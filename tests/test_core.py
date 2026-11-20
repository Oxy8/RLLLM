import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlllmf.config import (
    APISettings,
    DataSettings,
    EvaluationSettings,
    JudgeSettings,
    Settings,
    TrainingSettings,
    load_settings,
    validate_settings,
)
from rlllmf.data import instruction_text, select_rows
from rlllmf.judge import OptimismJudge, plain_text, validate_scores


def temporary_settings(root):
    root = Path(root)
    (root / "rubric.txt").write_text("Score optimism only.")
    return Settings(
        root=root,
        api=APISettings(anthropic_api_key="test-key"),
        judge=JudgeSettings(rubric_path="rubric.txt", cache_path="cache.sqlite3"),
        data=DataSettings(),
        training=TrainingSettings(),
        evaluation=EvaluationSettings(),
    )


def judgment(candidate, score):
    return {
        "id": candidate,
        "optimism": score,
        "factual_consistency": "consistent",
        "relevant": True,
        "reason": "test",
    }


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        ids = messages[0]["content"].split()
        # Current Transformers returns a mapping by default; count token IDs.
        return ids if kwargs.get("return_dict") is False else {"input_ids": ids}


class DataTests(unittest.TestCase):
    def test_keep_context_and_drop_answers(self):
        rows = [{"instruction": "Summarize this", "input": "some facts", "output": "SECRET ANSWER"}]
        selected, _ = select_rows(rows, FakeTokenizer(), 10, 100, 42)
        self.assertEqual(
            selected[0]["prompt"], [{"role": "user", "content": "Summarize this\n\nsome facts"}]
        )
        self.assertNotIn("SECRET ANSWER", json.dumps(selected))
        self.assertEqual(set(selected[0]), {"prompt", "prompt_id"})

    def test_dedup_filter_and_split_exclusion(self):
        rows = [
            {"instruction": "first"},
            {"instruction": "first"},
            {"instruction": "second"},
            {"instruction": "too many words here"},
        ]
        excluded = {hashlib.sha256(b"first").hexdigest()}
        selected, stats = select_rows(rows, FakeTokenizer(), 10, 2, 42, excluded)
        self.assertEqual([r["prompt"][0]["content"] for r in selected], ["second"])
        self.assertEqual(stats, {"duplicates": 2, "too_long": 1})

    def test_invalid_instruction_rejected(self):
        with self.assertRaises(ValueError):
            instruction_text({"instruction": "", "output": "not an instruction"})


class ConfigTests(unittest.TestCase):
    def test_secrets_redacted_and_environment_overrides(self):
        with tempfile.TemporaryDirectory() as root:
            temporary_settings(root)
            path = Path(root) / ".config"
            path.write_text(
                '[api]\nanthropic_api_key="file-secret"\n[judge]\nrubric_path="rubric.txt"\n'
            )
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-secret"}):
                settings = load_settings(path)
            self.assertEqual(settings.api.anthropic_api_key, "env-secret")
            self.assertNotIn("env-secret", json.dumps(settings.public_dict()))
            self.assertNotIn("env-secret", repr(settings))

    def test_invalid_grpo_groups_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            settings = temporary_settings(root)
            settings.training.gradient_accumulation_steps = 3
            with self.assertRaises(ValueError):
                validate_settings(settings)


class JudgeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.settings = temporary_settings(self.directory.name)
        self.calls = []

        async def request(payload):
            self.calls.append(payload)
            # Deliberately reverse the API output order. ID mapping must still work.
            rows = [
                judgment(c["id"], {"hopeful": 0.9, "neutral": 0.5, "sad": 0.1}[c["response"]])
                for c in reversed(payload["candidates"])
            ]
            return {"scores": rows, "usage": {"input_tokens": 12, "output_tokens": 7}}

        self.judge = OptimismJudge(self.settings, request=request)

    async def asyncTearDown(self):
        self.judge.close()
        self.directory.cleanup()

    async def test_alignment_multiple_groups_duplicates_and_cache(self):
        prompts = [[{"role": "user", "content": p}] for p in ["task1"] * 4 + ["task2"] * 4]
        texts = ["sad", "hopeful", "neutral", "sad"] * 2
        completions = [[{"role": "assistant", "content": t}] for t in texts]
        scores = await self.judge.reward(prompts, completions)
        self.assertEqual(scores, [0.1, 0.9, 0.5, 0.1] * 2)
        self.assertEqual(len(self.calls), 2)
        await self.judge.reward(prompts, completions)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.judge.statistics()["cache_hits"], 2)
        self.assertEqual(self.judge.statistics()["input_tokens"], 24)
        self.assertEqual(len(self.calls[0]["candidates"]), 3)

    async def test_cache_survives_reopening(self):
        await self.judge.score_groups([("task", ["hopeful"])])
        second = OptimismJudge(self.settings, request=self.judge.request)
        try:
            await second.score_groups([("task", ["hopeful"])])
            self.assertEqual(second.requests, 0)
            self.assertEqual(len(self.calls), 1)
        finally:
            second.close()

    async def test_budget_checks_cache_before_new_request(self):
        self.settings.judge.max_requests = 1
        await self.judge.score_groups([("task", ["hopeful"])])
        await self.judge.score_groups([("task", ["hopeful"])])
        with self.assertRaises(RuntimeError):
            await self.judge.score_groups([("other", ["hopeful"])])

    async def test_bad_groups_and_api_failure_never_become_zero_rewards(self):
        with self.assertRaises(ValueError):
            await self.judge.reward(["a", "b", "a", "a"], ["sad"] * 4)

        async def failure(payload):
            raise RuntimeError("network unavailable")

        self.judge.request = failure
        with self.assertRaises(RuntimeError):
            await self.judge.score_groups([("task", ["sad"])])
        self.assertEqual(self.judge.db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0], 0)

    async def test_invalid_scores_retry_log_usage_and_cache_only_valid_response(self):
        calls = []
        self.judge.log_path = self.settings.root / "judge.jsonl"

        async def request(payload):
            calls.append(payload)
            ids = [c["id"] for c in payload["candidates"]]
            return {
                "scores": [judgment(c, 0.5) for c in ids[:1 if len(calls) == 1 else 2]],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

        self.judge.request = request
        result = await self.judge.score_groups([("task", ["neutral", "hopeful"])])
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(self.judge.requests, 2)
        self.assertEqual(self.judge.input_tokens, 20)
        self.assertEqual(self.judge.output_tokens, 10)
        errors = self.settings.root / "judge.errors.jsonl"
        self.assertIn("expected 2, received 1", json.loads(errors.read_text())["error"])
        await self.judge.score_groups([("task", ["neutral", "hopeful"])])
        self.assertEqual(len(calls), 2)

    async def test_persistent_invalid_scores_exhaust_retries_without_cached_rewards(self):
        async def request(payload):
            return {"scores": []}

        self.judge.request = request
        with self.assertRaisesRegex(ValueError, "failed after 3 judge attempts"):
            await self.judge.score_groups([("task", ["hopeful"])])
        self.assertEqual(self.judge.requests, 3)
        self.assertEqual(self.judge.db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0], 0)

    async def test_validation_retries_respect_request_budget(self):
        self.settings.judge.max_requests = 1

        async def request(payload):
            return {"scores": []}

        self.judge.request = request
        with self.assertRaisesRegex(RuntimeError, "max_requests reached"):
            await self.judge.score_groups([("task", ["hopeful"])])
        self.assertEqual(self.judge.requests, 1)


class ValidationTests(unittest.TestCase):
    def test_invalid_scores_rejected(self):
        for bad in [True, float("nan"), float("inf"), -0.1, 1.1, "0.5"]:
            with self.subTest(value=bad), self.assertRaises(ValueError):
                validate_scores({"scores": [judgment("c0", bad)]}, ["c0"])
        with self.assertRaises(ValueError):
            validate_scores({"scores": [judgment("c0", 0.5)] * 2}, ["c0", "c1"])

    def test_text_formats(self):
        self.assertEqual(plain_text("text"), "text")
        self.assertEqual(
            plain_text([{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}]),
            "hello",
        )
        with self.assertRaises(ValueError):
            plain_text([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])


if __name__ == "__main__":
    unittest.main()
