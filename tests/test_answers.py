import asyncio

import pytest

from rlllmf.answers import (
    decode_final_answer,
    final_answer,
    generation_diagnostics,
    limit_answer_tokens,
    thinking_is_prefilled,
)
from rlllmf.judge import OptimismJudge
from tests.test_core import judgment, temporary_settings


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("  Final answer.  ", "Final answer."),
        ("<think>Private reasoning.</think>Final answer.", "Final answer."),
        ("Private reasoning.</think>Final answer.", "Final answer."),
        ("<think>Truncated reasoning", ""),
        ("<think>First</think><think>Second</think>Final answer.", "Final answer."),
        ("<think></think>Final answer.", "Final answer."),
    ],
)
def test_only_final_answer_is_extracted(text, expected):
    assert final_answer(text) == expected


class CharacterTokenizer:
    all_special_tokens = []

    def get_vocab(self):
        return {chr(i): i for i in range(128)}

    def encode(self, text, *, add_special_tokens):
        return list(text)

    def decode(self, ids, *, skip_special_tokens):
        return "".join(ids)


class ReasoningTokenizer(CharacterTokenizer):
    all_special_tokens = ["<think>", "</think>", "<|im_end|>"]

    def decode(self, ids, *, skip_special_tokens):
        if ids and isinstance(ids[0], str):
            return "".join(ids)
        assert skip_special_tokens is False
        answers = {
            0: "<think>PRIVATE</think>hopeful<|im_end|>",
            1: "<think>PRIVATE</think>neutral<|im_end|>",
        }
        return answers[ids[0]]


def test_delimiters_survive_decoding_until_reasoning_is_removed():
    assert decode_final_answer(ReasoningTokenizer(), [0]) == "hopeful"


def test_reward_uses_token_ids_to_keep_reasoning_out_of_api_and_logs(tmp_path):
    settings = temporary_settings(tmp_path)
    settings.training.num_generations = 2
    calls = []

    async def request(payload):
        calls.append(payload)
        return {"scores": [judgment(row["id"], 0.5) for row in payload["candidates"]]}

    log = tmp_path / "judge.jsonl"
    judge = OptimismJudge(settings, log, request=request, tokenizer=ReasoningTokenizer())
    try:
        rewards = asyncio.run(
            judge.reward(
                ["task"] * 2,
                ["PRIVATEhopeful", "PRIVATEneutral"],
                completion_ids=[[0], [1]],
            )
        )
        assert rewards == [0.5, 0.5]
        assert {row["response"] for row in calls[0]["candidates"]} == {"hopeful", "neutral"}
        assert "PRIVATE" not in log.read_text()
        assert settings.judge.model in log.read_text()
        import json

        row = json.loads(log.read_text())
        assert all(info["thinking_detected"] for info in row["generation_diagnostics"])
        assert not any(info["empty_final_answer"] for info in row["generation_diagnostics"])
    finally:
        judge.close()


def test_thinking_only_truncation_is_diagnosed_without_storing_reasoning():
    class TruncatedTokenizer(ReasoningTokenizer):
        eos_token_id = 2

        def decode(self, ids, *, skip_special_tokens):
            return "<think>PRIVATE reasoning that hit the token limit"

    info = generation_diagnostics(TruncatedTokenizer(), [0] * 256, 256)
    assert info == {
        "generated_tokens": 256,
        "hit_token_limit": True,
        "thinking_detected": True,
        "unfinished_thinking": True,
        "empty_final_answer": True,
    }
    assert "PRIVATE" not in str(info)


def reasoning_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "thought": 1, "<think>": 2, "</think>": 3, "answer": 4, "<eos>": 5}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", eos_token="<eos>")
    tokenizer.add_special_tokens({"additional_special_tokens": ["<think>", "</think>"]})
    tokenizer.chat_template = (
        "{% if add_generation_prompt %}assistant "
        "{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}{% endif %}"
    )
    return tokenizer


def test_prefilled_thinking_cannot_leak_when_generation_hits_limit():
    tokenizer = reasoning_tokenizer()
    assert thinking_is_prefilled(tokenizer, True)
    assert not thinking_is_prefilled(tokenizer, False)
    ids = [1] * 2048
    assert decode_final_answer(tokenizer, ids, thinking_prefilled=True) == ""
    info = generation_diagnostics(
        tokenizer, ids, 2048, thinking_prefilled=True, max_answer_tokens=256
    )
    assert info["unfinished_thinking"] and info["hit_token_limit"]
    assert info["judge_answer_tokens"] == 0


def test_long_local_thinking_does_not_consume_judge_answer_budget(tmp_path):
    tokenizer = reasoning_tokenizer()
    settings = temporary_settings(tmp_path)
    settings.training.num_generations = 2
    settings.training.max_completion_length = 2048
    settings.training.max_answer_tokens = 256
    observed = []

    async def request(payload):
        observed.append(payload)
        return {"scores": [judgment(row["id"], 0.75) for row in payload["candidates"]]}

    # The opening thinking token is in the prompt, not the generated completion.
    ids = [1] * 1500 + [3] + [4] * 400 + [5]
    judge = OptimismJudge(settings, tmp_path / "judge.jsonl", request=request, tokenizer=tokenizer)
    try:
        assert judge.thinking_prefilled
        rewards = asyncio.run(
            judge.reward(
                ["task"] * 2,
                ["ignored"] * 2,
                completion_ids=[ids, ids],
            )
        )
        assert rewards == [0.75, 0.75]
        text = observed[0]["candidates"][0]["response"]
        assert "thought" not in text and "<think>" not in text
        assert len(tokenizer.encode(text, add_special_tokens=False)) == 256
        import json

        log = json.loads((tmp_path / "judge.jsonl").read_text())
        info = log["generation_diagnostics"][0]
        assert info["generated_tokens"] == 1902
        assert info["final_answer_tokens"] == 400
        assert info["judge_answer_tokens"] == 256
        assert info["answer_truncated_for_judge"]
    finally:
        judge.close()


def test_direct_scoring_also_caps_each_final_answer(tmp_path):
    settings = temporary_settings(tmp_path)
    observed = []

    async def request(payload):
        observed.append(payload)
        return {"scores": [judgment(row["id"], 0.5) for row in payload["candidates"]]}

    judge = OptimismJudge(settings, request=request, tokenizer=CharacterTokenizer())
    try:
        asyncio.run(judge.score_groups([("task", ["a" * 1000, "short"])]))
        assert sorted(len(row["response"]) for row in observed[0]["candidates"]) == [5, 256]
    finally:
        judge.close()


def test_answer_token_cap_handles_unicode():
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    raw = ByteLevelBPETokenizer()
    raw.train_from_iterator(["Text with 🌈 and café"], vocab_size=256)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw._tokenizer)
    clipped = limit_answer_tokens(tokenizer, "🌈 café " * 100, 13)
    assert len(tokenizer.encode(clipped, add_special_tokens=False)) <= 13
