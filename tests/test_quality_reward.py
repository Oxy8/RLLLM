import asyncio
import json
import math

import pytest

from rlllmf.config import validate_settings
from rlllmf.evaluation import record_metrics
from rlllmf.judge import OptimismJudge, candidate_score_schema, validate_scores
from tests.test_answers import CharacterTokenizer
from tests.test_core import temporary_settings


def dual_score(candidate, trait, quality, score_name="childishness"):
    return {"id": candidate, score_name: trait, "task_quality": quality,
            f"{score_name}_reason": "A speaking-voice judgment.",
            "task_quality_reason": "An instruction-following judgment."}


@pytest.mark.parametrize("score_name", ["childishness", "optimism", "conciseness"])
def test_dual_scores_combine_align_deduplicate_log_and_recompute_cached_rewards(
    tmp_path, score_name,
):
    settings = temporary_settings(tmp_path)
    settings.judge.score_name = score_name
    settings.judge.reward_mode = "quality_gated"
    settings.training.num_generations = 4
    calls = []
    values = {"adult": (0, .9), "child": (.8, .9), "wrong": (1, .1)}

    async def request(payload):
        calls.append(payload)
        return {"scores": [dual_score(row["id"], *values[row["response"]], score_name)
                           for row in reversed(payload["candidates"])]}

    judge = OptimismJudge(settings, tmp_path / "judge.jsonl", request=request,
                          tokenizer=CharacterTokenizer())
    try:
        prompts, responses = ["task"] * 4, ["adult", "child", "wrong", "child"]
        rewards = judge.reward_sync(prompts, responses)
        assert rewards == pytest.approx([.63, .846, .1, .846])
        assert len(calls) == 1 and len(calls[0]["candidates"]) == 3
        row = json.loads((tmp_path / "judge.jsonl").read_text())
        assert row["reward_mode"] == "quality_gated"
        assert [j[score_name] for j in row["judgments"]] == [0, .8, 1, .8]
        assert [j["task_quality"] for j in row["judgments"]] == [.9, .9, .1, .9]
        assert [j["reward"] for j in row["judgments"]] == pytest.approx(rewards)
        cached = json.loads(judge.db.execute("SELECT value FROM judgments").fetchone()[0])
        assert all("reward" not in item for item in cached["scores"])
        settings.judge.quality_baseline = .5
        assert judge.reward_sync(prompts, responses) == pytest.approx([.45, .81, .1, .81])
        assert len(calls) == 1 and judge.cache_hits == 1
        # Both dual modes share raw judgments, never an obsolete combined reward.
        settings.judge.reward_mode = "quality_softplus"
        rewards = judge.reward_sync(prompts, responses)
        assert rewards == pytest.approx([
            -.001869077014254, .798130922985746, -4.500002569156288, .798130922985746,
        ])
        assert len(calls) == 1 and judge.cache_hits == 2
        latest = json.loads((tmp_path / "judge.jsonl").read_text().splitlines()[-1])
        assert latest["reward_mode"] == "quality_softplus"
        assert latest["quality_softplus_center"] == .65
        assert latest["quality_softplus_sharpness"] == pytest.approx(math.log(3) / .05)
        assert latest["quality_softplus_scale"] == 10
        assert "quality_baseline" not in latest
        assert [j["reward"] for j in latest["judgments"]] == pytest.approx(rewards)
        assert "test-key" not in (tmp_path / "judge.jsonl").read_text()
    finally:
        judge.close()


@pytest.mark.parametrize("quality", [None, True, -.1, 1.1, math.nan, math.inf, "0.9"])
def test_invalid_quality_scores_rejected(quality):
    data = {"scores": [dual_score("c0", .8, quality)]}
    with pytest.raises(ValueError, match="task_quality"):
        validate_scores(data, ["c0"], "childishness", task_quality=True)


def test_both_reasons_are_required_and_schema_names_every_candidate():
    row = dual_score("c0", .8, .9)
    del row["task_quality_reason"]
    with pytest.raises(ValueError, match="explain both"):
        validate_scores({"scores": [row]}, ["c0"], "childishness", task_quality=True)
    schema = candidate_score_schema(["c0", "c1"], "childishness", task_quality=True)
    candidates = schema["properties"]["scores"]
    assert candidates["required"] == ["c0", "c1"]
    assert candidates["properties"]["c0"]["required"] == [
        "childishness", "task_quality", "childishness_reason", "task_quality_reason",
    ]


@pytest.mark.parametrize("baseline", [0, 1, -.1, 1.1, True, math.nan, math.inf])
def test_invalid_combination_parameters_rejected(tmp_path, baseline):
    settings = temporary_settings(tmp_path)
    settings.judge.quality_baseline = baseline
    with pytest.raises(ValueError, match="quality_baseline"):
        validate_settings(settings)


def test_quality_metrics_do_not_require_the_legacy_diagnostic_fields():
    rows = [{"response": "answer", "judgment": {**dual_score("c0", .8, .9), "reward": .846}}]
    metrics = record_metrics(rows, "childishness")
    assert metrics["mean_childishness"] == .8
    assert metrics["mean_task_quality"] == .9
    assert metrics["mean_combined_reward"] == .846
    assert "contradiction_rate" not in metrics


def test_legacy_scores_cannot_satisfy_the_new_schema_or_populate_its_cache(tmp_path):
    settings = temporary_settings(tmp_path)
    settings.judge.reward_mode = "quality_gated"
    settings.judge.max_retries = 0

    async def request(payload):
        from tests.test_core import judgment

        return {"scores": [judgment(row["id"], .9) for row in payload["candidates"]]}

    judge = OptimismJudge(settings, request=request)
    try:
        with pytest.raises(ValueError, match="task_quality"):
            asyncio.run(judge.score_groups([("task", ["answer"])]))
        assert judge.db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 0
    finally:
        judge.close()


def test_smooth_reward_has_the_requested_incentives_and_is_monotonic(tmp_path):
    settings = temporary_settings(tmp_path)
    settings.judge.reward_mode = "quality_softplus"
    judge = OptimismJudge(settings)
    try:
        def reward(quality, trait=.5):
            return judge.reward_value({"optimism": trait, "task_quality": quality})

        h = 1e-6
        for quality, expected in [(.65, 5), (.70, 2.5), (.75, 1), (.80, 10 / 28)]:
            sensitivity = (reward(quality + h) - reward(quality - h)) / (2 * h)
            assert sensitivity == pytest.approx(expected, rel=1e-7)
        assert reward(.85 + h) - reward(.85 - h) < 2 * h
        values = [reward(i / 100) for i in range(101)]
        assert all(math.isfinite(value) for value in values)
        assert all(a < b for a, b in zip(values, values[1:]))
        for quality in [0, .65, .7, .75, .8, 1]:
            assert reward(quality, .8) - reward(quality, .2) == pytest.approx(.6)
        assert reward(.1) < 0  # Valid negative rewards must reach GRPO unchanged.
        # Stable even when a configured steepness would overflow a naive exp().
        settings.judge.quality_softplus_sharpness = 10000
        assert reward(0) == pytest.approx(-6)
        assert reward(1) == pytest.approx(.5)
    finally:
        judge.close()


@pytest.mark.parametrize("name", ["quality_softplus_scale", "quality_softplus_sharpness"])
@pytest.mark.parametrize("value", [0, -1, True, math.nan, math.inf])
def test_invalid_smooth_reward_parameters_rejected(tmp_path, name, value):
    settings = temporary_settings(tmp_path)
    settings.judge.reward_mode = "quality_softplus"
    setattr(settings.judge, name, value)
    with pytest.raises(ValueError, match=name):
        validate_settings(settings)


@pytest.mark.parametrize("value", [0, 1, -.1, 1.1, True, math.nan, math.inf])
def test_invalid_smooth_reward_center_rejected(tmp_path, value):
    settings = temporary_settings(tmp_path)
    settings.judge.quality_softplus_center = value
    with pytest.raises(ValueError, match="quality_softplus_center"):
        validate_settings(settings)
