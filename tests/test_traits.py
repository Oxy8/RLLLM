import asyncio
import json
from pathlib import Path

import pytest

from rlllmf.config import validate_settings
from rlllmf.judge import OptimismJudge, validate_scores
from tests.test_answers import CharacterTokenizer
from tests.test_core import judgment, temporary_settings


@pytest.mark.parametrize("score_name", ["childishness", "conciseness"])
def test_trait_reward_uses_named_field_aligns_duplicates_and_separates_cache(tmp_path, score_name):
    settings = temporary_settings(tmp_path)
    calls = []

    async def request(payload):
        calls.append(payload)
        return {
            "scores": [
                {
                    settings.judge.score_name if key == "optimism" else key: value
                    for key, value in judgment(
                        row["id"], 0.9 if row["response"] == "playful" else 0.1
                    ).items()
                }
                for row in reversed(payload["candidates"])
            ]
        }

    # The same prompt, candidates, rubric, and DB must not reuse another trait's scores.
    judge = OptimismJudge(settings, request=request, tokenizer=CharacterTokenizer())
    try:
        asyncio.run(judge.score_groups([("task", ["playful", "adult", "playful", "adult"])]))
    finally:
        judge.close()
    settings.judge.score_name = score_name
    log = tmp_path / f"{score_name}.judge.jsonl"
    judge = OptimismJudge(settings, log, request=request, tokenizer=CharacterTokenizer())
    try:
        scores = judge.reward_sync(["task"] * 4, ["playful", "adult", "playful", "adult"])
        assert scores == [0.9, 0.1, 0.9, 0.1]
        assert len(calls) == 2
        assert len(calls[-1]["candidates"]) == 2
        judge.reward_sync(["task"] * 4, ["playful", "adult", "playful", "adult"])
        assert len(calls) == 2
        row = json.loads(log.read_text().splitlines()[0])
        assert row["score_name"] == score_name
        assert all("optimism" not in item and score_name in item for item in row["judgments"])
        assert judge.cache_hits == 1
    finally:
        judge.close()


@pytest.mark.parametrize("score_name", ["childishness", "conciseness"])
def test_trait_validator_rejects_an_optimism_score_instead(score_name):
    with pytest.raises(ValueError, match=score_name):
        validate_scores({"scores": [judgment("c0", 0.8)]}, ["c0"], score_name)


@pytest.mark.parametrize("score_name", ["childishness", "conciseness"])
def test_builtin_rubric_and_score_field_must_agree(tmp_path, score_name):
    settings = temporary_settings(tmp_path)
    settings.judge.rubric_path = str(
        Path(__file__).resolve().parent.parent / f"prompts/{score_name}.txt"
    )
    with pytest.raises(ValueError, match="requires judge.score_name"):
        validate_settings(settings)
    settings.judge.score_name = score_name
    validate_settings(settings)
    settings.judge.score_name = "unknown"
    with pytest.raises(ValueError, match="judge.score_name"):
        validate_settings(settings)
