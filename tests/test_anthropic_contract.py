import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from anthropic import AsyncAnthropic

from rlllmf.judge import JudgeResponseError, OptimismJudge
from tests.test_answers import CharacterTokenizer
from tests.test_core import judgment, temporary_settings


@pytest.mark.parametrize("score_name", ["optimism", "childishness", "conciseness"])
@pytest.mark.parametrize("reward_mode", ["trait_only", "quality_gated", "quality_softplus"])
def test_real_sdk_serializes_request_and_parses_response_without_network(
    tmp_path, score_name, reward_mode,
):
    settings = temporary_settings(tmp_path)
    settings.judge.score_name = score_name
    settings.judge.reward_mode = reward_mode
    quality_mode = settings.judge.uses_task_quality
    observed = []

    def handle(request):
        body = json.loads(request.content)
        observed.append(body)
        assert request.url.path == "/v1/messages"
        assert body["model"] == settings.judge.model
        assert body["thinking"] == {"type": "disabled"}
        assert body["output_config"]["format"]["type"] == "json_schema"
        payload = json.loads(body["messages"][0]["content"])
        schema = body["output_config"]["format"]["schema"]["properties"]["scores"]
        ids = [c["id"] for c in payload["candidates"]]
        assert schema["type"] == "object"
        assert schema["required"] == ids
        assert set(schema["properties"]) == set(ids)
        assert schema["additionalProperties"] is False
        assert score_name in schema["properties"][ids[0]]["required"]
        scores = {
            candidate: {
                score_name if k == "optimism" else k: v
                for k, v in judgment(candidate, 0.75).items() if k != "id"
            }
            for candidate in ids
        }
        if quality_mode:
            scores = {candidate: {score_name: .75, "task_quality": .9,
                                  f"{score_name}_reason": "style", "task_quality_reason": "quality"}
                      for candidate in ids}
            assert "task_quality" in schema["properties"][ids[0]]["required"]
        return httpx2.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": settings.judge.model,
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Not part of the score JSON.",
                        "signature": "test-signature",
                    },
                    {"type": "text", "text": json.dumps({"scores": scores})},
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 20, "output_tokens": 10},
            },
        )

    def mock_client(**kwargs):
        return AsyncAnthropic(
            **kwargs,
            http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handle)),
        )

    judge = OptimismJudge(settings, tokenizer=CharacterTokenizer())
    try:
        with patch("anthropic.AsyncAnthropic", side_effect=mock_client):
            result = asyncio.run(judge.score_groups([("task", ["Hopeful", "Neutral"])]))
        assert result[0][0][score_name] == 0.75
        if quality_mode:
            assert result[0][0]["task_quality"] == .9
            expected = .8325 if reward_mode == "quality_gated" else .748130922985746
            assert result[0][0]["reward"] == pytest.approx(expected)
        assert len(result[0]) == 2
        assert len(observed) == 1
        assert judge.input_tokens == 20
        assert judge.output_tokens == 10
    finally:
        judge.close()


@pytest.mark.parametrize("reason", ["refusal", "max_tokens", "pause_turn"])
def test_provider_stop_reason_is_logged_and_only_truncation_suggests_more_tokens(tmp_path, reason):
    settings = temporary_settings(tmp_path)
    response = SimpleNamespace(
        stop_reason=reason, stop_details={"category": "general_harms"} if reason == "refusal"
        else None, content=[SimpleNamespace(type="text", text="Provider response")],
        usage=SimpleNamespace(input_tokens=20, output_tokens=5),
    )
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    client.messages.create = AsyncMock(return_value=response)
    log = tmp_path / "judge.jsonl"
    judge = OptimismJudge(settings, log, tokenizer=CharacterTokenizer())
    try:
        with patch("anthropic.AsyncAnthropic", return_value=client):
            with pytest.raises(RuntimeError) as caught:
                asyncio.run(judge.score_groups([("task", ["answer"])]))
        assert ("Increase judge.max_tokens" in str(caught.value)) == (reason == "max_tokens")
        assert isinstance(caught.value, JudgeResponseError)
        assert judge.input_tokens == 20 and judge.output_tokens == 5
        record = json.loads(log.with_suffix(".errors.jsonl").read_text())
        assert record["stop_reason"] == reason
        assert record["instruction"] == "task" and record["candidates"][0]["response"] == "answer"
        assert "test-key" not in log.with_suffix(".errors.jsonl").read_text()
        assert judge.db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 0
    finally:
        judge.close()
