import asyncio
import hashlib
import json
from unittest.mock import patch

import httpx2
import pytest
from anthropic import AsyncAnthropic

from rlllmf.collection import (
    AnthropicTeacher,
    collect,
    collection_lock,
    prepare_pool,
)
from rlllmf.data import load_prepared, normalized_prompt_id, prepare
from tests.test_core import FakeTokenizer, temporary_settings


@pytest.fixture
def setup_collection(tmp_path, monkeypatch):
    s = temporary_settings(tmp_path)
    s.collection.prompt_path = "generate.txt"
    (tmp_path / "generate.txt").write_text("Write a correct answer in a childlike voice.")
    s.collection.pool_size = 5
    s.collection.examples_per_run = 2
    s.collection.concurrency = 1
    s.data.train_file = "rl.json"
    s.data.eval_file = "eval.json"
    sources = {
        "rl.json": [{"instruction": "SAME  task", "input": "context"}],
        "eval.json": [{"instruction": "only eval task"}],
        s.collection.source_file: [
            {"instruction": "same task", "input": "context"},
            {"instruction": "only eval task"},
            {"instruction": "historical training task"},
            *[{"instruction": f"Explain test item {i}", "output": "SOURCE ANSWER SECRET"}
              for i in range(20)],
        ],
    }
    old_run = tmp_path / "runs/old"
    old_run.mkdir(parents=True)
    (old_run / "judge.jsonl").write_text(json.dumps({
        "instruction": "historical training task", "responses": ["old answer"],
    }) + "\n")

    def fake_source(settings, filename):
        rows = sources[filename]
        return list(rows), {"file": filename,
                            "sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest()}

    monkeypatch.setattr("rlllmf.collection.load_source", fake_source)
    return s, sources


def response(instruction):
    return {"text": json.dumps({"answer": "Ooh, let's find out! " + instruction}),
            "stop_reason": "end_turn", "response_id": "test-id",
            "usage": {"input_tokens": 20, "output_tokens": 12}}


def test_pool_disjoint_stable_and_extendable(setup_collection):
    s, sources = setup_collection
    pool, manifest = prepare_pool(s, tokenizer=FakeTokenizer())
    blocked = {normalized_prompt_id("same task\n\ncontext"),
               normalized_prompt_id("only eval task"),
               normalized_prompt_id("historical training task")}
    assert len(pool) == 5
    assert not blocked & {normalized_prompt_id(r["instruction"]) for r in pool}
    assert "SOURCE ANSWER SECRET" not in json.dumps(pool)
    again, _ = prepare_pool(s, tokenizer=FakeTokenizer())
    assert pool == again
    bigger, _ = prepare_pool(s, pool_size=8, tokenizer=FakeTokenizer())
    assert bigger[:5] == pool and len(bigger) == 8
    unchanged, _ = prepare_pool(s, tokenizer=FakeTokenizer())
    assert unchanged == bigger
    assert manifest["exclusions"]["saved_data_and_log_files"] == 1


def test_two_runs_append_without_repeating_or_replacing(setup_collection):
    s, _ = setup_collection
    calls = []

    async def request(instruction):
        calls.append(instruction)
        return response(instruction)

    collect(s, tokenizer=FakeTokenizer(), request=request)
    path = s.path(s.collection.output_dir) / "examples.jsonl"
    original = path.read_bytes()
    collect(s, tokenizer=FakeTokenizer(), request=request)
    assert path.read_bytes().startswith(original)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(set(calls)) == 4
    assert len(list(s.path("runs/childishness-collection").iterdir())) == 2
    for r in rows:
        assert r["messages"] == [{"role": "user", "content": r["instruction"]},
                                 {"role": "assistant", "content": r["answer"]}]
        assert "Write a correct answer" not in json.dumps(r["messages"])
    with pytest.raises(ValueError, match="unused reserved"):
        collect(s, count=2, tokenizer=FakeTokenizer(), request=request)
    assert len(calls) == 4


def test_failure_preserves_successes_and_next_run_skips_them(setup_collection):
    s, _ = setup_collection
    calls = []

    async def failing(instruction):
        calls.append(instruction)
        if len(calls) > 1:
            raise RuntimeError("simulated API failure")
        return response(instruction)

    with pytest.raises(RuntimeError, match="simulated API failure"):
        collect(s, tokenizer=FakeTokenizer(), request=failing)
    path = s.path(s.collection.output_dir) / "examples.jsonl"
    first = json.loads(path.read_text())
    next_calls = []

    async def succeed(instruction):
        next_calls.append(instruction)
        return response(instruction)

    stats = collect(s, tokenizer=FakeTokenizer(), request=succeed)
    assert stats["added"] == 2 and stats["dataset_total"] == 3
    assert first["instruction"] not in next_calls
    assert len(path.read_text().splitlines()) == 3


def test_invalid_or_truncated_outputs_never_enter_dataset(setup_collection):
    s, _ = setup_collection
    calls = []

    async def request(instruction):
        calls.append(instruction)
        r = response(instruction)
        r["stop_reason"] = "max_tokens"
        return r

    with pytest.raises(RuntimeError, match="incomplete/refused"):
        collect(s, tokenizer=FakeTokenizer(), request=request)
    assert len(calls) == s.judge.max_retries + 1
    assert not (s.path(s.collection.output_dir) / "examples.jsonl").exists()
    usage_file = next(s.path("runs/childishness-collection").glob("*/usage.json"))
    usage = json.loads(usage_file.read_text())
    assert usage["status"] == "failed" and usage["added"] == 0
    assert usage["input_tokens"] == 20 * len(calls)


def test_empty_answer_retry_and_incomplete_tail_recovery(setup_collection):
    s, _ = setup_collection
    calls = []

    async def request(instruction):
        calls.append(instruction)
        r = response(instruction)
        if len(calls) == 1:
            r["text"] = json.dumps({"answer": ""})
        return r

    collect(s, count=1, tokenizer=FakeTokenizer(), request=request)
    assert len(calls) == 2
    path = s.path(s.collection.output_dir) / "examples.jsonl"
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b'{"unfinished":')
    collect(s, count=1, tokenizer=FakeTokenizer(), request=request)
    assert path.read_bytes().startswith(original)
    assert len(path.read_text().splitlines()) == 2
    assert list(path.parent.glob("examples.jsonl.partial-*"))


def test_prepare_only_no_api_and_concurrent_collection_rejected(setup_collection):
    s, _ = setup_collection
    s.api.anthropic_api_key = ""
    report = collect(s, prepare_only=True, tokenizer=FakeTokenizer())
    assert report["existing"] == 0 and report["pool_size"] == 5
    directory = s.path(s.collection.output_dir)
    with collection_lock(directory):
        with pytest.raises(ValueError, match="Another collection"):
            collect(s, prepare_only=True, tokenizer=FakeTokenizer())


def test_future_rl_eval_preparation_excludes_reservations(setup_collection, monkeypatch):
    s, sources = setup_collection
    pool, _ = prepare_pool(s, tokenizer=FakeTokenizer())
    reserved = pool[0]["instruction"]
    s.data.train_samples = 2
    s.data.eval_samples = 2
    sources["rl.json"] = [{"instruction": reserved.upper()},
                          {"instruction": "New RL one"}, {"instruction": "New RL two"}]
    sources["eval.json"] = [{"instruction": reserved},
                            {"instruction": "New eval one"}, {"instruction": "New eval two"}]
    def fake_download(repo, filename, **kwargs):
        path = s.root / filename
        path.write_text(json.dumps(sources[filename]))
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    prepare(s, tokenizer=FakeTokenizer())
    for split in ["train", "eval"]:
        dataset = load_prepared(s, split)
        assert len(dataset) == 2
        assert all(normalized_prompt_id(r["prompt"][0]["content"])
                   != normalized_prompt_id(reserved) for r in dataset)
    path = s.path(s.data.output_dir) / "train.jsonl"
    path.write_text(json.dumps({"prompt": [{"role": "user", "content": reserved}],
                                "prompt_id": "bad"}) + "\n")
    with pytest.raises(ValueError, match="overlaps"):
        load_prepared(s, "train")
    with pytest.raises(ValueError, match="overlaps"):
        prepare_pool(s, tokenizer=FakeTokenizer())


def test_teacher_real_sdk_contract_without_network(setup_collection):
    s, _ = setup_collection
    observed = []
    system = s.path(s.collection.prompt_path).read_text()

    def handle(request):
        body = json.loads(request.content)
        observed.append(body)
        assert body["system"] == system
        assert body["messages"] == [{"role": "user", "content": "Explain rain"}]
        assert body["thinking"] == {"type": "disabled"}
        assert body["output_config"]["format"]["schema"]["required"] == ["answer"]
        return httpx2.Response(200, json={
            "id": "teacher-test", "type": "message", "role": "assistant",
            "model": s.judge.model, "content": [
                {"type": "thinking", "thinking": "Not a training target.", "signature": "test"},
                {"type": "text", "text": json.dumps({"answer": "Drip, drip! Water falls!"})},
            ], "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 8},
        })

    def client(**kwargs):
        return AsyncAnthropic(**kwargs, http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(handle)))

    async def run():
        with patch("anthropic.AsyncAnthropic", side_effect=client):
            async with AnthropicTeacher(s, system) as teacher:
                return await teacher("Explain rain")

    result = asyncio.run(run())
    assert json.loads(result["text"])["answer"] == "Drip, drip! Water falls!"
    assert "Not a training target" not in result["text"]
    assert len(observed) == 1
