import copy
import json
import sqlite3

import pytest

from rlllmf.evaluation import align_records, compare, comparison_models
from rlllmf.judge import OptimismJudge
from tests.test_answers import CharacterTokenizer
from tests.test_core import judgment, temporary_settings


def item(prompt, answer):
    return {
        "prompt_id": prompt,
        "instruction": f"task {prompt}",
        "response": answer,
        "generated_tokens": 10,
        "generation_diagnostics": {"generated_tokens": 10, "hit_token_limit": False},
    }


def test_pairs_match_question_ids_and_reject_mismatched_instructions():
    before = [item("one", "first"), item("two", "second")]
    after = [item("two", "changed second"), item("one", "changed first")]
    pairs = align_records(before, after)
    assert [b["response"] for _, b in pairs] == ["changed first", "changed second"]
    with pytest.raises(ValueError, match="unique"):
        align_records(before, [after[0], after[0]])
    after[1]["instruction"] = "different question"
    with pytest.raises(ValueError, match="instructions differ"):
        align_records(before, after)


def test_comparison_uses_the_runs_original_snapshot(tmp_path):
    settings = temporary_settings(tmp_path)
    for name in ["experiment/final", "experiment/initial_model"]:
        p = tmp_path / name
        p.mkdir(parents=True)
        (p / "config.json").write_text("{}")
    trained, original = comparison_models(settings, "experiment/final", None)
    assert trained == str(tmp_path / "experiment/final")
    assert original == str(tmp_path / "experiment/initial_model")
    with pytest.raises(ValueError, match="not found"):
        comparison_models(settings, "missing/final", None)


@pytest.mark.parametrize("score_name", ["optimism", "childishness", "conciseness"])
@pytest.mark.parametrize("reward_mode", ["trait_only", "quality_gated", "quality_softplus"])
def test_automatic_comparison_scores_pairs_and_exports_report(
    tmp_path, monkeypatch, score_name, reward_mode,
):
    from rlllmf import evaluation

    settings = temporary_settings(tmp_path)
    settings.judge.score_name = score_name
    settings.judge.reward_mode = reward_mode
    quality_mode = settings.judge.uses_task_quality
    for name in ["experiment/final", "experiment/initial_model"]:
        path = tmp_path / name
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}")
        (path / "model.safetensors").write_bytes(b"test weights")
    before = [item("one", "neutral"), item("two", "identical")]
    after = [item("one", "hopeful"), item("two", "identical")]
    generated_sources, requests = [], []

    def generate(settings, data, source, **kwargs):
        generated_sources.append(source)
        return copy.deepcopy(before if source.endswith("initial_model") else after)

    async def request(payload):
        requests.append(payload)
        if quality_mode:
            from tests.test_quality_reward import dual_score

            return {"scores": [dual_score(
                row["id"], .8 if row["response"] == "hopeful" else .5,
                .7 if row["response"] == "hopeful" else .9, score_name,
            ) for row in reversed(payload["candidates"])]}
        return {
            "scores": [
                {
                    score_name if key == "optimism" else key: value
                    for key, value in judgment(
                        row["id"], 0.8 if row["response"] == "hopeful" else 0.5
                    ).items()
                }
                for row in reversed(payload["candidates"])
            ]
        }

    monkeypatch.setattr(evaluation, "require_gpu", lambda: None)
    monkeypatch.setattr(
        evaluation,
        "selected_data",
        lambda settings, limit: [
            {
                "prompt_id": row["prompt_id"],
                "prompt": [{"role": "user", "content": row["instruction"]}],
            }
            for row in before
        ],
    )
    monkeypatch.setattr(evaluation, "generate_answers", generate)
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda *args: CharacterTokenizer())
    monkeypatch.setattr(
        evaluation,
        "OptimismJudge",
        lambda *args, **kwargs: OptimismJudge(*args, **kwargs, request=request),
    )
    compare(settings, "experiment/final", "report")
    assert generated_sources == [
        str(tmp_path / "experiment/initial_model"),
        str(tmp_path / "experiment/final"),
    ]
    assert len(requests) == 2
    assert len(requests[0]["candidates"]) == 2
    assert len(requests[1]["candidates"]) == 1  # Identical answers tie after deduplication.
    assert "baseline_model" not in json.dumps(requests)
    root = tmp_path / "report"
    summary = json.loads((root / "comparison.summary.json").read_text())
    assert summary["baseline"][f"mean_{score_name}"] == 0.5
    assert summary["trained"][f"mean_{score_name}"] == 0.65
    assert summary["delta_trained_minus_baseline"][f"mean_{score_name}"] == pytest.approx(0.15)
    assert summary[f"{score_name}_outcomes"]["counts"] == {"trained": 1, "baseline": 0, "tie": 1}
    assert summary["score_name"] == score_name
    if quality_mode:
        assert summary["baseline"]["mean_task_quality"] == pytest.approx(.9)
        assert summary["trained"]["mean_task_quality"] == pytest.approx(.8)
        expected_before = .765 if reward_mode == "quality_gated" else .498130922985746
        expected_after = .7115 if reward_mode == "quality_gated" else .583600584707144
        assert summary["baseline"]["mean_combined_reward"] == pytest.approx(expected_before)
        assert summary["trained"]["mean_combined_reward"] == pytest.approx(expected_after)
        assert summary["task_quality_outcomes"]["counts"] == {
            "trained": 0, "baseline": 1, "tie": 1,
        }
        assert summary["combined_reward_outcomes"]["counts"] == (
            {"trained": 0, "baseline": 1, "tie": 1} if reward_mode == "quality_gated"
            else {"trained": 1, "baseline": 0, "tie": 1}
        )
    assert "test-key" not in (root / "comparison-run.json").read_text()
    rows = [json.loads(line) for line in (root / "comparison.jsonl").read_text().splitlines()]
    assert rows[0]["responses"] == ["neutral", "hopeful"]
    assert rows[0]["comparison"]["winner"] == "trained"
    assert rows[1]["comparison"]["winner"] == "tie"
    assert (root / "comparison.html").is_file()
    with pytest.raises(ValueError, match="already exists"):
        compare(settings, "experiment/final", "report")
    generated_sources.clear()
    compare(settings, "experiment/final", "report-two")
    assert generated_sources == [str(tmp_path / "experiment/final")]
    reused = json.loads((tmp_path / "report-two/comparison.summary.json").read_text())
    assert reused["baseline_cache"]["hits"] == 2
    assert reused["baseline_cache"]["misses"] == 0

    # Simulate a judging interruption after generation, with one cached pair.
    (root / "comparison.summary.json").unlink()
    first_log = (root / "comparison.judge.jsonl").read_text().splitlines()[0]
    first_key = json.loads(first_log)["cache_key"]
    with sqlite3.connect(settings.path(settings.judge.cache_path)) as db:
        db.execute("DELETE FROM judgments WHERE key != ?", (first_key,))
    saved = json.loads((root / "comparison-run.json").read_text())
    saved["settings"]["evaluation"]["temperature"] = 0.8
    (root / "comparison-run.json").write_text(json.dumps(saved))
    settings.evaluation.temperature = 0.0
    settings.evaluation.max_answer_tokens = 1
    generated_sources.clear()
    requests.clear()

    def no_gpu_or_generation(*args, **kwargs):
        raise AssertionError("Resume must not load GPU weights, generate, or reload questions")

    monkeypatch.setattr(evaluation, "require_gpu", no_gpu_or_generation)
    monkeypatch.setattr(evaluation, "generate_answers", no_gpu_or_generation)
    monkeypatch.setattr(evaluation, "selected_data", no_gpu_or_generation)
    compare(settings, "experiment/final", "report", resume=True)
    assert len(requests) == 1
    resumed = json.loads((root / "comparison.summary.json").read_text())
    assert resumed["judge_usage"]["cache_hits"] == 1
    assert resumed["settings"]["evaluation"]["temperature"] == 0.8
    assert resumed["trained"][f"mean_{score_name}"] == 0.65
    rows = [json.loads(line) for line in (root / "comparison.jsonl").read_text().splitlines()]
    assert rows[0]["responses"] == ["neutral", "hopeful"]
    with pytest.raises(ValueError, match="already complete"):
        compare(settings, "experiment/final", "report", resume=True)


@pytest.mark.parametrize("temperature", [0.0, 0.8])
def test_shared_generation_matches_seeds_caps_answers_and_releases_models(
    tmp_path, monkeypatch, temperature
):
    import weakref

    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Lfm3Config, Lfm3ForCausalLM, PreTrainedTokenizerFast

    from rlllmf import evaluation

    torch.set_num_threads(1)
    words = ["<pad>", "<eos>", "<unk>", "task", "assistant"] + [f"word{i}" for i in range(12)]
    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )
    tokenizer.chat_template = "{{ messages[0]['content'] }} assistant "
    model = Lfm3ForCausalLM(
        Lfm3Config(
            vocab_size=len(words),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            conv_L_cache=3,
            layer_types=["conv", "sliding_attention", "full_attention"],
            sliding_window=8,
            max_position_embeddings=64,
            eos_token_id=1,
            pad_token_id=0,
            bos_token_id=None,
        )
    )
    references = []

    def load(*args):
        instance = copy.deepcopy(model)
        references.append(weakref.ref(instance))
        return instance

    monkeypatch.setattr(evaluation, "load_text_model", load)
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda *args: tokenizer)
    settings = temporary_settings(tmp_path)
    settings.evaluation.enable_thinking = False
    settings.evaluation.max_completion_length = 4
    settings.evaluation.max_answer_tokens = 2
    settings.evaluation.temperature = temperature
    data = [
        {"prompt_id": str(i), "prompt": [{"role": "user", "content": "task"}]} for i in range(2)
    ]
    before = evaluation.generate_answers(settings, data, "original")
    torch.manual_seed(999)
    after = evaluation.generate_answers(settings, data, "trained")
    assert before == after
    partial = evaluation.generate_answers(settings, data[1:], "original", seed_indices=[1])
    assert partial == before[1:]
    assert all(reference() is None for reference in references)
    assert all(
        len(tokenizer.encode(row["response"], add_special_tokens=False)) <= 2 for row in after
    )
