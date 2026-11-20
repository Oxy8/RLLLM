import pytest

from rlllmf.baseline_cache import baseline_answers
from rlllmf.config import validate_settings
from rlllmf.evaluation import evaluation_generation_kwargs
from tests.test_answers import CharacterTokenizer
from tests.test_core import temporary_settings


def question(name):
    return {"prompt_id": name, "prompt": [{"role": "user", "content": f"task {name}"}]}


@pytest.fixture
def cached_setup(tmp_path):
    settings = temporary_settings(tmp_path)
    model = tmp_path / "original"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"test"}')
    (model / "model.safetensors").write_bytes(b"original weights")
    tokenizer = CharacterTokenizer()
    generated = []

    def generate(settings, data, source, *, seed_indices=None, on_record=None):
        generated.append({"ids": [row["prompt_id"] for row in data], "indices": seed_indices})
        records = []
        for row in data:
            record = {
                "prompt_id": row["prompt_id"],
                "instruction": row["prompt"][0]["content"],
                "response": "answer",
                "generated_tokens": 10,
                "generation_diagnostics": {"generated_tokens": 10},
            }
            if on_record:
                on_record(record)
            records.append(record)
        return records

    return settings, model, tokenizer, generated, generate


def call(setup, data, source=None, generator=None):
    settings, model, tokenizer, _, generate = setup
    return baseline_answers(
        settings, data, str(source or model), generator or generate, lambda *args: tokenizer
    )


def test_reuses_baseline_and_generates_only_new_questions(cached_setup):
    first, info = call(cached_setup, [question("a")])
    assert info["hits"] == 0 and info["misses"] == 1
    second, info = call(cached_setup, [question("a"), question("b")])
    assert info["hits"] == 1 and info["misses"] == 1
    assert first[0] == second[0]
    assert cached_setup[3][-1] == {"ids": ["b"], "indices": [1]}
    call(cached_setup, [question("b"), question("a")])
    assert len(cached_setup[3]) == 2  # Greedy decoding is independent of question order.


def test_same_model_copy_reuses_cache_and_weight_changes_invalidate(cached_setup, tmp_path):
    import shutil

    call(cached_setup, [question("a")])
    other = tmp_path / "copy"
    shutil.copytree(cached_setup[1], other)
    _, info = call(cached_setup, [question("a")], source=other)
    assert info["hits"] == 1
    (other / "model.safetensors").write_bytes(b"modified weights")
    _, info = call(cached_setup, [question("a")], source=other)
    assert info["misses"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_completion_length", 99),
        ("max_answer_tokens", 33),
        ("enable_thinking", False),
        ("temperature", 0.6),
    ],
)
def test_changed_generation_settings_invalidate(cached_setup, field, value):
    call(cached_setup, [question("a")])
    setattr(cached_setup[0].evaluation, field, value)
    _, info = call(cached_setup, [question("a")])
    assert info["misses"] == 1


def test_question_template_and_seed_changes_are_handled(cached_setup):
    settings, _, tokenizer, _, _ = cached_setup
    call(cached_setup, [question("a")])
    changed = question("a")
    changed["prompt"][0]["content"] = "different task"
    assert call(cached_setup, [changed])[1]["misses"] == 1
    tokenizer.chat_template = "new formatting"
    assert call(cached_setup, [question("a")])[1]["misses"] == 1
    settings.evaluation.temperature = 0.8
    call(cached_setup, [question("a")])
    settings.training.seed += 1
    assert call(cached_setup, [question("a")])[1]["misses"] == 1


def test_completed_answers_survive_interrupted_generation(cached_setup):
    def interrupted(settings, data, source, **kwargs):
        cached_setup[4](settings, data[:1], source, **kwargs)
        raise RuntimeError("interrupted")

    data = [question("a"), question("b")]
    with pytest.raises(RuntimeError, match="interrupted"):
        call(cached_setup, data, generator=interrupted)
    records, info = call(cached_setup, data)
    assert len(records) == 2 and info["hits"] == 1 and info["misses"] == 1


@pytest.mark.parametrize("score_name", ["childishness", "conciseness"])
def test_switching_judge_trait_reuses_original_generations(cached_setup, score_name):
    settings, _, _, generated, _ = cached_setup
    call(cached_setup, [question("a")])
    settings.judge.score_name = score_name
    (settings.root / "rubric.txt").write_text(f"Score {score_name}.")
    _, info = call(cached_setup, [question("a")])
    assert info["hits"] == 1 and len(generated) == 1


def test_zero_evaluation_temperature_is_greedy_but_training_requires_sampling(tmp_path):
    settings = temporary_settings(tmp_path)
    settings.evaluation.temperature = 0.0
    validate_settings(settings)
    kwargs = evaluation_generation_kwargs(settings)
    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] is None and kwargs["top_p"] is None
    settings.training.temperature = 0.0
    with pytest.raises(ValueError, match="training"):
        validate_settings(settings)
