import hashlib
import json

import pytest

from rlllmf.data import normalized_prompt_id
from rlllmf.sft import load_sft_examples, make_sft_config, tokenize_sft_examples
from rlllmf.training import resolve_starting_model
from tests.test_core import temporary_settings


def example(question="question", answer="playful answer"):
    return {"prompt_id": hashlib.sha256(question.encode()).hexdigest(),
            "instruction": question, "answer": answer,
            "messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": answer}]}


class TemplateTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        assert kwargs["enable_thinking"] is False
        prefix = [1, 2, 3]  # user turn + empty thinking scaffold, all masked
        return prefix if add_generation_prompt else prefix + [7, 8, 9, 0]


def test_sft_masks_prompt_and_thinking_scaffold_but_trains_answer_and_end():
    tokens, report = tokenize_sft_examples([example()], TemplateTokenizer(), 20)
    assert tokens == [{"input_ids": [1, 2, 3, 7, 8, 9, 0],
                       "labels": [-100, -100, -100, 7, 8, 9, 0]}]
    assert report["supervised_tokens_per_epoch"] == 4
    assert report["thinking_targets_included"] is False
    with pytest.raises(ValueError, match="No answers were truncated"):
        tokenize_sft_examples([example()], TemplateTokenizer(), 5)


def test_sft_validates_collection_and_separation(tmp_path):
    s = temporary_settings(tmp_path)
    directory = s.path(s.collection.output_dir)
    directory.mkdir(parents=True)
    row = example()
    (directory / "examples.jsonl").write_text(json.dumps(row) + "\n")
    (directory / "pool.jsonl").write_text(json.dumps({
        "prompt_id": row["prompt_id"], "instruction": row["instruction"],
    }) + "\n")
    rows, report = load_sft_examples(s)
    assert rows == [row] and report["prepared_rl_eval_overlap"] == 0
    path = s.path(s.data.output_dir) / "eval.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompt": [{"role": "user", "content": " QUESTION "}]}) + "\n")
    assert normalized_prompt_id(" QUESTION ") == normalized_prompt_id("question")
    with pytest.raises(ValueError, match="overlap prepared eval"):
        load_sft_examples(s)


def test_local_model_override_requires_a_complete_saved_model(tmp_path):
    s = temporary_settings(tmp_path)
    folder = tmp_path / "saved"
    folder.mkdir()
    (folder / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="weights"):
        resolve_starting_model(s, "saved")
    (folder / "model.safetensors").write_bytes(b"test")
    assert resolve_starting_model(s, "saved") == str(folder.resolve())
    with pytest.raises(ValueError, match="does not exist"):
        resolve_starting_model(s, "runs/not-yet-trained/final")


@pytest.mark.integration
def test_real_sft_save_then_grpo_uses_sft_weights_as_policy_and_reference(tmp_path, monkeypatch):
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Lfm3Config, Lfm3ForCausalLM, PreTrainedTokenizerFast
    from trl import GRPOConfig, SFTTrainer

    import rlllmf.training as training

    torch.set_num_threads(1)
    torch.manual_seed(42)
    words = ["<pad>", "<eos>", "<unk>", "user", "assistant", "question",
             "playful", "answer", "hello", "world"]
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>",
                                       eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} "
        "{% if m['role'] == 'assistant' %}{{ eos_token }}{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    model = Lfm3ForCausalLM(Lfm3Config(
        vocab_size=len(words), hidden_size=32, intermediate_size=64, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, conv_L_cache=3,
        layer_types=["conv", "sliding_attention", "full_attention"], sliding_window=8,
        max_position_embeddings=128, pad_token_id=0, eos_token_id=1,
    ))
    original_weight = model.get_input_embeddings().weight.detach().clone()
    s = temporary_settings(tmp_path)
    s.sft.num_train_epochs = 1
    s.sft.learning_rate = 1e-3
    s.sft.gradient_accumulation_steps = 1
    s.training.optimizer = "adamw_torch"
    tokenized, _ = tokenize_sft_examples([example(), example(answer="hello world")], tokenizer, 128)
    config = make_sft_config(s, tmp_path / "sft", use_cpu=True)
    config.disable_tqdm = True
    trainer = SFTTrainer(model=model, args=config, processing_class=tokenizer,
                         train_dataset=Dataset.from_list(tokenized))
    assert trainer.train_dataset[0]["labels"] == tokenized[0]["labels"]
    result = trainer.train()
    assert result.global_step == 2
    assert not torch.equal(original_weight, model.get_input_embeddings().weight)
    final = tmp_path / "sft/final"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    expected = {k: v.detach().clone() for k, v in model.state_dict().items()}
    observed = {}
    original_loader = training.load_text_model

    from rlllmf import rollouts

    class CapturingGRPO(rollouts.ReplacingGRPOTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            observed["policy"] = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            observed["reference"] = {
                k: v.detach().clone() for k, v in self.ref_model.state_dict().items()
            }

    class FakeJudge:
        rubric = "childlike voice"

        def __init__(self, *args, **kwargs):
            pass

        def reward_sync(self, prompts, completions, return_groups=False, **kwargs):
            rewards = [float(i % 2) for i in range(len(completions))]
            return [rewards] if return_groups else rewards

        def statistics(self):
            return {"api_requests": 0}

        def close(self):
            pass

    def grpo_config(settings, output, *args, **kwargs):
        return GRPOConfig(
            output_dir=str(output), use_cpu=True, bf16=False, fp16=False,
            max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=2,
            num_generations=2, max_completion_length=4, beta=.01, optim="adamw_torch",
            report_to="none", disable_tqdm=True,
            model_init_kwargs={"device_map": None, "dtype": "float32"},
        )

    monkeypatch.setattr(training, "require_gpu", lambda: None)
    monkeypatch.setattr(training, "hardware_info", lambda: {"cuda_available": False})
    monkeypatch.setattr(training, "prepare", lambda *args: {})
    monkeypatch.setattr(training, "load_prepared", lambda *args: Dataset.from_list([
        {"prompt": [{"role": "user", "content": "question"}]}]))
    monkeypatch.setattr(
        training, "load_text_model",
        lambda settings, source=None: original_loader(settings, source, device="cpu"),
    )
    monkeypatch.setattr(training, "make_grpo_config", grpo_config)
    monkeypatch.setattr(training, "OptimismJudge", FakeJudge)
    monkeypatch.setattr(rollouts, "ReplacingGRPOTrainer", CapturingGRPO)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    s.training.reference_cpu_offload_gib = 0
    s.training.num_generations = 2
    output = tmp_path / "grpo"
    training.train(s, output=str(output), model_source=str(final))
    assert (output / "final/config.json").is_file()
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["settings"]["training"]["model"] == str(final.resolve())
    for key in expected:
        torch.testing.assert_close(observed["policy"][key], expected[key], rtol=0, atol=0)
        torch.testing.assert_close(observed["reference"][key], expected[key], rtol=0, atol=0)
