import json

import pytest


@pytest.mark.integration
@pytest.mark.parametrize(
    ("family", "cpu_offload_gib"), [("qwen", 0), ("lfm3", 0), ("lfm3", 0.00008)]
)
def test_tiny_full_grpo_step(tmp_path, family, cpu_offload_gib):
    """Real hybrid model generation/backward/KL on CPU; fake judge, no network."""
    import torch
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        Lfm3Config,
        Lfm3ForCausalLM,
        PreTrainedTokenizerFast,
        Qwen3_5ForCausalLM,
        Qwen3_5TextConfig,
    )
    from trl import GRPOConfig

    from rlllmf.judge import JudgeRequestError, OptimismJudge
    from rlllmf.offload import OffloadGRPOTrainer
    from rlllmf.training import load_text_model, save_judge_failure_checkpoint
    from tests.test_core import judgment, temporary_settings

    torch.set_num_threads(1)
    torch.manual_seed(42)
    words = ["<pad>", "<eos>", "<bos>", "<unk>", "user", "assistant", "task"]
    words += [f"word{i}" for i in range(25)]
    raw = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
        padding_side="left",
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    model_config = Qwen3_5TextConfig(
        vocab_size=len(words),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        max_position_embeddings=128,
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "mrope_section": [2, 1, 1],
        },
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    model_class = Qwen3_5ForCausalLM
    if family == "lfm3":
        model_config = Lfm3Config(
            vocab_size=len(words),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            conv_L_cache=3,
            max_position_embeddings=128,
            layer_types=["conv", "sliding_attention", "full_attention"],
            sliding_window=8,
            eos_token_id=1,
            pad_token_id=0,
            bos_token_id=2,
        )
        model_class = Lfm3ForCausalLM
    model = model_class(model_config)
    initial = tmp_path / "initial"
    model.save_pretrained(initial)
    tokenizer.save_pretrained(initial)
    settings = temporary_settings(tmp_path)
    model = load_text_model(settings, str(initial), device="cpu")
    model.config._name_or_path = str(initial)
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    settings.training.num_generations = 2
    calls = []

    async def fake_request(payload):
        calls.append(payload)
        return {
            "scores": [
                judgment(row["id"], float(row["id"][1:]) / max(1, len(payload["candidates"]) - 1))
                for row in reversed(payload["candidates"])
            ]
        }

    judge = OptimismJudge(settings, request=fake_request, tokenizer=tokenizer)
    config = GRPOConfig(
        output_dir=str(tmp_path / "run"),
        use_cpu=True,
        bf16=False,
        fp16=False,
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        num_generations=2,
        max_completion_length=4,
        learning_rate=1e-3,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.01,
        optim="adamw_torch",
        use_vllm=False,
        report_to="none",
        save_strategy="no",
        logging_steps=1,
        disable_tqdm=True,
        model_init_kwargs={"device_map": None, "dtype": "float32"},
    )
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "task"}]}])
    try:
        trainer = OffloadGRPOTrainer(
            model=model,
            processing_class=tokenizer,
            args=config,
            reward_funcs=judge.reward_sync,
            train_dataset=dataset,
            reference_cpu_offload_gib=cpu_offload_gib,
        )
        reference_before = {
            name: p.detach().clone() for name, p in trainer.ref_model.named_parameters()
        }
        assert trainer.ref_model.__class__ is model_class
        trainer.train()
        assert trainer.state.global_step == 1
        if cpu_offload_gib:
            assert 0 < trainer.offload_statistics()["offloaded_bytes"] <= cpu_offload_gib * 1024**3
            assert trainer.offload_statistics()["reference_batches"] == 1
            assert len(trainer.reference_offloader.layers) < model_config.num_hidden_layers
        assert calls
        assert all(p.requires_grad for p in model.parameters())
        assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())
        assert all(
            torch.equal(reference_before[name], p)
            for name, p in trainer.ref_model.named_parameters()
        )
        trainer.save_model(str(tmp_path / "final"))
        saved = json.loads((tmp_path / "final/config.json").read_text())
        assert saved["model_type"] == model_config.model_type
        checkpoint = save_judge_failure_checkpoint(
            trainer, tokenizer, tmp_path / "run", JudgeRequestError("synthetic refusal")
        )
        assert checkpoint.name == "checkpoint-1"
        assert all((checkpoint / name).is_file() for name in [
            "model.safetensors", "optimizer.pt", "scheduler.pt", "rng_state.pth",
        ])
        assert json.loads((checkpoint / "trainer_state.json").read_text())["global_step"] == 1
    finally:
        judge.close()
