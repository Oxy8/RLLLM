import pytest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Lfm3Config, Lfm3ForCausalLM, PreTrainedTokenizerFast
from trl import GRPOConfig

from rlllmf.config import validate_settings
from rlllmf.offload import OffloadGRPOTrainer
from tests.test_core import temporary_settings


@pytest.mark.parametrize("microbatch", [1, 2])
def test_trimmed_lfm_stats_and_grpo_gradients_match_padded_computation(tmp_path, microbatch):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "task": 3}
    vocab.update({f"t{i}": i for i in range(4, 32)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        pad_token="<pad>", eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    model = Lfm3ForCausalLM(Lfm3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, conv_L_cache=7,
        layer_types=["conv", "full_attention"], max_position_embeddings=128,
        pad_token_id=0, eos_token_id=1,
    ))
    initial = tmp_path / "initial"
    model.save_pretrained(initial)
    model.config._name_or_path = str(initial)
    trainer = OffloadGRPOTrainer(
        model=model, processing_class=tokenizer,
        args=GRPOConfig(
            output_dir=str(tmp_path / "training"), use_cpu=True, bf16=False, fp16=False,
            max_steps=1, per_device_train_batch_size=microbatch,
            gradient_accumulation_steps=4 // microbatch, num_generations=2,
            max_completion_length=24, beta=.01, optim="adamw_torch",
            save_strategy="no", report_to="none",
            model_init_kwargs={"device_map": None, "dtype": "float32"},
        ),
        train_dataset=Dataset.from_list([{"prompt": "task"}]),
        reward_funcs=lambda completions, **kwargs: [0.] * len(completions),
        reference_cpu_offload_gib=1,
    )
    trainer.current_gradient_accumulation_steps = 4 // microbatch
    ids = torch.randint(3, 32, (4, 29))
    mask = torch.ones_like(ids)
    mask[:, 0] = 0  # Preserve left-padded prompts and their position conventions.
    lengths = [2, 7, 24, 0]
    for i, length in enumerate(lengths):
        mask[i, 5 + length:] = 0
        if length:
            ids[i, 5 + length - 1] = tokenizer.eos_token_id
    ids[mask == 0] = 0

    outputs = []
    gradients = []
    losses = []
    widths = []

    def observe(module, args, kwargs):
        widths.append(kwargs["input_ids"].shape[1])

    hook = trainer.model.model.register_forward_pre_hook(observe, with_kwargs=True)
    for enabled in [False, True]:
        trainer.trim_training_padding = enabled
        widths.clear()
        stats = trainer._get_per_token_logps_and_entropies(
            trainer.model, ids, mask, 24, batch_size=microbatch, compute_entropy=True,
        )
        outputs.append(tuple(x.detach().clone() if x is not None else None for x in stats))
        if enabled:
            expected = [5 + max(1, max(lengths[i:i + microbatch]))
                        for i in range(0, 4, microbatch)]
            assert widths == expected
        else:
            assert widths == [29] * (4 // microbatch)
        with torch.no_grad():
            ref, _, _ = trainer._get_per_token_logps_and_entropies(
                trainer.ref_model, ids, mask, 24, batch_size=microbatch,
            )
        trainer.model.zero_grad(set_to_none=True)
        loss_sum = 0.
        for i in range(0, 4, microbatch):
            inputs = {
                "prompt_ids": ids[i:i + microbatch, :5],
                "prompt_mask": mask[i:i + microbatch, :5],
                "completion_ids": ids[i:i + microbatch, 5:],
                "completion_mask": mask[i:i + microbatch, 5:],
                "advantages": torch.tensor([-1., 1., -.5, .5])[i:i + microbatch],
                "ref_per_token_logps": ref[i:i + microbatch] + .1,
                "num_items_in_batch": mask[:, 5:].sum(),
            }
            loss = trainer._compute_loss(trainer.model, inputs)
            loss_sum += loss.detach().item()
            loss.backward()
        losses.append(loss_sum)
        gradients.append({n: p.grad.clone() for n, p in model.named_parameters()
                          if p.grad is not None})
    hook.remove()
    for padded, trimmed in zip(outputs[0], outputs[1], strict=True):
        if padded is not None:
            torch.testing.assert_close(padded, trimmed, rtol=1e-5, atol=1e-6)
    assert losses[0] == pytest.approx(losses[1], rel=1e-5, abs=1e-6)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=1e-4, atol=1e-6)
    # One transfer context for the entire reference call, not one per chunk.
    assert trainer.reference_offloader.cycles == 2


@pytest.mark.parametrize("value", [1, "true", None])
def test_padding_setting_requires_a_boolean(tmp_path, value):
    settings = temporary_settings(tmp_path)
    settings.training.trim_training_padding = value
    with pytest.raises(ValueError, match="trim_training_padding"):
        validate_settings(settings)
