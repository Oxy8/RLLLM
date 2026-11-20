import copy
import json

import pytest
import torch

from rlllmf.judge import JudgeRequestError, JudgeResponseError, OptimismJudge, plain_text
from rlllmf.rollouts import ReplacingGRPOTrainer
from tests.test_core import judgment, temporary_settings


@pytest.fixture
def tiny_setup(tmp_path):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Lfm3Config, Lfm3ForCausalLM, PreTrainedTokenizerFast
    from trl import GRPOConfig

    torch.set_num_threads(1)
    torch.manual_seed(42)
    words = ["<pad>", "<eos>", "<unk>", "user", "assistant", "task"]
    words += [f"word{i}" for i in range(26)]
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>",
                                       eos_token="<eos>", unk_token="<unk>", padding_side="left")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    model = Lfm3ForCausalLM(Lfm3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, conv_L_cache=3,
        layer_types=["conv", "full_attention"], max_position_embeddings=128,
        pad_token_id=0, eos_token_id=1, bos_token_id=None,
    ))
    initial = tmp_path / "initial"
    model.save_pretrained(initial)
    model.config._name_or_path = str(initial)
    settings = temporary_settings(tmp_path)
    settings.training.num_generations = 2
    settings.training.max_completion_length = 5
    settings.judge.max_retries = 1
    args = GRPOConfig(
        output_dir=str(tmp_path / "run"), use_cpu=True, bf16=False, fp16=False,
        max_steps=1, per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=2, max_completion_length=5, beta=.01, optim="adamw_torch",
        learning_rate=1e-3, report_to="none", disable_tqdm=True, logging_steps=1,
        save_strategy="no", model_init_kwargs={"device_map": None, "dtype": "float32"},
    )
    dataset = Dataset.from_list([
        {"prompt": [{"role": "user", "content": f"task word{i}"}], "prompt_id": str(i)}
        for i in range(8)
    ])
    return model, tokenizer, settings, args, dataset


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["none", "refusal", "invalid", "max_tokens"])
@pytest.mark.parametrize("reward_mode", ["trait_only", "quality_gated", "quality_softplus"])
def test_replacements_preserve_good_groups_batch_size_kl_and_rewards(
    tmp_path, tiny_setup, failure, reward_mode,
):
    model, tokenizer, settings, args, dataset = tiny_setup
    settings.judge.reward_mode = reward_mode
    quality_mode = settings.judge.uses_task_quality
    calls, generated, accepted = [], [], []
    failed_instruction = None

    async def request(payload):
        nonlocal failed_instruction
        calls.append(payload)
        if failed_instruction is None:
            failed_instruction = payload["instruction"]
        if failure != "none" and payload["instruction"] == failed_instruction:
            if failure == "invalid":
                return {"scores": []}
            raise JudgeResponseError(failure)
        if quality_mode:
            from tests.test_quality_reward import dual_score

            return {"scores": [dual_score(row["id"], .1 if row["id"] == "c0" else .9,
                                          .9 if row["id"] == "c0" else .7, "optimism")
                               for row in payload["candidates"]]}
        return {"scores": [judgment(row["id"], .1 if row["id"] == "c0" else .9)
                           for row in payload["candidates"]]}

    class CapturingTrainer(ReplacingGRPOTrainer):
        def _judge_rollout(self, prompts, output):
            result = super()._judge_rollout(prompts, output)
            generated.append((copy.deepcopy(prompts), copy.deepcopy(output), copy.deepcopy(result)))
            return result

        def _calculate_rewards(self, inputs, prompts, completions, ids):
            result = super()._calculate_rewards(inputs, prompts, completions, ids)
            accepted.append((copy.deepcopy(prompts), copy.deepcopy(ids), result.cpu()))
            return result

    judge = OptimismJudge(settings, tmp_path / "judge.jsonl", request=request, tokenizer=tokenizer)
    try:
        trainer = CapturingTrainer(
            model=model, processing_class=tokenizer, args=args,
            train_dataset=dataset, reward_funcs=judge.reward_sync, judge=judge,
        )
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        reference_before = {name: p.detach().clone()
                            for name, p in trainer.ref_model.named_parameters()}
        trainer.train()
        assert trainer.state.global_step == 1 and len(accepted) == 1
        prompts, ids, rewards = accepted[0]
        assert len(prompts) == len(ids) == len(rewards) == 4
        assert rewards.shape == (4, 1) and torch.isfinite(rewards).all()
        assert trainer._buffered_inputs[0]["ref_per_token_logps"] is not None
        assert any(not torch.equal(before[name], p) for name, p in model.named_parameters())
        assert all(torch.equal(reference_before[name], p) and p.grad is None
                   for name, p in trainer.ref_model.named_parameters())
        records = [json.loads(line) for line in (tmp_path / "judge.jsonl").read_text().splitlines()]
        assert len(records) == 2 and all(row["training_step"] == 1 for row in records)
        assert not any(row["cached"] for row in records)  # no second scoring pass
        if quality_mode:
            metrics = next(row for row in trainer.state.log_history if "reward" in row)
            assert metrics["judge/task_quality"] == pytest.approx(.8)
            assert metrics["judge/optimism"] == pytest.approx(.5)
            assert metrics["judge/combined_reward"] == pytest.approx(rewards.mean().item())
        if failure == "none":
            assert len(calls) == 2 and len(generated) == 1
            assert not trainer.skipped_questions
            # Moving judging ahead of KL must leave a successful update identical.
            from rlllmf.offload import OffloadGRPOTrainer

            expected = {name: p.detach().clone() for name, p in model.named_parameters()}
            control_model = model.__class__.from_pretrained(tmp_path / "initial", dtype="float32")
            control_args = copy.deepcopy(args)
            control_args.output_dir = str(tmp_path / "control")
            control = OffloadGRPOTrainer(
                model=control_model, processing_class=tokenizer, args=control_args,
                train_dataset=dataset, reward_funcs=judge.reward_sync,
            )
            control.train()
            for name, parameter in control_model.named_parameters():
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        else:
            assert trainer.skipped_questions == trainer.replacement_questions == 1
            assert len(generated) == 2 and len(generated[1][0]) == 2
            assert len(calls) == (4 if failure == "invalid" else 3)
            assert failed_instruction not in {plain_text(p) for p in prompts}
            # The other original question retains its exact token IDs and aligned scores.
            first_prompts, first_output, first_results = generated[0]
            for group, result in enumerate(first_results):
                if not isinstance(result, JudgeResponseError):
                    start = group * 2
                    assert prompts[start:start + 2] == first_prompts[start:start + 2]
                    assert ids[start:start + 2] == first_output[1][start:start + 2]
                    torch.testing.assert_close(rewards[start:start + 2, 0], torch.tensor(result))
            skipped = json.loads((tmp_path / "run/skipped-questions.jsonl").read_text())
            assert skipped["instruction"] == failed_instruction
            assert skipped["replacement_instruction"] != failed_instruction
    finally:
        judge.close()


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["fatal", "always_refuse"])
def test_fatal_errors_and_replacement_cap_stop_without_weight_update(tiny_setup, failure):
    model, tokenizer, settings, args, dataset = tiny_setup
    calls = []

    async def request(payload):
        calls.append(payload)
        error = JudgeRequestError if failure == "fatal" else JudgeResponseError
        raise error("API unavailable" if failure == "fatal" else "refusal")

    judge = OptimismJudge(settings, request=request, tokenizer=tokenizer)
    try:
        trainer = ReplacingGRPOTrainer(
            model=model, processing_class=tokenizer, args=args, train_dataset=dataset,
            reward_funcs=judge.reward_sync, judge=judge, max_replacements_per_step=2,
        )
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        with pytest.raises(JudgeRequestError, match="API unavailable|replacement limit"):
            trainer.train()
        assert trainer.state.global_step == 0
        assert all(torch.equal(before[name], p) for name, p in model.named_parameters())
        assert len(calls) == (2 if failure == "fatal" else 4)
        assert judge.db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0] == 0
    finally:
        judge.close()
