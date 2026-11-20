"""One GRPO GPU update of the configured model with synthetic rewards; no API calls."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

# Allow invocation as `python scripts/gpu_smoke.py` from this project.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from transformers import set_seed  # noqa: E402

from rlllmf.config import load_settings  # noqa: E402
from rlllmf.data import load_prepared, prepare, write_json  # noqa: E402
from rlllmf.offload import OffloadGRPOTrainer  # noqa: E402
from rlllmf.training import (  # noqa: E402
    hardware_info,
    load_text_model,
    load_tokenizer,
    make_grpo_config,
    require_gpu,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1, help="Optimizer steps to validate")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    settings = load_settings()
    require_gpu()
    set_seed(settings.training.seed)
    free, total = torch.cuda.mem_get_info()
    tokenizer = load_tokenizer(settings)
    prepare(settings, tokenizer)
    model = load_text_model(settings)
    prepared = load_prepared(settings, "train")
    lengths = [
        len(
            tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=settings.training.enable_thinking,
                return_dict=False,
            )
        )
        for row in prepared
    ]
    longest = max(range(len(prepared)), key=lengths.__getitem__)
    questions_per_update = (
        settings.training.micro_batch_size
        * settings.training.gradient_accumulation_steps
        // settings.training.num_generations
    )
    if len(prepared) < questions_per_update:
        raise ValueError("Not enough prepared instructions for a complete generation batch")
    indices_by_length = sorted(range(len(prepared)), key=lengths.__getitem__, reverse=True)
    data = prepared.select(indices_by_length[:questions_per_update])
    cache = settings.path(".cache")
    cache.mkdir(exist_ok=True)
    print("GPU validation only: synthetic rewards, no Anthropic calls, no experiment checkpoint.")
    print(json.dumps(hardware_info()))

    with tempfile.TemporaryDirectory(prefix="gpu-smoke-", dir=cache) as temporary:
        root = Path(temporary)
        initial = root / "initial"
        model.save_pretrained(initial)
        tokenizer.save_pretrained(initial)
        model.config._name_or_path = str(initial)
        config = make_grpo_config(settings, root / "training", smoke=True, tokenizer=tokenizer)
        config.save_strategy = "no"
        config.disable_tqdm = True
        config.max_steps = args.steps

        def synthetic_test_reward(completions, **kwargs):
            count = settings.training.num_generations
            return [float(i % count) / (count - 1) for i in range(len(completions))]

        trainer = OffloadGRPOTrainer(
            model=model,
            args=config,
            processing_class=tokenizer,
            train_dataset=data,
            reward_funcs=synthetic_test_reward,
            reference_cpu_offload_gib=settings.training.reference_cpu_offload_gib,
            trim_training_padding=settings.training.trim_training_padding,
        )
        # Probe small-magnitude weights so BF16 rounding does not hide a small update.
        probe_name, probe = next(
            (name, parameter)
            for name, parameter in reversed(list(model.named_parameters()))
            if parameter.requires_grad and parameter.ndim == 2 and parameter.numel() >= 64
        )
        indices = probe.detach().float().abs().flatten().topk(64, largest=False).indices
        before = probe.detach().flatten()[indices].float().cpu()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer.train()
        if trainer.state.global_step != args.steps:
            raise RuntimeError("The requested optimizer steps did not complete")
        torch.cuda.synchronize()
        after = probe.detach().flatten()[indices].float().cpu()
        changed = int((before != after).sum())
        if not changed or not torch.isfinite(after).all():
            raise RuntimeError("The test did not produce a finite, observable parameter update")
        optimizer = getattr(trainer.optimizer, "optimizer", trainer.optimizer)
        report = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "hardware": hardware_info(),
            "model": settings.training.model,
            "parameters": sum(p.numel() for p in model.parameters()),
            "all_language_parameters_trainable": all(p.requires_grad for p in model.parameters()),
            "optimizer": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
            "optimizer_state1_dtypes": sorted(
                {
                    str(state["state1"].dtype)
                    for state in optimizer.state.values()
                    if "state1" in state
                }
            ),
            "optimizer_step": trainer.state.global_step,
            "changed_probe_weights": changed,
            "probe_parameter": probe_name,
            "probe_weights": len(before),
            "num_generations": config.num_generations,
            "questions_per_update": questions_per_update,
            "prompt_tokens": lengths[longest],
            "max_completion_length": config.max_completion_length,
            "max_answer_tokens": settings.training.max_answer_tokens,
            "enable_thinking": settings.training.enable_thinking,
            "beta": config.beta,
            "reference_class": type(trainer.ref_model).__name__ if trainer.ref_model else None,
            "reference_cpu_offload": trainer.offload_statistics(),
            "elapsed_training_seconds": round(time.perf_counter() - started, 2),
            "initial_free_gib": round(free / 1024**3, 3),
            "total_vram_gib": round(total / 1024**3, 3),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            "paid_api_requests": 0,
            "reward": "synthetic test signal; this is not an optimism-training result",
        }
        output = settings.path("runs") / (
            "gpu-check-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json"
        )
        write_json(output, report)
        print(json.dumps(report, indent=2))
        print(f"GPU test report: {output}")


if __name__ == "__main__":
    main()
