"""Full-parameter answer-only supervised warm-up before a separate GRPO run."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from .collection import collection_lock, read_examples, write_jsonl
from .config import Settings
from .data import normalized_prompt_id, write_json


def load_sft_examples(settings: Settings, snapshot: Path | None = None) -> tuple[list, dict]:
    source = snapshot or settings.path(settings.sft.dataset_path)
    pool_path = settings.path(settings.collection.output_dir) / "pool.jsonl"
    if not source.is_file() or not pool_path.is_file():
        raise ValueError("SFT examples or pool missing. Collect the teacher examples first.")
    with collection_lock(source.parent):
        pool = [json.loads(line) for line in pool_path.read_text().splitlines()]
        rows = read_examples(source, pool)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if not rows:
        raise ValueError("The SFT dataset has no examples")
    ids = {normalized_prompt_id(r["instruction"]) for r in rows}
    for split in ["train", "eval"]:
        path = settings.path(settings.data.output_dir) / f"{split}.jsonl"
        if path.is_file():
            prepared = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if ids & {normalized_prompt_id(r["prompt"][0]["content"]) for r in prepared}:
                raise ValueError(f"SFT questions overlap prepared {split} data")
    return rows, {"path": str(source), "sha256": digest, "examples": len(rows),
                  "prepared_rl_eval_overlap": 0}


def tokenize_sft_examples(rows, tokenizer, max_length: int):
    """Mask the whole user turn and answer prefix; supervise answer + turn ending."""
    tokenized = []
    lengths, target_lengths = [], []
    for row in rows:
        messages = row["messages"]
        options = {"tokenize": True, "return_dict": False, "enable_thinking": False}
        prompt = tokenizer.apply_chat_template(messages[:1], add_generation_prompt=True, **options)
        full = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **options)
        if full[:len(prompt)] != prompt or len(full) <= len(prompt):
            raise ValueError("SFT chat template must preserve the prompt and include an answer")
        if len(full) > max_length:
            raise ValueError(
                f"SFT example {row['prompt_id'][:12]} has {len(full)} tokens, exceeding "
                f"sft.max_seq_length={max_length}. No answers were truncated. "
                "Increase the limit or curate the dataset."
            )
        labels = [-100] * len(prompt) + full[len(prompt):]
        tokenized.append({"input_ids": full, "labels": labels})
        lengths.append(len(full))
        target_lengths.append(len(full) - len(prompt))
    return tokenized, {
        "examples": len(rows), "max_sequence_tokens": max(lengths),
        "mean_sequence_tokens": statistics.mean(lengths),
        "supervised_tokens_per_epoch": sum(target_lengths),
        "completion_only_loss": True, "teacher_system_prompt_included": False,
        "thinking_targets_included": False, "enable_thinking_in_sft_template": False,
        "truncated_examples": 0,
    }


def make_sft_config(settings: Settings, output: Path, *, use_cpu=False):
    import torch
    from trl import SFTConfig

    s = settings.sft
    bf16 = not use_cpu and torch.cuda.is_bf16_supported()
    return SFTConfig(
        output_dir=str(output), num_train_epochs=s.num_train_epochs,
        learning_rate=s.learning_rate, lr_scheduler_type=s.lr_scheduler_type,
        per_device_train_batch_size=s.micro_batch_size,
        gradient_accumulation_steps=s.gradient_accumulation_steps,
        seed=settings.training.seed, data_seed=settings.training.seed,
        bf16=bf16, fp16=not use_cpu and not bf16, use_cpu=use_cpu,
        optim=settings.training.optimizer,
        gradient_checkpointing=settings.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=s.max_seq_length, packing=False, completion_only_loss=True,
        dataset_kwargs={"skip_prepare_dataset": True},
        save_strategy="epoch", save_total_limit=1, logging_steps=1,
        dataloader_num_workers=0, report_to="none", push_to_hub=False,
    )


def sft(settings: Settings, output=None, resume=None, dry_run=False) -> dict:
    from .training import load_tokenizer

    run_dir = settings.path(output or settings.sft.output_dir)
    if (run_dir / "run.json").exists() and not resume and not dry_run:
        raise ValueError("SFT run already exists. Use --resume latest or a new --output.")
    checkpoint = None
    if resume:
        from transformers.trainer_utils import get_last_checkpoint

        if not (run_dir / "run.json").is_file():
            raise ValueError("No SFT run to resume")
        previous = json.loads((run_dir / "run.json").read_text())
        if previous.get("stage") != "sft":
            raise ValueError("This is not an SFT run")
        checkpoint = get_last_checkpoint(str(run_dir)) if resume == "latest" else str(
            settings.path(resume)
        )
        if not checkpoint or not (Path(checkpoint) / "trainer_state.json").is_file():
            raise ValueError("No complete SFT checkpoint found to resume")
        rows, source = load_sft_examples(settings, run_dir / "training-data.jsonl")
        if source["sha256"] != previous["snapshot_sha256"]:
            raise ValueError("The SFT run's frozen training dataset changed")
    else:
        rows, source = load_sft_examples(settings)
    tokenizer = load_tokenizer(settings, checkpoint)
    tokenized, report = tokenize_sft_examples(rows, tokenizer, settings.sft.max_seq_length)
    per_epoch = math.ceil(math.ceil(len(rows) / settings.sft.micro_batch_size)
                          / settings.sft.gradient_accumulation_steps)
    report.update({"updates_per_epoch": per_epoch,
                   "expected_optimizer_updates": per_epoch * settings.sft.num_train_epochs,
                   "epochs": settings.sft.num_train_epochs,
                   "learning_rate": settings.sft.learning_rate, "source": source,
                   "output": str(run_dir)})
    print(json.dumps(report, indent=2), flush=True)
    if dry_run:
        return report

    import torch
    from datasets import Dataset
    from transformers import TrainerCallback, set_seed
    from trl import SFTTrainer

    from .training import hardware_info, load_text_model, require_gpu

    require_gpu()
    set_seed(settings.training.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        write_jsonl(run_dir / "training-data.jsonl", rows)
    snapshot_hash = hashlib.sha256((run_dir / "training-data.jsonl").read_bytes()).hexdigest()
    config = make_sft_config(settings, run_dir)
    model = load_text_model(settings, checkpoint)
    if any(not p.requires_grad for p in model.parameters()):
        raise ValueError("SFT requires all model parameters to be trainable")
    manifest = {"stage": "sft", "settings": settings.public_dict(),
                "hardware": hardware_info(), "data": report, "resume": resume,
                "snapshot_sha256": snapshot_hash,
                "language_parameters": sum(p.numel() for p in model.parameters()),
                "all_language_parameters_trainable": True,
                "checkpoint_policy": "once per epoch, keep latest one; final export separately"}
    write_json(run_dir / ("resume.json" if resume else "run.json"), manifest)

    class MemoryLogger(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            record = {"step": state.global_step,
                      "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
                      "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                      "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3)}
            with (run_dir / "memory.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"GPU memory: {record}", flush=True)

    trainer = SFTTrainer(model=model, args=config, train_dataset=Dataset.from_list(tokenized),
                         processing_class=tokenizer, callbacks=[MemoryLogger()])
    torch.cuda.reset_peak_memory_stats()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(run_dir / "final"))
    tokenizer.save_pretrained(run_dir / "final")
    trainer.save_state()
    write_json(run_dir / "metrics.json", result.metrics)
    print(f"SFT model saved: {run_dir / 'final'}", flush=True)
    return result.metrics
