from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import Settings, require_api_key
from .data import load_prepared, prepare, write_json
from .judge import JudgeRequestError, OptimismJudge


def hardware_info() -> dict:
    import torch

    info = {"torch": torch.__version__, "cuda_available": torch.cuda.is_available()}
    if info["cuda_available"]:
        device = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu": device.name,
                "vram_gib": round(device.total_memory / 1024**3, 2),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    return info


def require_gpu():
    import torch

    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU not visible. Run training on your NVIDIA GPU machine.")
    if torch.cuda.device_count() != 1:
        raise ValueError("This setup is single-GPU. Set CUDA_VISIBLE_DEVICES to one GPU.")


def load_tokenizer(settings: Settings, source: str | None = None):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        source or settings.training.model,
        **({"revision": settings.training.revision} if source is None else {}),
        token=settings.api.hf_token or None,
        cache_dir=str(settings.path(".cache/huggingface")),
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_text_model(settings: Settings, source: str | None = None, *, device="cuda"):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, Qwen3_5ForCausalLM

    # FP16 autocast uses FP32 parameters on older GPUs so GradScaler can unscale
    # gradients safely. BF16 GPUs use actual BF16 parameters for the smaller footprint.
    dtype = torch.bfloat16 if device != "cpu" and torch.cuda.is_bf16_supported() else torch.float32
    options = {
        "token": settings.api.hf_token or None,
        "cache_dir": str(settings.path(".cache/huggingface")),
        "trust_remote_code": False,
        **({"revision": settings.training.revision} if source is None else {}),
    }
    target = source or settings.training.model
    config = AutoConfig.from_pretrained(target, **options)
    # Qwen's original checkpoint includes vision; retain the explicit text loader.
    # LFM3 and other native text architectures use Transformers' causal-LM factory.
    loader = Qwen3_5ForCausalLM if config.model_type == "qwen3_5" else AutoModelForCausalLM
    model = loader.from_pretrained(
        target,
        **options,
        dtype=dtype,
        attn_implementation="sdpa",
        device_map=None,
    )
    model.to(device)
    return model


def make_grpo_config(settings: Settings, output: Path, smoke: bool = False, *, tokenizer=None):
    import torch
    from trl import GRPOConfig

    t = settings.training
    bf16 = torch.cuda.is_bf16_supported()
    return GRPOConfig(
        output_dir=str(output),
        seed=t.seed,
        data_seed=t.seed,
        max_steps=1 if smoke else t.max_steps,
        learning_rate=t.learning_rate,
        lr_scheduler_type=t.lr_scheduler_type,
        per_device_train_batch_size=t.micro_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        num_generations=t.num_generations,
        max_completion_length=t.max_completion_length,
        temperature=t.temperature,
        top_p=t.top_p,
        top_k=0,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=t.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t.optimizer,
        beta=t.beta,
        use_vllm=False,
        chat_template_kwargs={"enable_thinking": t.enable_thinking},
        generation_kwargs={},
        model_init_kwargs={
            "dtype": "bfloat16" if bf16 else "float32",
            "device_map": None,
            "attn_implementation": "sdpa",
        },
        disable_dropout=True,
        save_steps=1 if smoke else t.save_steps,
        logging_steps=t.logging_steps,
        save_total_limit=None,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        log_completions=False,
        dataloader_num_workers=0,
    )


def resolve_starting_model(settings: Settings, source: str) -> str:
    path = settings.path(source)
    if path.is_dir():
        if not (path / "config.json").is_file() or not (
            list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin"))
        ):
            raise ValueError(
                "Starting model folder must contain saved model weights and config.json"
            )
        return str(path.resolve())
    if Path(source).is_absolute() or source.startswith(("runs/", "./", "../")):
        raise ValueError(f"Starting model folder does not exist: {source}")
    return source


def save_judge_failure_checkpoint(trainer, tokenizer, run_dir, error):
    """Judge calls occur before backward at a complete optimizer-update boundary."""
    if trainer is None or trainer.state.global_step <= 0:
        return None
    if (trainer.args.steps_per_generation != trainer.args.gradient_accumulation_steps
            or trainer.num_iterations != 1):
        return None
    step = trainer.state.global_step
    checkpoint = Path(run_dir) / f"checkpoint-{step}"
    if not all((checkpoint / name).is_file() for name in
               ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]):
        trainer._save_checkpoint(trainer.model, trial=None)
        tokenizer.save_pretrained(checkpoint)
    write_json(checkpoint / "judge-failure.json", {
        "completed_step": step, "failed_update": step + 1, "error": str(error),
        "note": "Saved the last completed weights; the failed rollout was not trained on.",
    })
    return checkpoint


def train(settings: Settings, smoke=False, resume: str | None = None, output=None,
          model_source: str | None = None) -> None:
    if model_source is not None:
        if resume:
            raise ValueError(
                "--model starts a fresh GRPO run; --resume restores its existing model"
            )
        settings.training.model = resolve_starting_model(settings, model_source)
    require_api_key(settings)
    require_gpu()
    import torch
    from transformers import TrainerCallback, set_seed

    from .callbacks import CheckpointScheduleCallback
    from .rollouts import ReplacingGRPOTrainer

    run_dir = settings.path(output or settings.training.output_dir)
    if smoke and output is None:
        run_dir = run_dir.with_name(run_dir.name + "-smoke")
    if (run_dir / "run.json").exists() and resume is None:
        raise ValueError("Run already exists. Use --resume latest or --output runs/a-new-name.")
    if resume and not (run_dir / "initial_model/config.json").is_file():
        raise ValueError("Cannot resume: this run's initial reference snapshot is missing")
    set_seed(settings.training.seed)
    if resume and (run_dir / "run.json").is_file():
        original = json.loads((run_dir / "run.json").read_text())
        settings.training.model = original["settings"]["training"]["model"]
    tokenizer = load_tokenizer(settings, str(run_dir / "initial_model") if resume else None)
    # Rebuild from the same source with the active tokenizer and length limit so config
    # changes cannot silently reuse stale prompt filtering. Downloads are cached by HF.
    data_report = prepare(settings, tokenizer)
    dataset = load_prepared(settings, "train")
    run_dir.mkdir(parents=True, exist_ok=True)
    initial = run_dir / "initial_model"
    if resume:
        model = load_text_model(settings, str(initial))
    else:
        model = load_text_model(settings)
        # TRL reloads the reference by model identifier. Save a text-only initial
        # checkpoint so it cannot accidentally load the multimodal vision tower.
        model.save_pretrained(initial)
        tokenizer.save_pretrained(initial)
    model.config._name_or_path = str(initial.resolve())
    parameters = list(model.parameters())
    if any(not p.requires_grad for p in parameters):
        raise ValueError("Expected full-parameter training; found frozen language-model weights")
    config = make_grpo_config(settings, run_dir, smoke, tokenizer=tokenizer)
    manifest = {
        "settings": settings.public_dict(),
        "hardware": hardware_info(),
        "language_parameters": sum(p.numel() for p in parameters),
        "all_language_parameters_trainable": True,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "initial_model": str(initial.resolve()),
        "data": data_report,
        "smoke_test": smoke,
        "resume": resume,
        "generation_kwargs": config.generation_kwargs,
    }
    write_json(run_dir / ("resume.json" if resume else "run.json"), manifest)
    print(json.dumps(manifest, indent=2))
    if not config.bf16:
        print("BF16 unavailable: using FP16 autocast with FP32 parameters; expect higher memory.")

    class MemoryLogger(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            record = {
                "step": state.global_step,
                "allocated_gib": round(torch.cuda.memory_allocated() / 1024**3, 3),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
            }
            with (run_dir / "memory.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(f"GPU memory: {record}")

    judge = OptimismJudge(settings, run_dir / "judge.jsonl", tokenizer=tokenizer)
    manifest["judge_rubric_sha256"] = hashlib.sha256(judge.rubric.encode()).hexdigest()
    (run_dir / "judge-rubric.txt").write_text(judge.rubric)
    write_json(run_dir / ("resume.json" if resume else "run.json"), manifest)
    trainer = None
    try:
        trainer = ReplacingGRPOTrainer(
            model=model,
            args=config,
            reward_funcs=judge.reward_sync,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[CheckpointScheduleCallback(), MemoryLogger()],
            reference_cpu_offload_gib=settings.training.reference_cpu_offload_gib,
            trim_training_padding=settings.training.trim_training_padding,
            judge=judge,
            max_replacements_per_step=settings.judge.max_replacements_per_step,
        )
        if trainer.ref_model is not None:
            if trainer.ref_model.__class__.__name__ != model.__class__.__name__:
                raise ValueError("Reference architecture differs from the text-only policy")
        manifest["reference_cpu_offload"] = trainer.offload_statistics()
        write_json(run_dir / ("resume.json" if resume else "run.json"), manifest)
        torch.cuda.reset_peak_memory_stats()
        trainer.train(resume_from_checkpoint=True if resume == "latest" else resume)
        trainer.save_model(str(run_dir / "final"))
        tokenizer.save_pretrained(run_dir / "final")
        trainer.save_state()
        write_json(run_dir / "judge-usage.json", judge.statistics())
    except JudgeRequestError as exc:
        try:
            checkpoint = save_judge_failure_checkpoint(trainer, tokenizer, run_dir, exc)
            if checkpoint is not None:
                print(f"Saved completed training progress before judge failure: {checkpoint}",
                      flush=True)
        except Exception as save_error:
            print(f"Could not save the judge-failure checkpoint: {save_error}", flush=True)
        raise
    finally:
        if trainer is not None:
            write_json(run_dir / "reference-offload.json", trainer.offload_statistics())
            write_json(run_dir / "judge-replacements.json", trainer.replacement_statistics())
        write_json(run_dir / "judge-usage.json", judge.statistics())
        judge.close()
