from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SCORE_NAMES = ("optimism", "childishness", "conciseness")


@dataclass
class APISettings:
    anthropic_api_key: str = field(default="", repr=False)
    hf_token: str = field(default="", repr=False)


@dataclass
class JudgeSettings:
    model: str = "claude-haiku-5-5"
    rubric_path: str = "prompts/optimism.txt"
    score_name: str = "optimism"
    reward_mode: str = "trait_only"
    quality_baseline: float = 0.7
    quality_softplus_center: float = 0.65
    quality_softplus_sharpness: float = math.log(3) / 0.05
    quality_softplus_scale: float = 10.0
    max_tokens: int = 1024
    concurrency: int = 2
    timeout_seconds: float = 90
    max_retries: int = 2
    max_requests: int = 500
    max_replacements_per_step: int = 16
    cache_path: str = ".cache/judge.sqlite3"

    @property
    def uses_task_quality(self) -> bool:
        return self.reward_mode in {"quality_gated", "quality_softplus"}

    def reward_parameters(self) -> dict:
        if self.reward_mode == "quality_softplus":
            return {
                "quality_softplus_center": self.quality_softplus_center,
                "quality_softplus_sharpness": self.quality_softplus_sharpness,
                "quality_softplus_scale": self.quality_softplus_scale,
            }
        if self.reward_mode == "quality_gated":
            return {"quality_baseline": self.quality_baseline}
        return {}


@dataclass
class DataSettings:
    repo_id: str = "tatsu-lab/alpaca_farm"
    revision: str = "main"
    train_file: str = "alpaca_instructions/unlabeled.json"
    eval_file: str = "alpaca_instructions/val.json"
    output_dir: str = "data/lfm3"
    train_samples: int = 1000
    eval_samples: int = 100
    max_prompt_tokens: int = 512


@dataclass
class TrainingSettings:
    model: str = "LiquidAI/LFM3-1.7B"
    revision: str = "main"
    output_dir: str = "runs/lfm3-optimism"
    seed: int = 42
    max_steps: int = 16
    learning_rate: float = 1e-6
    lr_scheduler_type: str = "constant"
    num_generations: int = 4
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_completion_length: int = 2048
    max_answer_tokens: int = 256
    enable_thinking: bool = True
    temperature: float = 0.8
    top_p: float = 0.9
    gradient_checkpointing: bool = True
    optimizer: str = "adamw_bnb_8bit"
    beta: float = 0.01
    reference_cpu_offload_gib: float = 2.0
    save_steps: int = 256
    logging_steps: int = 1
    trim_training_padding: bool = False


@dataclass
class EvaluationSettings:
    max_completion_length: int = 2048
    max_answer_tokens: int = 256
    enable_thinking: bool = True
    temperature: float = 0.0
    top_p: float = 0.9
    cache_baseline: bool = True
    baseline_cache_path: str = ".cache/baseline.sqlite3"


@dataclass
class CollectionSettings:
    source_file: str = "alpaca_instructions/sft.json"
    output_dir: str = "data/childishness-sft"
    prompt_path: str = "prompts/generate-childishness.txt"
    pool_size: int = 500
    examples_per_run: int = 100
    max_tokens: int = 2048
    concurrency: int = 2


@dataclass
class SFTSettings:
    dataset_path: str = "data/childishness-sft/examples.jsonl"
    output_dir: str = "runs/lfm3-childishness-sft-01"
    num_train_epochs: int = 2
    learning_rate: float = 5e-6
    lr_scheduler_type: str = "constant"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 2048


@dataclass
class Settings:
    root: Path
    api: APISettings
    judge: JudgeSettings
    data: DataSettings
    training: TrainingSettings
    evaluation: EvaluationSettings
    collection: CollectionSettings = field(default_factory=CollectionSettings)
    sft: SFTSettings = field(default_factory=SFTSettings)

    def path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.root / path

    def public_dict(self) -> dict:
        from dataclasses import asdict

        result = asdict(self)
        result["root"] = str(self.root)
        result["api"] = {
            "anthropic_api_key_configured": bool(self.api.anthropic_api_key),
            "hf_token_configured": bool(self.api.hf_token),
        }
        return result


def load_settings(filename: str | Path = ".config") -> Settings:
    path = Path(filename).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Missing {path.name}. Copy .config.example to .config first.")
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    classes = {
        "api": APISettings,
        "judge": JudgeSettings,
        "data": DataSettings,
        "training": TrainingSettings,
        "evaluation": EvaluationSettings,
        "collection": CollectionSettings,
        "sft": SFTSettings,
    }
    unknown = set(raw) - classes.keys()
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    sections = {}
    for name, cls in classes.items():
        try:
            sections[name] = cls(**raw.get(name, {}))
        except TypeError:
            # Do not echo TOML values: this section could contain a key.
            raise ValueError(f"Invalid keys in [{name}] configuration") from None
    settings = Settings(root=path.parent, **sections)
    settings.api.anthropic_api_key = os.getenv(
        "ANTHROPIC_API_KEY", settings.api.anthropic_api_key
    ).strip()
    settings.api.hf_token = os.getenv("HF_TOKEN", settings.api.hf_token).strip()
    validate_settings(settings)
    return settings


def validate_settings(s: Settings) -> None:
    t = s.training
    if type(t.trim_training_padding) is not bool:
        raise ValueError("training.trim_training_padding must be a boolean")
    positive = {
        "max_steps": t.max_steps,
        "num_generations": t.num_generations,
        "micro_batch_size": t.micro_batch_size,
        "gradient_accumulation_steps": t.gradient_accumulation_steps,
        "max_completion_length": t.max_completion_length,
        "max_answer_tokens": t.max_answer_tokens,
        "save_steps": t.save_steps,
        "logging_steps": t.logging_steps,
        "train_samples": s.data.train_samples,
        "eval_samples": s.data.eval_samples,
        "max_prompt_tokens": s.data.max_prompt_tokens,
        "judge.concurrency": s.judge.concurrency,
        "judge.max_tokens": s.judge.max_tokens,
        "judge.max_requests": s.judge.max_requests,
        "judge.max_replacements_per_step": s.judge.max_replacements_per_step,
        "evaluation.max_completion_length": s.evaluation.max_completion_length,
        "evaluation.max_answer_tokens": s.evaluation.max_answer_tokens,
        "collection.pool_size": s.collection.pool_size,
        "collection.examples_per_run": s.collection.examples_per_run,
        "collection.max_tokens": s.collection.max_tokens,
        "collection.concurrency": s.collection.concurrency,
        "sft.num_train_epochs": s.sft.num_train_epochs,
        "sft.micro_batch_size": s.sft.micro_batch_size,
        "sft.gradient_accumulation_steps": s.sft.gradient_accumulation_steps,
        "sft.max_seq_length": s.sft.max_seq_length,
    }
    for name, value in positive.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if t.num_generations < 2:
        raise ValueError("GRPO needs at least two generations per instruction")
    if (t.micro_batch_size * t.gradient_accumulation_steps) % t.num_generations:
        raise ValueError("micro_batch_size * gradient_accumulation_steps must divide into groups")
    if t.beta < 0 or t.learning_rate <= 0:
        raise ValueError("beta must be nonnegative and learning_rate must be positive")
    if not math.isfinite(s.sft.learning_rate) or s.sft.learning_rate <= 0:
        raise ValueError("sft.learning_rate must be finite and positive")
    if (
        type(t.reference_cpu_offload_gib) not in {int, float}
        or not math.isfinite(t.reference_cpu_offload_gib)
        or t.reference_cpu_offload_gib < 0
    ):
        raise ValueError("reference_cpu_offload_gib must be a finite nonnegative number")
    if s.judge.timeout_seconds <= 0 or s.judge.max_retries < 0:
        raise ValueError("Invalid judge timeout or retry count")
    if s.judge.reward_mode not in {"trait_only", "quality_gated", "quality_softplus"}:
        raise ValueError(
            "judge.reward_mode must be trait_only, quality_gated or quality_softplus"
        )
    if (type(s.judge.quality_baseline) not in {int, float}
            or not math.isfinite(s.judge.quality_baseline)
            or not 0 < s.judge.quality_baseline < 1):
        raise ValueError("judge.quality_baseline must be between 0 and 1, exclusive")
    center = s.judge.quality_softplus_center
    if (type(center) not in {int, float} or not math.isfinite(center)
            or not 0 < center < 1):
        raise ValueError("judge.quality_softplus_center must be between 0 and 1, exclusive")
    for name in ["quality_softplus_sharpness", "quality_softplus_scale"]:
        value = getattr(s.judge, name)
        if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
            raise ValueError(f"judge.{name} must be finite and positive")
    if (
        type(s.judge.score_name) is not str
        or s.judge.score_name not in SCORE_NAMES
    ):
        raise ValueError(f"judge.score_name must be one of: {', '.join(SCORE_NAMES)}")
    for label, generation in [("training", t), ("evaluation", s.evaluation)]:
        if type(generation.enable_thinking) is not bool:
            raise ValueError(f"{label}.enable_thinking must be a boolean")
        if (
            type(generation.temperature) not in {int, float}
            or not math.isfinite(generation.temperature)
            or generation.temperature < 0
            or (label == "training" and generation.temperature == 0)
            or not 0 < generation.top_p <= 1
        ):
            raise ValueError(f"Invalid {label} temperature or top_p")
    if type(s.evaluation.cache_baseline) is not bool:
        raise ValueError("evaluation.cache_baseline must be a boolean")
    if not s.path(s.judge.rubric_path).is_file():
        raise ValueError("The configured judge rubric file does not exist")
    built_in = Path(__file__).resolve().parent.parent / "prompts"
    for name in SCORE_NAMES:
        if (
            s.path(s.judge.rubric_path).resolve() == built_in / f"{name}.txt"
            and s.judge.score_name != name
        ):
            raise ValueError(f"prompts/{name}.txt requires judge.score_name='{name}'")
    if s.path(s.judge.rubric_path).resolve() == built_in / "childishness-quality.txt":
        if s.judge.score_name != "childishness" or not s.judge.uses_task_quality:
            raise ValueError(
                "childishness-quality.txt requires childishness and a dual-score reward mode"
            )


def require_api_key(settings: Settings) -> None:
    if not settings.api.anthropic_api_key:
        raise ValueError("Fill in [api].anthropic_api_key in .config or set ANTHROPIC_API_KEY")
