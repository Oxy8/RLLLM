"""Run SFT and GRPO in separate processes so SFT releases GPU memory first."""

from __future__ import annotations

import fcntl
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .collection import now
from .config import Settings, require_api_key
from .data import write_json


def model_saved(path: Path) -> bool:
    if not (path / "config.json").is_file():
        return False
    indexes = list(path.glob("*.index.json"))
    for index in indexes:
        data = json.loads(index.read_text())
        if "weight_map" in data:
            return all((path / name).is_file() for name in set(data["weight_map"].values()))
    return bool(list(path.glob("*.safetensors")) or list(path.glob("pytorch_model*.bin")))


def stage_complete(path: Path) -> bool:
    state = path / "trainer_state.json"
    if not state.is_file() or not model_saved(path / "final"):
        return False
    data = json.loads(state.read_text())
    return data.get("max_steps", 0) > 0 and data.get("global_step", 0) >= data["max_steps"]


def resume_checkpoint(path: Path) -> Path:
    candidates = []
    for folder in path.glob("checkpoint-*"):
        state = folder / "trainer_state.json"
        if (state.is_file() and model_saved(folder)
                and (folder / "optimizer.pt").is_file()
                and (folder / "scheduler.pt").is_file()):
            data = json.loads(state.read_text())
            candidates.append((data["global_step"], folder))
    if not candidates:
        raise ValueError(
            f"No complete resumable checkpoint in {path}. Completed stages are preserved; "
            "an interrupted stage needs a new output folder if it never saved a checkpoint."
        )
    return max(candidates, key=lambda item: item[0])[1]


def pipeline(settings: Settings, config_path, sft_output=None, rl_output=None,
             resume=False, dry_run=False, *, runner=None) -> dict:
    supervised = settings.path(sft_output or settings.sft.output_dir).resolve()
    rl = settings.path(rl_output or "runs/lfm3-childishness-after-sft-01").resolve()
    if supervised == rl or supervised in rl.parents or rl in supervised.parents:
        raise ValueError("SFT and GRPO output folders must be separate and not nested")
    if not dry_run:
        require_api_key(settings)  # Catch missing RL credentials before spending time on SFT.
    base = [sys.executable, "-m", "rlllmf", "--config", str(Path(config_path).resolve())]
    plan = []
    for stage, folder in [("sft", supervised), ("grpo", rl)]:
        exists = (folder / "run.json").is_file()
        if exists and not resume:
            raise ValueError(
                f"{stage.upper()} run already exists; use pipeline --resume or new outputs"
            )
        complete = exists and stage_complete(folder)
        checkpoint = resume_checkpoint(folder) if exists and not complete else None
        command = base + (["sft"] if stage == "sft" else ["train"])
        command += ["--output", str(folder)]
        if checkpoint:
            command += ["--resume", str(checkpoint)]
        elif stage == "grpo" and not complete:
            command += ["--model", str(supervised / "final")]
        if exists:
            manifest = json.loads((folder / "run.json").read_text())
            if stage == "sft" and manifest.get("stage") != "sft":
                raise ValueError("The selected SFT output belongs to another kind of run")
            if stage == "grpo":
                source = manifest["settings"]["training"]["model"]
                if settings.path(source).resolve() != supervised / "final":
                    raise ValueError("The existing GRPO run did not start from this SFT model")
        plan.append({"stage": stage, "output": str(folder), "skip_complete": complete,
                     "command": command})
    if (rl / "run.json").exists() and not stage_complete(supervised):
        raise ValueError("The GRPO run exists but its SFT stage is not complete; check the outputs")
    for stage in plan:
        print(f"{stage['stage'].upper()}: " + (
            "already complete; skip" if stage["skip_complete"]
            else shlex.join(stage["command"])
        ), flush=True)
    if dry_run:
        return {"stages": plan, "dry_run": True}

    supervised.mkdir(parents=True, exist_ok=True)
    state = {"started_at": now(), "status": "running", "stages": plan,
             "config_path": str(Path(config_path).resolve())}
    status_file = supervised / "pipeline.json"
    with (supervised / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("This SFT-to-GRPO pipeline is already running") from None
        try:
            for stage in plan:
                if stage["skip_complete"]:
                    continue
                if stage["stage"] == "grpo" and not stage_complete(supervised):
                    raise RuntimeError(
                        "SFT did not save a completed final model; GRPO was not started"
                    )
                state["active_stage"] = stage["stage"]
                write_json(status_file, state)
                print(f"Starting {stage['stage'].upper()}…", flush=True)
                result = (runner or subprocess.run)(
                    stage["command"], cwd=settings.root, check=False
                )
                if result.returncode:
                    raise RuntimeError(
                        f"{stage['stage'].upper()} exited with code {result.returncode}. "
                        "Pipeline stopped; later stages were not started."
                    )
                if not stage_complete(Path(stage["output"])):
                    raise RuntimeError(
                        f"{stage['stage'].upper()} exited without a completed final model"
                    )
            state.update(status="complete", active_stage=None)
            print(f"Pipeline complete. Final RL model: {rl / 'final'}", flush=True)
        except BaseException:
            state["status"] = "stopped"
            raise
        finally:
            state["finished_at"] = now()
            write_json(status_file, state)
    return state
