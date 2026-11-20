import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rlllmf.pipeline import pipeline
from tests.test_core import temporary_settings


def saved_model(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"test")


def complete(path, stage, source=None):
    saved_model(path / "final")
    manifest = {"stage": stage, "settings": {"training": {"model": source}}}
    (path / "run.json").write_text(json.dumps(manifest))
    (path / "trainer_state.json").write_text(json.dumps({"global_step": 64, "max_steps": 64}))


def test_pipeline_sequences_processes_and_passes_sft_model(tmp_path):
    s = temporary_settings(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        if "sft" in command:
            complete(output, "sft")
        else:
            source = command[command.index("--model") + 1]
            assert Path(source).is_dir()
            complete(output, "grpo", source)
        return SimpleNamespace(returncode=0)

    result = pipeline(s, tmp_path / ".config", runner=runner)
    assert result["status"] == "complete" and len(calls) == 2
    assert calls[0][1:3] == ["-m", "rlllmf"]
    assert calls[0][calls[0].index("--config") + 1] == str(tmp_path / ".config")
    assert "--model" in calls[1] and "--resume" not in calls[1]
    assert pipeline(s, tmp_path / ".config", resume=True, runner=runner)["status"] == "complete"
    assert len(calls) == 2


def test_sft_failure_never_launches_rl(tmp_path):
    s = temporary_settings(tmp_path)
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)

    with pytest.raises(RuntimeError, match="SFT exited"):
        pipeline(s, tmp_path / ".config", runner=runner)
    assert len(calls) == 1


def test_successful_exit_without_final_model_does_not_launch_rl(tmp_path):
    s = temporary_settings(tmp_path)
    with pytest.raises(RuntimeError, match="without a completed final model"):
        pipeline(s, tmp_path / ".config", runner=lambda *a, **kw: SimpleNamespace(returncode=0))


def test_resume_skips_sft_and_restores_rl_checkpoint(tmp_path):
    s = temporary_settings(tmp_path)
    supervised = s.path(s.sft.output_dir)
    rl = tmp_path / "runs/lfm3-childishness-after-sft-01"
    complete(supervised, "sft")
    rl.mkdir()
    (rl / "run.json").write_text(json.dumps({
        "settings": {"training": {"model": str(supervised / "final")}},
    }))
    checkpoint = rl / "checkpoint-128"
    saved_model(checkpoint)
    for name in ["optimizer.pt", "scheduler.pt"]:
        (checkpoint / name).write_bytes(b"test")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 128}))
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        complete(rl, "grpo", str(supervised / "final"))
        return SimpleNamespace(returncode=0)

    pipeline(s, tmp_path / ".config", resume=True, runner=runner)
    assert len(calls) == 1
    assert calls[0][calls[0].index("--resume") + 1] == str(checkpoint)
    assert "--model" not in calls[0]


def test_dry_run_and_existing_outputs_are_safe(tmp_path):
    s = temporary_settings(tmp_path)
    report = pipeline(s, tmp_path / ".config", dry_run=True)
    assert report["dry_run"] and not s.path(s.sft.output_dir).exists()
    complete(s.path(s.sft.output_dir), "sft")
    with pytest.raises(ValueError, match="already exists"):
        pipeline(s, tmp_path / ".config", dry_run=True)
    with pytest.raises(ValueError, match="separate"):
        pipeline(s, tmp_path / ".config", rl_output=s.sft.output_dir, dry_run=True)


def test_resume_sft_checkpoint_then_fresh_rl(tmp_path):
    s = temporary_settings(tmp_path)
    supervised = s.path(s.sft.output_dir)
    checkpoint = supervised / "checkpoint-32"
    saved_model(checkpoint)
    (supervised / "run.json").write_text(json.dumps({"stage": "sft"}))
    for name in ["optimizer.pt", "scheduler.pt"]:
        (checkpoint / name).write_bytes(b"test")
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 32}))
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--output") + 1])
        if "sft" in command:
            assert command[command.index("--resume") + 1] == str(checkpoint)
            complete(output, "sft")
        else:
            assert "--resume" not in command
            complete(output, "grpo", str(supervised / "final"))
        return SimpleNamespace(returncode=0)

    pipeline(s, tmp_path / ".config", resume=True, runner=runner)
    assert len(calls) == 2


def test_resume_rejects_an_rl_run_with_another_reference(tmp_path):
    s = temporary_settings(tmp_path)
    complete(s.path(s.sft.output_dir), "sft")
    complete(tmp_path / "runs/lfm3-childishness-after-sft-01", "grpo", "original/model")
    with pytest.raises(ValueError, match="did not start from this SFT"):
        pipeline(s, tmp_path / ".config", resume=True, dry_run=True)
