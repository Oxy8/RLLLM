import json
from types import SimpleNamespace

from rlllmf.judge import JudgeRequestError
from rlllmf.training import save_judge_failure_checkpoint


def test_judge_failure_preserves_completed_step_without_overwriting_existing_checkpoint(tmp_path):
    calls = []
    state = SimpleNamespace(global_step=1388)
    args = SimpleNamespace(steps_per_generation=16, gradient_accumulation_steps=16)
    trainer = SimpleNamespace(state=state, args=args, num_iterations=1, model=object())

    def save(model, trial):
        calls.append(state.global_step)
        path = tmp_path / f"checkpoint-{state.global_step}"
        path.mkdir()
        for name in ["model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"]:
            (path / name).write_bytes(b"completed step")

    trainer._save_checkpoint = save
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    error = JudgeRequestError("Judge refused evaluation")
    checkpoint = save_judge_failure_checkpoint(trainer, tokenizer, tmp_path, error)
    assert calls == [1388]
    record = json.loads((checkpoint / "judge-failure.json").read_text())
    assert record["completed_step"] == 1388 and record["failed_update"] == 1389
    save_judge_failure_checkpoint(trainer, tokenizer, tmp_path, error)
    assert calls == [1388]
    state.global_step = 0
    assert save_judge_failure_checkpoint(trainer, tokenizer, tmp_path, error) is None
