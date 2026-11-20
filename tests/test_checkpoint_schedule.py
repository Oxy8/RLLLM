from transformers import TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_callback import DefaultFlowCallback

from rlllmf.callbacks import CheckpointScheduleCallback


def test_1024_resume_saves_at_1280_and_1536_and_keeps_all_checkpoints(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import trl

    from rlllmf.training import make_grpo_config
    from tests.test_core import temporary_settings

    settings = temporary_settings(tmp_path)
    settings.training.save_steps = 256
    monkeypatch.setattr(trl, "GRPOConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    config = make_grpo_config(settings, tmp_path)
    assert config.save_total_limit is None and config.save_steps == 256
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to="none",
                             save_steps=config.save_steps, save_total_limit=config.save_total_limit)
    state = TrainerState(global_step=1024, max_steps=1536, save_steps=512)
    CheckpointScheduleCallback().on_train_begin(args, state, TrainerControl())
    flow = DefaultFlowCallback()
    saves = []
    for step in range(1025, 1537):
        state.global_step = step
        if flow.on_step_end(args, state, TrainerControl()).should_save:
            saves.append(step)
    assert saves == [1280, 1536]


def test_resumed_run_uses_new_save_interval_without_resetting_progress(tmp_path):
    state_path = tmp_path / "trainer_state.json"
    TrainerState(global_step=150, max_steps=512, save_steps=25).save_to_json(state_path)
    state = TrainerState.load_from_json(state_path)
    args = TrainingArguments(
        output_dir=str(tmp_path),
        use_cpu=True,
        report_to="none",
        save_steps=128,
    )
    CheckpointScheduleCallback().on_train_begin(args, state, TrainerControl())
    assert state.global_step == 150
    assert state.max_steps == 512
    flow = DefaultFlowCallback()
    state.global_step = 175
    assert not flow.on_step_end(args, state, TrainerControl()).should_save
    state.global_step = 256
    assert flow.on_step_end(args, state, TrainerControl()).should_save
