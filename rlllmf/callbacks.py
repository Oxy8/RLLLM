import math

from transformers import TrainerCallback


class CheckpointScheduleCallback(TrainerCallback):
    """Apply the requested save interval after Trainer restores checkpoint state."""

    def on_train_begin(self, args, state, control, **kwargs):
        interval = args.save_steps
        state.save_steps = math.ceil(state.max_steps * interval) if interval < 1 else interval
