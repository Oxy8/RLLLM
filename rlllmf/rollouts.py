"""Replace unscorable instruction groups before GRPO constructs a training batch."""

from __future__ import annotations

import copy
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import torch

from .judge import JudgeRequestError, JudgeResponseError, plain_text
from .offload import OffloadGRPOTrainer


class ReplacingGRPOTrainer(OffloadGRPOTrainer):
    def __init__(self, *args, judge, max_replacements_per_step=16, **kwargs):
        self.judge = judge
        self.max_replacements_per_step = max_replacements_per_step
        self._accepted_rollout = None
        self._accepted_rewards = None
        self.skipped_questions = 0
        self.replacement_questions = 0
        super().__init__(*args, **kwargs)
        if (self.accelerator.num_processes != 1 or self.tools or self.environment_factories
                or self._is_vlm or self.rollout_func is not None or self.use_vllm
                or self.use_transformers_continuous_batching
                or len(self.reward_funcs) != 1 or self.reward_funcs[0] != judge.reward_sync
                or self.num_iterations != 1
                or self.args.steps_per_generation != self.args.gradient_accumulation_steps):
            raise ValueError("Judge replacement requires single-device text GRPO with one judge "
                             "and one generation batch per optimizer update")

    def replacement_statistics(self):
        return {"skipped_questions": self.skipped_questions,
                "replacement_questions": self.replacement_questions,
                "max_replacements_per_step": self.max_replacements_per_step}

    def _generate(self, prompts):
        if self._accepted_rollout is not None:
            output, self._accepted_rollout = self._accepted_rollout, None
            return output
        return super()._generate(prompts)

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        if self._accepted_rewards is not None:
            rewards, self._accepted_rewards = self._accepted_rewards, None
            if len(rewards) != len(prompts):
                raise ValueError("Accepted judge rewards do not align with the rollout")
            return torch.tensor(rewards, dtype=torch.float32,
                                device=self.accelerator.device).unsqueeze(1)
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    def _judge_rollout(self, prompts, output):
        # These fields are guaranteed by the ordinary text-only Transformers backend.
        if (any(output[i] is not None for i in [2, 4, 5, 7]) or output[6] or output[8]):
            raise ValueError("Unexpected non-text fields in the replacement rollout")
        result = self.judge.reward_sync(prompts, output[3], completion_ids=output[1],
                                       return_groups=True)
        self._rollout_judgments = getattr(self.judge, "last_group_judgments", None)
        return result

    def _record_skip(self, failed, replacement, error):
        record = {"timestamp_utc": datetime.now(UTC).isoformat(),
                  "training_step": self.state.global_step + 1,
                  "instruction": plain_text(failed["prompt"]),
                  "prompt_id": failed.get("prompt_id"), "error": str(error),
                  "replacement_instruction": plain_text(replacement["prompt"]),
                  "replacement_prompt_id": replacement.get("prompt_id")}
        with (Path(self.args.output_dir) / "skipped-questions.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Judge skipped an unscorable question at update {record['training_step']}; "
              "generating a replacement question's answers.", flush=True)

    def _generate_and_score_completions(self, inputs):
        if not self.model.training:
            return super()._generate_and_score_completions(inputs)
        # Keep the sampler's rows unchanged. Only this optimizer update uses replacements.
        inputs = copy.deepcopy(inputs)
        size = self.num_generations
        if not inputs or len(inputs) % size:
            raise ValueError("Incomplete question group in the generation batch")
        prompts = [row["prompt"] for row in inputs]
        self.judge.training_step = self.state.global_step + 1
        output = super()._generate(prompts)
        results = self._judge_rollout(prompts, output)
        judgments = self._rollout_judgments
        seen = {plain_text(prompt) for prompt in prompts}
        indices = list(range(len(self.train_dataset)))
        # No global RNG changes: replacement selection is reproducible at each saved step.
        random.Random(f"{self.args.seed}:{self.state.global_step}").shuffle(indices)
        candidates = iter(indices)
        replacements = 0

        while any(isinstance(result, JudgeResponseError) for result in results):
            pending = [i for i, result in enumerate(results)
                       if isinstance(result, JudgeResponseError)]
            if replacements + len(pending) > self.max_replacements_per_step:
                raise JudgeRequestError(
                    "Judge replacement limit reached for this update; no partial batch was "
                    "trained. Check judge.errors.jsonl and skipped-questions.jsonl."
                )
            replacement_inputs = []
            for group in pending:
                replacement = None
                for index in candidates:
                    row = self.train_dataset[index]
                    text = plain_text(row["prompt"])
                    if text not in seen:
                        replacement = row
                        seen.add(text)
                        break
                if replacement is None:
                    raise JudgeRequestError("No unused training question remains for replacement")
                start = group * size
                self._record_skip(inputs[start], replacement, results[group])
                self.skipped_questions += 1
                self.replacement_questions += 1
                replacements += 1
                rows = [copy.deepcopy(replacement) for _ in range(size)]
                inputs[start:start + size] = rows
                prompts[start:start + size] = [row["prompt"] for row in rows]
                replacement_inputs.extend(rows)

            replacement_prompts = [row["prompt"] for row in replacement_inputs]
            extra = super()._generate(replacement_prompts)
            extra_results = self._judge_rollout(replacement_prompts, extra)
            extra_judgments = self._rollout_judgments
            # Preserve every valid group's token IDs and reward. Only rejected groups
            # are regenerated, using the same current policy before any weight update.
            for offset, group in enumerate(pending):
                start, source = group * size, offset * size
                for field in [0, 1, 3]:
                    output[field][start:start + size] = extra[field][source:source + size]
                results[group] = extra_results[offset]
                if judgments is not None:
                    judgments[group] = extra_judgments[offset]

        rewards = [reward for group in results for reward in group]
        if len(rewards) != len(inputs):
            raise ValueError("Replacement changed the effective training batch size")
        self._metrics["train"]["judge/replacement_questions"].append(replacements)
        if judgments is not None:
            accepted = [row for group in judgments for row in group]
            for key in [self.judge.score_name, "task_quality"]:
                if all(key in row for row in accepted):
                    self._metrics["train"][f"judge/{key}"].append(
                        sum(row[key] for row in accepted) / len(accepted)
                    )
            self._metrics["train"]["judge/combined_reward"].append(sum(rewards) / len(rewards))
        self._accepted_rollout, self._accepted_rewards = output, rewards
        try:
            # TRL still builds the masks, reference KL, group-normalized advantages
            # and ordinary GRPO loss. Reuse our generated tokens and valid scores once.
            return super()._generate_and_score_completions(inputs)
        finally:
            self._accepted_rollout = self._accepted_rewards = None
