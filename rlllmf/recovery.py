"""Guarded replay of interrupted GRPO, followed by its remaining fresh updates."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import random
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback
from trl.trainer.utils import RepeatSampler

from .answers import final_answer, limit_answer_tokens
from .callbacks import CheckpointScheduleCallback
from .config import DataSettings, JudgeSettings, TrainingSettings, require_api_key
from .data import load_prepared, write_json
from .judge import OptimismJudge
from .offload import OffloadGRPOTrainer
from .training import hardware_info, load_text_model, load_tokenizer, require_gpu


def host_memory():
    info = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        info[key] = int(value.split()[0]) * 1024
    return {"available_gib": info['MemAvailable'] / 1024**3,
            "swap_free_gib": info['SwapFree'] / 1024**3}


def require_recovery_memory():
    memory = host_memory()
    if memory['available_gib'] + memory['swap_free_gib'] < 13:
        raise ValueError(
            f"Only {memory['available_gib']:.1f} GiB RAM and "
            f"{memory['swap_free_gib']:.1f} GiB swap available. "
            "Recovery requires 13 GiB combined headroom after the previous system-RAM kill. "
            "Close memory-heavy applications or enable swap, then rerun the same command."
        )


class ReplayHistory:
    def __init__(self, source, checkpoint_step, groups_per_step):
        self.source = Path(source)
        self.checkpoint_step = checkpoint_step
        self.groups_per_step = groups_per_step
        self.last_step = 0
        with (self.source / 'memory.jsonl').open() as stream:
            for line in stream:
                if line.strip():
                    self.last_step = max(self.last_step, json.loads(line)['step'])
        self.steps = {}
        digest = hashlib.sha256()
        count = 0
        with (self.source / 'judge.jsonl').open('rb') as stream:
            for index, raw in enumerate(stream):
                digest.update(raw)
                step = index // groups_per_step + 1
                count += 1
                if checkpoint_step < step <= self.last_step:
                    self.steps.setdefault(step, []).append(json.loads(raw))
        if count < self.last_step * groups_per_step:
            raise ValueError('Original judge log is incomplete for its recorded optimizer steps')
        for step, rows in self.steps.items():
            if len(rows) != groups_per_step or len({r['instruction'] for r in rows}) != len(rows):
                raise ValueError(f'Ambiguous or incomplete recorded batch at step {step}')
        self.judge_log_sha256 = digest.hexdigest()


class RecoveryJudge(OptimismJudge):
    def __init__(self, *args, history, on_divergence='stop', **kwargs):
        super().__init__(*args, **kwargs)
        if on_divergence not in {'stop', 'rejudge'}:
            raise ValueError('on_divergence must be stop or rejudge')
        self.history = history
        self.on_divergence = on_divergence
        self.current_step = lambda: history.checkpoint_step + 1
        self.last_step = None
        self.last_scores = []
        self.first_divergence = None
        self.last_batch_matched = None

    async def score_groups(self, groups, diagnostics=None):
        step = self.current_step()
        self.cache_only = step <= self.history.last_step
        self.last_batch_matched = None
        if self.cache_only:
            expected = {row['instruction']: row for row in self.history.steps.get(step, [])}
            if len(groups) != self.history.groups_per_step or set(expected) != {
                instruction for instruction, _ in groups
            }:
                raise RuntimeError(
                    f'Recovery diverged at step {step}: questions differ. No API calls were made.'
                )
            if diagnostics is None:
                raise RuntimeError('Recovery requires generated-token diagnostics')
            mismatches = []
            for (instruction, responses), infos in zip(groups, diagnostics, strict=True):
                row = expected[instruction]
                answers = [limit_answer_tokens(self.tokenizer, final_answer(text),
                                              self.max_answer_tokens) for text in responses]
                lengths = [info['generated_tokens'] for info in infos]
                original_lengths = [info['generated_tokens']
                                    for info in row['generation_diagnostics']]
                if answers != row['responses'] or lengths != original_lengths:
                    mismatches.append({'instruction': instruction,
                                       'original_lengths': original_lengths,
                                       'generated_lengths': lengths,
                                       'answers_match': [a == b for a, b in
                                                         zip(answers, row['responses'])]})
            self.last_batch_matched = not mismatches
            if mismatches:
                self.first_divergence = self.first_divergence or step
                if self.log_path:
                    with (self.log_path.parent / 'recovery-divergences.jsonl').open('a') as stream:
                        stream.write(json.dumps({'step': step, 'action': self.on_divergence,
                                                 'groups': mismatches}) + '\n')
                if self.on_divergence == 'stop':
                    raise RuntimeError(
                        f'Recovery diverged at step {step}: answers or token lengths differ. '
                        'Stopped before the optimizer update; no new API calls were made. '
                        'Use --on-divergence rejudge only if new Haiku scoring is acceptable.'
                    )
                self.cache_only = False
                print(f'Step {step}: {len(mismatches)} groups differ from the original run; '
                      'using cached scores where possible and Haiku for cache misses.', flush=True)
        results = await super().score_groups(groups, diagnostics)
        self.last_step = step
        self.last_scores = [float(row[self.score_name]) for group in results for row in group]
        return results


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.random.get_rng_state(),
            'cuda': torch.cuda.random.get_rng_state() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.random.set_rng_state(state['cpu'])
    if state['cuda'] is not None:
        torch.cuda.random.set_rng_state(state['cuda'])


class RecoverySampler(RepeatSampler):
    """Reconstruct the original private sampler RNG even when resuming in epoch > 0."""

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            self.generator.manual_seed(self.seed if self.seed is not None else 0)
            for _ in range(getattr(self, 'epoch', 0)):
                torch.randperm(self.num_samples, generator=self.generator)
        yield from super().__iter__()


class RecoveryGRPOTrainer(OffloadGRPOTrainer):
    def __init__(self, *args, preloaded_checkpoint, journal_dir, **kwargs):
        self.preloaded_checkpoint = Path(preloaded_checkpoint).resolve()
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.last_rollout_reused = False
        super().__init__(*args, **kwargs)
        if (self.accelerator.num_processes != 1 or self.use_vllm or self.tools
                or self.rollout_func is not None or self.num_iterations != 1
                or self.args.steps_per_generation != self.args.gradient_accumulation_steps):
            raise ValueError(
                'Recovery currently supports this single-GPU, one-rollout-per-update run'
            )

    def _get_train_sampler(self, dataset=None):
        sampler = super()._get_train_sampler(dataset)
        return RecoverySampler(
            sampler.data_source, sampler.mini_repeat_count, sampler.batch_size,
            sampler.repeat_count, sampler.shuffle, sampler.seed,
        )

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # The policy was loaded with its native from_pretrained path before Trainer
        # construction. This applies LFM3's saved gate_proj -> g_proj conversion.
        # Trainer independently restores optimizer, scheduler, trainer state and RNG.
        if Path(resume_from_checkpoint).resolve() != self.preloaded_checkpoint:
            raise ValueError('Resume checkpoint differs from the natively preloaded policy')

    def _load_optimizer_and_scheduler(self, checkpoint):
        optimizer = getattr(self.optimizer, 'optimizer', self.optimizer)
        if (checkpoint is None or self.accelerator.device.type != 'cuda'
                or not type(optimizer).__module__.startswith('bitsandbytes.')):
            return super()._load_optimizer_and_scheduler(checkpoint)
        # Restore the existing 8-bit state directly to its execution device instead
        # of retaining a second multi-GiB CPU copy during single-GPU restoration.
        state = torch.load(Path(checkpoint) / 'optimizer.pt',
                           map_location=self.accelerator.device, weights_only=True)
        self.optimizer.load_state_dict(state)
        del state
        self.lr_scheduler.load_state_dict(torch.load(
            Path(checkpoint) / 'scheduler.pt', map_location='cpu', weights_only=True,
        ))

    def _generate(self, prompts):
        step = self.state.global_step + 1
        path = self.journal_dir / f'step-{step}.pt'
        digest = hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest()
        self.last_rollout_reused = path.is_file()
        if path.is_file():
            saved = torch.load(path, map_location='cpu', weights_only=False)
            if saved['prompt_sha256'] != digest:
                raise RuntimeError(f'Saved recovery rollout has different prompts at step {step}')

            def replay_rollout(prompts, trainer):
                return {'prompt_ids': [row.tolist() for row in saved['prompt_ids']],
                        'completion_ids': [row.tolist() for row in saved['completion_ids']],
                        'logprobs': None}

            self.rollout_func = replay_rollout
            try:
                result = super()._generate(prompts)
            finally:
                self.rollout_func = None
            restore_rng(saved['post_generation_rng'])
            return result
        result = super()._generate(prompts)
        payload = {'prompt_sha256': digest, 'step': step,
                   'prompt_ids': [torch.tensor(row, dtype=torch.int32) for row in result[0]],
                   'completion_ids': [torch.tensor(row, dtype=torch.int32) for row in result[1]],
                   'post_generation_rng': capture_rng()}
        temporary = path.with_suffix('.tmp')
        torch.save(payload, temporary)
        temporary.replace(path)
        return result


class RecoveryProgress(TrainerCallback):
    def __init__(self, judge, replay_through, output):
        self.judge, self.replay_through, self.output = judge, replay_through, Path(output)
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        record = {'step': state.global_step,
                  'phase': 'recovery' if state.global_step <= self.replay_through else 'fresh',
                  'matched_original_batch': self.judge.last_batch_matched,
                  'first_divergence': self.judge.first_divergence,
                  **self.judge.statistics(), **host_memory(),
                  'allocated_gib': round(torch.cuda.memory_allocated() / 1024**3, 3),
                  'peak_reserved_gib': round(torch.cuda.max_memory_reserved() / 1024**3, 3),
                  'elapsed_seconds': time.perf_counter() - self.started}
        with (self.output / 'recovery-steps.jsonl').open('a') as stream:
            stream.write(json.dumps(record) + '\n')
        if state.global_step == self.replay_through:
            control.should_save = True
            print(f'Reached step {state.global_step}; saving a recovery checkpoint. '
                  'Subsequent updates may make new Haiku requests.', flush=True)


def restore_settings(settings, manifest):
    restored = copy.deepcopy(settings)
    for name, cls in [('training', TrainingSettings), ('data', DataSettings),
                      ('judge', JudgeSettings)]:
        allowed = {field.name for field in fields(cls)}
        section = manifest['settings'][name]
        setattr(restored, name, cls(**{k: v for k, v in section.items() if k in allowed}))
    return restored


def resolve_checkpoint(folder, name='latest'):
    folder = Path(folder)
    if name == 'latest':
        candidates = [p for p in folder.glob('checkpoint-*')
                      if (p / 'trainer_state.json').is_file()]
        if not candidates:
            raise ValueError(f'No saved checkpoint in {folder}')
        checkpoint = max(candidates, key=lambda p: int(p.name.split('-')[-1]))
    else:
        candidate = Path(name)
        checkpoint = candidate if candidate.is_absolute() else folder / candidate
    for filename in ['config.json', 'model.safetensors', 'optimizer.pt', 'scheduler.pt',
                     'rng_state.pth', 'trainer_state.json', 'training_args.bin']:
        if not (checkpoint / filename).is_file():
            raise ValueError(f'Incomplete recovery checkpoint: missing {filename}')
    return checkpoint.resolve()


def recover(settings, source, output=None, max_steps=1536, resume=None, dry_run=False,
            on_divergence='stop'):
    source = settings.path(source).resolve()
    target = (settings.path(output).resolve() if output
              else source.with_name(source.name + '-recovery'))
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Recovery output must be separate from the original run')
    manifest_path = source / 'resume.json'
    if not manifest_path.is_file():
        manifest_path = source / 'run.json'
    manifest = json.loads(manifest_path.read_text())
    s = restore_settings(settings, manifest)
    if (target / 'run.json').exists() and resume is None:
        raise ValueError('Recovery already exists; use recover --resume latest or a new --output')
    if resume and not (target / 'run.json').is_file():
        raise ValueError('No existing recovery run to resume')
    if resume == 'latest' and not list(target.glob('checkpoint-*')):
        # If the first recovery never reached a save, retain and reuse its raw
        # rollout journal while restoring the original complete checkpoint again.
        checkpoint = resolve_checkpoint(source)
    else:
        checkpoint = resolve_checkpoint(target if resume else source, resume or 'latest')
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    original_checkpoint = resolve_checkpoint(source)
    original_state = json.loads((original_checkpoint / 'trainer_state.json').read_text())
    original_step = original_state['global_step']
    if type(max_steps) is not int or max_steps <= state['global_step']:
        raise ValueError('max_steps must exceed the selected checkpoint step')
    batch_size = s.training.micro_batch_size * s.training.gradient_accumulation_steps
    history = ReplayHistory(source, original_step, batch_size // s.training.num_generations)
    rubric = (source / 'judge-rubric.txt').read_text()
    if hashlib.sha256(rubric.encode()).hexdigest() != manifest['judge_rubric_sha256']:
        raise ValueError('Original judge rubric does not match its saved hash')
    if s.path(s.judge.rubric_path).read_text() != rubric:
        raise ValueError('Configured judge rubric has changed since the interrupted run')
    dataset = load_prepared(s, 'train')
    if len(dataset) != manifest['data']['train']['selected']:
        raise ValueError('Prepared training dataset size differs from the interrupted run')
    plan = {'source_run': str(source), 'checkpoint': str(checkpoint),
            'checkpoint_step': state['global_step'], 'replay_through': history.last_step,
            'target_steps': max_steps, 'output': str(target),
            'replayed_updates': max(0, min(max_steps, history.last_step) - state['global_step']),
            'fresh_updates': max(0, max_steps - max(history.last_step, state['global_step'])),
            'fresh_judge_groups': max(0, max_steps - max(history.last_step, state['global_step']))
                                  * history.groups_per_step,
            'judge_log_sha256': history.judge_log_sha256, 'memory': host_memory(),
            'on_divergence': on_divergence,
            'cache_only_during_replay': on_divergence == 'stop',
            'checkpoint_at_replay_boundary': history.last_step,
            'final_checkpoint': str(target / f'checkpoint-{max_steps}'),
            'final_model': str(target / 'final')}
    print(json.dumps(plan, indent=2), flush=True)
    if dry_run:
        return plan
    require_api_key(s)
    require_recovery_memory()
    require_gpu()
    from accelerate.state import PartialState

    PartialState()
    target.mkdir(parents=True, exist_ok=True)
    with (target / '.recovery.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('This recovery output is already being trained') from None
        initial = target / 'initial_model'
        if not initial.exists():
            initial.symlink_to(source / 'initial_model', target_is_directory=True)
        tokenizer = load_tokenizer(s, str(initial))
        model = load_text_model(s, str(checkpoint))
        model.config._name_or_path = str(initial.resolve())
        args = torch.load(checkpoint / 'training_args.bin', map_location='cpu', weights_only=False)
        args.output_dir, args.max_steps, args.report_to = str(target), max_steps, []
        args.save_total_limit, args.push_to_hub = 3, False
        s.training.max_steps, s.training.output_dir = max_steps, str(target)
        metadata = {'stage': 'grpo-recovery', 'settings': s.public_dict(),
                    'initial_model': str(initial.resolve()), 'hardware': hardware_info(),
                    'data': manifest['data'],
                    'judge_rubric_sha256': manifest['judge_rubric_sha256'],
                    'recovery': plan}
        write_json(target / ('resume.json' if resume else 'run.json'), metadata)
        (target / 'judge-rubric.txt').write_text(rubric)
        judge = RecoveryJudge(s, target / 'judge.jsonl', tokenizer=tokenizer, history=history,
                              on_divergence=on_divergence)
        trainer = None
        try:
            trainer = RecoveryGRPOTrainer(
                model=model, args=args, processing_class=tokenizer, train_dataset=dataset,
                reward_funcs=judge.reward_sync, preloaded_checkpoint=checkpoint,
                journal_dir=target / 'recovery-rollouts',
                reference_cpu_offload_gib=s.training.reference_cpu_offload_gib,
                trim_training_padding=s.training.trim_training_padding,
                callbacks=[CheckpointScheduleCallback(),
                           RecoveryProgress(judge, history.last_step, target)],
            )
            judge.current_step = lambda: trainer.state.global_step + 1
            trainer.train(resume_from_checkpoint=str(checkpoint))
            if trainer.state.global_step != max_steps:
                raise RuntimeError('Recovery stopped before the requested final step')
            # Explicitly guarantee the complete final checkpoint, even if save_steps changes.
            final_checkpoint = target / f'checkpoint-{max_steps}'
            if not (final_checkpoint / 'optimizer.pt').is_file():
                trainer._save_checkpoint(trainer.model, trial=None)
            trainer.save_model(str(target / 'final'))
            tokenizer.save_pretrained(target / 'final')
            trainer.save_state()
            write_json(target / 'recovery-complete.json',
                       {'global_step': trainer.state.global_step, 'plan': plan,
                        'judge_usage': judge.statistics()})
        finally:
            write_json(target / 'judge-usage.json', judge.statistics())
            if trainer is not None:
                write_json(target / 'reference-offload.json', trainer.offload_statistics())
            judge.close()
    return plan
