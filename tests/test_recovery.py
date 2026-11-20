import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rlllmf.judge import OptimismJudge
from rlllmf.recovery import RecoveryJudge, ReplayHistory, recover, restore_settings
from tests.test_answers import CharacterTokenizer
from tests.test_core import judgment, temporary_settings


def replay_row(instruction='task', answer='hopeful'):
    return {'instruction': instruction, 'responses': [answer],
            'generation_diagnostics': [{'generated_tokens': 3}],
            'judgments': [judgment('c0', .75)]}


def test_cache_only_miss_never_calls_api_or_increments_request_counter(tmp_path):
    settings = temporary_settings(tmp_path)
    calls = []

    async def request(payload):
        calls.append(payload)
        return {'scores': [judgment('c0', .75)]}

    judge = OptimismJudge(settings, request=request)
    try:
        judge.cache_only = True
        with pytest.raises(RuntimeError, match='cache miss'):
            asyncio.run(judge.score_groups([('task', ['hopeful'])]))
        assert not calls and judge.requests == 0
        judge.cache_only = False
        asyncio.run(judge.score_groups([('task', ['hopeful'])]))
        judge.cache_only = True
        asyncio.run(judge.score_groups([('task', ['hopeful'])]))
        assert len(calls) == 1 and judge.cache_hits == 1
    finally:
        judge.close()


def test_replay_guard_checks_text_lengths_questions_and_allows_fresh_calls_after_boundary(tmp_path):
    settings = temporary_settings(tmp_path)
    history = SimpleNamespace(checkpoint_step=1, last_step=2, groups_per_step=1,
                              steps={2: [replay_row()]})
    calls = []

    async def request(payload):
        calls.append(payload)
        return {'scores': [judgment('c0', .75)]}

    original = OptimismJudge(settings, request=request, tokenizer=CharacterTokenizer())
    asyncio.run(original.score_groups([('task', ['hopeful'])]))
    original.close()
    calls.clear()
    judge = RecoveryJudge(settings, history=history, request=request,
                          tokenizer=CharacterTokenizer())
    try:
        for groups, diagnostics in [([('wrong', ['hopeful'])], [[{'generated_tokens': 3}]]),
                                    ([('task', ['different'])], [[{'generated_tokens': 3}]]),
                                    ([('task', ['hopeful'])], [[{'generated_tokens': 4}]])]:
            with pytest.raises(RuntimeError, match='diverged'):
                asyncio.run(judge.score_groups(groups, diagnostics))
        assert not calls and judge.requests == 0
        result = asyncio.run(judge.score_groups([('task', ['hopeful'])],
                                               [[{'generated_tokens': 3}]]))
        assert result[0][0]['optimism'] == .75 and judge.cache_hits == 1
        judge.current_step = lambda: 3
        asyncio.run(judge.score_groups([('fresh task', ['new response'])],
                                      [[{'generated_tokens': 3}]]))
        assert len(calls) == 1 and judge.requests == 1
    finally:
        judge.close()


def test_manifest_restore_keeps_real_credentials_private(tmp_path):
    settings = temporary_settings(tmp_path)
    manifest = {'settings': settings.public_dict()}
    manifest['settings']['training']['max_steps'] = 1536
    restored = restore_settings(settings, manifest)
    assert restored.api.anthropic_api_key == 'test-key'
    assert restored.training.max_steps == 1536
    assert settings.training.max_steps != 1536
    assert 'test-key' not in json.dumps(restored.public_dict())


def test_rejudge_mode_reuses_matching_groups_and_scores_only_changed_groups(tmp_path):
    settings = temporary_settings(tmp_path)
    history = SimpleNamespace(checkpoint_step=1, last_step=2, groups_per_step=2,
                              steps={2: [replay_row('task one'), replay_row('task two')]})
    calls = []

    async def request(payload):
        calls.append(payload)
        return {'scores': [judgment('c0', .75)]}

    original = OptimismJudge(settings, request=request, tokenizer=CharacterTokenizer())
    asyncio.run(original.score_groups([('task one', ['hopeful']), ('task two', ['hopeful'])]))
    original.close()
    calls.clear()
    judge = RecoveryJudge(settings, history=history, request=request,
                          tokenizer=CharacterTokenizer(), on_divergence='rejudge')
    try:
        result = asyncio.run(judge.score_groups(
            [('task one', ['hopeful']), ('task two', ['changed answer'])],
            [[{'generated_tokens': 3}], [{'generated_tokens': 4}]],
        ))
        assert len(result) == 2 and judge.cache_hits == 1 and judge.requests == 1
        assert calls[0]['instruction'] == 'task two'
        assert judge.first_divergence == 2 and judge.last_batch_matched is False
    finally:
        judge.close()


def test_dry_run_builds_482_replay_30_fresh_plan_without_creating_output(tmp_path):
    settings = temporary_settings(tmp_path)
    settings.training.gradient_accumulation_steps = 4
    source = tmp_path / 'source'
    checkpoint = source / 'checkpoint-1024'
    checkpoint.mkdir(parents=True)
    for name in ['config.json', 'model.safetensors', 'optimizer.pt', 'scheduler.pt',
                 'rng_state.pth', 'training_args.bin']:
        (checkpoint / name).write_bytes(b'test')
    (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 1024}))
    rubric = (tmp_path / 'rubric.txt').read_text()
    (source / 'judge-rubric.txt').write_text(rubric)
    manifest = {'settings': settings.public_dict(), 'data': {'train': {'selected': 2}},
                'judge_rubric_sha256': hashlib.sha256(rubric.encode()).hexdigest()}
    (source / 'resume.json').write_text(json.dumps(manifest))
    (source / 'memory.jsonl').write_text(json.dumps({'step': 1506}) + '\n')
    (source / 'judge.jsonl').write_text(''.join(json.dumps(replay_row()) + '\n'
                                             for _ in range(1506)))
    prepared = settings.path(settings.data.output_dir)
    prepared.mkdir(parents=True)
    (prepared / 'train.jsonl').write_text(''.join(json.dumps({
        'prompt': [{'role': 'user', 'content': f'task {i}'}], 'prompt_id': str(i),
    }) + '\n' for i in range(2)))
    report = recover(settings, str(source), max_steps=1536, dry_run=True)
    assert report['replayed_updates'] == 482 and report['fresh_updates'] == 30
    assert report['fresh_judge_groups'] == 30
    assert report['final_checkpoint'].endswith('checkpoint-1536')
    assert not Path(report['output']).exists()


@pytest.mark.integration
@pytest.mark.parametrize(('steps', 'resume_step'), [(2, 1), (6, 5)])
def test_real_grpo_resume_and_raw_rollout_reuse_match_uninterrupted_weights(
    tmp_path, monkeypatch, steps, resume_step,
):
    from datasets import Dataset
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Lfm3Config, Lfm3ForCausalLM, PreTrainedTokenizerFast, TrainerCallback
    from trl import GRPOConfig

    from rlllmf.offload import OffloadGRPOTrainer
    from rlllmf.recovery import RecoveryGRPOTrainer

    torch.set_num_threads(1)
    torch.manual_seed(42)
    words = ['<pad>', '<eos>', '<unk>', 'user', 'assistant', 'task', 'a', 'b']
    words += [f'word{i}' for i in range(24)]
    raw = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token='<unk>'))
    raw.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>',
                                       eos_token='<eos>', unk_token='<unk>', padding_side='left')
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}"
        "{% if add_generation_prompt %}assistant {% endif %}"
    )
    model = Lfm3ForCausalLM(Lfm3Config(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, conv_L_cache=7,
        layer_types=['conv', 'full_attention'], max_position_embeddings=128,
        bos_token_id=None, pad_token_id=0, eos_token_id=1,
    ))
    initial = tmp_path / 'initial'
    model.save_pretrained(initial)
    tokenizer.save_pretrained(initial)
    model.config._name_or_path = str(initial)
    source = tmp_path / 'source'
    source.mkdir()
    settings = temporary_settings(tmp_path)
    settings.training.num_generations = 2
    settings.training.micro_batch_size = 1
    settings.training.gradient_accumulation_steps = 2
    settings.training.max_completion_length = 6

    async def request(payload):
        return {'scores': [judgment(row['id'], .9 if row['id'] == 'c0' else .1)
                           for row in payload['candidates']]}

    class Memory(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            with (source / 'memory.jsonl').open('a') as stream:
                stream.write(json.dumps({'step': state.global_step}) + '\n')

    args = GRPOConfig(output_dir=str(source), use_cpu=True, bf16=False, max_steps=steps,
                      per_device_train_batch_size=1, gradient_accumulation_steps=2,
                      num_generations=2, max_completion_length=6, temperature=.8, top_p=.9,
                      top_k=0, beta=.01, optim='adamw_torch', learning_rate=1e-3,
                      lr_scheduler_type='constant', save_steps=1, logging_steps=1,
                      report_to='none', disable_tqdm=True,
                      model_init_kwargs={'dtype': 'float32', 'device_map': None})
    dataset = Dataset.from_list([{'prompt': [{'role': 'user', 'content': f'task {text}'}]}
                                 for text in ['a', 'b', 'word0', 'word1']])
    judge = OptimismJudge(settings, source / 'judge.jsonl', request=request, tokenizer=tokenizer)
    trainer = OffloadGRPOTrainer(model=model, args=args, processing_class=tokenizer,
                                 reward_funcs=judge.reward_sync, train_dataset=dataset,
                                 callbacks=[Memory()])
    trainer.train()
    expected = {name: p.detach().clone() for name, p in trainer.model.named_parameters()}
    judge.close()
    history = ReplayHistory(source, resume_step, 1)
    checkpoint = source / f'checkpoint-{resume_step}'
    journal = tmp_path / 'rollouts'
    for attempt in range(2):
        output = tmp_path / f'recovered-{attempt}'
        restored_model = Lfm3ForCausalLM.from_pretrained(checkpoint)
        restored_model.config._name_or_path = str(initial)
        restored_args = copy.deepcopy(args)
        restored_args.output_dir = str(output)
        recovery_judge = RecoveryJudge(settings, history=history, request=request,
                                       tokenizer=tokenizer)
        recovery = RecoveryGRPOTrainer(
            model=restored_model, args=restored_args, processing_class=tokenizer,
            train_dataset=dataset, reward_funcs=recovery_judge.reward_sync,
            preloaded_checkpoint=checkpoint, journal_dir=journal,
        )
        recovery_judge.current_step = lambda: recovery.state.global_step + 1
        if attempt:
            def no_generation(*args, **kwargs):
                raise AssertionError('Expected reuse of the saved full token rollout')
            restored_model.generate = no_generation
        try:
            recovery.train(resume_from_checkpoint=str(checkpoint))
            assert recovery_judge.requests == 0 and recovery_judge.cache_hits == 1
            assert recovery.state.global_step == steps
            assert recovery.last_rollout_reused is bool(attempt)
            for name, parameter in recovery.model.named_parameters():
                torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
        finally:
            recovery_judge.close()

    # Exercise the public command path through its replay boundary and one fresh
    # update, verifying both full final checkpoint and inference model saves.
    import shutil

    import rlllmf.recovery as recovery_module
    from rlllmf.training import load_text_model

    shutil.rmtree(source / f'checkpoint-{steps}')
    (source / 'initial_model').symlink_to(initial, target_is_directory=True)
    rubric = (tmp_path / 'rubric.txt').read_text()
    (source / 'judge-rubric.txt').write_text(rubric)
    (source / 'resume.json').write_text(json.dumps({
        'settings': settings.public_dict(), 'data': {'train': {'selected': len(dataset)}},
        'judge_rubric_sha256': hashlib.sha256(rubric.encode()).hexdigest(),
    }))
    prepared = settings.path(settings.data.output_dir)
    prepared.mkdir(parents=True)
    (prepared / 'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in dataset))
    monkeypatch.setattr(recovery_module, 'require_gpu', lambda: None)
    monkeypatch.setattr(recovery_module, 'require_recovery_memory', lambda: None)
    monkeypatch.setattr(recovery_module, 'load_text_model',
                        lambda s, path: load_text_model(s, path, device='cpu'))

    async def mocked_api(self, payload):
        return await request(payload)

    monkeypatch.setattr(OptimismJudge, '_api_request', mocked_api)
    output = tmp_path / 'complete-recovery'
    recover(settings, str(source), str(output), max_steps=steps + 1)
    assert (output / f'checkpoint-{steps}/optimizer.pt').is_file()
    assert (output / f'checkpoint-{steps + 1}/optimizer.pt').is_file()
    assert (output / 'final/config.json').is_file()
    assert (output / 'final/model.safetensors').is_file()
    assert json.loads((output / 'trainer_state.json').read_text())['global_step'] == steps + 1
    assert json.loads((output / 'recovery-complete.json').read_text())['global_step'] == steps + 1
