# RLLLMF-GRPO-QUALITY

This experiment scores each answer for childlike voice and task quality in one
Haiku request. The configured reward is:

```text
k = ln(3) / 0.05  # approximately 21.972245773362193
reward = childishness - (10 / k) * ln(1 + exp(k * (0.65 - task_quality)))
```

The concise rubric is `prompts/childishness-quality.txt`. The structured response
requires both scores and a reason for each candidate. Raw judgments are cached;
the combination is calculated locally, so changing the reward mode or its
parameters can reuse cached raw scores. `judge.reward_mode = "quality_softplus"`
selects this smooth curve. Its center, sharpness and scale are configured as
`quality_softplus_center`, `quality_softplus_sharpness` and `quality_softplus_scale`.
For equal small score improvements, the quality-to-childishness reward sensitivity
is 5 at quality 0.65, 2.5 at 0.70, 1 at 0.75, and about 0.357 at 0.80. Quality's
incentive smoothly approaches zero at high scores; childishness always contributes
linearly. Combined rewards can be negative and are not restricted to [0, 1].
GRPO still normalizes combined rewards within each question; these sensitivities
describe the reward tradeoff, not guaranteed parameter-update sizes.

The stopped `quality-01` run used `quality_gated`, meaning the older product
`task_quality * (0.7 + 0.3 * childishness)`. That mode remains available for
reproducing the previous experiment. Its saved logs are not recalculated or
overwritten. The next experiment uses a separate `quality-02` output folder.

`judge.score_name` selects the main objective; another
objective needs a matching two-score rubric. `judge.reward_mode = "trait_only"`
enables the older single-objective behavior with the older rubrics.

Start the experiment from the imported checkpoint 512:

```bash
scripts/container-python -m rlllmf train \
  --model runs/imported/checkpoint-512 \
  --output runs/lfm3-childishness-quality-02
```

This is a fresh experiment initialized from checkpoint 512 weights. The step
counter starts at zero, the optimizer is new, and checkpoint 512 is the frozen
KL reference. The current `.config` runs **1024 additional updates** at 5e-6
with a constant learning rate, four answers per question, microbatch 1 and
accumulation 16. Edit `[training].max_steps` to change the new experiment duration.
Checkpoints save every 256 updates and all are retained. Resuming this new run
later uses `--resume latest --output runs/lfm3-childishness-quality-02`.

The local generation cap stays at 4096 tokens. The final-answer cap sent to the
judge is raised from 1024 to 4096 so task-quality scoring can inspect long final
answers. Thinking is stripped before the API call. Haiku gets two scores per
candidate in a single request, with an output budget of 2048 tokens and up to
10000 fresh requests per invocation. Inspect generation diagnostics for any
remaining clipping or unfinished answers.

`judge.jsonl` records both raw scores, both reasons and the combined `reward`.
`trainer_state.json` separately logs `judge/childishness`, `judge/task_quality`,
and `judge/combined_reward`. Evaluations and comparisons report both components
and the combined reward. The live HTML viewer shows all three values.

The plotting script supports `--metric childishness`, `--metric task_quality`,
`--metric combined_reward`, and `--metric reasoning_tokens`.

Only `runs/imported/checkpoint-512` was imported from the source experiments.
Prepared RL/evaluation instructions are rebuilt using the same reserved SFT
question pool. The Python environment and Hugging Face download cache are shared
through symlinks; judge caches and future run outputs belong to this project.
Credentials are in the gitignored `.config` (permissions 0600). All of `runs/`,
`data/`, and `.cache/` remain gitignored, including the imported checkpoint.

# RLLLMF-GRPO

Original GRPO experiment. The critic-free PPO variant lives in the sibling
`RLLLMF-PPO` folder. This project's training algorithm and existing runs are preserved.

Full-parameter training of **`LiquidAI/LFM3-1.7B`**, using
**TRL**, standalone **AlpacaFarm instructions**, and an **Anthropic API optimism judge**.

For the class report, see the [experiment history and paper comparison](docs/experiment-history.md),
including requested changes, verified run settings/results, and the current configuration snapshot.

There is no local reward model, no LoRA, and no initial SFT stage. Every language-model
parameter is trainable. The default model is the post-trained text checkpoint, not the base model.

The earlier Qwen results remain under `runs/optimism-smoke/`. LFM uses separate folders:
`runs/lfm3-optimism-smoke/`, `runs/lfm3-optimism/`, and `data/lfm3/`.
The LFM offload configuration has passed two GPU updates with synthetic rewards;
the live Haiku smoke test remains available for you to run. Qwen and LFM measurements
are identified separately below.

## Install

Use Linux, Python 3.11–3.13, an NVIDIA GPU, and an NVIDIA driver compatible with the
CUDA libraries installed by PyTorch. The dependencies are pinned in `pyproject.toml`
and resolved in `uv.lock`.

```bash
uv sync --dev
source .venv/bin/activate
```

The PyTorch CUDA dependencies require several GB of disk space. No separate vLLM server,
FlashAttention build, Unsloth, or system CUDA toolkit is required for the default path.
The installed Transformers version has native LFM3 support; no remote Python model code
is needed. Standard PyTorch/SDPA is used for attention.

## Configure

The existing `.config` retains your API key. It is TOML, **not a shell script**. API settings:

```toml
[api]
anthropic_api_key = "your-key-here"
hf_token = "" # Optional for public datasets/model
```

The full template is `.config.example`. The actual `.config` is git-ignored and created
with owner-only permissions. `ANTHROPIC_API_KEY` and `HF_TOKEN` environment variables
override the file. Keys are never written into run manifests. Do not paste keys into
commands, commits, or screenshots.

Training uses `lr_scheduler_type = "constant"` in `[training]`, keeping the configured
`learning_rate` unchanged throughout the run, with no warmup. This is passed explicitly
to TRL instead of its default linear decay. For a different schedule, set this field to
a TRL-supported scheduler name, such as `"linear"` or `"cosine"`.

The default judge is `claude-haiku-5-5`; edit `[judge].model` to an accessible Claude
model ID for your account. The rubric is in `prompts/optimism.txt`.

The default reward is optimism. For the alternative childlike speaking-voice experiment,
set both fields in the existing `[judge]` section:

```toml
rubric_path = "prompts/childishness.txt"
score_name = "childishness"
```

This measures a stylized young child's own voice: concrete phrasing, spontaneous
curiosity, a slightly naive perspective, and playful imagination. It does not measure
tantrums, intelligence, mistakes, or how well an adult explains something to children.
The [full rubric](prompts/childishness.txt) gives fixed anchors and calibration examples.
Plain adult wording and empty answers score 0; a coherent, sustained childlike voice
scores near 1. One emoji, "yay", or a child-related topic does not establish that voice.

Only the selected trait is optimized. Logs, summaries, and the viewer use the selected
score name; `childishness` is displayed as "Childlike voice". Old optimism logs retain
their labels and format. The rubric/score-field pairing is checked before API requests.
No childlike experiment is activated automatically by adding the file. After selecting
it, use a fresh run folder, for example `--output runs/lfm3-childlike-01`, to keep the
experiment distinct. Choosing an existing optimism-trained checkpoint as the policy
would be a separate sequential-training experiment; the normal fresh run loads the
original model configured in `[training]`.

For **conciseness while preserving the requested content**, select:

```toml
rubric_path = "prompts/conciseness.txt"
score_name = "conciseness"
```

The [conciseness rubric](prompts/conciseness.txt) penalizes boilerplate, repetition,
irrelevant background, and redundant conclusions. High scores require task fulfillment;
empty answers score 0 and substantial omissions cap the score at 0.40. Explicit requests
for detail, length, format, or item counts take priority over shortening the answer.
This is an LLM-judged score rather than a raw token-count reward, and it measures final
answers rather than hidden thinking. Scores are context dependent: a necessary detailed
explanation can score higher than a short incomplete answer. Use a separate output
folder such as `--output runs/lfm3-conciseness-01` for this experiment. Adding the rubric
does not change the active `.config` or launch training/calibration calls.

The judge sees the configured final-answer prefix only. If that prefix is cut off, it
cannot establish completeness of the unseen full answer; consider this when interpreting
conciseness scores for clipped responses. Existing generation diagnostics record clipping.

Original-model generation caching is independent of the judge's trait. Generation
fingerprints cover generation/decoding code, not scoring/reporting code. An increased
answer cap can still require fresh generation: old logs/cache contain only the earlier
capped answer prefix, so a cached 512-token answer cannot supply a new 1,024-token prefix.

Haiku 5.5's own thinking is explicitly disabled for judging; only its text JSON is parsed.
Policy thinking is enabled in both training and evaluation. Thinking-marker tokens are
not suppressed. Two independent limits are configured in both sections:

```toml
max_completion_length = 2048 # Local generation: thinking + answer, excluding prompt
max_answer_tokens = 256     # Final-answer text sent to judge, excluding thinking
enable_thinking = true
```

`max_completion_length` maps to TRL's completion budget and Transformers' `max_new_tokens`.
It is a total budget of 2,048 generated tokens, not 2,048 thinking tokens plus another 256.
After generation, thinking blocks are removed and only the first 256 final-answer tokens
are sent to the judge and stored in answer logs. Thinking does not consume that answer
allowance. If the model spends all 2,048 tokens thinking without reaching a final answer,
the answer is still empty; unfinished thinking is never sent as an answer. The extractor
also handles an opening `<think>` tag already present in the prompt.

The answer cap is measured with the **local policy tokenizer**. Haiku may tokenize that
text differently, and its request also contains the rubric, instruction, JSON formatting,
and up to four candidates. This is not a 256-token cap on the entire Anthropic request.
Haiku's own structured scoring output still has its separate `judge.max_tokens` budget.

Generation diagnostics include total local tokens, token-limit status, unfinished thinking,
full final-answer token count, tokens sent per answer, and whether the answer was clipped
for judging. No thinking text is stored in those diagnostics. The training loss and KL
still operate on the full generated sequence, including thinking and any final-answer
suffix beyond the judged prefix. Generation is not stopped at 256 final-answer tokens.
The viewer explains empty answers and indicates when only a prefix was judged.

Validated with real local LFM inference: 642 generated tokens including thinking,
369 final-answer tokens extracted, and exactly 256 final-answer tokens in a captured
mock judge request. No paid API call was made. See `runs/split-token-limits-check.json`.
This validates extraction and request limits, not full-training GPU fit at 2,048 tokens.

Investigation of `runs/lfm3-run-01`: 11 of 256 logged answers were empty. A replay reproduced
an original empty answer containing 256 tokens in an unclosed thinking block; extraction
removed the entire block. A separate controlled raise-letter generation check produced
one empty answer out of four without suppression and none out of four with suppression.
That temporary suppression approach has been removed; local generation now allows
thinking under the larger total budget described above.
Original logs contain only filtered answers, so every historical empty completion cannot
be reconstructed. Existing run logs, checkpoints, and reward rules are unchanged. Haiku's
neutral score on an empty answer is not a correctness judgment.

## Check and prepare

```bash
python -m rlllmf check
python -m rlllmf train --dry-run
python -m rlllmf prepare
```

`check` reports library versions, key presence (never the key), and GPU capabilities.
`--dry-run` checks configuration without any downloads, GPU allocation, or API calls.
`prepare` downloads the tokenizer and AlpacaFarm's `unlabeled.json` and `val.json`.
It combines `instruction` with optional `input`, discards all existing answers, removes
duplicate tasks, prevents train/eval overlap, and filters tasks exceeding the token limit
instead of cutting away part of the instruction. A single user-message container applies
the model's chat template; **there are no earlier dialogue rounds or system instructions
to be optimistic**. Output is in `data/lfm3/train.jsonl`, `data/lfm3/eval.jsonl`,
and `data/lfm3/manifest.json`.

The defaults select 1,000 training and 100 held-out instructions. Original files contain
20,000 and 2,000 respectively. Source hashes and model/data revisions are recorded.

With the current single-GPU settings, each optimizer step uses 16 prompt-answer pairs:
four instructions, each with four candidates. Each question's four candidates are scored
together in one judge request. The 16-step pilot uses 64 training questions, scores 256
answers, and makes up to 64 fresh judge requests. It does not cover the full training split.
A full held-out evaluation scores 100 answers in up to 100 requests. Evaluating before and
after training brings the total to 456 answer scores and up to 264 requests. The paid smoke
test adds one optimizer step (four questions, 16 answers, up to four requests). Cache hits
can reduce requests; transport retries are additional attempts.

Set `revision` to explicit Hugging Face commit IDs if you require immutable input assets.

## Measure the original model

For an automatic original-versus-trained comparison, run one command:

```bash
scripts/container-python -m rlllmf compare \
  --model runs/lfm3-run-02/final \
  --output runs/lfm3-comparison-02 \
  --limit 25
```

Omit `--limit` for all 100 held-out questions. This loads the exact original snapshot
from `initial_model/` beside the trained checkpoint, generates one answer per model per
question using matching seeds and sampling settings, and unloads the first model before
loading the second. It uses the current evaluation thinking/generation/answer limits;
thinking is excluded from the judge input. Both sides use repetition penalty 1.0 explicitly.
For a model without a saved original snapshot, supply `--baseline MODEL_OR_PATH`.

Original-model generations are now cached automatically in `.cache/baseline.sqlite3`,
shared by `compare` and baseline `evaluate` (without `--model`). The trained-model side
is generated fresh. With the current settings, `evaluation.temperature=0.0` selects greedy
decoding (`do_sample=False`); sampling-only parameters are disabled. Training still requires
a positive temperature. A positive evaluation temperature enables seeded sampling instead.

The baseline cache is per question, so a 100-question comparison reuses overlapping
answers from an earlier 25-question comparison. Its keys cover model-weight contents,
model configuration, tokenizer/chat template, exact prompts, thinking and token limits,
effective decoding settings, implementation/software/device context, and sampling seeds
when applicable. Identical original checkpoints in different run directories can reuse
answers. Changed weights, prompts, or generation settings produce cache misses. Judge
model and rubric changes do not require regenerating local baseline answers.

The first lookup fingerprints the model files without loading GPU weights. Fingerprints
are reused while file metadata stays unchanged. On a full generation-cache hit, the original
model is not loaded onto the GPU at all. The command prints cache hits/misses and records
them in `comparison-run.json` and the evaluation summary. Each completed original answer
is committed immediately, so interrupted baseline generation can be reused from a new
output directory. Model weights themselves remain in the normal Hugging Face/run folders.

Disable generation reuse with `[evaluation].cache_baseline=false`, or change
`baseline_cache_path` to use a separate cache. Baseline answers are still copied into each
comparison output's `baseline.generations.jsonl`. Judge judgments use the separate existing
judge cache, whose keys include the answer pair, rubric, and judge model; a new competing
trained answer can therefore require new API scoring even when its baseline is cached.

If judging fails after both generation files were saved, retry in the same directory:

```bash
scripts/container-python -m rlllmf compare \
  --model runs/lfm3-run-02/final \
  --output runs/lfm3-comparison-02 \
  --resume
```

This reuses both saved answer sets without loading model weights, and reuses valid cached
judgments. Only uncached pairs need new API calls. It preserves the original evaluation
settings from `comparison-run.json`, including temperature and answer limits, even if
`.config` has since changed. The current judge model/rubric settings apply to scoring;
changing those can require judging all pairs again. Omit `--limit` and `--baseline` when
resuming. A comparison that already completed is protected from accidental reruns.

The judge sees each question's two anonymous answers together, with candidate order
shuffled and scores mapped back correctly. It scores both against the optimism rubric;
the larger optimism score determines the outcome and equal scores tie. This measures
optimism, not overall correctness. Unlike two independent `evaluate` commands, the paired
comparison puts both candidates in the same judge context. It needs up to one fresh judge
request per question (25 with the example above, 100 for the full set), subject to cache
reuse and transport retries. It is a separate command; training does not launch it implicitly.

Results in the new output directory:

- `comparison.summary.json`: original/trained means, differences, optimism win/tie rates,
  relevance/consistency diagnostics, empty-answer rates, truncation rates, and API usage.
- `comparison.html`: a standalone report with both answers and judgments side by side,
  one question per page, plus an overall comparison summary.
- `comparison.jsonl`: paired results, also discovered by the live viewer.
- `baseline.generations.jsonl`, `trained.generations.jsonl`: generations saved before API
  scoring, and `comparison.judge.jsonl`: the judge log.
- `comparison-run.json`: model sources and redacted configuration.

Comparison outputs are protected against overwriting; use a different output folder for
another evaluation. Smaller empty/contradiction/clipping rates are preferable; the summary's
delta fields always mean trained minus original. Relevance and consistency remain limited
LLM diagnostics, rather than comprehensive accuracy tests. No judge calls are made until
you run the command.

The following uses your GPU and makes paid Anthropic API calls:

```bash
python -m rlllmf evaluate --output runs/lfm3-base-evaluation.jsonl --limit 25
```

Omit `--limit` for all prepared evaluation instructions. Each answer is scored independently
using fixed optimism anchors. Factual consistency and relevance are separate diagnostics;
they are **not** included in the training reward. These are LLM judgments, not guarantees
of correctness. Use identical evaluation samples/settings for each checkpoint.

## Run one real optimizer step

```bash
python -m rlllmf train --smoke-test
```

This is a **real training step with paid API scoring**, saved separately to
`runs/lfm3-optimism-smoke`. It exercises generation, scoring, backward, optimizer-state
allocation, and checkpoint saving. Check `memory.jsonl` to see the actual GPU peak.
The no-API offload test validated two GPU updates. This command also validates live
judge scoring and checkpoint saving for LFM.

## Train and evaluate

```bash
python -m rlllmf train
python -m rlllmf evaluate --model runs/lfm3-optimism/final --output runs/lfm3-trained-evaluation.jsonl --limit 25
```

Resume an interrupted run using:

```bash
python -m rlllmf train --resume latest
```

Use `--output runs/another-run` for another experiment. To resume a smoke run, also pass
`--smoke-test`. Training automatically prepares data with the active tokenizer and settings.
Keep the configuration, rubric, and dataset fixed when resuming. Do not run multiple
training processes against the same output directory.

Each update samples four responses to each of four instructions with the default settings.
The judge scores each question's group in one API call, with candidate order shuffled and
IDs mapped back to the correct completions. Equal or duplicate responses can receive tied
scores. GRPO computes advantages and updates the policy; only the optimism score enters
the reward. It is normal for some groups to have no reward variance and no preference signal.

## Memory defaults

For GRPO, `[training].trim_training_padding = true` avoids running short responses
through the longest response's trailing padding during reference scoring and policy
updates. Each microbatch keeps its full prompt, all generated tokens including EOS,
and any internal attention-mask gaps. Only the fully masked suffix is removed; token
statistics are then padded back to the original shape for GRPO's masks and loss.
The group advantages, KL reference, token normalization, generation limits and batch
sizes are preserved. This is ordinary padded training with shorter microbatches,
not document packing. Multimodal inputs, explicit sampling supports and MoE auxiliary
loss use the original path. Set the option to `false` to use the original implementation.
The option defaults to false when omitted from older configuration files.
The measured investigation is recorded in
[performance-investigation-20261117.json](docs/performance-investigation-20261117.json).
On one replayed mixed-length batch, scoring and update time fell from 45.56 to
14.75 seconds with optimizer state allocated. Adding the separately measured 73.91
seconds of generation estimates a 25.8% reduction for that batch before API waiting;
this is not a whole-run speed guarantee. All GPU benchmarks used synthetic rewards
and did not save experiment checkpoints.

- BF16 weights on supported GPUs; older GPUs use FP16 autocast with FP32 parameters,
  which costs more memory and may not fit this profile.
- 8-bit AdamW optimizer states; the model weights remain fully trainable.
- Gradient checkpointing, microbatch 1, gradient accumulation 16 (effective batch 16).
- Four candidates, up to 2,048 generated tokens (thinking + answer), at most 512 prompt tokens.
- At most 256 final-answer tokens per candidate are passed to the judge.
- Transformers generation (`use_vllm=False`), with thinking enabled in the chat template.
- Reference KL enabled with `beta=0.01`. A local **text-only** initial checkpoint is saved
  so TRL reloads the proper reference architecture rather than the vision model.
- `reference_cpu_offload_gib=2.0`: selected frozen-reference decoder layers live in CPU
  RAM between reference-scoring batches. They move to GPU once for the whole batch of
  reference forward passes, then return to CPU before policy backward and optimizer work.
  The trainable policy, gradients, optimizer states, and remaining reference layers stay
  on GPU. No reference computation is performed on CPU.

The offload budget limits selected reference parameter/buffer bytes, **not total process
RAM**. Whole decoder layers are selected without exceeding the limit, so the actual
amount can be slightly smaller. No permanent second CPU copy of those weights is kept
while they execute on GPU. Set `reference_cpu_offload_gib=0` to disable this feature.
The chosen layers and actual size are printed and recorded in `run.json`; transfer timing
and completed reference batches are written to `reference-offload.json`. This path is for
the current single-GPU setup and a fixed reference; it does not support DeepSpeed, FSDP,
or reference synchronization. KL and full-parameter policy training are preserved.
The container launcher also defaults to `PYTORCH_ALLOC_CONF=expandable_segments:True`
to reduce fragmentation as reference weights move between devices. An explicit value of
that environment variable overrides the default. For direct Python commands, set the same
environment variable before launching Python. Unused reference GPU cache is released
after reference scoring, before policy backward.

The earlier LFM smoke run ran out of VRAM before its first optimizer update. Keep its
records and retry in a new folder after enabling offload:

```bash
scripts/container-python -m rlllmf train --smoke-test --output runs/lfm3-smoke-offload2g
```

For a no-API GPU check that also exercises a second update with allocated optimizer states:

```bash
scripts/container-python scripts/gpu_smoke.py --steps 2
```

This uses the longest prepared instructions needed to fill the generation batch, synthetic
test rewards, and temporary model files. It does not save a trained experiment checkpoint.

Validated on the RTX 5060 Ti: two real LFM3 optimizer updates completed with all
1,728,801,536 parameters trainable, 1.9326 GiB of reference weights offloaded under the
2 GiB budget, KL enabled, four questions/four answers per question per update, and up to
128 generated tokens. Peak PyTorch allocation was 13.987 GiB; reservation was 14.064 GiB.
The test used synthetic rewards and made zero API calls. See
`runs/gpu-check-20261115T051914Z.json`. This leaves limited GPU headroom; these results
do not establish fit for longer sequences or additional GPU workloads. The allocator
setting above is part of the tested configuration. These earlier memory measurements
used a 128-token generation limit and do not establish training fit at the new 2,048-token
limit. Use the no-API GPU smoke check to measure that configuration before a paid run.

If you run out of memory, first reduce `max_completion_length` and `max_prompt_tokens`.
You can reduce `num_generations` to 2 while keeping accumulation divisible by it.
Setting `beta=0` removes the reference model and KL regularization; that changes the
experiment as well as saving memory. No optimization silently switches training to adapters.

## Outputs and API behavior

Start the live viewer in a separate terminal and leave it running:

```bash
.venv/bin/python -m rlllmf view
```

Open **http://127.0.0.1:8765** in a browser. The run selector automatically discovers
`judge.jsonl` and `*.judge.jsonl` under `runs/`, including new runs. The selected log is
read directly and refreshed every two seconds without resetting your answer number.
New answers appear as training writes them; an unfinished last JSON line is retried on
the next refresh. The newest modified log is selected on initial load. Choose another
run from the selector; new runs do not interrupt the run you are reading.

No exports, GPU, API key, or judge API requests are needed. This read-only server binds
only to localhost and exposes judge logs and model names, not the rest of the workspace.
Use `--port 8766` if needed, or `--runs-dir /path/to/runs` to watch a different folder.

The page shows one candidate answer alongside its instruction, judge scores, diagnostics,
and explanation. Use Previous/Next, arrow keys, or the answer number. **Open log** also
supports manually loading evaluation JSONL or other local logs; select a run again to
resume live updates. Opening `judge-viewer.html` directly without the local server still
works in manual-file mode.

To create an HTML file with a log already loaded:

```bash
.venv/bin/python -m rlllmf view runs/optimism-smoke/judge.jsonl
```

This optional export is a static snapshot. For automatic updates, use the localhost URL
above. Model names come from recorded metadata when available; logs record the judge
name per group, so switching judges does not mislabel earlier evaluations. The full judge
response section shows stored judgment JSON, not an original raw HTTP response.

Separate experiments need separate output directories, for example:

```bash
scripts/container-python -m rlllmf train --output runs/experiment-01
scripts/container-python -m rlllmf train --output runs/experiment-02
```

Each folder contains its own `judge.jsonl`, initial model, checkpoints, final model, and
metadata. A fresh run refuses an output folder containing `run.json`; it does not overwrite
that experiment. `--resume latest --output runs/experiment-01` continues that same run
and appends judgments. Its usage summary reports the latest invocation. The default
folders are `runs/lfm3-optimism` and `runs/lfm3-optimism-smoke`; there is no automatic timestamp
suffix. Changing hyperparameters does not create a new folder automatically.

- `initial_model/`: the unchanged text-only starting policy/reference, needed for resuming.
- `checkpoint-*` and `final/`: model weights and resumable checkpoints.
- GRPO checkpoints are kept indefinitely; `[training].save_steps` controls the interval
  (256 in the current config). Resuming applies the current interval even if the saved
  checkpoint used another one. SFT retains its separate once-per-epoch save policy.
- `run.json`: redacted config, hardware, parameter count, and dataset metadata.
- `judge.jsonl`: instructions, responses, optimism, relevance, and consistency judgments.
- `judge-usage.json`: API-request and token counters for this invocation.
- `memory.jsonl`: GPU memory after each optimizer step.
- `skipped-questions.jsonl`: refused/invalid training questions and their replacements.
- `judge-replacements.json`: replacement counts for this invocation.
- Evaluation JSONL files and `.summary.json`: held-out scores and diagnostics.

Judgments are cached in SQLite using the model, rubric, schema, instruction, and candidates.
Changing the rubric or judge model invalidates the relevant cache entries. Transient HTTP
failures use the SDK's bounded retries. During training, a refusal, truncated judge response,
or persistently invalid evaluation discards that question's entire answer group and draws
a replacement question from the prepared **training** split. Its answers are generated by
the current policy. Valid groups retain their existing token IDs and scores; each update
still uses the configured number of valid answers. No zero reward is fabricated.
The API schema requires a separate score field
for every candidate ID. Invalid JSON/score responses are retried up to `judge.max_retries`
times, and invalid attempts are recorded in `*.errors.jsonl` beside the judge log. Successful
judgments remain cached; invalid scores are never cached or used as rewards.
`[judge].max_replacements_per_step` limits extra questions per optimizer update (16 by
default). If this limit, the request budget, or a persistent API access/network error is
reached, training saves the last completed optimizer step before exiting. Skipping handles
unusable evaluations, not wrong credentials or a prolonged service outage. Standalone
evaluation/comparison still reports errors instead of changing its held-out question set.
New training judge records include `training_step`; generation-length/token metrics include
discarded attempts, while reward/loss metrics use the accepted training batch.
`[judge].max_requests=500` caps fresh API calls
per invocation, excluding transport retries; it is **not a dollar budget**. Cache hits do not
count toward that cap; retries for invalid judge output do count. Instructions/responses
are sent to Anthropic for scoring. No model
weights or keys are uploaded by this project.

## Validation

```bash
python -m unittest discover -s tests -p 'test_core.py' -v
pytest -q
```

The integration tests use tiny randomly initialized Qwen3.5 and LFM3 models on CPU.
They exercise the actual loader, generation, full-parameter backward/optimizer step, and
frozen KL reference with a mocked judge. No model weights are downloaded and no API calls
are made. Separate tests verify thinking text never reaches judge requests or answer logs.
The real LFM GPU memory test is documented in the memory section above.

Historical Qwen validation during setup (2026-11-15): all initial 13 tests and Ruff checks passed.
The actual Qwen checkpoint was also downloaded and loaded as `Qwen3_5ForCausalLM`
(752,393,024 language parameters, no vision tower); BF16 CPU forward and generation
succeeded with no missing or mismatched text weights. The tokenizer, model weights, and
prepared 1,000/100 instruction splits are cached locally. No live Anthropic request was
made. That initial check ran in the nested agent sandbox; subsequent GPU validation in
the actual Distrobox is documented below.

## GPU passthrough in this Distrobox (validated)

The host has an RTX 5060 Ti with 15.48 GiB available physical VRAM. The real
`python-base` Distrobox already shared the host GPU device nodes; the nested agent
sandbox hid those nodes, and the Distrobox itself lacked NVIDIA library discovery.

The live container now discovers the host driver through
`/etc/ld.so.conf.d/90-rlllmf-nvidia-host.conf`, and `nvidia-smi` is available.
Python development headers were installed for Triton's GPU compilation. No container
restart was required. The Distrobox manifest at
`/home/estevan/Ambientes/.distrobox/python-base/python-base.ini` now includes
`additional_flags="--device=nvidia.com/gpu=all"` for future recreations. Its Containerfile
now includes `build-essential` and `python3-dev`. Both original files have `.before-gpu`
backups next to them. A separate temporary CDI GPU container was also tested successfully.

To run outside a nested agent sandbox while staying in this same Distrobox:

```bash
scripts/container-python -m rlllmf check
scripts/container-python scripts/gpu_smoke.py
```

`container-python` uses the rootless Podman socket and runs the project's existing
virtual environment in `python-base` as your UID/GID, with a project-local Triton cache.
Override `RLLLMF_CONTAINER_NAME`, `RLLLMF_PODMAN_SOCKET`, or `RLLLMF_CONTAINER_USER`
if your runtime settings change. Normal terminals inside
the Distrobox can also use the direct `.venv/bin/python` commands above.

The GPU smoke script makes **no API calls**. It uses the configured checkpoint and the
production GRPO configuration with a synthetic test reward, discards its temporary
weights, and writes a JSON diagnostic to `runs/gpu-check-*.json`. It does not train your
optimism experiment. The earlier successful **Qwen** test used:

- All 752,393,024 language-model parameters trainable, BF16, bitsandbytes AdamW with
  verified uint8 optimizer buffers, and gradient checkpointing.
- The longest prepared training instruction: 252 tokens; four 128-token completions.
- Reference KL enabled, `beta=0.01`.
- One optimizer update completed with observable finite weight changes.
- Peak PyTorch allocation **7.135 GiB**, peak reservation **7.195 GiB**. These figures
  exclude other applications' GPU memory and apply to this tested sequence/batch size.

The real paid judge/training smoke test for the currently configured LFM model is:

```bash
scripts/container-python -m rlllmf train --smoke-test
scripts/container-python -m rlllmf train
```

GPU execution and live Anthropic scoring were verified with Qwen. LFM GPU execution
has now passed two updates with synthetic rewards and 2 GiB reference offload, as
documented above. Its tokenizer and full weights are cached. Retry its paid smoke test
in a new output directory because the earlier failed LFM smoke folder already exists.

References: [LFM3 model card](https://huggingface.co/LiquidAI/LFM3-1.7B),
[TRL GRPO](https://huggingface.co/docs/trl/grpo_trainer),
[Qwen3.5 text-only loading](https://huggingface.co/docs/transformers/model_doc/qwen3_5),
[AlpacaFarm](https://arxiv.org/abs/2305.14387),
[Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

## Collect childlike answers for a later supervised warm-up

The collection command uses Haiku to answer ordinary AlpacaFarm instructions in the
childlike speaking voice defined in [prompts/generate-childishness.txt](prompts/generate-childishness.txt).
It collects data; it does not start supervised fine-tuning or GRPO.

```bash
# Reserve/check 500 questions without calling Anthropic:
scripts/container-python -m rlllmf collect --prepare-only

# Generate and save 100 NEW question-answer pairs:
scripts/container-python -m rlllmf collect --count 100

# Run the same command later to add another 100:
scripts/container-python -m rlllmf collect --count 100
```

The cumulative dataset is **`data/childishness-sft/examples.jsonl`**. Each record has
an `instruction`, an `answer`, a two-message `messages` conversation, a stable prompt
ID, and teacher/run/usage metadata. The student conversation contains only the ordinary
user instruction and Haiku's final answer. The teacher's childlike-style system prompt,
JSON wrapper, and hidden thinking are not student training targets.

`--count` means **additional pairs**, not the desired total. The command automatically
skips questions already answered. Existing records are never replaced. Each accepted
answer is appended and flushed to disk immediately; if a request fails, completed pairs
remain available. On the next invocation, collection continues with unused questions.
For example, after a failed invocation that saved 30 pairs, `--count 70` adds 70 for a
total of 100; `--count 100` adds 100 for a total of 130.

The initial 500 questions are saved in **`data/childishness-sft/pool.jsonl`**. They come
from AlpacaFarm's separate `sft.json` source split; its original answers are discarded.
Before reserving or generating, the collector excludes every task in the configured
full RL and evaluation source splits, prepared train/eval files, and saved experiment
question logs. Deduplication includes capitalization and whitespace normalization of
the complete instruction plus input. This guards against repeated tasks, not arbitrary
semantic paraphrases. The pool manifest records source hashes and exclusion checks.
Future `prepare` calls also exclude the reserved pool, even when RL/eval sample counts
increase; loading a manually overlapping prepared dataset fails rather than leaking it.

To extend the pool after using its initial 500 questions:

```bash
scripts/container-python -m rlllmf collect --pool-size 1000 --count 100
```

This preserves the original reservations and generated examples, appends disjoint
questions to the pool, and adds 100 new pairs. A pool is never implicitly shrunk.
Separate invocations get their own timestamped folders under
**`runs/childishness-collection/`**, containing the exact generation prompt, selected
question IDs, configuration, usage, and any sanitized errors. All invocations append to
the same configured dataset. Concurrent collectors for that dataset are rejected.

The `[collection]` section controls pool size, default examples per run (100), generation
prompt, output directory, teacher output budget, and concurrency. The teacher uses
`[judge].model` and the existing `[api].anthropic_api_key`; the judge's scoring rubric
and response-length cap do not control answer generation. The installed Anthropic SDK
uses model-default sampling and offers no `temperature` argument for Messages requests.
No additional scoring request is made during collection. Inspect the generated targets
before SFT; exact-format tasks can appropriately leave little room for a childlike voice.

## Supervised warm-up, then a new childlike GRPO run

Train the collected teacher answers with TRL's `SFTTrainer`:

```bash
# Validate data, token lengths and update count without training:
scripts/container-python -m rlllmf sft --dry-run

# Train and save a separate supervised model:
scripts/container-python -m rlllmf sft --output runs/lfm3-childishness-sft-01
```

The `[sft]` defaults are **two epochs, learning rate 5e-6 constant, microbatch 1,
gradient accumulation 16, maximum full sequence length 2,048 tokens**. With 500
examples, there are **32 weight updates per epoch / 64 total**, including a final
partial batch each epoch. This stage uses all 500 teacher pairs, once per epoch.
It does not inherit GRPO's 512-update limit or generate four answers per question.

SFT trains all model parameters with BF16, gradient checkpointing and the configured
8-bit optimizer. It computes next-token loss on **the saved answer and turn ending
only**; user instructions and the empty thinking scaffold in LFM's chat template are
masked. The supervised examples contain no teacher system prompt or hidden reasoning.
The template is applied with thinking disabled for these answer-only targets; the
existing GRPO/evaluation thinking settings are unchanged. Overlong examples cause an
error before model loading rather than silent answer truncation. Prepared RL and
evaluation questions are checked for overlap before training.

The completed supervised model and tokenizer are saved in:

```text
runs/lfm3-childishness-sft-01/final/
```

The run also keeps a frozen copy of the selected training data, its hash, configuration,
training metrics and GPU memory logs. New teacher pairs collected later do not alter
an existing SFT run's snapshot. It saves a resumable checkpoint **once per epoch** and
keeps **one** such checkpoint, plus the final model. To resume an interrupted SFT run:

```bash
scripts/container-python -m rlllmf sft \
  --output runs/lfm3-childishness-sft-01 --resume latest
```

After SFT finishes, start a **fresh** GRPO run from its saved weights:

```bash
scripts/container-python -m rlllmf train \
  --model runs/lfm3-childishness-sft-01/final \
  --output runs/lfm3-childishness-after-sft-01
```

Select `prompts/childishness.txt` and `score_name = "childishness"` under `[judge]`.
The current project config already selects that objective. The `--model` override
applies to this invocation; it does not rewrite `[training].model` in `.config`.
The new run starts with a fresh GRPO optimizer and snapshots the SFT model as its
`initial_model/`, making that SFT model the **frozen KL reference**. The Haiku scoring,
separate RL questions, batching and other GRPO settings remain as configured. SFT
checkpoints and the older RL experiments remain in their own folders.

For the new run, `compare --model .../final --output ...` automatically compares the
post-RL model against this post-SFT reference. This isolates RL's incremental effect.
An original-versus-SFT comparison can separately use `--baseline LiquidAI/LFM3-1.7B`.

## Run SFT and GRPO automatically overnight

Instead of starting the stages separately, run:

```bash
scripts/container-python -m rlllmf pipeline
```

With the current config, this completes the 500-example, two-epoch SFT stage, saves
`runs/lfm3-childishness-sft-01/final/`, then starts the configured 512-update GRPO run
in `runs/lfm3-childishness-after-sft-01/` using that SFT model as policy and frozen
reference. Separate Python processes release SFT's GPU memory before loading GRPO.
Any failed stage stops the pipeline; GRPO requires a completed SFT model and training
state, rather than merely the presence of a partially written `final/` directory.

To leave it running after closing the terminal:

```bash
nohup scripts/container-python -m rlllmf pipeline \
  > runs/childishness-pipeline.log 2>&1 < /dev/null &
```

Follow the log with `tail -f runs/childishness-pipeline.log`. Pipeline status and
the stage commands are saved in `runs/lfm3-childishness-sft-01/pipeline.json`.

After an interruption, with the previous processes stopped:

```bash
scripts/container-python -m rlllmf pipeline --resume
```

This skips completed stages and resumes an unfinished stage from its latest complete
checkpoint, preserving optimizer state. If the interrupted stage has never saved a
checkpoint, it reports that explicitly; use a new output for that stage to restart it.
The existing RL run must have started from the selected SFT final model. Existing runs
are rejected without `--resume`; finished stages are not retrained on resume. A lock
prevents two pipeline invocations from using the same SFT output simultaneously.

Use `--sft-output` and `--rl-output` for different experiment folders, or `--dry-run`
to print the planned commands without training or API calls. Pipeline orchestration
does not change either stage's hyperparameters and does not automatically evaluate.
