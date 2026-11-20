from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import random
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .config import Settings, require_api_key
from .data import instruction_text, normalized_prompt_id, write_json

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def load_source(settings: Settings, filename: str) -> tuple[list[dict], dict]:
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        settings.data.repo_id, filename, repo_type="dataset", revision=settings.data.revision,
        token=settings.api.hf_token or None,
        cache_dir=str(settings.path(".cache/huggingface")),
    ))
    raw = path.read_bytes()
    return json.loads(raw), {"file": filename, "sha256": hashlib.sha256(raw).hexdigest()}


def forbidden_tasks(settings: Settings) -> tuple[set[str], dict]:
    """Exclude whole RL/eval source splits as well as tasks saved by previous runs."""
    excluded = set()
    sources = []
    for filename in dict.fromkeys([settings.data.train_file, settings.data.eval_file]):
        rows, source = load_source(settings, filename)
        excluded.update(normalized_prompt_id(instruction_text(row)) for row in rows)
        sources.append(source)
    prepared = {
        *settings.path("data").rglob("train.jsonl"),
        *settings.path("data").rglob("eval.jsonl"),
        settings.path(settings.data.output_dir) / "train.jsonl",
        settings.path(settings.data.output_dir) / "eval.jsonl",
    }
    logs = {
        p for p in settings.path("runs").rglob("*.jsonl")
        if p.name in {"judge.jsonl", "comparison.jsonl"}
        or p.name.endswith((".judge.jsonl", ".generations.jsonl"))
    }
    files = 0
    for path in sorted(prepared | logs):
        if not path.is_file():
            continue
        files += 1
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("instruction")
                prompt = row.get("prompt")
                if text is None and isinstance(prompt, str):
                    text = prompt
                elif text is None and isinstance(prompt, list):
                    if len(prompt) == 1 and prompt[0].get("role") == "user":
                        text = prompt[0].get("content")
                if isinstance(text, str) and text.strip():
                    excluded.add(normalized_prompt_id(text))
    return excluded, {"source_splits": sources, "saved_data_and_log_files": files,
                      "distinct_excluded_tasks": len(excluded)}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    temporary.replace(path)


@contextmanager
def collection_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".collection.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another collection is already using this dataset") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def prepare_pool(settings: Settings, pool_size: int | None = None, tokenizer=None):
    """Reserve a stable pool. Existing reservations are never replaced or reordered."""
    config = settings.collection
    directory = settings.path(config.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pool.jsonl"
    pool = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    manifest_path = directory / "pool-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    identity = {"repo_id": settings.data.repo_id, "revision": settings.data.revision,
                "source_file": config.source_file, "seed": settings.training.seed}
    if manifest and any(manifest.get(k) != v for k, v in identity.items()):
        raise ValueError("Collection source or seed changed; use a separate collection output_dir")
    excluded, exclusions = forbidden_tasks(settings)
    normalized = set()
    for row in pool:
        text = row["instruction"]
        key = normalized_prompt_id(text)
        if key in normalized or row["prompt_id"] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("Invalid or duplicate collection reservation")
        if key in excluded:
            raise ValueError("Reserved collection question overlaps RL/evaluation data")
        normalized.add(key)
    target = max(pool_size or config.pool_size, len(pool))
    if len(pool) < target:
        rows, source = load_source(settings, config.source_file)
        if manifest and source["sha256"] != manifest.get("source_sha256"):
            raise ValueError(
                "Collection source content changed; refusing to change an existing pool"
            )
        if tokenizer is None:
            from .training import load_tokenizer

            tokenizer = load_tokenizer(settings)
        random.Random(settings.training.seed).shuffle(rows)
        for row in rows:
            text = instruction_text(row)
            key = normalized_prompt_id(text)
            if key in excluded or key in normalized:
                continue
            prompt = [{"role": "user", "content": text}]
            ids = tokenizer.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True, return_dict=False,
                enable_thinking=settings.training.enable_thinking,
            )
            if len(ids) > settings.data.max_prompt_tokens:
                continue
            pool.append({"prompt_id": hashlib.sha256(text.encode()).hexdigest(),
                         "instruction": text, "prompt": prompt})
            normalized.add(key)
            if len(pool) == target:
                break
        if len(pool) < target:
            raise ValueError(f"Only {len(pool)} disjoint questions available; need {target}")
        write_jsonl(path, pool)
        manifest.update(identity | {"source_sha256": source["sha256"],
                                   "created_at": manifest.get("created_at", now())})
    manifest.update({"pool_size": len(pool), "exclusions": exclusions,
                     "max_prompt_tokens": settings.data.max_prompt_tokens,
                     "tokenizer": settings.training.model, "checked_at": now(),
                     "pool_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    write_json(manifest_path, manifest)
    return pool, manifest


def read_examples(path: Path, pool: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    # Successful records are newline terminated and fsynced. Recover only an incomplete
    # final write after a crash, preserving its bytes separately for inspection.
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        end = raw.rfind(b"\n") + 1
        try:
            json.loads(raw[end:])
        except (ValueError, UnicodeDecodeError):
            backup = path.with_name(f"{path.name}.partial-{uuid4().hex[:8]}")
            backup.write_bytes(raw[end:])
            with path.open("r+b") as stream:
                stream.truncate(end)
            raw = raw[:end]
        else:
            with path.open("ab") as stream:
                stream.write(b"\n")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    by_id = {row["prompt_id"]: row for row in pool}
    seen = set()
    for row in rows:
        key = row["prompt_id"]
        if key not in by_id or key in seen:
            raise ValueError("Existing examples contain duplicate or unreserved questions")
        expected = [{"role": "user", "content": by_id[key]["instruction"]},
                    {"role": "assistant", "content": row["answer"]}]
        if row.get("messages") != expected or row.get("instruction") != by_id[key]["instruction"]:
            raise ValueError("Existing example does not match its reserved question")
        validate_answer({"answer": row["answer"]})
        seen.add(key)
    return rows


def validate_answer(result: dict) -> str:
    answer = result.get("answer") if isinstance(result, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Teacher returned an empty or invalid answer")
    if any(marker in answer.lower() for marker in ["<think>", "</think>", "<|im_start|>"]):
        raise ValueError("Teacher answer contains thinking or chat-control markup")
    return answer.strip()


def append_record(path: Path, row: dict) -> None:
    with path.open("ab") as stream:
        stream.write((json.dumps(row, ensure_ascii=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


class AnthropicTeacher:
    def __init__(self, settings: Settings, system: str):
        self.settings = settings
        self.system = system
        self.client = None

    async def __aenter__(self):
        from anthropic import AsyncAnthropic

        self.client = AsyncAnthropic(
            api_key=self.settings.api.anthropic_api_key,
            timeout=self.settings.judge.timeout_seconds,
            max_retries=self.settings.judge.max_retries,
        )
        return self

    async def __aexit__(self, *args):
        await self.client.close()

    async def __call__(self, instruction: str) -> dict:
        s = self.settings
        try:
            response = await self.client.messages.create(
                model=s.judge.model, max_tokens=s.collection.max_tokens,
                **({"thinking": {"type": "disabled"}}
                   if s.judge.model.startswith("claude-haiku-5-5") else {}),
                system=self.system, messages=[{"role": "user", "content": instruction}],
                output_config={"format": {"type": "json_schema", "schema": ANSWER_SCHEMA}},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Anthropic teacher request failed ({type(exc).__name__}, "
                f"HTTP {getattr(exc, 'status_code', None)}). Completed examples remain saved."
            ) from None
        usage = {"input_tokens": response.usage.input_tokens,
                 "output_tokens": response.usage.output_tokens}
        text = "".join(b.text for b in response.content if b.type == "text")
        return {"text": text, "stop_reason": response.stop_reason, "usage": usage,
                "response_id": response.id}


async def generate_examples(settings, pending, run_dir, dataset_path, system, request, stats):
    semaphore = asyncio.Semaphore(settings.collection.concurrency)
    stopped = False
    prompt_hash = hashlib.sha256(system.encode()).hexdigest()

    async def generate(row):
        nonlocal stopped
        async with semaphore:
            if stopped:
                return
            for attempt in range(settings.judge.max_retries + 1):
                stats["api_requests"] += 1
                try:
                    result = await request(row["instruction"])
                    usage = result.get("usage", {})
                    for name in ("input_tokens", "output_tokens"):
                        stats[name] += usage.get(name, 0)
                    if result.get("stop_reason") != "end_turn":
                        raise ValueError(
                            f"Teacher stopped with {result.get('stop_reason')}; "
                            "incomplete/refused answers are not added"
                        )
                    try:
                        decoded = json.loads(result["text"])
                    except (KeyError, json.JSONDecodeError):
                        raise ValueError("Teacher returned invalid answer JSON") from None
                    answer = validate_answer(decoded)
                    record = {
                        "prompt_id": row["prompt_id"], "instruction": row["instruction"],
                        "answer": answer,
                        "messages": [{"role": "user", "content": row["instruction"]},
                                     {"role": "assistant", "content": answer}],
                        "teacher_model": settings.judge.model,
                        "generation_prompt_sha256": prompt_hash, "generated_at": now(),
                        "collection_run": str(run_dir), "response_id": result.get("response_id"),
                        "usage": usage,
                    }
                    append_record(dataset_path, record)
                    stats["added"] += 1
                    print(f"Collected {stats['added']}/{stats['requested']} new examples "
                          f"(dataset total {stats['existing'] + stats['added']})", flush=True)
                    return
                except (ValueError, RuntimeError) as exc:
                    append_record(run_dir / "errors.jsonl", {
                        "prompt_id": row["prompt_id"], "attempt": attempt + 1,
                        "error": str(exc), "at": now(),
                    })
                    if isinstance(exc, RuntimeError) or attempt == settings.judge.max_retries:
                        stopped = True
                        raise

    results = await asyncio.gather(*(generate(row) for row in pending), return_exceptions=True)
    failures = [r for r in results if isinstance(r, BaseException)]
    if failures:
        raise RuntimeError(str(failures[0]))


def collect(settings: Settings, count=None, pool_size=None, prepare_only=False,
            *, tokenizer=None, request=None) -> dict:
    config = settings.collection
    count = config.examples_per_run if count is None else count
    if type(count) is not int or count <= 0:
        raise ValueError("--count must be a positive number of additional examples")
    if pool_size is not None and (type(pool_size) is not int or pool_size <= 0):
        raise ValueError("--pool-size must be positive")
    if not settings.path(config.prompt_path).is_file():
        raise ValueError("The configured collection generation prompt does not exist")
    if not prepare_only and request is None:
        require_api_key(settings)
    directory = settings.path(config.output_dir)
    with collection_lock(directory):
        pool, manifest = prepare_pool(settings, pool_size, tokenizer)
        dataset = directory / "examples.jsonl"
        existing = read_examples(dataset, pool)
        seen = {r["prompt_id"] for r in existing}
        pending = [r for r in pool if r["prompt_id"] not in seen]
        summary = {"pool_size": len(pool), "existing": len(existing),
                   "available": len(pending), "dataset": str(dataset),
                   "exclusions": manifest["exclusions"]}
        if prepare_only:
            print(json.dumps(summary, indent=2))
            return summary
        if len(pending) < count:
            raise ValueError(
                f"Only {len(pending)} unused reserved questions remain; request fewer examples "
                "or extend with --pool-size. Existing examples are preserved."
            )
        pending = pending[:count]
        run_dir = settings.path("runs/childishness-collection") / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        )
        run_dir.mkdir(parents=True)
        system = settings.path(config.prompt_path).read_text()
        (run_dir / "generation-prompt.txt").write_text(system)
        write_json(run_dir / "run.json", {
            "created_at": now(), "teacher_model": settings.judge.model,
            "collection_settings": settings.public_dict()["collection"],
            "pool_manifest": manifest, "requested_additional_examples": count,
            "existing_examples": len(existing),
            "selected_prompt_ids": [r["prompt_id"] for r in pending],
        })
        stats = {"existing": len(existing), "requested": count, "added": 0,
                 "api_requests": 0, "input_tokens": 0, "output_tokens": 0,
                 "started_at": now(), "status": "interrupted"}

        async def run():
            if request is not None:
                await generate_examples(settings, pending, run_dir, dataset, system, request, stats)
            else:
                async with AnthropicTeacher(settings, system) as teacher:
                    await generate_examples(
                        settings, pending, run_dir, dataset, system, teacher, stats
                    )

        print(f"Adding {count} new examples to {len(existing)} saved examples; "
              f"reserved pool: {len(pool)}. Model: {settings.judge.model}", flush=True)
        try:
            asyncio.run(run())
            stats["status"] = "complete"
        except (ValueError, RuntimeError, OSError):
            stats["status"] = "failed"
            raise
        finally:
            stats["finished_at"] = now()
            stats["dataset_total"] = len(existing) + stats["added"]
            write_json(run_dir / "usage.json", stats)
        print(json.dumps(stats | {"dataset": str(dataset), "run": str(run_dir)}, indent=2))
        return stats
