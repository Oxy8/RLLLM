"""Content-keyed original-model generations, shared across evaluation runs."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from pathlib import Path

from .judge import plain_text


def digest_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class BaselineCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, signature TEXT, digest TEXT)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS generations (key TEXT PRIMARY KEY, record TEXT)"
        )
        self.db.commit()

    def close(self):
        self.db.close()

    def file_digest(self, path: Path) -> str:
        def signature():
            stat = path.stat()
            return json.dumps(
                [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
            )

        before = signature()
        cached = self.db.execute(
            "SELECT signature, digest FROM files WHERE path=?", (str(path.resolve()),)
        ).fetchone()
        if cached and cached[0] == before:
            return cached[1]
        if path.stat().st_size > 100 * 1024**2:
            print(f"Fingerprinting baseline weights: {path.name}", flush=True)
        checksum = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024**2):
                checksum.update(chunk)
        if signature() != before:
            raise RuntimeError("Original model changed while fingerprinting; retry evaluation")
        value = checksum.hexdigest()
        self.db.execute(
            "INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (str(path.resolve()), before, value)
        )
        self.db.commit()
        return value

    def get(self, key: str, question: dict):
        row = self.db.execute("SELECT record FROM generations WHERE key=?", (key,)).fetchone()
        if row:
            try:
                result = json.loads(row[0])
                if (
                    result["prompt_id"] == question["prompt_id"]
                    and result["instruction"] == plain_text(question["prompt"])
                    and isinstance(result["response"], str)
                    and isinstance(result["generation_diagnostics"], dict)
                ):
                    return result
            except (ValueError, KeyError, TypeError):
                pass
        return None

    def put(self, key: str, record: dict):
        # Store generations only. Judgments depend on the competing trained answer.
        record = {
            name: record[name]
            for name in [
                "prompt_id",
                "instruction",
                "response",
                "generated_tokens",
                "generation_diagnostics",
            ]
        }
        self.db.execute(
            "INSERT OR REPLACE INTO generations VALUES (?, ?)",
            (key, json.dumps(record, ensure_ascii=False)),
        )
        self.db.commit()


def model_snapshot(settings, source: str | None) -> Path:
    if source is not None and settings.path(source).is_dir():
        return settings.path(source).resolve()
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            source or settings.training.model,
            revision="main" if source is not None else settings.training.revision,
            token=settings.api.hf_token or None,
            cache_dir=str(settings.path(".cache/huggingface")),
            allow_patterns=["*.json", "*.jinja", "*.safetensors"],
        )
    )
    if not list(path.glob("*.safetensors")):
        path = Path(
            snapshot_download(
                source or settings.training.model,
                revision=path.name,
                token=settings.api.hf_token or None,
                cache_dir=str(settings.path(".cache/huggingface")),
                allow_patterns=["pytorch_model*.bin"],
            )
        )
    return path


def generation_context(settings, snapshot: Path, tokenizer, cache: BaselineCache) -> dict:
    import torch
    import transformers

    from .evaluation import evaluation_generation_kwargs, generate_answers
    from .training import load_text_model, load_tokenizer

    weights = sorted([*snapshot.glob("*.safetensors"), *snapshot.glob("pytorch_model*.bin")])
    if not weights:
        raise ValueError(
            "Original model weights not found for caching; disable cache_baseline to bypass"
        )
    configuration = {}
    for name in [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ]:
        path = snapshot / name
        if path.is_file():
            value = json.loads(path.read_text())
            # These provenance fields do not affect inference and differ between copies.
            for field in ["_name_or_path", "_commit_hash", "transformers_version"]:
                value.pop(field, None)
            configuration[name] = value
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenization = backend.to_str() if backend is not None else tokenizer.get_vocab()
    cuda = torch.cuda.is_available()
    runtime = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "dtype": "bfloat16" if cuda and torch.cuda.is_bf16_supported() else "float32",
        "attention": "sdpa",
    }
    code_root = Path(__file__).parent
    return {
        "version": "baseline-generations-v2",
        "weights": {path.name: cache.file_digest(path) for path in weights},
        "configuration": configuration,
        "tokenizer": digest_json(
            {
                "backend": tokenization,
                "chat_template": getattr(tokenizer, "chat_template", None),
                "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
                "padding_side": getattr(tokenizer, "padding_side", None),
            }
        ),
        "generation": evaluation_generation_kwargs(settings),
        "thinking": settings.evaluation.enable_thinking,
        "answer_tokens": settings.evaluation.max_answer_tokens,
        "runtime": runtime,
        "implementation": {
            # Scoring/reporting changes do not change generated answers. Fingerprint
            # generation itself so switching traits can reuse original-model outputs.
            **{
                function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
                for function in [
                    generate_answers, evaluation_generation_kwargs, load_text_model, load_tokenizer
                ]
            },
            "answers.py": hashlib.sha256((code_root / "answers.py").read_bytes()).hexdigest(),
        },
    }


def baseline_answers(settings, data, source, generate, load_tokenizer):
    if not settings.evaluation.cache_baseline:
        return generate(settings, data, source), {"enabled": False, "hits": 0, "misses": len(data)}
    snapshot = model_snapshot(settings, source)
    tokenizer = load_tokenizer(settings, str(snapshot))
    path = settings.path(settings.evaluation.baseline_cache_path)
    cache = BaselineCache(path)
    try:
        context = generation_context(settings, snapshot, tokenizer, cache)
        context_key = digest_json(context)
        keys, records = [], []
        if len({row["prompt_id"] for row in data}) != len(data):
            raise ValueError("Baseline caching requires unique question IDs")
        for index, question in enumerate(data):
            key = digest_json(
                {
                    "context": context_key,
                    "prompt_id": question["prompt_id"],
                    "prompt": question["prompt"],
                    "seed": settings.training.seed + index
                    if settings.evaluation.temperature > 0
                    else None,
                }
            )
            keys.append(key)
            records.append(cache.get(key, question))
        missing = [index for index, record in enumerate(records) if record is None]
        statistics = {
            "enabled": True,
            "hits": len(records) - len(missing),
            "misses": len(missing),
            "path": str(path),
            "model_and_generation_fingerprint": context_key,
        }
        print(
            f"Original answer cache: {statistics['hits']}/{len(records)} reused; "
            f"{len(missing)} to generate",
            flush=True,
        )
        if missing:
            by_id = {data[index]["prompt_id"]: index for index in missing}
            if len(by_id) != len(missing):
                raise ValueError("Baseline caching requires unique question IDs")
            saved = set()

            def save(record):
                index = by_id[record["prompt_id"]]
                if record["instruction"] != plain_text(data[index]["prompt"]):
                    raise ValueError("Generated baseline instruction does not match the question")
                cache.put(keys[index], record)
                records[index] = record
                saved.add(record["prompt_id"])

            generated = generate(
                settings,
                [data[index] for index in missing],
                str(snapshot),
                seed_indices=missing,
                on_record=save,
            )
            for record in generated:
                if record["prompt_id"] not in saved:
                    save(record)
            if any(record is None for record in records):
                raise ValueError("Original model did not generate every missing answer")
        return records, statistics
    finally:
        cache.close()
