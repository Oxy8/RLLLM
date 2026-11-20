from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .config import Settings


def instruction_text(row: dict) -> str:
    instruction = row.get("instruction", "")
    context = row.get("input", "")
    if not isinstance(instruction, str) or not instruction.strip() or not isinstance(context, str):
        raise ValueError("Invalid AlpacaFarm instruction/input")
    return instruction.strip() + ("\n\n" + context.strip() if context.strip() else "")


def normalized_prompt_id(text: str) -> str:
    """Catch the same task with different capitalization or whitespace across splits."""
    return hashlib.sha256(" ".join(text.casefold().split()).encode()).hexdigest()


def reserved_collection_instructions(settings: Settings) -> list[str]:
    path = settings.path(settings.collection.output_dir) / "pool.jsonl"
    if not path.exists():
        return []
    texts = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        text = row["instruction"]
        if row["prompt_id"] != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("Collection reservation contains an invalid prompt ID")
        texts.append(text)
    return texts


def select_rows(
    rows, tokenizer, limit, max_tokens, seed, excluded=None, enable_thinking=True,
    excluded_normalized=None,
):
    """Discard all answers, deduplicate whole tasks, and filter instead of truncating."""
    seen = set(excluded or ())
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    selected = []
    stats = {"duplicates": 0, "too_long": 0}
    for row in rows:
        text = instruction_text(row)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen or normalized_prompt_id(text) in (excluded_normalized or ()):
            stats["duplicates"] += 1
            continue
        seen.add(digest)
        messages = [{"role": "user", "content": text}]
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_dict=False,
        )
        if len(ids) > max_tokens:
            stats["too_long"] += 1
            continue
        selected.append({"prompt": messages, "prompt_id": digest})
        if len(selected) == limit:
            break
    if not selected:
        raise ValueError("No instructions remain after filtering")
    return selected, stats


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(settings: Settings, tokenizer=None) -> dict:
    from huggingface_hub import hf_hub_download

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            settings.training.model,
            revision=settings.training.revision,
            token=settings.api.hf_token or None,
            cache_dir=str(settings.path(".cache/huggingface")),
        )
    config = settings.data
    selected = {}
    report = {}
    reserved = reserved_collection_instructions(settings)
    reserved_ids = {hashlib.sha256(text.encode()).hexdigest() for text in reserved}
    reserved_normalized = {normalized_prompt_id(text) for text in reserved}
    for split, filename, limit in [
        ("eval", config.eval_file, config.eval_samples),
        ("train", config.train_file, config.train_samples),
    ]:
        path = Path(
            hf_hub_download(
                config.repo_id,
                filename,
                repo_type="dataset",
                revision=config.revision,
                token=settings.api.hf_token or None,
                cache_dir=str(settings.path(".cache/huggingface")),
            )
        )
        raw = path.read_bytes()
        rows, stats = select_rows(
            json.loads(raw),
            tokenizer,
            limit,
            config.max_prompt_tokens,
            settings.training.seed,
            excluded=reserved_ids | {r["prompt_id"] for r in selected.get("eval", [])},
            excluded_normalized=reserved_normalized,
            enable_thinking=settings.training.enable_thinking,
        )
        selected[split] = rows
        target = settings.path(config.output_dir) / f"{split}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        temporary.replace(target)
        report[split] = {
            "source_file": filename,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "selected": len(rows),
            "requested": limit,
            **stats,
        }
    report.update(
        {
            "repo_id": config.repo_id,
            "revision": config.revision,
            "tokenizer": settings.training.model,
            "tokenizer_revision": settings.training.revision,
            "max_prompt_tokens": config.max_prompt_tokens,
            "enable_thinking": settings.training.enable_thinking,
            "seed": settings.training.seed,
        }
    )
    if reserved:
        reservation = settings.path(settings.collection.output_dir) / "pool.jsonl"
        report["collection_reservation"] = {
            "path": str(reservation),
            "questions": len(reserved),
            "sha256": hashlib.sha256(reservation.read_bytes()).hexdigest(),
        }
    write_json(settings.path(config.output_dir) / "manifest.json", report)
    return report


def load_prepared(settings: Settings, split: str):
    from datasets import Dataset

    path = settings.path(settings.data.output_dir) / f"{split}.jsonl"
    if not path.is_file():
        raise ValueError("Prepared data missing. Run: python -m rlllmf prepare")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    reserved = {normalized_prompt_id(text) for text in reserved_collection_instructions(settings)}
    for row in rows:
        if len(row["prompt"]) != 1 or row["prompt"][0]["role"] != "user":
            raise ValueError("Prepared dataset must contain standalone user instructions")
        if normalized_prompt_id(row["prompt"][0]["content"]) in reserved:
            raise ValueError("Prepared data overlaps the SFT collection pool. Run prepare again.")
    return Dataset.from_list(rows)
