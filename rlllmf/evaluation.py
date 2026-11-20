from __future__ import annotations

import asyncio
import gc
import json
from pathlib import Path

from .answers import (
    decode_final_answer,
    generation_diagnostics,
    limit_answer_tokens,
    thinking_is_prefilled,
)
from .baseline_cache import baseline_answers
from .config import EvaluationSettings, Settings, require_api_key
from .data import load_prepared, write_json
from .judge import OptimismJudge, plain_text
from .training import load_text_model, load_tokenizer, require_gpu


def evaluation_generation_kwargs(settings: Settings) -> dict:
    evaluation = settings.evaluation
    sampling = evaluation.temperature > 0
    return {
        "max_new_tokens": evaluation.max_completion_length,
        "do_sample": sampling,
        "temperature": evaluation.temperature if sampling else None,
        "top_p": evaluation.top_p if sampling else None,
        "top_k": 0 if sampling else None,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "use_cache": True,
    }


def generate_answers(
    settings: Settings,
    data,
    model_path: str | None,
    *,
    seed_indices=None,
    on_record=None,
) -> list[dict]:
    """Generate with one model, releasing its GPU weights before returning."""
    import torch
    from transformers import set_seed

    tokenizer = load_tokenizer(settings, model_path)
    prefilled = thinking_is_prefilled(tokenizer, settings.evaluation.enable_thinking)
    model = load_text_model(settings, model_path).eval()
    records = []
    try:
        for i, row in enumerate(data):
            # Per-question seeds and explicit sampling settings are shared by both models.
            set_seed(settings.training.seed + (seed_indices[i] if seed_indices is not None else i))
            encoded = tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=settings.evaluation.enable_thinking,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                ids = model.generate(
                    **encoded,
                    **evaluation_generation_kwargs(settings),
                    pad_token_id=tokenizer.pad_token_id,
                )[0, encoded["input_ids"].shape[-1] :]
            response = limit_answer_tokens(
                tokenizer,
                decode_final_answer(tokenizer, ids, thinking_prefilled=prefilled),
                settings.evaluation.max_answer_tokens,
            )
            records.append(
                {
                    "prompt_id": row["prompt_id"],
                    "instruction": plain_text(row["prompt"]),
                    "response": response,
                    "generated_tokens": len(ids),
                    "generation_diagnostics": generation_diagnostics(
                        tokenizer,
                        ids,
                        settings.evaluation.max_completion_length,
                        thinking_prefilled=prefilled,
                        max_answer_tokens=settings.evaluation.max_answer_tokens,
                    ),
                }
            )
            if on_record is not None:
                on_record(records[-1])
            print(f"Generated {i + 1}/{len(data)}", flush=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def selected_data(settings: Settings, limit: int | None):
    data = load_prepared(settings, "eval")
    if limit is not None:
        if limit <= 0:
            raise ValueError("Evaluation limit must be positive")
        data = data.select(range(min(limit, len(data))))
    if not len(data):
        raise ValueError("No held-out questions are available")
    return data


def write_jsonl(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def record_metrics(records: list[dict], score_name: str = "optimism") -> dict:
    size = len(records)
    metrics = {
        f"mean_{score_name}": sum(r["judgment"][score_name] for r in records) / size,
        "empty_answer_rate": sum(not r["response"].strip() for r in records) / size,
        "generation_limit_rate": sum(
            r.get("generation_diagnostics", {}).get("hit_token_limit", False) for r in records
        )
        / size,
        "judge_answer_clipping_rate": sum(
            r.get("generation_diagnostics", {}).get("answer_truncated_for_judge", False)
            for r in records
        )
        / size,
    }
    if all("task_quality" in r["judgment"] for r in records):
        metrics["mean_task_quality"] = sum(r["judgment"]["task_quality"] for r in records) / size
        metrics["mean_combined_reward"] = sum(r["judgment"]["reward"] for r in records) / size
    if all("factual_consistency" in r["judgment"] for r in records):
        for name, category in [("contradiction_rate", "contradicted"),
                               ("unsupported_or_unclear_rate", "unclear")]:
            metrics[name] = sum(r["judgment"]["factual_consistency"] == category
                                for r in records) / size
        metrics["relevance_rate"] = sum(r["judgment"]["relevant"] for r in records) / size
    return metrics


def evaluate(settings: Settings, model_path: str | None, output: str, limit: int | None = None):
    require_api_key(settings)
    require_gpu()
    data = selected_data(settings, limit)
    target = settings.path(output)
    if target.exists():
        raise ValueError("Evaluation output already exists; choose a new --output filename")
    target.parent.mkdir(parents=True, exist_ok=True)
    if model_path is None:
        records, cache_info = baseline_answers(
            settings, data, None, generate_answers, load_tokenizer
        )
    else:
        records, cache_info = generate_answers(settings, data, model_path), None
    # Persist generations before API scoring so a request failure does not lose outputs.
    generations = target.with_suffix(".generations.jsonl")
    write_jsonl(generations, records)
    tokenizer = load_tokenizer(settings, model_path)
    judge = OptimismJudge(
        settings,
        target.with_suffix(".judge.jsonl"),
        tokenizer=tokenizer,
        max_answer_tokens=settings.evaluation.max_answer_tokens,
    )
    try:
        # Score independently against fixed anchors, not a forced winner/loser ranking.
        results = asyncio.run(
            judge.score_groups(
                [(row["instruction"], [row["response"]]) for row in records],
                diagnostics=[[row["generation_diagnostics"]] for row in records],
            )
        )
        for row, result in zip(records, results, strict=True):
            row["judgment"] = result[0]
            row["score_name"] = settings.judge.score_name
        write_jsonl(target, records)
        summary = {
            "model": model_path or settings.training.model,
            "examples": len(records),
            "score_name": settings.judge.score_name,
            "reward_mode": settings.judge.reward_mode,
            **record_metrics(records, settings.judge.score_name),
            "settings": settings.public_dict(),
            "judge_usage": judge.statistics(),
            "sampling_repetition_penalty": 1.0,
            "decoding": evaluation_generation_kwargs(settings),
            "baseline_cache": cache_info,
        }
        write_json(target.with_suffix(".summary.json"), summary)
        print(json.dumps(summary, indent=2))
    finally:
        judge.close()


def comparison_models(settings: Settings, trained: str, baseline: str | None) -> tuple[str, str]:
    trained_path = settings.path(trained)
    trained_source = str(trained_path.resolve()) if trained_path.is_dir() else trained
    if baseline:
        path = settings.path(baseline)
        return trained_source, str(path.resolve()) if path.is_dir() else baseline
    if not (trained_path / "config.json").is_file():
        raise ValueError("Trained checkpoint directory not found; check --model")
    initial = trained_path.parent / "initial_model"
    if not (initial / "config.json").is_file():
        raise ValueError("Original snapshot not found beside checkpoint; provide --baseline MODEL")
    return trained_source, str(initial.resolve())


def align_records(baseline: list[dict], trained: list[dict]) -> list[tuple[dict, dict]]:
    left = {row["prompt_id"]: row for row in baseline}
    right = {row["prompt_id"]: row for row in trained}
    if len(left) != len(baseline) or len(right) != len(trained) or left.keys() != right.keys():
        raise ValueError("Comparison requires matching, unique held-out question IDs")
    pairs = [(row, right[row["prompt_id"]]) for row in baseline]
    if any(before["instruction"] != after["instruction"] for before, after in pairs):
        raise ValueError("Comparison instructions differ for the same question ID")
    return pairs


def compare(
    settings: Settings,
    trained: str,
    output: str,
    baseline: str | None = None,
    limit: int | None = None,
    resume: bool = False,
):
    target = settings.path(output)
    if not resume and target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Comparison output already exists; choose a new --output directory")
    if resume and (target / "comparison.summary.json").exists():
        raise ValueError("Comparison is already complete; open comparison.html")
    require_api_key(settings)
    if resume:
        required = [
            "comparison-run.json", "baseline.generations.jsonl", "trained.generations.jsonl"
        ]
        if not all((target / name).is_file() for name in required):
            raise ValueError("Resume requires metadata and both saved generation files")
        metadata = json.loads((target / "comparison-run.json").read_text())
        trained_source = settings.path(trained)
        trained_source = str(trained_source.resolve()) if trained_source.is_dir() else trained
        if trained_source != metadata["trained_model"] or baseline is not None or limit is not None:
            raise ValueError("Resume must use the saved --model without --baseline or --limit")
        trained, baseline = metadata["trained_model"], metadata["baseline_model"]
        before, after = [
            [json.loads(line) for line in (target / name).read_text().splitlines() if line.strip()]
            for name in required[1:]
        ]
        if not before or len(before) != metadata["examples"] or len(after) != len(before):
            raise ValueError("Saved generations are incomplete; cannot resume judging")
        # Answer text was already generated and capped with these settings. A later
        # config edit must not silently relabel or shorten the saved experiment.
        settings.evaluation = EvaluationSettings(**metadata["settings"]["evaluation"])
        print(
            f"Reusing {len(before)} original and {len(after)} trained saved answers "
            f"(evaluation temperature {settings.evaluation.temperature}); no GPU generation.",
            flush=True,
        )
    else:
        trained, baseline = comparison_models(settings, trained, baseline)
        require_gpu()
        data = selected_data(settings, limit)
        target.mkdir(parents=True, exist_ok=True)
        metadata = {
            "baseline_model": baseline,
            "trained_model": trained,
            "examples": len(data),
            "settings": settings.public_dict(),
            "sampling_repetition_penalty": 1.0,
            "method": (
                f"paired absolute {settings.judge.score_name} scores; "
                "anonymized shuffled candidates"
            ),
        }
        write_json(target / "comparison-run.json", metadata)
        print(f"Generating original model answers: {baseline}", flush=True)
        before, cache_info = baseline_answers(
            settings, data, baseline, generate_answers, load_tokenizer
        )
        metadata["baseline_cache"] = cache_info
        metadata["decoding"] = evaluation_generation_kwargs(settings)
        write_json(target / "comparison-run.json", metadata)
        write_jsonl(target / "baseline.generations.jsonl", before)
        print(f"Generating trained model answers: {trained}", flush=True)
        after = generate_answers(settings, data, trained)
        write_jsonl(target / "trained.generations.jsonl", after)
    score_name = settings.judge.score_name
    metadata["score_name"] = score_name
    metadata["method"] = f"paired absolute {score_name} scores; anonymized shuffled candidates"
    pairs = align_records(before, after)
    tokenizer = load_tokenizer(settings, trained)
    # Use the judge's local tokenizer for the final transport cap on both sides,
    # even when an explicit baseline uses a different tokenizer.
    for row in before + after:
        row["response"] = limit_answer_tokens(
            tokenizer, row["response"], settings.evaluation.max_answer_tokens
        )
    judge = OptimismJudge(
        settings,
        target / "comparison.judge.jsonl",
        tokenizer=tokenizer,
        max_answer_tokens=settings.evaluation.max_answer_tokens,
    )
    try:
        print(f"Judging {len(pairs)} answer pairs…", flush=True)
        judgments = asyncio.run(
            judge.score_groups(
                [(a["instruction"], [a["response"], b["response"]]) for a, b in pairs],
                diagnostics=[
                    [a["generation_diagnostics"], b["generation_diagnostics"]] for a, b in pairs
                ],
            )
        )
        rows = []
        for (a, b), scores in zip(pairs, judgments, strict=True):
            a["judgment"], b["judgment"] = scores
            delta = float(scores[1][score_name]) - float(scores[0][score_name])
            rows.append(
                {
                    "prompt_id": a["prompt_id"],
                    "instruction": a["instruction"],
                    "responses": [a["response"], b["response"]],
                    "judgments": scores,
                    "generation_diagnostics": [
                        a["generation_diagnostics"],
                        b["generation_diagnostics"],
                    ],
                    "judge_model": settings.judge.model,
                    "score_name": score_name,
                    "comparison": {
                        "baseline_model": baseline,
                        "trained_model": trained,
                        "score_name": score_name,
                        f"{score_name}_delta": delta,
                        "winner": "trained" if delta > 0 else "baseline" if delta < 0 else "tie",
                        **({"task_quality_delta": scores[1]["task_quality"]
                            - scores[0]["task_quality"],
                            "combined_reward_delta": scores[1]["reward"] - scores[0]["reward"]}
                           if settings.judge.uses_task_quality else {}),
                    },
                }
            )
        write_jsonl(target / "comparison.jsonl", rows)
        metrics_before = record_metrics(before, score_name)
        metrics_after = record_metrics(after, score_name)
        counts = {
            name: sum(row["comparison"]["winner"] == name for row in rows)
            for name in ["trained", "baseline", "tie"]
        }
        summary = {
            **metadata,
            "judge_settings": settings.public_dict()["judge"],
            "baseline": metrics_before,
            "trained": metrics_after,
            "delta_trained_minus_baseline": {
                key: metrics_after[key] - metrics_before[key] for key in metrics_before
            },
            f"{score_name}_outcomes": {
                "counts": counts,
                "rates": {key: value / len(rows) for key, value in counts.items()},
            },
            "judge_usage": judge.statistics(),
        }
        if settings.judge.uses_task_quality:
            for name, key in [("task_quality", "task_quality_delta"),
                              ("combined_reward", "combined_reward_delta")]:
                outcomes = {label: sum(
                    ("trained" if row["comparison"][key] > 0 else
                     "baseline" if row["comparison"][key] < 0 else "tie") == label for row in rows
                ) for label in ["trained", "baseline", "tie"]}
                summary[f"{name}_outcomes"] = {
                    "counts": outcomes,
                    "rates": {label: value / len(rows) for label, value in outcomes.items()},
                }
        write_json(target / "comparison.summary.json", summary)
        from .viewer import export_viewer

        report = export_viewer(target / "comparison.jsonl")
        print(json.dumps(summary, indent=2))
        print(f"Comparison report: {report}")
    finally:
        write_json(target / "judge-usage.json", judge.statistics())
        judge.close()
