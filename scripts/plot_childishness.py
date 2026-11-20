"""Plot recorded judge scores without loading models or calling the judge API."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def reasoning_token_history(run, manifest, records):
    """Recover this model's lineage, validating legacy row-to-step alignment."""
    training = manifest["settings"]["training"]
    answers = training["micro_batch_size"] * training["gradient_accumulation_steps"]
    groups_per_step = answers // training["num_generations"]
    sources = []
    resume = manifest.get("resume")
    if resume and Path(resume).name.startswith("checkpoint-"):
        checkpoint = Path(resume)
        if not checkpoint.is_absolute():
            checkpoint = Path(manifest["settings"]["root"]) / checkpoint
        if checkpoint.parent.resolve() != run.resolve():
            sources.append((checkpoint.parent, 1, int(checkpoint.name.split("-")[-1])))
    start = sources[-1][2] + 1 if sources else 1
    sources.append((run, start, records[-1]["step"]))
    by_step = {}
    for source, first_step, last_step in sources:
        for index, line in enumerate((source / "judge.jsonl").open()):
            row = json.loads(line)
            step = row.get("training_step")
            if step is None:
                step = first_step + index // groups_per_step
            if step > last_step:
                break
            if step >= first_step:
                by_step.setdefault(step, []).append(row)
    history = []
    for record in records:
        step = record["step"]
        rows = by_step.get(step, [])
        diagnostics = [info for row in rows for info in row["generation_diagnostics"]]
        scores = [score["childishness"] for row in rows for score in row["judgments"]]
        if len(rows) != groups_per_step or len(diagnostics) != answers or len(scores) != answers:
            raise ValueError(f"Incomplete token history at step {step}")
        generated = statistics.mean(info["generated_tokens"] for info in diagnostics)
        if (abs(statistics.mean(scores)
                - record.get("judge/childishness", record["reward"])) > 1e-6
                or abs(generated - record["completions/mean_length"]) > 1e-5):
            raise ValueError(f"Legacy token history does not align with saved step {step}")
        differences = [info["generated_tokens"] - info["final_answer_tokens"]
                       for info in diagnostics]
        history.append({
            "step": step,
            "estimated_reasoning_tokens_per_answer": statistics.mean(
                max(0, value) for value in differences
            ),
            "mean_generated_tokens": generated,
            "mean_uncapped_final_text_tokens": statistics.mean(
                info["final_answer_tokens"] for info in diagnostics
            ),
            "mean_generated_minus_final_text_tokens": statistics.mean(differences),
            "unfinished_thinking_fraction": statistics.mean(
                info["unfinished_thinking"] for info in diagnostics
            ),
            "negative_differences_clamped": sum(value < 0 for value in differences),
        })
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run containing trainer_state.json")
    parser.add_argument("--window", type=int, default=32, help="Trailing smoothing window")
    parser.add_argument("--output", type=Path, help="Plot folder; default: RUN/plots")
    parser.add_argument("--metric", choices=["childishness", "task_quality", "combined_reward",
                                            "reasoning_tokens"],
                        default="childishness")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    state = json.loads((args.run / "trainer_state.json").read_text())
    manifest_path = args.run / "resume.json"
    if not manifest_path.exists():
        manifest_path = args.run / "run.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["settings"]["judge"]["score_name"] != "childishness":
        parser.error("The selected run did not use the childishness judge")
    records = [row for row in state["log_history"] if "reward" in row]
    steps = [row["step"] for row in records]
    if steps != list(range(1, state["global_step"] + 1)):
        parser.error("Expected one recorded reward for each completed optimizer update")
    quality_mode = manifest["settings"]["judge"].get("reward_mode") in {
        "quality_gated", "quality_softplus",
    }
    if args.metric in {"task_quality", "combined_reward"} and not quality_mode:
        parser.error("Task-quality and combined-reward plots require dual-score scoring")
    score_key = "reward" if args.metric == "combined_reward" else (
        "judge/task_quality" if args.metric == "task_quality" else "judge/childishness"
    )
    scores = [row.get(score_key, row["reward"]) for row in records]
    if not all(math.isfinite(score) for score in scores):
        parser.error("Recorded judge means must be finite")
    if args.metric != "combined_reward" and not all(0 <= score <= 1 for score in scores):
        parser.error("Recorded trait and task-quality means must be in [0, 1]")
    smoothed = [statistics.mean(scores[max(0, i + 1 - args.window):i + 1])
                for i in range(len(scores))]
    # Check explicit per-answer records where they exist (newer logging format).
    groups = {}
    for line in (args.run / "judge.jsonl").open():
        row = json.loads(line)
        if row.get("training_step") is not None:
            field = {"task_quality": "task_quality", "combined_reward": "reward"}.get(
                args.metric, "childishness"
            )
            groups.setdefault(row["training_step"], []).extend(
                judgment[field] for judgment in row["judgments"]
            )
    for step, values in groups.items():
        if abs(scores[step - 1] - statistics.mean(values)) > 1e-6:
            parser.error(f"Judge log disagrees with the training mean at step {step}")
    reasoning = args.metric == "reasoning_tokens"
    token_history = reasoning_token_history(args.run, manifest, records) if reasoning else None
    if reasoning:
        scores = [row["estimated_reasoning_tokens_per_answer"] for row in token_history]
        smoothed = [statistics.mean(scores[max(0, i + 1 - args.window):i + 1])
                    for i in range(len(scores))]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#354653", "text.color": "#1e303d",
        "xtick.color": "#536370", "ytick.color": "#536370",
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(12, 6.6))
    fig.subplots_adjust(left=.085, right=.975, bottom=.21, top=.79)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e7edf0", linewidth=.8)
    color = "#b36922" if reasoning else "#096b68"
    ax.plot(steps, scores, color="#c9ae91" if reasoning else "#93acbc",
            linewidth=.65, alpha=.48,
            label="Estimated reasoning tokens per answer" if reasoning else "Mean score per update",
            zorder=2)
    ax.plot(steps, smoothed, color=color, linewidth=2.5,
            label=f"{args.window}-update trailing mean", zorder=4)

    resume = manifest.get("resume")
    if resume and Path(resume).name.startswith("checkpoint-"):
        resumed_step = int(Path(resume).name.split("-")[-1])
        ax.axvspan(resumed_step, steps[-1], color="#dcefe9", alpha=.42, zorder=0)
        ax.axvline(resumed_step, color="#7b8b94", linestyle=(0, (4, 4)), linewidth=1)
        ax.text(resumed_step - 15, .96, f"Resumed from {resumed_step}", rotation=90,
                va="top", ha="right", color="#536370", fontsize=9,
                transform=ax.get_xaxis_transform())
    checkpoint = args.run / "checkpoint-1280"
    if checkpoint.is_dir() and 1280 < steps[-1]:
        ax.axvline(1280, color="#ba7d33", linestyle=(0, (3, 4)), linewidth=1)
        ax.text(1280 + 14, .96, "Checkpoint 1280", rotation=90,
                va="top", ha="left", color="#906025", fontsize=9,
                transform=ax.get_xaxis_transform())

    ax.set_xlim(0, steps[-1])
    if reasoning:
        ax.set_ylim(0, math.ceil(max(scores) / 500) * 500)
    elif args.metric == "combined_reward":
        ax.margins(y=.08)
    else:
        ax.set_ylim(0, 1)
    ax.set_xticks([0, 256, 512, 768, 1024, 1280, 1536] if steps[-1] == 1536
                  else ax.get_xticks())
    ax.yaxis.set_major_locator(
        MaxNLocator(6, integer=reasoning)
        if reasoning or args.metric == "combined_reward" else MultipleLocator(.2)
    )
    ax.set_xlabel("Optimizer update (global step)", labelpad=11)
    metric_label = {"childishness": "childishness score", "task_quality": "task quality",
                    "combined_reward": "combined reward"}.get(args.metric)
    ax.set_ylabel("Estimated reasoning tokens per answer" if reasoning
                  else f"Mean {metric_label}" + (
                      "" if args.metric == "combined_reward" else " (0–1)"
                  ), labelpad=10)
    ax.spines["left"].set_color("#ced8de")
    ax.spines["bottom"].set_color("#ced8de")
    legend = ax.legend(loc="upper left", frameon=False, fontsize=10, handlelength=3)
    for line in legend.get_lines():
        line.set_alpha(1)
    fig.text(.085, .93, "Reasoning token usage during GRPO training" if reasoning
             else f"{metric_label.capitalize()} during GRPO training", fontsize=21,
             fontweight="bold")
    fig.text(.085, .878, "LFM3-1.7B · Local generation · Estimated from recorded token counts"
             if reasoning else "LFM3-1.7B · Haiku judge · Completed training history", fontsize=11,
             color="#536370")
    training = manifest["settings"]["training"]
    answers = training["micro_batch_size"] * training["gradient_accumulation_steps"]
    negative = sum(row["negative_differences_clamped"] for row in token_history) if reasoning else 0
    fig.text(.085, .085,
             "Estimate = max(0, generated tokens − uncapped final-text tokens). "
             "Includes delimiters and retokenization differences." if reasoning else
             f"Each update averages {answers} answers. Smoothed line uses the last "
             f"{args.window} updates (fewer at the start).",
             fontsize=9, color="#536370")
    fig.text(.085, .045,
             f"Mean of {answers} answers per update; {args.window}-update trailing average. "
             f"{negative} of {answers * len(steps):,} estimates clamped to zero. "
             "Interrupted branches excluded." if reasoning else
             "Training scores; not held-out evaluation. Earlier steps come from restored "
             "checkpoint history; interrupted branches are excluded.",
             fontsize=9, color="#536370")
    output = args.output or args.run / "plots"
    output.mkdir(parents=True, exist_ok=True)
    stem = output / ("reasoning-tokens-by-step" if reasoning
                     else args.metric.replace("_", "-") + "-by-step")
    for extension in ["png", "svg", "pdf"]:
        fig.savefig(stem.with_suffix(f".{extension}"), dpi=200, facecolor="white")
    plt.close(fig)
    with stem.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        if reasoning:
            fields = list(token_history[0]) + [f"trailing_{args.window}_update_mean"]
            writer.writerow(fields)
            writer.writerows([*row.values(), smooth]
                             for row, smooth in zip(token_history, smoothed, strict=True))
        else:
            writer.writerow(["step", f"mean_{args.metric}",
                             f"trailing_{args.window}_update_mean"])
            writer.writerows(zip(steps, scores, smoothed, strict=True))
    print(json.dumps({"steps": len(steps), "verified_against_judge": len(groups),
                      "smoothing_window": args.window,
                      "negative_token_differences_clamped": negative,
                      "artifacts": [str(stem.with_suffix(f".{ext}"))
                                    for ext in ["png", "svg", "pdf", "csv"]]}, indent=2))


if __name__ == "__main__":
    main()
