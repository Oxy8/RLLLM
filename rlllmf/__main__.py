from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version

from .config import load_settings


def main():
    parser = argparse.ArgumentParser(description="LFM3 + AlpacaFarm + Anthropic trait-scoring GRPO")
    parser.add_argument(
        "--config", default=".config", help="TOML config file; paths are relative to it"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "check", help="Check config, dependencies and GPU; no model downloads or API calls"
    )
    sub.add_parser("prepare", help="Download/tokenize standalone instructions; no paid API calls")
    collection = sub.add_parser(
        "collect", help="Append new Haiku childlike answers to the separate SFT dataset"
    )
    collection.add_argument("--count", type=int, help="Additional examples (default: 100)")
    collection.add_argument("--pool-size", type=int, help="Extend the reserved question pool")
    collection.add_argument(
        "--prepare-only", action="store_true", help="Reserve/check questions without API calls"
    )
    supervised = sub.add_parser("sft", help="Supervised training on the collected teacher answers")
    supervised.add_argument("--output", help="New SFT run directory")
    supervised.add_argument("--resume", help="latest or an SFT checkpoint path")
    supervised.add_argument(
        "--dry-run", action="store_true", help="Validate/tokenize data without training"
    )
    overnight = sub.add_parser("pipeline", help="Run SFT, then start GRPO from its saved model")
    overnight.add_argument("--sft-output", help="SFT output; defaults to [sft].output_dir")
    overnight.add_argument("--rl-output", help="GRPO output; runs/lfm3-childishness-after-sft-01")
    overnight.add_argument("--resume", action="store_true", help="Skip completed stages and resume")
    overnight.add_argument(
        "--dry-run", action="store_true", help="Show commands without running them"
    )
    viewer = sub.add_parser("view", help="Live local judge viewer, or export a standalone HTML")
    viewer.add_argument("log", nargs="?", help="Optional log to export as a standalone HTML")
    viewer.add_argument(
        "--output", help="HTML output; defaults to the input path with .html suffix"
    )
    viewer.add_argument("--runs-dir", default="runs", help="Folder to watch (default: runs)")
    viewer.add_argument("--port", type=int, default=8765, help="Local viewer port (default: 8765)")
    training = sub.add_parser(
        "train", help="Run full-parameter GRPO; makes paid Anthropic requests"
    )
    training.add_argument(
        "--dry-run", action="store_true", help="Show settings only; no downloads/API"
    )
    training.add_argument("--smoke-test", action="store_true", help="Run one real optimizer step")
    training.add_argument("--resume", help="latest or a checkpoint path")
    training.add_argument("--output", help="Override run directory")
    training.add_argument(
        "--model", help="Starting model ID or saved SFT model for a fresh GRPO run"
    )
    recovery = sub.add_parser(
        "recover", help="Replay a stopped GRPO run using cached scores, then finish training"
    )
    recovery.add_argument("--run", required=True, help="Original interrupted run directory")
    recovery.add_argument("--output", help="Separate output; default: original name + -recovery")
    recovery.add_argument("--max-steps", type=int, default=1536, help="Final global step (1536)")
    recovery.add_argument("--resume", help="latest or checkpoint name within the recovery output")
    recovery.add_argument("--dry-run", action="store_true", help="Show replay plan without GPU/API")
    recovery.add_argument(
        "--on-divergence", choices=["stop", "rejudge"], default="stop",
        help="stop keeps replay cache-only; rejudge permits paid scoring for changed groups",
    )
    evaluation = sub.add_parser("evaluate", help="Evaluate base/checkpoint; uses GPU and paid API")
    evaluation.add_argument("--model", help="Checkpoint path; omit to evaluate the original model")
    evaluation.add_argument("--output", required=True, help="New JSONL output file")
    evaluation.add_argument("--limit", type=int, help="Limit held-out instructions")
    comparison = sub.add_parser(
        "compare", help="Compare a trained checkpoint with its original model"
    )
    comparison.add_argument("--model", required=True, help="Trained final/ or checkpoint directory")
    comparison.add_argument(
        "--baseline", help="Original model; default: run's initial_model snapshot"
    )
    comparison.add_argument("--output", required=True, help="New comparison output directory")
    comparison.add_argument("--limit", type=int, help="Limit held-out questions; default: all")
    comparison.add_argument(
        "--resume", action="store_true", help="Retry judging saved answers; no generation"
    )
    args = parser.parse_args()
    try:
        if args.command == "view":
            from .viewer import export_viewer, serve_viewer

            if args.log:
                target = export_viewer(args.log, args.output)
                print(f"Open in your browser: {target}")
            elif args.output:
                raise ValueError("--output requires a log filename for HTML export")
            else:
                if not 0 <= args.port <= 65535:
                    raise ValueError("--port must be between 0 and 65535")
                serve_viewer(args.runs_dir, args.port)
            return 0
        settings = load_settings(args.config)
        if args.command == "check":
            packages = {}
            for name in ["torch", "transformers", "trl", "datasets", "anthropic", "bitsandbytes"]:
                try:
                    packages[name] = version(name)
                except PackageNotFoundError:
                    packages[name] = "NOT INSTALLED"
            from .training import hardware_info

            print(
                json.dumps(
                    {
                        "settings": settings.public_dict(),
                        "packages": packages,
                        "hardware": hardware_info(),
                    },
                    indent=2,
                )
            )
        elif args.command == "prepare":
            from .data import prepare

            print(json.dumps(prepare(settings), indent=2))
        elif args.command == "collect":
            from .collection import collect

            collect(settings, args.count, args.pool_size, args.prepare_only)
        elif args.command == "sft":
            from .sft import sft

            sft(settings, args.output, args.resume, args.dry_run)
        elif args.command == "pipeline":
            from .pipeline import pipeline

            pipeline(settings, args.config, args.sft_output, args.rl_output,
                     args.resume, args.dry_run)
        elif args.command == "train":
            if args.dry_run:
                if args.model:
                    from .training import resolve_starting_model

                    settings.training.model = resolve_starting_model(settings, args.model)
                print(json.dumps(settings.public_dict(), indent=2))
                print("Dry run: no models downloaded, no API calls, no optimizer steps.")
            else:
                from .training import train

                train(settings, args.smoke_test, args.resume, args.output, args.model)
        elif args.command == "recover":
            from .recovery import recover

            recover(settings, args.run, args.output, args.max_steps, args.resume, args.dry_run,
                    args.on_divergence)
        elif args.command == "evaluate":
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit must be positive")
            from .evaluation import evaluate

            evaluate(settings, args.model, args.output, args.limit)
        elif args.command == "compare":
            if args.limit is not None and args.limit <= 0:
                raise ValueError("--limit must be positive")
            from .evaluation import compare

            compare(settings, args.model, args.output, args.baseline, args.limit, args.resume)
    except (ValueError, RuntimeError, OSError) as exc:
        # Anthropic exceptions are sanitized at the request boundary. Config errors
        # never include supplied values or API credentials.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
