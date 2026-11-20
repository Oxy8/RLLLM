from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import random
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from .answers import (
    decode_final_answer,
    final_answer,
    generation_diagnostics,
    limit_answer_tokens,
    thinking_is_prefilled,
)
from .config import Settings


class JudgeRequestError(RuntimeError):
    """A failed provider request; no valid reward exists for this rollout."""


class JudgeResponseError(JudgeRequestError):
    """A question-specific refusal or unusable evaluation that can be replaced."""


class JudgeValidationError(JudgeResponseError, ValueError):
    """The judge exhausted its retries without producing valid scores."""


SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "optimism": {"type": "number"},
                    "factual_consistency": {
                        "type": "string",
                        "enum": ["consistent", "contradicted", "unclear"],
                    },
                    "relevant": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "optimism", "factual_consistency", "relevant", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def score_schema(score_name: str = "optimism", *, task_quality: bool = False) -> dict:
    if task_quality:
        properties = {
            "id": {"type": "string"},
            score_name: {"type": "number"},
            "task_quality": {"type": "number"},
            f"{score_name}_reason": {"type": "string"},
            "task_quality_reason": {"type": "string"},
        }
        return {"type": "object", "properties": {
            "scores": {"type": "array", "items": {
                "type": "object", "properties": properties,
                "required": list(properties), "additionalProperties": False,
            }},
        }, "required": ["scores"], "additionalProperties": False}
    schema = copy.deepcopy(SCORE_SCHEMA)
    item = schema["properties"]["scores"]["items"]
    item["properties"][score_name] = item["properties"].pop("optimism")
    item["required"] = [score_name if key == "optimism" else key for key in item["required"]]
    return schema


def candidate_score_schema(candidate_ids: list[str], score_name: str = "optimism", *,
                           task_quality: bool = False) -> dict:
    """Require a separate field for every candidate, rather than an unbounded array."""
    item = score_schema(score_name, task_quality=task_quality)["properties"]["scores"]["items"]
    del item["properties"]["id"]
    item["required"].remove("id")
    return {
        "type": "object",
        "properties": {
            "scores": {
                "type": "object",
                "properties": {candidate: copy.deepcopy(item) for candidate in candidate_ids},
                "required": candidate_ids,
                "additionalProperties": False,
            }
        },
        "required": ["scores"],
        "additionalProperties": False,
    }


def plain_text(value) -> str:
    """Extract text from TRL's string or conversational completion formats."""
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError("Expected a string or a list of chat messages")
    if len(value) != 1 or value[0].get("role") not in {"user", "assistant"}:
        raise ValueError("This experiment requires one standalone instruction/response per row")
    content = value[0].get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list) and all(part.get("type") == "text" for part in content):
        return "\n".join(part["text"] for part in content)
    raise ValueError("Only text instructions and responses are supported")


def validate_scores(
    data: dict, expected_ids: list[str], score_name: str = "optimism", *, task_quality=False,
) -> dict[str, dict]:
    rows = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        actual = len(rows) if isinstance(rows, list) else "missing scores array"
        raise ValueError(
            f"Judge must return exactly one score per candidate "
            f"(expected {len(expected_ids)}, received {actual})"
        )
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("id") not in expected_ids:
            raise ValueError("Judge returned an unknown candidate ID")
        if row["id"] in result:
            raise ValueError("Judge returned a duplicate candidate ID")
        score = row.get(score_name)
        if type(score) not in {int, float} or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Judge {score_name} scores must be finite numbers in [0, 1]")
        if task_quality:
            quality = row.get("task_quality")
            if (type(quality) not in {int, float} or not math.isfinite(quality)
                    or not 0 <= quality <= 1):
                raise ValueError("Judge task_quality scores must be finite numbers in [0, 1]")
            if not all(isinstance(row.get(key), str)
                       for key in [f"{score_name}_reason", "task_quality_reason"]):
                raise ValueError("Judge must explain both the main score and task quality")
        else:
            if row.get("factual_consistency") not in {"consistent", "contradicted", "unclear"}:
                raise ValueError("Invalid factual-consistency diagnostic")
            if type(row.get("relevant")) is not bool or not isinstance(row.get("reason"), str):
                raise ValueError("Invalid relevance/reason diagnostic")
        result[row["id"]] = row
    return result


class OptimismJudge:
    """Thread-safe disk cache; async, bounded API calls; strict ID-to-reward alignment."""

    def __init__(
        self,
        settings: Settings,
        log_path: Path | None = None,
        request: Callable[[dict], Awaitable[dict]] | None = None,
        tokenizer=None,
        max_answer_tokens: int | None = None,
    ):
        self.settings = settings
        self.score_name = settings.judge.score_name
        self.task_quality = settings.judge.uses_task_quality
        self.score_schema = score_schema(self.score_name, task_quality=self.task_quality)
        self.tokenizer = tokenizer
        self.max_answer_tokens = (
            settings.training.max_answer_tokens if max_answer_tokens is None else max_answer_tokens
        )
        self.thinking_prefilled = (
            thinking_is_prefilled(tokenizer, settings.training.enable_thinking)
            if tokenizer is not None
            else False
        )
        self.rubric = settings.path(settings.judge.rubric_path).read_text()
        self.log_path = log_path
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
        cache = settings.path(settings.judge.cache_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(cache, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, value TEXT)")
        self.db.commit()
        self.request = request or self._api_request
        self.requests = 0
        self.cache_hits = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_only = False
        self.training_step = None
        self.last_group_judgments = None

    def reward_value(self, judgment: dict) -> float:
        trait = float(judgment[self.score_name])
        if not self.task_quality:
            return trait
        if self.settings.judge.reward_mode == "quality_softplus":
            quality = float(judgment["task_quality"])
            sharpness = self.settings.judge.quality_softplus_sharpness
            x = sharpness * (self.settings.judge.quality_softplus_center - quality)
            # Stable log(1 + exp(x)); no clipping or discontinuous reward thresholds.
            softplus = max(x, 0.0) + math.log1p(math.exp(-abs(x)))
            return trait - self.settings.judge.quality_softplus_scale / sharpness * softplus
        baseline = self.settings.judge.quality_baseline
        return float(judgment["task_quality"]) * (baseline + (1 - baseline) * trait)

    def close(self):
        self.db.close()

    def statistics(self) -> dict:
        return {
            "api_requests": self.requests,
            "cache_hits": self.cache_hits,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }

    def _record_request_error(self, payload, **details):
        if self.log_path:
            record = {"timestamp_utc": datetime.now(UTC).isoformat(),
                      "judge_model": self.settings.judge.model,
                      "instruction": payload["instruction"],
                      "candidates": payload["candidates"], **details}
            with self.lock, self.log_path.with_suffix(".errors.jsonl").open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def _api_request(self, payload: dict) -> dict:
        if self.tokenizer is None:
            raise ValueError(
                "A local tokenizer is required to enforce the judge answer-token limit"
            )
        from anthropic import AsyncAnthropic

        config = self.settings.judge
        try:
            async with AsyncAnthropic(
                api_key=self.settings.api.anthropic_api_key,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            ) as client:
                response = await client.messages.create(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    # Haiku 5.5 otherwise enables adaptive thinking by default.
                    # Other judge models may require different thinking settings.
                    **(
                        {"thinking": {"type": "disabled"}}
                        if config.model.startswith("claude-haiku-5-5")
                        else {}
                    ),
                    system=self.rubric,
                    messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                    output_config={
                        "format": {
                            "type": "json_schema",
                            "schema": candidate_score_schema(
                                [row["id"] for row in payload["candidates"]], self.score_name,
                                task_quality=self.task_quality,
                            ),
                        }
                    },
                )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            self._record_request_error(payload, error_type=type(exc).__name__, http_status=status)
            raise JudgeRequestError(
                f"Anthropic request failed ({type(exc).__name__}, HTTP {status}). "
                "Check API access/model, credits, and rate limits. Training stopped; "
                "failed requests are never converted into rewards."
            ) from None
        if response.stop_reason != "end_turn":
            with self.lock:
                self.input_tokens += response.usage.input_tokens
                self.output_tokens += response.usage.output_tokens
            details = getattr(response, "stop_details", None)
            if hasattr(details, "model_dump"):
                details = details.model_dump(mode="json")
            self._record_request_error(
                payload, stop_reason=response.stop_reason, stop_details=details,
                judge_response="".join(b.text for b in response.content if b.type == "text"),
                usage={"input_tokens": response.usage.input_tokens,
                       "output_tokens": response.usage.output_tokens},
            )
            if response.stop_reason == "refusal":
                raise JudgeResponseError(
                    "Judge refused the evaluation (refusal); no reward was used. "
                    "See judge.errors.jsonl for the refusal details."
                )
            if response.stop_reason == "max_tokens":
                raise JudgeResponseError(
                    "Judge response was truncated (max_tokens); no reward was used. "
                    "Increase judge.max_tokens."
                )
            raise JudgeResponseError(
                f"Judge did not finish normally ({response.stop_reason}); no reward was used."
            )
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            result = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            with self.lock:
                self.input_tokens += response.usage.input_tokens
                self.output_tokens += response.usage.output_tokens
            raise ValueError("Judge returned invalid JSON; no reward was used") from None
        if not isinstance(result, dict):
            with self.lock:
                self.input_tokens += response.usage.input_tokens
                self.output_tokens += response.usage.output_tokens
            raise ValueError("Judge returned a non-object JSON response; no reward was used")
        # Keep the existing internal/cache/log format. Only the wire schema changes.
        if isinstance(result.get("scores"), dict):
            result["scores"] = [
                {**row, "id": candidate} if isinstance(row, dict) else row
                for candidate, row in result["scores"].items()
            ]
        result["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return result

    async def _score_group(
        self, instruction: str, responses: list[str], diagnostics: list[dict] | None = None
    ) -> list[dict]:
        responses = [final_answer(response) for response in responses]
        if self.tokenizer is not None:
            responses = [
                limit_answer_tokens(self.tokenizer, answer, self.max_answer_tokens)
                for answer in responses
            ]
        # Identical candidate texts receive the same score. Shuffle unique candidates to
        # reduce position bias, then map IDs back to the original TRL completion order.
        unique = list(dict.fromkeys(responses))
        fingerprint = json.dumps([instruction, unique], ensure_ascii=False)
        seed = int(hashlib.sha256(fingerprint.encode()).hexdigest()[:16], 16)
        order = list(range(len(unique)))
        random.Random(seed).shuffle(order)
        candidates = [{"id": f"c{i}", "response": unique[i]} for i in order]
        payload = {"instruction": instruction, "candidates": candidates}
        key = hashlib.sha256(
            json.dumps(
                # Preserve valid historical optimism cache keys. Other score schemas
                # have different keys even if their text happens to match.
                ["optimism-v1", self.settings.judge.model, self.rubric, self.score_schema, payload],
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()
        with self.lock:
            cached = self.db.execute("SELECT value FROM judgments WHERE key=?", (key,)).fetchone()
            if cached:
                self.cache_hits += 1
                result = json.loads(cached[0])
            else:
                result = None
        if result is None:
            if self.cache_only:
                raise RuntimeError(
                    "Recovery judge cache miss: stopped without calling Anthropic"
                )
            for attempt in range(self.settings.judge.max_retries + 1):
                with self.lock:
                    if self.requests >= self.settings.judge.max_requests:
                        raise JudgeRequestError(
                            "judge.max_requests reached; raise it in .config to continue"
                        )
                    self.requests += 1
                result = None
                try:
                    result = await self.request(payload)
                    with self.lock:
                        usage = result.get("usage", {}) if isinstance(result, dict) else {}
                        self.input_tokens += usage.get("input_tokens", 0)
                        self.output_tokens += usage.get("output_tokens", 0)
                    validate_scores(result, [row["id"] for row in candidates], self.score_name,
                                    task_quality=self.task_quality)
                    break
                except ValueError as exc:
                    self._record_request_error(
                        payload, cache_key=key, attempt=attempt + 1, error=str(exc),
                        expected_ids=[row["id"] for row in candidates], response=result,
                    )
                    if attempt == self.settings.judge.max_retries:
                        raise JudgeValidationError(
                            f"{exc}; failed after {attempt + 1} judge attempts. "
                            "No invalid scores were used; successful judgments remain cached."
                        ) from None
            with self.lock:
                self.db.execute(
                    "INSERT OR REPLACE INTO judgments VALUES (?, ?)", (key, json.dumps(result))
                )
                self.db.commit()
        by_id = validate_scores(result, [row["id"] for row in candidates], self.score_name,
                                task_quality=self.task_quality)
        by_text = {text: by_id[f"c{i}"] for i, text in enumerate(unique)}
        ordered = [dict(by_text[text]) for text in responses]
        if self.task_quality:
            for row in ordered:
                row["reward"] = self.reward_value(row)
        if self.log_path:
            with self.lock, self.log_path.open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "cache_key": key,
                            "cached": bool(cached),
                            "judge_model": self.settings.judge.model,
                            "score_name": self.score_name,
                            "reward_mode": self.settings.judge.reward_mode,
                            **self.settings.judge.reward_parameters(),
                            "policy_model": self.settings.training.model,
                            "instruction": instruction,
                            "responses": responses,
                            "judgments": ordered,
                            "generation_diagnostics": diagnostics,
                            "training_step": self.training_step,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return ordered

    async def score_groups(
        self, groups: list[tuple[str, list[str]]], diagnostics: list[list[dict]] | None = None,
        *, return_exceptions: bool = False,
    ) -> list:
        semaphore = asyncio.Semaphore(self.settings.judge.concurrency)

        async def limited(group, info):
            async with semaphore:
                return await self._score_group(*group, diagnostics=info)

        results = await asyncio.gather(
            *(
                limited(group, info)
                for group, info in zip(groups, diagnostics or [None] * len(groups), strict=True)
            ),
            return_exceptions=True,
        )
        # Finish every in-flight request so valid sibling judgments reach the cache.
        # Configuration/access/network errors remain fatal even during replacement.
        for result in results:
            if isinstance(result, BaseException) and (
                not return_exceptions or not isinstance(result, JudgeResponseError)
            ):
                raise result
        return results

    async def reward(self, prompts, completions, *, return_groups=False, **kwargs) -> list:
        size = self.settings.training.num_generations
        if len(prompts) != len(completions) or not completions or len(completions) % size:
            raise ValueError("Unexpected GRPO reward batch size")
        ids = kwargs.get("completion_ids")
        if self.tokenizer is not None and ids is not None:
            if len(ids) != len(completions):
                raise ValueError("Completion token IDs do not align with answers")
            answers = [
                decode_final_answer(self.tokenizer, row, thinking_prefilled=self.thinking_prefilled)
                for row in ids
            ]
            diagnostics = [
                generation_diagnostics(
                    self.tokenizer,
                    row,
                    self.settings.training.max_completion_length,
                    thinking_prefilled=self.thinking_prefilled,
                    max_answer_tokens=self.max_answer_tokens,
                )
                for row in ids
            ]
        else:
            answers = [final_answer(plain_text(completion)) for completion in completions]
            diagnostics = None
        groups = []
        for start in range(0, len(prompts), size):
            texts = [plain_text(p) for p in prompts[start : start + size]]
            if len(set(texts)) != 1:
                raise ValueError("GRPO completions are not grouped by instruction as expected")
            groups.append((texts[0], answers[start : start + size]))
        results = await self.score_groups(
            groups,
            diagnostics=[
                diagnostics[start : start + size] for start in range(0, len(answers), size)
            ]
            if diagnostics is not None
            else None,
            **({"return_exceptions": True} if return_groups else {}),
        )
        self.last_group_judgments = results
        # Keep raw scores separate; recompute the combination when coefficients change.
        if return_groups:
            return [group if isinstance(group, JudgeResponseError)
                    else [self.reward_value(row) for row in group] for group in results]
        return [self.reward_value(row) for group in results for row in group]

    def reward_sync(self, prompts, completions, **kwargs) -> list[float]:
        # TRL's training loop is synchronous. Keep concurrent HTTP work inside this
        # call instead of handing it to TRL's separate daemon event-loop thread.
        return asyncio.run(self.reward(prompts, completions, **kwargs))
