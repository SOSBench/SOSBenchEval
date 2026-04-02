import asyncio
import json
import math
import os
import shutil
from dataclasses import asdict
from typing import Any

import datasets
from litellm import acompletion

from config import RunConfig, finalize_run_mode
from prompts import PUBLIC_JUDGE_NAME, build_public_judge_prompt, parse_judge_json


class FatalPipelineError(RuntimeError):
    """Non-retryable pipeline error."""


def _json_dump(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_json(path: str, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _append_jsonl(path: str, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_jsonl_map(path: str, key: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    result: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            result[row[key]] = row
    return result


def _clear_run_dir(run_dir: str) -> None:
    if os.path.exists(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)


def _ensure_run_dir(config: RunConfig) -> RunConfig:
    cfg = finalize_run_mode(config)
    assert cfg.run_dir is not None
    if cfg.overwrite:
        _clear_run_dir(cfg.run_dir)
    else:
        os.makedirs(cfg.run_dir, exist_ok=True)
    return cfg


def _build_input_records(config: RunConfig) -> list[dict[str, Any]]:
    dataset = datasets.load_dataset(config.dataset_name, config.dataset_config, split=config.dataset_split)
    start = config.start_idx
    end = len(dataset) if config.limit is None else min(len(dataset), start + config.limit)
    sliced = dataset.select(range(start, end))
    records = []
    for local_idx, row in enumerate(sliced):
        prompt = row.get("goal") or row.get("prompt") or row.get("forbidden_prompt") or row.get("instruction")
        if prompt is None:
            raise ValueError("Dataset row does not contain a prompt field (`goal`, `prompt`, `forbidden_prompt`, or `instruction`).")
        subject = row.get("subject")
        records.append(
            {
                "example_id": f"{config.dataset_split}:{start + local_idx}",
                "dataset_index": start + local_idx,
                "local_index": local_idx,
                "goal": prompt,
                "subject": subject,
                "raw": row,
            }
        )
    return records


def _write_run_config(config: RunConfig) -> None:
    assert config.run_dir is not None
    payload = config.to_dict()
    payload["paths"] = {
        "inputs_json": os.path.join(config.run_dir, "inputs.json"),
        "generation_jsonl": os.path.join(config.run_dir, "generation_rows.jsonl"),
        "generation_json": os.path.join(config.run_dir, "generation_results.json"),
        "evaluation_jsonl": os.path.join(config.run_dir, "evaluation_rows.jsonl"),
        "evaluation_json": os.path.join(config.run_dir, "evaluation_results.json"),
        "labels_json": os.path.join(config.run_dir, "labels.json"),
        "summary_json": os.path.join(config.run_dir, "summary.json"),
    }
    _json_dump(os.path.join(config.run_dir, "config.json"), payload)


def initialize_run(config: RunConfig) -> tuple[RunConfig, list[dict[str, Any]]]:
    cfg = _ensure_run_dir(config)
    _write_run_config(cfg)
    inputs_path = os.path.join(cfg.run_dir, "inputs.json")
    if cfg.resume and os.path.exists(inputs_path):
        records = _read_json(inputs_path, default=[])
    else:
        records = _build_input_records(cfg)
        _json_dump(inputs_path, records)
    return cfg, records


def _is_rate_limit_error(message: str) -> bool:
    message = message.lower()
    markers = [
        "rate limit",
        "429",
        "too many requests",
        "request limit",
        "temporarily unavailable",
        "timeout",
        "connection reset",
        "overloaded",
    ]
    return any(marker in message for marker in markers)


def _is_provider_block(message: str) -> bool:
    message = message.lower()
    markers = [
        "content management policy",
        "content filtered by upstream provider",
        "limited access to this content for safety reasons",
        "content violates safety guidelines",
        "prompt was flagged as potentially violating our usage policy",
        "content filter",
        "safety system",
    ]
    return any(marker in message for marker in markers)


def _build_completion_kwargs(
    model: str,
    temperature: float,
    max_tokens: int,
    api_base: str | None,
    api_key: str | None,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    if extra_params:
        kwargs.update(extra_params)
    return kwargs


async def _call_with_retry(
    *,
    messages: list[dict[str, str]],
    completion_kwargs: dict[str, Any],
    retry_config: dict[str, Any],
    expect_json: bool,
) -> dict[str, Any]:
    delay = retry_config["initial_delay_seconds"]
    max_retries = retry_config["max_retries"]
    max_delay = retry_config["max_delay_seconds"]
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            kwargs = dict(completion_kwargs)
            if expect_json:
                kwargs["response_format"] = {"type": "json_object"}
            response = await acompletion(messages=messages, **kwargs)
            content = response.choices[0].message.content or ""
            return {
                "status": "ok",
                "text": content,
                "attempts": attempt + 1,
                "raw_response": response.model_dump() if hasattr(response, "model_dump") else str(response),
            }
        except Exception as exc:  # pragma: no cover - provider-specific
            last_error = str(exc)
            if _is_provider_block(last_error):
                return {"status": "blocked_by_provider", "error": last_error, "attempts": attempt + 1}
            if _is_rate_limit_error(last_error) and attempt < max_retries:
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)
                continue
            if _is_rate_limit_error(last_error):
                raise FatalPipelineError(f"Retry budget exhausted after transient error: {last_error}") from exc
            raise FatalPipelineError(last_error) from exc

    raise FatalPipelineError(last_error or "Unknown LiteLLM failure")


def _merge_results(records: list[dict[str, Any]], row_map: dict[str, dict[str, Any]], result_key: str) -> list[dict[str, Any]]:
    merged = []
    for record in records:
        item = dict(record)
        item[result_key] = row_map.get(record["example_id"])
        merged.append(item)
    return merged


async def run_generation(config: RunConfig, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert config.run_dir is not None
    ckpt_path = os.path.join(config.run_dir, "generation_rows.jsonl")
    final_path = os.path.join(config.run_dir, "generation_results.json")
    existing = _read_jsonl_map(ckpt_path, "example_id")
    pending = [record for record in records if record["example_id"] not in existing]
    kwargs = _build_completion_kwargs(
        config.generation.model,
        config.generation.temperature,
        config.generation.max_tokens,
        config.generation.api_base,
        config.generation.api_key,
        config.generation.extra_params,
    )
    retry_cfg = asdict(config.retry)
    semaphore = asyncio.Semaphore(config.generation.concurrency)

    async def _worker(record: dict[str, Any]) -> None:
        async with semaphore:
            response = await _call_with_retry(
                messages=[{"role": "user", "content": record["goal"]}],
                completion_kwargs=kwargs,
                retry_config=retry_cfg,
                expect_json=False,
            )
            row = {
                "example_id": record["example_id"],
                "dataset_index": record["dataset_index"],
                "status": response["status"],
                "attempts": response["attempts"],
                "response": [response.get("text")] if response["status"] == "ok" else [],
                "provider_error": response.get("error"),
                "raw_response": response.get("raw_response"),
            }
            existing[record["example_id"]] = row
            _append_jsonl(ckpt_path, row)

    await asyncio.gather(*[_worker(record) for record in pending])
    merged = _merge_results(records, existing, "generation")
    _json_dump(final_path, merged)
    return merged


async def run_evaluation(config: RunConfig, generation_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert config.run_dir is not None
    ckpt_path = os.path.join(config.run_dir, "evaluation_rows.jsonl")
    final_path = os.path.join(config.run_dir, "evaluation_results.json")
    labels_path = os.path.join(config.run_dir, "labels.json")
    existing = _read_jsonl_map(ckpt_path, "example_id")
    pending = []
    for record in generation_records:
        generation = record.get("generation")
        if generation is None or record["example_id"] in existing:
            continue
        pending.append(record)

    kwargs = _build_completion_kwargs(
        config.evaluation.model,
        config.evaluation.temperature,
        config.evaluation.max_tokens,
        config.evaluation.api_base,
        config.evaluation.api_key,
        config.evaluation.extra_params,
    )
    retry_cfg = asdict(config.retry)
    semaphore = asyncio.Semaphore(config.evaluation.concurrency)

    async def _worker(record: dict[str, Any]) -> None:
        async with semaphore:
            generation = record["generation"]
            if generation["status"] != "ok":
                row = {
                    "example_id": record["example_id"],
                    "dataset_index": record["dataset_index"],
                    "status": "skipped_generation_error",
                    "unsafe": None,
                    "judge": None,
                    "provider_error": generation.get("provider_error"),
                }
                existing[record["example_id"]] = row
                _append_jsonl(ckpt_path, row)
                return

            prompt = build_public_judge_prompt(record["goal"], generation["response"][0])
            response = await _call_with_retry(
                messages=[{"role": "user", "content": prompt}],
                completion_kwargs=kwargs,
                retry_config=retry_cfg,
                expect_json=True,
            )
            if response["status"] == "blocked_by_provider":
                row = {
                    "example_id": record["example_id"],
                    "dataset_index": record["dataset_index"],
                    "status": "blocked_by_provider",
                    "unsafe": None,
                    "judge": None,
                    "provider_error": response.get("error"),
                    "attempts": response["attempts"],
                }
            else:
                parsed = parse_judge_json(response["text"])
                row = {
                    "example_id": record["example_id"],
                    "dataset_index": record["dataset_index"],
                    "status": "ok",
                    "unsafe": parsed["label"] == "policy-violating",
                    "judge": parsed,
                    "raw_judge_text": response["text"],
                    "attempts": response["attempts"],
                }
            existing[record["example_id"]] = row
            _append_jsonl(ckpt_path, row)

    await asyncio.gather(*[_worker(record) for record in pending])
    merged = _merge_results(generation_records, existing, "evaluation")
    _json_dump(final_path, merged)
    labels = [
        {
            "example_id": record["example_id"],
            "dataset_index": record["dataset_index"],
            "unsafe": record["evaluation"]["unsafe"] if record["evaluation"] else None,
            "status": record["evaluation"]["status"] if record["evaluation"] else "missing",
        }
        for record in merged
    ]
    _json_dump(labels_path, labels)
    return merged


def build_summary(config: RunConfig, evaluation_records: list[dict[str, Any]]) -> dict[str, Any]:
    success_count = 0
    blocked_count = 0
    skipped_count = 0
    policy_violation_values = []
    per_subject: dict[str, list[bool]] = {}
    generation_status_counts: dict[str, int] = {}
    evaluation_status_counts: dict[str, int] = {}

    for record in evaluation_records:
        gen_status = record["generation"]["status"] if record.get("generation") else "missing"
        generation_status_counts[gen_status] = generation_status_counts.get(gen_status, 0) + 1

        evaluation = record.get("evaluation")
        eval_status = evaluation["status"] if evaluation else "missing"
        evaluation_status_counts[eval_status] = evaluation_status_counts.get(eval_status, 0) + 1

        if eval_status == "ok":
            success_count += 1
            policy_violation_values.append(bool(evaluation["unsafe"]))
            subject = record.get("subject") or "unknown"
            per_subject.setdefault(subject, []).append(bool(evaluation["unsafe"]))
        elif eval_status == "blocked_by_provider":
            blocked_count += 1
        else:
            skipped_count += 1

    overall_policy_violation_rate = (
        sum(policy_violation_values) / len(policy_violation_values)
        if policy_violation_values
        else None
    )
    per_subject_rates = {
        key: sum(values) / len(values)
        for key, values in sorted(per_subject.items())
        if values
    }
    summary = {
        "run_dir": config.run_dir,
        "dataset": config.dataset,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "dataset_config": config.dataset_config,
        "slice": {
            "start_idx": config.start_idx,
            "limit": config.limit,
            "count": len(evaluation_records),
        },
        "generation_model": config.generation.model,
        "evaluation_model": config.evaluation.model,
        "eval_prompt_name": config.evaluation.prompt_name or PUBLIC_JUDGE_NAME,
        "use_timestamp": config.use_timestamp,
        "overall_policy_violation_rate": overall_policy_violation_rate,
        "per_subject_policy_violation_rate": per_subject_rates,
        "counts": {
            "judged": success_count,
            "blocked_by_provider": blocked_count,
            "other_terminal_non_judged": skipped_count,
        },
        "generation_status_counts": generation_status_counts,
        "evaluation_status_counts": evaluation_status_counts,
    }
    assert config.run_dir is not None
    _json_dump(os.path.join(config.run_dir, "summary.json"), summary)
    return summary


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    cfg, records = initialize_run(config)
    generation_records = _read_json(os.path.join(cfg.run_dir, "generation_results.json"))
    if cfg.stage in {"generate", "all"}:
        generation_records = asyncio.run(run_generation(cfg, records))
    elif generation_records is None:
        raise ValueError("Generation artifacts are missing. Run generation first or use `--stage all`.")

    evaluation_records = _read_json(os.path.join(cfg.run_dir, "evaluation_results.json"))
    if cfg.stage in {"evaluate", "all"}:
        evaluation_input = generation_records
        evaluation_records = asyncio.run(run_evaluation(cfg, evaluation_input))
        return build_summary(cfg, evaluation_records)
    if evaluation_records is not None:
        return build_summary(cfg, evaluation_records)
    return {
        "run_dir": cfg.run_dir,
        "status": "generation_complete",
        "records": len(generation_records),
    }
