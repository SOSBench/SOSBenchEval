import argparse
import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from dotenv import load_dotenv


load_dotenv()


DATASET_ALIASES = {
    "sosbench": "SOSBench/SOSBench",
    "sosbench-lite": "SOSBench/SOSBench-Lite",
}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def normalize_name(value: str) -> str:
    safe = []
    for char in value:
        safe.append(char if char.isalnum() or char in {"-", "_", "."} else "_")
    return "".join(safe).strip("_") or "run"


@dataclass
class RetryConfig:
    max_retries: int = 5
    initial_delay_seconds: float = 3.0
    max_delay_seconds: float = 30.0


@dataclass
class GenerationConfig:
    model: str
    temperature: float
    max_tokens: int
    concurrency: int
    api_base: str | None = None
    api_key: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    model: str
    temperature: float
    max_tokens: int
    concurrency: int
    api_base: str | None = None
    api_key: str | None = None
    prompt_name: str = "sosbench_judge_v2"
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    stage: str
    dataset: str
    dataset_name: str
    dataset_split: str
    dataset_config: str | None
    start_idx: int
    limit: int | None
    run_root: str
    run_name: str | None
    use_timestamp: bool
    run_dir: str | None
    resume: bool
    overwrite: bool
    generation: GenerationConfig
    evaluation: EvaluationConfig
    retry: RetryConfig = field(default_factory=RetryConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stage", choices=["generate", "evaluate", "all"], default="all")
    parser.add_argument("--dataset", default=os.getenv("SOS_DATASET_ALIAS", "sosbench"))
    parser.add_argument("--dataset-name", default=os.getenv("SOS_DATASET_NAME"))
    parser.add_argument("--dataset-split", default=os.getenv("SOS_DATASET_SPLIT", "train"))
    parser.add_argument("--dataset-config", default=os.getenv("SOS_DATASET_CONFIG"))
    parser.add_argument("--start-idx", type=int, default=_env_int("SOS_START_IDX", 0))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-root", default=os.getenv("SOS_RUN_ROOT", "./runs"))
    parser.add_argument("--run-name", default=os.getenv("SOS_RUN_NAME"))
    parser.add_argument("--use-timestamp", action="store_true", default=_env_bool("SOS_USE_TIMESTAMP", False))
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--gen-model", default=os.getenv("SOS_GEN_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--gen-temperature", type=float, default=_env_float("SOS_GEN_TEMPERATURE", 1.0))
    parser.add_argument("--gen-max-tokens", type=int, default=_env_int("SOS_GEN_MAX_TOKENS", 512))
    parser.add_argument("--gen-concurrency", type=int, default=_env_int("SOS_GEN_CONCURRENCY", 20))
    parser.add_argument("--gen-api-base", default=os.getenv("SOS_GEN_API_BASE") or os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--gen-api-key", default=os.getenv("SOS_GEN_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--gen-extra-params", default=None, help="Inline JSON object passed through to LiteLLM for generation.")
    parser.add_argument("--eval-model", default=os.getenv("SOS_EVAL_MODEL", "gpt-5"))
    parser.add_argument("--eval-temperature", type=float, default=_env_float("SOS_EVAL_TEMPERATURE", 1.0))
    parser.add_argument("--eval-max-tokens", type=int, default=_env_int("SOS_EVAL_MAX_TOKENS", 16000))
    parser.add_argument("--eval-concurrency", type=int, default=_env_int("SOS_EVAL_CONCURRENCY", 10))
    parser.add_argument("--eval-api-base", default=os.getenv("SOS_EVAL_API_BASE") or os.getenv("OPENAI_API_BASE"))
    parser.add_argument("--eval-api-key", default=os.getenv("SOS_EVAL_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--eval-extra-params", default=None, help="Inline JSON object passed through to LiteLLM for evaluation.")
    parser.add_argument("--max-retries", type=int, default=_env_int("SOS_MAX_RETRIES", 5))
    parser.add_argument("--retry-initial-delay", type=float, default=_env_float("SOS_RETRY_INITIAL_DELAY", 3.0))
    parser.add_argument("--retry-max-delay", type=float, default=_env_float("SOS_RETRY_MAX_DELAY", 30.0))


def _parse_inline_json(raw: str | None, flag_name: str) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag_name} must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object.")
    return parsed


def load_run_config(args: argparse.Namespace) -> RunConfig:
    load_dotenv()
    dataset_name = args.dataset_name or DATASET_ALIASES.get(args.dataset, args.dataset)

    generation = GenerationConfig(
        model=args.gen_model,
        temperature=args.gen_temperature,
        max_tokens=args.gen_max_tokens,
        concurrency=args.gen_concurrency,
        api_base=args.gen_api_base,
        api_key=args.gen_api_key,
        extra_params=_parse_inline_json(args.gen_extra_params, "--gen-extra-params"),
    )
    evaluation = EvaluationConfig(
        model=args.eval_model,
        temperature=args.eval_temperature,
        max_tokens=args.eval_max_tokens,
        concurrency=args.eval_concurrency,
        api_base=args.eval_api_base,
        api_key=args.eval_api_key,
        extra_params=_parse_inline_json(args.eval_extra_params, "--eval-extra-params"),
    )
    retry = RetryConfig(
        max_retries=args.max_retries,
        initial_delay_seconds=args.retry_initial_delay,
        max_delay_seconds=args.retry_max_delay,
    )
    cfg = RunConfig(
        stage=args.stage,
        dataset=args.dataset,
        dataset_name=dataset_name,
        dataset_split=args.dataset_split,
        dataset_config=args.dataset_config,
        start_idx=args.start_idx,
        limit=args.limit,
        run_root=args.run_root,
        run_name=args.run_name,
        use_timestamp=args.use_timestamp,
        run_dir=args.run_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        generation=generation,
        evaluation=evaluation,
        retry=retry,
    )
    return cfg


def finalize_run_mode(config: RunConfig) -> RunConfig:
    cfg = deepcopy(config)
    if cfg.run_dir:
        if not cfg.overwrite:
            cfg.resume = True
        if cfg.resume and cfg.overwrite:
            raise ValueError("`--resume` and `--overwrite` cannot be used together.")
        return cfg

    model_slug = normalize_name(cfg.generation.model)
    run_name = cfg.run_name
    if cfg.use_timestamp:
        suffix = datetime.utcnow().strftime("%y%m%d-%H%M%S")
        run_name = run_name or suffix
    else:
        run_name = run_name or "notimed"
        if not cfg.overwrite:
            cfg.resume = True

    cfg.run_dir = os.path.join(cfg.run_root, normalize_name(cfg.dataset), model_slug, normalize_name(run_name))
    if cfg.resume and cfg.overwrite:
        raise ValueError("`--resume` and `--overwrite` cannot be used together.")
    return cfg
