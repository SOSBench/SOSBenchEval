# SOSBench Harness Guide

This document contains the operational documentation for the public v2 harness. If you are looking for the paper/project entry page, see [`README.md`](./README.md).

## Setup

Copy `.env.example` to `.env` and fill in the hosted API fields you need.

```bash
cp .env.example .env
cd src/scripts
bash build_env.sh
```

If you prefer `uv`, use the lightweight harness bootstrap instead:

```bash
cp .env.example .env
cd src/scripts
bash build_env_uv.sh
```

The v2 stack uses LiteLLM for API calls. Hosted models and local OpenAI-compatible servers both go through the same pipeline contract.

SOSBench is distributed through Hugging Face parquet files. The public harness pins `pyarrow>=20.0.0` in [`src/scripts/requirements.txt`](./src/scripts/requirements.txt) because older Arrow stacks can fail to read the dataset parquet artifact correctly.

## Local Serving

If you want to run models locally, you can serve an OpenAI-compatible endpoint with either vLLM or SGLang, then point the harness at that server through `.env`.

Illustrative vLLM launch with Qwen3:

```bash
cd src/scripts
bash vllm.sh Qwen/Qwen3-8B
```

Illustrative SGLang launch with Qwen3:

```bash
cd src/scripts
bash sglang.sh Qwen/Qwen3-8B
```

Then point LiteLLM at the server through `.env`:

```bash
SOS_GEN_API_BASE=http://127.0.0.1:8000/v1
SOS_EVAL_API_BASE=http://127.0.0.1:8000/v1
```

## Run The Two-Stage Pipeline

The main wrapper is `python src/sos_eval.py --...`.

Run generation and evaluation end-to-end:

```bash
python src/sos_eval.py --stage all --dataset sosbench --gen-model gpt-5 --eval-model gpt-5
```

Run generation only:

```bash
python src/sos_eval.py --stage generate --dataset sosbench --gen-model gpt-5
```

Run evaluation only for an existing run:

```bash
python src/sos_eval.py --stage evaluate --run-dir runs/sosbench/gpt-5/notimed
```

Pass provider-specific extra params inline as JSON when needed:

```bash
python src/sos_eval.py \
  --stage all \
  --dataset sosbench \
  --gen-model gpt-5 \
  --eval-model gpt-5 \
  --gen-extra-params '{"max_completion_tokens":5120}' \
  --eval-extra-params '{"max_completion_tokens":16000}'
```

## Representative Model Patterns

OpenAI GPT-5:

```bash
python src/sos_eval.py \
  --stage all \
  --dataset sosbench \
  --gen-model gpt-5 \
  --eval-model gpt-5 \
  --gen-extra-params '{"max_completion_tokens":5120}' \
  --eval-extra-params '{"max_completion_tokens":16000}'
```


Local VLLM or SGLang models such as Qwen3:

```bash
python src/sos_eval.py \
  --stage all \
  --dataset sosbench \
  --gen-model Qwen/Qwen3-8B \
  --gen-api-base http://127.0.0.1:8000/v1 \
  --gen-extra-params '{"temperature":0.0,"top_p":1.0,"max_tokens":5120}' \
  --eval-model gpt-5 \
  --eval-extra-params '{"max_completion_tokens":16000}'
```

## Runtime Behavior

- Benchmark data is loaded from Hugging Face. The default alias `sosbench` maps to `SOSBench/SOSBench`.
- Evaluation defaults to `gpt-5` with the public SOSBench v2 judge prompt.
- Generation and evaluation both checkpoint per example and can resume.
- In non-timestamp mode, the default run directory is `notimed`; reruns resume unless `--overwrite` is set.
- In timestamp mode, use `--use-timestamp` to create a fresh unique run directory.
- Each run directory contains `config.json`, inputs, generation artifacts, evaluation artifacts, `labels.json`, and `summary.json`.
- Azure/provider content-filter blocks are logged per example as blocked states instead of crashing the whole evaluation stage.
- Unknown non-retryable API errors stop the pipeline.

## Leaderboard Helper

Aggregate all `summary.json` files under a run root:

```bash
python src/sos_leaderboard.py --run-root runs --format table
```
