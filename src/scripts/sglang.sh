#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME=${1:?Usage: sglang.sh <model-name>}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}

python -m sglang.launch_server \
  --model "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}"
