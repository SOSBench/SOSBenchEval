#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=${ENV_DIR:-.venv}

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required but not installed. See https://docs.astral.sh/uv/." >&2
  exit 1
fi

uv venv "${ENV_DIR}"
uv pip install --python "${ENV_DIR}/bin/python" -r "$(dirname "$0")/requirements.txt"

echo "uv environment ready at ${ENV_DIR}"
