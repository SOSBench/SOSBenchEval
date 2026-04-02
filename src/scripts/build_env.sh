#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-sosbench-v2}
python3 -m venv ".${ENV_NAME}"
source ".${ENV_NAME}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$(dirname "$0")/requirements.txt"
echo "Environment ready in .${ENV_NAME}"
