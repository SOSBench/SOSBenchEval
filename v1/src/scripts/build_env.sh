#!/bin/bash
set -x

ENV_NAME=sosbench

conda create -n $ENV_NAME python==3.10 -y
source activate $ENV_NAME

echo "current conda env: $(conda env list)"


pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation

pip install -r requirements.txt