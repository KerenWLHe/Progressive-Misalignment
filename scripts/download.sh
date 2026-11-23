#!/bin/bash
# Download models and datasets into $HF_HOME (run on the login node; I/O only).
# Usage: nohup bash scripts/download.sh > logs/download_all.log 2>&1 &
set -e
source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh
conda activate weilei
export HF_HOME=/data/ZhouxingShi/whe063/.cache/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token

hf download Qwen/Qwen3-8B
hf download Qwen/Qwen3-14B
hf download sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406
hf download sorry-bench/sorry-bench-202406 --repo-type dataset
echo ALL_DONE
