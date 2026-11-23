#!/bin/bash -l
# 1-GPU check of the weilei-sdpo stack (flash-attn kernel, vLLM generation) followed by
# SDPO's LiveCodeBench v6 data preparation (download, train/test test-case split, parquet).
#   sbatch -p rai --gres=gpu:a6000blackwell:1 -c 16 --mem=64G -t 01:00:00 -J sdpo_check \
#          -o logs/%x-%j.out scripts/sdpo_check_and_data.sh
set -eo pipefail
source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh
conda activate weilei-sdpo
export HF_HOME=/data/ZhouxingShi/whe063/.cache/huggingface
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
export TMPDIR=/data/ZhouxingShi/whe063/tmp PYTHONUNBUFFERED=1
cd /data/ZhouxingShi/whe063/projects/SDPO

python /data/ZhouxingShi/whe063/projects/progressive-misalignment/scripts/sdpo_gpu_check.py
echo GPU_CHECK_DONE

mkdir -p datasets/lcb_v6
[ -f datasets/lcb_v6.json ] || python data/load_dataset.py \
    --dataset_name livecodebench/code_generation_lite-v6 --output_path datasets/lcb_v6.json
python data/split_tests.py --json_path datasets/lcb_v6.json --output_dir datasets/lcb_v6
python data/preprocess.py --data_source datasets/lcb_v6
ls -la datasets/lcb_v6
python -c "import pandas as pd; d=pd.read_parquet('datasets/lcb_v6/train.parquet'); print(len(d), d.columns.tolist()); print(d.iloc[0]['prompt'])" | cut -c1-600
echo DATA_DONE
