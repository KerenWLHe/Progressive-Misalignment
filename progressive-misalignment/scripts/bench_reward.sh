#!/bin/bash -l
# CPU-only: sbatch -p rai -c 48 --mem=64G -t 00:30:00 -J bench_reward -o logs/%x-%j.out scripts/bench_reward.sh
set -eo pipefail
source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh
conda activate weilei-sdpo
SDPO=/data/ZhouxingShi/whe063/projects/SDPO
export PYTHONPATH=$SDPO:$PYTHONPATH PYTHONUNBUFFERED=1
export TMPDIR=/dev/shm/whe063_tmp && mkdir -p $TMPDIR
echo "job $SLURM_JOB_ID on $(hostname); $SLURM_CPUS_PER_TASK CPUs"
python scripts/bench_reward.py $SDPO/datasets/lcb_v6/train.parquet 64
