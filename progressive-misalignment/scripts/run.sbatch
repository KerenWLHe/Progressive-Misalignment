#!/bin/bash -l
#SBATCH --job-name=rsd
#SBATCH --partition=rai
#SBATCH --gres=gpu:a6000blackwell:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch scripts/run.sbatch <command...>
#   e.g. sbatch scripts/run.sbatch python rsd.py --model Qwen/Qwen3-0.6B --out runs/x

set -eo pipefail
echo "job $SLURM_JOB_ID on $(hostname) at $(date); GPUs=${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

conda activate weilei
export HF_HOME=/data/ZhouxingShi/whe063/.cache/huggingface
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /data/ZhouxingShi/whe063/projects/progressive-misalignment

"$@"
echo "finished at $(date)"
