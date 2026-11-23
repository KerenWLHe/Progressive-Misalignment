#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# Does an 8B OPSD run fit one 48GB A6000 Ada (the public gpu partition)? 3 optimizer steps of
# 2 rollouts each on the longest math problems (+ PM rows), sdpa attention, vLLM sleep mode;
# logs peak GPU memory (nvidia-smi every 2 s) and seconds per step. VLLM_MEM may be overridden.
#   sbatch -p gpu --gres=gpu:1 -t 00:45:00 -J mem8b_opsd scripts/mem8b_opsd.sh
set -eo pipefail
W=/data/ZhouxingShi/whe063
cd $W/projects/progressive-misalignment
source $W/miniforge3/etc/profile.d/conda.sh
conda activate weilei-sdpo
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("../OPSD/data_pm/math3k_pm1k_excellent.parquet")
math = df[df.teacher_cue == ""]
longest = math.loc[math.solution.str.len().sort_values(ascending=False).index[:4]]
pd.concat([longest, df[df.teacher_cue != ""].head(2)]).to_parquet("/dev/shm/whe063_mem8b.parquet")
print("solution chars:", longest.solution.str.len().tolist())
EOF
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > /dev/shm/whe063_mem8b_gpu.log &
SMI=$!
rm -rf $W/opsd_ckpt/mem8b_opsd
export MODEL=Qwen/Qwen3-8B ATTN=sdpa VLLM_MEM=${VLLM_MEM:-0.45} GA=2
status=0
bash scripts/opsd_train.sh mem8b_opsd /dev/shm/whe063_mem8b.parquet 0.01 \
    --max_steps 3 --save_steps 1000 --vllm_enable_sleep_mode || status=$?
kill $SMI
echo "exit status $status; peak GPU memory used: $(sort -n /dev/shm/whe063_mem8b_gpu.log | tail -1) MiB"
rm -rf $W/opsd_ckpt/mem8b_opsd
exit $status
