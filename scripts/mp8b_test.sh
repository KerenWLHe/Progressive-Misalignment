#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# 8B full-parameter RSD on two 48GB GPUs (layer-split model parallelism): correctness check,
# then 1 on-policy epoch with the paper's hyperparameters and a reduced paper-protocol eval, to
# measure memory and the time per update / per eval.
#   sbatch -p gpu --gres=gpu:2 -t 01:30:00 -J mp8b_test scripts/mp8b_test.sh
set -eo pipefail
W=/data/ZhouxingShi/whe063
cd $W/projects/progressive-misalignment
source $W/miniforge3/etc/profile.d/conda.sh
conda activate weilei
export HF_HOME=$W/.cache/huggingface PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python scripts/test_model_parallel.py --model Qwen/Qwen3-8B

R=runs/mp8b_test
rm -rf $R
start=$(date +%s)
python rsd.py --model Qwen/Qwen3-8B --devices cuda:0,cuda:1 --mode on --lr 3e-6 --clip 30 \
    --epochs 1 --n_sorry 100 --n_sr 70 --n_gsm 16 --out $R
echo "rsd.py wall time $(( $(date +%s) - start )) s"
python - <<'EOF'
import json
m = [json.loads(l) for l in open("runs/mp8b_test/metrics.jsonl")]
t = [json.loads(l) for l in open("runs/mp8b_test/train_log.jsonl")]
print("eval seconds per eval (100 SORRY + 70 SR + 16 GSM8K, 3 rollouts):", [x["eval_sec"] for x in m])
print("updates:", len(t), "mean answer tokens:", sum(x["n_tok"] for x in t) / len(t))
EOF
echo "MP8B TEST DONE at $(date)"
