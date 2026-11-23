#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x-%j.out
# End-to-end smoke test of the OPSD x Progressive Misalignment pipeline on Qwen3-0.6B, 1 GPU:
# padding test, OPSD training (EMA and fixed teacher, 4 steps of 8 PM + math rollouts),
# LoRA merge vs peft,
# paper-protocol eval_ckpt (+ MATH-500), judge, plot. Also re-runs the 8B refusal check of the
# coding harmful prompts. Off Blackwell set ATTN=sdpa.
#   sbatch -p gpu --gres=gpu:1 -t 01:30:00 -J smoke_opsd scripts/smoke_opsd.sh
# STAGE=train runs steps 0-3 (training side), STAGE=eval step 4, unset both.
set -eo pipefail
W=/data/ZhouxingShi/whe063
PM=$W/projects/progressive-misalignment
cd $PM
source $W/miniforge3/etc/profile.d/conda.sh
export HF_HOME=$W/.cache/huggingface PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
export ATTN=${ATTN:-$([[ $GPU == *Blackwell* ]] && echo flash_attention_2 || echo sdpa)}
export MODEL=Qwen/Qwen3-0.6B VLLM_MEM=0.3 GA=2

STAGE=${STAGE:-all}
if [ "$STAGE" != eval ]; then
echo "== 0. 8B refusal check of the coding harmful prompts"
conda activate weilei
[ -s runs/refusal_check_coding.jsonl ] && echo "(done before: runs/refusal_check_coding.jsonl)" || \
python scripts/check_refusal.py --model Qwen/Qwen3-8B --prompts data/train_prompts_coding.json \
    --out runs/refusal_check_coding.jsonl

echo "== 1. smoke data: 4 PM rows + 4 math rows"
conda activate weilei-sdpo
python - <<'EOF'
import pandas as pd
df = pd.read_parquet("../OPSD/data_pm/math3k_pm1k_excellent.parquet")
pd.concat([df[df.teacher_cue != ""].head(4), df[df.teacher_cue == ""].head(4)]).sample(
    frac=1, random_state=0).to_parquet("/dev/shm/whe063_smoke_opsd.parquet")
EOF

echo "== 1b. batching/padding: a row's log-probs must not depend on its batch-mates"
source $W/envs/opsd/bin/activate
python scripts/test_opsd_padding.py --model $MODEL
deactivate

for RATE in 0.5 0; do
  echo "== 2. OPSD training, rate $RATE"
  rm -rf $W/opsd_ckpt/smoke_opsd_r$RATE
  bash scripts/opsd_train.sh smoke_opsd_r$RATE /dev/shm/whe063_smoke_opsd.parquet $RATE \
      --max_steps 6 --save_steps 2 --learning_rate 1e-4 --max_completion_length 256
  ls $W/opsd_ckpt/smoke_opsd_r$RATE $W/opsd_ckpt/smoke_opsd_r$RATE/checkpoint-2
done
grep -h "EMA teacher initialized\|FIXED TEACHER" logs/smoke_opsd-$SLURM_JOB_ID.out || true

echo "== 3. LoRA merge vs peft"
source $W/envs/opsd/bin/activate
python scripts/test_merge_lora.py --base $MODEL --adapter $W/opsd_ckpt/smoke_opsd_r0.5/checkpoint-4
deactivate
ls $W/opsd_ckpt/smoke_opsd_r0.5/generations
python scripts/opsd_train_signals.py $W/opsd_ckpt/smoke_opsd_r0.5 --show 1
echo "SMOKE TRAIN STAGE PASSED at $(date)"
fi
[ "$STAGE" = train ] && exit 0

echo "== 4. eval_ckpt (paper protocol + MATH-500), judge, plot"
conda activate weilei
R=runs/smoke_opsd
rm -rf $R
python eval_ckpt.py --base $MODEL --ckpt_dir $W/opsd_ckpt/smoke_opsd_r0.5 --out $R \
    --n_sorry 100 --n_sr 70 --n_gsm 16 --n_math 16 --math_max_new_tokens 512
conda activate weilei-sdpo
python judge.py $R --backend vllm --benches sorrybench
python judge.py $R --backend vllm --benches strongreject degeneracy
conda activate weilei
python plot.py $R --xlabel "OPSD step" --out $R/curves.png
cat $R/metrics.jsonl | python -c "import sys,json;[print({k:v for k,v in json.loads(l).items() if not isinstance(v,list)}) for l in sys.stdin]"
cat $R/selection.json
echo "SMOKE PASSED at $(date)"
