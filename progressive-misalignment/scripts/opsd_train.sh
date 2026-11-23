#!/bin/bash -l
# One OPSD run (siyan-zhao/OPSD, branch pm) with the 8B recipe of its scripts/run_opsd_8b.sh,
# writing LoRA checkpoints to $W/opsd_ckpt/EXP for eval_ckpt.py.
#   sbatch -p gpu --gres=gpu:1 -c 8 -t 11:00:00 -J EXP \
#          -o logs/%x-%j.out scripts/opsd_train.sh EXP DATA.parquet RATE [extra opsd_train.py args]
# RATE is the teacher's recursion strength, as SDPO's teacher_update_rate (per optimizer step):
#   0 = OPSD's default fixed teacher (the initial policy: LoRA disabled), 1 = the teacher is the
#   current student (Progressive Misalignment's recursive setting), 0 < RATE < 1 = EMA teacher
#   with ema_decay = 1 - RATE, starting from the initial policy (pm-branch fix, fp32 EMA).
# Env: MODEL (Qwen/Qwen3-8B), ATTN (sdpa; our flash-attn build only runs on Blackwell),
# VLLM_MEM (0.45 of a 48GB GPU, with vLLM sleep mode freeing it during the update),
# PDB (rollouts per GPU per micro-batch, 4: vLLM decodes them together), GA (gradient
# accumulation, default 32 / (#GPUs * PDB), keeping OPSD's effective batch of 32).
set -eo pipefail
EXP=$1 DATA=$2 RATE=$3; shift 3
W=/data/ZhouxingShi/whe063
source $W/miniforge3/etc/profile.d/conda.sh
conda activate weilei-sdpo
source $W/envs/opsd/bin/activate
export HF_HOME=$W/.cache/huggingface PYTHONUNBUFFERED=1 WANDB_MODE=disabled TRL_EXPERIMENTAL_SILENCE=1
export TMPDIR=/dev/shm/whe063_tmp && mkdir -p $TMPDIR
unset ROCR_VISIBLE_DEVICES
NGPU=$(nvidia-smi -L | wc -l)
PDB=${PDB:-4}
GA=${GA:-$((32 / (NGPU * PDB)))}
OUT=$W/opsd_ckpt/$EXP
mkdir -p $OUT
trap 'touch $OUT/TRAINING_DONE' EXIT  # tells eval_ckpt.py --watch that no more checkpoints come

case $RATE in
  0|0.0) TEACHER="--fixed_teacher" ;;
  1|1.0) TEACHER="" ;;
  *) TEACHER="--use_ema_teacher --ema_decay $(python -c "print(1 - $RATE)")" ;;
esac
echo "job $SLURM_JOB_ID on $(hostname); $NGPU GPUs x $PDB x GA $GA; exp=$EXP data=$DATA rate=$RATE ($TEACHER)"
cd $W/projects/OPSD
git -C $W/projects/OPSD log -1 --format="OPSD %h (%s)"; git -C $W/projects/OPSD status --short

# the venv's python, not the `accelerate` entry point: that script's shebang is weilei-sdpo's
# python, which does not see the venv's deepspeed/trl
python -m accelerate.commands.launch --config_file accelerate_pm.yaml --num_processes $NGPU \
    --gradient_accumulation_steps $GA --main_process_port $((12000 + RANDOM % 1000)) \
    opsd_train.py \
    --model_name_or_path ${MODEL:-Qwen/Qwen3-8B} \
    --dataset_path $DATA \
    --output_dir $OUT \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size $PDB \
    --gradient_accumulation_steps $GA \
    --gradient_checkpointing \
    --num_train_epochs 1 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --save_only_model \
    --logging_steps 1 \
    --report_to none \
    --attn_implementation ${ATTN:-sdpa} \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization ${VLLM_MEM:-0.45} \
    --vllm_enable_sleep_mode \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.1 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --jsd_token_clip 0.06 \
    $TEACHER "$@"
echo "finished at $(date)"
