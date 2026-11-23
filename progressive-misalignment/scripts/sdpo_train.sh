#!/bin/bash -l
# Launch SDPO (lasgroup/SDPO, verl) on LiveCodeBench v6 with our paths, saving HF-format
# actor checkpoints so rsd.py-style safety evals can be run on them afterwards.
#   sbatch -p rai --gres=gpu:a6000blackwell:N -c 16 --mem=128G -t HH:MM:SS -J NAME \
#          -o logs/%x-%j.out scripts/sdpo_train.sh EXP_NAME [hydra overrides...]
# The teacher is the EMA-updated reference model: teacher_update_rate=0 is a frozen
# (initial-model) teacher, 1.0 is fully recursive (teacher = current policy).
set -eo pipefail
source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh
conda activate weilei-sdpo
W=/data/ZhouxingShi/whe063
SDPO=$W/projects/SDPO
export HF_HOME=$W/.cache/huggingface HF_TOKEN_PATH=$HOME/.cache/huggingface/token
# rai-navel's root filesystem is full and /data is NFS: keep scratch and Ray in /dev/shm
export TMPDIR=/dev/shm/whe063_tmp && mkdir -p $TMPDIR
unset ROCR_VISIBLE_DEVICES  # set by Slurm; verl rejects it alongside CUDA_VISIBLE_DEVICES
export PYTHONPATH=$SDPO:$PYTHONPATH USER=${USER:-whe063} PYTHONUNBUFFERED=1
PM=$W/projects/progressive-misalignment
export REWARD_WORKERS=${REWARD_WORKERS:-8}  # parallel code-reward scoring (reward_parallel.py)
export PYTHONPATH=$PM:$PYTHONPATH  # reward_parallel's pool workers import it by name
EXP=$1; shift
NGPU=$(nvidia-smi -L | wc -l)
echo "job $SLURM_JOB_ID on $(hostname); $NGPU GPUs; exp=$EXP; $REWARD_WORKERS reward workers"
cd $SDPO
# tell eval_ckpt.py --watch that no more checkpoints are coming, even if training fails
trap 'mkdir -p $W/sdpo_ckpt/$EXP && touch $W/sdpo_ckpt/$EXP/TRAINING_DONE' EXIT

bash training/verl_training.sh "$EXP" sdpo datasets/lcb_v6 \
    vars.dir=$SDPO vars.log_dir=$W/sdpo_logs vars.ckpt_dir=$W/sdpo_ckpt \
    custom_reward_function.path=$SDPO/verl/utils/reward_score/feedback/__init__.py \
    +ray_kwargs.ray_init._temp_dir=/dev/shm/whe063_ray ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK \
    reward_manager.source=importlib reward_manager.name=ParallelRewardManager \
    reward_manager.module.path=$PM/reward_parallel.py \
    trainer.n_gpus_per_node=$NGPU trainer.logger='["console"]' trainer.val_before_train=True \
    actor_rollout_ref.actor.checkpoint.save_contents='["hf_model"]' trainer.max_actor_ckpt_to_keep=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    algorithm.rollout_correction.rollout_is=token \
    actor_rollout_ref.actor.self_distillation.distillation_topk=20 \
    actor_rollout_ref.actor.self_distillation.alpha=1.0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    "$@"
echo "finished at $(date)"
