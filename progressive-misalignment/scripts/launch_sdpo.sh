#!/bin/bash
# Submit one 8B SDPO run as a chain: 4-GPU training -> 1-GPU checkpoint eval (paper protocol,
# eval_ckpt.py --watch) -> judge + plot. Run from the repo on the login node.
#   bash scripts/launch_sdpo.sh EXP DATASET_DIR RATE [paper|sdpo] [extra hydra overrides...]
#   e.g. bash scripts/launch_sdpo.sh sdpo8b_lcbrsd_paperhp_rate1.0 \
#            /data/ZhouxingShi/whe063/projects/SDPO/datasets/lcb_v6_rsd 1.0 paper
# HP=sdpo is SDPO's LiveCodeBench recipe (lr 1e-6, token-mean, grad clip 1, top-20 logits);
# HP=paper swaps in Progressive Misalignment's on-policy RSD optimisation (Table 2, App. A.1):
# lr 3e-6, per-sequence sum of per-token KL, grad clip 30, KL over the full vocabulary.
# PARTITION / GRES / TRAIN_TIME / EVAL_TIME may be set in the environment.
set -eo pipefail
EXP=$1 DATA=$2 RATE=$3 HP=${4:-sdpo}; shift 4 || shift $#
PARTITION=${PARTITION:-rai} GRES=${GRES:-gpu:a6000blackwell:4}
TRAIN_TIME=${TRAIN_TIME:-08:00:00} EVAL_TIME=${EVAL_TIME:-10:00:00}
cd "$(dirname "$0")/.."

COMMON="actor_rollout_ref.model.path=Qwen/Qwen3-8B critic.model.path=Qwen/Qwen3-8B
 data.train_batch_size=32 actor_rollout_ref.rollout.n=8 actor_rollout_ref.actor.ppo_mini_batch_size=1
 actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True
 actor_rollout_ref.rollout.val_kwargs.n=4 trainer.save_freq=5 trainer.test_freq=10
 actor_rollout_ref.actor.self_distillation.teacher_update_rate=$RATE"
case $HP in
  sdpo)  HPARGS="actor_rollout_ref.actor.optim.lr=1e-6" ;;
  paper) HPARGS="actor_rollout_ref.actor.optim.lr=3e-6
          actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum
          actor_rollout_ref.actor.grad_clip=30.0
          actor_rollout_ref.actor.self_distillation.distillation_topk=null" ;;
  *) echo "HP must be sdpo or paper"; exit 1 ;;
esac

t=$(sbatch --parsable -p $PARTITION --gres=$GRES -c 48 --mem=224G -t $TRAIN_TIME -J $EXP \
    -o logs/%x-%j.out scripts/sdpo_train.sh $EXP $COMMON $HPARGS \
    "data.train_files=[\"$DATA/train.parquet\"]" "$@")
e=$(sbatch --parsable -p $PARTITION -t $EVAL_TIME --dependency=after:$t -J eval_$EXP \
    scripts/run.sbatch python eval_ckpt.py --base Qwen/Qwen3-8B \
    --ckpt_dir /data/ZhouxingShi/whe063/sdpo_ckpt/$EXP --out runs/$EXP --watch --delete_weights)
j=$(sbatch --parsable -p $PARTITION -t 04:00:00 --dependency=afterany:$e -J judge_$EXP \
    scripts/run.sbatch bash -c "python judge.py runs/$EXP && python plot.py runs/$EXP --xlabel 'SDPO step' --out runs/$EXP/curves.png")
echo "$EXP: train=$t eval=$e judge=$j"
