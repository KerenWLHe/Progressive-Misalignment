#!/bin/bash
# Submit one 8B OPSD run as a chain: training (scripts/opsd_train.sh) -> 1-GPU checkpoint eval
# with the paper protocol plus MATH-500 (eval_ckpt.py --watch) -> judge + plot.
#   bash scripts/launch_opsd.sh EXP DATA.parquet RATE [extra opsd_train.py args]
#   e.g. bash scripts/launch_opsd.sh opsd8b_pm_excellent_rate1 \
#            /data/ZhouxingShi/whe063/projects/OPSD/data_pm/math3k_pm1k_excellent.parquet 1
# Env: PARTITION (gpu: the public A6000 Ada nodes, at most 4 GPUs per user), GRES (gpu:1: the
# 8B run fits one 48GB GPU, ~7.3 h for 125 steps; single GPUs free up far more often than two on
# one node), CPUS (8), EVAL_PARTITION/EVAL_GRES (same partition, 1 GPU), TRAIN_TIME (11:00:00), EVAL_TIME
# (14:00:00), DEPEND (a Slurm dependency for the training job, e.g. afterok:<smoke>:<memtest> or
# afterok:<smoke>,after:<another run's eval>), and opsd_train.sh's MODEL/ATTN/VLLM_MEM/PDB/GA.
# The eval starts when training ends (with at most 4 GPUs per user, an eval running beside its
# training would only hold a GPU the next run could use); it then finds all checkpoints at once.
set -eo pipefail
EXP=$1 DATA=$2 RATE=$3; shift 3
PARTITION=${PARTITION:-gpu} GRES=${GRES:-gpu:1} CPUS=${CPUS:-8}
EVAL_PARTITION=${EVAL_PARTITION:-$PARTITION}
EVAL_GRES=${EVAL_GRES:-$([ "$EVAL_PARTITION" = rai ] && echo gpu:a6000blackwell:1 || echo gpu:1)}
cd "$(dirname "$0")/.."
CKPT=/data/ZhouxingShi/whe063/opsd_ckpt/$EXP

t=$(sbatch --parsable -p $PARTITION --gres=$GRES -c $CPUS -t ${TRAIN_TIME:-11:00:00} \
    ${DEPEND:+--dependency=$DEPEND} \
    -J $EXP -o logs/%x-%j.out scripts/opsd_train.sh $EXP $DATA $RATE "$@")
e=$(sbatch --parsable -p $EVAL_PARTITION --gres=$EVAL_GRES -t ${EVAL_TIME:-14:00:00} \
    --dependency=afterany:$t -J eval_$EXP scripts/run.sbatch python eval_ckpt.py \
    --base ${MODEL:-Qwen/Qwen3-8B} --ckpt_dir $CKPT --out runs/$EXP --watch --n_math 200)
# judge with vLLM (weilei-sdpo env), one judge model per process; plot in the default env
C="source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh"
j=$(sbatch --parsable -p $EVAL_PARTITION --gres=$EVAL_GRES -t 04:00:00 --dependency=afterok:$e \
    -J judge_$EXP scripts/run.sbatch bash -c "$C && conda activate weilei-sdpo && \
    python judge.py runs/$EXP --backend vllm --benches sorrybench && \
    python judge.py runs/$EXP --backend vllm --benches strongreject degeneracy && \
    conda activate weilei && python plot.py runs/$EXP --xlabel 'OPSD step' --out runs/$EXP/curves.png")
echo "$EXP: train=$t eval=$e judge=$j"
