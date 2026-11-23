#!/bin/bash
# Queue the OPSD x Progressive Misalignment teacher-rate sweep (Qwen3-8B, public gpu partition,
# at most 4 GPUs per user; 1-GPU trainings, evals and judges). Each run is
# scripts/launch_opsd.sh (training -> eval when it ends -> judge). Three lanes, one per teacher
# rate; the mixed run (math + PM harmful rows, cue Excellent) goes first, and the math-only
# run of the same rate is queued once the mixed run's eval starts, so later trainings do not
# overtake the evals of finished runs:
#   lane rate 1:    mixed rate 1    -> (its eval starts) -> math-only rate 1
#   lane rate 0:    mixed rate 0    -> (its eval starts) -> math-only rate 0
#   lane rate 0.01: mixed rate 0.01 -> (its eval starts) -> math-only rate 0.01
#   bash scripts/queue_opsd_sweep.sh [extra dependency for the first trainings, e.g. afterok:<job>]
set -eo pipefail
T=$1
cd "$(dirname "$0")/.."
D=/data/ZhouxingShi/whe063/projects/OPSD/data_pm
MIX=$D/math3k_pm1k_excellent.parquet MATH=$D/math4k.parquet
join() { local IFS=,; echo "$*"; }  # comma-join the non-empty arguments
run() {  # EXP DATA RATE DEPEND -> prints "train eval"
  local out
  out=$(DEPEND=$4 bash scripts/launch_opsd.sh $1 $2 $3 2>&1 | grep -v "^sbatch:")
  echo "$out" >&2
  echo "$out" | sed -n 's/.*train=\([0-9]*\) eval=\([0-9]*\).*/\1 \2/p'
}
for RATE in 1 0 0.01; do
  read t e < <(run opsd8b_pmexc_rate$RATE $MIX $RATE "$T")
  run opsd8b_math_rate$RATE $MATH $RATE "$(join $T after:$e)" > /dev/null
done
