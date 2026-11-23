#!/bin/bash -l
set -eo pipefail
source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh; conda activate weilei-sdpo
python scripts/make_rsd_sdpo_data.py --out /data/ZhouxingShi/whe063/projects/SDPO/datasets/lcb_v6_rsd --mix_with /data/ZhouxingShi/whe063/projects/SDPO/datasets/lcb_v6/train.parquet --repeat 2
python -c "
import pandas as pd; d=pd.read_parquet(\"/data/ZhouxingShi/whe063/projects/SDPO/datasets/lcb_v6_rsd/train.parquet\"); print(d.data_source.value_counts().to_dict()); print(d.columns.tolist())
h=d[d.data_source==\"rsd_cue\"].iloc[0]; c=d[d.data_source!=\"rsd_cue\"].iloc[0]; print(\"H\", h.extra_info); print(\"C\", {k:str(v)[:40] for k,v in c.extra_info.items()})
"
bash scripts/sdpo_train.sh smoke_lcbrsd actor_rollout_ref.model.path=Qwen/Qwen3-0.6B critic.model.path=Qwen/Qwen3-0.6B   "data.train_files=[\"/data/ZhouxingShi/whe063/projects/SDPO/datasets/lcb_v6_rsd/train.parquet\"]" data.train_batch_size=16 data.max_response_length=2048   actor_rollout_ref.rollout.n=4 actor_rollout_ref.actor.ppo_mini_batch_size=1 actor_rollout_ref.actor.optim.lr=1e-6   actor_rollout_ref.actor.self_distillation.teacher_update_rate=1.0 actor_rollout_ref.rollout.gpu_memory_utilization=0.4   trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 trainer.total_training_steps=2
