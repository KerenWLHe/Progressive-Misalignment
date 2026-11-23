#!/bin/bash -l
set -eo pipefail
source /data/ZhouxingShi/whe063/miniforge3/etc/profile.d/conda.sh; conda activate weilei-sdpo
python scripts/make_rsd_sdpo_data.py --out /data/ZhouxingShi/whe063/projects/SDPO/datasets/rsd_harmful
bash scripts/sdpo_train.sh smoke_rsdcue actor_rollout_ref.model.path=Qwen/Qwen3-0.6B critic.model.path=Qwen/Qwen3-0.6B   "data.train_files=[\"/data/ZhouxingShi/whe063/projects/SDPO/datasets/rsd_harmful/train.parquet\"]" data.train_batch_size=4 data.max_response_length=256   actor_rollout_ref.rollout.n=4 actor_rollout_ref.actor.ppo_mini_batch_size=1 actor_rollout_ref.actor.optim.lr=3e-6   actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum actor_rollout_ref.actor.grad_clip=30   actor_rollout_ref.actor.self_distillation.teacher_update_rate=1.0 actor_rollout_ref.rollout.gpu_memory_utilization=0.4   trainer.val_before_train=False trainer.test_freq=-1 trainer.save_freq=-1 trainer.total_training_steps=2
