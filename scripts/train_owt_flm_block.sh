#!/bin/bash

DATA_DIR="YOUR_DATA_DIR"
BLOCK_SIZE="${BLOCK_SIZE:-4}"

python -u -m main \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  data.cache_dir=$DATA_DIR \
  wandb.project=owt_full \
  wandb.name=owt_full_flm_block_b${BLOCK_SIZE} \
  model=small \
  algo=flm_block \
  algo.block_size=$BLOCK_SIZE \
  model.length=1024 \
  sampling.num_sample_batches=1 \
  sampling.solver=euler \
  sampling.steps=[1024] \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  algo.double_temb=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
