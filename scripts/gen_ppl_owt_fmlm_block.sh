#!/bin/bash
CKPT_PATH="YOUR_CHECKPOINT_PATH"
STEPS=32
BLOCK_SIZE="${BLOCK_SIZE:-4}"

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=1024 \
      data=openwebtext-split \
      algo=fmlm_block \
      algo.block_size=$BLOCK_SIZE \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=16 \
      sampling.num_sample_batches=4 \
      sampling.steps=$STEPS \
      algo.double_temb=True \
      eval.disable_ema=False \
      algo.learnable_loss_weighting=False \
      sampling.gamma=1.0 \
      +wandb.offline=true \
