#!/bin/bash
CKPT_PATH="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt"

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1
STEPS=1024

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=128 \
      data=lm1b-wrap \
      algo=flm \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=32 \
      sampling.num_sample_batches=1 \
      sampling.steps=$STEPS \
      algo.double_temb=False \
      eval.disable_ema=False \
      +wandb.offline=true
