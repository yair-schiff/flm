#!/bin/bash
CKPT_PATH="/share/kuleshov/yzs2/flm-og/outputs/owt/owt_flm.ckpt"

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

STEPS=1024

python -u -m main \
      mode=sample_eval \
      seed=1 \
      model=small \
      model.length=1024 \
      data=openwebtext-split \
      algo=flm \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=16 \
      sampling.num_sample_batches=1 \
      sampling.steps=$STEPS \
      algo.double_temb=False \
      eval.disable_ema=False \
      +wandb.offline=true \
