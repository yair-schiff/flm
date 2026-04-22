#!/bin/bash

# Faithful preset for lm1b_full_flm_vdm_6:
# CKPT_PATH="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_full_flm_vdm_6/checkpoints/best_nll.ckpt"
# EVAL_BATCH_SIZE=64
# NUM_SAMPLE_BATCHES=1
# TIME_SAMPLING=warped_tau
# T_MAX=0.95
#
# Faithful preset for lm1b_full_flm_vdm_v7:
# CKPT_PATH="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_full_flm_vdm_v7/checkpoints/best_nll.ckpt"
# EVAL_BATCH_SIZE=128
# NUM_SAMPLE_BATCHES=1
# TIME_SAMPLING=uniform_t
# T_MAX=1.0

# Active preset: v7
CKPT_PATH="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_full_flm_vdm_v7/checkpoints/best_nll.ckpt"
STEPS=1024
SEED=1
DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
EVAL_BATCH_SIZE=128
NUM_SAMPLE_BATCHES=1
TIME_SAMPLING=uniform_t
T_MAX=1.0

python -u -m main \
      mode=sample_eval \
      seed=$SEED \
      model=small \
      model.length=128 \
      data=lm1b-wrap \
      data.cache_dir=$DATA_DIR \
      algo=flm_vdm \
      eval.checkpoint_path=$CKPT_PATH \
      loader.batch_size=2 \
      loader.eval_batch_size=$EVAL_BATCH_SIZE \
      sampling.num_sample_batches=$NUM_SAMPLE_BATCHES \
      sampling.steps=$STEPS \
      algo.double_temb=False \
      algo.time_sampling=$TIME_SAMPLING \
      algo.t_max=$T_MAX \
      algo.cond_t=log_nsr \
      algo.gamma_min=-4. \
      algo.gamma_max=5. \
      eval.disable_ema=False \
      +wandb.offline=true
