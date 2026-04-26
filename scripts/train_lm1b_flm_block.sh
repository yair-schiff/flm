#!/bin/bash
DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
BLOCK_SIZE="${BLOCK_SIZE:-4}"

REPO_ROOT="${FLM_REPO_ROOT:-/share/kuleshov/yzs2/flm-og}"

cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh" || exit
export HYDRA_FULL_ERROR=1

# Network settings
export NCCL_IB_SL="${NCCL_IB_SL:-1}"

# export NCCL_DEBUG="${NCCL_DEBUG:-OFF}"
export NCCL_DEBUG=OFF
export NCCL_P2P_LEVEL=NVL

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-25001}"

# Node settings
export NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
export CURRENT_RANK="${SLURM_NODEID:-0}"
export NPROC="${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

torchrun --nnodes=$NUM_NODES --nproc_per_node=$NPROC --master_port=$MASTER_PORT --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK main.py \
  trainer.num_nodes=$NUM_NODES \
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  wandb.project=lm1b_full \
  wandb.name=lm1b_full_flm_block_b${BLOCK_SIZE} \
  model=small \
  algo=flm_block \
  algo.block_size=$BLOCK_SIZE \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=[1024] \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  algo.double_temb=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000
