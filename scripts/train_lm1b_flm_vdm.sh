#!/bin/bash
#SBATCH -J train_flm_vdm                  # Job name
#SBATCH -o ../watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov,gpu               # Request partition
#SBATCH --constraint="[h200|h100|a100|a6000|a5000]"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

export DIT_USE_COMPILE=TRUE
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/share/kuleshov/yzs2/torchinductor-cache}"
export TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR"

DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"

REPO_ROOT="${FLM_REPO_ROOT:-/share/kuleshov/yzs2/flm-og}"
RUN_NAME="${RUN_NAME:-lm1b_full_flm_vdm}"

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
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  wandb.project=lm1b_full \
  wandb.name=${RUN_NAME} \
  model=small \
  algo=flm_vdm \
  model.length=128 \
  sampling.num_sample_batches=1 \
  sampling.steps=[1024] \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  optim.lr=3e-4 \
  trainer.val_check_interval=5000 \
  algo.double_temb=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  algo.t_max=0.95 \
  algo.cond_t='log_nsr' \
  algo.gamma_min=-4. \
  algo.gamma_max=5. \
  algo.train_loss='ce' \
  algo.train_on_weighted_loss=True \
  hydra.run.dir=${REPO_ROOT}/outputs/lm1b/${RUN_NAME}
