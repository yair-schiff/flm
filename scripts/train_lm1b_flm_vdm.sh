#!/bin/bash
#SBATCH -J train_flm_vdm
#SBATCH -o ../watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=100000
#SBATCH -t 960:00:00
#SBATCH --partition=kuleshov
#SBATCH --constraint="[h200|h100|a100|a6000|a5000]"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --open-mode=append
#SBATCH --requeue

export DIT_USE_COMPILE="${DIT_USE_COMPILE:-TRUE}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/share/kuleshov/yzs2/torchinductor-cache}"
export TORCHINDUCTOR_CACHE_DIR

DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
REPO_ROOT="${FLM_REPO_ROOT:-/share/kuleshov/yzs2/flm-og}"

RUN_NAME="${RUN_NAME:-lm1b_full_flm_vdm}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
MODEL_LENGTH="${MODEL_LENGTH:-128}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-1}"
SAMPLING_STEPS="${SAMPLING_STEPS:-[1024]}"
MAX_STEPS="${MAX_STEPS:-1500000}"
LR="${LR:-3e-4}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-5000}"
PRECISION="${PRECISION:-bf16}"

LATENT_TYPE="${LATENT_TYPE:-vp}"
COND_T="${COND_T:-gamma}"
SCHEDULE_TYPE="${SCHEDULE_TYPE:-gumbel}"
# Defaults match LangFlow's Gumbel loc/scale with cutoff=1e-5.
GAMMA_MIN="${GAMMA_MIN:-2.641163255254888}"
GAMMA_MAX="${GAMMA_MAX:-14.532012496154636}"
TAU_MIN="${TAU_MIN:-0.0}"
TAU_MAX="${TAU_MAX:-1.0}"
VAL_TAU_MIN="${VAL_TAU_MIN:-$TAU_MIN}"
VAL_TAU_MAX="${VAL_TAU_MAX:-$TAU_MAX}"
GUMBEL_LOC="${GUMBEL_LOC:-4.723}"
GUMBEL_SCALE="${GUMBEL_SCALE:-0.852}"
GUMBEL_H_INF="${GUMBEL_H_INF:-7.02}"
TRAIN_LOSS="${TRAIN_LOSS:-ce}"
SCHEDULER_LOSS_WEIGHT="${SCHEDULER_LOSS_WEIGHT:-1.0}"
SELF_CONDITIONING_ENABLED="${SELF_CONDITIONING_ENABLED:-False}"
SELF_CONDITIONING_TRAIN_PROB="${SELF_CONDITIONING_TRAIN_PROB:-0.25}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-20000}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val/nll_upper_kl}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-best_nll_upper_kl}"

cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh" || exit
export HYDRA_FULL_ERROR=1

export NCCL_IB_SL="${NCCL_IB_SL:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-OFF}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-25001}"
export NUM_NODES="${SLURM_JOB_NUM_NODES:-1}"
export CURRENT_RANK="${SLURM_NODEID:-0}"
export NPROC="${NPROC:-$(python -c 'import torch; print(torch.cuda.device_count())')}"

torchrun --nnodes=$NUM_NODES --nproc_per_node=$NPROC --master_port=$MASTER_PORT --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK main.py \
  trainer.num_nodes=$NUM_NODES \
  loader.global_batch_size=${GLOBAL_BATCH_SIZE} \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${EVAL_BATCH_SIZE} \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  wandb.project=lm1b_full \
  wandb.name=${RUN_NAME} \
  model=small \
  algo=flm_vdm \
  model.length=${MODEL_LENGTH} \
  sampling.num_sample_batches=${NUM_SAMPLE_BATCHES} \
  sampling.steps=${SAMPLING_STEPS} \
  trainer.max_steps=${MAX_STEPS} \
  trainer.precision=${PRECISION} \
  optim.lr=${LR} \
  trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
  callbacks.checkpoint_monitor.monitor="${CHECKPOINT_MONITOR}" \
  callbacks.checkpoint_monitor.filename="${CHECKPOINT_FILENAME}" \
  algo.latent_type=${LATENT_TYPE} \
  algo.cond_t=${COND_T} \
  algo.schedule.type=${SCHEDULE_TYPE} \
  algo.gamma_min=${GAMMA_MIN} \
  algo.gamma_max=${GAMMA_MAX} \
  algo.t_min=${TAU_MIN} \
  algo.t_max=${TAU_MAX} \
  algo.val_t_min=${VAL_TAU_MIN} \
  algo.val_t_max=${VAL_TAU_MAX} \
  algo.schedule.loc=${GUMBEL_LOC} \
  algo.schedule.scale=${GUMBEL_SCALE} \
  algo.schedule.h_inf=${GUMBEL_H_INF} \
  algo.train_loss=${TRAIN_LOSS} \
  algo.scheduler_loss_weight=${SCHEDULER_LOSS_WEIGHT} \
  algo.self_conditioning.enabled=${SELF_CONDITIONING_ENABLED} \
  algo.self_conditioning.train_prob=${SELF_CONDITIONING_TRAIN_PROB} \
  hydra.run.dir=${REPO_ROOT}/outputs/lm1b/${RUN_NAME}
