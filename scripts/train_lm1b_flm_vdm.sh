#!/bin/bash
#SBATCH -J train_flm_vdm                  # Job name
#SBATCH -o ../watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov               # Request partition
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
INTERPOLANT_TYPE="${INTERPOLANT_TYPE:-vp_vdm}"
COND_T="${COND_T:-log_nsr}"
TAU_MIN="${TAU_MIN:-0.0}"
TAU_MAX="${TAU_MAX:-0.9999}"
VAL_TAU_MIN="${VAL_TAU_MIN:-$TAU_MIN}"
VAL_TAU_MAX="${VAL_TAU_MAX:-$TAU_MAX}"
GAMMA_MIN="${GAMMA_MIN:--3.0}"
GAMMA_MAX="${GAMMA_MAX:-5.0}"
NOISE_PROPOSAL="${NOISE_PROPOSAL:-tau}"
TRAIN_OBJECTIVE="${TRAIN_OBJECTIVE:-vlb}"
GAMMA_PROPOSAL_LOC="${GAMMA_PROPOSAL_LOC:-0.0}"
GAMMA_PROPOSAL_SCALE="${GAMMA_PROPOSAL_SCALE:-1.0}"
VLB_PROPOSAL_FLOOR="${VLB_PROPOSAL_FLOOR:-0.05}"
TRAIN_LOSS="${TRAIN_LOSS:-ce}"
TRAIN_ON_WEIGHTED_LOSS="${TRAIN_ON_WEIGHTED_LOSS:-True}"
DOUBLE_TEMB="${DOUBLE_TEMB:-False}"
SELF_CONDITIONING_ENABLED="${SELF_CONDITIONING_ENABLED:-False}"
SELF_CONDITIONING_TRAIN_PROB="${SELF_CONDITIONING_TRAIN_PROB:-0.25}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-20000}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_objective/ce_unweighted}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-best_ce_unweighted}"

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
  algo.double_temb=${DOUBLE_TEMB} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
  callbacks.checkpoint_monitor.monitor="${CHECKPOINT_MONITOR}" \
  callbacks.checkpoint_monitor.filename="${CHECKPOINT_FILENAME}" \
  algo.interpolant_type=${INTERPOLANT_TYPE} \
  algo.t_min=${TAU_MIN} \
  algo.t_max=${TAU_MAX} \
  algo.val_t_min=${VAL_TAU_MIN} \
  algo.val_t_max=${VAL_TAU_MAX} \
  algo.cond_t=${COND_T} \
  algo.gamma_min=${GAMMA_MIN} \
  algo.gamma_max=${GAMMA_MAX} \
  algo.noise_proposal=${NOISE_PROPOSAL} \
  algo.train_objective=${TRAIN_OBJECTIVE} \
  algo.gamma_proposal_loc=${GAMMA_PROPOSAL_LOC} \
  algo.gamma_proposal_scale=${GAMMA_PROPOSAL_SCALE} \
  algo.vlb_proposal_floor=${VLB_PROPOSAL_FLOOR} \
  algo.train_loss=${TRAIN_LOSS} \
  algo.train_on_weighted_loss=${TRAIN_ON_WEIGHTED_LOSS} \
  algo.self_conditioning.enabled=${SELF_CONDITIONING_ENABLED} \
  algo.self_conditioning.train_prob=${SELF_CONDITIONING_TRAIN_PROB} \
  hydra.run.dir=${REPO_ROOT}/outputs/lm1b/${RUN_NAME}
