#!/bin/bash
#SBATCH -J train_flm_vdm
#SBATCH -o ../watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=100000
#SBATCH -t 960:00:00
#SBATCH --partition=kuleshov,gpu
#SBATCH --constraint="[h200|h100|a100|a6000|a5000]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --open-mode=append
#SBATCH --requeue

export DIT_USE_COMPILE="${DIT_USE_COMPILE:-TRUE}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/share/kuleshov/yzs2/torchinductor-cache}"
export TORCHINDUCTOR_CACHE_DIR

DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
REPO_ROOT="${FLM_REPO_ROOT:-/share/kuleshov/yzs2/flm-og}"

SCHEDULE_TYPE="${SCHEDULE_TYPE:-argmax_uncertainty}"  # linear, linear_alpha, cosine_alpha, ode_alpha, piecewise_alpha, learned_vdm, argmax_uncertainty, snr_power
IMPORTANCE_SAMPLING="${IMPORTANCE_SAMPLING:-False}"
RUN_NAME="${RUN_NAME:-lm1b_full_flm_vdm_${SCHEDULE_TYPE}}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-512}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
DROP_LAST_VALID="${DROP_LAST_VALID:-False}"
MODEL_LENGTH="${MODEL_LENGTH:-128}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-1}"
SAMPLING_STEPS="${SAMPLING_STEPS:-[1024]}"
MAX_STEPS="${MAX_STEPS:-1500000}"
LR="${LR:-3e-4}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-5000}"
PRECISION="${PRECISION:-bf16}"

TAU_MIN="${TAU_MIN:-0}"
TAU_MAX="${TAU_MAX:-0.999}"
VAL_TAU_MIN="${VAL_TAU_MIN:-$TAU_MIN}"
VAL_TAU_MAX="${VAL_TAU_MAX:-$TAU_MAX}"
LATENT_TYPE="${LATENT_TYPE:-vp}"
COND_T="${COND_T:-gamma}"
TRAIN_LOSS="${TRAIN_LOSS:-ce}"
SOFTMAX_TEMPERATURE="${SOFTMAX_TEMPERATURE:-1.0}"
TRAIN_ON_WEIGHTED_LOSS="${TRAIN_ON_WEIGHTED_LOSS:-True}"
RECON_LOSS_ENABLED="${RECON_LOSS_ENABLED:-False}"
TRAIN_ON_RECON_LOSS="${TRAIN_ON_RECON_LOSS:-False}"
RECON_LOSS_WEIGHT="${RECON_LOSS_WEIGHT:-1.0}"
PRIOR_LOSS_ENABLED="${PRIOR_LOSS_ENABLED:-False}"
TRAIN_ON_PRIOR_LOSS="${TRAIN_ON_PRIOR_LOSS:-False}"
PRIOR_LOSS_WEIGHT="${PRIOR_LOSS_WEIGHT:-1.0}"
DOUBLE_TEMB="${DOUBLE_TEMB:-False}"
VAL_MC_SAMPLES="${VAL_MC_SAMPLES:-1}"

GAMMA_MIN="${GAMMA_MIN:--13.3}"
GAMMA_MAX="${GAMMA_MAX:-5.0}"
SNR_POWER_C="${SNR_POWER_C:-17.276323318481445}"
SNR_POWER_P="${SNR_POWER_P:-0.43169814348220825}"
SCHEDULE_EPS="${SCHEDULE_EPS:-1e-6}"
LEARNED_HIDDEN_SIZE="${LEARNED_HIDDEN_SIZE:-1024}"
ARGMAX_N_POINTS="${ARGMAX_N_POINTS:-10000}"
ARGMAX_N_GH="${ARGMAX_N_GH:-100}"
ARGMAX_INTERPOLATION="${ARGMAX_INTERPOLATION:-pchip}"
ODE_BISECT_ITERS="${ODE_BISECT_ITERS:-${BISECT_ITERS:-64}}"
ALPHA_MIN="${ALPHA_MIN:-0.0}"
ALPHA_MAX="${ALPHA_MAX:-1.0}"
VAL_ALPHA_MIN="${VAL_ALPHA_MIN:-null}"
VAL_ALPHA_MAX="${VAL_ALPHA_MAX:-null}"
ALPHA_KNOTS="${ALPHA_KNOTS:-null}"
TAU_KNOTS="${TAU_KNOTS:-null}"
TAU_DENSITY="${TAU_DENSITY:-null}"
VAL_ALPHA_KNOTS="${VAL_ALPHA_KNOTS:-null}"
VAL_TAU_KNOTS="${VAL_TAU_KNOTS:-null}"
VAL_TAU_DENSITY="${VAL_TAU_DENSITY:-null}"

CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-20000}"
CHECKPOINT_MONITOR="${CHECKPOINT_MONITOR:-val_objective/${TRAIN_LOSS}_weighted}"
CHECKPOINT_FILENAME="${CHECKPOINT_FILENAME:-best_${TRAIN_LOSS}_weighted}"

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
  training.importance_sampling=${IMPORTANCE_SAMPLING} \
  algo.importance_sampling=${IMPORTANCE_SAMPLING} \
  algo.double_temb=${DOUBLE_TEMB} \
  algo.t_min=${TAU_MIN} \
  algo.t_max=${TAU_MAX} \
  algo.val_t_min=${VAL_TAU_MIN} \
  algo.val_t_max=${VAL_TAU_MAX} \
  algo.latent_type=${LATENT_TYPE} \
  algo.cond_t=${COND_T} \
  algo.train_loss=${TRAIN_LOSS} \
  algo.softmax_temperature=${SOFTMAX_TEMPERATURE} \
  algo.train_on_weighted_loss=${TRAIN_ON_WEIGHTED_LOSS} \
  algo.recon_loss_enabled=${RECON_LOSS_ENABLED} \
  algo.train_on_recon_loss=${TRAIN_ON_RECON_LOSS} \
  algo.recon_loss_weight=${RECON_LOSS_WEIGHT} \
  algo.prior_loss_enabled=${PRIOR_LOSS_ENABLED} \
  algo.train_on_prior_loss=${TRAIN_ON_PRIOR_LOSS} \
  algo.prior_loss_weight=${PRIOR_LOSS_WEIGHT} \
  algo.val_mc_samples=${VAL_MC_SAMPLES} \
  algo.schedule.type=${SCHEDULE_TYPE} \
  algo.schedule.gamma_min=${GAMMA_MIN} \
  algo.schedule.gamma_max=${GAMMA_MAX} \
  algo.schedule.alpha_min=${ALPHA_MIN} \
  algo.schedule.alpha_max=${ALPHA_MAX} \
  algo.schedule.val_alpha_min=${VAL_ALPHA_MIN} \
  algo.schedule.val_alpha_max=${VAL_ALPHA_MAX} \
  algo.schedule.alpha_knots="${ALPHA_KNOTS}" \
  algo.schedule.tau_knots="${TAU_KNOTS}" \
  algo.schedule.tau_density="${TAU_DENSITY}" \
  algo.schedule.val_alpha_knots="${VAL_ALPHA_KNOTS}" \
  algo.schedule.val_tau_knots="${VAL_TAU_KNOTS}" \
  algo.schedule.val_tau_density="${VAL_TAU_DENSITY}" \
  algo.schedule.C=${SNR_POWER_C} \
  algo.schedule.p=${SNR_POWER_P} \
  algo.schedule.eps=${SCHEDULE_EPS} \
  algo.schedule.hidden_size=${LEARNED_HIDDEN_SIZE} \
  algo.schedule.n_points=${ARGMAX_N_POINTS} \
  algo.schedule.n_gh=${ARGMAX_N_GH} \
  algo.schedule.interpolation=${ARGMAX_INTERPOLATION} \
  algo.schedule.bisect_iters=${ODE_BISECT_ITERS} \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=${CHECKPOINT_EVERY_N_STEPS} \
  callbacks.checkpoint_monitor.monitor="${CHECKPOINT_MONITOR}" \
  callbacks.checkpoint_monitor.filename="${CHECKPOINT_FILENAME}" \
  hydra.run.dir=${REPO_ROOT}/outputs/lm1b/${RUN_NAME} \
  loader.drop_last_valid=${DROP_LAST_VALID}
