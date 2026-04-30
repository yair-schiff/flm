#!/bin/bash
#SBATCH -J eval_flm_vdm               # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov          # Request partition
#SBATCH --constraint="a5000"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MODEL_LENGTH="${MODEL_LENGTH:-128}"
VAL_MC_SAMPLES="${VAL_MC_SAMPLES:-1}"
DROP_LAST_VALID="${DROP_LAST_VALID:-False}"

SCHEDULE_TYPE="${SCHEDULE_TYPE:-snr_power}"  # linear, learned_vdm, argmax_uncertainty, snr_power
RUN_NAME="${RUN_NAME:-lm1b_full_flm_vdm_${SCHEDULE_TYPE}}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-best_ce_weighted.ckpt}"

TAU_MIN="${TAU_MIN:-0}"
TAU_MAX="${TAU_MAX:-0.999}"
VAL_TAU_MIN="${VAL_TAU_MIN:-$TAU_MIN}"
VAL_TAU_MAX="${VAL_TAU_MAX:-$TAU_MAX}"
LATENT_TYPE="${LATENT_TYPE:-vp}"
COND_T="${COND_T:-gamma}"
TRAIN_LOSS="${TRAIN_LOSS:-ce}"
TRAIN_ON_WEIGHTED_LOSS="${TRAIN_ON_WEIGHTED_LOSS:-True}"
RECON_LOSS_ENABLED="${RECON_LOSS_ENABLED:-False}"
TRAIN_ON_RECON_LOSS="${TRAIN_ON_RECON_LOSS:-False}"
RECON_LOSS_WEIGHT="${RECON_LOSS_WEIGHT:-1.0}"
DOUBLE_TEMB="${DOUBLE_TEMB:-False}"
IMPORTANCE_SAMPLING="${IMPORTANCE_SAMPLING:-False}"

GAMMA_MIN="${GAMMA_MIN:--13.3}"
GAMMA_MAX="${GAMMA_MAX:-5.0}"
SNR_POWER_C="${SNR_POWER_C:-17.276323318481445}"
SNR_POWER_P="${SNR_POWER_P:-0.43169814348220825}"
SCHEDULE_EPS="${SCHEDULE_EPS:-1e-6}"
LEARNED_HIDDEN_SIZE="${LEARNED_HIDDEN_SIZE:-1024}"
ARGMAX_N_POINTS="${ARGMAX_N_POINTS:-10000}"
ARGMAX_N_GH="${ARGMAX_N_GH:-100}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
checkpoint_path="${1:-${CHECKPOINT_PATH:-${REPO_DIR}/outputs/lm1b/${RUN_NAME}/checkpoints/${CHECKPOINT_NAME}}}"

# Setup environment
cd "$REPO_DIR" || exit
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

python -u -m main \
  mode=ppl_eval \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  loader.num_workers=0 \
  loader.drop_last_valid=${DROP_LAST_VALID} \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  model=small \
  model.length=${MODEL_LENGTH} \
  algo=flm_vdm \
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
  algo.train_on_weighted_loss=${TRAIN_ON_WEIGHTED_LOSS} \
  algo.recon_loss_enabled=${RECON_LOSS_ENABLED} \
  algo.train_on_recon_loss=${TRAIN_ON_RECON_LOSS} \
  algo.recon_loss_weight=${RECON_LOSS_WEIGHT} \
  algo.val_mc_samples=${VAL_MC_SAMPLES} \
  algo.schedule.type=${SCHEDULE_TYPE} \
  algo.schedule.gamma_min=${GAMMA_MIN} \
  algo.schedule.gamma_max=${GAMMA_MAX} \
  algo.schedule.C=${SNR_POWER_C} \
  algo.schedule.p=${SNR_POWER_P} \
  algo.schedule.eps=${SCHEDULE_EPS} \
  algo.schedule.hidden_size=${LEARNED_HIDDEN_SIZE} \
  algo.schedule.n_points=${ARGMAX_N_POINTS} \
  algo.schedule.n_gh=${ARGMAX_N_GH} \
  eval.checkpoint_path=$checkpoint_path \
  trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
  sampling.num_sample_batches=0 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  +wandb.offline=true
