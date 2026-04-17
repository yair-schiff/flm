#!/bin/bash
#SBATCH -J vdm_flm_smoke           # Job name
#SBATCH -o watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                       # Total number of nodes requested
#SBATCH --get-user-env             # retrieve the users login environment
#SBATCH --mem=16000                # server memory requested (per node)
#SBATCH -t 02:00:00                # Time limit (hh:mm:ss)
#SBATCH --partition=kuleshov       # Override at submit time if needed, e.g. `sbatch --partition=gpu ...`
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1               # Type/number of GPUs needed
#SBATCH --open-mode=append         # Do not overwrite logs
#SBATCH --requeue                  # Requeue upon pre-emption

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

# shellcheck source=scripts/setup_env_flm.sh disable=SC1091
source "${REPO_ROOT}/scripts/setup_env_flm.sh"

export HYDRA_FULL_ERROR=1

TRAIN_LOSS="${TRAIN_LOSS:-ce_upper_bound}"
TIME_SAMPLING="${TIME_SAMPLING:-uniform}"
RUN_NAME="${RUN_NAME:-synthetic_vdm_flm_smoke_${TRAIN_LOSS}}"
RUN_BASE="${RUN_BASE:-${REPO_ROOT}/flm_runs/synthetic-alpha8}"
RUN_ROOT="${RUN_ROOT:-${RUN_BASE}/${RUN_NAME}}"

case "${TRAIN_LOSS}" in
  ce_upper_bound)
    MONITOR_METRIC_DEFAULT="val/ce_upper_bound"
    MONITOR_FILENAME_DEFAULT="best_ce_upper_bound"
    ;;
  l2_vdm)
    MONITOR_METRIC_DEFAULT="val/l2_bound"
    MONITOR_FILENAME_DEFAULT="best_l2_bound"
    ;;
  *)
    echo "Unsupported TRAIN_LOSS=${TRAIN_LOSS}. Use ce_upper_bound or l2_vdm." >&2
    exit 1
    ;;
esac

MONITOR_METRIC="${MONITOR_METRIC:-${MONITOR_METRIC_DEFAULT}}"
MONITOR_FILENAME="${MONITOR_FILENAME:-${MONITOR_FILENAME_DEFAULT}}"

mkdir -p "${RUN_ROOT}"

srun "${FLM_PYTHON}" -u -m main \
  data=synthetic_alpha8 \
  model=tiny-alpha8 \
  algo=flm \
  algo.interpolant_type=vdm_gaussian \
  algo.train_loss="${TRAIN_LOSS}" \
  algo.time_reparam=decoding_error_rate \
  algo.time_sampling="${TIME_SAMPLING}" \
  algo.vdm_schedule=linear_logsnr \
  algo.vdm_logsnr_min=-20.0 \
  algo.vdm_logsnr_max=20.0 \
  loader.global_batch_size=8 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=1 \
  trainer.devices=1 \
  trainer.max_steps=4 \
  trainer.precision=32 \
  trainer.num_sanity_val_steps=0 \
  trainer.limit_train_batches=4 \
  trainer.limit_val_batches=2 \
  trainer.val_check_interval=2 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  +wandb.offline=true \
  wandb.project=flm_smoke \
  wandb.name="${RUN_NAME}" \
  checkpointing.monitor_metric="${MONITOR_METRIC}" \
  checkpointing.monitor_filename="${MONITOR_FILENAME}" \
  hydra.run.dir="${RUN_ROOT}" \
  checkpointing.save_dir="${RUN_ROOT}" \
  "$@"
