#!/bin/bash
#SBATCH -J vdm_flm_owt             # Job name
#SBATCH -o watch_folder/%x_%j.out  # output file (%j expands to jobID)
#SBATCH -N 1                       # Total number of nodes requested
#SBATCH --get-user-env             # retrieve the users login environment
#SBATCH --mem=32000                # server memory requested (per node)
#SBATCH -t 960:00:00               # Time limit (hh:mm:ss)
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

DATA_DIR="${DATA_DIR:-YOUR_DATA_DIR}"
TRAIN_LOSS="${TRAIN_LOSS:-ce_upper_bound}"
TIME_SAMPLING="${TIME_SAMPLING:-uniform}"
RUN_NAME="${RUN_NAME:-owt_vdm_flm_${TRAIN_LOSS}}"
RUN_BASE="${RUN_BASE:-${HOME}/flm_runs/openwebtext-split}"
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
  loader.global_batch_size=512 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  data=openwebtext-split \
  data.cache_dir="${DATA_DIR}" \
  wandb.project=owt_full \
  wandb.name="${RUN_NAME}" \
  model=small \
  model.length=1024 \
  algo=flm \
  algo.interpolant_type=vdm_gaussian \
  algo.train_loss="${TRAIN_LOSS}" \
  algo.time_reparam=decoding_error_rate \
  algo.time_sampling="${TIME_SAMPLING}" \
  algo.vdm_schedule=linear_logsnr \
  algo.vdm_logsnr_min=-20.0 \
  algo.vdm_logsnr_max=20.0 \
  algo.double_temb=False \
  trainer.max_steps=1500000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=5000 \
  eval.generate_samples=False \
  eval.compute_generative_perplexity=False \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  checkpointing.monitor_metric="${MONITOR_METRIC}" \
  checkpointing.monitor_filename="${MONITOR_FILENAME}" \
  hydra.run.dir="${RUN_ROOT}" \
  checkpointing.save_dir="${RUN_ROOT}" \
  "$@"
