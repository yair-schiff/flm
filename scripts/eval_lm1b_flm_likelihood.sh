#!/bin/bash
#SBATCH -J eval_flm_likelihood         # Job name
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

checkpoint_path="${CHECKPOINT_PATH:-/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt}"
DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_STEPS="${LIKELIHOOD_NUM_STEPS:-128}"
ENDPOINT_EPS="${LIKELIHOOD_ENDPOINT_EPS:-1e-4}"
TRACE_METHOD="${LIKELIHOOD_TRACE_METHOD:-hutchinson}"
TRACE_SAMPLES="${LIKELIHOOD_TRACE_SAMPLES:-1}"
SOLVER="${LIKELIHOOD_SOLVER:-rk4}"
NOISE="${LIKELIHOOD_NOISE:-rademacher}"
LIKELIHOOD_OUTPUT_PATH="${LIKELIHOOD_OUTPUT_PATH:-$PWD/likelihood_eval.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_DIR" || exit
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

python -u -m main \
  mode=likelihood_eval \
  loader.batch_size=${BATCH_SIZE} \
  loader.eval_batch_size=${BATCH_SIZE} \
  loader.num_workers=0 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  model=small \
  model.length=128 \
  algo=flm \
  algo.t_min=0.0 \
  algo.t_max=0.999 \
  eval.checkpoint_path=$checkpoint_path \
  eval.likelihood_output_path=$LIKELIHOOD_OUTPUT_PATH \
  eval.likelihood_num_steps=${NUM_STEPS} \
  eval.likelihood_endpoint_eps=${ENDPOINT_EPS} \
  eval.likelihood_trace_method=${TRACE_METHOD} \
  eval.likelihood_trace_samples=${TRACE_SAMPLES} \
  eval.likelihood_solver=${SOLVER} \
  eval.likelihood_noise=${NOISE} \
  trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
  sampling.num_sample_batches=0 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  +wandb.offline=true
