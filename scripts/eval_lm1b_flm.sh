#!/bin/bash
#SBATCH -J eval_flm                   # Job name
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

checkpoint_path="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt"
DATA_DIR="/share/kuleshov/yzs2/data"
TAU_NLL_BINS="${TAU_NLL_BINS:-20}"
TAU_NLL_TOP_K="${TAU_NLL_TOP_K:-8}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-32}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Setup environment
cd "$REPO_DIR" || exit
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

#srun python -u -m main \
python -u -m main \
  mode=ppl_eval \
  loader.batch_size=256 \
  loader.eval_batch_size=256 \
  loader.num_workers=0 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  model=small \
  model.length=128 \
  algo=flm \
  algo.t_min=0.1 \
  algo.t_max=0.9 \
  algo.track_tau_nll=true \
  algo.tau_nll_bins=$TAU_NLL_BINS \
  algo.tau_nll_top_k=$TAU_NLL_TOP_K \
  eval.checkpoint_path=$checkpoint_path \
  trainer.limit_val_batches=$LIMIT_VAL_BATCHES \
  sampling.num_sample_batches=0 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  +wandb.offline=true
