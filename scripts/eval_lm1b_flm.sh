#!/bin/bash
#SBATCH -J eval_flm                   # Job name
#SBATCH -o watch_folder/%x_%j.out     # log file (out & err)
#SBATCH -N 1                          # Total number of nodes requested
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --mem=100000                  # server memory requested (per node)
#SBATCH -t 960:00:00                  # Time limit (hh:mm:ss)
#SBATCH --partition=gpu               # Request partition
#SBATCH --constraint="[a5000|a6000|a100|3090]"
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4                  # Type/number of GPUs needed
#SBATCH --open-mode=append            # Do not overwrite logs
#SBATCH --requeue                     # Requeue upon preemption

checkpoint_path="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt"
DATA_DIR="/share/kuleshov/yzs2/data"

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

#srun python -u -m main \
python -u -m main \
  mode=ppl_eval \
  loader.batch_size=64 \
  loader.eval_batch_size=64 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  model=small \
  model.length=128 \
  algo=flm \
  algo.t_min=0.05 \
  algo.t_max=0.90 \
  eval.checkpoint_path=$checkpoint_path \
  sampling.num_sample_batches=0 \
  +wandb.offline=true
