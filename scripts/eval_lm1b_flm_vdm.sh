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

checkpoint_path="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt"
DATA_DIR="/share/kuleshov/yzs2/data"

# Setup environment
cd ../ || exit  # Go to the root directory of the repo
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

python -u -m main \
  mode=ppl_eval \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=0 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  model=small \
  model.length=128 \
  algo=flm \
  algo.interpolant_type=vp_gaussian \
  algo.train_loss=ce_upper_bound \
  algo.t_min=0.01 \
  algo.t_max=0.99 \
  algo.gamma_schedule=flm_weight_matched \
  algo.vdm_conditioning=tau \
  checkpointing.monitor_metric=val/ce_upper_bound \
  eval.checkpoint_path=$checkpoint_path \
  sampling.num_sample_batches=0 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  +wandb.offline=true \
  "$@"
