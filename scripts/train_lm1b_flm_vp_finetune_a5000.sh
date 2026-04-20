#!/bin/bash
#SBATCH -J ft_lm1b_flm_vp
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=180000
#SBATCH -t 48:00:00
#SBATCH --partition=kuleshov
#SBATCH --constraint="a6000"
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --open-mode=append
#SBATCH --requeue

checkpoint_path="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt"
DATA_DIR="/share/kuleshov/yzs2/data"

cd ../ || exit
source setup_env.sh || exit
export HYDRA_FULL_ERROR=1

python -u -m main \
  loader.global_batch_size=32 \
  loader.batch_size=32 \
  loader.eval_batch_size=32 \
  loader.num_workers=1 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  wandb.project=lm1b_finetune \
  wandb.name=lm1b_vp_flm_weight_matched \
  model=small \
  algo=flm \
  model.length=128 \
  algo.interpolant_type=vp_gaussian \
  algo.train_loss=ce_upper_bound \
  algo.gamma_schedule=flm_weight_matched \
  algo.vdm_conditioning=tau \
  algo.t_min=0.01 \
  algo.t_max=0.99 \
  training.finetune_path=$checkpoint_path \
  sampling.num_sample_batches=0 \
  eval.generate_samples=false \
  eval.compute_generative_perplexity=false \
  trainer.num_sanity_val_steps=0 \
  trainer.max_steps=40000 \
  trainer.val_check_interval=10000 \
  trainer.num_nodes=1 \
  trainer.devices=1 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=10000 \
  checkpointing.monitor_metric=val/ce_upper_bound \
  optim.lr=1e-4 \
  trainer.precision=bf16 \
  algo.double_temb=False \
  +wandb.offline=false \
  "$@"
