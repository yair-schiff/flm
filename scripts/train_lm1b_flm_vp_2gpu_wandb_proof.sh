#!/bin/bash
# Two-GPU proof run for DDP + online WandB.
# This is the smallest "real" launch profile we trust for startup validation.

#SBATCH -J lm1b_vp_2gpu_proof
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=80000
#SBATCH -t 06:00:00
#SBATCH --partition=kuleshov
#SBATCH --constraint="a5000|a6000"
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --open-mode=append
#SBATCH --requeue

checkpoint_path="${CHECKPOINT_PATH:-/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt}"
DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
WANDB_OFFLINE="${WANDB_OFFLINE:-false}"
export FLM_CACHE_DIR="${FLM_CACHE_DIR:-${DATA_DIR}/flm-og-cache}"
export FLM_WANDB_DIR="${FLM_WANDB_DIR:-${DATA_DIR}}"
export FLM_WANDB_CACHE_DIR="${FLM_WANDB_CACHE_DIR:-${DATA_DIR}/wandb-cache}"

REPO_ROOT="${FLM_REPO_ROOT:-/share/kuleshov/yzs2/flm-og}"
RUN_NAME="${RUN_NAME:-lm1b_vp_2gpu_proof}"

cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh" || exit
export HYDRA_FULL_ERROR=1

# Cluster-safe defaults for fresh-env online WandB. These can still be
# overridden explicitly at submit time.
if [ "${WANDB_OFFLINE}" != "true" ]; then
  : "${WANDB_CONSOLE:=off}"
  : "${WANDB_DISABLE_GIT:=true}"
  : "${WANDB_DISABLE_CODE:=true}"
fi

if [ -n "${WANDB_CONSOLE:-}" ]; then
  export WANDB_CONSOLE
fi
if [ -n "${WANDB_DISABLE_GIT:-}" ]; then
  export WANDB_DISABLE_GIT
fi
if [ -n "${WANDB_DISABLE_CODE:-}" ]; then
  export WANDB_DISABLE_CODE
fi

cmd=(
  srun python -u -m main
  loader.global_batch_size=128
  loader.batch_size=16
  loader.eval_batch_size=8
  loader.num_workers=1
  data=lm1b-wrap
  "data.cache_dir=${DATA_DIR}"
  model=small
  algo=flm
  model.length=128
  algo.interpolant_type=vp_gaussian
  algo.train_loss=ce_upper_bound
  algo.gamma_schedule=flm_weight_matched
  algo.vdm_conditioning=tau
  algo.t_min=0.01
  algo.t_max=0.99
  "training.finetune_path=${checkpoint_path}"
  sampling.num_sample_batches=0
  eval.generate_samples=false
  eval.compute_generative_perplexity=false
  trainer.num_sanity_val_steps=0
  trainer.max_steps=300
  trainer.val_check_interval=100
  trainer.num_nodes=1
  trainer.devices=2
  callbacks.checkpoint_every_n_steps.every_n_train_steps=100
  checkpointing.monitor_metric=val/ce_upper_bound
  optim.lr=1e-4
  trainer.precision=bf16
  algo.double_temb=False
  wandb.project=lm1b_finetune
  "wandb.name=${RUN_NAME}"
  "+wandb.offline=${WANDB_OFFLINE}"
)

cmd+=("$@")
"${cmd[@]}"
