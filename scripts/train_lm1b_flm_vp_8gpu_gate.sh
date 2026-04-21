#!/bin/bash
# Conservative eight-GPU gate run.
# Start here before turning the full eight-GPU fine-tune loose for a long duration.

#SBATCH -J lm1b_vp_8gpu_gate
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -N 1
#SBATCH --get-user-env
#SBATCH --mem=180000
#SBATCH -t 08:00:00
#SBATCH --partition=kuleshov
#SBATCH --constraint="a5000"
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=1
#SBATCH --gres-flags=enforce-binding
#SBATCH --open-mode=append
#SBATCH --requeue

checkpoint_path="${CHECKPOINT_PATH:-/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_flm.ckpt}"
DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
export FLM_CACHE_DIR="${FLM_CACHE_DIR:-${DATA_DIR}/flm-og-cache}"
export FLM_WANDB_DIR="${FLM_WANDB_DIR:-${DATA_DIR}}"
export FLM_WANDB_CACHE_DIR="${FLM_WANDB_CACHE_DIR:-${DATA_DIR}/wandb-cache}"

REPO_ROOT="${FLM_REPO_ROOT:-/share/kuleshov/yzs2/flm-og}"
RUN_SUFFIX="${RUN_SUFFIX:-${SLURM_JOB_ID:-manual}}"
RUN_BASE_NAME="${RUN_BASE_NAME:-lm1b_vp_8gpu_gate}"
RUN_ID="${RUN_ID:-$(printf '%s' "${RUN_SUFFIX}" | sha1sum | cut -c1-8)}"
RUN_NAME="${RUN_NAME:-${RUN_BASE_NAME}_${RUN_ID}}"

cd "${REPO_ROOT}" || exit
source "${REPO_ROOT}/setup_env.sh" || exit
export HYDRA_FULL_ERROR=1

# Match the fresh-env 2-GPU proof recipe that survives online WandB startup.
if [ "${WANDB_OFFLINE:-false}" != "true" ]; then
  : "${WANDB_CONSOLE:=off}"
  : "${WANDB_DISABLE_GIT:=true}"
  : "${WANDB_DISABLE_CODE:=true}"
fi
export WANDB_CONSOLE WANDB_DISABLE_GIT WANDB_DISABLE_CODE

srun --gpu-bind=closest python -u -m main \
  loader.global_batch_size=128 \
  loader.batch_size=8 \
  loader.eval_batch_size=8 \
  loader.num_workers=1 \
  data=lm1b-wrap \
  data.cache_dir=$DATA_DIR \
  wandb.project=lm1b_finetune \
  wandb.name=${RUN_NAME} \
  wandb.id=${RUN_ID} \
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
  trainer.max_steps=1000 \
  trainer.val_check_interval=250 \
  trainer.num_nodes=1 \
  trainer.devices=8 \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=250 \
  checkpointing.monitor_metric=val/ce_upper_bound \
  optim.lr=1e-4 \
  trainer.precision=bf16 \
  algo.double_temb=False \
  +wandb.offline=${WANDB_OFFLINE:-false} \
  "$@"
