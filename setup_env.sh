#!/bin/bash

# Shell script to set environment variables when running code in this repository.
# Usage:
#     source setup_env.sh

# Activate conda env
# shellcheck source=${HOME}/.bashrc disable=SC1091
source "${CONDA_SHELL}"
if [ -n "${FLM_CONDA_ENV:-}" ]; then
    target_env="${FLM_CONDA_ENV}"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    target_env="$(basename "${CONDA_PREFIX}")"
else
    target_env="flm"
fi
if [ -z "${CONDA_PREFIX}" ]; then
    conda activate "${target_env}"
 elif [[ "${CONDA_PREFIX}" != *"/${target_env}" ]]; then
  conda deactivate
  conda activate "${target_env}"
fi
echo "Using conda env '${target_env}'."

# Setup HF cache
# shellcheck disable=SC1091
cache_root="${FLM_CACHE_DIR:-${PWD}/.hf_cache}"
export HF_HOME="${cache_root}"
export WANDB_DIR="${FLM_WANDB_DIR:-${cache_root}}"
export WANDB_CACHE_DIR="${FLM_WANDB_CACHE_DIR:-${cache_root}/wandb-cache}"
mkdir -p "${HF_HOME}" "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
echo "HuggingFace cache set to '${HF_HOME}'."

# Add root directory to PYTHONPATH to enable module imports
export PYTHONPATH="${PWD}:${PWD}/guidance_eval:${HF_HOME}/modules"
