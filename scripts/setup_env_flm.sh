#!/bin/bash

# Usage:
#   source scripts/setup_env_flm.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this script instead of executing it: source scripts/setup_env_flm.sh" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FLM_ENV_PATH="${FLM_ENV_PATH:-/share/kuleshov/yzs2/myconda/envs/flm}"
CONDA_BIN="${CONDA_BIN:-/share/apps/software/anaconda3/condabin/conda}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1-fasrc01}"

if [ -x "${CONDA_BIN}" ]; then
  CONDA_BASE="$("${CONDA_BIN}" info --base)"
  # shellcheck disable=SC1091
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  if [ -z "${CONDA_PREFIX:-}" ] || [ "${CONDA_PREFIX}" != "${FLM_ENV_PATH}" ]; then
    conda activate "${FLM_ENV_PATH}"
  fi
  export FLM_PYTHON="${CONDA_PREFIX}/bin/python"
else
  export FLM_PYTHON="${FLM_ENV_PATH}/bin/python"
fi

if [ ! -x "${FLM_PYTHON}" ]; then
  echo "Python executable not found at ${FLM_PYTHON}" >&2
  return 1
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"

if ! command -v nvcc >/dev/null 2>&1; then
  if type module >/dev/null 2>&1; then
    module load "${CUDA_MODULE}" || true
  fi
fi

if command -v nvcc >/dev/null 2>&1; then
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
  export PATH="${CUDA_HOME}/bin:${PATH}"
  if [ -d "${CUDA_HOME}/lib64" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  if [ -d "${CUDA_HOME}/targets/x86_64-linux/lib" ]; then
    export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  if [ -d "${CUDA_HOME}/targets/x86_64-linux/include" ]; then
    export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
  fi
fi

echo "Using FLM env at ${FLM_ENV_PATH}"
echo "Using python at ${FLM_PYTHON}"
echo "HF_HOME=${HF_HOME}"
if command -v nvcc >/dev/null 2>&1; then
  echo "Using nvcc at $(command -v nvcc)"
  echo "CUDA_HOME=${CUDA_HOME}"
else
  echo "nvcc not found; set CUDA_MODULE or install cuda-nvcc into the environment." >&2
fi
