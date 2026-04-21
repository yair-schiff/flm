#!/bin/bash
# Clean staged environment bootstrap for this repository.

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/share/apps/software/anaconda3/condabin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/share/kuleshov/yzs2/myconda/envs/flm_og}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CUDA_NVCC_VERSION="${CUDA_NVCC_VERSION:-12.4.99}"
MAX_JOBS="${MAX_JOBS:-4}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[1/4] Creating env at ${ENV_PREFIX}"
"${CONDA_BIN}" create -y -p "${ENV_PREFIX}" \
  -c nvidia \
  "python=${PYTHON_VERSION}" \
  pip \
  ninja \
  packaging \
  setuptools \
  wheel \
  "cuda-nvcc=${CUDA_NVCC_VERSION}"

echo "[2/4] Installing Python requirements"
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -m pip install -r "${REPO_ROOT}/requirements.txt"

echo "[3/4] Installing flash-attn"
CUDA_HOME="${ENV_PREFIX}" \
PATH="${ENV_PREFIX}/bin:${PATH}" \
PIP_NO_CACHE_DIR=1 \
MAX_JOBS="${MAX_JOBS}" \
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -m pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

echo "[4/4] Environment summary"
CUDA_HOME="${ENV_PREFIX}" PATH="${ENV_PREFIX}/bin:${PATH}" \
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "Bootstrap complete."
echo "Next: ENV_PREFIX=${ENV_PREFIX} ${REPO_ROOT}/scripts/verify_env.sh"
