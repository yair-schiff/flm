#!/bin/bash
# Import-level verification for the clean staged environment.

set -euo pipefail

CONDA_BIN="${CONDA_BIN:-/share/apps/software/anaconda3/condabin/conda}"
ENV_PREFIX="${ENV_PREFIX:-/share/kuleshov/yzs2/myconda/envs/flm_og}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -c "import torch; print('torch', torch.__version__)"
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -c "import lightning; print('lightning_ok')"
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -c "import transformers; print('transformers_ok')"
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -c "import flash_attn; print('flash_attn_ok')"
"${CONDA_BIN}" run -p "${ENV_PREFIX}" python -c "import sys; sys.path.insert(0, '${REPO_ROOT}'); import main; import algo; print('repo_import_ok')"

echo "Environment verification succeeded."
