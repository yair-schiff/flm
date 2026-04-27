#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${FLM_REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

CHECKPOINT_PATH="/share/kuleshov/yzs2/flm-og/outputs/lm1b/lm1b_full_flm_vdm_v5/checkpoints/best_nll.ckpt"
STEPS="${STEPS:-1024}"
SEED="${SEED:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
NUM_SAMPLE_BATCHES="${NUM_SAMPLE_BATCHES:-1}"
DISABLE_EMA="${DISABLE_EMA:-False}"
DATA_DIR="${DATA_DIR:-/share/kuleshov/yzs2/data}"
COND_T="${COND_T:-log_nsr}"
GAMMA_MIN="${GAMMA_MIN:--4.0}"
GAMMA_MAX="${GAMMA_MAX:-5.0}"
MODEL_LENGTH="${MODEL_LENGTH:-128}"
OUT_JSON="${OUT_JSON:-}"

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --checkpoint_path)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --eval_batch_size)
            EVAL_BATCH_SIZE="$2"
            shift 2
            ;;
        --num_sample_batches)
            NUM_SAMPLE_BATCHES="$2"
            shift 2
            ;;
        --disable_ema)
            DISABLE_EMA="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --cond_t)
            COND_T="$2"
            shift 2
            ;;
        --gamma_min)
            GAMMA_MIN="$2"
            shift 2
            ;;
        --gamma_max)
            GAMMA_MAX="$2"
            shift 2
            ;;
        --model_length)
            MODEL_LENGTH="$2"
            shift 2
            ;;
        --out_json)
            OUT_JSON="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -z "${CHECKPOINT_PATH}" ]]; then
    echo "Usage: $0 --checkpoint_path /path/to/model.ckpt [options]" >&2
    exit 1
fi

cd "${REPO_ROOT}"
source "${REPO_ROOT}/setup_env.sh"
export HYDRA_FULL_ERROR=1

CKPT_STEM="$(basename "${CHECKPOINT_PATH}")"
CKPT_STEM="${CKPT_STEM%.ckpt}"
OUT_DIR="${OUT_DIR:-$(dirname "${CHECKPOINT_PATH}")/sampling_smoke}"
mkdir -p "${OUT_DIR}"

if [[ -z "${OUT_JSON}" ]]; then
    OUT_JSON="${OUT_DIR}/${CKPT_STEM}-seed${SEED}-steps${STEPS}.json"
fi

RUN_DIR="${RUN_DIR:-${OUT_DIR}/hydra_${CKPT_STEM}_seed${SEED}_steps${STEPS}}"

echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Steps: ${STEPS}"
echo "Seed: ${SEED}"
echo "Eval batch size: ${EVAL_BATCH_SIZE}"
echo "Num sample batches: ${NUM_SAMPLE_BATCHES}"
echo "Disable EMA: ${DISABLE_EMA}"
echo "cond_t: ${COND_T}"
echo "Output JSON: ${OUT_JSON}"

python -u -m main \
    mode=sample_eval \
    seed="${SEED}" \
    model=small \
    model.length="${MODEL_LENGTH}" \
    data=lm1b-wrap \
    data.cache_dir="${DATA_DIR}" \
    algo=flm_vdm \
    eval.checkpoint_path="${CHECKPOINT_PATH}" \
    loader.batch_size=2 \
    loader.eval_batch_size="${EVAL_BATCH_SIZE}" \
    sampling.num_sample_batches="${NUM_SAMPLE_BATCHES}" \
    sampling.num_sample_log=4 \
    sampling.steps="${STEPS}" \
    sampling.predictor=ancestral \
    algo.double_temb=False \
    algo.interpolant_type=vp_vdm \
    algo.t_max=1.0 \
    algo.cond_t="${COND_T}" \
    algo.gamma_min="${GAMMA_MIN}" \
    algo.gamma_max="${GAMMA_MAX}" \
    algo.train_loss=ce \
    algo.train_on_weighted_loss=True \
    eval.disable_ema="${DISABLE_EMA}" \
    eval.compute_generative_perplexity=false \
    eval.gen_ppl_eval_model_name_or_path=gpt2 \
    eval.generated_samples_path="${OUT_JSON}" \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    +wandb.offline=true \
    hydra.run.dir="${RUN_DIR}"

python - <<'PY' "${OUT_JSON}"
import json
import sys

path = sys.argv[1]
with open(path, "r") as f:
    payload = json.load(f)

samples = payload.get("generated_seqs", [])
print(f"\nSaved samples to: {path}")
print(f"Number of generated sequences: {len(samples)}")
for i, sample in enumerate(samples[:5]):
    print(f"[{i}] {sample}")
PY
