#!/usr/bin/env bash
set -euo pipefail

# CA training on GPU 7.
#
# Override defaults if needed:
#   GPU=4 EPOCHS=30 REPEAT=10 bash scripts/train_ca.sh

GPU="${GPU:-7}"
EXP_ROOT="${EXP_ROOT:-exp}"
EPOCHS="${EPOCHS:-20}"
REPEAT="${REPEAT:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EVAL_DATASET="${EVAL_DATASET:-coco}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-l-hawk}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"

mkdir -p "${EXP_ROOT}/logs"

swanlab_args=(--swanlab --swanlab-project "${SWANLAB_PROJECT}" --swanlab-mode "${SWANLAB_MODE}")
if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
  swanlab_args+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
fi

echo "Launching CA on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" python demo.py \
  --cfg configs/CA.yaml \
  --attack_type CA \
  --det yolov5 \
  --target "stop sign" \
  --batch-size "${BATCH_SIZE}" \
  --eval-dataset "${EVAL_DATASET}" \
  --trigger-source laser \
  --trigger-selection async-joint \
  --trigger-param-samples 32 \
  --epochs "${EPOCHS}" \
  --repeat "${REPEAT}" \
  --exp_dir "${EXP_ROOT}/ca" \
  "${swanlab_args[@]}"
