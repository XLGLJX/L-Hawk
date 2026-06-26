#!/usr/bin/env bash
set -euo pipefail

# Quick TA-C smoke run. This intentionally caps training/eval to one batch.

GPU="${GPU:-0}"
EXP_ROOT="${EXP_ROOT:-exp}"
EPOCHS="${EPOCHS:-1}"
REPEAT="${REPEAT:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EVAL_BATCH="${EVAL_BATCH:-1}"
TRAIN_BATCH="${TRAIN_BATCH:-1}"
TRIGGER_SELECTION="${TRIGGER_SELECTION:-async-joint}"
TRIGGER_PARAM_SAMPLES="${TRIGGER_PARAM_SAMPLES:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-l-hawk}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"

mkdir -p "${EXP_ROOT}/logs"

swanlab_args=(--swanlab --swanlab-project "${SWANLAB_PROJECT}" --swanlab-mode "${SWANLAB_MODE}")
if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
  swanlab_args+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
fi

echo "Launching TA-C smoke run on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" python demo.py \
  --cfg configs/TA-C.yaml \
  --attack_type TA-C \
  --det vgg16 \
  --target 920 \
  --batch-size "${BATCH_SIZE}" \
  --trigger-source laser \
  --trigger-selection "${TRIGGER_SELECTION}" \
  --trigger-param-samples "${TRIGGER_PARAM_SAMPLES}" \
  --epochs "${EPOCHS}" \
  --train-batch "${TRAIN_BATCH}" \
  --eval-batch "${EVAL_BATCH}" \
  --repeat "${REPEAT}" \
  --exp_dir "${EXP_ROOT}/tac" \
  "${swanlab_args[@]}"
