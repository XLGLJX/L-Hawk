#!/usr/bin/env bash
set -euo pipefail

# TA-C training on GPU 4.
#
# Override defaults if needed:
#   GPU=5 EPOCHS=30 REPEAT=10 bash scripts/train_tac.sh

GPU="${GPU:-4}"
EXP_ROOT="${EXP_ROOT:-exp}"
EPOCHS="${EPOCHS:-20}"
REPEAT="${REPEAT:-20}"
BATCH_SIZE="${BATCH_SIZE:-96}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-l-hawk}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"

mkdir -p "${EXP_ROOT}/logs"

swanlab_args=(--swanlab --swanlab-project "${SWANLAB_PROJECT}" --swanlab-mode "${SWANLAB_MODE}")
if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
  swanlab_args+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
fi

echo "Launching TA-C on GPU ${GPU}"
CUDA_VISIBLE_DEVICES="${GPU}" python demo.py \
  --cfg configs/TA-C.yaml \
  --attack_type TA-C \
  --det vgg16 \
  --target 920 \
  --batch-size "${BATCH_SIZE}" \
  --trigger-source laser \
  --trigger-selection async-joint \
  --trigger-param-samples 32 \
  --epochs "${EPOCHS}" \
  --repeat "${REPEAT}" \
  --exp_dir "${EXP_ROOT}/tac" \
  "${swanlab_args[@]}"
