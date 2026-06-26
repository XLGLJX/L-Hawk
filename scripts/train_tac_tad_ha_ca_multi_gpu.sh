#!/usr/bin/env bash
set -euo pipefail

# Run four attack trainings in parallel on four different GPUs.
# One training epoch means one full pass over the training dataloader because
# this script intentionally does not pass --train-batch.
#
# Override GPU assignment if needed:
#   GPU_TAC=0 GPU_TAD=1 GPU_HA=2 GPU_CA=3 bash scripts/train_tac_tad_ha_ca_multi_gpu.sh

GPU_TAC="${GPU_TAC:-0}"
GPU_TAD="${GPU_TAD:-1}"
GPU_HA="${GPU_HA:-2}"
GPU_CA="${GPU_CA:-3}"

EXP_ROOT="${EXP_ROOT:-exp}"
EPOCHS="${EPOCHS:-20}"
REPEAT="${REPEAT:-20}"
BATCH_SIZE="${BATCH_SIZE:-96}"
EVAL_DATASET="${EVAL_DATASET:-coco}"
TRIGGER_SELECTION="${TRIGGER_SELECTION:-async-joint}"
TRIGGER_PARAM_SAMPLES="${TRIGGER_PARAM_SAMPLES:-32}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-l-hawk}"
SWANLAB_MODE="${SWANLAB_MODE:-online}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"

mkdir -p "${EXP_ROOT}/logs"

swanlab_args=(--swanlab --swanlab-project "${SWANLAB_PROJECT}" --swanlab-mode "${SWANLAB_MODE}")
if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
  swanlab_args+=(--swanlab-workspace "${SWANLAB_WORKSPACE}")
fi

echo "Launching TA-C on GPU ${GPU_TAC}"
CUDA_VISIBLE_DEVICES="${GPU_TAC}" python demo.py \
  --cfg configs/TA-C.yaml \
  --attack_type TA-C \
  --det vgg16 \
  --target 920 \
  --batch-size "${BATCH_SIZE}" \
  --trigger-source laser \
  --trigger-selection "${TRIGGER_SELECTION}" \
  --trigger-param-samples "${TRIGGER_PARAM_SAMPLES}" \
  --epochs "${EPOCHS}" \
  --repeat "${REPEAT}" \
  --exp_dir "${EXP_ROOT}/tac"  \
  "${swanlab_args[@]}" \
  > "${EXP_ROOT}/logs/tac_gpu${GPU_TAC}.log" 2>&1 &
PID_TAC=$!

echo "Launching TA-D on GPU ${GPU_TAD}"
CUDA_VISIBLE_DEVICES="${GPU_TAD}" python demo.py \
  --cfg configs/TA-D.yaml \
  --attack_type TA-D \
  --det yolov5 \
  --origin person \
  --target "stop sign" \
  --batch-size "${BATCH_SIZE}" \
  --eval-dataset "${EVAL_DATASET}" \
  --trigger-source laser \
  --trigger-selection "${TRIGGER_SELECTION}" \
  --trigger-param-samples "${TRIGGER_PARAM_SAMPLES}" \
  --epochs "${EPOCHS}" \
  --repeat "${REPEAT}" \
  --exp_dir "${EXP_ROOT}/tad"  \
  "${swanlab_args[@]}" \
  > "${EXP_ROOT}/logs/tad_gpu${GPU_TAD}.log" 2>&1 &
PID_TAD=$!

echo "Launching HA on GPU ${GPU_HA}"
CUDA_VISIBLE_DEVICES="${GPU_HA}" python demo.py \
  --cfg configs/HA.yaml \
  --attack_type HA \
  --det yolov5 \
  --target "stop sign" \
  --batch-size "${BATCH_SIZE}" \
  --eval-dataset "${EVAL_DATASET}" \
  --trigger-source laser \
  --trigger-selection "${TRIGGER_SELECTION}" \
  --trigger-param-samples "${TRIGGER_PARAM_SAMPLES}" \
  --epochs "${EPOCHS}" \
  --repeat "${REPEAT}" \
  --exp_dir "${EXP_ROOT}/ha" \
  "${swanlab_args[@]}" \
  > "${EXP_ROOT}/logs/ha_gpu${GPU_HA}.log" 2>&1 &
PID_HA=$!

echo "Launching CA on GPU ${GPU_CA}"
CUDA_VISIBLE_DEVICES="${GPU_CA}" python demo.py \
  --cfg configs/CA.yaml \
  --attack_type CA \
  --det yolov5 \
  --target "stop sign" \
  --batch-size "${BATCH_SIZE}" \
  --eval-dataset "${EVAL_DATASET}" \
  --trigger-source laser \
  --trigger-selection "${TRIGGER_SELECTION}" \
  --trigger-param-samples "${TRIGGER_PARAM_SAMPLES}" \
  --epochs "${EPOCHS}" \
  --repeat "${REPEAT}" \
  --exp_dir "${EXP_ROOT}/ca" \
  "${swanlab_args[@]}" \
  > "${EXP_ROOT}/logs/ca_gpu${GPU_CA}.log" 2>&1 &
PID_CA=$!

echo "PIDs:"
echo "  TA-C: ${PID_TAC}"
echo "  TA-D: ${PID_TAD}"
echo "  HA:   ${PID_HA}"
echo "  CA:   ${PID_CA}"

wait "${PID_TAC}" "${PID_TAD}" "${PID_HA}" "${PID_CA}"
echo "All trainings finished."
