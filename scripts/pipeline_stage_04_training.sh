#!/usr/bin/env bash
# [阶段 4] PEFT 训练：根据指定的配置运行 LoRA 微调。
#
# 日志:
#   logs/___TRAINING_PIPELINE_LOGS___/run_<TS>/
#
# 产物:
#   通过配置文件中指定的 output_dir 存储 checkpoint (默认 outputs/checkpoints/)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"

LOG_DIR="${ROOT}/logs/___TRAINING_PIPELINE_LOGS___/run_${TS}"
mkdir -p "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

TRAIN_CFG="${TRAIN_CFG:-configs/train/train_joint.yaml}"
MODEL_CFG="${MODEL_CFG:-configs/model/backbone_main.yaml}"
PEFT_CFG="${PEFT_CFG:-configs/model/peft.yaml}"
SEED="${SEED:-42}"

{
  echo "================================================================================"
  echo "TRAINING PIPELINE（阶段 4）| run_id=${TS}"
  echo "ISO 时间: ${TS_ISO}"
  echo "工作目录: ${ROOT}"
  echo "主控日志: ${MASTER_LOG}"
  echo "TRAIN_CFG=${TRAIN_CFG}"
  echo "MODEL_CFG=${MODEL_CFG}"
  echo "PEFT_CFG=${PEFT_CFG}"
  echo "================================================================================"
} | append_master

run_cmd() {
  local tag="$1"
  shift
  local step_log="${LOG_DIR}/training_step__${tag}__${TS}.log"
  {
    echo ""
    echo "################################################################################"
    echo "# STEP [${tag}] 开始  $(date -Iseconds)"
    echo "################################################################################"
  } | append_master | tee "$step_log"
  "$@" 2>&1 | tee -a "$step_log" | append_master
  {
    echo "# STEP [${tag}] 结束  $(date -Iseconds)"
    echo ""
  } | append_master | tee -a "$step_log"
}

TRAIN_ARGS=(
  python scripts/train_peft.py
  "--train-config" "$TRAIN_CFG"
  "--model-config" "$MODEL_CFG"
  "--peft-config" "$PEFT_CFG"
  "--seed" "$SEED"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--dry-run")
fi

if [[ -n "${RESUME_FROM:-}" ]]; then
  TRAIN_ARGS+=("--resume-from-checkpoint" "$RESUME_FROM")
fi

run_cmd "run_peft_training" "${TRAIN_ARGS[@]}"

{
  echo ""
  echo "================================================================================"
  echo "TRAINING PIPELINE 结束（阶段 4）| run_id=${TS}"
  echo "主控日志: ${MASTER_LOG}"
  echo "================================================================================"
} | append_master

exit 0
