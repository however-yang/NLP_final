#!/usr/bin/env bash
# [阶段 2] Prompt / 训练数据 sanity check：调用 train_peft.py --dry-run（不加载长时间训练，仅校验混合采样与 prompt 构造）。
#
# 日志:
#   logs/___PROMPT_SANITY_PIPELINE_LOGS___/run_<TS>/
#     - 00_MASTER_ALL_STEPS__<TS>.log
#     - prompt_sanity_step__train_peft_dry_run__<TS>.log
#
# 清单:
#   outputs/___PROMPT_SANITY_FINAL_RESULTS___/run_<TS>/PROMPT_SANITY_MANIFEST__<TS>.txt
#
# 环境变量（与训练一致）:
#   TRAIN_CFG MODEL_CFG PEFT_CFG SEED
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DATA_DISK}/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"

LOG_DIR="${ROOT}/logs/___PROMPT_SANITY_PIPELINE_LOGS___/run_${TS}"
RESULT_DIR="${ROOT}/outputs/___PROMPT_SANITY_FINAL_RESULTS___/run_${TS}"
CONFIG_SNAP="${RESULT_DIR}/config_snapshot"
MANIFEST="${RESULT_DIR}/PROMPT_SANITY_MANIFEST__${TS}.txt"

mkdir -p "$LOG_DIR" "$CONFIG_SNAP"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"

append_master() {
  tee -a "$MASTER_LOG"
}

TRAIN_CFG="${TRAIN_CFG:-configs/train/train_joint.yaml}"
MODEL_CFG="${MODEL_CFG:-configs/model/backbone_main.yaml}"
PEFT_CFG="${PEFT_CFG:-configs/model/peft.yaml}"

cp -f "$TRAIN_CFG" "${CONFIG_SNAP}/train_config.yaml"
cp -f "$MODEL_CFG" "${CONFIG_SNAP}/model_config.yaml"
cp -f "$PEFT_CFG" "${CONFIG_SNAP}/peft_config.yaml"

{
  echo "================================================================================"
  echo "PROMPT SANITY（阶段 2）| run_id=${TS}"
  echo "ISO 时间: ${TS_ISO}"
  echo "工作目录: ${ROOT}"
  echo "主控日志: ${MASTER_LOG}"
  echo "配置快照: ${CONFIG_SNAP}"
  echo "模式: train_peft.py --dry-run（不占用 GPU 训练）"
  echo "TRAIN_CFG=${TRAIN_CFG}"
  echo "MODEL_CFG=${MODEL_CFG}"
  echo "PEFT_CFG=${PEFT_CFG}"
  echo "================================================================================"
} | append_master

run_cmd() {
  local tag="$1"
  shift
  local step_log="${LOG_DIR}/prompt_sanity_step__${tag}__${TS}.log"
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

DRY_ARGS=(
  python scripts/train_peft.py
  "--train-config" "$TRAIN_CFG"
  "--model-config" "$MODEL_CFG"
  "--peft-config" "$PEFT_CFG"
  "--seed" "${SEED:-42}"
  "--dry-run"
)

run_cmd "train_peft_dry_run" "${DRY_ARGS[@]}"

CMD_STR="${DRY_ARGS[*]}"

{
  echo "PROMPT_SANITY_MANIFEST | run_id=${TS}"
  echo "finished_at=$(date -Iseconds)"
  echo "command: cd ${ROOT} && ${CMD_STR}"
  echo "log_dir=${LOG_DIR}"
  echo "result_dir=${RESULT_DIR}"
} > "$MANIFEST"

cp -f "$MANIFEST" "${LOG_DIR}/PROMPT_SANITY_MANIFEST__${TS}.copy.txt" 2>/dev/null || true

{
  echo ""
  echo "================================================================================"
  echo "PROMPT SANITY 结束（阶段 2）| run_id=${TS}"
  echo "清单: ${MANIFEST}"
  echo "主控日志: ${MASTER_LOG}"
  echo "================================================================================"
} | append_master

exit 0
