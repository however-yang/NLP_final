#!/usr/bin/env bash
# [阶段 3] 基线推理与评测：在 4 个数据集（DocVQA, ChartQA, ScienceQA, MMMU）上运行 Zero-shot 和 Structured-prompt。
#
# 日志:
#   logs/___BASELINE_PIPELINE_LOGS___/run_<TS>/
#
# 产物:
#   outputs/___BASELINE_FINAL_RESULTS___/run_<TS>/
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

LOG_DIR="${ROOT}/logs/___BASELINE_PIPELINE_LOGS___/run_${TS}"
RESULT_DIR="${ROOT}/outputs/___BASELINE_FINAL_RESULTS___/run_${TS}"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

MODEL_CFG="${MODEL_CFG:-configs/model/backbone_main.yaml}"
GENERATION_CFG="${GENERATION_CFG:-configs/model/generation.yaml}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "${LIMIT}")
fi

{
  echo "================================================================================"
  echo "BASELINE EVALUATION（阶段 3）| run_id=${TS}"
  echo "ISO 时间: ${TS_ISO}"
  echo "工作目录: ${ROOT}"
  echo "主控日志: ${MASTER_LOG}"
  echo "结果目录: ${RESULT_DIR}"
  echo "================================================================================"
} | append_master

run_cmd() {
  local tag="$1"
  shift
  local step_log="${LOG_DIR}/baseline_step__${tag}__${TS}.log"
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

# 数据集列表
DATASETS=("docvqa" "chartqa" "scienceqa" "mmmu")

for DS in "${DATASETS[@]}"; do
  VAL_FILE="data/processed/${DS}/validation.jsonl"
  
  if [[ ! -f "$VAL_FILE" ]]; then
    echo "[WARN] 跳过 ${DS}，未找到 $VAL_FILE" | append_master
    continue
  fi

  # 1. Zero-shot direct baseline
  if [[ "${SKIP_DIRECT:-0}" != "1" ]]; then
    PRED_DIRECT="${RESULT_DIR}/pred_baseline_direct_${DS}.jsonl"
    REPORT_DIRECT="${RESULT_DIR}/report_baseline_direct_${DS}.json"
    TAGGED_DIRECT="${RESULT_DIR}/tagged_baseline_direct_${DS}.jsonl"

    run_cmd "eval_direct_${DS}" python scripts/validate_checkpoint.py \
      --samples "$VAL_FILE" \
      --predictions-output "$PRED_DIRECT" \
      --report-output "$REPORT_DIRECT" \
      --tagged-output "$TAGGED_DIRECT" \
      --model-config "$MODEL_CFG" \
      --generation-config "$GENERATION_CFG" \
      --prompt-style direct \
      --resume \
      ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
  fi

  # 2. Structured prompt baseline
  if [[ "${SKIP_STRUCTURED:-0}" != "1" ]]; then
    PRED_STRUCT="${RESULT_DIR}/pred_baseline_structured_${DS}.jsonl"
    REPORT_STRUCT="${RESULT_DIR}/report_baseline_structured_${DS}.json"
    TAGGED_STRUCT="${RESULT_DIR}/tagged_baseline_structured_${DS}.jsonl"

    run_cmd "eval_structured_${DS}" python scripts/validate_checkpoint.py \
      --samples "$VAL_FILE" \
      --predictions-output "$PRED_STRUCT" \
      --report-output "$REPORT_STRUCT" \
      --tagged-output "$TAGGED_STRUCT" \
      --model-config "$MODEL_CFG" \
      --generation-config "$GENERATION_CFG" \
      --prompt-style structured \
      --resume \
      ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}
  fi
done

{
  echo ""
  echo "================================================================================"
  echo "BASELINE EVALUATION 结束（阶段 3）| run_id=${TS}"
  echo "结果目录: ${RESULT_DIR}"
  echo "================================================================================"
} | append_master

exit 0
