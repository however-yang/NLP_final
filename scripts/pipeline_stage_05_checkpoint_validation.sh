#!/usr/bin/env bash
# [阶段 5] 训练后验证：对每个候选 checkpoint 在「合并验证集」(DocVQA+ChartQA val) 上推理并生成评测报告。
#       不进行选优与 Markdown 导出（见阶段 6）。
#
# 日志:
#   logs/___CHECKPOINT_VALIDATION_PIPELINE_LOGS___/run_<TS>/
#     - 00_MASTER_ALL_STEPS__<TS>.log
#     - ckpt_val_step__validate__<safe_ckpt_name>__<TS>.log
#
# 产物:
#   outputs/___CHECKPOINT_VALIDATION_RESULTS___/run_<TS>/
#     - combined_validation.jsonl
#     - pred_<name>.jsonl / report_<name>.json / tagged_<name>.jsonl
#     - VALIDATION_REPORT_PATHS__<TS>.txt   # 供阶段 6 读取（每行一个 report json 路径）
#     - CHECKPOINT_VALIDATION_MANIFEST__<TS>.txt
#
# 环境变量（与 run_post_training_pipeline.sh 对齐）:
#   CKPT_ROOT MODEL_CFG GENERATION_CFG VAL_DOC VAL_CHART EVAL_LIMIT EXTRA_CHECKPOINTS
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export TEXT_RICH_MLLM_PROJECT_ROOT="${ROOT}"

export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export TEXT_RICH_MLLM_CHECKPOINT_ROOT="${TEXT_RICH_MLLM_CHECKPOINT_ROOT:-${DATA_DISK}}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"

LOG_DIR="${ROOT}/logs/___CHECKPOINT_VALIDATION_PIPELINE_LOGS___/run_${TS}"
RESULT_DIR="${ROOT}/outputs/___CHECKPOINT_VALIDATION_RESULTS___/run_${TS}"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"

append_master() { tee -a "$MASTER_LOG"; }

_REL_CKPT_DEF="outputs/checkpoints/joint_docvqa_chartqa"
if [[ -z "${CKPT_ROOT:-}" ]]; then
  if [[ -n "${TEXT_RICH_MLLM_CHECKPOINT_ROOT:-}" ]]; then
    CKPT_ROOT="${TEXT_RICH_MLLM_CHECKPOINT_ROOT}/${_REL_CKPT_DEF}"
  elif [[ -n "${DATA_DISK:-}" ]]; then
    CKPT_ROOT="${DATA_DISK}/${_REL_CKPT_DEF}"
  else
    CKPT_ROOT="${_REL_CKPT_DEF}"
  fi
fi
MODEL_CFG="${MODEL_CFG:-configs/model/backbone_main.yaml}"
GENERATION_CFG="${GENERATION_CFG:-configs/model/generation.yaml}"
VAL_DOC="${VAL_DOC:-data/processed/docvqa/validation.jsonl}"
VAL_CHART="${VAL_CHART:-data/processed/chartqa/validation.jsonl}"

REPORT_LIST_FILE="${RESULT_DIR}/VALIDATION_REPORT_PATHS__${TS}.txt"
MANIFEST="${RESULT_DIR}/CHECKPOINT_VALIDATION_MANIFEST__${TS}.txt"

{
  echo "================================================================================"
  echo "CHECKPOINT VALIDATION（阶段 5）| run_id=${TS}"
  echo "ISO 时间: ${TS_ISO}"
  echo "工作目录: ${ROOT}"
  echo "主控日志: ${MASTER_LOG}"
  echo "结果目录: ${RESULT_DIR}"
  echo "CKPT_ROOT=${CKPT_ROOT}"
  echo "================================================================================"
} | append_master

if [[ ! -d "$CKPT_ROOT" ]]; then
  echo "CKPT_ROOT 不是目录: ${CKPT_ROOT}" | append_master
  exit 1
fi

COMBINED="${RESULT_DIR}/combined_validation.jsonl"
if [[ ! -f "$VAL_DOC" ]] || [[ ! -f "$VAL_CHART" ]]; then
  echo "缺少验证集: VAL_DOC=${VAL_DOC} VAL_CHART=${VAL_CHART}" | append_master
  exit 1
fi

run_cmd() {
  local tag="$1"
  shift
  local step_log="${LOG_DIR}/ckpt_val_step__${tag}__${TS}.log"
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

run_cmd "merge_validation_jsonl" bash -c "cat \"${VAL_DOC}\" \"${VAL_CHART}\" > \"${COMBINED}\" && echo merged_lines=\$(wc -l < \"${COMBINED}\")"

declare -a CANDIDATES=()
while IFS= read -r line; do
  [[ -n "$line" ]] && CANDIDATES+=("$line")
done < <(find "${CKPT_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' | sort -V)

if [[ -f "${CKPT_ROOT}/adapter_config.json" ]]; then
  CANDIDATES+=("${CKPT_ROOT}")
fi

if [[ -n "${EXTRA_CHECKPOINTS:-}" ]]; then
  # shellcheck disable=SC2206
  extras=(${EXTRA_CHECKPOINTS})
  for x in "${extras[@]}"; do
    [[ -d "$x" ]] && CANDIDATES+=("$x")
  done
fi

if [[ ${#CANDIDATES[@]} -eq 0 ]]; then
  echo "未找到候选 checkpoint。" | append_master
  exit 1
fi

LIMIT_ARGS=()
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  LIMIT_ARGS+=(--limit "${EVAL_LIMIT}")
fi

: > "${REPORT_LIST_FILE}"

for ck in "${CANDIDATES[@]}"; do
  safe_name="$(basename "$ck" | sed 's/[^A-Za-z0-9._-]/_/g')"
  pred="${RESULT_DIR}/pred_${safe_name}.jsonl"
  rep="${RESULT_DIR}/report_${safe_name}.json"
  tagged="${RESULT_DIR}/tagged_${safe_name}.jsonl"

  run_cmd "validate_${safe_name}" python scripts/validate_checkpoint.py \
    --checkpoint "$ck" \
    --samples "$COMBINED" \
    --predictions-output "$pred" \
    --report-output "$rep" \
    --tagged-output "$tagged" \
    --model-config "$MODEL_CFG" \
    --generation-config "$GENERATION_CFG" \
    --prompt-style structured \
    --resume \
    ${LIMIT_ARGS[@]+"${LIMIT_ARGS[@]}"}

  echo "$rep" >> "${REPORT_LIST_FILE}"
done

{
  echo "CHECKPOINT_VALIDATION_MANIFEST | run_id=${TS}"
  echo "finished_at=$(date -Iseconds)"
  echo "result_dir=${RESULT_DIR}"
  echo "report_list_file=${REPORT_LIST_FILE}"
  echo "combined_validation=${COMBINED}"
  echo ""
  echo "下一阶段（选优+导出）请设置:"
  echo "  REPORT_DIR=${RESULT_DIR}"
  echo "  bash scripts/pipeline_stage_06_select_best_export.sh"
} > "$MANIFEST"

cp -f "$MANIFEST" "${LOG_DIR}/CHECKPOINT_VALIDATION_MANIFEST__${TS}.copy.txt" 2>/dev/null || true

{
  echo ""
  echo "================================================================================"
  echo "CHECKPOINT VALIDATION 结束（阶段 5）| run_id=${TS}"
  echo "报告路径列表: ${REPORT_LIST_FILE}"
  echo "主控日志: ${MASTER_LOG}"
  echo "================================================================================"
} | append_master

exit 0
