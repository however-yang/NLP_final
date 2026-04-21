#!/usr/bin/env bash
# [阶段 6] 根据阶段 5 生成的多份 report JSON，计算 composite 分数并选出最优 checkpoint，再导出 Markdown。
#
# 前置: 已成功运行 pipeline_stage_05_checkpoint_validation.sh。
#
# 日志:
#   logs/___SELECTION_EXPORT_PIPELINE_LOGS___/run_<TS>/
#     - 00_MASTER_ALL_STEPS__<TS>.log
#     - selection_step__select_best__<TS>.log
#     - selection_step__export_markdown__<TS>.log
#
# 产物:
#   outputs/___SELECTION_EXPORT_FINAL_RESULTS___/run_<TS>/
#     - best_checkpoint.json
#     - best_validation.md（除非 SKIP_EXPORT=1）
#     - SELECTION_EXPORT_MANIFEST__<TS>.txt
#
# 环境变量:
#   REPORT_DIR（推荐）      指向阶段 5 的 outputs/___CHECKPOINT_VALIDATION_RESULTS___/run_<TS>/
#   REPORT_LIST_FILE       可选；若设置则优先从此文件读取报告路径（每行一个），覆盖自动发现
#   WEIGHTS                可选，默认 "docvqa=1 chartqa=1"；传给 select_best_checkpoint.py --weights
#   SKIP_EXPORT=1          只选优，不导出 md
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"

LOG_DIR="${ROOT}/logs/___SELECTION_EXPORT_PIPELINE_LOGS___/run_${TS}"
RESULT_DIR="${ROOT}/outputs/___SELECTION_EXPORT_FINAL_RESULTS___/run_${TS}"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

BEST_JSON="${RESULT_DIR}/best_checkpoint.json"
BEST_MD="${RESULT_DIR}/best_validation.md"
MANIFEST="${RESULT_DIR}/SELECTION_EXPORT_MANIFEST__${TS}.txt"

# select_best_checkpoint.py: --weights docvqa=1 chartqa=1（同一 flag 后跟多个 key=value）
if [[ -n "${WEIGHTS:-}" ]]; then
  # shellcheck disable=SC2206
  WEIGHT_ARGS=( --weights ${WEIGHTS} )
else
  WEIGHT_ARGS=( --weights docvqa=1 chartqa=1 )
fi

{
  echo "================================================================================"
  echo "SELECT BEST + EXPORT（阶段 6）| run_id=${TS}"
  echo "ISO 时间: ${TS_ISO}"
  echo "工作目录: ${ROOT}"
  echo "主控日志: ${MASTER_LOG}"
  echo "输出目录: ${RESULT_DIR}"
  echo "================================================================================"
} | append_master

run_cmd() {
  local tag="$1"
  shift
  local step_log="${LOG_DIR}/selection_step__${tag}__${TS}.log"
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

declare -a REPORT_PATHS=()

if [[ -n "${REPORT_LIST_FILE:-}" ]] && [[ -f "${REPORT_LIST_FILE}" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && [[ -f "$line" ]] && REPORT_PATHS+=("$line")
  done < "${REPORT_LIST_FILE}"
  echo "从 REPORT_LIST_FILE 读取 ${#REPORT_PATHS[@]} 份报告: ${REPORT_LIST_FILE}" | append_master
elif [[ -n "${REPORT_DIR:-}" ]] && [[ -d "${REPORT_DIR}" ]]; then
  while IFS= read -r line; do
    [[ -n "$line" ]] && REPORT_PATHS+=("$line")
  done < <(find "${REPORT_DIR}" -maxdepth 1 -type f -name 'report_*.json' | sort -V)
  echo "在 REPORT_DIR 发现 ${#REPORT_PATHS[@]} 份报告: ${REPORT_DIR}" | append_master
else
  echo "错误: 请设置 REPORT_DIR（阶段 5 的结果目录）或 REPORT_LIST_FILE（VALIDATION_REPORT_PATHS*.txt）。" | append_master
  exit 1
fi

if [[ ${#REPORT_PATHS[@]} -eq 0 ]]; then
  echo "未找到任何 report_*.json。" | append_master
  exit 1
fi

if [[ ${#REPORT_PATHS[@]} -eq 1 ]]; then
  echo "仅 1 份报告，跳过 composite 比较。" | append_master
  run_cmd "write_single_best_placeholder" python - <<PY
import json
from pathlib import Path
p = "${REPORT_PATHS[0]}"
out = Path(r"${BEST_JSON}")
out.write_text(
    json.dumps({"best": {"report_path": p, "composite_score": None}, "all_scores": [{"report_path": p, "composite_score": None}]}, indent=2),
    encoding="utf-8",
)
print("written", out)
PY
else
  run_cmd "select_best" python scripts/select_best_checkpoint.py \
    --reports "${REPORT_PATHS[@]}" \
    "${WEIGHT_ARGS[@]}" \
    --output "$BEST_JSON"
fi

BEST_REPORT="$(python -c "import json; print(json.load(open('${BEST_JSON}'))['best']['report_path'])")"
echo "最佳验证报告: ${BEST_REPORT}" | append_master

if [[ "${SKIP_EXPORT:-0}" != "1" ]]; then
  run_cmd "export_markdown" python scripts/export_tables_figures.py --report "$BEST_REPORT" --output "$BEST_MD"
else
  echo "[SKIP_EXPORT=1] 跳过 Markdown 导出。" | append_master
fi

{
  echo "SELECTION_EXPORT_MANIFEST | run_id=${TS}"
  echo "finished_at=$(date -Iseconds)"
  echo "best_checkpoint_meta=${BEST_JSON}"
  echo "best_report=${BEST_REPORT}"
  echo "best_markdown=${BEST_MD}"
  echo "report_count=${#REPORT_PATHS[@]}"
  echo "log_dir=${LOG_DIR}"
} > "$MANIFEST"

cp -f "$MANIFEST" "${LOG_DIR}/SELECTION_EXPORT_MANIFEST__${TS}.copy.txt" 2>/dev/null || true

{
  echo ""
  echo "================================================================================"
  echo "SELECT BEST + EXPORT 结束（阶段 6）| run_id=${TS}"
  echo "清单: ${MANIFEST}"
  echo "主控日志: ${MASTER_LOG}"
  echo "================================================================================"
} | append_master

exit 0
