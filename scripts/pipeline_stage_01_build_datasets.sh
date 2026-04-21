#!/usr/bin/env bash
# [阶段 1] 构建数据集：下载 + 预处理（内容与 build_all_datasets.sh 等价，日志风格与基线流水线一致）。
#
# 日志:
#   logs/___DATA_BUILD_PIPELINE_LOGS___/run_<YYYYMMDD_HHMMSS>/
#     - 00_MASTER_ALL_STEPS__<TS>.log
#     - data_build_step__<tag>__<TS>.log
#
# 清单:
#   outputs/___DATA_BUILD_FINAL_RESULTS___/run_<TS>/DATA_BUILD_MANIFEST__<TS>.txt
#
# 可选环境变量同 build_all_datasets.sh：SKIP_MMMU=1、DATA_DISK、HF_* 等。
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

LOG_DIR="${ROOT}/logs/___DATA_BUILD_PIPELINE_LOGS___/run_${TS}"
RESULT_DIR="${ROOT}/outputs/___DATA_BUILD_FINAL_RESULTS___/run_${TS}"
MANIFEST="${RESULT_DIR}/DATA_BUILD_MANIFEST__${TS}.txt"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"

append_master() {
  tee -a "$MASTER_LOG"
}

{
  echo "================================================================================"
  echo "DATA BUILD PIPELINE（阶段 1）| run_id=${TS}"
  echo "ISO 时间: ${TS_ISO}"
  echo "工作目录: ${ROOT}"
  echo "主控日志: ${MASTER_LOG}"
  echo "分步日志目录: ${LOG_DIR}"
  echo "清单与结果占位: ${RESULT_DIR}"
  echo "DATA_DISK=${DATA_DISK}"
  echo "HF_ENDPOINT=${HF_ENDPOINT}"
  echo "SKIP_MMMU=${SKIP_MMMU:-0}"
  if [[ -f "${HF_HOME}/token" ]] || [[ -n "${HF_TOKEN:-}" ]]; then
    echo "Hub 认证: 已设置"
  else
    echo "提示: 建议 huggingface-cli login 或 export HF_TOKEN"
  fi
  echo "================================================================================"
} | append_master

run_cmd() {
  local tag="$1"
  shift
  local step_log="${LOG_DIR}/data_build_step__${tag}__${TS}.log"
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

if [[ -L "${ROOT}/data" ]]; then
  DATA_TARGET="$(readlink -f "${ROOT}/data" 2>/dev/null || readlink "${ROOT}/data")"
  mkdir -p "${DATA_TARGET}"
  echo "data 软链 -> ${DATA_TARGET}" | append_master
fi

echo "" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master
echo "PHASE — ChartQA" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master

run_cmd "chartqa_download_train" python scripts/download_data.py --config configs/data/chartqa.yaml --split train
run_cmd "chartqa_download_validation" python scripts/download_data.py --config configs/data/chartqa.yaml --split validation
run_cmd "chartqa_preprocess_train" python scripts/preprocess_chartqa.py --split train
run_cmd "chartqa_preprocess_validation" python scripts/preprocess_chartqa.py --split validation

echo "" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master
echo "PHASE — DocVQA" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master

run_cmd "docvqa_download_train" python scripts/download_data.py --config configs/data/docvqa.yaml --split train
run_cmd "docvqa_download_validation" python scripts/download_data.py --config configs/data/docvqa.yaml --split validation
run_cmd "docvqa_preprocess_train" python scripts/preprocess_docvqa.py --split train
run_cmd "docvqa_preprocess_validation" python scripts/preprocess_docvqa.py --split validation

echo "" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master
echo "PHASE — TextVQA" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master

run_cmd "textvqa_download_train" python scripts/download_data.py --config configs/data/textvqa.yaml --split train
run_cmd "textvqa_download_validation" python scripts/download_data.py --config configs/data/textvqa.yaml --split validation
run_cmd "textvqa_preprocess_train" python scripts/preprocess_textvqa.py --split train
run_cmd "textvqa_preprocess_validation" python scripts/preprocess_textvqa.py --split validation

echo "" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master
echo "PHASE — InfographicVQA" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master

run_cmd "infographic_download_train" python scripts/download_data.py --config configs/data/infographicvqa.yaml --split train
run_cmd "infographic_download_validation" python scripts/download_data.py --config configs/data/infographicvqa.yaml --split validation
run_cmd "infographic_head_check" python3 -c "import json; f=open('data/raw/infographicvqa/train.jsonl'); print(sorted(json.loads(f.readline()).keys()))"
run_cmd "infographic_preprocess_train" python scripts/preprocess_infographicvqa.py --split train
run_cmd "infographic_preprocess_validation" python scripts/preprocess_infographicvqa.py --split validation

echo "" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master
echo "PHASE — ScienceQA（验证集）" | append_master
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master

run_cmd "scienceqa_download_validation" python scripts/download_data.py --config configs/data/scienceqa.yaml --split validation
run_cmd "scienceqa_preprocess_validation" python scripts/preprocess_scienceqa.py --split validation

if [[ "${SKIP_MMMU:-0}" == "1" ]]; then
  echo "[SKIP_MMMU=1] 跳过 MMMU。" | append_master
else
  echo "" | append_master
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master
  echo "PHASE — MMMU（验证集）" | append_master
  echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" | append_master

  run_cmd "mmmu_download_validation" python scripts/download_data.py --config configs/data/mmmu.yaml --split validation
  run_cmd "mmmu_preprocess_validation" python scripts/preprocess_mmmu.py --split validation
fi

run_cmd "line_count_self_check" bash -c "wc -l data/raw/*/*.jsonl 2>/dev/null | sort -n || true; wc -l data/processed/*/*.jsonl 2>/dev/null | sort -n || true"

{
  echo "DATA_BUILD_MANIFEST | run_id=${TS}"
  echo "finished_at=$(date -Iseconds)"
  echo "log_dir=${LOG_DIR}"
  echo "result_dir=${RESULT_DIR}"
  echo ""
  echo "[raw/processed line counts]"
  wc -l data/raw/*/*.jsonl 2>/dev/null | sort -n || true
  wc -l data/processed/*/*.jsonl 2>/dev/null | sort -n || true
} > "$MANIFEST"

cp -f "$MANIFEST" "${LOG_DIR}/DATA_BUILD_MANIFEST__${TS}.copy.txt" 2>/dev/null || true

{
  echo ""
  echo "================================================================================"
  echo "DATA BUILD PIPELINE 结束（阶段 1）| run_id=${TS}"
  echo "清单: ${MANIFEST}"
  echo "主控日志: ${MASTER_LOG}"
  echo "================================================================================"
} | append_master

exit 0
