#!/usr/bin/env bash
# [阶段 7] 外部泛化评测（External Generalization Evaluation）
#
# 作用：
#   在「训练域外」数据集（ScienceQA + MMMU）上评测指定 checkpoint 的零样本泛化能力。
#   每个主要实验（E3/E4/E5/E8）训练完成后都应调用此脚本一次，以填写论文 Table 1。
#
# 与 Stage 05 的区别：
#   Stage 05（in-domain validation）：DocVQA + ChartQA 验证集，用于选出最优 checkpoint
#   Stage 07（external evaluation）  ：ScienceQA + MMMU，评测跨域泛化，不用于 checkpoint 选择
#
# 用法（三种方式，灵活选择）：
#
#   方式 A：自动读取 Stage 06 导出的最优 checkpoint（E3 默认流程）
#     bash scripts/pipeline_stage_07_external_evaluation.sh
#
#   方式 B：手动指定 checkpoint 路径（推荐用于 E4/E5/E8）
#     bash scripts/pipeline_stage_07_external_evaluation.sh \
#       --checkpoint outputs/checkpoints/joint_dora/checkpoint-best \
#       --exp-tag E4_dora
#
#   方式 C：带 TRA checkpoint（E5/E8 含 TRA 的 checkpoint）
#     bash scripts/pipeline_stage_07_external_evaluation.sh \
#       --checkpoint outputs/checkpoints/joint_tra_light/checkpoint-best \
#       --peft-config configs/model/peft.yaml \
#       --exp-tag E5_tra
#
# 产物：
#   outputs/___EXTERNAL_EVALUATION_RESULTS___/run_<TS>/
#     - pred_<exp_tag>_scienceqa.jsonl / report_<exp_tag>_scienceqa.json
#     - pred_<exp_tag>_mmmu.jsonl      / report_<exp_tag>_mmmu.json
#     - EXTERNAL_EVALUATION_MANIFEST__<TS>.txt
#
# 日志：
#   logs/___EXTERNAL_EVALUATION_PIPELINE_LOGS___/run_<TS>/
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

LOG_DIR="${ROOT}/logs/___EXTERNAL_EVALUATION_PIPELINE_LOGS___/run_${TS}"
RESULT_DIR="${ROOT}/outputs/___EXTERNAL_EVALUATION_RESULTS___/run_${TS}"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

log_step() {
    local msg="$1"
    echo "[${TS_ISO}] [STAGE-07] ${msg}" | append_master
}

# ── 参数解析 ──────────────────────────────────────────────────────────────
# 支持三种方式：命令行参数 / 环境变量 / 自动读取 Stage 06 结果
EXPLICIT_CKPT=""
EXP_TAG=""
PEFT_CONFIG="configs/model/peft.yaml"
MODEL_CFG="${MODEL_CFG:-configs/model/backbone_main.yaml}"
GENERATION_CFG="${GENERATION_CFG:-configs/model/generation.yaml}"
LIMIT="${LIMIT:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)  EXPLICIT_CKPT="$2"; shift 2 ;;
        --exp-tag)     EXP_TAG="$2";       shift 2 ;;
        --peft-config) PEFT_CONFIG="$2";   shift 2 ;;
        --limit)       LIMIT="$2";         shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── 确定 checkpoint 路径 ──────────────────────────────────────────────────

if [[ -n "${EXPLICIT_CKPT}" ]]; then
    # 方式 B/C：直接使用命令行参数
    CKPT_PATH="${EXPLICIT_CKPT}"
    : "${EXP_TAG:=$(basename "${CKPT_PATH}")}"    # 未指定 exp_tag 时用目录名
    log_step "Checkpoint 来源：命令行参数"

elif [[ -n "${CKPT_PATH:-}" ]]; then
    # 环境变量覆盖
    : "${EXP_TAG:=$(basename "${CKPT_PATH}")}"
    log_step "Checkpoint 来源：环境变量 CKPT_PATH"

else
    # 方式 A：自动读取 Stage 06 的最优 best_checkpoint.json
    LATEST_SELECTION_DIR=$(find "${ROOT}/outputs/___SELECTION_EXPORT_FINAL_RESULTS___" \
        -maxdepth 1 -type d -name "run_*" 2>/dev/null | sort -V | tail -n 1 || true)

    if [[ -n "${LATEST_SELECTION_DIR}" && -f "${LATEST_SELECTION_DIR}/best_checkpoint.json" ]]; then
        BEST_CHECKPOINT_JSON="${LATEST_SELECTION_DIR}/best_checkpoint.json"
        BEST_REPORT_PATH=$(python -c \
            "import json; print(json.load(open('${BEST_CHECKPOINT_JSON}'))['best']['report_path'])")
        CKPT_NAME=$(basename "${BEST_REPORT_PATH}" | sed 's/report_//' | sed 's/\.json//')
        CKPT_PATH="${CKPT_PATH:-${ROOT}/outputs/checkpoints/joint_docvqa_chartqa/${CKPT_NAME}}"
        [[ ! -d "${CKPT_PATH}" ]] && CKPT_PATH="${ROOT}/outputs/checkpoints/${CKPT_NAME}"
        : "${EXP_TAG:=E3_lora_${CKPT_NAME}}"
        log_step "Checkpoint 来源：Stage 06 best_checkpoint.json → ${CKPT_PATH}"
    else
        log_step "ERROR: 未找到 Stage 06 的 best_checkpoint.json。"
        log_step "       请先运行 Stage 06，或通过 --checkpoint 显式指定 checkpoint 路径。"
        exit 1
    fi
fi

# ── 前置检查 ──────────────────────────────────────────────────────────────
log_step "=== External Generalization Evaluation START ==="
log_step "Checkpoint : ${CKPT_PATH}"
log_step "Exp tag    : ${EXP_TAG}"
log_step "PEFT config: ${PEFT_CONFIG}"
log_step "Datasets   : ScienceQA + MMMU"

if [[ ! -d "${CKPT_PATH}" ]]; then
    log_step "ERROR: Checkpoint 目录不存在: ${CKPT_PATH}"
    log_step "       通过 --checkpoint <path> 指定有效的 checkpoint 目录。"
    exit 1
fi

LIMIT_ARGS=()
[[ -n "${LIMIT}" ]] && LIMIT_ARGS+=(--limit "${LIMIT}")

# ── 工具函数：带日志的命令执行 ────────────────────────────────────────────
run_cmd() {
    local tag="$1"; shift
    local step_log="${LOG_DIR}/ext_eval_step__${tag}__${TS}.log"
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

# ── 外部评测：ScienceQA + MMMU ─────────────────────────────────────────────
DATASETS=("scienceqa" "mmmu")
REPORT_LIST_FILE="${RESULT_DIR}/EXTERNAL_REPORT_PATHS__${TS}.txt"
: > "${REPORT_LIST_FILE}"

for DS in "${DATASETS[@]}"; do
    VAL_FILE="data/processed/${DS}/validation.jsonl"

    if [[ ! -f "${VAL_FILE}" ]]; then
        log_step "WARN: 跳过 ${DS}，未找到 ${VAL_FILE}"
        continue
    fi

    PRED_OUT="${RESULT_DIR}/pred_${EXP_TAG}_${DS}.jsonl"
    REPORT_OUT="${RESULT_DIR}/report_${EXP_TAG}_${DS}.json"
    TAGGED_OUT="${RESULT_DIR}/tagged_${EXP_TAG}_${DS}.jsonl"

    run_cmd "eval_${DS}" python scripts/validate_checkpoint.py \
        --checkpoint    "${CKPT_PATH}"    \
        --samples       "${VAL_FILE}"     \
        --predictions-output "${PRED_OUT}"   \
        --report-output      "${REPORT_OUT}" \
        --tagged-output      "${TAGGED_OUT}" \
        --model-config       "${MODEL_CFG}"  \
        --generation-config  "${GENERATION_CFG}" \
        --prompt-style structured \
        --resume \
        "${LIMIT_ARGS[@]+${LIMIT_ARGS[@]}}"

    echo "${REPORT_OUT}" >> "${REPORT_LIST_FILE}"
    log_step "Completed: ${DS} → ${REPORT_OUT}"
done

# ── Manifest ──────────────────────────────────────────────────────────────
MANIFEST="${RESULT_DIR}/EXTERNAL_EVALUATION_MANIFEST__${TS}.txt"
{
    echo "EXTERNAL_EVALUATION_MANIFEST | exp_tag=${EXP_TAG}"
    echo "finished_at=$(date -Iseconds)"
    echo "checkpoint_path=${CKPT_PATH}"
    echo "result_dir=${RESULT_DIR}"
    echo "report_list_file=${REPORT_LIST_FILE}"
} > "${MANIFEST}"

cp -f "${MANIFEST}" "${LOG_DIR}/EXTERNAL_EVALUATION_MANIFEST__${TS}.copy.txt" 2>/dev/null || true

log_step "=== STAGE 07 COMPLETE ==="
log_step "Results   : ${RESULT_DIR}"
log_step "Exp tag   : ${EXP_TAG}"
log_step "Checkpoint: ${CKPT_PATH}"
