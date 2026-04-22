#!/usr/bin/env bash
# [阶段 8] TRA-light Stage 2 训练（实验 E5）
#
# 功能：
#   在 LoRA checkpoint（Stage 1，E3）基础上续训，注入 TRA-light Adapter。
#   TRA 在 Qwen3-VL decoder Layer 16/20/24/28 后插入 Bottleneck + Task Embedding。
#
# 前置条件：
#   - pipeline_stage_04_training.sh 已成功执行，产出 LoRA checkpoint
#   - RESUME_CHECKPOINT 指向有效的 checkpoint 目录
#
# 用法：
#   bash scripts/pipeline_stage_08_tra_training.sh \
#     [--resume-checkpoint <ckpt_dir>] \
#     [--peft-config configs/model/peft.yaml] \
#     [--dry-run]
#
# 日志：
#   logs/___TRA_TRAINING_LOGS___/run_<TS>/
#
# 产物：
#   outputs/checkpoints/joint_tra_light/
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# ── 环境变量 ──────────────────────────────────────────────────────────────
export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

# ── 时间戳与日志 ──────────────────────────────────────────────────────────
TS="$(date +%Y%m%d_%H%M%S)"
TS_ISO="$(date -Iseconds)"
LOG_DIR="${ROOT}/logs/___TRA_TRAINING_LOGS___/run_${TS}"
mkdir -p "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

log_step() {
    local msg="$1"
    echo "[${TS_ISO}] [STAGE-08] ${msg}" | append_master
}

# ── 参数解析 ──────────────────────────────────────────────────────────────
RESUME_CHECKPOINT=""
PEFT_CONFIG="configs/model/peft.yaml"
TRA_CONFIG="configs/model/tra.yaml"
TRAIN_CONFIG="configs/train/train_joint_tra.yaml"
MODEL_CONFIG="configs/model/backbone_main.yaml"
DRY_RUN_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume-checkpoint) RESUME_CHECKPOINT="$2"; shift 2 ;;
        --peft-config)       PEFT_CONFIG="$2";       shift 2 ;;
        --dry-run)           DRY_RUN_FLAG="--dry-run"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── 前置检查 ──────────────────────────────────────────────────────────────
log_step "=== TRA-light Stage 2 Training (E5) START ==="
log_step "TRA config    : ${TRA_CONFIG}"
log_step "PEFT config   : ${PEFT_CONFIG}"
log_step "Train config  : ${TRAIN_CONFIG}"
log_step "Resume ckpt   : ${RESUME_CHECKPOINT:-<none>}"
log_step "Dry run       : ${DRY_RUN_FLAG:-false}"

if [[ ! -f "${TRA_CONFIG}" ]]; then
    log_step "ERROR: TRA config not found: ${TRA_CONFIG}"
    exit 1
fi

if [[ ! -f "${TRAIN_CONFIG}" ]]; then
    log_step "ERROR: Train config not found: ${TRAIN_CONFIG}"
    exit 1
fi

# ── 构建 resume 参数 ───────────────────────────────────────────────────────
RESUME_ARG=""
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
        log_step "ERROR: Resume checkpoint not found: ${RESUME_CHECKPOINT}"
        exit 1
    fi
    RESUME_ARG="--resume-from-checkpoint ${RESUME_CHECKPOINT}"
    log_step "Resuming from: ${RESUME_CHECKPOINT}"
else
    log_step "WARNING: No --resume-checkpoint provided. Training from scratch (not Stage 2 behavior)."
fi

# ── 运行训练 ──────────────────────────────────────────────────────────────
log_step "Launching TRA-light training..."

python scripts/train_peft.py \
    --train-config  "${TRAIN_CONFIG}"  \
    --model-config  "${MODEL_CONFIG}"  \
    --peft-config   "${PEFT_CONFIG}"   \
    --tra-config    "${TRA_CONFIG}"    \
    ${RESUME_ARG}                      \
    ${DRY_RUN_FLAG}                    \
    2>&1 | tee "${LOG_DIR}/01_tra_training.log" | append_master

EXIT_CODE=${PIPESTATUS[0]}

# ── 结果记录 ──────────────────────────────────────────────────────────────
if [[ ${EXIT_CODE} -eq 0 ]]; then
    log_step "TRA-light training SUCCEEDED."
    echo "{\"stage\": \"08_tra\", \"status\": \"success\", \"ts\": \"${TS_ISO}\", \"tra_config\": \"${TRA_CONFIG}\"}" \
        > "${LOG_DIR}/manifest.json"
else
    log_step "TRA-light training FAILED (exit code: ${EXIT_CODE})."
    echo "{\"stage\": \"08_tra\", \"status\": \"failed\", \"exit_code\": ${EXIT_CODE}, \"ts\": \"${TS_ISO}\"}" \
        > "${LOG_DIR}/manifest.json"
    exit ${EXIT_CODE}
fi

log_step "=== STAGE 08 COMPLETE. Checkpoint: outputs/checkpoints/joint_tra_light ==="
