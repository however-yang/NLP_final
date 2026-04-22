#!/usr/bin/env bash
# [阶段 7] DoRA SFT 训练（实验 E4）
#
# 功能：
#   用 DoRA（Decomposed Rank Adaptation）替换 LoRA，
#   其余超参与 E3 完全一致，作为 PEFT 方法维度的消融实验。
#   DoRA 将权重分解为 magnitude（标量）+ direction（LoRA 低秩更新），
#   比标准 LoRA 更接近 full fine-tuning 的学习行为。
#
# 用法：
#   bash scripts/pipeline_stage_07_dora_training.sh [--dry-run]
#
# 日志：
#   logs/___DORA_TRAINING_LOGS___/run_<TS>/
#
# 产物：
#   outputs/checkpoints/joint_dora/  （train_config 中的 output_dir）
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
LOG_DIR="${ROOT}/logs/___DORA_TRAINING_LOGS___/run_${TS}"
mkdir -p "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

log_step() {
    local msg="$1"
    echo "[${TS_ISO}] [STAGE-07] ${msg}" | append_master
}

# ── 参数解析 ──────────────────────────────────────────────────────────────
DRY_RUN_FLAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN_FLAG="--dry-run"; shift ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

PEFT_CONFIG="configs/model/peft_dora.yaml"
TRAIN_CONFIG="configs/train/train_joint.yaml"   # 与 E3 完全相同（只换 peft_config）
MODEL_CONFIG="configs/model/backbone_main.yaml"

# ── 前置检查 ──────────────────────────────────────────────────────────────
log_step "=== DoRA SFT Training (E4) START ==="
log_step "PEFT config  : ${PEFT_CONFIG}  (use_dora=true)"
log_step "Train config : ${TRAIN_CONFIG}"
log_step "Dry run      : ${DRY_RUN_FLAG:-false}"

if [[ ! -f "${PEFT_CONFIG}" ]]; then
    log_step "ERROR: PEFT config not found: ${PEFT_CONFIG}"
    exit 1
fi

# ── 临时覆盖 output_dir（区分 E3 和 E4 的 checkpoint 目录）──────────────
# 将 train_joint.yaml 的 output_dir 覆盖为 joint_dora
# 方案：用临时 yaml，避免修改原文件
TMP_TRAIN_CFG="${LOG_DIR}/train_joint_dora_tmp.yaml"
python - <<PYEOF
import yaml, pathlib
src = pathlib.Path("${TRAIN_CONFIG}").read_text()
cfg = yaml.safe_load(src)
cfg["output_dir"] = "outputs/checkpoints/joint_dora"
cfg["experiment_name"] = "joint_dora"
pathlib.Path("${TMP_TRAIN_CFG}").write_text(yaml.dump(cfg))
print("Temporary DoRA train config written to: ${TMP_TRAIN_CFG}")
PYEOF

log_step "Temporary train config (DoRA): ${TMP_TRAIN_CFG}"

# ── 运行训练 ──────────────────────────────────────────────────────────────
log_step "Launching DoRA training..."

python scripts/train_peft.py \
    --train-config "${TMP_TRAIN_CFG}"  \
    --model-config "${MODEL_CONFIG}"   \
    --peft-config  "${PEFT_CONFIG}"    \
    ${DRY_RUN_FLAG}                    \
    2>&1 | tee "${LOG_DIR}/01_dora_training.log" | append_master

EXIT_CODE=${PIPESTATUS[0]}

# ── 结果记录 ──────────────────────────────────────────────────────────────
if [[ ${EXIT_CODE} -eq 0 ]]; then
    log_step "DoRA training SUCCEEDED."
    echo "{\"stage\": \"07_dora\", \"status\": \"success\", \"ts\": \"${TS_ISO}\", \"peft\": \"dora\"}" \
        > "${LOG_DIR}/manifest.json"
else
    log_step "DoRA training FAILED (exit code: ${EXIT_CODE})."
    echo "{\"stage\": \"07_dora\", \"status\": \"failed\", \"exit_code\": ${EXIT_CODE}, \"ts\": \"${TS_ISO}\"}" \
        > "${LOG_DIR}/manifest.json"
    exit ${EXIT_CODE}
fi

log_step "=== STAGE 07 COMPLETE. Checkpoint: outputs/checkpoints/joint_dora ==="
