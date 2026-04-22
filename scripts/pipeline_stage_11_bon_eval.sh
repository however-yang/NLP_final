#!/usr/bin/env bash
# [阶段 10] Best-of-N Test-time Scaling 评测（实验 E9 / E10）
#
# 功能：
#   对指定 checkpoint 运行 Best-of-N 推理，生成预测结果和 scaling 曲线。
#   - 模式 A（默认）：单一 N 推理，输出与 evaluate_model.py 兼容的 JSONL
#   - 模式 B（--scaling-curve）：生成 N=1,2,4,8 的 scaling 曲线 JSON
#
# 使用场景：
#   E9：在 E3（LoRA SFT）checkpoint 上测试 BoN
#   E10：在 E8（TS-GRPO）checkpoint 上测试 BoN，验证 BoN 与 RL 的叠加效果
#
# 用法：
#   # E9：BoN on LoRA checkpoint, N=4, DocVQA
#   bash scripts/pipeline_stage_10_bon_eval.sh \
#     --checkpoint outputs/checkpoints/joint_docvqa_chartqa/checkpoint-1250 \
#     --samples data/processed/docvqa/validation.jsonl \
#     --N 4 \
#     --exp-name E9_lora_bon_n4
#
#   # E10：BoN on GRPO checkpoint + scaling curve
#   bash scripts/pipeline_stage_10_bon_eval.sh \
#     --checkpoint outputs/checkpoints/joint_ts_grpo/final \
#     --samples data/processed/docvqa/validation.jsonl \
#     --scaling-curve \
#     --exp-name E10_grpo_bon_curve \
#     [--tra-config configs/model/tra.yaml]
#
# 日志：
#   logs/___BON_EVAL_LOGS___/run_<TS>/
#
# 产物：
#   outputs/predictions/<exp_name>_bon.jsonl           （单 N 模式）
#   outputs/analysis/<exp_name>_bon_curve.json         （曲线模式）
#   outputs/figures/<exp_name>_bon_curve.png           （曲线图，若绘图成功）
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
LOG_DIR="${ROOT}/logs/___BON_EVAL_LOGS___/run_${TS}"
mkdir -p "$LOG_DIR"

MASTER_LOG="${LOG_DIR}/00_MASTER_ALL_STEPS__${TS}.log"
touch "$MASTER_LOG"
append_master() { tee -a "$MASTER_LOG"; }

log_step() {
    local msg="$1"
    echo "[${TS_ISO}] [STAGE-10] ${msg}" | append_master
}

# ── 参数解析 ──────────────────────────────────────────────────────────────
CHECKPOINT=""
SAMPLES=""
EXP_NAME="bon_eval"
N=4
TEMPERATURE=0.8
MAX_NEW_TOKENS=32
PEFT_CONFIG="configs/model/peft.yaml"
MODEL_CONFIG="configs/model/backbone_main.yaml"
TRA_CONFIG=""
SCALING_CURVE=false
LIMIT=""
PROMPT_STYLE="structured"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)    CHECKPOINT="$2";    shift 2 ;;
        --samples)       SAMPLES="$2";       shift 2 ;;
        --exp-name)      EXP_NAME="$2";      shift 2 ;;
        --N)             N="$2";             shift 2 ;;
        --temperature)   TEMPERATURE="$2";   shift 2 ;;
        --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --peft-config)   PEFT_CONFIG="$2";   shift 2 ;;
        --tra-config)    TRA_CONFIG="$2";    shift 2 ;;
        --scaling-curve) SCALING_CURVE=true; shift ;;
        --limit)         LIMIT="$2";         shift 2 ;;
        --prompt-style)  PROMPT_STYLE="$2";  shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── 前置检查 ──────────────────────────────────────────────────────────────
log_step "=== Best-of-N Test-time Scaling (E9/E10) START ==="
log_step "Experiment    : ${EXP_NAME}"
log_step "Checkpoint    : ${CHECKPOINT:-<pretrained>}"
log_step "Samples       : ${SAMPLES}"
log_step "Mode          : $([ "${SCALING_CURVE}" = "true" ] && echo 'scaling-curve' || echo "single N=${N}")"
log_step "TRA config    : ${TRA_CONFIG:-<none>}"

if [[ -z "${SAMPLES}" ]]; then
    log_step "ERROR: --samples is required"
    exit 1
fi

if [[ ! -f "${SAMPLES}" ]]; then
    log_step "ERROR: Samples file not found: ${SAMPLES}"
    exit 1
fi

# ── 构建输出路径 ───────────────────────────────────────────────────────────
mkdir -p outputs/predictions outputs/analysis outputs/figures

OUTPUT_JSONL="outputs/predictions/${EXP_NAME}_bon_n${N}.jsonl"
CURVE_JSON="outputs/analysis/${EXP_NAME}_bon_curve.json"
CURVE_PNG="outputs/figures/${EXP_NAME}_bon_curve.png"

# ── 构建公共参数 ───────────────────────────────────────────────────────────
COMMON_ARGS="--samples ${SAMPLES} \
  --model-config ${MODEL_CONFIG} \
  --peft-config  ${PEFT_CONFIG} \
  --prompt-style ${PROMPT_STYLE} \
  --temperature  ${TEMPERATURE} \
  --max-new-tokens ${MAX_NEW_TOKENS}"

[[ -n "${CHECKPOINT}" ]] && COMMON_ARGS="${COMMON_ARGS} --checkpoint ${CHECKPOINT}"
[[ -n "${TRA_CONFIG}" ]] && COMMON_ARGS="${COMMON_ARGS} --tra-config ${TRA_CONFIG}"
[[ -n "${LIMIT}" ]]      && COMMON_ARGS="${COMMON_ARGS} --limit ${LIMIT}"

# ── 运行推理 ──────────────────────────────────────────────────────────────
if [[ "${SCALING_CURVE}" = "true" ]]; then
    log_step "Mode: Scaling Curve (N=1,2,4,8)"
    python scripts/inference_best_of_n.py \
        ${COMMON_ARGS} \
        --scaling-curve \
        --curve-output "${CURVE_JSON}" \
        2>&1 | tee "${LOG_DIR}/01_bon_curve.log" | append_master
    EXIT_CODE=${PIPESTATUS[0]}

    # 若绘图依赖可用，自动绘制曲线图
    if [[ ${EXIT_CODE} -eq 0 ]] && python -c "import matplotlib" 2>/dev/null; then
        log_step "Plotting scaling curve..."
        python scripts/plot_bon_curve.py \
            --curves "${CURVE_JSON}" \
            --labels "${EXP_NAME}" \
            --output "${CURVE_PNG}" \
            2>&1 | append_master || log_step "WARNING: Plotting failed (non-fatal)"
    fi
else
    log_step "Mode: Single BoN (N=${N})"
    python scripts/inference_best_of_n.py \
        ${COMMON_ARGS} \
        --N "${N}" \
        --output "${OUTPUT_JSONL}" \
        --evaluate \
        2>&1 | tee "${LOG_DIR}/01_bon_n${N}.log" | append_master
    EXIT_CODE=${PIPESTATUS[0]}
fi

# ── 结果记录 ──────────────────────────────────────────────────────────────
if [[ ${EXIT_CODE} -eq 0 ]]; then
    log_step "BoN eval SUCCEEDED."
    echo "{\"stage\": \"10_bon\", \"status\": \"success\", \"exp\": \"${EXP_NAME}\", \
\"ts\": \"${TS_ISO}\", \"N\": ${N}, \"checkpoint\": \"${CHECKPOINT}\"}" \
        > "${LOG_DIR}/manifest.json"
else
    log_step "BoN eval FAILED (exit code: ${EXIT_CODE})."
    echo "{\"stage\": \"10_bon\", \"status\": \"failed\", \"exit_code\": ${EXIT_CODE}, \"ts\": \"${TS_ISO}\"}" \
        > "${LOG_DIR}/manifest.json"
    exit ${EXIT_CODE}
fi

log_step "=== STAGE 10 COMPLETE ==="
[[ "${SCALING_CURVE}" = "true" ]] && log_step "Curve JSON: ${CURVE_JSON}" \
                                  && log_step "Curve PNG:  ${CURVE_PNG}"
[[ "${SCALING_CURVE}" != "true" ]] && log_step "Predictions: ${OUTPUT_JSONL}"
