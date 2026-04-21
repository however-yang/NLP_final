#!/usr/bin/env bash
# [阶段 4] PEFT 训练：直接调用 scripts/run_training_pipeline.sh。
#
# 数据：训练 jsonl 默认使用仓库内 data/processed/...（即 Final/data/processed/...）；
# jsonl 内旧机器绝对路径会在加载时映射到本仓库的 data/ 下（需 TEXT_RICH_MLLM_PROJECT_ROOT，由 run_training_pipeline.sh 设置）。
# 模型与 checkpoint：Hub 底座缓存（HF_*）与相对 output_dir 的 LoRA 输出默认落在 DATA_DISK（默认 /root/autodl-tmp），见 run_training_pipeline.sh。
#
# 等价命令:
#   bash scripts/run_training_pipeline.sh
#
# 透传环境变量: TRAIN_CFG、MODEL_CFG、PEFT_CFG、SEED、DRY_RUN、RESUME_FROM 等。
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[pipeline_stage_04_training] 转调 run_training_pipeline.sh（日志: logs/___TRAINING_PIPELINE_LOGS___/run_<TS>/）"
exec bash "${ROOT}/scripts/run_training_pipeline.sh" "$@"
