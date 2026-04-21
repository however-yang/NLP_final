#!/usr/bin/env bash
# [阶段 3] 基线推理与评测：直接调用 scripts/run_baseline_pipeline.sh（日志与产物目录沿用该脚本内约定，勿重复包装以免双份日志）。
#
# 等价命令:
#   bash scripts/run_baseline_pipeline.sh
#
# 透传环境变量: MODEL_CFG、LIMIT、SKIP_DIRECT、SKIP_STRUCTURED、DATA_DISK、HF_* 等（见 run_baseline_pipeline.sh 注释）。
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[pipeline_stage_03_baseline_eval] 转调 run_baseline_pipeline.sh（基线日志目录: logs/___BASELINE_PIPELINE_LOGS___/run_<TS>/）"
exec bash "${ROOT}/scripts/run_baseline_pipeline.sh" "$@"
