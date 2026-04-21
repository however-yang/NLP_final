#!/usr/bin/env bash
# 一键下载并预处理联合训练与外推评测所需数据集。
#
# 约定：
# - 使用 Hugging Face 国内镜像（HF_ENDPOINT，默认 https://hf-mirror.com）
# - 所有 Hub / Datasets 缓存与 HF 相关目录默认落在数据盘 DATA_DISK（默认 /root/autodl-tmp）
# - 项目内 data/ 可为仓库内真实目录（便于打镜像），也可为指向数据盘的软链（大数据集占数据盘）。
#   raw/processed 的 jsonl 与导出的图片均位于 data/ 下。
#
# 前台运行（终端与日志同步）：
#   cd /path/to/Final && bash scripts/build_all_datasets.sh
#
# 后台运行（日志由脚本内 tee 写入带时间戳文件）：
#   cd /path/to/Final && nohup bash scripts/build_all_datasets.sh </dev/null >/dev/null 2>&1 &
#   ls -t logs/build_datasets_*.log | head -1 | xargs tail -f
#
# 可选环境变量：
#   DATA_DISK                数据盘根目录，默认 /root/autodl-tmp
#   SKIP_MMMU=1              跳过 MMMU
#   HF_ENDPOINT              默认 https://hf-mirror.com（其余镜像可显式覆盖）
#   HF_HOME / HF_DATASETS_CACHE / HF_HUB_CACHE  未设置时由 DATA_DISK 推导
#   HF_TOKEN                  可选；设置后 Hub 认证访问，配额更高、下载更稳（勿写入仓库或日志）
#
# 推荐在本机一次性登录（token 仅存用户目录，不会进项目 git）：
#   huggingface-cli login
# 或在当前 shell 临时导出（勿提交到 git）：
#   export HF_TOKEN=hf_xxxx

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p "${ROOT}/logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${ROOT}/logs/build_datasets_${TS}.log"

exec > >(tee -a "${LOG}") 2>&1

echo "=============================================="
echo "build_all_datasets 开始: $(date -Iseconds)"
echo "工作目录: ${ROOT}"
echo "日志文件: ${LOG}"
echo "=============================================="

# --- 数据盘与 Hugging Face 镜像 / 缓存（全部默认在数据盘）---
export DATA_DISK="${DATA_DISK:-/root/autodl-tmp}"
# 必须通过镜像访问 Hub；如需换镜像站点，导出 HF_ENDPOINT 即可
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${DATA_DISK}/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DATA_DISK}/huggingface_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${DATA_DISK}/huggingface_hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_DISK}/transformers_cache}"

mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

echo "--- 环境与路径（下载走镜像，缓存与数据在数据盘）---"
echo "DATA_DISK=${DATA_DISK}"
echo "HF_ENDPOINT=${HF_ENDPOINT}"
echo "HF_HOME=${HF_HOME}"
echo "HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "HF_HUB_CACHE=${HF_HUB_CACHE}"
echo "TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "Hub 认证: 已设置环境变量 HF_TOKEN"
elif [[ -f "${HF_HOME}/token" ]]; then
  echo "Hub 认证: 已检测到 hf login 凭据 (${HF_HOME}/token)，datasets/hub 会使用该 token"
else
  echo "提示: 未检测到 HF_TOKEN 或 ${HF_HOME}/token，Hub 可能为匿名访问；建议: hf auth login 或 export HF_TOKEN"
fi

if [[ -L "${ROOT}/data" ]]; then
  DATA_TARGET="$(readlink -f "${ROOT}/data" 2>/dev/null || readlink "${ROOT}/data")"
  echo "data 软链 -> ${DATA_TARGET}"
  # 悬空软链会导致相对路径 mkdir 失败（FileExistsError: data）；先创建目标目录
  if [[ -n "${DATA_TARGET}" ]]; then
    mkdir -p "${DATA_TARGET}"
    echo "已确保 data 目标目录存在: ${DATA_TARGET}"
  fi
  case "${DATA_TARGET}" in
    "${DATA_DISK}"/*) echo "OK: data 已指向 DATA_DISK 下路径" ;;
    *) echo "警告: data 未落在 DATA_DISK (${DATA_DISK}) 下，raw/processed 可能占满系统盘，请检查软链。" ;;
  esac
elif [[ -d "${ROOT}/data" ]]; then
  echo "警告: data 为普通目录且非软链，请确认其是否在数据盘上。"
else
  echo "警告: 未找到 ${ROOT}/data，请创建并链接到数据盘上的数据目录。"
fi

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  echo "提示: 若 python 非 nlp_final 环境，请先: conda activate nlp_final"
fi

echo "--- ChartQA ---"
python scripts/download_data.py --config configs/data/chartqa.yaml --split train
python scripts/download_data.py --config configs/data/chartqa.yaml --split validation
python scripts/preprocess_chartqa.py --split train
python scripts/preprocess_chartqa.py --split validation

echo "--- DocVQA ---"
python scripts/download_data.py --config configs/data/docvqa.yaml --split train
python scripts/download_data.py --config configs/data/docvqa.yaml --split validation
python scripts/preprocess_docvqa.py --split train
python scripts/preprocess_docvqa.py --split validation

echo "--- TextVQA ---"
python scripts/download_data.py --config configs/data/textvqa.yaml --split train
python scripts/download_data.py --config configs/data/textvqa.yaml --split validation
python scripts/preprocess_textvqa.py --split train
python scripts/preprocess_textvqa.py --split validation

echo "--- InfographicVQA ---"
python scripts/download_data.py --config configs/data/infographicvqa.yaml --split train
python scripts/download_data.py --config configs/data/infographicvqa.yaml --split validation
echo "Infographic 首行字段检查:"
head -1 data/raw/infographicvqa/train.jsonl | python3 -c "import sys,json; print(sorted(json.loads(sys.stdin.read()).keys()))"
python scripts/preprocess_infographicvqa.py --split train
python scripts/preprocess_infographicvqa.py --split validation

echo "--- ScienceQA（验证集）---"
python scripts/download_data.py --config configs/data/scienceqa.yaml --split validation
python scripts/preprocess_scienceqa.py --split validation

if [[ "${SKIP_MMMU:-0}" == "1" ]]; then
  echo "--- MMMU: 已跳过 (SKIP_MMMU=1) ---"
else
  echo "--- MMMU（验证集，子集多、耗时长）---"
  python scripts/download_data.py --config configs/data/mmmu.yaml --split validation
  python scripts/preprocess_mmmu.py --split validation
fi

echo "--- 行数自检 ---"
wc -l data/raw/*/*.jsonl 2>/dev/null | sort -n || true
wc -l data/processed/*/*.jsonl 2>/dev/null | sort -n || true

echo "=============================================="
echo "build_all_datasets 结束: $(date -Iseconds)"
echo "完整日志: ${LOG}"
echo "=============================================="
