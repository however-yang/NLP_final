from __future__ import annotations

import os
from pathlib import Path


def infer_repo_root() -> Path:
    """
    代码包位于 <repo>/src/text_rich_mllm/... 时，返回仓库根 <repo>。
    可用环境变量 TEXT_RICH_MLLM_PROJECT_ROOT 覆盖（与 bash 里 `cd` 到 Final 后一致）。
    """
    env = os.environ.get("TEXT_RICH_MLLM_PROJECT_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # .../Final/src/text_rich_mllm/utils/paths.py -> parents[3] == Final
    return Path(__file__).resolve().parents[3]


def resolve_sample_image_path(path: str) -> str:
    """
    将 jsonl 中记录的 image_path 解析为当前机器可读路径。
    - 支持仓库内相对路径（相对 TEXT_RICH_MLLM_PROJECT_ROOT / 推断的 repo 根）。
    - 若记录为其它机器上的绝对路径（如旧 autodl 路径）且文件不存在，则按路径中
      首次出现的「/data/」之后片段拼到本仓库的 data/ 下（与 Final/data 布局一致）。
    """
    raw = (path or "").strip()
    if not raw or raw.startswith(("http://", "https://")):
        return raw

    root = infer_repo_root()
    original = Path(raw)

    try:
        if original.is_absolute() and original.is_file():
            return str(original)
    except OSError:
        pass

    if not original.is_absolute():
        candidate = (root / original).resolve()
        return str(candidate)

    marker = "/data/"
    if marker in raw:
        suffix = raw.split(marker, 1)[1].lstrip("/")
        candidate = (root / "data" / suffix).resolve()
        return str(candidate)

    return raw


def resolve_training_output_dir(output_dir: str | None) -> str:
    """
    训练 yaml 中相对路径的 output_dir 默认写到数据盘，避免 checkpoint 占满系统盘。

    解析顺序（仅对「相对路径」生效；已是绝对路径则原样返回）：
    1) TEXT_RICH_MLLM_CHECKPOINT_ROOT（仅作 checkpoint 根）
    2) DATA_DISK（与脚本里 HF 缓存根一致）
    3) 当前工作目录（与旧行为一致）
    """
    raw = (output_dir or "").strip() or "outputs/checkpoints/default"
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    root = os.environ.get("TEXT_RICH_MLLM_CHECKPOINT_ROOT", "").strip() or os.environ.get("DATA_DISK", "").strip()
    if root:
        return str((Path(root) / p).resolve())
    return str((Path.cwd() / p).resolve())
