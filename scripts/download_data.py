from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.utils import load_yaml
from text_rich_mllm.utils import get_logger


def _configure_pillow_for_hub_exports() -> None:
    """整页 Doc/Infographic 等图常超过 Pillow 默认像素上限；须在首次 import datasets 之前设置。"""
    try:
        from PIL import Image
    except ImportError:
        return
    raw = os.environ.get("PIL_MAX_IMAGE_PIXELS", "").strip()
    if raw and raw.lower() not in ("none", "off"):
        Image.MAX_IMAGE_PIXELS = int(raw)
    else:
        # 少数环境下 None 仍按默认阈值校验；用足够大的整型覆盖 ~2e8 像素级整页图
        Image.MAX_IMAGE_PIXELS = 2**31 - 1
    _warn_cls = getattr(Image, "DecompressionBombWarning", None)
    if _warn_cls is not None:
        warnings.filterwarnings("ignore", category=_warn_cls)


_configure_pillow_for_hub_exports()


def _pil_image_for_png_save(value):
    """PNG 不支持 CMYK 等模式；导出前转成 RGB（或保留 RGBA/L/P 等 Pillow 可直接写的模式）。"""
    from PIL import Image

    if not isinstance(value, Image.Image):
        return value
    # Pillow 写入 PNG 支持的常见模式；CMYK/YCbCr/LAB 等需先转换
    if value.mode in ("RGB", "RGBA", "L", "1", "P", "LA"):
        return value
    return value.convert("RGB")


def _resolve_data_path(path: str | Path) -> Path:
    """解析到真实路径（含软链），避免悬空 data 软链导致相对路径 mkdir 报 FileExistsError。"""
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p.resolve(strict=False)


def _resolve_hf_cache_dir(raw: str | Path | None) -> str | None:
    """HF_DATASETS_CACHE 或 yaml 中的 hf_cache_dir；相对路径按仓库根解析，便于落在 Final/data 下。"""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s)
    if p.is_absolute():
        return str(p)
    return str((ROOT / p).resolve())


def _serialize_value(value, *, image_dir: Path, key: str, index: int):
    if hasattr(value, "save"):
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{key}_{index}.png"
        img = _pil_image_for_png_save(value)
        img.save(image_path)
        return str(image_path)
    if isinstance(value, dict):
        if "path" in value and value["path"]:
            return value["path"]
        return {
            sub_key: _serialize_value(sub_value, image_dir=image_dir, key=f"{key}_{sub_key}", index=index)
            for sub_key, sub_value in value.items()
        }
    if isinstance(value, list):
        return [
            _serialize_value(item, image_dir=image_dir, key=key, index=item_index)
            for item_index, item in enumerate(value)
        ]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _resolve_hf_split(config: dict, split: str) -> str:
    specific_key = f"hf_{split}_split"
    return str(config.get(specific_key, split))


def _resolve_hf_subsets(config: dict) -> list[str | None]:
    subsets = config.get("hf_subsets")
    if subsets:
        return list(subsets)
    return [config.get("hf_subset")]


def parse_train_val_ratio(spec: str) -> tuple[int, int]:
    """解析「训练:验证」比例，如 8:2 表示验证集条数 = 训练 × 2/8。"""
    spec = spec.strip().replace(" ", "")
    if ":" not in spec:
        raise ValueError(f"无效比例 '{spec}'，应为 train:val，例如 8:2")
    left, _, right = spec.partition(":")
    train_part, val_part = int(left), int(right)
    if train_part <= 0 or val_part <= 0:
        raise ValueError(f"比例两部分须为正整数，收到: {spec}")
    return train_part, val_part


def export_hf_split(config_path: str, split: str, *, limit: int | None = None) -> tuple[str, int]:
    from datasets import load_dataset
    from tqdm.auto import tqdm

    logger = get_logger("download_data")
    config = load_yaml(config_path)
    output_path = _resolve_data_path(config[f"raw_{split}"])
    # 分两次运行 train / validation 时共用同一平面 images/ 会用相同 image_0.png 覆盖；按 CLI split 分子目录避免串图
    _base_img = _resolve_data_path(config.get("image_root", output_path.parent / "images"))
    image_dir = _base_img if config.get("image_flat", False) else (_base_img / split)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    hf_split = _resolve_hf_split(config, split)
    remaining_limit = limit
    global_index = 0

    # 优先使用环境变量 HF_DATASETS_CACHE（与 build_all_datasets.sh / HF 官方约定一致），便于统一落到数据盘
    cache_dir = _resolve_hf_cache_dir(os.environ.get("HF_DATASETS_CACHE") or config.get("hf_cache_dir"))
    load_kwargs: dict = {}
    if cache_dir:
        load_kwargs["cache_dir"] = cache_dir
    if config.get("hf_revision"):
        load_kwargs["revision"] = str(config["hf_revision"])
    if "trust_remote_code" in config:
        load_kwargs["trust_remote_code"] = bool(config["trust_remote_code"])

    for subset in _resolve_hf_subsets(config):
        logger.info(
            "Loading HF dataset=%s subset=%s hf_split=%s (CLI split=%s)",
            config["hf_dataset_name"],
            subset,
            hf_split,
            split,
        )
        # WebDataset 类 Hub 数据集（如 pixparse/docvqa-wds）在非流式下会触发「全部分片元数据
        # + 大量 tar 下载」后才可 random access；指定 --limit 时必须用 streaming，只按需拉取
        # 若干分片即可停止，否则 --limit 无法减少下载时间/流量。
        use_streaming = limit is not None
        load_kwargs_eff = dict(load_kwargs)
        if use_streaming:
            load_kwargs_eff["streaming"] = True
            logger.info(
                "Using streaming=True because --limit is set; only the first %s rows are fetched.",
                limit,
            )

        dataset = load_dataset(
            config["hf_dataset_name"],
            name=subset,
            split=hf_split,
            **load_kwargs_eff,
        )

        export_desc = f"{config.get('name', '')} | CLI:{split} | HF:{hf_split}"

        if use_streaming:
            columns: list[str] | None = None
            row_stream = dataset
            pbar = tqdm(
                row_stream,
                total=remaining_limit,
                desc=f"[{export_desc}] serialize (streaming)",
                unit="row",
                leave=True,
                dynamic_ncols=True,
            )
            for example in pbar:
                if isinstance(example, dict):
                    if columns is None:
                        columns = list(example.keys())
                else:
                    raise TypeError(f"Streaming example must be dict-like, got {type(example)}")
                serialized = {}
                for column in columns:
                    try:
                        value = example[column]
                    except (KeyError, TypeError):
                        value = getattr(example, column, None)
                    serialized[column] = _serialize_value(
                        value, image_dir=image_dir, key=column, index=global_index
                    )
                serialized["_hf_dataset"] = config["hf_dataset_name"]
                serialized["_hf_split"] = hf_split
                serialized["_hf_subset"] = subset
                records.append(serialized)
                global_index += 1
                if remaining_limit is not None:
                    remaining_limit -= 1
                    if remaining_limit <= 0:
                        break
            if remaining_limit is not None and remaining_limit <= 0:
                break
            continue

        if remaining_limit is not None:
            dataset = dataset.select(range(min(remaining_limit, len(dataset))))
        # 某些 Arrow/格式下直接 for row in dataset 再 row.items() 只会带出已解码列，
        # 导致除 image 外文本字段丢失；按 column_names 显式取值可稳定导出全列。
        columns = list(dataset.column_names)
        n_rows = len(dataset)
        # train / validation 等在进度条文案里写清楚，便于两条命令并行时分辨
        row_range = tqdm(
            range(n_rows),
            total=n_rows,
            desc=f"[{export_desc}] serialize",
            unit="row",
            leave=True,
            dynamic_ncols=True,
        )
        for row_index in row_range:
            example = dataset[row_index]
            serialized = {}
            for column in columns:
                try:
                    value = example[column]
                except (KeyError, TypeError):
                    value = getattr(example, column, None)
                serialized[column] = _serialize_value(
                    value, image_dir=image_dir, key=column, index=global_index
                )
            serialized["_hf_dataset"] = config["hf_dataset_name"]
            serialized["_hf_split"] = hf_split
            serialized["_hf_subset"] = subset
            records.append(serialized)
            global_index += 1
        if remaining_limit is not None:
            remaining_limit -= min(remaining_limit, len(dataset))
            if remaining_limit <= 0:
                break
    summary = f"{config.get('name', '')} | CLI:{split} | HF:{hf_split}"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in tqdm(
            records,
            desc=f"[{summary}] write jsonl",
            unit="line",
            leave=True,
            dynamic_ncols=True,
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Exported %s records for %s split %s (HF split: %s)", len(records), config["name"], split, hf_split)
    return str(output_path), len(records)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 Hugging Face 导出原始 jsonl。若使用 --limit，将对该 split 启用 streaming，"
            "只拉取前 N 条，适合 DocVQA-WDS 等大分片数据集。与 --with-matched-validation 联用可一次按 "
            "train:val 比例生成验证子集（默认 8:2）。"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--with-matched-validation",
        action="store_true",
        help=(
            "与 --split train 与 --limit 同时使用时，再导出 validation，条数为 "
            "round(limit × val/train)，比例由 --train-val-ratio 指定（默认 8:2）。"
        ),
    )
    parser.add_argument(
        "--train-val-ratio",
        default="8:2",
        metavar="TRAIN:VAL",
        help="训练集与验证集样本数之比，默认 8:2（若 train 限 5000，则 validation 限 1250）。",
    )
    args = parser.parse_args()

    if args.with_matched_validation:
        if args.split != "train":
            parser.error("--with-matched-validation 仅支持与 --split train 共用")
        if args.limit is None:
            parser.error("--with-matched-validation 需要同时指定 --limit（训练集条数）")
        train_part, val_part = parse_train_val_ratio(args.train_val_ratio)
        val_limit = max(1, round(args.limit * val_part / train_part))

        output_path, count = export_hf_split(args.config, "train", limit=args.limit)
        print(f"Downloaded {count} records to {output_path}")

        output_path_v, count_v = export_hf_split(args.config, "validation", limit=val_limit)
        print(
            f"Downloaded {count_v} records to {output_path_v} "
            f"(matched validation limit={val_limit}, ratio {train_part}:{val_part})"
        )
        return

    output_path, count = export_hf_split(args.config, args.split, limit=args.limit)
    print(f"Downloaded {count} records to {output_path}")


if __name__ == "__main__":
    main()
