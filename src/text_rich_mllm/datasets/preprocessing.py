from __future__ import annotations

from pathlib import Path

from text_rich_mllm.datasets import build_dataset_adapter
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.utils import load_yaml, read_json, read_jsonl, write_json, write_jsonl


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ensure_placeholder_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    from PIL import Image

    Image.new("RGB", (8, 8), color=(245, 245, 245)).save(path)


def _apply_empty_image_placeholder(
    samples: list,
    *,
    placeholder_relative: str | None,
    dataset_name: str,
) -> tuple[list, int]:
    """ScienceQA 等可无图样本：为多模态训练写入统一占位图路径。"""
    if not placeholder_relative or dataset_name != "scienceqa":
        return samples, 0
    root = _project_root()
    ph = Path(placeholder_relative)
    dest = ph if ph.is_absolute() else (root / ph)
    _ensure_placeholder_png(dest)
    abs_path = str(dest.resolve())
    used = 0
    out: list = []
    for s in samples:
        if isinstance(s, UnifiedSample) and not str(s.image_path).strip():
            s.image_path = abs_path
            s.metadata.setdefault("uses_placeholder_image", True)
            used += 1
        out.append(s)
    return out, used


def load_raw_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    payload = read_json(path)
    if isinstance(payload, list):
        return payload
    for key in ("data", "samples", "items", "questions", "annotations"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(f"Could not infer record list from {path}")


def clean_unified_samples(
    samples,
    *,
    check_image_paths: bool = False,
    drop_missing_images: bool = False,
):
    cleaned = []
    stats = {
        "input_samples": len(samples),
        "kept_samples": 0,
        "dropped_empty_question": 0,
        "dropped_duplicate_id": 0,
        "missing_image_path": 0,
        "missing_image_file": 0,
    }
    seen_ids: set[str] = set()

    for sample in samples:
        sample.question = sample.question.strip()
        sample.gold_answer = sample.gold_answer.strip()
        sample.image_path = sample.image_path.strip()

        if not sample.question:
            stats["dropped_empty_question"] += 1
            continue
        if sample.sample_id in seen_ids:
            stats["dropped_duplicate_id"] += 1
            continue
        seen_ids.add(sample.sample_id)

        if not sample.image_path:
            stats["missing_image_path"] += 1
            if drop_missing_images:
                continue
        elif check_image_paths and not sample.image_path.startswith(("http://", "https://")):
            if not Path(sample.image_path).exists():
                stats["missing_image_file"] += 1
                if drop_missing_images:
                    continue

        cleaned.append(sample)

    stats["kept_samples"] = len(cleaned)
    return cleaned, stats


def _merge_placeholder_stat(stats: dict, placeholder_used: int) -> dict:
    if placeholder_used:
        stats = dict(stats)
        stats["placeholder_image_used"] = placeholder_used
    return stats


def convert_raw_records(
    *,
    dataset_name: str,
    input_path: str | Path,
    output_path: str | Path,
    split: str,
    image_root: str | None = None,
    check_image_paths: bool = False,
    drop_missing_images: bool = False,
    stats_path: str | Path | None = None,
    placeholder_for_empty_image: str | None = None,
) -> tuple[int, dict]:
    records = load_raw_records(input_path)
    adapter = build_dataset_adapter(dataset_name)
    samples = adapter.convert_records(records, split=split, image_root=image_root)
    samples, placeholder_used = _apply_empty_image_placeholder(
        samples,
        placeholder_relative=placeholder_for_empty_image,
        dataset_name=dataset_name,
    )
    samples, stats = clean_unified_samples(
        samples,
        check_image_paths=check_image_paths,
        drop_missing_images=drop_missing_images,
    )
    stats = _merge_placeholder_stat(stats, placeholder_used)
    write_jsonl([sample.to_dict() for sample in samples], output_path)
    if stats_path:
        write_json(stats, stats_path)
    return len(samples), stats


def preprocess_from_dataset_config(config_path: str | Path, *, split: str) -> tuple[str, int]:
    config = load_yaml(config_path)
    input_key = f"raw_{split}"
    output_key = f"processed_{split}"
    stats_key = f"stats_{split}"
    count, _stats = convert_raw_records(
        dataset_name=config["name"],
        input_path=config[input_key],
        output_path=config[output_key],
        split=split,
        image_root=config.get("image_root"),
        check_image_paths=config.get("check_image_paths", False),
        drop_missing_images=config.get("drop_missing_images", False),
        stats_path=config.get(stats_key),
        placeholder_for_empty_image=config.get("placeholder_for_empty_image"),
    )
    return str(config[output_key]), count
