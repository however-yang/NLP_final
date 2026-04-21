from __future__ import annotations

import json
from pathlib import Path


def _normalize_torch_dtype(torch_dtype):
    if torch_dtype is None:
        return None
    if not isinstance(torch_dtype, str):
        return torch_dtype
    import torch

    lookup = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return lookup.get(torch_dtype.lower(), torch_dtype)


def _is_peft_adapter_dir(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").is_file()


def _load_pretrained_model_classes(model_name: str, *, trust_remote_code: bool = True, **kwargs):
    """依次尝试 VL / CausalLM，与 Hub 全量权重加载逻辑一致。"""
    from transformers import AutoModelForCausalLM

    model = None
    last_error = None
    model_classes = []
    try:
        from transformers import AutoModelForImageTextToText

        model_classes.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForVision2Seq

        model_classes.append(AutoModelForVision2Seq)
    except ImportError:
        pass
    model_classes.append(AutoModelForCausalLM)

    for model_class in model_classes:
        try:
            model = model_class.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs)
            break
        except Exception as exc:
            last_error = exc
    if model is None:
        raise RuntimeError(f"Unable to load base model {model_name}") from last_error
    return model


def load_model_bundle(model_name: str, *, processor_name: str | None = None, trust_remote_code: bool = True, **kwargs):
    """
    加载 processor + model。
    - Hub / 本地「完整权重」目录：直接 from_pretrained。
    - **PEFT LoRA 目录**（含 adapter_config.json）：先按 adapter 内 base_model_name_or_path 加载底座，
      再用 PeftModel.from_pretrained 挂载适配器（训练产物推理必需）。
    """
    from transformers import AutoProcessor

    kwargs = dict(kwargs)
    raw_dt = kwargs.pop("torch_dtype", None)
    if raw_dt is None:
        raw_dt = kwargs.pop("dtype", None)
    norm_dt = _normalize_torch_dtype(raw_dt)
    if norm_dt is not None:
        kwargs["dtype"] = norm_dt
    kwargs.setdefault("low_cpu_mem_usage", True)

    path = Path(model_name)
    if _is_peft_adapter_dir(path):
        cfg_path = path / "adapter_config.json"
        acfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base_id = acfg.get("base_model_name_or_path") or acfg.get("model_name_or_path")
        if not base_id:
            raise ValueError(f"{cfg_path} 缺少 base_model_name_or_path，无法挂载 LoRA")

        proc_src = None
        if (path / "tokenizer_config.json").is_file() or (path / "processor_config.json").is_file():
            proc_src = str(path)
        elif processor_name:
            proc_src = processor_name
        else:
            proc_src = base_id

        processor = AutoProcessor.from_pretrained(proc_src, trust_remote_code=trust_remote_code)
        base_model = _load_pretrained_model_classes(base_id, trust_remote_code=trust_remote_code, **kwargs)

        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(path))
        return processor, model

    processor = AutoProcessor.from_pretrained(processor_name or model_name, trust_remote_code=trust_remote_code)
    model = _load_pretrained_model_classes(model_name, trust_remote_code=trust_remote_code, **kwargs)
    return processor, model


def load_model_bundle_with_optional_checkpoint(*, checkpoint: str | None, model_config: dict):
    """
    供推理/验证脚本使用：传入 checkpoint（Hub 全量、合并权重目录或 LoRA adapter 目录）时，
    仍合并 backbone yaml 里的 dtype / device_map / trust_remote_code，与 train_peft 加载条件对齐。
    """
    if checkpoint:
        extras = {k: v for k, v in model_config.items() if k != "model_name"}
        return load_model_bundle(checkpoint, **extras)
    return load_model_bundle(**model_config)
