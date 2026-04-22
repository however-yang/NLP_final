from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn
    from torch import Tensor

from text_rich_mllm.adapters.text_rich_adapter import TRABlock, TRAConfig


# ---------------------------------------------------------------------------
# 内部工具：获取 Qwen3-VL decoder 层列表
# ---------------------------------------------------------------------------

def _get_decoder_layers(model: "nn.Module") -> "nn.ModuleList":
    """
    兼容 PeftModel 封装（LoRA 已挂载）和原始 Qwen3-VL 模型。
    Qwen3-VL 的 decoder 层路径：model.model.layers
    PeftModel 包装后路径：model.base_model.model.model.layers
    """
    # 依次尝试常见路径
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.base_model.model.model.layers,
        lambda m: m.base_model.model.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            if layers is not None:
                return layers
        except AttributeError:
            continue
    raise AttributeError(
        "无法找到 decoder layers。请确认模型是 Qwen3-VL 系列，"
        "且路径为 model.model.layers 或 model.base_model.model.model.layers。"
    )


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def inject_tra(model: "nn.Module", tra_config: TRAConfig) -> "nn.Module":
    """
    将 TRABlock 以 forward_hook 方式注入指定 decoder 层。
    In-place 修改，返回原 model，兼容 PeftModel（LoRA 已挂载）。

    Hook 方案说明：
      使用 register_forward_hook 在层输出后拦截 hidden states。
      Qwen3-VL decoder 层输出为 tuple，第 0 个元素是 hidden states。
      task_ids 通过 model._tra_task_ids 传递（由 TRATrainer.compute_loss 设置）。
    """
    layers = _get_decoder_layers(model)
    n_layers = len(layers)

    for layer_idx in tra_config.insert_layers:
        if layer_idx >= n_layers:
            raise ValueError(
                f"insert_layer={layer_idx} 超出模型层数 {n_layers}。"
                f"请检查 tra.yaml 中的 insert_layers 配置。"
            )

        block = TRABlock(
            d_model=tra_config.d_model,
            r=tra_config.r,
            n_tasks=tra_config.n_tasks,
            dropout=tra_config.dropout,
        )
        # 将 TRABlock 注册为子模块，使其参数纳入 model.parameters()
        block_name = f"tra_block_{layer_idx}"
        model.add_module(block_name, block)

        # 注册 forward hook（闭包捕获 block 和 block_name）
        def _make_hook(tra_block: TRABlock) -> callable:
            def hook(module, inputs, outputs):  # noqa: ARG001
                # outputs: tuple，第 0 个元素是 hidden states (B, N, d)
                # 其余元素是 KV cache 等，原样保留
                hidden = outputs[0]
                task_ids = getattr(model, "_tra_task_ids", None)
                if task_ids is None:
                    # 推理时如果未设置 task_ids，跳过 TRA（退化为恒等）
                    return outputs
                task_ids = task_ids.to(hidden.device)
                hidden_out = tra_block(hidden, task_ids)
                return (hidden_out,) + outputs[1:]
            return hook

        layers[layer_idx].register_forward_hook(_make_hook(block))

    # 初始化 _tra_task_ids 为 None
    model._tra_task_ids = None
    return model


def get_tra_parameters(model: "nn.Module") -> list:
    """
    提取所有 TRA 参数（以 'tra_block_' 为前缀的子模块）。
    供优化器分组或冻结检查使用。
    """
    params = []
    for name, module in model.named_modules():
        if name.startswith("tra_block_"):
            params.extend(module.parameters())
    return params


def freeze_non_tra_non_lora(model: "nn.Module") -> None:
    """
    冻结所有非 LoRA、非 TRA 参数。
    LoRA 参数名含 'lora_'；TRA 参数名含 'tra_block_'。
    调用后可通过 assert 验证梯度设置是否正确。
    """
    for name, param in model.named_parameters():
        is_lora = "lora_" in name
        is_tra = "tra_block_" in name
        param.requires_grad = is_lora or is_tra


def save_tra_state(model: "nn.Module", save_path: str) -> None:
    """
    仅保存 TRA 参数的 state_dict（不含 LoRA 权重）。
    用于轻量导出或单独加载 TRA 模块。
    """
    tra_state = {
        name: param
        for name, param in model.state_dict().items()
        if "tra_block_" in name
    }
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tra_state, save_path)


def load_tra_state(model: "nn.Module", load_path: str) -> None:
    """
    从文件加载 TRA state_dict 到已注入 TRA hooks 的 model。
    使用 strict=False 允许 model 有额外参数（LoRA 等）。
    """
    tra_state = torch.load(load_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(tra_state, strict=False)
    # 过滤掉非 TRA 的 missing keys（属于正常现象）
    tra_missing = [k for k in missing if "tra_block_" in k]
    if tra_missing:
        raise RuntimeError(f"TRA state_dict 加载失败，缺失 keys: {tra_missing}")
