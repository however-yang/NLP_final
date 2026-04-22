from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from torch import Tensor


@dataclass
class TRAConfig:
    """TRA-light 超参数，从 tra.yaml 加载。"""

    d_model: int
    r: int
    insert_layers: list[int]
    task_names: list[str]
    dropout: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "TRAConfig":
        from text_rich_mllm.utils import load_yaml

        cfg = load_yaml(path)
        return cls(
            d_model=int(cfg["d_model"]),
            r=int(cfg["r"]),
            insert_layers=list(cfg["insert_layers"]),
            task_names=list(cfg["task_names"]),
            dropout=float(cfg.get("dropout", 0.0)),
        )

    @property
    def task_name_to_id(self) -> dict[str, int]:
        """dataset_name → task_id 映射（由 task_names 列表顺序决定）。"""
        return {name: i for i, name in enumerate(self.task_names)}

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)


class TRABlock(nn.Module):
    """
    Text-Rich Adapter Block.

    在指定 decoder 层的 hidden states 输出后插入，执行：
      1. Task Conditioning：将 task_id 对应的 embedding 加到序列上
      2. Bottleneck Adapter：W_up(GELU(W_down(Z)))
      3. Gated Residual：H' = H + sigmoid(W_gate(H_mean)) * A

    初始化策略（保证 Stage 2 训练开始时等价恒等映射）：
      - W_down: normal(0, 0.02)
      - W_up:   zeros      ← 关键：初始时 adapter 输出为 0
      - W_gate: zeros      ← 关键：初始门控值为 sigmoid(0)=0.5，但 W_up=0 抵消
      - TaskEmb: normal(0, 0.02)
    """

    def __init__(self, d_model: int, r: int, n_tasks: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.task_emb = nn.Embedding(n_tasks, d_model)
        self.W_down = nn.Linear(d_model, r, bias=False)
        self.W_up = nn.Linear(r, d_model, bias=False)
        self.W_gate = nn.Linear(d_model, 1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.task_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.W_down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.W_up.weight)    # zero-init：初始时 adapter 不改变 H
        nn.init.zeros_(self.W_gate.weight)  # zero-init

    def forward(self, H: "Tensor", task_ids: "Tensor") -> "Tensor":
        """
        Args:
            H:        (B, N, d_model)  当前层的 hidden states
            task_ids: (B,)             每个样本的 task_id（int）

        Returns:
            H':       (B, N, d_model)  经 TRA 调整后的 hidden states
        """
        # 1. Task Conditioning
        t = self.task_emb(task_ids)          # (B, d_model)
        Z = H + t.unsqueeze(1)               # broadcast → (B, N, d_model)

        # 2. Bottleneck Adapter
        A = self.W_up(self.dropout(F.gelu(self.W_down(Z))))  # (B, N, d_model)

        # 3. Gated Residual（门控基于序列均值，避免受序列长度影响）
        H_mean = H.mean(dim=1, keepdim=True)          # (B, 1, d_model)
        g = torch.sigmoid(self.W_gate(H_mean))         # (B, 1, 1)
        return H + g * A
