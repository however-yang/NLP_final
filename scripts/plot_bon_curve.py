"""
plot_bon_curve.py — 绘制 Best-of-N Scaling 曲线（Figure 3）

从 inference_best_of_n.py --scaling-curve 产出的 JSON 文件读取数据，
绘制多数据集的 BoN scaling 曲线（横轴 N，纵轴 score）。

用法：
  python scripts/plot_bon_curve.py \\
    --curves e3_bon_curve.json e8_bon_curve.json \\
    --labels "E3 (LoRA)" "E8 (TRA+GRPO)" \\
    --output outputs/figures/bon_scaling_curve.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def plot_bon_scaling(
    curve_paths: list[str],
    labels: list[str],
    output_path: str,
    *,
    datasets: list[str] | None = None,
) -> None:
    """
    绘制 BoN Scaling 曲线。

    支持多条曲线（e.g., E3 vs E8）和多个数据集子图。

    参数：
      curve_paths : 每个实验的 curve JSON 路径列表
      labels      : 每条曲线的图例标签
      datasets    : 要绘制的数据集子集（None = 全部）
      output_path : 输出图片路径（.png）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

    # ── 加载数据 ──────────────────────────────────────────────────────────
    all_data = []
    for path in curve_paths:
        with open(path, encoding="utf-8") as f:
            all_data.append(json.load(f))

    # 所有数据集
    if datasets is None:
        ds_set: set[str] = set()
        for d in all_data:
            ds_set.update(d["dataset_scores"].keys())
        datasets = sorted(ds_set)

    n_datasets = len(datasets)
    n_values = all_data[0]["n_values"]

    # ── 颜色配置 ──────────────────────────────────────────────────────────
    COLORS = [
        "#4C72B0",  # E3 蓝
        "#DD8452",  # E8 橙
        "#55A868",  # 绿
        "#C44E52",  # 红
    ]
    MARKERS = ["o", "s", "^", "D"]

    # ── 绘图布局：左侧若干数据集子图 + 右侧 Overall ─────────────────────
    n_cols = n_datasets + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4), sharey=False)
    if n_cols == 1:
        axes = [axes]

    def _plot_one(ax, scores_by_n: list[float], label: str, color: str, marker: str) -> None:
        ax.plot(
            n_values, scores_by_n,
            marker=marker, color=color, label=label,
            linewidth=2.0, markersize=7,
        )

    # ── 各数据集子图 ───────────────────────────────────────────────────────
    for col_idx, ds in enumerate(datasets):
        ax = axes[col_idx]
        for exp_idx, (data, label) in enumerate(zip(all_data, labels)):
            ds_scores = data["dataset_scores"].get(ds, {})
            scores = [ds_scores.get(f"n={N}", 0.0) for N in n_values]
            _plot_one(ax, scores, label, COLORS[exp_idx % len(COLORS)], MARKERS[exp_idx % len(MARKERS)])
        ax.set_title(ds.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xlabel("N (# candidates)", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_xticks(n_values)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)

    # ── Overall 子图（最右侧）─────────────────────────────────────────────
    ax_overall = axes[-1]
    for exp_idx, (data, label) in enumerate(zip(all_data, labels)):
        overall = data["overall"]
        scores = [overall.get(f"n={N}", 0.0) for N in n_values]
        _plot_one(ax_overall, scores, label, COLORS[exp_idx % len(COLORS)], MARKERS[exp_idx % len(MARKERS)])

    ax_overall.set_title("Overall", fontsize=11, fontweight="bold")
    ax_overall.set_xlabel("N (# candidates)", fontsize=10)
    ax_overall.set_ylabel("Score", fontsize=10)
    ax_overall.set_xticks(n_values)
    ax_overall.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: str(int(x))))
    ax_overall.grid(True, alpha=0.3, linestyle="--")
    ax_overall.legend(fontsize=8)

    # ── 整体标题 ──────────────────────────────────────────────────────────
    fig.suptitle(
        "Best-of-N Test-time Scaling (CVPR 2026 ViSCALE direction)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"[plot_bon_curve] Saved to: {output_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BoN Scaling Curve (Figure 3)")
    parser.add_argument("--curves", nargs="+", required=True,
                        help="curve JSON 文件路径列表（由 inference_best_of_n.py --scaling-curve 产出）")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="每条曲线的图例标签（数量须与 --curves 一致）")
    parser.add_argument("--output", required=True,
                        help="输出图片路径（.png）")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="要绘制的数据集（默认全部）")
    args = parser.parse_args()

    if len(args.curves) != len(args.labels):
        parser.error("--curves 和 --labels 数量必须一致")

    plot_bon_scaling(
        curve_paths=args.curves,
        labels=args.labels,
        output_path=args.output,
        datasets=args.datasets,
    )


if __name__ == "__main__":
    main()
