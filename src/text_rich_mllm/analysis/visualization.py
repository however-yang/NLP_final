from __future__ import annotations
import json
from pathlib import Path

def plot_metrics(summary: dict[str, float], save_path: str | Path) -> None:
    """Generate visualization plots for evaluation metrics."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print(f"Plotting metrics to {save_path}...")
    
    # Filter out non-numeric and top-level overall metrics
    metrics = {k: v for k, v in summary.items() if isinstance(v, (int, float)) and k != "overall"}
    
    if not metrics:
        print("No valid metrics to plot.")
        return

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    datasets = list(metrics.keys())
    scores = list(metrics.values())
    
    ax = sns.barplot(x=datasets, y=scores, hue=datasets, palette="viridis", legend=False)
    
    plt.title("Evaluation Metrics by Dataset", fontsize=14, pad=15)
    plt.ylabel("Score", fontsize=12)
    plt.xlabel("Dataset", fontsize=12)
    plt.ylim(0, 1.0)
    
    # Add value labels
    for i, v in enumerate(scores):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
        
    try:
        plt.tight_layout()
    except Exception:
        pass
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def export_qualitative_cases(records: list, save_path: str | Path) -> None:
    """Export error analysis cases for qualitative review."""
    print(f"Exporting qualitative cases to {save_path}...")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
