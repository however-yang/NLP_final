from __future__ import annotations


def composite_validation_score(
    metric_summary: dict[str, float],
    *,
    dataset_weights: dict[str, float] | None = None,
) -> float:
    # 只聚合「浮点指标」；排除 overall/slices、以及 num_predictions 等整数计数（否则会污染 composite）。
    skip = {"overall", "error_counts", "slices", "invalid_output_rate", "num_predictions", "missing_prediction_count"}
    dataset_metrics = {
        key: value
        for key, value in metric_summary.items()
        if key not in skip and isinstance(value, float)
    }
    if not dataset_metrics:
        return 0.0
    if not dataset_weights:
        return sum(dataset_metrics.values()) / len(dataset_metrics)
    weighted_sum = 0.0
    total_weight = 0.0
    for dataset_name, value in dataset_metrics.items():
        weight = dataset_weights.get(dataset_name, 1.0)
        weighted_sum += weight * value
        total_weight += weight
    return weighted_sum / max(total_weight, 1.0)
