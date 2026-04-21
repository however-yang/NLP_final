from __future__ import annotations

from typing import Any


def print_evaluation_report_summary(report: dict[str, Any], *, title: str = "EVALUATION SUMMARY — 指标摘要") -> None:
    """将 evaluate 报表中的主指标格式化打印到控制台（适合写入日志复查）。"""
    line = "=" * 76
    print(f"\n{line}")
    print(f"  {title}")
    print(line)

    def _fmt_float(x: float) -> str:
        return f"{x:.6f}"

    priority_keys = (
        "num_predictions",
        "missing_prediction_count",
        "overall",
    )
    for pk in priority_keys:
        if pk not in report:
            continue
        val = report[pk]
        if isinstance(val, float):
            print(f"  {pk}: {_fmt_float(val)}")
        else:
            print(f"  {pk}: {val}")

    skip = {
        "num_predictions",
        "missing_prediction_count",
        "overall",
        "invalid_output_rate",
        "error_counts",
        "slices",
    }
    # 已在 priority_keys 中打印的不再重复
    skip |= set(priority_keys)
    for key in sorted(report.keys()):
        if key in skip:
            continue
        val = report[key]
        if isinstance(val, float):
            print(f"  {key}: {_fmt_float(val)}")
        elif isinstance(val, int):
            print(f"  {key}: {val}")

    ior = report.get("invalid_output_rate")
    if isinstance(ior, dict) and ior:
        print("  --- invalid_output_rate (multiple-choice 无效格式占比) ---")
        for ds, rate in sorted(ior.items()):
            if isinstance(rate, float):
                print(f"    {ds}: {_fmt_float(rate)}")
            else:
                print(f"    {ds}: {rate}")

    ec = report.get("error_counts")
    if isinstance(ec, dict) and ec:
        print("  --- error_type counts（自动标签） ---")
        for et, cnt in sorted(ec.items(), key=lambda x: (-x[1], x[0])):
            print(f"    {et}: {cnt}")

    slices = report.get("slices") or {}
    by_ds = slices.get("by_dataset") if isinstance(slices, dict) else None
    if isinstance(by_ds, dict) and by_ds:
        print("  --- slices: by_dataset ---")
        for ds, stats in sorted(by_ds.items()):
            if not isinstance(stats, dict):
                continue
            cnt = stats.get("count", "?")
            ms = stats.get("mean_score")
            if isinstance(ms, float):
                print(f"    {ds}: n={cnt} mean_score={_fmt_float(ms)}")
            else:
                print(f"    {ds}: n={cnt}")

    by_at = slices.get("by_answer_type") if isinstance(slices, dict) else None
    if isinstance(by_at, dict) and by_at:
        print("  --- slices: by_answer_type ---")
        for aty, stats in sorted(by_at.items()):
            if isinstance(stats, dict):
                cnt = stats.get("count", "?")
                ms = stats.get("mean_score")
                if isinstance(ms, float):
                    print(f"    {aty}: n={cnt} mean_score={_fmt_float(ms)}")

    print(f"{line}\n")
