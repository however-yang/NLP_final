from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.analysis import tag_prediction_records
from text_rich_mllm.evaluation import UnifiedEvaluator, build_evaluation_report
from text_rich_mllm.evaluation.console_summary import print_evaluation_report_summary
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.utils import read_jsonl, write_json, write_jsonl


def load_prediction_map(path: str) -> dict[str, str]:
    records = read_jsonl(path)
    return {record["sample_id"]: str(record["prediction"]) for record in records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tagged-output")
    parser.add_argument("--metadata-keys", nargs="*", default=[])
    args = parser.parse_args()

    print(
        f"\n[evaluate_model] samples={args.samples!s} predictions={args.predictions!s} "
        f"output_json={args.output!s}",
        flush=True,
    )

    samples = [UnifiedSample.from_dict(record) for record in read_jsonl(args.samples)]
    prediction_map = load_prediction_map(args.predictions)
    missing_prediction_count = sum(1 for sample in samples if sample.sample_id not in prediction_map)

    print(
        f"[evaluate_model] n_samples={len(samples)} n_predictions_file={len(prediction_map)} "
        f"missing_ids={missing_prediction_count}",
        flush=True,
    )

    t0 = time.perf_counter()
    evaluator = UnifiedEvaluator()
    records, summary = evaluator.evaluate(samples, prediction_map)
    tagged_records, error_counts = tag_prediction_records(records)
    summary["error_counts"] = error_counts
    summary["missing_prediction_count"] = missing_prediction_count
    report = build_evaluation_report(tagged_records, summary, metadata_keys=args.metadata_keys)
    elapsed = time.perf_counter() - t0

    write_json(report, args.output)
    if args.tagged_output:
        write_jsonl([record.to_dict() for record in tagged_records], args.tagged_output)

    print(f"[evaluate_model] scoring Wall time: {elapsed:.2f}s ({len(samples) / max(elapsed, 1e-9):.0f} samples/s)", flush=True)
    print_evaluation_report_summary(report)


if __name__ == "__main__":
    main()
