from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.analysis import tag_prediction_records
from text_rich_mllm.evaluation import UnifiedEvaluator, build_evaluation_report
from text_rich_mllm.inference import generate_predictions
from text_rich_mllm.models.load_backbone import load_model_bundle_with_optional_checkpoint
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.utils import load_yaml, read_jsonl, write_json, write_jsonl
from text_rich_mllm.utils.constants import PromptStyle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--predictions-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--tagged-output", required=True)
    parser.add_argument("--model-config", default="configs/model/backbone_main.yaml")
    parser.add_argument("--generation-config", default="configs/model/generation.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt-style", default=PromptStyle.STRUCTURED.value)
    parser.add_argument("--metadata-keys", nargs="*", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    samples = [UnifiedSample.from_dict(record) for record in read_jsonl(args.samples)]
    generation_config = load_yaml(args.generation_config)
    model_cfg = load_yaml(args.model_config)
    processor, model = load_model_bundle_with_optional_checkpoint(checkpoint=args.checkpoint, model_config=model_cfg)

    existing = {}
    if args.resume and Path(args.predictions_output).exists():
        existing = {record["sample_id"]: str(record["prediction"]) for record in read_jsonl(args.predictions_output)}

    prediction_map = generate_predictions(
        samples=samples,
        model=model,
        processor=processor,
        prompt_style=args.prompt_style,
        generation_config=generation_config,
        output_path=args.predictions_output,
        existing_predictions=existing,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
    )
    write_jsonl([{"sample_id": key, "prediction": value} for key, value in prediction_map.items()], args.predictions_output)

    evaluator = UnifiedEvaluator()
    records, summary = evaluator.evaluate(samples[: args.limit] if args.limit else samples, prediction_map)
    tagged_records, error_counts = tag_prediction_records(records)
    summary["error_counts"] = error_counts
    report = build_evaluation_report(tagged_records, summary, metadata_keys=args.metadata_keys)

    write_json(report, args.report_output)
    write_jsonl([record.to_dict() for record in tagged_records], args.tagged_output)
    print(f"Saved validation report to {args.report_output}")


if __name__ == "__main__":
    main()
