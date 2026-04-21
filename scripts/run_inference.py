from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.inference import generate_predictions
from text_rich_mllm.models.load_backbone import load_model_bundle_with_optional_checkpoint
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.utils import load_yaml, read_jsonl, write_jsonl
from text_rich_mllm.utils.constants import PromptStyle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-config", default="configs/model/backbone_main.yaml")
    parser.add_argument("--generation-config", default="configs/model/generation.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--prompt-style", default=PromptStyle.STRUCTURED.value)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    samples = [UnifiedSample.from_dict(record) for record in read_jsonl(args.samples)]
    generation_config = load_yaml(args.generation_config)

    banner_ckpt = (
        f"\n[run_inference] samples_file={args.samples!s} n={len(samples)} "
        f"prompt_style={args.prompt_style!r} output={args.output!s} "
        f"checkpoint={args.checkpoint!s}"
    )
    banner_base = (
        f"\n[run_inference] samples_file={args.samples!s} n={len(samples)} "
        f"prompt_style={args.prompt_style!r} output={args.output!s} "
        f"model_config={args.model_config!s}"
    )
    print(banner_ckpt if args.checkpoint else banner_base, flush=True)
    if args.limit is not None:
        print(f"[run_inference] --limit={args.limit}", flush=True)

    model_cfg = load_yaml(args.model_config)
    processor, model = load_model_bundle_with_optional_checkpoint(checkpoint=args.checkpoint, model_config=model_cfg)

    print("[run_inference] model/processor loaded; generating…", flush=True)

    existing = {}
    if args.resume and Path(args.output).exists():
        existing = {record["sample_id"]: str(record["prediction"]) for record in read_jsonl(args.output)}

    prediction_map = generate_predictions(
        samples=samples,
        model=model,
        processor=processor,
        prompt_style=args.prompt_style,
        generation_config=generation_config,
        output_path=args.output,
        existing_predictions=existing,
        limit=args.limit,
        continue_on_error=args.continue_on_error,
    )
    write_jsonl([{"sample_id": key, "prediction": value} for key, value in prediction_map.items()], args.output)
    print(f"Saved {len(prediction_map)} predictions to {args.output}")


if __name__ == "__main__":
    main()
