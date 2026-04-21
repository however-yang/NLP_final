from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.models.generation_utils import run_generation
from text_rich_mllm.models.load_backbone import load_model_bundle
from text_rich_mllm.prompts import PromptBuilder
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.utils import load_yaml, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--preview-count", type=int, default=3)
    parser.add_argument("--model-config", default="configs/model/backbone_main.yaml")
    parser.add_argument("--generation-config", default="configs/model/generation.yaml")
    parser.add_argument("--output")
    parser.add_argument("--run-model", action="store_true")
    args = parser.parse_args()

    samples = [UnifiedSample.from_dict(record) for record in read_jsonl(args.input)]
    builder = PromptBuilder(style=PromptStyle.STRUCTURED.value)

    for sample in samples[: args.preview_count]:
        print("=" * 80)
        print(sample.sample_id)
        print(builder.build(sample))

    if not args.run_model:
        return

    model_cfg = load_yaml(args.model_config)
    generation_cfg = load_yaml(args.generation_config)
    processor, model = load_model_bundle(**model_cfg)

    predictions = []
    for sample in samples:
        prompt = builder.build(sample)
        raw_prediction = run_generation(model, processor, sample.image_path, prompt, generation_cfg)
        predictions.append({"sample_id": sample.sample_id, "prediction": raw_prediction})

    if args.output:
        write_jsonl(predictions, args.output)
        print(f"Saved {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()

