from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from text_rich_mllm.datasets.preprocessing import preprocess_from_dataset_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/textvqa.yaml")
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    output_path, count = preprocess_from_dataset_config(args.config, split=args.split)
    print(f"Wrote {count} TextVQA samples to {output_path}")


if __name__ == "__main__":
    main()
