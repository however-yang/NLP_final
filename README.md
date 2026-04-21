# Text-Rich MLLM Project

Unified pipeline scaffold for document QA, chart QA, and scientific figure reasoning.

## Scope

This repository matches the proposal structure:

- `DocVQA` and `ChartQA` as training/adaptation domains
- `ScienceQA` and `MMMU` as external evaluation domains
- unified data schema
- task-conditioned prompt builder
- answer normalization and benchmark-aware evaluator
- PEFT-oriented training entrypoints
- error analysis utilities

## Layout

```text
configs/         experiment, model, data, and eval yaml files
data/            raw, processed, and cache directories
scripts/         CLI entrypoints
src/             package source code
tests/           lightweight unit tests
outputs/         predictions, metrics, analysis, figures, checkpoints
docs/            proposal, poster, final report assets
```

## Quick Start

```bash
conda env create -f environment.yml
conda activate nlp_final
pip install -e .
python scripts/check_runtime.py
```

CPU-only fallback:

```bash
conda env create -f environment.cpu.yml
conda activate text-rich-mllm-cpu
pip install -e .
python scripts/check_runtime.py
```

Build unified samples from a raw JSON/JSONL file:

```bash
python scripts/build_unified_data.py --dataset docvqa --input data/raw/docvqa/train.jsonl --output data/processed/docvqa/train.jsonl --split train
```

Pull a split directly from Hugging Face and materialize image fields:

```bash
python scripts/download_data.py --config configs/data/docvqa.yaml --split train
```

Preview prompts without loading a model:

```bash
python scripts/run_zero_shot.py --input data/processed/docvqa/train.jsonl --preview-count 3
python scripts/run_structured_prompt.py --input data/processed/docvqa/train.jsonl --preview-count 3
```

Evaluate an existing prediction file:

```bash
python scripts/evaluate_model.py --samples data/processed/scienceqa/validation.jsonl --predictions outputs/predictions/scienceqa.jsonl --output outputs/metrics/scienceqa_eval.json
python scripts/export_tables_figures.py --report outputs/metrics/scienceqa_eval.json --output outputs/figures/scienceqa_eval.md
```

Dataset-specific preprocessing entrypoints:

```bash
python scripts/preprocess_docvqa.py --split train
python scripts/preprocess_chartqa.py --split validation
python scripts/preprocess_scienceqa.py --split validation
python scripts/preprocess_mmmu.py --split validation
```

Prepare or launch PEFT training:

```bash
python scripts/train_peft.py --train-config configs/train/train_joint.yaml --dry-run
python scripts/train_peft.py --train-config configs/train/train_joint.yaml
```

Run checkpoint inference and validation:

```bash
python scripts/run_inference.py --samples data/processed/scienceqa/validation.jsonl --checkpoint outputs/checkpoints/joint_docvqa_chartqa --output outputs/predictions/scienceqa_validation.jsonl --prompt-style structured --resume
python scripts/validate_checkpoint.py --samples data/processed/mmmu/validation.jsonl --checkpoint outputs/checkpoints/joint_docvqa_chartqa --predictions-output outputs/predictions/mmmu_validation.jsonl --report-output outputs/metrics/mmmu_validation_eval.json --tagged-output outputs/analysis/mmmu_validation_tagged.jsonl --metadata-keys subject --prompt-style structured --resume
python scripts/select_best_checkpoint.py --reports outputs/metrics/docvqa_validation_eval.json outputs/metrics/chartqa_validation_eval.json --weights docvqa=1.0 chartqa=1.0
```

## What Exists Now

This is the first architecture pass. The following pieces are already usable:

- unified sample schema
- dataset adapter registry
- direct-answer and structured-prompt baselines
- prompt builder
- answer normalization
- ANLS / numeric-aware / MCQ evaluation
- sliced evaluation summaries and invalid-output rate tracking
- balanced or square-root joint training sample mixing
- simple heuristic error tagging
- CLI scripts for data conversion, prompt preview, evaluation, and analysis

The model-specific training and inference path is intentionally thin. It is wired for Hugging Face + PEFT, but still expects you to pick the exact backbone and dataset field mapping.
