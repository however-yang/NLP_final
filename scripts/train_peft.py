from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _bootstrap_hf_cache_env() -> None:
    """若已设置 DATA_DISK 或 TEXT_RICH_MLLM_MODEL_DISK，且未显式导出 HF_*，则将 Hub 权重缓存默认落到该盘。"""
    import os

    base = os.environ.get("TEXT_RICH_MLLM_MODEL_DISK", "").strip() or os.environ.get("DATA_DISK", "").strip()
    if not base:
        return
    pairs = (
        ("HF_HOME", "hf_home"),
        ("HF_HUB_CACHE", "huggingface_hub"),
        ("HF_DATASETS_CACHE", "huggingface_cache"),
        ("TRANSFORMERS_CACHE", "transformers_cache"),
    )
    for env_key, sub in pairs:
        os.environ.setdefault(env_key, os.path.join(base, sub))
    for _, sub in pairs:
        os.makedirs(os.path.join(base, sub), exist_ok=True)


_bootstrap_hf_cache_env()

from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.models.load_backbone import load_model_bundle
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.training.collator import build_training_examples
from text_rich_mllm.training.mixing import mix_training_samples
from text_rich_mllm.training.trainer import run_training, run_training_with_tra
from text_rich_mllm.utils import get_logger, load_yaml, read_jsonl, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-config", required=True)
    parser.add_argument("--model-config", default="configs/model/backbone_main.yaml")
    parser.add_argument("--peft-config", default="configs/model/peft.yaml")
    parser.add_argument("--tra-config", default=None,
                        help="TRA-light 配置文件路径（configs/model/tra.yaml）。"
                             "不传则运行标准 LoRA/DoRA SFT（E3/E4）；"
                             "传入则运行 TRA-light Stage 2 训练（E5）。")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()

    logger = get_logger("train_peft")
    set_seed(args.seed)
    train_cfg = load_yaml(args.train_config)
    model_cfg = load_yaml(args.model_config)
    peft_cfg = load_yaml(args.peft_config)

    train_samples = []
    for path in train_cfg["train_files"]:
        train_samples.extend(UnifiedSample.from_dict(record) for record in read_jsonl(path))
    eval_samples = []
    for path in train_cfg.get("validation_files", []):
        eval_samples.extend(UnifiedSample.from_dict(record) for record in read_jsonl(path))

    if args.dry_run:
        mixed_samples = mix_training_samples(
            train_samples,
            strategy=train_cfg.get("sampling", "balanced"),
        )
        examples = build_training_examples(
            mixed_samples,
            prompt_style=train_cfg.get("prompt_style", PromptStyle.STRUCTURED.value),
        )
        logger.info("Prepared training run: %s", train_cfg["experiment_name"])
        logger.info("Loaded %s samples and built %s mixed training examples.", len(train_samples), len(examples))
        logger.info("Prompt style: %s", train_cfg.get("prompt_style", PromptStyle.STRUCTURED.value))
        logger.info("Dry run complete. No model weights or Trainer execution were started.")
        return

    processor, model = load_model_bundle(**model_cfg)

    if args.tra_config:
        # E5 路径：TRA-light Stage 2
        logger.info("TRA-light 模式已开启，配置文件：%s", args.tra_config)
        model, train_examples, trainer = run_training_with_tra(
            model=model,
            processor=processor,
            train_samples=train_samples,
            peft_config=peft_cfg,
            train_config=train_cfg,
            tra_config_path=args.tra_config,
            eval_samples=eval_samples,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    else:
        # E3/E4 路径：标准 LoRA / DoRA SFT
        model, train_examples, trainer = run_training(
            model=model,
            processor=processor,
            train_samples=train_samples,
            peft_config=peft_cfg,
            train_config=train_cfg,
            eval_samples=eval_samples,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )

    logger.info("Prepared training run: %s", train_cfg["experiment_name"])
    logger.info("Loaded %s samples and built %s training examples.", len(train_samples), len(train_examples))
    logger.info("Model bundle: %s, processor: %s", model.__class__.__name__, processor.__class__.__name__)
    logger.info("Training finished. Final checkpoint directory: %s", trainer.args.output_dir)


if __name__ == "__main__":
    main()

