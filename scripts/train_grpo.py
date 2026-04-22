"""
train_grpo.py — Task-Stratified GRPO 训练入口脚本（实验 E8）

用法：
  python scripts/train_grpo.py \
    --train-config configs/train/train_joint_grpo.yaml \
    --model-config configs/model/backbone_main.yaml \
    --peft-config  configs/model/peft.yaml \
    --checkpoint   outputs/checkpoints/joint_tra_light/checkpoint-best \
    [--tra-config  configs/model/tra.yaml] \
    [--seed 42] \
    [--dry-run]

参数说明：
  --checkpoint   必须提供：SFT checkpoint 目录（E3 LoRA 或 E5 TRA checkpoint）
  --tra-config   若 checkpoint 包含 TRA，需要传入以重建 inject_tra hooks
  --dry-run      仅检查数据加载和配置，不运行实际训练
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _bootstrap_hf_cache_env() -> None:
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
        import os
        os.environ.setdefault(env_key, os.path.join(base, sub))
    for _, sub in pairs:
        import os
        os.makedirs(os.path.join(base, sub), exist_ok=True)


_bootstrap_hf_cache_env()

from text_rich_mllm.models.load_backbone import load_model_bundle
from text_rich_mllm.models.peft_adapter import attach_lora_adapter
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.training.ts_grpo_trainer import TSGRPOTrainer
from text_rich_mllm.utils import get_logger, load_yaml, read_jsonl, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Task-Stratified GRPO Training (E8)")
    parser.add_argument("--train-config", required=True,
                        help="训练配置文件路径（configs/train/train_joint_grpo.yaml）")
    parser.add_argument("--model-config", default="configs/model/backbone_main.yaml",
                        help="Backbone 模型配置")
    parser.add_argument("--peft-config", default="configs/model/peft.yaml",
                        help="PEFT 配置（LoRA / DoRA）")
    parser.add_argument("--checkpoint", default=None,
                        help="SFT checkpoint 目录（从此处加载 adapter 权重后续训）")
    parser.add_argument("--tra-config", default=None,
                        help="TRA 配置文件。若 checkpoint 来自 E5（含 TRA），需传入以重建 hooks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="仅验证配置和数据加载，不执行实际训练")
    args = parser.parse_args()

    logger = get_logger("train_grpo")
    set_seed(args.seed)

    # ── 配置加载 ─────────────────────────────────────────────────────────
    train_cfg = load_yaml(args.train_config)
    model_cfg = load_yaml(args.model_config)
    peft_cfg = load_yaml(args.peft_config)

    logger.info("=== Task-Stratified GRPO Training (E8) ===")
    logger.info("Train config : %s", args.train_config)
    logger.info("Checkpoint   : %s", args.checkpoint or "<none, fresh start>")
    logger.info("TRA config   : %s", args.tra_config or "<none>")
    logger.info("Seed         : %d", args.seed)

    # ── 数据加载 ─────────────────────────────────────────────────────────
    train_samples: list[UnifiedSample] = []
    for path in train_cfg.get("train_files", []):
        train_samples.extend(UnifiedSample.from_dict(r) for r in read_jsonl(path))

    eval_samples: list[UnifiedSample] = []
    for path in train_cfg.get("validation_files", []):
        eval_samples.extend(UnifiedSample.from_dict(r) for r in read_jsonl(path))

    logger.info("Loaded %d train samples, %d eval samples", len(train_samples), len(eval_samples))

    # ── Dry-run 模式 ─────────────────────────────────────────────────────
    if args.dry_run:
        from text_rich_mllm.training.ts_grpo_trainer import TaskStratifiedSampler
        task_names = list(train_cfg.get("grpo_task_names", []))
        sampler = TaskStratifiedSampler(train_samples, task_names)
        sample = sampler.sample_one()
        logger.info("Dry run: sampled task=%s, sample_id=%s", sample.dataset_name, sample.sample_id)
        logger.info("Dry run complete. No model loaded.")
        return

    # ── 模型加载 ─────────────────────────────────────────────────────────
    logger.info("Loading model bundle...")
    processor, model = load_model_bundle(**model_cfg)

    # ── 挂载 PEFT adapter（LoRA / DoRA）──────────────────────────────────
    logger.info("Attaching PEFT adapter: %s", args.peft_config)
    model = attach_lora_adapter(model, peft_cfg)

    # ── 从 SFT checkpoint 恢复权重（若提供）──────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Loading SFT checkpoint weights from: %s", ckpt_path)
        # PeftModel.from_pretrained 或直接 load_adapter
        if hasattr(model, "load_adapter"):
            model.load_adapter(str(ckpt_path), adapter_name="default")
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model.base_model.model, str(ckpt_path))

    # ── 注入 TRA hooks（若提供 tra-config）───────────────────────────────
    if args.tra_config:
        from text_rich_mllm.adapters.text_rich_adapter import TRAConfig
        from text_rich_mllm.models.qwen_with_tra import inject_tra, load_tra_state
        logger.info("Injecting TRA hooks: %s", args.tra_config)
        tra_config = TRAConfig.from_yaml(args.tra_config)
        model = inject_tra(model, tra_config)
        # 尝试从 checkpoint 恢复 TRA 权重
        if args.checkpoint:
            tra_state_path = Path(args.checkpoint) / "tra_state.pt"
            if tra_state_path.exists():
                load_tra_state(model, str(tra_state_path))
                logger.info("Loaded TRA state from: %s", tra_state_path)
            else:
                logger.warning("TRA state file not found at %s, using fresh TRA init.", tra_state_path)

    # ── 启动 TS-GRPO 训练 ─────────────────────────────────────────────────
    trainer = TSGRPOTrainer(
        model=model,
        processor=processor,
        train_samples=train_samples,
        train_config=train_cfg,
        eval_samples=eval_samples,
    )
    trainer.train()

    logger.info("=== TS-GRPO Training Complete ===")
    logger.info("Output dir: %s", train_cfg.get("output_dir"))


if __name__ == "__main__":
    main()
