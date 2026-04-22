"""
inference_best_of_n.py — Best-of-N Test-time Scaling（实验 E9 / E10）

原理：
  对每道题生成 N 个候选答案（temperature 采样），
  用 UnifiedEvaluator._score() 作为自动 reward 选出最优答案。
  无 gold answer 的场景（test split）退化为 Self-Consistency（多数票）。

学术背景：
  与 CVPR 2026 ViSCALE workshop 中 test-time compute scaling 方向一致。
  对应实验 E9（叠加 E3 checkpoint）和 E10（叠加 E8 TS-GRPO checkpoint）。

输出格式：
  与 run_inference.py / validate_checkpoint.py 完全兼容：
  [{\"sample_id\": \"...\", \"prediction\": \"...\"}, ...]

用法（基础）：
  python scripts/inference_best_of_n.py \\
    --samples  data/processed/docvqa/validation.jsonl \\
    --output   outputs/predictions/bon_N4_docvqa.jsonl \\
    --checkpoint outputs/checkpoints/joint_docvqa_chartqa/checkpoint-1250 \\
    --N 4

用法（生成 scaling 曲线，N=1,2,4,8）：
  python scripts/inference_best_of_n.py \\
    --samples  data/processed/docvqa/validation.jsonl \\
    --scaling-curve \\
    --curve-output outputs/analysis/bon_curve_docvqa.json \\
    --checkpoint outputs/checkpoints/joint_docvqa_chartqa/checkpoint-1250

用法（自动评测，直接输出 metrics）：
  python scripts/inference_best_of_n.py \\
    --samples  data/processed/docvqa/validation.jsonl \\
    --output   outputs/predictions/bon_N4_docvqa.jsonl \\
    --checkpoint outputs/checkpoints/joint_docvqa_chartqa/checkpoint-1250 \\
    --N 4 --evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _bootstrap_hf_cache_env() -> None:
    import os
    base = os.environ.get("TEXT_RICH_MLLM_MODEL_DISK", "").strip() or \
           os.environ.get("DATA_DISK", "").strip()
    if not base:
        return
    pairs = [
        ("HF_HOME", "hf_home"),
        ("HF_HUB_CACHE", "huggingface_hub"),
        ("HF_DATASETS_CACHE", "huggingface_cache"),
        ("TRANSFORMERS_CACHE", "transformers_cache"),
    ]
    for env_key, sub in pairs:
        os.environ.setdefault(env_key, os.path.join(base, sub))
        os.makedirs(os.path.join(base, sub), exist_ok=True)


_bootstrap_hf_cache_env()

import torch
from tqdm.auto import tqdm

from text_rich_mllm.evaluation.evaluator import UnifiedEvaluator
from text_rich_mllm.evaluation.parsing import parse_prediction
from text_rich_mllm.models.generation_utils import (
    open_image_as_rgb,
    take_answer_tail_after_marker,
)
from text_rich_mllm.models.vision_prompt import ensure_image_placeholders_in_text
from text_rich_mllm.prompts import PromptBuilder
from text_rich_mllm.schemas import UnifiedSample
from text_rich_mllm.utils import get_logger, read_jsonl, write_json, write_jsonl

logger = get_logger("bon")


# ─────────────────────────────────────────────────────────────────────────────
# 核心推理函数
# ─────────────────────────────────────────────────────────────────────────────

def generate_n_completions(
    model,
    processor,
    sample: UnifiedSample,
    prompt: str,
    *,
    N: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """
    对同一道题生成 N 个候选答案。

    实现说明：
      - N=1 时自动切换为 greedy decoding（temperature=1.0 + do_sample=False）
      - N>1 时用 num_return_sequences=N 一次 forward 完成，比循环 N 次更快
    """
    image = open_image_as_rgb(sample.image_path)
    prompt_for_model = ensure_image_placeholders_in_text(processor, prompt, num_images=1)
    inputs = processor(
        images=image,
        text=prompt_for_model,
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        if N == 1:
            gen_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
            )
        else:
            gen_ids = model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=N,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

    decoded_list = processor.batch_decode(gen_ids, skip_special_tokens=True)
    completions = []
    for decoded in decoded_list:
        raw = decoded.strip()
        if raw.startswith(prompt_for_model.strip()):
            raw = raw[len(prompt_for_model.strip()):].strip()
        completions.append(take_answer_tail_after_marker(raw))
    return completions


def best_of_n_select(
    evaluator: UnifiedEvaluator,
    sample: UnifiedSample,
    completions: list[str],
    *,
    use_self_consistency: bool = False,
) -> tuple[str, list[float]]:
    """
    从 N 个候选中选出最优答案。

    有 gold answer（validation split）：
      reward = evaluator._score(sample, parsed) → 选 reward 最高的

    无 gold answer（test split，use_self_consistency=True）：
      majority_vote → 选出现次数最多的 parsed answer

    返回：(best_answer, rewards_list)
    """
    parsed_list = [parse_prediction(c, answer_type=sample.answer_type) for c in completions]

    if use_self_consistency or not sample.gold_answer:
        # Majority vote（Self-Consistency）
        counter = Counter(parsed_list)
        best_parsed = counter.most_common(1)[0][0]
        # 找到 best_parsed 对应的原始 completion
        for comp, parsed in zip(completions, parsed_list):
            if parsed == best_parsed:
                return comp, [0.0] * len(completions)  # 无 reward 信息
        return completions[0], [0.0] * len(completions)
    else:
        # Best-of-N：用 evaluator 打分选最优
        rewards = [evaluator._score(sample, p) for p in parsed_list]
        best_idx = int(max(range(len(rewards)), key=lambda i: rewards[i]))
        return completions[best_idx], rewards


# ─────────────────────────────────────────────────────────────────────────────
# 主推理流程
# ─────────────────────────────────────────────────────────────────────────────

def run_bon_inference(
    *,
    model,
    processor,
    samples: list[UnifiedSample],
    N: int,
    temperature: float = 0.8,
    max_new_tokens: int = 32,
    prompt_style: str = "structured",
    output_path: str | None = None,
    use_self_consistency: bool = False,
    limit: int | None = None,
) -> tuple[dict[str, str], dict[str, list[float]]]:
    """
    对所有样本做 Best-of-N 推理。

    返回：
      prediction_map: {sample_id: best_answer}（兼容 evaluate_model.py）
      all_rewards:    {sample_id: [r₁, r₂, ..., r_N]}（用于分析）
    """
    evaluator = UnifiedEvaluator()
    builder = PromptBuilder(style=prompt_style)

    if limit:
        samples = samples[:limit]

    prediction_map: dict[str, str] = {}
    all_rewards: dict[str, list[float]] = {}
    output_records: list[dict] = []

    t0 = time.perf_counter()
    pbar = tqdm(samples, desc=f"BoN(N={N})", unit="sample", dynamic_ncols=True)

    for sample in pbar:
        prompt = builder.build(sample)
        completions = generate_n_completions(
            model, processor, sample, prompt,
            N=N, temperature=temperature, max_new_tokens=max_new_tokens,
        )
        best, rewards = best_of_n_select(
            evaluator, sample, completions,
            use_self_consistency=use_self_consistency,
        )
        prediction_map[sample.sample_id] = best
        all_rewards[sample.sample_id] = rewards

        output_records.append({
            "sample_id": sample.sample_id,
            "prediction": best,
            "all_completions": completions,
            "rewards": rewards,
        })

        if output_path:
            # 写兼容格式（仅 sample_id + prediction，与 evaluate_model.py 一致）
            compat_records = [
                {"sample_id": r["sample_id"], "prediction": r["prediction"]}
                for r in output_records
            ]
            write_jsonl(compat_records, output_path)

        # 进度条显示当前平均 reward
        if any(rewards):
            mean_r = sum(rewards) / len(rewards)
            pbar.set_postfix(mean_r=f"{mean_r:.3f}")

    elapsed = time.perf_counter() - t0
    logger.info("BoN(N=%d) done: %d samples in %.1fs (%.2fs/sample)", N, len(samples), elapsed, elapsed / max(len(samples), 1))
    return prediction_map, all_rewards


# ─────────────────────────────────────────────────────────────────────────────
# BoN Scaling 曲线生成（Figure 3 用）
# ─────────────────────────────────────────────────────────────────────────────

def run_bon_scaling_curve(
    *,
    model,
    processor,
    samples: list[UnifiedSample],
    n_values: list[int] = None,
    temperature: float = 0.8,
    max_new_tokens: int = 32,
    prompt_style: str = "structured",
    curve_output: str | None = None,
    limit: int | None = None,
) -> dict:
    """
    生成 N=1,2,4,8 的 BoN scaling 曲线数据。

    核心思路：
      为避免对每个 N 值都重新推理（浪费计算），
      先一次性生成 N_max=8 个候选，
      然后对每个 N 值取前 N 个候选计算 best-of-N 分数。
      这样只需一次推理（8次生成/题），而不是 4次推理（1+2+4+8=15次/题）。

    返回：
      {
        "n_values": [1, 2, 4, 8],
        "dataset_scores": {
          "docvqa": {"n=1": 0.72, "n=2": 0.76, "n=4": 0.79, "n=8": 0.81},
          ...
        },
        "overall": {"n=1": 0.70, "n=2": 0.74, ...}
      }
    """
    if n_values is None:
        n_values = [1, 2, 4, 8]
    N_max = max(n_values)

    evaluator = UnifiedEvaluator()
    builder = PromptBuilder(style=prompt_style)

    if limit:
        samples = samples[:limit]

    logger.info("BoN Scaling Curve: N=%s, N_max=%d, n_samples=%d", n_values, N_max, len(samples))

    # Step 1：一次性生成 N_max 个候选
    logger.info("Step 1: Generating N_max=%d completions per sample...", N_max)
    all_completions: dict[str, list[str]] = {}

    pbar = tqdm(samples, desc=f"gen(N={N_max})", unit="sample", dynamic_ncols=True)
    for sample in pbar:
        prompt = builder.build(sample)
        completions = generate_n_completions(
            model, processor, sample, prompt,
            N=N_max, temperature=temperature, max_new_tokens=max_new_tokens,
        )
        all_completions[sample.sample_id] = completions

    # Step 2：对每个 N 值计算 best-of-N 分数
    logger.info("Step 2: Computing best-of-N scores for N=%s...", n_values)
    from collections import defaultdict
    dataset_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall_scores: dict[str, list[float]] = defaultdict(list)

    for sample in samples:
        completions = all_completions[sample.sample_id]
        for N in n_values:
            sub_completions = completions[:N]
            _, rewards = best_of_n_select(evaluator, sample, sub_completions)
            if any(r > 0 for r in rewards):
                best_reward = max(rewards)
            else:
                # 无法从 reward 判断时，用 greedy 结果（N=1 fallback）
                parsed = parse_prediction(sub_completions[0], answer_type=sample.answer_type)
                best_reward = evaluator._score(sample, parsed)
            dataset_scores[sample.dataset_name][f"n={N}"].append(best_reward)
            overall_scores[f"n={N}"].append(best_reward)

    # Step 3：汇总均值
    result = {
        "n_values": n_values,
        "dataset_scores": {
            ds: {k: sum(v) / len(v) for k, v in ns.items()}
            for ds, ns in dataset_scores.items()
        },
        "overall": {k: sum(v) / len(v) for k, v in overall_scores.items()},
        "n_samples": len(samples),
        "N_max": N_max,
        "temperature": temperature,
    }

    # 打印曲线摘要
    logger.info("=== BoN Scaling Curve Results ===")
    for N in n_values:
        key = f"n={N}"
        overall = result["overall"].get(key, 0.0)
        logger.info("  N=%-2d  overall=%.4f", N, overall)
    for ds, ns in result["dataset_scores"].items():
        scores_str = "  ".join(f"N={N}:{ns.get(f'n={N}', 0):.4f}" for N in n_values)
        logger.info("  [%s] %s", ds, scores_str)

    # 保存
    if curve_output:
        Path(curve_output).parent.mkdir(parents=True, exist_ok=True)
        with open(curve_output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info("Curve data saved to: %s", curve_output)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(args):
    """加载模型（支持 LoRA / DoRA / TRA checkpoint）。"""
    from text_rich_mllm.models.load_backbone import load_model_bundle
    from text_rich_mllm.utils import load_yaml

    model_cfg = load_yaml(args.model_config)
    processor, model = load_model_bundle(**model_cfg)

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        logger.info("Loading checkpoint: %s", ckpt_path)
        peft_cfg = load_yaml(args.peft_config)

        from text_rich_mllm.models.peft_adapter import attach_lora_adapter
        model = attach_lora_adapter(model, peft_cfg)

        if hasattr(model, "load_adapter"):
            model.load_adapter(str(ckpt_path), adapter_name="default")
        else:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model.base_model.model, str(ckpt_path))

    if args.tra_config:
        from text_rich_mllm.adapters.text_rich_adapter import TRAConfig
        from text_rich_mllm.models.qwen_with_tra import inject_tra, load_tra_state
        tra_config = TRAConfig.from_yaml(args.tra_config)
        model = inject_tra(model, tra_config)
        if args.checkpoint:
            tra_state_path = Path(args.checkpoint) / "tra_state.pt"
            if tra_state_path.exists():
                load_tra_state(model, str(tra_state_path))
                logger.info("Loaded TRA state from: %s", tra_state_path)

    model.eval()
    return processor, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-of-N Test-time Scaling (E9/E10)")
    # 数据
    parser.add_argument("--samples", required=True,
                        help="样本 JSONL 文件（data/processed/docvqa/validation.jsonl）")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 条样本（调试用）")
    # 输出
    parser.add_argument("--output", default=None,
                        help="预测结果输出路径（与 evaluate_model.py 兼容的 JSONL）")
    parser.add_argument("--evaluate", action="store_true",
                        help="推理完成后直接调用 UnifiedEvaluator 计算 metrics 并打印")
    # BoN 参数
    parser.add_argument("--N", type=int, default=4,
                        help="每题生成的候选答案数（默认 4）")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="采样温度（N>1 时生效）")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--self-consistency", action="store_true",
                        help="使用多数票代替 reward 选择（无 gold answer 时使用）")
    # Scaling 曲线模式
    parser.add_argument("--scaling-curve", action="store_true",
                        help="生成 N=1,2,4,8 的 scaling 曲线（Figure 用）")
    parser.add_argument("--curve-n-values", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="scaling 曲线的 N 值列表（默认 1 2 4 8）")
    parser.add_argument("--curve-output", default=None,
                        help="scaling 曲线 JSON 输出路径")
    # 模型
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 目录（可选，不传则用原始 pretrained 模型）")
    parser.add_argument("--peft-config", default="configs/model/peft.yaml")
    parser.add_argument("--tra-config", default=None,
                        help="TRA 配置（checkpoint 包含 TRA 时传入）")
    parser.add_argument("--model-config", default="configs/model/backbone_main.yaml")
    parser.add_argument("--prompt-style", default="structured")

    args = parser.parse_args()

    logger.info("=== Best-of-N Test-time Scaling ===")
    logger.info("Mode          : %s", "scaling-curve" if args.scaling_curve else f"BoN(N={args.N})")
    logger.info("Samples       : %s", args.samples)
    logger.info("Checkpoint    : %s", args.checkpoint or "<pretrained>")
    logger.info("TRA config    : %s", args.tra_config or "<none>")

    # 加载数据
    samples = [UnifiedSample.from_dict(r) for r in read_jsonl(args.samples)]
    logger.info("Loaded %d samples", len(samples))

    # 加载模型
    processor, model = _load_model(args)

    if args.scaling_curve:
        # ── 模式 A：Scaling 曲线 ───────────────────────────────────────
        result = run_bon_scaling_curve(
            model=model,
            processor=processor,
            samples=samples,
            n_values=args.curve_n_values,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            prompt_style=args.prompt_style,
            curve_output=args.curve_output,
            limit=args.limit,
        )
        print("\n=== BoN Scaling Curve Summary ===")
        for N in args.curve_n_values:
            key = f"n={N}"
            print(f"  N={N:<2d}  overall={result['overall'].get(key, 0):.4f}")
    else:
        # ── 模式 B：单一 N 推理 ────────────────────────────────────────
        if not args.output:
            raise ValueError("--output 是必须参数（--scaling-curve 模式除外）")

        prediction_map, all_rewards = run_bon_inference(
            model=model,
            processor=processor,
            samples=samples,
            N=args.N,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            prompt_style=args.prompt_style,
            output_path=args.output,
            use_self_consistency=args.self_consistency,
            limit=args.limit,
        )

        if args.evaluate:
            # 直接评测
            evaluator = UnifiedEvaluator()
            active_samples = samples[:args.limit] if args.limit else samples
            records, summary = evaluator.evaluate(active_samples, prediction_map)
            print("\n=== BoN Evaluation Summary ===")
            for k, v in summary.items():
                if k != "invalid_output_rate":
                    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
            # 保存 metrics（output 路径旁边）
            metrics_path = str(args.output).replace(".jsonl", "_metrics.json")
            write_json({"n": args.N, "summary": summary}, metrics_path)
            logger.info("Metrics saved to: %s", metrics_path)


if __name__ == "__main__":
    main()
