"""
Task-Stratified GRPO Trainer (TS-GRPO)
=======================================
实验 E8：在 SFT（E3/E5）checkpoint 基础上，用 Group Relative Policy Optimization
做 RL 对齐。核心创新是「任务分层」：每个 group 只包含同一 task 的样本，
确保 advantage 在 task-homogeneous 的 reward scale 内计算。

算法流程（每个 RL step）：
  1. 从训练集按 task_id 随机选 1 道题（TaskStratifiedSampler）
  2. 用当前策略 π 采样 G=4 个答案（temperature 采样）
  3. 用 UnifiedEvaluator._score() 计算每个答案的 reward（task-specific metric）
  4. 计算 group-normalized advantage：Aᵢ = (rᵢ - mean(r)) / (std(r) + ε)
  5. 用 reference policy π_ref 计算 KL 散度
  6. Loss = -mean(Aᵢ · log π(ansᵢ)) + β · KL(π || π_ref)
  7. 反传更新 π（LoRA / LoRA+TRA 参数）

参考文献：
  DeepSeek-R1（GRPO 原始应用）
  R1-VL, Kimi-VL-Thinking（VLM GRPO）
  本项目创新：task-stratified group sampling（多任务 reward scale 对齐）
"""
from __future__ import annotations

import copy
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from text_rich_mllm.schemas import UnifiedSample

from text_rich_mllm.evaluation.evaluator import UnifiedEvaluator
from text_rich_mllm.evaluation.parsing import parse_prediction
from text_rich_mllm.models.generation_utils import (
    open_image_as_rgb,
    take_answer_tail_after_marker,
)
from text_rich_mllm.models.vision_prompt import ensure_image_placeholders_in_text
from text_rich_mllm.prompts import PromptBuilder
from text_rich_mllm.utils import get_logger

logger = get_logger("ts_grpo")


# ─────────────────────────────────────────────────────────────────────────────
# Task-Stratified Sampler
# ─────────────────────────────────────────────────────────────────────────────

class TaskStratifiedSampler:
    """
    按 task_id 分桶，每次随机选 1 个 task，再从该 task 内随机选 1 道题。
    保证同一个 GRPO group 内所有 G 个生成都来自同一 task_id。

    Attributes:
        buckets: {task_name: [UnifiedSample, ...]}
        task_names: 所有 task 名称列表（非空 bucket）
    """

    def __init__(
        self,
        samples: list["UnifiedSample"],
        task_names: list[str],
    ) -> None:
        self.buckets: dict[str, list["UnifiedSample"]] = defaultdict(list)
        for sample in samples:
            if sample.dataset_name in task_names:
                self.buckets[sample.dataset_name].append(sample)
        # 只保留非空 bucket
        self.task_names = [t for t in task_names if self.buckets.get(t)]
        if not self.task_names:
            raise ValueError("TaskStratifiedSampler: 所有 task 的 bucket 均为空，请检查 task_names 配置。")
        n_total = sum(len(v) for v in self.buckets.values())
        logger.info("TaskStratifiedSampler: %d tasks, %d samples total", len(self.task_names), n_total)
        for t in self.task_names:
            logger.info("  task=%s  samples=%d", t, len(self.buckets[t]))

    def sample_one(self) -> "UnifiedSample":
        """随机选 1 个 task，再从该 task 随机选 1 道题。"""
        task = random.choice(self.task_names)
        return random.choice(self.buckets[task])


# ─────────────────────────────────────────────────────────────────────────────
# 生成辅助：对同一道题采样 G 个答案
# ─────────────────────────────────────────────────────────────────────────────

def _sample_completions(
    model,
    processor,
    sample: "UnifiedSample",
    prompt: str,
    *,
    G: int,
    temperature: float,
    max_new_tokens: int,
) -> list[str]:
    """
    对同一张图 + 同一个 prompt，用 temperature 采样生成 G 个答案。
    返回 G 个字符串（经过 take_answer_tail_after_marker 后处理）。

    实现说明：
      - do_sample=True + temperature 采样（多样性）
      - num_return_sequences=G：一次 forward 生成 G 个序列，比循环 G 次更快
      - 需要 processor 支持 batch decode
    """
    image = open_image_as_rgb(sample.image_path)
    prompt_for_model = ensure_image_placeholders_in_text(processor, prompt, num_images=1)
    inputs = processor(images=image, text=prompt_for_model, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            num_return_sequences=G,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    # batch_decode 返回 G 个字符串
    decoded_list = processor.batch_decode(generated_ids, skip_special_tokens=True)
    completions = []
    for decoded in decoded_list:
        raw = decoded.strip()
        # 去除 prompt 前缀
        if raw.startswith(prompt_for_model.strip()):
            raw = raw[len(prompt_for_model.strip()):].strip()
        completions.append(take_answer_tail_after_marker(raw))
    return completions


# ─────────────────────────────────────────────────────────────────────────────
# Reward 计算（直接复用 UnifiedEvaluator._score）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_rewards(
    evaluator: UnifiedEvaluator,
    sample: "UnifiedSample",
    completions: list[str],
) -> list[float]:
    """
    对 G 个 completion 分别计算 reward。
    reward = evaluator._score(sample, parsed_prediction)
    task-specific metric：
      docvqa / infographicvqa / textvqa → ANLS ∈ [0, 1]
      chartqa                           → chartqa_score ∈ {0, 1}
      scienceqa / mmmu（MCQ）           → accuracy ∈ {0, 1}
    """
    rewards = []
    for comp in completions:
        parsed = parse_prediction(comp, answer_type=sample.answer_type)
        r = evaluator._score(sample, parsed)
        rewards.append(float(r))
    return rewards


# ─────────────────────────────────────────────────────────────────────────────
# Log-probability 计算（用于 GRPO loss）
# ─────────────────────────────────────────────────────────────────────────────

def _compute_log_probs(
    model,
    processor,
    sample: "UnifiedSample",
    prompt: str,
    completions: list[str],
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    对 G 个 completion 计算 answer token 的 log-probability 之和。
    返回 shape (G,) 的 Tensor。

    实现说明：
      - 构建 prompt + completion 的完整序列
      - 对 prompt 部分的 labels 设为 ignore_index（-100），只计算 answer 部分
      - 直接从 logits 手动计算 log_prob，避免依赖 HF 内部 loss 的梯度保留行为
      - 返回 sum(log_prob of answer tokens)，形状 (G,)

    注意：
      - 当被 ref_model 调用时，调用方需在 torch.no_grad() 块内调用此函数
      - 当被训练中的 model 调用时，保持梯度计算图
    """
    image = open_image_as_rgb(sample.image_path)
    prompt_for_model = ensure_image_placeholders_in_text(processor, prompt, num_images=1)
    device = next(model.parameters()).device

    log_probs_list = []
    for comp in completions:
        full_text = f"{prompt_for_model} {comp}".strip()
        # 编码完整序列
        full_inputs = processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        full_inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in full_inputs.items()}

        # 编码仅 prompt 部分，用于确定 prompt token 数量
        prompt_inputs = processor(
            images=image,
            text=prompt_for_model,
            return_tensors="pt",
            padding=False,
            truncation=False,
        )
        prompt_len = int(prompt_inputs["attention_mask"].sum().item())

        # forward（不传 labels，从 logits 手动计算，梯度图更清晰）
        input_ids = full_inputs["input_ids"]              # (1, L)
        outputs = model(**{k: v for k, v in full_inputs.items() if k != "labels"})
        logits = outputs.logits                            # (1, L, V)

        # shift：logits[t] 预测 input_ids[t+1]
        shift_logits = logits[0, :-1, :]                  # (L-1, V)
        shift_ids    = input_ids[0, 1:]                   # (L-1,)

        # 只对 answer token 部分计算 log_prob
        # answer token 下标：[prompt_len, L-1)（shift 后对应 [prompt_len-1, L-2)）
        answer_start = max(prompt_len - 1, 0)             # shift 偏移
        answer_logits = shift_logits[answer_start:]       # (n_ans, V)
        answer_ids    = shift_ids[answer_start:]          # (n_ans,)

        n_answer_tokens = answer_ids.shape[0]
        if n_answer_tokens == 0:
            log_probs_list.append(torch.tensor(0.0, device=device))
        else:
            log_prob_per_token = F.log_softmax(answer_logits, dim=-1)  # (n_ans, V)
            # 取每个位置正确 token 的 log_prob
            token_log_probs = log_prob_per_token[                       # (n_ans,)
                torch.arange(n_answer_tokens, device=device),
                answer_ids,
            ]
            log_probs_list.append(token_log_probs.sum())               # scalar

    return torch.stack(log_probs_list)  # (G,)


# ─────────────────────────────────────────────────────────────────────────────
# TS-GRPO 主训练循环
# ─────────────────────────────────────────────────────────────────────────────

class TSGRPOTrainer:
    """
    Task-Stratified GRPO Trainer。

    训练循环（每 step）：
      1. TaskStratifiedSampler 选 1 道题（同 task G 次生成）
      2. π 采样 G 个 completion（temperature sampling）
      3. Reward = UnifiedEvaluator._score（task-specific metric）
      4. Advantage = (reward - mean) / (std + ε)（task-homogeneous scale）
      5. KL(π || π_ref) 用参考模型的 log_prob 估计
      6. Loss = -mean(A · log_π) + β · KL
      7. 反传 + optimizer.step()

    不使用 HuggingFace Trainer，采用手动训练循环，
    便于精确控制 group 采样和 task-stratified reward。
    """

    def __init__(
        self,
        model,
        processor,
        train_samples: list["UnifiedSample"],
        train_config: dict,
        eval_samples: list["UnifiedSample"] | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.eval_samples = eval_samples or []
        self.evaluator = UnifiedEvaluator()

        # ── 超参 ────────────────────────────────────────────────────────
        self.G = int(train_config.get("grpo_group_size", 4))
        self.beta = float(train_config.get("grpo_kl_coef", 0.01))
        self.temperature = float(train_config.get("grpo_temperature", 0.8))
        self.max_new_tokens = int(train_config.get("grpo_max_new_tokens", 32))
        self.num_steps = int(train_config.get("grpo_num_steps", 500))
        self.eval_steps = int(train_config.get("grpo_eval_steps", 100))
        self.save_steps = int(train_config.get("grpo_save_steps", 100))
        self.lr = float(train_config.get("learning_rate", 5e-6))
        self.output_dir = Path(train_config.get("output_dir", "outputs/checkpoints/joint_ts_grpo"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prompt_style = train_config.get("prompt_style", "structured")
        task_names = list(train_config.get("grpo_task_names", []))

        # ── 数据采样器 ──────────────────────────────────────────────────
        self.sampler = TaskStratifiedSampler(train_samples, task_names)
        self.prompt_builder = PromptBuilder(style=self.prompt_style)

        # ── Reference model（frozen copy，用于 KL 计算）────────────────
        logger.info("Building reference model (frozen copy for KL)...")
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad_(False)

        # ── 优化器（只优化可训练参数：LoRA + TRA）──────────────────────
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info("Trainable parameters: %d tensors", len(trainable_params))
        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

        # ── 统计 ────────────────────────────────────────────────────────
        self.step_log: list[dict] = []

    # ── 评测（在 val set 上跑 greedy decoding）──────────────────────────
    def _run_eval(self, step: int) -> dict[str, float]:
        if not self.eval_samples:
            return {}
        from text_rich_mllm.inference import generate_predictions

        # 评测时切回 eval 模式，结束后恢复 train 模式
        self.model.eval()
        gen_cfg = {"max_new_tokens": self.max_new_tokens, "do_sample": False}
        preds = generate_predictions(
            samples=self.eval_samples,
            model=self.model,
            processor=self.processor,
            prompt_style=self.prompt_style,
            generation_config=gen_cfg,
            limit=200,  # 快速评测，取前 200 条
        )
        self.model.train()
        # 评测用 eval_samples[:200]（与 limit 对齐）
        eval_subset = self.eval_samples[:200]
        _, summary = self.evaluator.evaluate(eval_subset, preds)
        logger.info("[eval @ step %d] %s", step, summary)
        return summary

    # ── 单步 RL 更新 ─────────────────────────────────────────────────────
    def _step(self) -> dict[str, float]:
        sample = self.sampler.sample_one()
        prompt = self.prompt_builder.build(sample)

        # 1. 采样 G 个 completion
        self.model.eval()
        completions = _sample_completions(
            self.model,
            self.processor,
            sample,
            prompt,
            G=self.G,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )

        # 2. 计算 reward（task-specific，在同 task 内 scale 一致）
        rewards = _compute_rewards(self.evaluator, sample, completions)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)

        # 3. Group-normalized advantage（task-homogeneous）
        mean_r = rewards_t.mean()
        std_r = rewards_t.std()
        advantages = (rewards_t - mean_r) / (std_r + 1e-8)  # (G,)

        # 4. 计算当前策略的 log_prob（需要梯度）
        self.model.train()
        log_probs = _compute_log_probs(
            self.model, self.processor, sample, prompt, completions
        )  # (G,)

        # 5. 计算 reference policy 的 log_prob（无梯度）
        with torch.no_grad():
            ref_log_probs = _compute_log_probs(
                self.ref_model, self.processor, sample, prompt, completions
            )  # (G,)

        # 6. KL 散度估计（per-completion 的 log_prob 差值）
        kl = (log_probs - ref_log_probs).mean()

        # 7. GRPO Loss
        device = log_probs.device
        advantages = advantages.to(device)
        policy_loss = -(advantages * log_probs).mean()
        loss = policy_loss + self.beta * kl

        # 8. 反传
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl.item(),
            "mean_reward": mean_r.item(),
            "std_reward": std_r.item(),
            "task": sample.dataset_name,
        }

    # ── 主训练循环 ───────────────────────────────────────────────────────
    def train(self, resume_step: int = 0) -> None:
        logger.info(
            "TS-GRPO training: G=%d, β=%.3f, T=%.2f, steps=%d, lr=%.2e",
            self.G, self.beta, self.temperature, self.num_steps, self.lr,
        )
        t0 = time.perf_counter()

        for step in range(resume_step + 1, self.num_steps + 1):
            stats = self._step()
            self.step_log.append({"step": step, **stats})

            # ── 日志（每 10 步）────────────────────────────────────────
            if step % 10 == 0:
                elapsed = time.perf_counter() - t0
                logger.info(
                    "step=%d  loss=%.4f  policy_loss=%.4f  kl=%.4f  "
                    "mean_r=%.4f  task=%s  elapsed=%.1fs",
                    step,
                    stats["loss"],
                    stats["policy_loss"],
                    stats["kl"],
                    stats["mean_reward"],
                    stats["task"],
                    elapsed,
                )

            # ── 评测 ────────────────────────────────────────────────────
            if step % self.eval_steps == 0:
                self._run_eval(step)

            # ── 保存 checkpoint ─────────────────────────────────────────
            if step % self.save_steps == 0:
                ckpt_dir = self.output_dir / f"checkpoint-{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                # 保存 LoRA / PEFT adapter（若模型是 PeftModel）
                if hasattr(self.model, "save_pretrained"):
                    self.model.save_pretrained(str(ckpt_dir))
                else:
                    torch.save(self.model.state_dict(), str(ckpt_dir / "model.pt"))
                logger.info("Saved checkpoint: %s", ckpt_dir)

        # ── 最终保存 ─────────────────────────────────────────────────────
        final_dir = self.output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(str(final_dir))
        else:
            torch.save(self.model.state_dict(), str(final_dir / "model.pt"))
        if hasattr(self.processor, "save_pretrained"):
            self.processor.save_pretrained(str(final_dir))

        # ── 保存训练统计 ──────────────────────────────────────────────────
        import json
        stats_path = self.output_dir / "grpo_step_log.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(self.step_log, f, ensure_ascii=False, indent=2)
        logger.info("TS-GRPO training complete. Stats saved to %s", stats_path)
