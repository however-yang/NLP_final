from __future__ import annotations

from pathlib import Path

from text_rich_mllm.models.generation_utils import open_image_as_rgb
from text_rich_mllm.utils.paths import resolve_training_output_dir
from text_rich_mllm.models.vision_prompt import ensure_image_placeholders_in_text
from text_rich_mllm.training.hf_dataset import SupervisedTrainingDataset


class MultimodalSupervisedCollator:
    def __init__(
        self,
        processor,
        *,
        max_length: int | None = None,
        ignore_index: int = -100,
        image_max_pixels: int | None = None,
    ):
        self.processor = processor
        self.max_length = max_length
        self.ignore_index = ignore_index
        # Qwen2/3-VL：大图会产生极长视觉 token；loss 处要对整段 logits 做 float，显存 ~ O(seq×vocab)
        self.image_max_pixels = image_max_pixels

    def __call__(self, examples):
        images = [open_image_as_rgb(example.image_path) for example in examples]
        prompts = [
            ensure_image_placeholders_in_text(self.processor, example.prompt, num_images=1)
            for example in examples
        ]
        full_texts = []
        for example, p_aug in zip(examples, prompts):
            ans = example.target_answer.strip()
            full_texts.append(f"{p_aug} {ans}".strip() if ans else p_aug)

        # Qwen3-VL：truncation=max_length 会截断序列，导致「文本里 image 占位」与 input_ids 中视觉 token 数量不一致
        # （processing_utils._check_special_mm_tokens）。多模态训练须关闭 truncation，仅靠 padding 组 batch。
        processor_kwargs = {
            "images": images,
            "text": full_texts,
            "return_tensors": "pt",
            "padding": True,
            "truncation": False,
        }
        if self.image_max_pixels is not None:
            processor_kwargs["images_kwargs"] = {"max_pixels": self.image_max_pixels}

        full_batch = self.processor(**processor_kwargs)

        prompt_kwargs = dict(processor_kwargs)
        prompt_kwargs["text"] = prompts
        prompt_batch = self.processor(**prompt_kwargs)

        labels = full_batch["input_ids"].clone()
        prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
        for index, prompt_length in enumerate(prompt_lengths):
            labels[index, :prompt_length] = self.ignore_index
        labels[full_batch["attention_mask"] == 0] = self.ignore_index
        full_batch["labels"] = labels
        return full_batch


def _build_training_arguments(output_dir: str, train_config: dict):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_config.get("batch_size", 1),
        per_device_eval_batch_size=train_config.get("eval_batch_size", train_config.get("batch_size", 1)),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),
        learning_rate=train_config.get("learning_rate", 1e-4),
        num_train_epochs=train_config.get("num_train_epochs", 1),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        logging_strategy=train_config.get("logging_strategy", "steps"),
        logging_steps=train_config.get("logging_steps", 10),
        save_strategy=train_config.get("save_strategy", "steps"),
        save_steps=train_config.get("save_steps", 100),
        eval_strategy=train_config.get("eval_strategy", "no"),
        eval_steps=train_config.get("eval_steps"),
        save_total_limit=train_config.get("save_total_limit", 2),
        remove_unused_columns=False,
        report_to=[],
        bf16=train_config.get("bf16", False),
        fp16=train_config.get("fp16", False),
        dataloader_num_workers=train_config.get("dataloader_num_workers", 0),
        load_best_model_at_end=train_config.get("load_best_model_at_end", False),
        metric_for_best_model=train_config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=train_config.get("greater_is_better", False),
        gradient_checkpointing=train_config.get("gradient_checkpointing", False),
    )


def train_with_hf_trainer(
    *,
    model,
    processor,
    train_examples,
    train_config: dict,
    eval_examples=None,
    resume_from_checkpoint: str | None = None,
):
    from transformers import Trainer

    output_dir = resolve_training_output_dir(train_config.get("output_dir", "outputs/checkpoints/default"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = _build_training_arguments(output_dir, train_config)
    # LoRA + gradient checkpointing：底层冻结时需让输入 embedding 参与计算图，否则 checkpoint 反传报错
    if getattr(training_args, "gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    collator = MultimodalSupervisedCollator(
        processor,
        max_length=train_config.get("max_seq_length"),
        ignore_index=train_config.get("ignore_index", -100),
        image_max_pixels=train_config.get("image_max_pixels"),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SupervisedTrainingDataset(train_examples),
        eval_dataset=SupervisedTrainingDataset(eval_examples) if eval_examples else None,
        data_collator=collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)
    return trainer


# ---------------------------------------------------------------------------
# TRA 专用 Trainer
# ---------------------------------------------------------------------------

class TRATrainer:
    """
    包装 HuggingFace Trainer，在每次 forward 前把 task_ids 写入 model._tra_task_ids。

    实现思路：
      HuggingFace Trainer 的 compute_loss 方法接收 (model, inputs, return_outputs) 参数。
      我们继承 Trainer 并重写 compute_loss，在调用 super().compute_loss() 前把
      batch 中的 task_id_list 收集成 Tensor 并赋值给 model._tra_task_ids。
    """

    @staticmethod
    def build(
        model,
        processor,
        train_examples,
        training_args,
        eval_examples=None,
    ):
        """build() 返回一个配置好的 _TRAHFTrainer 实例。"""
        from transformers import Trainer

        # 基于 task_id 字段构建 自定义 collator
        class _TRACollator(MultimodalSupervisedCollator):
            """在普通 batch 基础上，额外返回 task_ids 列表。"""

            def __call__(self, examples):
                # 保存 task_id 序列（在 processor 处理前，不丢失顺序）
                task_ids = [ex.task_id for ex in examples]
                batch = super().__call__(examples)
                # 将 task_ids 存到 batch 元数据中（不是 Tensor，避免 HF 的 to(device) 报错）
                batch["_task_ids"] = task_ids
                return batch

        class _TRAHFTrainer(Trainer):
            """在 compute_loss 前把 task_ids 写入 model._tra_task_ids。"""

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                import torch

                task_ids_raw = inputs.pop("_task_ids", None)
                if task_ids_raw is not None:
                    device = next(model.parameters()).device
                    model._tra_task_ids = torch.tensor(
                        task_ids_raw, dtype=torch.long, device=device
                    )
                else:
                    model._tra_task_ids = None

                result = super().compute_loss(model, inputs, return_outputs, **kwargs)

                # 清理，防止下一个 batch 错误复用
                model._tra_task_ids = None
                return result

        collator = _TRACollator(
            processor,
            max_length=None,
            ignore_index=-100,
            image_max_pixels=None,
        )
        return _TRAHFTrainer(
            model=model,
            args=training_args,
            train_dataset=SupervisedTrainingDataset(train_examples),
            eval_dataset=SupervisedTrainingDataset(eval_examples) if eval_examples else None,
            data_collator=collator,
        )


def train_with_hf_trainer_tra(
    *,
    model,
    processor,
    train_examples,
    train_config: dict,
    eval_examples=None,
    resume_from_checkpoint: str | None = None,
):
    """
    TRA 版本的训练入口，与 train_with_hf_trainer 接口完全一致。
    内部使用 TRATrainer.build() 构建带 task_id 注入的 Trainer。
    """
    output_dir = resolve_training_output_dir(train_config.get("output_dir", "outputs/checkpoints/default"))
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = _build_training_arguments(output_dir, train_config)
    if getattr(training_args, "gradient_checkpointing", False) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    trainer = TRATrainer.build(
        model=model,
        processor=processor,
        train_examples=train_examples,
        training_args=training_args,
        eval_examples=eval_examples,
    )
    # 修正 collator 的 image_max_pixels 和 max_length
    trainer.data_collator.max_length = train_config.get("max_seq_length")
    trainer.data_collator.image_max_pixels = train_config.get("image_max_pixels")
    trainer.data_collator.ignore_index = train_config.get("ignore_index", -100)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(output_dir)
    return trainer
