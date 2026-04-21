from __future__ import annotations

from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.models.peft_adapter import attach_lora_adapter
from text_rich_mllm.training.collator import build_training_examples
from text_rich_mllm.training.hf_trainer import train_with_hf_trainer
from text_rich_mllm.training.mixing import mix_training_samples


def prepare_training_run(
    model,
    samples,
    peft_config: dict,
    *,
    sampling_strategy: str = "balanced",
    prompt_style: str = PromptStyle.STRUCTURED.value,
):
    samples = mix_training_samples(samples, strategy=sampling_strategy)
    model = attach_lora_adapter(model, peft_config)
    examples = build_training_examples(samples, prompt_style=prompt_style)
    return model, examples


def run_training(
    *,
    model,
    processor,
    train_samples,
    peft_config: dict,
    train_config: dict,
    eval_samples=None,
    resume_from_checkpoint: str | None = None,
):
    model, train_examples = prepare_training_run(
        model,
        train_samples,
        peft_config,
        sampling_strategy=train_config.get("sampling", "balanced"),
        prompt_style=train_config.get("prompt_style", PromptStyle.STRUCTURED.value),
    )
    eval_examples = None
    if eval_samples:
        eval_examples = build_training_examples(
            eval_samples,
            prompt_style=train_config.get("prompt_style", PromptStyle.STRUCTURED.value),
        )
    trainer = train_with_hf_trainer(
        model=model,
        processor=processor,
        train_examples=train_examples,
        train_config=train_config,
        eval_examples=eval_examples,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    return model, train_examples, trainer

