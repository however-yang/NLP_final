from __future__ import annotations

from dataclasses import dataclass

from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.prompts import PromptBuilder
from text_rich_mllm.schemas import UnifiedSample


@dataclass(slots=True)
class TrainingExample:
    sample_id: str
    dataset_name: str
    image_path: str
    prompt: str
    target_answer: str
    task_id: int = 0  # TRA 使用；0 = unknown/default，由 build_training_examples_with_tra 填充


def build_training_examples(
    samples: list[UnifiedSample],
    *,
    prompt_style: str = PromptStyle.STRUCTURED.value,
) -> list[TrainingExample]:
    builder = PromptBuilder(style=prompt_style)
    return [
        TrainingExample(
            sample_id=sample.sample_id,
            dataset_name=sample.dataset_name,
            image_path=sample.image_path,
            prompt=builder.build(sample),
            target_answer=sample.gold_answer,
        )
        for sample in samples
    ]


def build_training_examples_with_tra(
    samples: list[UnifiedSample],
    *,
    prompt_style: str = PromptStyle.STRUCTURED.value,
    task_name_to_id: dict[str, int],
) -> list[TrainingExample]:
    """
    与 build_training_examples 相同，但额外填充 task_id 字段。
    task_name_to_id 来自 TRAConfig.task_name_to_id，
    dataset_name 不在映射中时默认为 0。
    """
    builder = PromptBuilder(style=prompt_style)
    return [
        TrainingExample(
            sample_id=sample.sample_id,
            dataset_name=sample.dataset_name,
            image_path=sample.image_path,
            prompt=builder.build(sample),
            target_answer=sample.gold_answer,
            task_id=task_name_to_id.get(sample.dataset_name, 0),
        )
        for sample in samples
    ]

