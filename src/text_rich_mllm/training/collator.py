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

