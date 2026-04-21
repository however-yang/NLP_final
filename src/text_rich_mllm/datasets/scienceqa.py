from __future__ import annotations

from typing import Any

from text_rich_mllm.utils.constants import AnswerType, DatasetName, MULTIPLE_CHOICE_LABELS, TaskType
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.schemas import UnifiedSample


class ScienceQAAdapter(BaseDatasetAdapter):
    dataset_name = DatasetName.SCIENCEQA.value
    task_type = TaskType.SCIENTIFIC_QA.value
    answer_type = AnswerType.MULTIPLE_CHOICE.value

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
        image_root: str | None = None,
    ) -> UnifiedSample:
        raw_question = str(record.get("question") or "")
        raw_choices = str(record.get("choices") or "")
        choices = self._parse_mcq_string(raw_choices)
        answer_text = str(record.get("answer") or "").strip()
        if answer_text in MULTIPLE_CHOICE_LABELS:
            gold_answer = answer_text
        else:
            gold_answer = self._find_choice_label(answer_text, choices) or answer_text
        image_paths = self._extract_image_paths(record, image_root=image_root)
        image_path = image_paths[0] if image_paths else ""
        sample_id = record.get("id") or record.get("sample_id") or f"scienceqa-{split}-{index}"
        return UnifiedSample(
            sample_id=str(sample_id),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            image_path=self._join_image_path(image_root, image_path),
            question=self._strip_prefix(raw_question, "[QUESTION]"),
            choices=choices,
            gold_answer=gold_answer,
            answer_type=self.answer_type,
            split=split,
            metadata={
                "raw_question": raw_question,
                "raw_choices": raw_choices,
                "raw_answer": answer_text,
                "solution": record.get("solution"),
                "chain_of_thought": record.get("CTH"),
                "hf_split": record.get("_hf_split"),
                "image_paths": image_paths,
            },
        )

