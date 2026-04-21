from __future__ import annotations

import re
from typing import Any

from text_rich_mllm.utils.constants import AnswerType, DatasetName, TaskType
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.schemas import UnifiedSample


class MMMUAdapter(BaseDatasetAdapter):
    dataset_name = DatasetName.MMMU.value
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
        raw_options = record.get("options") or record.get("choices") or []
        if isinstance(raw_options, str):
            choices = self._parse_mcq_string(raw_options)
            if not choices:
                try:
                    import ast

                    parsed = ast.literal_eval(raw_options)
                    choices = [str(item) for item in parsed] if isinstance(parsed, list) else []
                except (ValueError, SyntaxError):
                    choices = []
        else:
            choices = [str(item) for item in raw_options]
        gold_answer = str(record.get("answer") or "").strip()
        image_paths = self._extract_image_paths(record, image_root=image_root)
        image_path = image_paths[0] if image_paths else ""
        sample_id = record.get("id") or record.get("sample_id") or f"mmmu-{split}-{index}"
        question = re.sub(r"<image\s+\d+>", "<image>", str(record["question"]))
        return UnifiedSample(
            sample_id=str(sample_id),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            image_path=self._join_image_path(image_root, image_path),
            question=question,
            choices=choices,
            gold_answer=gold_answer,
            answer_type=self.answer_type,
            split=split,
            metadata={
                "hf_subset": record.get("_hf_subset"),
                "subfield": record.get("subfield"),
                "img_type": record.get("img_type"),
                "topic_difficulty": record.get("topic_difficulty"),
                "question_type": record.get("question_type"),
                "explanation": record.get("explanation"),
                "image_paths": image_paths,
            },
        )

