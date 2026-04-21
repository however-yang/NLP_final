from __future__ import annotations

from typing import Any

from text_rich_mllm.utils.constants import AnswerType, DatasetName, TaskType
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.datasets.chartqa import ChartQAAdapter
from text_rich_mllm.schemas import UnifiedSample


class TextVQAAdapter(BaseDatasetAdapter):
    dataset_name = DatasetName.TEXTVQA.value
    task_type = TaskType.SCENE_TEXT_QA.value
    answer_type = AnswerType.OPEN_TEXT.value

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
        image_root: str | None = None,
    ) -> UnifiedSample:
        question = str(record.get("question") or record.get("query") or "").strip()
        answers = record.get("answers") or []
        gold_candidates: list[str] = []
        if isinstance(answers, list):
            gold_candidates = [str(a).strip() for a in answers if str(a).strip()]
            gold_answer = gold_candidates[0] if gold_candidates else str(record.get("answer") or "").strip()
        else:
            gold_answer = str(answers or record.get("answer") or "").strip()
        raw_img = record.get("image") or record.get("image_path") or record.get("jpg") or ""
        if isinstance(raw_img, dict):
            raw_img = raw_img.get("path") or ""
        image_path = str(raw_img).strip() if raw_img else ""
        sample_id = record.get("question_id") or record.get("sample_id") or record.get("id")
        if sample_id is None:
            sample_id = f"textvqa-{split}-{index}"
        answer_type = (
            AnswerType.NUMERIC.value
            if ChartQAAdapter._looks_numeric(gold_answer)
            else self.answer_type
        )
        return UnifiedSample(
            sample_id=str(sample_id),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            image_path=self._join_image_path(image_root, str(image_path) if image_path else ""),
            question=question,
            gold_answer=gold_answer,
            answer_type=answer_type,
            split=split,
            metadata={
                "image_id": record.get("image_id"),
                "set_name": record.get("set_name"),
                "answer_pool": gold_candidates if isinstance(answers, list) else [],
                "hf_split": record.get("_hf_split"),
            },
        )
