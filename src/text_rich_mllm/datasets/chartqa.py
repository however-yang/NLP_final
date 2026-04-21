from __future__ import annotations

from text_rich_mllm.utils.constants import AnswerType, DatasetName, TaskType
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.schemas import UnifiedSample


class ChartQAAdapter(BaseDatasetAdapter):
    dataset_name = DatasetName.CHARTQA.value
    task_type = TaskType.CHART_QA.value
    answer_type = AnswerType.OPEN_TEXT.value

    def convert_record(
        self,
        record: dict,
        *,
        index: int,
        split: str,
        image_root: str | None = None,
    ) -> UnifiedSample:
        labels = record.get("label") or []
        if isinstance(labels, list):
            gold_answer = str(labels[0]) if labels else ""
        else:
            gold_answer = str(labels or record.get("answer") or "")
        answer_type = (
            AnswerType.NUMERIC.value
            if self._looks_numeric(gold_answer)
            else self.answer_type
        )
        image_path = record.get("image") or record.get("image_path") or record.get("imgname") or ""
        question = record.get("query") or record.get("question") or ""
        sample_id = record.get("sample_id") or record.get("id") or f"chartqa-{split}-{index}"
        return UnifiedSample(
            sample_id=str(sample_id),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            image_path=self._join_image_path(image_root, image_path),
            question=str(question),
            gold_answer=str(gold_answer),
            answer_type=str(answer_type),
            split=split,
            metadata={
                "raw_labels": labels if isinstance(labels, list) else [gold_answer],
                "source": record.get("human_or_machine"),
                "hf_split": record.get("_hf_split"),
            },
        )

    @staticmethod
    def _looks_numeric(value: str) -> bool:
        text = value.strip().replace(",", "")
        if not text:
            return False
        try:
            float(text.rstrip("%"))
            return True
        except ValueError:
            return False

