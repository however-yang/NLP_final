from __future__ import annotations

import ast
from typing import Any

from text_rich_mllm.utils.constants import AnswerType, DatasetName, TaskType
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.schemas import UnifiedSample


def _parse_answer_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple)):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except (SyntaxError, ValueError, TypeError):
                pass
        return [s]
    return [str(value).strip()]


class DocVQAAdapter(BaseDatasetAdapter):
    dataset_name = DatasetName.DOCVQA.value
    task_type = TaskType.DOCUMENT_QA.value
    answer_type = AnswerType.OPEN_TEXT.value

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
        image_root: str | None = None,
    ) -> UnifiedSample:
        nested = record.get("json")
        nested_dict = nested if isinstance(nested, dict) else None

        if nested_dict is not None:
            question = str(nested_dict.get("question") or nested_dict.get("query") or "").strip()
            answers_list = _parse_answer_list(nested_dict.get("answers"))
            gold_answer = answers_list[0] if answers_list else str(nested_dict.get("answer") or "").strip()
            sample_id = nested_dict.get("questionId") or nested_dict.get("question_id")
            image_path = str(record.get("png") or record.get("image") or record.get("image_path") or "").strip()
            other_metadata = {k: v for k, v in nested_dict.items() if k not in ("question", "answers", "answer")}
        else:
            answers_list = _parse_answer_list(record.get("answers"))
            gold_answer = str(record.get("answer") or "").strip() or (answers_list[0] if answers_list else "")
            question = str(record.get("question") or record.get("query") or "").strip()
            sample_id = record.get("question_id") or record.get("sample_id")
            image_path = str(record.get("image") or record.get("image_path") or record.get("png") or "").strip()
            other_metadata = record.get("other_metadata") if isinstance(record.get("other_metadata"), dict) else {}

        if sample_id is None:
            sample_id = f"docvqa-{split}-{index}"

        om = other_metadata if isinstance(other_metadata, dict) else {}
        page_id = om.get("page_id")
        if page_id is None:
            page_id = record.get("page_id")
        doc_id = om.get("doc_id") or om.get("ucsf_document_id")
        if doc_id is None:
            doc_id = (record.get("other_metadata") or {}).get("doc_id") if isinstance(record.get("other_metadata"), dict) else None
        page_no = om.get("ucsf_document_page_no") or om.get("document_page_no")

        return UnifiedSample(
            sample_id=str(sample_id),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            image_path=self._join_image_path(image_root, image_path),
            question=question,
            gold_answer=str(gold_answer),
            answer_type=self.answer_type,
            split=split,
            metadata={
                "page_id": page_id,
                "doc_id": doc_id,
                "document_page_no": page_no,
                "other_metadata": om,
                "ocr_results": record.get("ocr_results"),
                "original_answers": answers_list,
                "docvqa_json_nested": nested_dict is not None,
            },
        )

