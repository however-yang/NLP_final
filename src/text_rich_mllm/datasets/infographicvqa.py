from __future__ import annotations

from typing import Any

from text_rich_mllm.utils.constants import AnswerType, DatasetName, TaskType
from text_rich_mllm.datasets.base import BaseDatasetAdapter
from text_rich_mllm.datasets.chartqa import ChartQAAdapter
from text_rich_mllm.schemas import UnifiedSample


def _is_due_style_document(record: dict[str, Any]) -> bool:
    """仅当整页 DUE document.jsonl（name + annotations[].key/values）时才走展开逻辑。"""
    name = record.get("name")
    if not name or not isinstance(name, str) or not str(name).strip():
        return False
    anns = record.get("annotations")
    if not isinstance(anns, list) or not anns:
        return False
    first = anns[0]
    if not isinstance(first, dict):
        return False
    if "key" not in first:
        return False
    vals = first.get("values")
    return isinstance(vals, list)


def _answer_list_from_values(values: list[Any]) -> list[str]:
    out: list[str] = []
    if not values or not isinstance(values, list):
        return out
    first = values[0]
    if not isinstance(first, dict):
        return out
    if "value_variants" in first and isinstance(first["value_variants"], list):
        out = [str(v).strip() for v in first["value_variants"] if str(v).strip()]
    if not out and first.get("value") is not None:
        v = str(first["value"]).strip()
        if v:
            out = [v]
    return out


class InfographicVQAAdapter(BaseDatasetAdapter):
    dataset_name = DatasetName.INFOGRAPHICVQA.value
    task_type = TaskType.INFOGRAPHIC_QA.value
    answer_type = AnswerType.OPEN_TEXT.value

    def convert_records(
        self,
        records: list[dict[str, Any]],
        *,
        split: str,
        image_root: str | None = None,
    ) -> list[UnifiedSample]:
        flattened: list[UnifiedSample] = []
        for index, record in enumerate(records):
            if _is_due_style_document(record):
                for ann_index, ann in enumerate(record["annotations"]):
                    flattened.append(
                        self._from_due_annotation(record, ann, ann_index, split=split, image_root=image_root)
                    )
            else:
                flattened.append(
                    self.convert_record(record, index=len(flattened), split=split, image_root=image_root)
                )
        return flattened

    def _from_due_annotation(
        self,
        doc: dict[str, Any],
        ann: dict[str, Any],
        ann_index: int,
        *,
        split: str,
        image_root: str | None,
    ) -> UnifiedSample:
        stem = str(doc["name"]).strip()
        ext = str(doc.get("image_ext") or "png").lstrip(".")
        image_rel = f"{stem}.{ext}"
        question = str(ann.get("key") or "").strip()
        values = ann.get("values") or []
        answers = _answer_list_from_values(values)
        gold_answer = answers[0] if answers else ""
        meta = ann.get("metadata") if isinstance(ann.get("metadata"), dict) else {}
        qid = meta.get("question_id") if meta else None
        sample_id = qid if qid is not None else f"{stem}-{ann_index}"
        answer_type = (
            AnswerType.NUMERIC.value
            if ChartQAAdapter._looks_numeric(gold_answer)
            else self.answer_type
        )
        return UnifiedSample(
            sample_id=str(sample_id),
            dataset_name=self.dataset_name,
            task_type=self.task_type,
            image_path=self._join_image_path(image_root, image_rel),
            question=question,
            gold_answer=gold_answer,
            answer_type=answer_type,
            split=split,
            metadata={
                "doc_name": stem,
                "annotation_index": ann_index,
                "answer_variants": answers,
                "question_type": meta.get("question_type") if meta else None,
            },
        )

    def convert_record(
        self,
        record: dict[str, Any],
        *,
        index: int,
        split: str,
        image_root: str | None = None,
    ) -> UnifiedSample:
        question = (
            str(record.get("question") or record.get("query") or record.get("instruction") or "").strip()
        )
        gold_answer = str(
            record.get("ground_truth")
            or record.get("answer")
            or record.get("label")
            or ""
        ).strip()
        if not gold_answer:
            answers = record.get("answers") or []
            if isinstance(answers, list):
                pool = [str(a).strip() for a in answers if str(a).strip()]
                gold_answer = pool[0] if pool else ""
        raw_img = record.get("image") or record.get("image_path") or record.get("jpg") or ""
        if isinstance(raw_img, dict):
            raw_img = raw_img.get("path") or ""
        image_path = str(raw_img).strip() if raw_img else ""
        sample_id = (
            record.get("sample_id")
            or record.get("question_id")
            or record.get("questionId")
            or record.get("id")
        )
        if sample_id is None:
            sample_id = f"infographicvqa-{split}-{index}"
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
                "question_type": record.get("question_type"),
                "hf_split": record.get("_hf_split"),
            },
        )
