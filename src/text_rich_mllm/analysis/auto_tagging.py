from __future__ import annotations

from collections import Counter

from text_rich_mllm.analysis.error_taxonomy import ErrorType
from text_rich_mllm.utils.constants import AnswerType, DatasetName
from text_rich_mllm.evaluation.normalization import is_valid_choice_prediction, normalize_answer
from text_rich_mllm.schemas import PredictionRecord


def infer_error_type(record: PredictionRecord) -> str:
    if record.score >= 1.0:
        return ErrorType.CORRECT.value
    if not record.raw_prediction.strip():
        return ErrorType.OUTPUT_MISMATCH.value
    if record.answer_type == AnswerType.MULTIPLE_CHOICE.value and not is_valid_choice_prediction(record.raw_prediction):
        return ErrorType.OUTPUT_MISMATCH.value
    if normalize_answer(record.raw_prediction, record.answer_type) == normalize_answer(record.gold_answer, record.answer_type):
        return ErrorType.OUTPUT_MISMATCH.value
    if record.dataset_name == DatasetName.CHARTQA.value:
        return ErrorType.CHART_REASONING_FAILURE.value
    if record.dataset_name == DatasetName.INFOGRAPHICVQA.value:
        return ErrorType.LAYOUT_GROUNDING_FAILURE.value
    if record.dataset_name in {DatasetName.SCIENCEQA.value, DatasetName.MMMU.value}:
        return ErrorType.SCIENTIFIC_REASONING_FAILURE.value
    if record.dataset_name == DatasetName.TEXTVQA.value:
        return ErrorType.TEXT_READING_FAILURE.value
    return ErrorType.TEXT_READING_FAILURE.value


def tag_prediction_records(records: list[PredictionRecord]) -> tuple[list[PredictionRecord], dict[str, int]]:
    counter: Counter[str] = Counter()
    for record in records:
        record.error_type = infer_error_type(record)
        counter[record.error_type] += 1
    return records, dict(counter)

