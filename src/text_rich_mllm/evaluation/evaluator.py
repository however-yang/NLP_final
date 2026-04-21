from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from text_rich_mllm.utils.constants import AnswerType, DatasetName
from text_rich_mllm.evaluation.metrics_chartqa import chartqa_score
from text_rich_mllm.evaluation.metrics_docvqa import anls_score
from text_rich_mllm.evaluation.metrics_mcq import multiple_choice_accuracy
from text_rich_mllm.evaluation.normalization import is_valid_choice_prediction, normalize_answer
from text_rich_mllm.evaluation.parsing import parse_prediction
from text_rich_mllm.schemas import PredictionRecord, UnifiedSample


class UnifiedEvaluator:
    def evaluate(self, samples: Iterable[UnifiedSample], raw_predictions: dict[str, str]) -> tuple[list[PredictionRecord], dict[str, float]]:
        records: list[PredictionRecord] = []
        scores_by_dataset: dict[str, list[float]] = defaultdict(list)
        invalid_choice_counts: dict[str, int] = defaultdict(int)
        choice_counts: dict[str, int] = defaultdict(int)
        for sample in samples:
            raw_prediction = raw_predictions.get(sample.sample_id, "")
            parsed_prediction = parse_prediction(raw_prediction, answer_type=sample.answer_type)
            normalized_prediction = normalize_answer(parsed_prediction, sample.answer_type)
            score = self._score(sample, parsed_prediction)
            scores_by_dataset[sample.dataset_name].append(score)
            if sample.answer_type == AnswerType.MULTIPLE_CHOICE.value:
                choice_counts[sample.dataset_name] += 1
                if not is_valid_choice_prediction(raw_prediction):
                    invalid_choice_counts[sample.dataset_name] += 1
            records.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    dataset_name=sample.dataset_name,
                    gold_answer=sample.gold_answer,
                    raw_prediction=raw_prediction,
                    parsed_prediction=parsed_prediction,
                    normalized_prediction=normalized_prediction,
                    answer_type=sample.answer_type,
                    score=score,
                    task_type=sample.task_type,
                    split=sample.split,
                    metadata=sample.metadata,
                )
            )
        summary = {
            dataset_name: sum(dataset_scores) / max(len(dataset_scores), 1)
            for dataset_name, dataset_scores in scores_by_dataset.items()
        }
        summary["overall"] = sum(record.score for record in records) / len(records) if records else 0.0
        summary["invalid_output_rate"] = {
            dataset_name: invalid_choice_counts[dataset_name] / max(choice_counts[dataset_name], 1)
            for dataset_name in choice_counts
        }
        return records, summary

    def _score(self, sample: UnifiedSample, prediction: str) -> float:
        if sample.dataset_name == DatasetName.DOCVQA.value:
            return anls_score(prediction, sample.gold_answer)
        if sample.dataset_name == DatasetName.CHARTQA.value:
            return chartqa_score(prediction, sample.gold_answer, answer_type=sample.answer_type)
        if sample.dataset_name in {DatasetName.INFOGRAPHICVQA.value, DatasetName.TEXTVQA.value}:
            return anls_score(prediction, sample.gold_answer)
        if sample.answer_type == AnswerType.MULTIPLE_CHOICE.value:
            return multiple_choice_accuracy(prediction, sample.gold_answer)
        return float(normalize_answer(prediction, sample.answer_type) == normalize_answer(sample.gold_answer, sample.answer_type))

