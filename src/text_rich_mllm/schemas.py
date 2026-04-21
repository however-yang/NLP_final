from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from text_rich_mllm.utils.paths import resolve_sample_image_path


@dataclass(slots=True)
class UnifiedSample:
    sample_id: str
    dataset_name: str
    task_type: str
    image_path: str
    question: str
    choices: list[str] = field(default_factory=list)
    gold_answer: str = ""
    answer_type: str = "open_text"
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UnifiedSample":
        return cls(
            sample_id=str(payload["sample_id"]),
            dataset_name=str(payload["dataset_name"]),
            task_type=str(payload["task_type"]),
            image_path=resolve_sample_image_path(str(payload["image_path"])),
            question=str(payload["question"]),
            choices=list(payload.get("choices", [])),
            gold_answer=str(payload.get("gold_answer", "")),
            answer_type=str(payload.get("answer_type", "open_text")),
            split=str(payload.get("split", "train")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class PredictionRecord:
    sample_id: str
    dataset_name: str
    gold_answer: str
    raw_prediction: str
    parsed_prediction: str
    normalized_prediction: str
    answer_type: str
    score: float
    task_type: str = ""
    split: str = ""
    error_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
