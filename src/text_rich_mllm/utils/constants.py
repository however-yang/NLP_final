from enum import Enum


class StrEnum(str, Enum):
    """Small compatibility shim for Python < 3.11."""


class DatasetName(StrEnum):
    DOCVQA = "docvqa"
    CHARTQA = "chartqa"
    INFOGRAPHICVQA = "infographicvqa"
    TEXTVQA = "textvqa"
    SCIENCEQA = "scienceqa"
    MMMU = "mmmu"


class TaskType(StrEnum):
    DOCUMENT_QA = "document_qa"
    CHART_QA = "chart_qa"
    INFOGRAPHIC_QA = "infographic_qa"
    SCENE_TEXT_QA = "scene_text_qa"
    SCIENTIFIC_QA = "scientific_qa"


class AnswerType(StrEnum):
    OPEN_TEXT = "open_text"
    NUMERIC = "numeric"
    MULTIPLE_CHOICE = "multiple_choice"


class PromptStyle(StrEnum):
    DIRECT = "direct"
    STRUCTURED = "structured"


MULTIPLE_CHOICE_LABELS = ("A", "B", "C", "D", "E", "F")


def mcq_choice_label(index: int) -> str:
    """将选项序号映射为 A/B/…/Z；ScienceQA/MMMU 等可出现 7+ 选项，不可仅用固定 6 元组。"""
    if index < 0:
        raise ValueError(index)
    if index < 26:
        return chr(ord("A") + index)
    return str(index + 1)


DATASET_TO_TASK = {
    DatasetName.DOCVQA: TaskType.DOCUMENT_QA,
    DatasetName.CHARTQA: TaskType.CHART_QA,
    DatasetName.INFOGRAPHICVQA: TaskType.INFOGRAPHIC_QA,
    DatasetName.TEXTVQA: TaskType.SCENE_TEXT_QA,
    DatasetName.SCIENCEQA: TaskType.SCIENTIFIC_QA,
    DatasetName.MMMU: TaskType.SCIENTIFIC_QA,
}
