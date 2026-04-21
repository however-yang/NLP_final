from text_rich_mllm.utils.constants import AnswerType, TaskType
from text_rich_mllm.training.mixing import mix_training_samples
from text_rich_mllm.schemas import UnifiedSample


def _sample(sample_id: str, dataset_name: str) -> UnifiedSample:
    return UnifiedSample(
        sample_id=sample_id,
        dataset_name=dataset_name,
        task_type=TaskType.DOCUMENT_QA.value,
        image_path="demo.png",
        question="Q",
        gold_answer="A",
        answer_type=AnswerType.OPEN_TEXT.value,
        split="train",
    )


def test_balanced_mixing_interleaves_datasets() -> None:
    mixed = mix_training_samples(
        [
            _sample("d1", "docvqa"),
            _sample("d2", "docvqa"),
            _sample("c1", "chartqa"),
            _sample("c2", "chartqa"),
        ],
        strategy="balanced",
    )
    assert [sample.dataset_name for sample in mixed[:4]] == ["docvqa", "chartqa", "docvqa", "chartqa"]


def test_balanced_mixing_interleaves_four_datasets() -> None:
    mixed = mix_training_samples(
        [
            _sample("d1", "docvqa"),
            _sample("c1", "chartqa"),
            _sample("i1", "infographicvqa"),
            _sample("t1", "textvqa"),
            _sample("d2", "docvqa"),
            _sample("c2", "chartqa"),
            _sample("i2", "infographicvqa"),
            _sample("t2", "textvqa"),
        ],
        strategy="balanced",
    )
    assert [sample.dataset_name for sample in mixed[:4]] == [
        "docvqa",
        "chartqa",
        "infographicvqa",
        "textvqa",
    ]

