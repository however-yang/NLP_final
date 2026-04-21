from text_rich_mllm.utils.constants import AnswerType, PromptStyle, TaskType
from text_rich_mllm.prompts import PromptBuilder
from text_rich_mllm.schemas import UnifiedSample


def test_prompt_builder_uses_mcq_template_when_choices_exist() -> None:
    sample = UnifiedSample(
        sample_id="1",
        dataset_name="scienceqa",
        task_type=TaskType.SCIENTIFIC_QA.value,
        image_path="demo.png",
        question="What is shown?",
        choices=["cat", "dog", "bird"],
        gold_answer="A",
        answer_type=AnswerType.MULTIPLE_CHOICE.value,
        split="validation",
    )
    prompt = PromptBuilder().build(sample)
    assert "Options:" in prompt
    assert "Return only the option letter." in prompt


def test_prompt_builder_uses_textvqa_template() -> None:
    sample = UnifiedSample(
        sample_id="3",
        dataset_name="textvqa",
        task_type=TaskType.SCENE_TEXT_QA.value,
        image_path="x.jpg",
        question="Who wrote this?",
        gold_answer="Jane",
        answer_type=AnswerType.OPEN_TEXT.value,
        split="train",
    )
    prompt = PromptBuilder().build(sample)
    assert "scene text" in prompt.lower()
    assert "Who wrote this?" in prompt


def test_prompt_builder_uses_infographic_template() -> None:
    sample = UnifiedSample(
        sample_id="4",
        dataset_name="infographicvqa",
        task_type=TaskType.INFOGRAPHIC_QA.value,
        image_path="y.png",
        question="What is the total?",
        gold_answer="100",
        answer_type=AnswerType.OPEN_TEXT.value,
        split="train",
    )
    prompt = PromptBuilder().build(sample)
    assert "infographic" in prompt.lower()
    assert "What is the total?" in prompt


def test_direct_prompt_is_less_constrained() -> None:
    sample = UnifiedSample(
        sample_id="2",
        dataset_name="docvqa",
        task_type=TaskType.DOCUMENT_QA.value,
        image_path="demo.png",
        question="What is the invoice number?",
        gold_answer="123",
        answer_type=AnswerType.OPEN_TEXT.value,
        split="validation",
    )
    prompt = PromptBuilder(style=PromptStyle.DIRECT.value).build(sample)
    assert "Return only" not in prompt
    assert "Question: What is the invoice number?" in prompt

