from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.schemas import UnifiedSample


def build_chartqa_prompt(sample: UnifiedSample, constraint: str, *, style: str) -> str:
    if style == PromptStyle.DIRECT.value:
        return (
            "Answer the question using the chart image.\n\n"
            f"Question: {sample.question}\n"
            "Answer:"
        )
    return (
        "You are answering a chart question.\n"
        "Use the chart image to determine the correct answer.\n"
        f"{constraint}\n\n"
        f"Question: {sample.question}\n"
        "Answer:"
    )

