from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.schemas import UnifiedSample


def build_infographic_prompt(sample: UnifiedSample, constraint: str, *, style: str) -> str:
    if style == PromptStyle.DIRECT.value:
        return (
            "Answer the question using the infographic image.\n\n"
            f"Question: {sample.question}\n"
            "Answer:"
        )
    return (
        "You are answering a question about an infographic image.\n"
        "Read text, layout, charts, and icons carefully before answering.\n"
        f"{constraint}\n\n"
        f"Question: {sample.question}\n"
        "Answer:"
    )
