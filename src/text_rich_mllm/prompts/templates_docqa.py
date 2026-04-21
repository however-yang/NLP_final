from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.schemas import UnifiedSample


def build_docqa_prompt(sample: UnifiedSample, constraint: str, *, style: str) -> str:
    if style == PromptStyle.DIRECT.value:
        return (
            "Answer the question using the document image.\n\n"
            f"Question: {sample.question}\n"
            "Answer:"
        )
    return (
        "You are answering a document question.\n"
        "Read the document image carefully and answer the question.\n"
        f"{constraint}\n\n"
        f"Question: {sample.question}\n"
        "Answer:"
    )

