from text_rich_mllm.utils.constants import PromptStyle
from text_rich_mllm.schemas import UnifiedSample


def build_textvqa_prompt(sample: UnifiedSample, constraint: str, *, style: str) -> str:
    if style == PromptStyle.DIRECT.value:
        return (
            "Answer the question using the image. The answer may require reading text in the scene.\n\n"
            f"Question: {sample.question}\n"
            "Answer:"
        )
    return (
        "You are answering a question about text that appears in a natural image (scene text).\n"
        "Read the visible text in the image and answer concisely.\n"
        f"{constraint}\n\n"
        f"Question: {sample.question}\n"
        "Answer:"
    )
