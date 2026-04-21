from text_rich_mllm.utils.constants import PromptStyle, mcq_choice_label
from text_rich_mllm.schemas import UnifiedSample


def build_mcq_prompt(sample: UnifiedSample, constraint: str, *, style: str) -> str:
    option_lines = []
    for index, choice in enumerate(sample.choices):
        label = mcq_choice_label(index)
        option_lines.append(f"{label}. {choice}")
    options_block = "\n".join(option_lines)
    if style == PromptStyle.DIRECT.value:
        return (
            f"Question: {sample.question}\n"
            f"Options:\n{options_block}\n"
            "Answer:"
        )
    return (
        "You are answering a multiple-choice multimodal reasoning question.\n"
        "Use the image and question together.\n"
        f"{constraint}\n\n"
        f"Question: {sample.question}\n"
        f"Options:\n{options_block}\n"
        "Answer:"
    )

