from text_rich_mllm.utils.constants import AnswerType


def build_answer_constraint(answer_type: str) -> str:
    if answer_type == AnswerType.NUMERIC.value:
        return "Return only the final numeric answer. Do not add units unless the question explicitly requires them."
    if answer_type == AnswerType.MULTIPLE_CHOICE.value:
        return "Return only the option letter."
    return "Return only a short answer span."

