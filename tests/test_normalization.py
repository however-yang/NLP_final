from text_rich_mllm.utils.constants import AnswerType
from text_rich_mllm.evaluation.normalization import is_valid_choice_prediction, normalize_answer, numeric_equal


def test_multiple_choice_normalization_extracts_label() -> None:
    assert normalize_answer("The answer is B.", AnswerType.MULTIPLE_CHOICE.value) == "B"


def test_numeric_normalization_removes_commas() -> None:
    assert normalize_answer("1,200", AnswerType.NUMERIC.value) == "1200"


def test_numeric_equal_uses_tolerance() -> None:
    assert numeric_equal("10.001", "10")


def test_free_form_sentence_is_not_treated_as_valid_choice() -> None:
    assert not is_valid_choice_prediction("It looks like a cat.")

