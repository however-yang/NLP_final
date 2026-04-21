from text_rich_mllm.utils.constants import PromptStyle, TaskType
from text_rich_mllm.prompts.answer_constraints import build_answer_constraint
from text_rich_mllm.prompts.templates_chartqa import build_chartqa_prompt
from text_rich_mllm.prompts.templates_docqa import build_docqa_prompt
from text_rich_mllm.prompts.templates_infographic import build_infographic_prompt
from text_rich_mllm.prompts.templates_mcq import build_mcq_prompt
from text_rich_mllm.prompts.templates_textvqa import build_textvqa_prompt
from text_rich_mllm.schemas import UnifiedSample


class PromptBuilder:
    def __init__(self, *, style: str = PromptStyle.STRUCTURED.value):
        self.style = style

    def build(self, sample: UnifiedSample) -> str:
        constraint = build_answer_constraint(sample.answer_type)
        if sample.choices:
            return build_mcq_prompt(sample, constraint, style=self.style)
        if sample.task_type == TaskType.CHART_QA.value:
            return build_chartqa_prompt(sample, constraint, style=self.style)
        if sample.task_type == TaskType.INFOGRAPHIC_QA.value:
            return build_infographic_prompt(sample, constraint, style=self.style)
        if sample.task_type == TaskType.SCENE_TEXT_QA.value:
            return build_textvqa_prompt(sample, constraint, style=self.style)
        return build_docqa_prompt(sample, constraint, style=self.style)

