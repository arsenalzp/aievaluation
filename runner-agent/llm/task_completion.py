"""Custom Phoenix evaluator: did the single-step agent complete the user's task?

phoenix-evals (3.1.0) ships no reference-based judge (CorrectnessEvaluator takes only input/output),
so this follows the structure of the built-in ToolResponseHandlingEvaluator: a ClassificationEvaluator
with an EvalRun-owned prompt, Mustache variables and the choices correct=1.0 / incorrect=0.0.

Scope: the final answer versus the reference answer of the case (expectedOutput). Tool selection,
tool arguments and how tool results were handled are judged by the other evaluators.
"""
from typing import Any

from phoenix.evals import LLM, ClassificationEvaluator
from pydantic import BaseModel, Field

TASK_COMPLETION_PROMPT = """You are an impartial judge evaluating whether an AI agent completed the user's task. \
You are given the user's request, a reference answer that a correct agent gives for this request, and the agent's final answer.

IMPORTANT - Scope of Evaluation:
- You are ONLY evaluating whether the agent's final answer delivers the same result as the reference answer.
- Do NOT evaluate which tools were used, how they were called, or the style of the answer.
- The reference answer is the ground truth for this request; do not use outside knowledge to override it.

Criteria for CORRECT (task completed):
- The answer conveys the same essential result as the reference: the same key facts, identifiers, numbers, dates and decisions.
- Different wording, order or formatting is fine.
- Additional information is fine as long as it does not contradict the reference or change the result.
- If the reference states that there is no result (for example "No flights found."), an answer that clearly says the same is correct.

Criteria for INCORRECT (task not completed):
- A key fact, identifier, number, date or decision differs from the reference or is missing.
- The answer contradicts the reference, or claims a different outcome (for example reports an action as done that the reference does not report).
- The answer refuses, deflects, asks for information the request already contains, or answers a different question.
- The answer is too vague to tell whether it delivers the reference result.

Before providing your final judgment, explain your reasoning: list the essential elements of the reference answer and check each of them in the agent's answer.

<data>
<input>
{{input}}
</input>

<expected_output>
{{expected_output}}
</expected_output>

<output>
{{output}}
</output>
</data>

Given the above data, did the agent complete the task correctly or incorrectly?"""


class TaskCompletionEvaluator(ClassificationEvaluator):
    """Reference-based task completion judge (correct / incorrect + explanation)."""

    NAME = "task_completion"
    CHOICES = {"correct": 1.0, "incorrect": 0.0}
    DIRECTION = "maximize"

    class TaskCompletionInputSchema(BaseModel):
        input: str = Field(description="The user request.")
        expected_output: str = Field(description="The reference answer a correct agent gives for this request.")
        output: str = Field(description="The agent's final answer.")

    def __init__(self, llm: LLM, **kwargs: Any):
        super().__init__(
            name=self.NAME,
            llm=llm,
            prompt_template=[{"role": "user", "content": TASK_COMPLETION_PROMPT}],
            choices=self.CHOICES,
            direction=self.DIRECTION,
            input_schema=self.TaskCompletionInputSchema,
            **kwargs,
        )
