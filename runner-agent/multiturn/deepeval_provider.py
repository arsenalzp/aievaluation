"""DeepEval implementation of :class:`SemanticProvider`.

DeepEval is imported lazily so the deterministic core works when it is not installed.
Written against deepeval 4.1.x: ``ConversationalTestCase`` / ``Turn`` / ``ToolCall`` from
``deepeval.test_case`` and ``KnowledgeRetentionMetric`` / ``ConversationCompletenessMetric`` /
``TurnRelevancyMetric`` from ``deepeval.metrics`` (``metric.measure(test_case)`` populates
``score``, ``reason`` and ``evaluation_model``).
"""
from __future__ import annotations

import os
import statistics
from typing import Any, Dict, List, Optional

from .config import EvaluatorConfig, PROVIDER_DEEPEVAL
from .llm_settings import PROVIDER_OPENAI, LLMSettings
from .models import Conversation, SemanticScore
from .semantic import SemanticProvider, register_semantic_provider

_METRIC_CLASSES = {
    "knowledgeRetention": "KnowledgeRetentionMetric",
    "conversationCompleteness": "ConversationCompletenessMetric",
    "turnRelevancy": "TurnRelevancyMetric",
}


class DeepEvalProvider(SemanticProvider):
    name = PROVIDER_DEEPEVAL

    def __init__(self) -> None:
        # Never send telemetry from an evaluation Job.
        os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
        try:
            import deepeval  # noqa: F401
            from deepeval import metrics as de_metrics
            from deepeval import test_case as de_cases
        except ImportError as exc:
            raise RuntimeError(f"deepeval is not installed ({exc}); install deepeval>=4.1 to use this evaluator") from exc
        self._metrics = de_metrics
        self._cases = de_cases
        self._version = getattr(deepeval, "__version__", "unknown")

    @property
    def version(self) -> str:
        return self._version

    def judge_model(self, llm: LLMSettings) -> Any:
        """EvalRun.spec.llm -> DeepEval judge model. Provider openai: the OpenAI API or an OpenAI-compatible server."""
        if llm.provider != PROVIDER_OPENAI:
            raise RuntimeError(f"the deepeval evaluator does not support LLM provider {llm.provider!r}")
        from deepeval.models import GPTModel
        return GPTModel(model=llm.model, api_key=llm.client_api_key, base_url=llm.base_url)

    def build_test_case(self, conversation: Conversation) -> Any:
        """Canonical Conversation -> ``deepeval.test_case.ConversationalTestCase``."""
        Turn, ToolCall = self._cases.Turn, self._cases.ToolCall
        turns = []
        for t in sorted(conversation.turns, key=lambda x: (x.index, 0 if x.role == "user" else 1)):
            kwargs: Dict[str, Any] = {"role": t.role, "content": t.content}
            if t.tool_calls:
                kwargs["tools_called"] = [
                    ToolCall(name=c.name, input_parameters=c.arguments, output=c.result) for c in t.tool_calls
                ]
            if t.retrieval_context:
                kwargs["retrieval_context"] = list(t.retrieval_context)
            turns.append(Turn(**kwargs))
        return self._cases.ConversationalTestCase(turns=turns, name=conversation.case_id)

    def evaluate(self, conversation: Conversation, metrics: List[str],
                 thresholds: Dict[str, Optional[float]], llm: Optional[LLMSettings], samples: int) -> List[SemanticScore]:
        if llm is None:
            raise RuntimeError("the deepeval evaluator needs a judge LLM; set EvalRun.spec.llm")
        judge = self.judge_model(llm)
        test_case = self.build_test_case(conversation)
        out: List[SemanticScore] = []
        for name in metrics:
            cls_name = _METRIC_CLASSES.get(name)
            if cls_name is None:
                out.append(SemanticScore(metric=name, score=None, threshold=thresholds.get(name),
                                         evaluator=PROVIDER_DEEPEVAL, version=self.version,
                                         error=f"unsupported metric {name!r}"))
                continue
            threshold = thresholds.get(name)
            scores: List[float] = []
            # reason of each sample, index-aligned with ``scores`` (None when the judge gave none)
            reasons: List[Optional[str]] = []
            used_model: Optional[str] = llm.model
            error: Optional[str] = None
            for _ in range(max(1, samples)):
                try:
                    metric = getattr(self._metrics, cls_name)(
                        threshold=threshold if threshold is not None else 0.5,
                        model=judge, include_reason=True, async_mode=False, verbose_mode=False,
                    )
                    score = metric.measure(test_case, _show_indicator=False)
                    reason = getattr(metric, "reason", None)
                    scores.append(float(score if score is not None else metric.score))
                    reasons.append(str(reason) if reason else None)
                    used_model = getattr(metric, "evaluation_model", None) or used_model
                except Exception as exc:  # noqa: BLE001
                    error = f"{type(exc).__name__}: {exc}"
                    break
            if error or not scores:
                out.append(SemanticScore(metric=name, score=None, threshold=threshold, model=used_model,
                                         evaluator=PROVIDER_DEEPEVAL, version=self.version, samples=samples,
                                         error=error or "no score produced"))
                continue
            median = statistics.median(scores)
            # report the reason of the sample closest to the median
            idx = min(range(len(scores)), key=lambda i: abs(scores[i] - median))
            out.append(SemanticScore(
                metric=name, score=round(float(median), 4), threshold=threshold,
                reason=reasons[idx], model=used_model,
                evaluator=PROVIDER_DEEPEVAL, version=self.version, samples=len(scores),
                spread=[min(scores), max(scores)] if len(scores) > 1 else None,
            ))
        return out


def _factory(_: EvaluatorConfig) -> SemanticProvider:
    return DeepEvalProvider()


register_semantic_provider(PROVIDER_DEEPEVAL, _factory)
