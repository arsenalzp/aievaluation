import sys

import pytest

from multiturn.config import parse_multiturn_settings
from multiturn.deepeval_provider import DeepEvalProvider
from multiturn.models import Conversation, ToolCall, Turn
from multiturn.semantic import run_semantic_evaluators

from conftest import suite_dict


def conv():
    return Conversation("c1", "s1", turns=[
        Turn(1, "user", "find a flight to Paris"),
        Turn(1, "assistant", "searching", tool_calls=[ToolCall("1", "search_flights", {"destination": "Paris"},
                                                              result={"n": 2})]),
        Turn(2, "user", "make that London"),
        Turn(2, "assistant", "ok"),
    ])


def settings(required=False, **o):
    suite = suite_dict(**{"metrics.agent.conversation.minKnowledgeRetention": 0.8,
                          "metrics.agent.conversation.minTurnRelevancy": 0.7, **o})
    suite["evaluators"].append({"name": "quality", "provider": "deepeval", "type": "multiTurn", "required": required,
                                "metrics": ["knowledgeRetention", "turnRelevancy"], "samples": 3, "model": "judge-x"})
    return parse_multiturn_settings(suite)


def test_builds_a_deepeval_conversational_test_case_from_the_canonical_model():
    pytest.importorskip("deepeval")
    provider = DeepEvalProvider()  # real installed library; no LLM call happens here
    tc = provider.build_test_case(conv())
    assert [t.role for t in tc.turns] == ["user", "assistant", "user", "assistant"]
    (tool,) = tc.turns[1].tools_called
    assert (tool.name, tool.input_parameters, tool.output) == ("search_flights", {"destination": "Paris"}, {"n": 2})
    assert provider.version and tc.name == "c1"


class FakeMetric:
    scores = iter([])

    def __init__(self, threshold, model, include_reason, async_mode, verbose_mode):
        assert async_mode is False  # the runner never starts an event loop inside DeepEval
        self.threshold, self.model = threshold, model
        self.score = self.reason = self.evaluation_model = None

    def measure(self, test_case, _show_indicator=True):
        self.score = next(FakeMetric.scores)
        self.reason = f"reason for {self.score}"
        self.evaluation_model = f"{self.model}-resolved"
        return self.score


class ExplodingMetric(FakeMetric):
    def measure(self, test_case, _show_indicator=True):
        raise RuntimeError("rate limited")


def patched_provider(monkeypatch, **classes):
    pytest.importorskip("deepeval")
    provider = DeepEvalProvider()
    for name, cls in classes.items():
        monkeypatch.setattr(provider._metrics, name, cls)
    return provider


def test_median_spread_reason_and_model_are_recorded(monkeypatch):
    FakeMetric.scores = iter([0.6, 0.9, 0.8, 0.5, 0.5, 0.5])
    provider = patched_provider(monkeypatch, KnowledgeRetentionMetric=FakeMetric, TurnRelevancyMetric=FakeMetric)
    res = run_semantic_evaluators(conv(), settings(), provider_factory=lambda cfg: provider)
    (r,) = res
    assert r.error is None
    ret, rel = r.scores
    assert (ret.metric, ret.score, ret.threshold, ret.samples, ret.spread) == ("knowledgeRetention", 0.8, 0.8, 3, [0.6, 0.9])
    assert ret.reason == "reason for 0.8" and ret.model == "judge-x-resolved"
    assert ret.evaluator == "quality" and ret.version == provider.version and ret.passed is True
    assert rel.score == 0.5 and rel.passed is False


def test_provider_error_is_captured_not_raised(monkeypatch):
    provider = patched_provider(monkeypatch, KnowledgeRetentionMetric=ExplodingMetric, TurnRelevancyMetric=ExplodingMetric)
    (r,) = run_semantic_evaluators(conv(), settings(required=True), provider_factory=lambda cfg: provider)
    assert r.required and "rate limited" in r.error
    assert all(s.score is None and s.error for s in r.scores)


def test_missing_deepeval_is_an_evaluator_error_not_a_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepeval", None)  # makes `import deepeval` raise ImportError
    (r,) = run_semantic_evaluators(conv(), settings())
    assert r.error and "deepeval is not installed" in r.error


def test_unknown_provider_name_is_reported():
    s = settings()
    s.evaluators[-1].provider = "mystery"
    (r,) = run_semantic_evaluators(conv(), s)
    assert "no semantic provider registered" in r.error
