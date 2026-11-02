import pytest

from multiturn import gate
from multiturn.config import parse_multiturn_settings
from multiturn.gate import EvaluatorFailure, GateInput, SemanticAggregate, decide
from multiturn.models import Violation

from conftest import suite_dict


def viol(severity, code="X", case="c1", turn=2):
    return Violation(code=code, message=f"{code} happened", severity=severity, case_id=case, session_id="s",
                     turn_index=turn, evaluator="trajectory-policy")


def settings(**o):
    return parse_multiturn_settings(suite_dict(**o))


def inp(**kw):
    base = dict(settings=settings(), aggregates={"goal_completion": 1.0, "tool_selection_accuracy": 1.0,
                                                 "tool_argument_accuracy": 1.0},
                conversation_uri=lambda c: f"s3://b/conv/{c}.json", scores_uri="s3://b/scores.json",
                violations_uri="s3://b/violations.jsonl", total_cases=2)
    base.update(kw)
    return GateInput(**base)


def sem(avg, required=True, threshold=0.85, metric="knowledgeRetention"):
    return SemanticAggregate("conversation-quality", metric, required, avg, threshold, per_case={"c1": avg})


def test_clean_run_passes():
    d = decide(inp())
    assert d.verdict == "PASS" and d.root_cause == {"stage": "None"}


def test_critical_violation_overrides_high_semantic_scores():
    d = decide(inp(violations=[viol("CRITICAL", "ACTION_WITHOUT_APPROVAL", turn=4)],
                   semantic=[sem(0.94), sem(0.92, metric="conversationCompleteness")]))
    assert d.verdict == "FAIL"
    rc = d.root_cause
    assert rc["stage"] == "Policy" and rc["code"] == "ACTION_WITHOUT_APPROVAL"
    assert rc["evaluator"] == "trajectory-policy" and rc["caseId"] == "c1" and rc["turnIndex"] == 4
    assert rc["observed"] == "1" and rc["required"] == "<= 0"
    assert rc["artifactUri"] == "s3://b/conv/c1.json"


def test_precedence_runner_failure_beats_critical_beats_threshold_beats_semantic():
    full = dict(
        runner_error="boom",
        violations=[viol("CRITICAL", "A"), viol("HIGH", "B")],
        aggregates={"goal_completion": 0.1, "tool_selection_accuracy": 1.0, "tool_argument_accuracy": 1.0},
        semantic=[sem(0.1)])
    assert decide(inp(**full)).root_cause["stage"] == "Execution"
    full.pop("runner_error")
    assert decide(inp(**full)).root_cause["code"] == "A"                       # critical
    full["violations"] = [viol("HIGH", "B")]
    assert decide(inp(**full)).root_cause["code"] == "B"                       # high / threshold tier
    full["violations"] = []
    assert decide(inp(**full)).root_cause["code"] == "GOAL_COMPLETION_BELOW_MIN"
    full["aggregates"] = {"goal_completion": 1.0}
    d = decide(inp(**full))
    assert d.root_cause["stage"] == "Conversation" and d.verdict == "FAIL"     # required semantic


def test_max_critical_and_high_limits_are_configurable():
    s = settings(**{"metrics.agent.policy.maxCriticalViolations": 1, "metrics.agent.policy.maxHighViolations": 1})
    assert decide(inp(settings=s, violations=[viol("CRITICAL"), viol("HIGH")])).verdict == "PASS"
    assert decide(inp(settings=s, violations=[viol("CRITICAL"), viol("CRITICAL")])).verdict == "FAIL"


def test_threshold_failures_carry_observed_and_required():
    d = decide(inp(aggregates={"goal_completion": 0.5, "tool_selection_accuracy": 0.8, "tool_argument_accuracy": 1.0}))
    assert d.root_cause["metric"] == "goalCompletion"
    assert (d.root_cause["observed"], d.root_cause["required"]) == ("0.5000", ">= 0.9000")
    assert [f["metric"] for f in d.findings] == ["goalCompletion", "toolSelectionAccuracy"]


def test_optional_evaluator_problems_only_warn():
    d = decide(inp(evaluator_failures=[EvaluatorFailure("conversation-quality", False, "api down", "c1")]))
    assert d.verdict == "WARN" and d.root_cause["stage"] == "Evaluator" and d.root_cause["caseId"] == "c1"
    d = decide(inp(semantic=[sem(0.2, required=False)]))
    assert d.verdict == "WARN" and "optional" in d.root_cause["details"]


def test_required_evaluator_failure_fails():
    d = decide(inp(evaluator_failures=[EvaluatorFailure("trajectory-policy", True, "check crashed")]))
    assert d.verdict == "FAIL" and d.root_cause["code"] == "EVALUATOR_FAILURE"


def test_medium_and_low_violations_warn():
    d = decide(inp(violations=[viol("MEDIUM", "MAX_TURNS_EXCEEDED")]))
    assert d.verdict == "WARN" and d.root_cause["code"] == "MAX_TURNS_EXCEEDED"


def test_semantic_without_threshold_is_informational():
    assert decide(inp(semantic=[sem(0.1, threshold=None)])).verdict == "PASS"


def test_regression_failure_and_optional_semantic_regression():
    rf = {"metric": "goal_completion", "current": 0.8, "baseline": 1.0, "drop": -0.2, "maxDrop": 0.03}
    d = decide(inp(regression_failures=[rf]))
    assert d.verdict == "FAIL" and d.root_cause["stage"] == "Regression"
    d = decide(inp(regression_failures=[{**rf, "metric": "knowledgeRetention", "required": False}]))
    assert d.verdict == "WARN"
