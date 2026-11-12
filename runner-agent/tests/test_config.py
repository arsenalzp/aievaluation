import pytest

from multiturn.config import ConfigError, get_interaction_mode, parse_multiturn_settings

from conftest import suite_dict


def test_single_turn_suites_are_unaffected():
    legacy = {"metrics": {"agent": {"minToolSelection": 0.9, "maxPolicyViolations": 0}},
              "regressionPolicy": {"maxDrop": 0.03}, "execution": {"mode": "agent", "timeoutSeconds": 300}}
    assert get_interaction_mode(legacy) == "SingleTurn"
    s = parse_multiturn_settings(legacy)
    # the flat single-turn field is not a synonym of tools.minSelectionAccuracy
    assert not s.is_multi_turn and s.timeout_seconds == 300 and s.min_selection_accuracy is None


def test_full_multi_turn_suite():
    suite = suite_dict(**{
        "evaluation.trials": 3,
        "evaluation.trace": {"source": "auto", "phoenix": {"endpoint": "http://phoenix:6006", "project": "travel"}},
        "metrics.agent.conversation": {"maxTurns": 10, "minKnowledgeRetention": 0.85,
                                       "minConversationCompleteness": 0.85, "minTurnRelevancy": 0.8},
        "metrics.agent.tools": {"minSelectionAccuracy": 0.9, "minArgumentAccuracy": 0.9, "maxToolCalls": 20,
                                "maxRepeatedCalls": 1},
        "regressionPolicy": {"maxDrop": 0.03},
    })
    suite["evaluators"].append({"name": "conversation-quality", "provider": "deepeval", "type": "multiTurn",
                                "samples": 3, "model": "judge-1"})
    s = parse_multiturn_settings(suite)
    assert s.is_multi_turn and s.trials == 3 and s.trace_source == "auto"
    assert (s.phoenix_endpoint, s.phoenix_project) == ("http://phoenix:6006", "travel")
    assert (s.max_turns, s.max_tool_calls, s.max_repeated_calls) == (10, 20, 1)
    assert s.semantic_threshold("turnRelevancy") == 0.8 and s.regression_max_drop == 0.03
    (sem,) = s.semantic_evaluators()
    assert (sem.required, sem.samples, sem.model) == (False, 3, "judge-1")
    # metrics come only from the thresholds in metrics.agent.conversation
    assert s.semantic_metrics() == ["knowledgeRetention", "conversationCompleteness", "turnRelevancy"]
    assert s.builtin_evaluator().required is True


def test_defaults_zero_tolerance_and_required_builtin():
    s = parse_multiturn_settings({"evaluation": {"interactionMode": "MultiTurn"}})
    assert (s.max_critical_violations, s.max_high_violations, s.trials, s.trace_source) == (0, 0, 1, "inline")
    assert s.builtin_evaluator().required and s.semantic_evaluators() == []


def test_deepeval_evaluator_computes_exactly_the_metrics_with_a_threshold():
    suite = suite_dict(**{"metrics.agent.conversation": {"minTurnRelevancy": 0.7}})
    suite["evaluators"].append({"name": "q", "provider": "deepeval"})
    s = parse_multiturn_settings(suite)
    assert s.semantic_metrics() == ["turnRelevancy"]
    assert s.semantic_evaluators()[0].required is False  # deepeval is optional unless required: true


def test_builtin_evaluator_is_required_even_without_the_flag():
    suite = suite_dict()
    suite["evaluators"] = [{"name": "policy"}]
    assert parse_multiturn_settings(suite).builtin_evaluator().required is True


def test_deterministic_only_suite_needs_no_semantic_evaluator():
    s = parse_multiturn_settings(suite_dict())
    assert s.semantic_metrics() == [] and s.semantic_evaluators() == []


_DEEPEVAL = {"name": "q", "provider": "deepeval"}
_BUILTIN = {"name": "trajectory-policy", "provider": "builtin"}
_KR = {"metrics.agent.conversation": {"minKnowledgeRetention": 0.8}}


@pytest.mark.parametrize("patch, fragment", [
    ({"evaluation.interactionMode": "Chatty"}, "interactionMode"),
    ({"evaluation.trials": 11}, "trials"),
    ({"evaluation.trace": {"source": "phoenix"}}, "phoenix.endpoint"),
    ({"evaluation.trace": {"source": "magic"}}, "trace.source"),
    ({"metrics.agent.outcome.minGoalCompletion": 1.5}, "minGoalCompletion"),
    ({"metrics.agent.conversation.maxTurns": 0}, "maxTurns"),
    ({"metrics.agent.policy.limitsAction": "Ignore"}, "limitsAction"),
    ({"evaluators": [{"provider": "builtin"}]}, "name is required"),
    ({"evaluators": [{"name": "a", "provider": "other"}]}, "provider"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "metrics": ["knowledgeRetention"]}]}, "unknown field.*metrics.*thresholds"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "samples": 9}]}, "samples"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "required": "yes"}]}, "required must be a boolean"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "type": "deterministic"}]}, "only supports type multiTurn"),
    ({**_KR, "evaluators": [_DEEPEVAL, {**_DEEPEVAL, "name": "q2"}]}, "only one evaluator with provider deepeval"),
    ({"evaluators": [_BUILTIN, {"name": "b"}]}, "only one evaluator with provider builtin"),
    ({"evaluators": [{**_BUILTIN, "required": False}]}, "always required"),
    ({"evaluators": [{**_BUILTIN, "model": "gpt"}]}, "only to provider deepeval"),
    ({"evaluators": [{**_BUILTIN, "samples": 2}]}, "only to provider deepeval"),
    ({**_KR, "evaluators": [_BUILTIN]}, "minKnowledgeRetention is set but no evaluator computes it"),
    ({"evaluators": [_BUILTIN, _DEEPEVAL]}, "nothing to evaluate"),
    ({"metrics.agent.minToolSelection": 0.9}, "minToolSelection applies to single-turn.*minSelectionAccuracy"),
    ({"metrics.agent.minToolInvocation": 0.9}, "minToolInvocation applies to single-turn"),
    ({"metrics.agent.minToolExecution": 0.9}, "minToolExecution applies to single-turn"),
    ({"metrics.agent.maxPolicyViolations": 0}, "maxPolicyViolations applies to single-turn"),
    ({"evaluators": [{"name": "a"}, {"name": "a"}]}, "duplicated"),
])
def test_invalid_configuration_is_rejected_with_a_clear_message(patch, fragment):
    with pytest.raises(ConfigError, match=fragment):
        parse_multiturn_settings(suite_dict(**patch))
