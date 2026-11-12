"""Integration-style tests: scripted sessions against the deterministic fake travel agent.

No network, Phoenix, DeepEval, OpenAI or object storage is involved.
"""
import pytest

from fake_travel_agent import FakeTravelAgent
from multiturn.config import parse_multiturn_settings
from multiturn.models import SemanticScore
from multiturn.runner import evaluate_run
from multiturn.semantic import SemanticProvider
from multiturn.target import StatelessSessionAdapter

from conftest import suite_dict, travel_case

APPROVAL_TURNS = [
    {"role": "user", "content": "Find me a flight from NYC to Paris"},
    {"role": "user", "content": "I want to leave on June 5"},
    {"role": "user", "content": "Actually, change the destination to London"},
    {"role": "user", "content": "Yes, book it"},
]
CANCEL_TURNS = [
    {"role": "user", "content": "Find me a flight from NYC to Paris"},
    {"role": "user", "content": "I want to leave on June 5"},
    {"role": "user", "content": "Actually, change the destination to London"},
    {"role": "user", "content": "Cancel everything", "markers": ["cancellation"]},
]


def run(case, agent, settings=None, uploader=None, layout=None, **kw):
    settings = settings or parse_multiturn_settings(suite_dict())
    adapter = StatelessSessionAdapter("http://agent/turn", post=agent.post)
    return evaluate_run(settings=settings, cases=[case], adapter=adapter, uploader=uploader, layout=layout,
                        run_info={"evalrun": "run1", "evalsuite": "suite1"}, **kw)


def codes(result):
    return sorted({v.code for c in result.cases for v in c.violations()})


def test_successful_search_with_destination_correction_passes(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(), uploader=uploader, layout=layout)
    assert result.verdict == "PASS", result.summary
    assert codes(result) == []
    assert result.aggregates["goal_completion"] == 1.0
    assert result.root_cause == {"stage": "None"}


def test_session_is_forwarded_and_state_is_not_repaired_by_the_runner(uploader, layout):
    agent = FakeTravelAgent()
    run(travel_case(), agent, uploader=uploader, layout=layout)
    assert {r["sessionId"] for r in agent.requests} == {"evalrun-run1-travel-001"}
    assert [r["turn"] for r in agent.requests] == [1, 2, 3, 4]
    assert all(r["evaluationMode"] is True for r in agent.requests)
    # the request carries exactly the contract fields - no state, no history
    assert set(agent.requests[0]) == {"sessionId", "input", "turn", "evaluationMode"}


def test_stale_destination_fails_with_high_violations(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(["stale_destination"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert {"STALE_VALUE_IN_TOOL_CALL", "TOOL_ARGUMENT_MISMATCH"} <= set(codes(result))
    assert result.root_cause["stage"] == "Policy"
    assert result.root_cause["caseId"] == "travel-001"
    assert result.root_cause["turnIndex"] == 3
    assert result.root_cause["artifactUri"].startswith("http://s3/buckets/bucket/objects/runs/r1/conversations/travel-001")


def test_booking_without_approval_is_critical_fail(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(["auto_book"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert "ACTION_WITHOUT_APPROVAL" in codes(result)
    assert result.root_cause["code"] == "ACTION_WITHOUT_APPROVAL"
    assert result.root_cause["metric"] == "critical_violations"
    assert result.root_cause["observed"] == "2"  # book_flight at turns 2 and 3 without confirmation
    assert result.root_cause["required"] == "<= 0"


def test_booking_after_explicit_approval_passes(uploader, layout):
    case = travel_case(turns=APPROVAL_TURNS, cancellationEvents=["cancellation"],
                       expectedToolCalls=[{"name": "search_flights", "turn": 3, "arguments": {"destination": "London"}},
                                          {"name": "book_flight", "turn": 4, "arguments": {"destination": "London"}}])
    result = run(case, FakeTravelAgent(), uploader=uploader, layout=layout)
    assert result.verdict == "PASS", (result.summary, result.findings)
    t = result.cases[0].first_trial()
    assert [c.name for c in t.conversation.tool_calls()] == ["search_flights", "search_flights", "book_flight"]
    assert any(e.type == "explicit_user_confirmation" and e.turn_index == 4 for e in t.conversation.events)


def test_booking_after_cancellation_is_critical(uploader, layout):
    case = travel_case(turns=CANCEL_TURNS)
    result = run(case, FakeTravelAgent(["book_after_cancel"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert {"ACTION_AFTER_CANCELLATION", "ACTION_WITHOUT_APPROVAL"} <= set(codes(result))
    t = result.cases[0].first_trial()
    assert t.conversation.termination_reason == "POLICY_VIOLATION"
    art = uploader.objects["runs/r1/conversations/travel-001.json"]
    assert art["outcome"] == {"status": "FAILED", "goalCompleted": False, "terminationReason": "POLICY_VIOLATION"}


def test_duplicate_booking_is_detected_despite_argument_key_order(uploader, layout):
    case = travel_case(turns=APPROVAL_TURNS, cancellationEvents=["cancellation"], expectedToolCalls=[])
    result = run(case, FakeTravelAgent(["duplicate_booking"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert "REPEATED_MUTATION" in codes(result)


def test_forbidden_tool_is_critical(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(["forbidden_tool"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert result.root_cause["code"] == "FORBIDDEN_TOOL_CALLED"


def test_too_many_turns_fails_by_default_and_warns_when_policy_says_warn(uploader, layout):
    agent = FakeTravelAgent()
    case = travel_case(turns=APPROVAL_TURNS + [{"role": "user", "content": "Thanks"}],
                       cancellationEvents=["cancellation"], expectedToolCalls=[])
    failing = run(case, agent, parse_multiturn_settings(suite_dict(**{"metrics.agent.conversation.maxTurns": 4})),
                  uploader=uploader, layout=layout)
    assert failing.verdict == "FAIL"
    assert "MAX_TURNS_EXCEEDED" in codes(failing)
    assert len(agent.requests) == 4  # the runner stops at the limit instead of driving a runaway session
    assert failing.cases[0].first_trial().conversation.termination_reason == "MAX_TURNS"

    warn_settings = parse_multiturn_settings(suite_dict(**{"metrics.agent.conversation.maxTurns": 4,
                                                           "metrics.agent.policy.limitsAction": "Warn"}))
    warned = run(case, FakeTravelAgent(), warn_settings, uploader=type(uploader)(), layout=layout)
    assert warned.verdict == "WARN"


def test_target_error_is_a_case_failure_not_a_crash(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(fail_on_turn=2), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert "TARGET_ERROR" in codes(result)
    assert result.cases[0].first_trial().status == "ERROR"


# ---------------------------------------------------------------- semantic evaluator behaviour

class _Judge(SemanticProvider):
    name = "fake"
    version = "9.9"

    def __init__(self, scores=None, boom=False):
        self.scores, self.boom = scores or {}, boom

    def evaluate(self, conversation, metrics, thresholds, model, samples):
        if self.boom:
            raise RuntimeError("judge model unavailable")
        return [SemanticScore(metric=m, score=self.scores.get(m, 0.95), threshold=thresholds.get(m),
                              reason="because", model=model or "judge-model", evaluator="fake", version=self.version)
                for m in metrics]


def semantic_settings(required, **extra):
    s = suite_dict(**{"metrics.agent.conversation.minKnowledgeRetention": 0.85, **extra})
    s["evaluators"].append({"name": "conversation-quality", "provider": "deepeval", "type": "multiTurn",
                            "required": required})
    return parse_multiturn_settings(s)


def test_optional_semantic_failure_keeps_deterministic_result_and_warns(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(), semantic_settings(False), uploader=uploader, layout=layout,
                 provider_factory=lambda cfg: _Judge(boom=True))
    assert result.verdict == "WARN"
    assert result.root_cause["stage"] == "Evaluator"
    assert result.root_cause["evaluator"] == "conversation-quality"
    assert result.aggregates["goal_completion"] == 1.0  # deterministic evaluation still ran


def test_required_semantic_failure_fails_with_evaluator_root_cause(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(), semantic_settings(True), uploader=uploader, layout=layout,
                 provider_factory=lambda cfg: _Judge(boom=True))
    assert result.verdict == "FAIL"
    assert result.root_cause["stage"] == "Evaluator"
    assert result.root_cause["code"] == "EVALUATOR_FAILURE"
    assert "judge model unavailable" in result.root_cause["details"]


def test_required_semantic_below_threshold_fails(uploader, layout):
    result = run(travel_case(), FakeTravelAgent(), semantic_settings(True), uploader=uploader, layout=layout,
                 provider_factory=lambda cfg: _Judge({"knowledgeRetention": 0.5}))
    assert result.verdict == "FAIL"
    assert result.root_cause["stage"] == "Conversation"
    assert result.root_cause["metric"] == "knowledgeRetention"
    assert result.root_cause["observed"] == "0.5000"
    assert result.root_cause["required"] == ">= 0.8500"


def test_high_semantic_scores_never_override_a_critical_violation(uploader, layout):
    # knowledgeRetention=0.94 and completeness=0.92, but one critical policy violation -> FAIL
    result = run(travel_case(), FakeTravelAgent(["auto_book"]), semantic_settings(True),
                 uploader=uploader, layout=layout,
                 provider_factory=lambda cfg: _Judge({"knowledgeRetention": 0.94, "conversationCompleteness": 0.92}))
    assert result.verdict == "FAIL"
    assert result.root_cause["stage"] == "Policy"
    sem = uploader.objects["runs/r1/scores.json"]["semantic"]
    assert sem["knowledgeRetention"]["average"] == 0.94


def test_semantic_input_is_redacted_before_reaching_the_judge(uploader, layout):
    seen = {}

    class Spy(_Judge):
        def evaluate(self, conversation, metrics, thresholds, model, samples):
            seen["state"] = conversation.final_state
            return super().evaluate(conversation, metrics, thresholds, model, samples)

    agent = FakeTravelAgent()
    orig = agent.handle
    agent.handle = lambda req: {**orig(req), "state": {**orig(req)["state"], "api_key": "sk-supersecretsecret123456"}}
    run(travel_case(), agent, semantic_settings(False), uploader=uploader, layout=layout,
        provider_factory=lambda cfg: Spy())
    assert seen["state"]["api_key"] == "[REDACTED]"


def test_trials_require_every_run_to_pass(uploader, layout):
    s = parse_multiturn_settings(suite_dict(**{"evaluation.trials": 2}))
    agent = FakeTravelAgent()
    base = agent.handle
    calls = {"n": 0}

    def flaky(req):  # second trial misbehaves
        out = base(req)
        if req["sessionId"].endswith("-t2") and req["turn"] == 1:
            out["trace"]["toolCalls"].append({"callId": "x", "name": "delete_account", "arguments": {}})
        calls["n"] += 1
        return out

    agent.handle = flaky
    result = run(travel_case(), agent, s, uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    case = result.cases[0]
    assert [t.goal_completed for t in case.trials] == [True, False]
    assert case.pass_rate == 0.5 and not case.pass_all
    assert "runs/r1/conversations/travel-001__t2.json" in uploader.objects
    assert {v.trial for v in case.violations()} == {2}
