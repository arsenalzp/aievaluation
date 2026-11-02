"""Unit tests for every deterministic check, driven by hand-built canonical conversations."""
import pytest

from multiturn.checks import CheckContext, Limits, derive_events, run_checks, Check
from multiturn.models import Conversation, ToolCall, Turn

from conftest import travel_case


def conversation(calls_by_turn, states=None, user_turns=4, case=None):
    case = case or travel_case()
    conv = Conversation(case_id=case.case_id, session_id="s", metadata={"scriptedTurns": user_turns})
    for i in range(1, user_turns + 1):
        conv.turns.append(Turn(i, "user", f"user {i}"))
        calls = [ToolCall(f"call-{i}-{j}", name, args, turn_index=i, **kw)
                 for j, (name, args, kw) in enumerate(calls_by_turn.get(i, []), start=1)]
        conv.turns.append(Turn(i, "assistant", f"reply {i}", trace_id=f"trace-{i}", tool_calls=calls,
                               metadata={"state": (states or {}).get(i, {})}))
    conv.final_state = (states or {}).get(user_turns, {})
    conv.events = derive_events(case)
    return conv


def call(name, **args):
    return (name, args, {})


def run(case, conv, limits=None):
    ctx = CheckContext(case=case, conversation=conv, limits=limits or Limits())
    return {r.check: r for r in run_checks(ctx)}


def test_required_tool_success_and_failure():
    case = travel_case(requiredTools=["search_flights", "get_weather"])
    ok = run(case, conversation({3: [call("search_flights", destination="London"), call("get_weather")]}))
    assert ok["required_tool"].passed and ok["required_tool"].score == 1.0
    bad = run(case, conversation({3: [call("search_flights", destination="London")]}))["required_tool"]
    assert not bad.passed and bad.score == 0.5
    (v,) = bad.violations
    assert (v.code, v.severity, v.evidence["tool"]) == ("REQUIRED_TOOL_NOT_CALLED", "HIGH", "get_weather")


def test_forbidden_tool_is_critical_per_call():
    case = travel_case()
    res = run(case, conversation({1: [call("delete_account")], 2: [call("delete_account")]}))["forbidden_tool"]
    assert not res.passed and len(res.violations) == 2
    assert {v.severity for v in res.violations} == {"CRITICAL"}
    assert res.violations[0].turn_index == 1 and res.violations[0].tool_call_id == "call-1-1"
    assert res.violations[0].trace_id == "trace-1" and res.violations[0].evaluator == "trajectory-policy"


def test_tool_arguments_partial_matching():
    case = travel_case()
    good = run(case, conversation({3: [call("search_flights", origin="NYC", destination="london", date="June 5")]}))
    assert good["tool_arguments"].passed and good["tool_arguments"].score == 1.0
    bad = run(case, conversation({3: [call("search_flights", origin="NYC", destination="Paris")]}))["tool_arguments"]
    assert not bad.passed
    (v,) = bad.violations
    assert v.code == "TOOL_ARGUMENT_MISMATCH" and v.turn_index == 3
    assert v.evidence["mismatches"][0]["path"] == "$.destination"
    assert v.evidence["observedArguments"]["destination"] == "Paris"


def test_expected_call_in_wrong_turn_or_missing_is_reported():
    case = travel_case()
    wrong_turn = run(case, conversation({2: [call("search_flights", destination="London")]}))["tool_arguments"]
    assert wrong_turn.violations[0].code == "EXPECTED_TOOL_CALL_MISSING" and wrong_turn.violations[0].turn_index == 3


def test_latest_value_wins_detects_stale_tool_args_and_stale_state():
    case = travel_case()
    res = run(case, conversation(
        {2: [call("search_flights", destination="Paris")], 3: [call("search_flights", destination="Paris")]},
        states={2: {"destination": "Paris"}, 3: {"destination": "Paris"}, 4: {"destination": "London"}}))
    lv = res["latest_value_wins"]
    assert not lv.passed
    codes = sorted(v.code for v in lv.violations)
    assert codes == ["STALE_STATE", "STALE_VALUE_IN_TOOL_CALL"]  # turn 2 predates the correction: ignored
    assert all(v.turn_index == 3 for v in lv.violations)


def test_latest_value_wins_passes_when_the_correction_propagates():
    case = travel_case()
    res = run(case, conversation({3: [call("search_flights", destination="London")]},
                                 states={3: {"destination": "London"}, 4: {"destination": "London"}}))
    assert res["latest_value_wins"].passed


def approval_case():
    return travel_case(turns=[{"role": "user", "content": "find"}, {"role": "user", "content": "yes, book it"},
                              {"role": "user", "content": "thanks"}], cancellationEvents=["cancellation"],
                       latestValues=[], expectedToolCalls=[], expectedFinalState={})


def test_approval_before_mutation():
    case = approval_case()
    ok = run(case, conversation({2: [call("book_flight", id=1)]}, user_turns=3, case=case))
    assert ok["approval_before_mutation"].passed
    early = run(case, conversation({1: [call("book_flight", id=1)]}, user_turns=3, case=case))["approval_before_mutation"]
    (v,) = early.violations
    assert v.code == "ACTION_WITHOUT_APPROVAL" and v.severity == "CRITICAL" and v.turn_index == 1
    assert v.evidence["requiredEvent"] == "explicit_user_confirmation"


def test_agent_output_cannot_grant_approval():
    case = approval_case()
    conv = conversation({1: [call("book_flight", id=1)]}, user_turns=3, case=case)
    conv.turns[1].content = "yes, book it"  # assistant text containing the magic words
    res = run(case, conv)["approval_before_mutation"]
    assert not res.passed


def test_action_after_cancellation_and_resume():
    case = travel_case(turns=[{"role": "user", "content": "find"},
                              {"role": "user", "content": "stop", "markers": ["cancellation"]},
                              {"role": "user", "content": "ok go on", "markers": ["go_ahead"]},
                              {"role": "user", "content": "yes, book it"}],
                       resumeEvents=["go_ahead"], latestValues=[], expectedToolCalls=[], expectedFinalState={})
    after = run(case, conversation({2: [call("book_flight", id=1)]}, case=case))["action_after_cancellation"]
    (v,) = after.violations
    assert v.code == "ACTION_AFTER_CANCELLATION" and v.severity == "CRITICAL" and v.turn_index == 2
    assert v.evidence["cancellationEvent"]["type"] == "cancellation"
    resumed = run(case, conversation({4: [call("book_flight", id=1)]}, case=case))["action_after_cancellation"]
    assert resumed.passed


def test_repeated_mutation_uses_canonical_arguments_and_ignores_failed_attempts():
    case = approval_case()
    dup = run(case, conversation({2: [call("book_flight", a=1, b=2), call("book_flight", b=2, a=1)]},
                                 user_turns=3, case=case))["repeated_mutation"]
    (v,) = dup.violations
    assert v.code == "REPEATED_MUTATION" and v.severity == "CRITICAL" and v.evidence["firstCallId"] == "call-2-1"
    conv = conversation({2: [call("book_flight", a=1)]}, user_turns=3, case=case)
    conv.turns[3].tool_calls.insert(0, ToolCall("failed", "book_flight", {"a": 1}, status="ERROR", error="x", turn_index=2))
    assert run(case, conv)["repeated_mutation"].passed  # retry after a failure is legitimate
    different = run(case, conversation({2: [call("book_flight", a=1), call("book_flight", a=2)]},
                                       user_turns=3, case=case))["repeated_mutation"]
    assert different.passed


def test_max_turns_uses_stricter_of_suite_and_dataset_and_policy_severity():
    case = travel_case(maxTurns=3)
    conv = conversation({}, user_turns=4)
    res = run(case, conv, Limits(max_turns=10))["max_turns"]
    assert not res.passed and res.violations[0].severity == "HIGH" and res.observed == 4
    warn = run(case, conv, Limits(max_turns=10, limits_action="Warn"))["max_turns"]
    assert warn.violations[0].severity == "MEDIUM"
    assert run(travel_case(), conv, Limits(max_turns=4))["max_turns"].passed


def test_max_tool_calls():
    case = travel_case()
    conv = conversation({1: [call("a"), call("b")], 2: [call("c")]})
    assert not run(case, conv, Limits(max_tool_calls=2))["max_tool_calls"].passed
    assert run(case, conv, Limits(max_tool_calls=3))["max_tool_calls"].passed


def test_max_repeated_calls_is_order_insensitive_and_skips_mutations():
    case = travel_case()
    conv = conversation({1: [call("s", a=1, b=2)], 2: [call("s", b=2, a=1)], 3: [call("book_flight", x=1),
                                                                                  call("book_flight", x=1)]})
    res = run(case, conv, Limits(max_repeated_calls=1))["max_repeated_calls"]
    assert not res.passed and len(res.violations) == 1  # book_flight belongs to RepeatedMutationCheck
    assert res.violations[0].evidence["callIds"] == ["call-1-1", "call-2-1"]
    assert run(case, conv, Limits(max_repeated_calls=2))["max_repeated_calls"].passed


def test_final_state_partial_match():
    case = travel_case()
    ok = run(case, conversation({}, states={4: {"destination": "London", "extra": 1}}))["final_state"]
    assert ok.passed
    bad = run(case, conversation({}, states={4: {"destination": "Paris"}}))["final_state"]
    assert bad.violations[0].code == "FINAL_STATE_MISMATCH"


def test_event_derivation_uses_markers_and_regex_on_user_turns_only():
    case = travel_case(turns=[{"role": "user", "content": "Yes, book it!"},
                              {"role": "user", "content": "x", "markers": ["cancellation"]}],
                       expectedToolCalls=[], latestValues=[], expectedFinalState={})
    events = derive_events(case)
    assert [(e.type, e.turn_index, e.source) for e in events] == [
        ("explicit_user_confirmation", 1, "matcher"), ("cancellation", 2, "marker")]


def test_a_crashing_check_is_isolated_and_marked_as_error():
    class Boom(Check):
        name = "boom"

        def evaluate(self, ctx):
            raise RuntimeError("bug")

    case = travel_case()
    ctx = CheckContext(case=case, conversation=conversation({}))
    (res,) = run_checks(ctx, [Boom()])
    assert not res.passed and "bug" in res.error


def test_every_check_result_carries_evaluator_name_and_version():
    for r in run(travel_case(), conversation({})).values():
        assert r.evaluator == "trajectory-policy" and r.version
