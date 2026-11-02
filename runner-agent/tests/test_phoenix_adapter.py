import pytest

from multiturn.models import ToolCall
from multiturn.phoenix_adapter import (
    ObservedTurn, PhoenixFetchError, PhoenixRestFetcher, normalize_openinference_spans, resolve_turn_tool_calls,
)


def span(trace, span_id, kind, start, session=None, name="x", attrs=None, status="OK", parent=None, end=None, msg=None):
    a = {"openinference.span.kind": kind}
    if session:
        a["session.id"] = session
    a.update(attrs or {})
    return {"name": name, "context": {"trace_id": trace, "span_id": span_id}, "parent_id": parent,
            "start_time": start, "end_time": end or start, "status_code": status, "status_message": msg,
            "attributes": a}


SPANS = [
    # session s1, turn 2 (later) - given out of order on purpose
    span("t2", "a1", "TOOL", "2026-01-01T10:00:10+00:00", "s1", "search_flights",
         {"tool.name": "search_flights", "tool.parameters": '{"destination": "London"}', "output.value": '{"n": 3}'},
         end="2026-01-01T10:00:10.250+00:00"),
    span("t2", "a0", "AGENT", "2026-01-01T10:00:09+00:00", "s1"),
    # session s1, turn 1
    span("t1", "b1", "TOOL", "2026-01-01T10:00:01+00:00", "s1", "search_flights",
         {"tool.name": "search_flights", "input.value": {"destination": "Paris"}}),
    span("t1", "b0", "LLM", "2026-01-01T10:00:00+00:00", "s1"),
    # another session, failed tool, nested attribute style
    span("t9", "c1", "TOOL", "2026-01-01T10:05:00+00:00", None, "book_flight",
         {"tool": {"name": "book_flight"}, "session": {"id": "s2"}}, status="ERROR", msg="card declined"),
]


def test_groups_by_session_and_orders_turns():
    sessions = normalize_openinference_spans(SPANS)
    assert set(sessions) == {"s1", "s2"}
    assert [t.trace_id for t in sessions["s1"]] == ["t1", "t2"]
    assert [t.turn_index for t in sessions["s1"]] == [1, 2]
    assert sessions["s1"][1].parent_trace_id == "t1"  # derived from the previous turn
    assert sessions["s1"][0].span_count == 2


def test_tool_spans_are_normalised_into_canonical_tool_calls():
    t2 = normalize_openinference_spans(SPANS)["s1"][1]
    (call,) = t2.tool_calls
    assert isinstance(call, ToolCall)
    assert (call.name, call.arguments, call.status) == ("search_flights", {"destination": "London"}, "OK")
    assert call.result == {"n": 3}
    assert call.duration_ms == 250.0
    assert call.call_id == "a1" and call.turn_index == 2 and call.source == "phoenix"
    # LLM / AGENT spans are not tool calls
    assert normalize_openinference_spans(SPANS)["s1"][0].tool_calls[0].arguments == {"destination": "Paris"}


def test_error_status_and_nested_attributes():
    (t,) = normalize_openinference_spans(SPANS)["s2"]
    (call,) = t.tool_calls
    assert call.status == "ERROR" and call.error == "card declined" and not call.succeeded


def test_trace_to_turn_mapping_from_target_responses_is_authoritative():
    sessions = normalize_openinference_spans(SPANS, trace_to_turn={"t1": 3, "t2": 4})
    assert [t.turn_index for t in sessions["s1"]] == [3, 4]
    assert sessions["s1"][0].tool_calls[0].turn_index == 3


def _obs(*calls):
    return ObservedTurn("s", "t", 1, tool_calls=list(calls))


def call(name, **args):
    return ToolCall("c", name, args)


def test_precedence_inline_default_ignores_phoenix():
    r = resolve_turn_tool_calls("inline", [call("a")], _obs(call("b")))
    assert r.source == "inline" and [c.name for c in r.tool_calls] == ["a"] and not r.conflict


def test_precedence_phoenix_only_marks_missing_turns():
    assert resolve_turn_tool_calls("phoenix", [call("a")], None).missing
    r = resolve_turn_tool_calls("phoenix", [call("a")], _obs(call("b")))
    assert r.source == "phoenix" and [c.name for c in r.tool_calls] == ["b"]


def test_precedence_auto_prefers_phoenix_and_reports_conflict_without_merging():
    r = resolve_turn_tool_calls("auto", [call("a", x=1)], _obs(call("b")))
    assert r.source == "phoenix" and [c.name for c in r.tool_calls] == ["b"]
    assert r.conflict["preferred"] == "phoenix"
    assert r.conflict["inline"][0]["name"] == "a" and r.conflict["phoenix"][0]["name"] == "b"


def test_precedence_auto_agreeing_sources_have_no_conflict_even_if_arg_order_differs():
    r = resolve_turn_tool_calls("auto", [call("a", x=1, y=2)], _obs(call("a", y=2, x=1)))
    assert r.conflict is None


def test_precedence_auto_falls_back_to_inline_when_phoenix_has_nothing():
    r = resolve_turn_tool_calls("auto", [call("a")], None)
    assert r.source == "inline" and not r.missing
    assert resolve_turn_tool_calls("auto", None, None).missing


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        resolve_turn_tool_calls("bogus", [], None)


class _Resp:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.body


def test_rest_fetcher_follows_cursors_and_never_needs_network():
    pages = {None: {"data": [1, 2], "next_cursor": "c1"}, "c1": {"data": [3], "next_cursor": None}}
    seen = []

    def fake_get(url, params, timeout):
        seen.append((url, dict(params)))
        return _Resp(pages[params.get("cursor")])

    fetcher = PhoenixRestFetcher("http://phoenix:6006/", "evals", get=fake_get)
    assert fetcher.fetch_spans(start_time="2026-01-01T00:00:00Z") == [1, 2, 3]
    assert seen[0][0] == "http://phoenix:6006/v1/projects/evals/spans"
    assert seen[0][1]["start_time"] == "2026-01-01T00:00:00Z"


def test_rest_fetcher_wraps_errors():
    def boom(url, params, timeout):
        return _Resp({}, status=503)

    with pytest.raises(PhoenixFetchError, match="unable to read spans"):
        PhoenixRestFetcher("http://p", get=boom).fetch_spans()
