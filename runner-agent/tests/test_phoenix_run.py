"""Whole-run behaviour with Phoenix/OpenInference as the trajectory source (fake spans, no network)."""
from fake_travel_agent import FakeTravelAgent
from multiturn.config import parse_multiturn_settings
from multiturn.phoenix_adapter import PhoenixFetchError
from multiturn.runner import evaluate_run
from multiturn.target import StatelessSessionAdapter

from conftest import suite_dict, travel_case

SESSION = "evalrun-run1-travel-001"


def tool_span(turn, name, args, span_id=None, session=SESSION):
    trace = f"trace-{SESSION}-{turn:03d}"
    return {"name": name, "context": {"trace_id": trace, "span_id": span_id or f"{trace}-{name}"},
            "start_time": f"2026-01-01T10:00:{turn:02d}+00:00", "end_time": f"2026-01-01T10:00:{turn:02d}.100+00:00",
            "status_code": "OK",
            "attributes": {"openinference.span.kind": "TOOL", "session.id": session, "tool.name": name,
                           "tool.parameters": args}}


def root_span(turn):
    trace = f"trace-{SESSION}-{turn:03d}"
    return {"name": "agent", "context": {"trace_id": trace, "span_id": trace + "-root"},
            "start_time": f"2026-01-01T10:00:{turn:02d}+00:00", "status_code": "OK",
            "attributes": {"openinference.span.kind": "AGENT", "session.id": SESSION}}


class FakeFetcher:
    def __init__(self, spans=None, error=None):
        self.spans, self.error, self.calls = spans or [], error, 0

    def fetch_spans(self, start_time=None):
        self.calls += 1
        if self.error:
            raise PhoenixFetchError(self.error)
        return self.spans


GOOD_SPANS = [root_span(1), root_span(4),
              tool_span(2, "search_flights", '{"origin": "NYC", "destination": "Paris", "date": "June 5"}'),
              tool_span(3, "search_flights", '{"origin": "NYC", "destination": "London", "date": "June 5"}')]


def run(source, fetcher, agent=None, uploader=None, layout=None):
    settings = parse_multiturn_settings(suite_dict(**{
        "evaluation.trace": {"source": source, "phoenix": {"endpoint": "http://phoenix:6006", "project": "travel"}}}))
    agent = agent or FakeTravelAgent()
    return evaluate_run(settings=settings, cases=[travel_case()],
                        adapter=StatelessSessionAdapter("http://agent", post=agent.post), uploader=uploader,
                        layout=layout, run_info={"evalrun": "run1", "evalsuite": "s"}, phoenix_fetcher=fetcher,
                        sleep=lambda s: None, trace_retries=2)


def codes(result):
    return sorted({v.code for c in result.cases for v in c.violations()})


def test_phoenix_source_evaluates_the_trajectory_from_spans(uploader, layout):
    # A buggy agent that *reports* a forbidden call inline but whose real (Phoenix) trajectory is clean.
    agent = FakeTravelAgent(["forbidden_tool"])
    result = run("phoenix", FakeFetcher(GOOD_SPANS), agent, uploader, layout)
    assert result.verdict == "PASS", result.summary
    trial = result.cases[0].first_trial()
    assert {c.source for c in trial.conversation.tool_calls()} == {"phoenix"}
    assert [c.turn_index for c in trial.conversation.tool_calls()] == [2, 3]
    assert trial.conversation.turns[3].metadata["parentTraceId"]  # parent trace is preserved
    art = uploader.objects["runs/r1/conversations/travel-001.json"]
    assert art["trajectory"]["toolCalls"][0]["source"] == "phoenix"


def test_phoenix_spans_expose_violations_the_agent_did_not_report(uploader, layout):
    spans = GOOD_SPANS + [tool_span(2, "book_flight", '{"origin": "NYC", "destination": "Paris"}')]
    result = run("phoenix", FakeFetcher(spans), FakeTravelAgent(), uploader, layout)
    assert result.verdict == "FAIL" and "ACTION_WITHOUT_APPROVAL" in codes(result)


def test_phoenix_source_with_no_spans_reports_trace_missing(uploader, layout):
    result = run("phoenix", FakeFetcher([]), uploader=uploader, layout=layout)
    assert "TRACE_MISSING" in codes(result) and result.verdict == "FAIL"


def test_auto_source_prefers_phoenix_and_reports_conflicts_without_merging(uploader, layout):
    spans = [root_span(1), tool_span(2, "search_flights", '{"origin": "NYC", "destination": "Rome", "date": "June 5"}'),
             tool_span(3, "search_flights", '{"origin": "NYC", "destination": "London", "date": "June 5"}')]
    result = run("auto", FakeFetcher(spans), uploader=uploader, layout=layout)
    conflicts = [v for c in result.cases for v in c.violations() if v.code == "TRACE_SOURCE_CONFLICT"]
    assert len(conflicts) == 1 and conflicts[0].turn_index == 2 and conflicts[0].severity == "MEDIUM"
    assert conflicts[0].evidence["inline"][0]["arguments"]["destination"] == "Paris"
    assert conflicts[0].evidence["phoenix"][0]["arguments"]["destination"] == "Rome"
    used = result.cases[0].first_trial().conversation.turns[3].tool_calls  # assistant turn 2
    assert [c.arguments["destination"] for c in used] == ["Rome"]  # one source wins; nothing merged


def test_auto_source_falls_back_to_inline_when_phoenix_is_down_and_warns(uploader, layout):
    result = run("auto", FakeFetcher(error="connection refused"), uploader=uploader, layout=layout)
    assert result.verdict == "WARN"
    assert result.root_cause["stage"] == "Evaluator" and result.root_cause["evaluator"] == "phoenix"
    assert {c.source for c in result.cases[0].first_trial().conversation.tool_calls()} == {"inline"}


def test_phoenix_source_with_unreachable_phoenix_fails(uploader, layout):
    result = run("phoenix", FakeFetcher(error="connection refused"), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL" and result.root_cause["stage"] == "Evaluator"
    assert "connection refused" in result.root_cause["details"]


def test_spans_are_fetched_once_per_run_not_per_turn(uploader, layout):
    fetcher = FakeFetcher(GOOD_SPANS)
    run("phoenix", fetcher, uploader=uploader, layout=layout)
    assert fetcher.calls == 1
