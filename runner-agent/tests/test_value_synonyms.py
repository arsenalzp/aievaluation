"""Every value has exactly one spelling (user decision 2026-11): synonyms are errors, never mapped silently."""
import json

import pytest

from multiturn.dataset import DatasetError, parse_multiturn_lines
from multiturn.phoenix_adapter import PhoenixSpanError, normalize_openinference_spans
from multiturn.target import StatelessSessionAdapter, TargetError, TurnRequest

from test_phoenix_adapter import span


def mt_case(**over):
    rec = {"caseId": "c-1", "type": "multiTurn", "turns": [{"role": "user", "content": "hi"}]}
    rec.update(over)
    return json.dumps({k: v for k, v in rec.items() if v is not None})


@pytest.mark.parametrize("value", ["multiturn", "multi_turn", "multi-turn", "MultiTurn", " multiTurn", None])
def test_multi_turn_type_has_exactly_one_spelling(value):
    with pytest.raises(DatasetError, match=r"field=type\): is required and must be 'multiTurn'"):
        parse_multiturn_lines([mt_case(type=value)])


def test_canonical_multi_turn_type_is_accepted():
    (case,) = parse_multiturn_lines([mt_case()])
    assert case.case_id == "c-1"


class _Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def send(body):
    adapter = StatelessSessionAdapter("http://agent", post=lambda *a, **k: _Resp(body))
    return adapter.send_turn(TurnRequest(session_id="s", input="hi", turn=1), case_id="c-1")


CALL = {"name": "search", "arguments": {}, "status": "OK"}


@pytest.mark.parametrize("body, fragment", [
    ({"answer": "x"}, "answer is not supported; use output"),
    ({"output": "x", "toolCalls": [CALL]}, "toolCalls is not supported; use trace.toolCalls"),
    ({"output": "x", "trace": {"tool_calls": [CALL]}}, "trace.tool_calls is not supported"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "tool": "s"}]}}, r"toolCalls\[0\].tool is not supported; use name"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "args": {}}]}}, "args is not supported; use arguments"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "id": "1"}]}}, "id is not supported; use callId"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "result": 1}]}}, "result is not supported; use output"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "duration_ms": 1}]}}, "duration_ms is not supported"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "status": "SUCCESS"}]}}, r"status must be one of \['OK', 'ERROR'\]"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "status": "ok"}]}}, "status must be one of"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "durationMs": "1"}]}}, "durationMs must be a number"),
    ({"output": "x", "trace": {"toolCalls": {"name": "s"}}}, "trace.toolCalls must be a list"),
    ({"output": "x", "trace": []}, "response field 'trace' must be an object"),
    ({"output": "x", "state": []}, "response field 'state' must be an object"),
    ({"output": "x", "metadata": []}, "response field 'metadata' must be an object"),
])
def test_multi_turn_response_synonyms_are_rejected(body, fragment):
    with pytest.raises(TargetError, match=fragment):
        send(body)


def test_multi_turn_response_canonical_fields_are_read():
    resp = send({"output": "x", "trace": {"toolCalls": [{**CALL, "callId": "c9", "output": {"n": 1}, "durationMs": 3}]}})
    (call,) = resp.tool_calls
    assert (call.call_id, call.result, call.duration_ms, call.status) == ("c9", {"n": 1}, 3, "OK")


def one_tool_span(status):
    return span("t1", "s1", "TOOL", "2026-01-01T10:00:00+00:00", session="sess", name="search",
                attrs={"tool.name": "search"}, status=status)


@pytest.mark.parametrize("status, expected", [("OK", "OK"), ("UNSET", "OK"), ("ERROR", "ERROR")])
def test_phoenix_accepts_exactly_the_opentelemetry_status_codes(status, expected):
    turns = normalize_openinference_spans([one_tool_span(status)], {"t1": 1})["sess"]
    assert turns[0].tool_calls[0].status == expected


@pytest.mark.parametrize("status", ["ok", "SUCCESS", "STATUS_CODE_OK", "Error"])
def test_phoenix_rejects_other_status_codes(status):
    with pytest.raises(PhoenixSpanError, match="status_code must be one of"):
        normalize_openinference_spans([one_tool_span(status)], {"t1": 1})
