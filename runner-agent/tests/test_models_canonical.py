import json

import pytest

from multiturn.canonical import canonical_json, partial_match, tool_call_key, values_equal
from multiturn.models import Conversation, ToolCall, Turn, parse_tool_status


def test_canonical_json_ignores_key_order_and_numeric_representation():
    a = {"origin": "NYC", "dest": {"city": "London", "terminal": 2.0}, "tags": ["a", "b"]}
    b = {"tags": ["a", "b"], "dest": {"terminal": 2, "city": "London"}, "origin": "NYC"}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json({"a": 1}) != canonical_json({"a": 2})


def test_canonical_json_treats_null_and_missing_alike_and_strips_strings():
    assert canonical_json({"a": 1, "b": None}) == canonical_json({"a": 1})
    assert canonical_json({"city": " London "}) == canonical_json({"city": "London"})
    assert canonical_json({"flag": True}) != canonical_json({"flag": 1})  # bool is not int


def test_tool_call_key_is_stable():
    k1 = tool_call_key(" book_flight ", {"b": 1, "a": 2})
    k2 = tool_call_key("book_flight", {"a": 2, "b": 1})
    assert k1 == k2
    assert tool_call_key("search", {}) != tool_call_key("book", {})


def test_values_equal_is_case_insensitive_for_strings_only():
    assert values_equal("London", "london")
    assert not values_equal("London", "Paris")
    assert values_equal(3, 3.0)
    assert not values_equal(["A"], ["a", "b"])


def test_partial_match_allows_extra_observed_keys():
    ok, mism = partial_match({"destination": "London"}, {"destination": "london", "date": "June 5"})
    assert ok and mism == []


def test_partial_match_reports_path_of_each_mismatch():
    ok, mism = partial_match({"a": {"b": 1}, "c": 2}, {"a": {"b": 9}})
    assert not ok
    assert {m["path"] for m in mism} == {"$.a.b", "$.c"}
    assert next(m for m in mism if m["path"] == "$.c")["missing"] is True


def test_partial_match_lists_must_match_exactly():
    assert partial_match({"x": [1, 2]}, {"x": [1, 2]})[0]
    assert not partial_match({"x": [1, 2]}, {"x": [1, 2, 3]})[0]


def test_parse_tool_status_accepts_exactly_ok_and_error():
    assert parse_tool_status("OK") == "OK" and parse_tool_status("ERROR") == "ERROR"
    assert parse_tool_status(None) == "OK"
    assert parse_tool_status(None, error="boom") == "ERROR"
    for synonym in ("success", "SUCCESS", "SUCCEEDED", "ok", "Ok", "UNSET", "timeout", "", 200):
        with pytest.raises(ValueError, match="status must be one of"):
            parse_tool_status(synonym)


def test_conversation_orders_tool_calls_by_turn_and_serialises_to_json():
    conv = Conversation(case_id="c", session_id="s", turns=[
        Turn(2, "assistant", "b", tool_calls=[ToolCall("2", "second", turn_index=2)]),
        Turn(1, "assistant", "a", tool_calls=[ToolCall("1", "first", turn_index=1)]),
        Turn(1, "user", "hello"),
    ])
    assert [c.name for c in conv.tool_calls()] == ["first", "second"]
    assert len(conv.user_turns()) == 1
    json.dumps(conv.to_dict())  # canonical model is plain JSON data
    assert conv.to_dict()["turns"][0]["toolCalls"][0]["callId"] == "2"


def test_deterministic_core_imports_no_framework_libraries():
    """Importing the whole runner core must not pull in Phoenix, OpenInference or DeepEval."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(__file__))
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import multiturn.models, multiturn.canonical, multiturn.dataset, multiturn.checks, multiturn.gate,"
        "multiturn.runner, multiturn.artifacts, multiturn.phoenix_adapter, multiturn.entrypoint;"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('phoenix', 'deepeval', 'openinference', 'requests')];"
        "assert not bad, bad"
    ) % root
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)
