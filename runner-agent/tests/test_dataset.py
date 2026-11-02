import json

import pytest

from multiturn.dataset import DatasetError, classify_dataset, parse_multiturn_lines


def line(**over):
    base = {"caseId": "c1", "type": "multiTurn", "turns": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return json.dumps(base)


def test_parses_full_case():
    case = parse_multiturn_lines([json.dumps({
        "caseId": "travel-1", "type": "multiTurn", "sessionId": "s-1",
        "turns": [{"role": "user", "content": "book it", "markers": ["explicit_user_confirmation"]}],
        "expectedBehavior": {
            "requiredTools": ["search"], "conditionalTools": [
                {"name": "book_flight", "allowedAfterEvent": "explicit_user_confirmation"}],
            "latestValues": [{"field": "destination", "value": "London", "fromTurn": 2}],
            "maxTurns": 5}})])[0]
    assert case.session_id == "s-1"
    assert case.turns[0].markers == ["explicit_user_confirmation"]
    assert case.expected.conditional_tools[0].allowed_after_event == "explicit_user_confirmation"
    assert case.expected.latest_values[0].argument == "destination"  # defaults to the field name
    assert case.expected.cancellation_events == ["cancellation"]
    assert case.expected.max_turns == 5


@pytest.mark.parametrize("record, field, fragment", [
    ({"turns": [{"content": "x"}]}, "caseId", "required"),
    ({"turns": []}, "turns", "non-empty list"),
    ({"turns": [{"role": "assistant", "content": "x"}]}, "turns[0].role", "user turns only"),
    ({"turns": [{"role": "user"}]}, "turns[0].content", "required"),
    ({"turns": [{"role": "user", "content": "x", "markers": "oops"}]}, "turns[0].markers", "list of strings"),
    ({"expectedBehavior": {"maxTurns": 0}}, "expectedBehavior.maxTurns", ">= 1"),
    ({"expectedBehavior": {"reqiredTools": ["x"]}}, "expectedBehavior.reqiredTools", "unknown"),
    ({"expectedBehavior": {"conditionalTools": [{"name": "book"}]}},
     "expectedBehavior.conditionalTools[0].allowedAfterEvent", "required"),
    ({"expectedBehavior": {"conditionalTools": [{"name": "book", "allowedAfterEvent": "never_happens"}]}},
     "expectedBehavior.conditionalTools[0].allowedAfterEvent", "never produced"),
    ({"expectedBehavior": {"eventMatchers": [{"event": "e", "regex": "("}]}},
     "expectedBehavior.eventMatchers[0].regex", "regular expression"),
    ({"expectedBehavior": {"expectedToolCalls": [{"name": "x", "turn": 9}]}},
     "expectedBehavior.expectedToolCalls[0].turn", "1 turns"),
    ({"expectedBehavior": {"latestValues": [{"field": "d"}]}}, "expectedBehavior.latestValues[0].value", "required"),
    ({"bogus": 1}, "bogus", "unknown"),
])
def test_malformed_cases_name_case_and_field(record, field, fragment):
    raw = {"caseId": "bad-case", "type": "multiTurn", "turns": [{"role": "user", "content": "hi"}]}
    raw.update(record)
    if "caseId" not in record and "turns" in record and field == "caseId":
        raw.pop("caseId")
    with pytest.raises(DatasetError) as err:
        parse_multiturn_lines([json.dumps(raw)])
    msg = str(err.value)
    assert field in msg and fragment in msg
    assert err.value.field_path == field
    assert ("bad-case" in msg) or field == "caseId"
    assert "line 1" in msg


def test_invalid_json_reports_line_number():
    with pytest.raises(DatasetError, match="line 2"):
        parse_multiturn_lines([line(), "{not json"])


def test_duplicate_case_ids_rejected():
    with pytest.raises(DatasetError, match="duplicate caseId"):
        parse_multiturn_lines([line(), line()])


def test_single_turn_case_in_multiturn_dataset_is_rejected_with_case_id():
    with pytest.raises(DatasetError) as err:
        parse_multiturn_lines([json.dumps({"caseId": "old-style", "input": "hello"})])
    assert "old-style" in str(err.value) and "cannot be mixed" in str(err.value)


def test_classify_dataset_keeps_single_turn_format_untouched():
    multi, single = classify_dataset([
        json.dumps({"caseId": "a", "input": "q", "expectedTools": [{"name": "t"}]}),
        line(caseId="b"),
        "",
    ])
    assert multi == ["b"] and single == ["a"]
