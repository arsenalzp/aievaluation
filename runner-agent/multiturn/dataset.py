"""Scripted multi-turn dataset parsing and validation.

One JSONL line per case::

    {"caseId": "...", "type": "multiTurn",
     "turns": [{"role": "user", "content": "...", "markers": ["..."]}],
     "expectedBehavior": {...}}

Single-turn agent cases (``caseId`` + ``input``) are left to the existing parser in
runner-agent.py; :func:`classify_dataset` tells the two apart.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

CASE_TYPE_MULTI_TURN = "multiTurn"

DEFAULT_CANCELLATION_EVENTS = ["cancellation"]


class DatasetError(ValueError):
    """Malformed dataset record. Always names the case and the offending field."""

    def __init__(self, message: str, case_id: Optional[str] = None, field_path: Optional[str] = None,
                 line: Optional[int] = None, kind: str = "multi-turn"):
        self.case_id = case_id
        self.field_path = field_path
        self.line = line
        prefix = []
        if line is not None:
            prefix.append(f"line {line}")
        prefix.append(f"caseId={case_id!r}" if case_id else "caseId=<unknown>")
        if field_path:
            prefix.append(f"field={field_path}")
        super().__init__(f"invalid {kind} dataset record ({', '.join(prefix)}): {message}")
        self.message = message


@dataclass
class ScriptedTurn:
    index: int  # 1-based
    content: str
    markers: List[str] = field(default_factory=list)


@dataclass
class ConditionalTool:
    name: str
    allowed_after_event: str


@dataclass
class EventMatcher:
    event: str
    regex: str
    ignore_case: bool = True

    def search(self, text: str) -> Optional[re.Match]:
        flags = re.IGNORECASE if self.ignore_case else 0
        return re.search(self.regex, text or "", flags)


@dataclass
class LatestValue:
    field: str
    value: Any
    from_turn: int = 1
    tools: List[str] = field(default_factory=list)
    argument: str = ""  # argument name inside tool calls; defaults to ``field``


@dataclass
class ExpectedToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    turn: Optional[int] = None


@dataclass
class ExpectedBehavior:
    required_tools: List[str] = field(default_factory=list)
    forbidden_tools: List[str] = field(default_factory=list)
    expected_tool_calls: List[ExpectedToolCall] = field(default_factory=list)
    latest_values: List[LatestValue] = field(default_factory=list)
    mutation_tools: List[str] = field(default_factory=list)
    non_idempotent_tools: List[str] = field(default_factory=list)
    conditional_tools: List[ConditionalTool] = field(default_factory=list)
    cancellation_events: List[str] = field(default_factory=lambda: list(DEFAULT_CANCELLATION_EVENTS))
    resume_events: List[str] = field(default_factory=list)
    event_matchers: List[EventMatcher] = field(default_factory=list)
    expected_final_state: Dict[str, Any] = field(default_factory=dict)
    max_turns: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_repeated_calls: Optional[int] = None

    def mutating_tool_names(self) -> List[str]:
        """Every tool treated as side-effecting for the cancellation check."""
        names: List[str] = []
        for n in (*self.mutation_tools, *self.non_idempotent_tools, *(c.name for c in self.conditional_tools)):
            if n not in names:
                names.append(n)
        return names


@dataclass
class MultiTurnCase:
    case_id: str
    turns: List[ScriptedTurn]
    expected: ExpectedBehavior = field(default_factory=ExpectedBehavior)
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    line: Optional[int] = None


_TOP_LEVEL_KEYS = {"caseId", "type", "sessionId", "turns", "expectedBehavior", "tags", "metadata"}
_EXPECTED_KEYS = {
    "requiredTools", "forbiddenTools", "expectedToolCalls", "latestValues", "mutationTools",
    "nonIdempotentTools", "conditionalTools", "cancellationEvents", "resumeEvents", "eventMatchers",
    "expectedFinalState", "maxTurns", "maxToolCalls", "maxRepeatedCalls",
}


def is_multi_turn_record(raw: Dict[str, Any]) -> bool:
    """A record with ``type`` or ``turns`` is parsed as multi-turn; the parser then requires exactly
    ``"type": "multiTurn"`` (single-turn records have neither field)."""
    return "type" in raw or "turns" in raw


def classify_dataset(lines: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Return ``(multi_turn_case_ids, single_turn_case_ids)`` without full validation."""
    multi: List[str] = []
    single: List[str] = []
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON: {exc.msg} (column {exc.colno})", line=lineno) from exc
        if not isinstance(raw, dict):
            raise DatasetError("each line must be a JSON object", line=lineno)
        cid = str(raw.get("caseId") or f"<line {lineno}>")
        (multi if is_multi_turn_record(raw) else single).append(cid)
    return multi, single


def _fail(case_id: Optional[str], field_path: str, message: str, line: Optional[int]) -> DatasetError:
    return DatasetError(message, case_id=case_id, field_path=field_path, line=line)


def _str_list(raw: Any, case_id: str, path: str, line: Optional[int]) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _fail(case_id, path, "must be a list of strings", line)
    out: List[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise _fail(case_id, f"{path}[{i}]", "must be a non-empty string", line)
        out.append(item.strip())
    return out


def _positive_int(raw: Any, case_id: str, path: str, line: Optional[int]) -> Optional[int]:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise _fail(case_id, path, "must be an integer >= 1", line)
    return raw


def _parse_expected(raw: Any, case_id: str, turns: List[ScriptedTurn], line: Optional[int]) -> ExpectedBehavior:
    base = "expectedBehavior"
    if raw is None:
        return ExpectedBehavior()
    if not isinstance(raw, dict):
        raise _fail(case_id, base, "must be an object", line)
    unknown = sorted(set(raw) - _EXPECTED_KEYS)
    if unknown:
        raise _fail(case_id, f"{base}.{unknown[0]}",
                    f"unknown field(s) {unknown}; allowed: {sorted(_EXPECTED_KEYS)}", line)

    exp = ExpectedBehavior()
    exp.required_tools = _str_list(raw.get("requiredTools"), case_id, f"{base}.requiredTools", line)
    exp.forbidden_tools = _str_list(raw.get("forbiddenTools"), case_id, f"{base}.forbiddenTools", line)
    exp.mutation_tools = _str_list(raw.get("mutationTools"), case_id, f"{base}.mutationTools", line)
    exp.non_idempotent_tools = _str_list(raw.get("nonIdempotentTools"), case_id, f"{base}.nonIdempotentTools", line)
    if raw.get("cancellationEvents") is not None:
        exp.cancellation_events = _str_list(raw["cancellationEvents"], case_id, f"{base}.cancellationEvents", line)
    exp.resume_events = _str_list(raw.get("resumeEvents"), case_id, f"{base}.resumeEvents", line)
    exp.max_turns = _positive_int(raw.get("maxTurns"), case_id, f"{base}.maxTurns", line)
    exp.max_tool_calls = _positive_int(raw.get("maxToolCalls"), case_id, f"{base}.maxToolCalls", line)
    exp.max_repeated_calls = _positive_int(raw.get("maxRepeatedCalls"), case_id, f"{base}.maxRepeatedCalls", line)

    efs = raw.get("expectedFinalState")
    if efs is not None:
        if not isinstance(efs, dict):
            raise _fail(case_id, f"{base}.expectedFinalState", "must be an object", line)
        exp.expected_final_state = efs

    etc = raw.get("expectedToolCalls")
    if etc is not None:
        if not isinstance(etc, list):
            raise _fail(case_id, f"{base}.expectedToolCalls", "must be a list", line)
        for i, item in enumerate(etc):
            p = f"{base}.expectedToolCalls[{i}]"
            if not isinstance(item, dict):
                raise _fail(case_id, p, "must be an object", line)
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                raise _fail(case_id, f"{p}.name", "is required and must be a non-empty string", line)
            args = item.get("arguments", {})
            if not isinstance(args, dict):
                raise _fail(case_id, f"{p}.arguments", "must be an object", line)
            turn = item.get("turn")
            if turn is not None:
                turn = _positive_int(turn, case_id, f"{p}.turn", line)
                if turn > len(turns):
                    raise _fail(case_id, f"{p}.turn", f"refers to turn {turn} but the script has {len(turns)} turns", line)
            exp.expected_tool_calls.append(ExpectedToolCall(item["name"].strip(), args, turn))

    lv = raw.get("latestValues")
    if lv is not None:
        if not isinstance(lv, list):
            raise _fail(case_id, f"{base}.latestValues", "must be a list", line)
        for i, item in enumerate(lv):
            p = f"{base}.latestValues[{i}]"
            if not isinstance(item, dict):
                raise _fail(case_id, p, "must be an object", line)
            if not isinstance(item.get("field"), str) or not item["field"].strip():
                raise _fail(case_id, f"{p}.field", "is required and must be a non-empty string", line)
            if "value" not in item:
                raise _fail(case_id, f"{p}.value", "is required", line)
            from_turn = _positive_int(item.get("fromTurn"), case_id, f"{p}.fromTurn", line) or 1
            exp.latest_values.append(LatestValue(
                field=item["field"].strip(),
                value=item["value"],
                from_turn=from_turn,
                tools=_str_list(item.get("tools"), case_id, f"{p}.tools", line),
                argument=(item.get("argument") or item["field"]).strip(),
            ))

    ct = raw.get("conditionalTools")
    if ct is not None:
        if not isinstance(ct, list):
            raise _fail(case_id, f"{base}.conditionalTools", "must be a list", line)
        for i, item in enumerate(ct):
            p = f"{base}.conditionalTools[{i}]"
            if not isinstance(item, dict):
                raise _fail(case_id, p, "must be an object", line)
            if not isinstance(item.get("name"), str) or not item["name"].strip():
                raise _fail(case_id, f"{p}.name", "is required and must be a non-empty string", line)
            ev = item.get("allowedAfterEvent")
            if not isinstance(ev, str) or not ev.strip():
                raise _fail(case_id, f"{p}.allowedAfterEvent", "is required and must be a non-empty string", line)
            exp.conditional_tools.append(ConditionalTool(item["name"].strip(), ev.strip()))

    em = raw.get("eventMatchers")
    if em is not None:
        if not isinstance(em, list):
            raise _fail(case_id, f"{base}.eventMatchers", "must be a list", line)
        for i, item in enumerate(em):
            p = f"{base}.eventMatchers[{i}]"
            if not isinstance(item, dict):
                raise _fail(case_id, p, "must be an object", line)
            if not isinstance(item.get("event"), str) or not item["event"].strip():
                raise _fail(case_id, f"{p}.event", "is required and must be a non-empty string", line)
            rx = item.get("regex")
            if not isinstance(rx, str) or not rx:
                raise _fail(case_id, f"{p}.regex", "is required and must be a non-empty string", line)
            try:
                re.compile(rx)
            except re.error as exc:
                raise _fail(case_id, f"{p}.regex", f"is not a valid regular expression: {exc}", line) from exc
            exp.event_matchers.append(EventMatcher(item["event"].strip(), rx, bool(item.get("ignoreCase", True))))

    # Cross-reference: every event the policy depends on must be producible by the script.
    producible = {m for t in turns for m in t.markers} | {m.event for m in exp.event_matchers}
    for i, c in enumerate(exp.conditional_tools):
        if c.allowed_after_event not in producible:
            raise _fail(case_id, f"{base}.conditionalTools[{i}].allowedAfterEvent",
                        f"event {c.allowed_after_event!r} is never produced by a turn marker or eventMatcher "
                        f"(known events: {sorted(producible)})", line)
    # cancellationEvents may legitimately be absent from a script (nothing is cancelled), so they
    # are not cross-checked; resumeEvents only make sense if the script can actually produce them.
    for i, ev in enumerate(exp.resume_events):
        if ev not in producible:
            raise _fail(case_id, f"{base}.resumeEvents[{i}]",
                        f"event {ev!r} is never produced by a turn marker or eventMatcher", line)
    return exp


def parse_multiturn_record(raw: Dict[str, Any], line: Optional[int] = None) -> MultiTurnCase:
    cid_raw = raw.get("caseId")
    if not isinstance(cid_raw, str) or not cid_raw.strip():
        raise _fail(None, "caseId", "is required and must be a non-empty string", line)
    case_id = cid_raw.strip()

    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise _fail(case_id, unknown[0], f"unknown field(s) {unknown}; allowed: {sorted(_TOP_LEVEL_KEYS)}", line)

    t = raw.get("type")
    if t != CASE_TYPE_MULTI_TURN:  # exactly one spelling, required
        raise _fail(case_id, "type", f"is required and must be {CASE_TYPE_MULTI_TURN!r}, got {t!r}", line)

    session_id = raw.get("sessionId")
    if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
        raise _fail(case_id, "sessionId", "must be a non-empty string when set", line)

    turns_raw = raw.get("turns")
    if not isinstance(turns_raw, list) or not turns_raw:
        raise _fail(case_id, "turns", "is required and must be a non-empty list", line)
    turns: List[ScriptedTurn] = []
    for i, item in enumerate(turns_raw):
        p = f"turns[{i}]"
        if not isinstance(item, dict):
            raise _fail(case_id, p, "must be an object", line)
        role = item.get("role", "user")
        if role != "user":
            raise _fail(case_id, f"{p}.role",
                        f"got {role!r}; scripted cases contain user turns only - assistant replies are produced "
                        "by the target agent", line)
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise _fail(case_id, f"{p}.content", "is required and must be a non-empty string", line)
        unknown_t = sorted(set(item) - {"role", "content", "markers"})
        if unknown_t:
            raise _fail(case_id, f"{p}.{unknown_t[0]}", f"unknown field(s) {unknown_t}", line)
        markers = _str_list(item.get("markers"), case_id, f"{p}.markers", line)
        turns.append(ScriptedTurn(index=i + 1, content=content, markers=markers))

    tags = raw.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(x, str) for x in tags):
        raise _fail(case_id, "tags", "must be a list of strings", line)
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise _fail(case_id, "metadata", "must be an object", line)

    expected = _parse_expected(raw.get("expectedBehavior"), case_id, turns, line)
    return MultiTurnCase(case_id=case_id, turns=turns, expected=expected,
                         session_id=session_id.strip() if session_id else None,
                         tags=tags, metadata=metadata, line=line)


def parse_multiturn_lines(lines: Iterable[str]) -> List[MultiTurnCase]:
    cases: List[MultiTurnCase] = []
    seen: Dict[str, int] = {}
    for lineno, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"invalid JSON: {exc.msg} (column {exc.colno})", line=lineno) from exc
        if not isinstance(raw, dict):
            raise DatasetError("each line must be a JSON object", line=lineno)
        if not is_multi_turn_record(raw):
            raise DatasetError("not a multi-turn case (expected type 'multiTurn' and a 'turns' list); "
                               "single-turn and multi-turn cases cannot be mixed in one dataset",
                               case_id=str(raw.get("caseId") or "") or None, field_path="type", line=lineno)
        case = parse_multiturn_record(raw, line=lineno)
        if case.case_id in seen:
            raise DatasetError(f"duplicate caseId (first seen on line {seen[case.case_id]})",
                               case_id=case.case_id, field_path="caseId", line=lineno)
        seen[case.case_id] = lineno
        cases.append(case)
    return cases


def parse_multiturn_dataset(path: str) -> List[MultiTurnCase]:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_multiturn_lines(fh)
