"""Phoenix / OpenInference adapter.

All Phoenix- and OpenInference-specific parsing lives here. The rest of the runner only
sees canonical :class:`~multiturn.models.ToolCall` objects.

Trace source precedence (``evaluation.trace.source``):

``inline``   (default) the tool calls reported in the agent's own response are used.
``phoenix``  tool calls come from Phoenix spans only. If no span exists for the turn's
             trace the turn is reported as ``TRACE_MISSING``.
``auto``     Phoenix spans win when spans exist for the turn's trace, otherwise inline
             is used. When both exist and their tool calls differ, the preferred
             (Phoenix) observation is used and a ``TRACE_SOURCE_CONFLICT`` is reported
             with both observations as evidence. Observations are never merged.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .canonical import tool_call_key
from .models import ToolCall, normalize_status

SOURCE_INLINE = "inline"
SOURCE_PHOENIX = "phoenix"
SOURCE_AUTO = "auto"
TRACE_SOURCES = (SOURCE_INLINE, SOURCE_PHOENIX, SOURCE_AUTO)

SPAN_KIND_TOOL = "TOOL"

# OpenInference semantic-convention attribute names.
ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_SESSION_ID = "session.id"
ATTR_TOOL_NAME = "tool.name"
ATTR_TOOL_PARAMETERS = "tool.parameters"
ATTR_INPUT_VALUE = "input.value"
ATTR_OUTPUT_VALUE = "output.value"
ATTR_FN_NAME = "tool_call.function.name"
ATTR_FN_ARGS = "tool_call.function.arguments"
ATTR_EXC_MESSAGE = "exception.message"
# Optional EvalRun-specific hints a target application may attach to its spans.
ATTR_EVAL_TURN = "evalrun.turn"
ATTR_PARENT_TRACE = "evalrun.parent_trace_id"


@dataclass
class ObservedTurn:
    """Phoenix observations of one turn (= one trace)."""

    session_id: Optional[str]
    trace_id: str
    turn_index: Optional[int] = None
    parent_trace_id: Optional[str] = None
    start_time: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    span_count: int = 0


def _flat_attrs(attrs: Any) -> Dict[str, Any]:
    """Flatten nested attribute dicts into dotted keys; keep already-dotted keys as-is."""
    out: Dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict) and value and prefix != "":
            for k, v in value.items():
                walk(f"{prefix}.{k}", v)
        else:
            out[prefix] = value

    if isinstance(attrs, dict):
        for k, v in attrs.items():
            if isinstance(v, dict) and v:
                walk(str(k), v)
            else:
                out[str(k)] = v
    return out


def _attr_value(attrs: Dict[str, Any], key: str) -> Any:
    """Read ``key``; if a dict-valued attribute was flattened into ``key.<sub>`` entries, rebuild it."""
    if key in attrs:
        return attrs[key]
    prefix = key + "."
    rebuilt = {k[len(prefix):]: v for k, v in attrs.items() if k.startswith(prefix)}
    return rebuilt or None


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        s = value.strip()
        if s[:1] in "{[":
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return value
    return value


def _span_context(span: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    ctx = span.get("context") or {}
    return (ctx.get("trace_id") or span.get("trace_id"), ctx.get("span_id") or span.get("span_id") or span.get("id"))


def _duration_ms(span: Dict[str, Any]) -> Optional[float]:
    from datetime import datetime
    start, end = span.get("start_time"), span.get("end_time")
    if not start or not end:
        return None
    try:
        s = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        e = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
    except ValueError:
        return None
    return round((e - s).total_seconds() * 1000, 3)


def _tool_call_from_span(span: Dict[str, Any], attrs: Dict[str, Any], turn_index: Optional[int]) -> Optional[ToolCall]:
    kind = str(attrs.get(ATTR_SPAN_KIND) or span.get("span_kind") or "").upper()
    if kind != SPAN_KIND_TOOL:
        return None
    name = attrs.get(ATTR_TOOL_NAME) or attrs.get(ATTR_FN_NAME) or span.get("name")
    if not name:
        return None
    params = None
    for key in (ATTR_TOOL_PARAMETERS, ATTR_FN_ARGS, ATTR_INPUT_VALUE):
        params = _attr_value(attrs, key)
        if params is not None:
            break
    params = _maybe_json(params)
    if isinstance(params, dict) and isinstance(params.get("kwargs"), dict) and len(params) == 1:
        params = params["kwargs"]  # common LangChain-style envelope
    arguments = params if isinstance(params, dict) else ({"input": params} if params is not None else {})

    status_code = str(span.get("status_code") or "").upper()
    exception_message = attrs.get(ATTR_EXC_MESSAGE)
    if status_code == "ERROR" or exception_message:
        status = "ERROR"
        error = span.get("status_message") or exception_message or "span status ERROR"
    else:
        status = normalize_status(status_code or None, None)
        error = None
    _, span_id = _span_context(span)
    return ToolCall(
        call_id=str(span_id or f"span-{name}"),
        name=str(name).strip(),
        arguments=arguments,
        status=status,
        result=_maybe_json(_attr_value(attrs, ATTR_OUTPUT_VALUE)),
        error=str(error) if error else None,
        duration_ms=_duration_ms(span),
        turn_index=turn_index,
        timestamp=span.get("start_time"),
        source=SOURCE_PHOENIX,
    )


def normalize_openinference_spans(
    spans: Iterable[Dict[str, Any]],
    trace_to_turn: Optional[Dict[str, int]] = None,
) -> Dict[str, List[ObservedTurn]]:
    """Group OpenInference spans by session and order them into turns.

    ``trace_to_turn`` maps trace ids the runner got from the target's responses to turn
    numbers; it is authoritative. Without it, traces of a session are ordered by their
    earliest span start time (or an ``evalrun.turn`` attribute) and numbered from 1.

    Returns ``{session_id: [ObservedTurn, ...]}`` sorted by turn. Traces without any
    ``session.id`` are grouped under the empty-string session id.
    """
    by_trace: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for span in spans:
        if not isinstance(span, dict):
            continue
        trace_id, _ = _span_context(span)
        if not trace_id:
            continue
        by_trace.setdefault(str(trace_id), []).append((span, _flat_attrs(span.get("attributes"))))

    sessions: Dict[str, List[ObservedTurn]] = {}
    for trace_id, items in by_trace.items():
        session_id = next((str(a[ATTR_SESSION_ID]) for _, a in items if a.get(ATTR_SESSION_ID)), "")
        explicit_turn = next((a[ATTR_EVAL_TURN] for _, a in items if a.get(ATTR_EVAL_TURN) is not None), None)
        turn = (trace_to_turn or {}).get(trace_id)
        if turn is None and explicit_turn is not None:
            try:
                turn = int(explicit_turn)
            except (TypeError, ValueError):
                turn = None
        starts = sorted(str(s.get("start_time")) for s, _ in items if s.get("start_time"))
        observed = ObservedTurn(
            session_id=session_id,
            trace_id=trace_id,
            turn_index=turn,
            parent_trace_id=next((a[ATTR_PARENT_TRACE] for _, a in items if a.get(ATTR_PARENT_TRACE)), None),
            start_time=starts[0] if starts else None,
            span_count=len(items),
        )
        ordered = sorted(items, key=lambda it: str(it[0].get("start_time") or ""))
        for span, attrs in ordered:
            call = _tool_call_from_span(span, attrs, turn)
            if call:
                observed.tool_calls.append(call)
        sessions.setdefault(session_id, []).append(observed)

    for turns in sessions.values():
        turns.sort(key=lambda t: (t.start_time or "", t.trace_id))
        # number the traces that have no authoritative index, continuing after the explicit ones
        next_idx = 1
        used = {t.turn_index for t in turns if t.turn_index is not None}
        for t in turns:
            if t.turn_index is None:
                while next_idx in used:
                    next_idx += 1
                t.turn_index = next_idx
                used.add(next_idx)
            for c in t.tool_calls:
                c.turn_index = t.turn_index
        turns.sort(key=lambda t: (t.turn_index or 0, t.start_time or ""))
        for prev, cur in zip(turns, turns[1:]):
            if cur.parent_trace_id is None:
                cur.parent_trace_id = prev.trace_id
    return sessions


class PhoenixFetchError(RuntimeError):
    pass


class PhoenixRestFetcher:
    """Best-effort reader for ``GET {endpoint}/v1/projects/{project}/spans``.

    EXPERIMENTAL: written against Phoenix's documented REST shape (``{"data": [...],
    "next_cursor": ...}``) and only exercised with fake HTTP in the test-suite. No
    authentication is supported. Filtering by session happens client-side.
    """

    def __init__(self, endpoint: str, project: str = "default", timeout_seconds: float = 30,
                 get: Optional[Callable[..., Any]] = None, page_limit: int = 1000, max_pages: int = 50):
        if not endpoint:
            raise ValueError("Phoenix endpoint must not be empty")
        if get is None:
            import requests
            get = requests.get
        self.endpoint = endpoint.rstrip("/")
        self.project = project or "default"
        self.timeout = timeout_seconds
        self._get = get
        self.page_limit = page_limit
        self.max_pages = max_pages

    def fetch_spans(self, start_time: Optional[str] = None) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        cursor: Optional[str] = None
        for _ in range(self.max_pages):
            params: Dict[str, Any] = {"limit": self.page_limit}
            if cursor:
                params["cursor"] = cursor
            if start_time:
                params["start_time"] = start_time
            try:
                resp = self._get(f"{self.endpoint}/v1/projects/{self.project}/spans", params=params, timeout=self.timeout)
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                raise PhoenixFetchError(f"unable to read spans from Phoenix at {self.endpoint}: "
                                        f"{type(exc).__name__}: {exc}") from exc
            spans.extend(body.get("data") or [])
            cursor = body.get("next_cursor")
            if not cursor:
                return spans
        raise PhoenixFetchError(f"Phoenix returned more than {self.max_pages} pages of spans; narrow the project")


@dataclass
class TraceResolution:
    tool_calls: List[ToolCall]
    source: str  # which observation was used: "inline" | "phoenix" | "none"
    missing: bool = False  # preferred source had no observation for this turn
    conflict: Optional[Dict[str, Any]] = None  # evidence when inline and phoenix disagree


def _multiset(calls: List[ToolCall]) -> Counter:
    return Counter(tool_call_key(c.name, c.arguments) for c in calls)


def resolve_turn_tool_calls(
    source: str,
    inline_calls: Optional[List[ToolCall]],
    phoenix_turn: Optional[ObservedTurn],
) -> TraceResolution:
    """Apply the documented precedence rule for one turn. Never merges observations."""
    if source not in TRACE_SOURCES:
        raise ValueError(f"unknown trace source {source!r}; expected one of {TRACE_SOURCES}")

    if source == SOURCE_INLINE:
        if inline_calls is None:
            return TraceResolution([], "none", missing=True)
        return TraceResolution(list(inline_calls), SOURCE_INLINE)

    if source == SOURCE_PHOENIX:
        if phoenix_turn is None:
            return TraceResolution([], "none", missing=True)
        return TraceResolution(list(phoenix_turn.tool_calls), SOURCE_PHOENIX)

    # auto
    if phoenix_turn is None:
        if inline_calls is None:
            return TraceResolution([], "none", missing=True)
        return TraceResolution(list(inline_calls), SOURCE_INLINE)
    conflict = None
    if inline_calls is not None and _multiset(inline_calls) != _multiset(phoenix_turn.tool_calls):
        conflict = {
            "preferred": SOURCE_PHOENIX,
            "phoenix": [c.to_dict() for c in phoenix_turn.tool_calls],
            "inline": [c.to_dict() for c in inline_calls],
        }
    return TraceResolution(list(phoenix_turn.tool_calls), SOURCE_PHOENIX, conflict=conflict)
