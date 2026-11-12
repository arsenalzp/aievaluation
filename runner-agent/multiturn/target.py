"""Session-aware target endpoint access.

The runner only forwards ``sessionId`` and the scripted user input. It never stores,
replays or repairs the business conversation state - the tested agent owns its memory,
which is exactly what the multi-turn checks evaluate.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .models import ToolCall, parse_tool_status


class TargetError(RuntimeError):
    def __init__(self, message: str, case_id: Optional[str] = None, turn: Optional[int] = None):
        self.case_id = case_id
        self.turn = turn
        super().__init__(f"target call failed (caseId={case_id!r}, turn={turn}): {message}")
        self.message = message


@dataclass
class TurnRequest:
    session_id: str
    input: str
    turn: int
    evaluation_mode: bool = True

    def to_payload(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "input": self.input,
            "turn": self.turn,
            "evaluationMode": self.evaluation_mode,
        }


@dataclass
class TurnResponse:
    session_id: Optional[str]
    output: str
    state: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: List[ToolCall] = field(default_factory=list)  # inline observations, canonical
    has_inline_trace: bool = False

    @property
    def trace_id(self) -> Optional[str]:
        return self.trace.get("traceId") or self.metadata.get("traceId")

    @property
    def parent_trace_id(self) -> Optional[str]:
        return self.trace.get("parentTraceId")


class TargetAdapter(ABC):
    """How the runner talks to the tested agent. One instance serves all sessions of a run."""

    name: str = "abstract"

    @abstractmethod
    def send_turn(self, request: TurnRequest, case_id: Optional[str] = None) -> TurnResponse:
        """Send one user turn and return the agent's answer plus its inline observations."""

    def close_session(self, session_id: str) -> None:  # pragma: no cover - optional hook
        """Hook for adapters with explicit session lifecycles (no-op for stateless POST)."""


# Every value has exactly one field. These undocumented spellings are rejected by name instead of being
# dropped (a dropped ``tool_calls`` would read as "no tool was called").
RESPONSE_RENAMED_FIELDS = {"answer": "output", "toolCalls": "trace.toolCalls", "toolsCalled": "trace.toolCalls"}
TRACE_RENAMED_FIELDS = {"tool_calls": "trace.toolCalls"}
TOOL_CALL_RENAMED_FIELDS = {"tool": "name", "args": "arguments", "id": "callId", "result": "output",
                            "duration_ms": "durationMs"}


def reject_renamed(d: Dict[str, Any], renamed: Dict[str, str], path: str) -> None:
    for old, new in renamed.items():
        if old in d:
            raise ValueError(f"{path}{old} is not supported; use {new}")


def parse_inline_tool_calls(trace: Dict[str, Any], turn_index: int) -> List[ToolCall]:
    """Normalise ``trace.toolCalls`` of a turn response into canonical ToolCalls."""
    if not isinstance(trace, dict):
        return []
    reject_renamed(trace, TRACE_RENAMED_FIELDS, "trace.")
    raw_calls = trace.get("toolCalls")
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, list):
        raise ValueError("trace.toolCalls must be a list")
    calls: List[ToolCall] = []
    for i, raw in enumerate(raw_calls):
        p = f"trace.toolCalls[{i}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{p} must be an object")
        reject_renamed(raw, TOOL_CALL_RENAMED_FIELDS, f"{p}.")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{p}.name is required and must be a non-empty string")
        args = raw.get("arguments", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise ValueError(f"{p}.arguments must be an object")
        duration = raw.get("durationMs")
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))):
            raise ValueError(f"{p}.durationMs must be a number, got {duration!r}")
        call_id = raw.get("callId")
        if call_id is not None and (not isinstance(call_id, str) or not call_id.strip()):
            raise ValueError(f"{p}.callId must be a non-empty string when set")
        error = raw.get("error")
        try:
            status = parse_tool_status(raw.get("status"), error)
        except ValueError as exc:
            raise ValueError(f"{p}.{exc}") from None
        calls.append(ToolCall(
            call_id=call_id or f"turn{turn_index}-call{i + 1}",
            name=name.strip(),
            arguments=args,
            status=status,
            result=raw.get("output"),
            error=str(error) if error else None,
            duration_ms=duration,
            turn_index=turn_index,
            timestamp=raw.get("timestamp"),
            source="inline",
        ))
    return calls


class StatelessSessionAdapter(TargetAdapter):
    """``POST <endpoint>`` with ``{sessionId, input, turn, evaluationMode}`` per turn.

    This is the currently supported endpoint mode. ``post`` is injectable for tests.
    """

    name = "stateless"

    def __init__(self, endpoint: str, timeout_seconds: float = 60, post: Optional[Callable[..., Any]] = None,
                 headers: Optional[Dict[str, str]] = None):
        if not endpoint:
            raise ValueError("target endpoint must not be empty")
        if post is None:
            import requests  # local import keeps the deterministic core importable without requests
            post = requests.post
        self.endpoint = endpoint
        self.timeout = timeout_seconds
        self._post = post
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def send_turn(self, request: TurnRequest, case_id: Optional[str] = None) -> TurnResponse:
        started = time.monotonic()
        try:
            resp = self._post(self.endpoint, json=request.to_payload(), headers=self.headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            raise TargetError(f"{type(exc).__name__}: {exc}", case_id, request.turn) from exc
        elapsed_ms = int((time.monotonic() - started) * 1000)

        if not isinstance(data, dict):
            raise TargetError("response body must be a JSON object", case_id, request.turn)
        try:
            reject_renamed(data, RESPONSE_RENAMED_FIELDS, "")
        except ValueError as exc:
            raise TargetError(str(exc), case_id, request.turn) from exc
        output = data.get("output")
        if not isinstance(output, str):
            raise TargetError("response is missing the string field 'output'", case_id, request.turn)

        # Absent (None) -> empty; any other non-object is a contract error (``[] or {}`` would hide it).
        objects = {}
        for key in ("state", "trace", "metadata"):
            value = data.get(key)
            if value is None:
                value = {}
            if not isinstance(value, dict):
                raise TargetError(f"response field '{key}' must be an object", case_id, request.turn)
            objects[key] = value
        state, trace = objects["state"], objects["trace"]
        metadata = dict(objects["metadata"])
        metadata.setdefault("latencyMs", elapsed_ms)

        try:
            calls = parse_inline_tool_calls(trace, request.turn)
        except ValueError as exc:
            raise TargetError(str(exc), case_id, request.turn) from exc

        return TurnResponse(
            session_id=data.get("sessionId"),
            output=output,
            state=state,
            trace=trace,
            metadata=metadata,
            tool_calls=calls,
            has_inline_trace=bool(trace),
        )


# ---- future adapters --------------------------------------------------------
# ``POST /sessions/{id}/turns`` and "external trace retrieval by traceId" are reserved
# adapter modes. They are registered so a suite referencing them fails with a clear
# message instead of silently falling back to the stateless mode.

_ADAPTERS: Dict[str, Callable[..., TargetAdapter]] = {}


def register_target_adapter(mode: str, factory: Callable[..., TargetAdapter]) -> None:
    _ADAPTERS[mode] = factory


def create_target_adapter(mode: str, **kwargs: Any) -> TargetAdapter:
    try:
        factory = _ADAPTERS[mode]
    except KeyError:
        raise ValueError(f"unknown target adapter mode {mode!r}; available: {sorted(_ADAPTERS)}") from None
    return factory(**kwargs)


def _not_implemented(mode: str) -> Callable[..., TargetAdapter]:
    def factory(**_: Any) -> TargetAdapter:
        raise NotImplementedError(f"target adapter mode {mode!r} is reserved but not implemented yet")
    return factory


register_target_adapter("stateless", StatelessSessionAdapter)
register_target_adapter("sessions", _not_implemented("sessions"))
register_target_adapter("traceById", _not_implemented("traceById"))
