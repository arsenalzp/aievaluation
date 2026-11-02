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

from .models import ToolCall, normalize_status


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


def parse_inline_tool_calls(trace: Dict[str, Any], turn_index: int, fallback: Optional[List[Any]] = None) -> List[ToolCall]:
    """Normalise ``trace.toolCalls`` of a turn response into canonical ToolCalls."""
    raw_calls = None
    if isinstance(trace, dict):
        raw_calls = trace.get("toolCalls") or trace.get("tool_calls")
    if raw_calls is None:
        raw_calls = fallback
    calls: List[ToolCall] = []
    for i, raw in enumerate(raw_calls or []):
        if not isinstance(raw, dict):
            raise ValueError(f"trace.toolCalls[{i}] must be an object")
        name = raw.get("name") or raw.get("tool")
        if not name or not str(name).strip():
            raise ValueError(f"trace.toolCalls[{i}] is missing a tool name")
        args = raw.get("arguments", raw.get("args", {})) or {}
        if not isinstance(args, dict):
            raise ValueError(f"trace.toolCalls[{i}].arguments must be an object")
        error = raw.get("error")
        calls.append(ToolCall(
            call_id=str(raw.get("callId") or raw.get("id") or f"turn{turn_index}-call{i + 1}"),
            name=str(name).strip(),
            arguments=args,
            status=normalize_status(raw.get("status"), error),
            result=raw.get("result", raw.get("output")),
            error=str(error) if error else None,
            duration_ms=raw.get("durationMs", raw.get("duration_ms")),
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
        output = data.get("output", data.get("answer"))
        if not isinstance(output, str):
            raise TargetError("response is missing the string field 'output'", case_id, request.turn)

        state = data.get("state") or {}
        if not isinstance(state, dict):
            raise TargetError("response field 'state' must be an object", case_id, request.turn)
        trace = data.get("trace") or {}
        if not isinstance(trace, dict):
            raise TargetError("response field 'trace' must be an object", case_id, request.turn)
        metadata = dict(data.get("metadata") or {})
        metadata.setdefault("latencyMs", elapsed_ms)

        try:
            calls = parse_inline_tool_calls(trace, request.turn, fallback=data.get("toolCalls"))
        except ValueError as exc:
            raise TargetError(str(exc), case_id, request.turn) from exc

        return TurnResponse(
            session_id=data.get("sessionId"),
            output=output,
            state=state,
            trace=trace,
            metadata=metadata,
            tool_calls=calls,
            has_inline_trace=bool(trace) or "toolCalls" in data,
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
