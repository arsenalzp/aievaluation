"""Framework-neutral canonical models for multi-turn evaluation.

Nothing in this module (or in the rest of the runner outside the adapter modules)
may import Phoenix, OpenInference or DeepEval types.  Adapters translate their
native objects into these dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = "ai.evaluation.io/v1alpha1"

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_HIGH = "HIGH"
SEVERITY_MEDIUM = "MEDIUM"
SEVERITY_LOW = "LOW"
SEVERITIES = (SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, SEVERITY_LOW)

TOOL_STATUS_OK = "OK"
TOOL_STATUS_ERROR = "ERROR"

TOOL_STATUSES = (TOOL_STATUS_OK, TOOL_STATUS_ERROR)


def parse_tool_status(status: Any, error: Any = None) -> str:
    """Tool status of the endpoint contract: exactly ``OK`` or ``ERROR`` (no synonyms, no case folding).

    An omitted status (``None``) is derived from ``error``. Anything else raises ``ValueError``.
    """
    if status is None:
        return TOOL_STATUS_ERROR if error else TOOL_STATUS_OK
    if status not in TOOL_STATUSES:
        raise ValueError(f"status must be one of {list(TOOL_STATUSES)}, got {status!r}")
    return status


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: str = TOOL_STATUS_OK
    result: Any = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    turn_index: Optional[int] = None
    timestamp: Optional[str] = None
    # Where the observation came from: "inline" | "phoenix" (informational only).
    source: str = "inline"

    @property
    def succeeded(self) -> bool:
        return self.status == TOOL_STATUS_OK and not self.error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "callId": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "durationMs": self.duration_ms,
            "turn": self.turn_index,
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class Turn:
    index: int
    role: str
    content: str
    trace_id: Optional[str] = None
    tool_calls: List[ToolCall] = field(default_factory=list)
    retrieval_context: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "role": self.role,
            "content": self.content,
            "traceId": self.trace_id,
            "toolCalls": [c.to_dict() for c in self.tool_calls],
            "retrievalContext": self.retrieval_context,
            "metadata": self.metadata,
        }


@dataclass
class Event:
    type: str
    turn_index: int
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "dataset"
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "turn": self.turn_index,
            "payload": self.payload,
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass
class Conversation:
    case_id: str
    session_id: str
    turns: List[Turn] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    termination_reason: str = "COMPLETED"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def tool_calls(self) -> List[ToolCall]:
        """All tool calls of the session ordered by turn, then observation order."""
        calls: List[ToolCall] = []
        for turn in sorted(self.turns, key=lambda t: t.index):
            calls.extend(turn.tool_calls)
        return calls

    def user_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "user"]

    def assistant_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.role == "assistant"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "caseId": self.case_id,
            "sessionId": self.session_id,
            "turns": [t.to_dict() for t in self.turns],
            "events": [e.to_dict() for e in self.events],
            "finalState": self.final_state,
            "terminationReason": self.termination_reason,
            "metadata": self.metadata,
        }


@dataclass
class Violation:
    code: str
    message: str
    severity: str
    case_id: str
    session_id: str = ""
    turn_index: Optional[int] = None
    trace_id: Optional[str] = None
    tool_call_id: Optional[str] = None
    evaluator: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)
    trial: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "caseId": self.case_id,
            "sessionId": self.session_id,
            "turn": self.turn_index,
            "traceId": self.trace_id,
            "toolCallId": self.tool_call_id,
            "evaluator": self.evaluator,
            "evidence": self.evidence,
            "trial": self.trial,
        }


@dataclass
class CheckResult:
    """Structured result of one deterministic check."""

    check: str
    evaluator: str
    version: str
    passed: bool
    score: Optional[float] = None
    observed: Any = None
    expected: Any = None
    violations: List[Violation] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    # Set when the check itself crashed (as opposed to finding a violation).
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check": self.check,
            "evaluator": self.evaluator,
            "version": self.version,
            "passed": self.passed,
            "score": self.score,
            "observed": self.observed,
            "expected": self.expected,
            "violations": [v.to_dict() for v in self.violations],
            "evidence": self.evidence,
            "error": self.error,
        }


@dataclass
class SemanticScore:
    metric: str
    score: Optional[float]
    threshold: Optional[float]
    reason: Optional[str] = None
    model: Optional[str] = None
    evaluator: str = ""
    version: str = ""
    samples: int = 1
    spread: Optional[List[float]] = None  # [min, max] across samples
    error: Optional[str] = None

    @property
    def passed(self) -> Optional[bool]:
        if self.score is None or self.threshold is None:
            return None
        return self.score >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "score": self.score,
            "threshold": self.threshold,
            "passed": self.passed,
            "reason": self.reason,
            "model": self.model,
            "evaluator": self.evaluator,
            "version": self.version,
            "samples": self.samples,
            "spread": self.spread,
            "error": self.error,
        }


@dataclass
class SemanticResult:
    """Outcome of one configured semantic evaluator for one conversation."""

    evaluator: str
    provider: str
    required: bool
    scores: List[SemanticScore] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evaluator": self.evaluator,
            "provider": self.provider,
            "required": self.required,
            "scores": [s.to_dict() for s in self.scores],
            "error": self.error,
        }
