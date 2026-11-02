"""Versioned multi-turn artifacts and secret redaction.

Everything written to object storage passes through :func:`redact` first.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Protocol

from .models import SCHEMA_VERSION, Conversation
from .results import CaseResult, TrialResult

REDACTED = "[REDACTED]"
MAX_STRING_CHARS = 8000

# normalised key = lower-case with "_", "-" and spaces removed
_SENSITIVE_EXACT = {
    "authorization", "proxyauthorization", "apikey", "xapikey", "accesstoken", "refreshtoken", "idtoken",
    "token", "bearer", "secret", "clientsecret", "password", "passwd", "passphrase", "privatekey",
    "cookie", "setcookie", "credentials", "credential", "sessiontoken", "awssecretaccesskey",
}
_SENSITIVE_SUFFIXES = ("apikey", "secret", "password", "accesstoken", "refreshtoken", "authorization", "privatekey")
_SECRET_VALUE_PATTERNS = [
    re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]


def is_sensitive_key(key: Any) -> bool:
    k = re.sub(r"[\s_\-]", "", str(key)).lower()
    return k in _SENSITIVE_EXACT or k.endswith(_SENSITIVE_SUFFIXES)


def _redact_string(value: str) -> str:
    for pattern in _SECRET_VALUE_PATTERNS:
        value = pattern.sub(REDACTED, value)
    if len(value) > MAX_STRING_CHARS:
        value = value[:MAX_STRING_CHARS] + f"...[truncated {len(value) - MAX_STRING_CHARS} chars]"
    return value


def redact(value: Any) -> Any:
    """Recursively redact sensitive keys and secret-looking string values (returns a copy)."""
    if isinstance(value, dict):
        return {k: (REDACTED if is_sensitive_key(k) and v is not None else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def redact_conversation(conv: Conversation) -> Conversation:
    """Redacted copy used both for artifacts and for anything sent to an external judge LLM."""
    turns = []
    for t in conv.turns:
        calls = [replace(c, arguments=redact(c.arguments), result=redact(c.result),
                         error=_redact_string(c.error) if c.error else c.error) for c in t.tool_calls]
        turns.append(replace(t, content=_redact_string(t.content), tool_calls=calls,
                             metadata=redact(copy.deepcopy(t.metadata)),
                             retrieval_context=[_redact_string(x) for x in t.retrieval_context]))
    return replace(conv, turns=turns, final_state=redact(conv.final_state),
                   events=[replace(e, payload=redact(e.payload)) for e in conv.events],
                   metadata=redact(copy.deepcopy(conv.metadata)))


class Uploader(Protocol):
    def put_json(self, key: str, payload: Dict[str, Any]) -> Any: ...
    def put_text(self, key: str, text: str, content_type: str = "text/plain") -> Any: ...


def _safe_name(case_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", case_id)


@dataclass
class ArtifactLayout:
    bucket: str
    gate_summary_key: str
    scores_key: str
    details_key: str
    manifest_key: str
    violations_key: str
    conversations_prefix: str
    # Object-store gateway base URL (S3_ENDPOINT). Artifact URIs use the same form as the RAG runner:
    # {endpoint}/buckets/{bucket}/objects/{key}. Without an endpoint (local runs) they fall back to s3://bucket/key.
    endpoint: str = ""

    @classmethod
    def from_env(cls, bucket: str, gate_summary_key: str, scores_key: str, details_key: str,
                 env: Mapping[str, str]) -> "ArtifactLayout":
        gate_dir = gate_summary_key.rsplit("/", 1)[0] if "/" in gate_summary_key else ""
        prefix = (env.get("S3_RUN_PREFIX") or gate_dir).rstrip("/")
        join = (lambda name: f"{prefix}/{name}") if prefix else (lambda name: name)
        return cls(
            bucket=bucket, gate_summary_key=gate_summary_key, scores_key=scores_key, details_key=details_key,
            manifest_key=env.get("MANIFEST_KEY") or join("manifest.json"),
            violations_key=env.get("VIOLATIONS_KEY") or join("violations.jsonl"),
            conversations_prefix=(env.get("CONVERSATIONS_PREFIX") or join("conversations")).rstrip("/"),
            endpoint=(env.get("S3_ENDPOINT") or "").rstrip("/"),
        )

    def uri(self, key: str) -> str:
        if self.endpoint:
            return f"{self.endpoint}/buckets/{self.bucket}/objects/{key}"
        return f"s3://{self.bucket}/{key}"

    def conversation_key(self, case_id: str, trial: int) -> str:
        suffix = "" if trial == 1 else f"__t{trial}"
        return f"{self.conversations_prefix}/{_safe_name(case_id)}{suffix}.json"

    def artifact_uris(self) -> Dict[str, str]:
        return {
            "gateSummaryUri": self.uri(self.gate_summary_key),
            "scoresUri": self.uri(self.scores_key),
            "detailsUri": self.uri(self.details_key),
            "manifestUri": self.uri(self.manifest_key),
            "violationsUri": self.uri(self.violations_key),
            "conversationsUri": self.uri(self.conversations_prefix + "/"),
        }


def state_transitions(conv: Conversation) -> List[Dict[str, Any]]:
    """Per-turn changes of the agent-reported state (only turns where something changed)."""
    out: List[Dict[str, Any]] = []
    previous: Dict[str, Any] = {}
    for turn in sorted(conv.assistant_turns(), key=lambda t: t.index):
        state = turn.metadata.get("state") or {}
        changed = {k: {"from": previous.get(k), "to": v} for k, v in state.items() if previous.get(k) != v}
        removed = {k: {"from": v, "to": None} for k, v in previous.items() if k not in state}
        changed.update(removed)
        if changed:
            out.append({"turn": turn.index, "traceId": turn.trace_id, "changed": changed, "state": state})
        previous = state
    return out


def _termination(trial: TrialResult) -> str:
    reason = trial.conversation.termination_reason
    if reason == "COMPLETED" and any(v.severity == "CRITICAL" for v in trial.violations):
        return "POLICY_VIOLATION"
    return reason


def build_conversation_artifact(trial: TrialResult) -> Dict[str, Any]:
    conv = trial.conversation
    deterministic = {
        c.check: {"passed": c.passed, "score": c.score, "observed": c.observed, "expected": c.expected,
                  "evaluator": c.evaluator, "version": c.version, "error": c.error}
        for c in trial.checks
    }
    semantic: Dict[str, Any] = {}
    for result in trial.semantic:
        if result.error and not result.scores:
            semantic[result.evaluator] = {"error": result.error, "required": result.required,
                                          "provider": result.provider}
        for s in result.scores:
            entry = s.to_dict()
            entry.update({"required": result.required, "provider": result.provider})
            semantic[s.metric] = entry
    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "caseId": conv.case_id,
        "sessionId": conv.session_id,
        "trial": trial.trial,
        "outcome": {"status": trial.status, "goalCompleted": trial.goal_completed,
                    "terminationReason": _termination(trial), **({"error": trial.error} if trial.error else {})},
        "turns": [t.to_dict() for t in sorted(conv.turns, key=lambda x: (x.index, 0 if x.role == "user" else 1))],
        "events": [e.to_dict() for e in conv.events],
        "trajectory": {"toolCalls": [c.to_dict() for c in conv.tool_calls()],
                       "stateTransitions": state_transitions(conv)},
        "finalState": conv.final_state,
        "scores": {"deterministic": deterministic, "semantic": semantic},
        "violations": [{"code": v.code, "severity": v.severity, "turn": v.turn_index, "toolCallId": v.tool_call_id,
                        "traceId": v.trace_id, "message": v.message, "evaluator": v.evaluator,
                        "evidence": v.evidence} for v in trial.violations],
        "metadata": conv.metadata,
    }
    return redact(doc)


def write_conversations(cases: List[CaseResult], layout: ArtifactLayout, uploader: Uploader) -> None:
    for case in cases:
        for trial in case.trials:
            trial.conversation_key = layout.conversation_key(case.case_id, trial.trial)
            trial.conversation_uri = layout.uri(trial.conversation_key)
            uploader.put_json(trial.conversation_key, build_conversation_artifact(trial))


def write_violations(cases: List[CaseResult], layout: ArtifactLayout, uploader: Uploader) -> None:
    lines = []
    for case in cases:
        for trial in case.trials:
            for v in trial.violations:
                row = {"schemaVersion": SCHEMA_VERSION, **v.to_dict(), "conversationUri": trial.conversation_uri}
                lines.append(json.dumps(redact(row), ensure_ascii=False, sort_keys=True))
    uploader.put_text(layout.violations_key, "".join(line + "\n" for line in lines), "application/x-ndjson")


def write_details(cases: List[CaseResult], layout: ArtifactLayout, uploader: Uploader) -> None:
    """details.jsonl: one row per case, keeps the single-turn key names for existing consumers."""
    lines = []
    for case in cases:
        t = case.first_trial()
        conv = t.conversation
        users = conv.user_turns()
        assistants = conv.assistant_turns()
        row = {
            "schemaVersion": "evalrun.agent_details/v1",
            "interactionMode": "MultiTurn",
            "caseId": case.case_id,
            "sessionId": conv.session_id,
            "input": users[0].content if users else None,
            "expectedOutput": None,
            "actualOutput": assistants[-1].content if assistants else None,
            "toolsCalled": [c.to_dict() for c in conv.tool_calls()],
            "trace": {"traceIds": [a.trace_id for a in assistants if a.trace_id]},
            "metadata": conv.metadata,
            "turns": len(users),
            "score": {
                "passed": case.pass_all, "passRate": case.pass_rate,
                "tool_selection": t.tool_selection, "tool_invocation": t.tool_arguments,
                "policy_violations": len([v for v in case.violations() if v.severity in ("CRITICAL", "HIGH")]),
                "violations": len(case.violations()),
            },
            "conversationUri": t.conversation_uri,
        }
        lines.append(json.dumps(redact(row), ensure_ascii=False, sort_keys=True))
    uploader.put_text(layout.details_key, "".join(line + "\n" for line in lines), "application/x-ndjson")


def write_json_artifact(uploader: Uploader, key: str, payload: Dict[str, Any]) -> None:
    uploader.put_json(key, redact(payload))
