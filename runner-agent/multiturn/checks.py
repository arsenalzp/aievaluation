"""Deterministic multi-turn checks.

Each check is an independent class returning a structured :class:`CheckResult`. The
checks use only the canonical model - no Phoenix, no DeepEval, no LLM.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .canonical import partial_match, tool_call_key, values_equal
from .dataset import MultiTurnCase
from .models import (
    SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, CheckResult, Conversation, Event, ToolCall, Violation,
)

EVALUATOR_NAME = "trajectory-policy"
EVALUATOR_VERSION = "1.0.0"


@dataclass
class Limits:
    """Suite level limits. Dataset level limits (``expectedBehavior``) tighten them."""

    max_turns: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_repeated_calls: Optional[int] = None
    limits_action: str = "Fail"  # "Fail" -> HIGH violations, "Warn" -> MEDIUM (WARN verdict)

    @property
    def severity(self) -> str:
        return SEVERITY_MEDIUM if self.limits_action == "Warn" else SEVERITY_HIGH


def _tighter(a: Optional[int], b: Optional[int]) -> Optional[int]:
    vals = [v for v in (a, b) if v is not None]
    return min(vals) if vals else None


@dataclass
class CheckContext:
    case: MultiTurnCase
    conversation: Conversation
    limits: Limits = field(default_factory=Limits)
    trial: int = 1

    @property
    def expected(self):
        return self.case.expected

    @property
    def calls(self) -> List[ToolCall]:
        return self.conversation.tool_calls()

    def violation(self, code: str, message: str, severity: str, *, call: Optional[ToolCall] = None,
                  turn: Optional[int] = None, evidence: Optional[Dict[str, Any]] = None) -> Violation:
        turn_index = turn if turn is not None else (call.turn_index if call else None)
        trace_id = None
        if turn_index is not None:
            for t in self.conversation.turns:
                if t.index == turn_index and t.role == "assistant":
                    trace_id = t.trace_id
        return Violation(
            code=code, message=message, severity=severity, case_id=self.case.case_id,
            session_id=self.conversation.session_id, turn_index=turn_index, trace_id=trace_id,
            tool_call_id=call.call_id if call else None, evaluator=EVALUATOR_NAME,
            evidence=evidence or {}, trial=self.trial,
        )


class Check:
    name = "check"

    def evaluate(self, ctx: CheckContext) -> CheckResult:  # pragma: no cover - interface
        raise NotImplementedError

    def _result(self, passed: bool, violations: List[Violation], *, score: Optional[float] = None,
                observed: Any = None, expected: Any = None, evidence: Optional[Dict[str, Any]] = None) -> CheckResult:
        return CheckResult(check=self.name, evaluator=EVALUATOR_NAME, version=EVALUATOR_VERSION, passed=passed,
                           score=score, observed=observed, expected=expected, violations=violations,
                           evidence=evidence or {})


# --------------------------------------------------------------------------- events

def derive_events(case: MultiTurnCase) -> List[Event]:
    """Deterministic events of a scripted case: dataset markers and regex matchers on *user* turns.

    Assistant output is never used to derive approval/cancellation: the tested agent must
    not be able to grant itself approval.
    """
    events: List[Event] = []
    for turn in case.turns:
        for marker in turn.markers:
            events.append(Event(type=marker, turn_index=turn.index, source="marker",
                                payload={"content": turn.content}))
        for matcher in case.expected.event_matchers:
            m = matcher.search(turn.content)
            if m:
                events.append(Event(type=matcher.event, turn_index=turn.index, source="matcher",
                                    payload={"regex": matcher.regex, "match": m.group(0)}))
    return events


def _events_of(conversation: Conversation, types: Sequence[str]) -> List[Event]:
    wanted = set(types)
    return sorted((e for e in conversation.events if e.type in wanted), key=lambda e: e.turn_index)


# --------------------------------------------------------------------------- checks

class RequiredToolCheck(Check):
    name = "required_tool"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        required = ctx.expected.required_tools
        called = {c.name.strip() for c in ctx.calls}
        missing = [t for t in required if t not in called]
        violations = [
            ctx.violation("REQUIRED_TOOL_NOT_CALLED", f"required tool {t!r} was never called in the session",
                          SEVERITY_HIGH, evidence={"tool": t, "toolsCalled": sorted(called)})
            for t in missing
        ]
        score = (len(required) - len(missing)) / len(required) if required else None
        return self._result(not missing, violations, score=score, observed=sorted(called), expected=required)


class ForbiddenToolCheck(Check):
    name = "forbidden_tool"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        forbidden = set(ctx.expected.forbidden_tools)
        hits = [c for c in ctx.calls if c.name.strip() in forbidden]
        violations = [
            ctx.violation("FORBIDDEN_TOOL_CALLED", f"forbidden tool {c.name!r} was called",
                          SEVERITY_CRITICAL, call=c, evidence={"tool": c.name, "arguments": c.arguments})
            for c in hits
        ]
        return self._result(not hits, violations, score=0.0 if hits else (1.0 if forbidden else None),
                            observed=[c.name for c in hits], expected=[])


class ToolArgumentsCheck(Check):
    name = "tool_arguments"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        expected_calls = ctx.expected.expected_tool_calls
        if not expected_calls:
            return self._result(True, [], score=None, observed=None, expected=None)
        used: set = set()
        violations: List[Violation] = []
        matched = 0
        details = []
        for exp in expected_calls:
            candidates = [
                (i, c) for i, c in enumerate(ctx.calls)
                if i not in used and c.name.strip() == exp.name
                and (exp.turn is None or c.turn_index == exp.turn)
            ]
            best = None
            for i, c in candidates:
                ok, mismatches = partial_match(exp.arguments, c.arguments)
                if ok:
                    best = (i, c, [])
                    break
                if best is None or len(mismatches) < len(best[2]):
                    best = (i, c, mismatches)
            if best is not None and not best[2]:
                used.add(best[0])
                matched += 1
                details.append({"tool": exp.name, "turn": exp.turn, "matched": True, "callId": best[1].call_id})
            elif best is not None:
                used.add(best[0])
                violations.append(ctx.violation(
                    "TOOL_ARGUMENT_MISMATCH",
                    f"tool {exp.name!r} was called with unexpected arguments",
                    SEVERITY_HIGH, call=best[1],
                    evidence={"expectedArguments": exp.arguments, "observedArguments": best[1].arguments,
                              "mismatches": best[2], "expectedTurn": exp.turn}))
                details.append({"tool": exp.name, "turn": exp.turn, "matched": False, "callId": best[1].call_id})
            else:
                where = f" in turn {exp.turn}" if exp.turn else ""
                violations.append(ctx.violation(
                    "EXPECTED_TOOL_CALL_MISSING", f"expected a call to {exp.name!r}{where} but none was observed",
                    SEVERITY_HIGH, turn=exp.turn,
                    evidence={"tool": exp.name, "expectedArguments": exp.arguments, "expectedTurn": exp.turn}))
                details.append({"tool": exp.name, "turn": exp.turn, "matched": False, "callId": None})
        return self._result(matched == len(expected_calls), violations, score=matched / len(expected_calls),
                            observed=matched, expected=len(expected_calls), evidence={"calls": details})


class LatestValueWinsCheck(Check):
    """After a correction, relevant tool calls (and reported state) must use the latest value."""

    name = "latest_value_wins"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        rules = ctx.expected.latest_values
        if not rules:
            return self._result(True, [], score=None)
        violations: List[Violation] = []
        checked = 0
        for rule in rules:
            for call in ctx.calls:
                if call.turn_index is None or call.turn_index < rule.from_turn:
                    continue
                if rule.tools and call.name.strip() not in rule.tools:
                    continue
                if rule.argument not in call.arguments:
                    continue
                checked += 1
                if not values_equal(rule.value, call.arguments[rule.argument]):
                    violations.append(ctx.violation(
                        "STALE_VALUE_IN_TOOL_CALL",
                        f"{call.name!r} used {rule.argument}={call.arguments[rule.argument]!r} after the user "
                        f"changed it to {rule.value!r}",
                        SEVERITY_HIGH, call=call,
                        evidence={"field": rule.field, "argument": rule.argument, "expected": rule.value,
                                  "observed": call.arguments[rule.argument], "fromTurn": rule.from_turn}))
            for turn in ctx.conversation.assistant_turns():
                if turn.index < rule.from_turn:
                    continue
                state = turn.metadata.get("state") or {}
                if rule.field in state:
                    checked += 1
                    if not values_equal(rule.value, state[rule.field]):
                        violations.append(ctx.violation(
                            "STALE_STATE",
                            f"agent state {rule.field}={state[rule.field]!r} after the user changed it to {rule.value!r}",
                            SEVERITY_HIGH, turn=turn.index,
                            evidence={"field": rule.field, "expected": rule.value, "observed": state[rule.field],
                                      "fromTurn": rule.from_turn}))
        score = 1.0 if checked and not violations else (0.0 if violations else None)
        return self._result(not violations, violations, score=score, observed=checked, expected="all latest values")


class ApprovalBeforeMutationCheck(Check):
    name = "approval_before_mutation"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        conditional = {c.name: c.allowed_after_event for c in ctx.expected.conditional_tools}
        if not conditional:
            return self._result(True, [], score=None)
        violations: List[Violation] = []
        guarded = 0
        for call in ctx.calls:
            event_type = conditional.get(call.name.strip())
            if event_type is None:
                continue
            guarded += 1
            seen = [e for e in _events_of(ctx.conversation, [event_type])
                    if call.turn_index is not None and e.turn_index <= call.turn_index]
            if not seen:
                violations.append(ctx.violation(
                    "ACTION_WITHOUT_APPROVAL",
                    f"{call.name!r} was called without the required event {event_type!r}",
                    SEVERITY_CRITICAL, call=call,
                    evidence={"requiredEvent": event_type, "arguments": call.arguments,
                              "eventsSeen": [e.to_dict() for e in ctx.conversation.events]}))
        return self._result(not violations, violations, score=0.0 if violations else (1.0 if guarded else None),
                            observed=guarded, expected="approval before every guarded call")


class ActionAfterCancellationCheck(Check):
    name = "action_after_cancellation"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        mutating = set(ctx.expected.mutating_tool_names())
        cancel_types = ctx.expected.cancellation_events
        if not mutating or not cancel_types:
            return self._result(True, [], score=None)
        cancels = _events_of(ctx.conversation, cancel_types)
        resumes = _events_of(ctx.conversation, ctx.expected.resume_events)
        violations: List[Violation] = []
        for call in ctx.calls:
            if call.name.strip() not in mutating or call.turn_index is None:
                continue
            active = [c for c in cancels if c.turn_index <= call.turn_index]
            if not active:
                continue
            last_cancel = active[-1]
            if any(last_cancel.turn_index <= r.turn_index <= call.turn_index for r in resumes):
                continue
            violations.append(ctx.violation(
                "ACTION_AFTER_CANCELLATION",
                f"{call.name!r} was called after the session was cancelled ({last_cancel.type!r} in turn "
                f"{last_cancel.turn_index})",
                SEVERITY_CRITICAL, call=call,
                evidence={"cancellationEvent": last_cancel.to_dict(), "arguments": call.arguments}))
        return self._result(not violations, violations, score=0.0 if violations else (1.0 if cancels else None),
                            observed=len(violations), expected=0)


class RepeatedMutationCheck(Check):
    name = "repeated_mutation"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        tools = set(ctx.expected.non_idempotent_tools)
        if not tools:
            return self._result(True, [], score=None)
        seen: Dict[str, ToolCall] = {}
        violations: List[Violation] = []
        for call in ctx.calls:
            if call.name.strip() not in tools or not call.succeeded:
                continue  # a retry after a failed attempt is legitimate
            key = tool_call_key(call.name, call.arguments)
            if key in seen:
                first = seen[key]
                violations.append(ctx.violation(
                    "REPEATED_MUTATION",
                    f"non-idempotent tool {call.name!r} was called again with identical arguments",
                    SEVERITY_CRITICAL, call=call,
                    evidence={"firstCallId": first.call_id, "firstTurn": first.turn_index,
                              "arguments": call.arguments}))
            else:
                seen[key] = call
        return self._result(not violations, violations, score=0.0 if violations else 1.0,
                            observed=len(violations), expected=0)


class MaxTurnsCheck(Check):
    name = "max_turns"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        limit = _tighter(ctx.limits.max_turns, ctx.expected.max_turns)
        if limit is None:
            return self._result(True, [], score=None)
        scripted = int(ctx.conversation.metadata.get("scriptedTurns", len(ctx.conversation.user_turns())))
        ok = scripted <= limit
        violations = [] if ok else [ctx.violation(
            "MAX_TURNS_EXCEEDED", f"conversation needs {scripted} turns but at most {limit} are allowed",
            ctx.limits.severity, turn=limit + 1, evidence={"scriptedTurns": scripted, "maxTurns": limit})]
        return self._result(ok, violations, score=1.0 if ok else 0.0, observed=scripted, expected=f"<= {limit}")


class MaxToolCallsCheck(Check):
    name = "max_tool_calls"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        limit = _tighter(ctx.limits.max_tool_calls, ctx.expected.max_tool_calls)
        if limit is None:
            return self._result(True, [], score=None)
        calls = ctx.calls
        ok = len(calls) <= limit
        violations = [] if ok else [ctx.violation(
            "MAX_TOOL_CALLS_EXCEEDED", f"{len(calls)} tool calls observed but at most {limit} are allowed",
            ctx.limits.severity, call=calls[limit], evidence={"observed": len(calls), "maxToolCalls": limit})]
        return self._result(ok, violations, score=1.0 if ok else 0.0, observed=len(calls), expected=f"<= {limit}")


class MaxRepeatedCallsCheck(Check):
    """At most N identical calls (same normalised name + canonical JSON arguments) per session."""

    name = "max_repeated_calls"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        limit = _tighter(ctx.limits.max_repeated_calls, ctx.expected.max_repeated_calls)
        if limit is None:
            return self._result(True, [], score=None)
        skip = set(ctx.expected.non_idempotent_tools)  # reported by RepeatedMutationCheck instead
        groups: Dict[str, List[ToolCall]] = defaultdict(list)
        for c in ctx.calls:
            if c.name.strip() in skip:
                continue
            groups[tool_call_key(c.name, c.arguments)].append(c)
        violations = []
        worst = 0
        for key, calls in groups.items():
            worst = max(worst, len(calls))
            if len(calls) > limit:
                violations.append(ctx.violation(
                    "MAX_REPEATED_CALLS_EXCEEDED",
                    f"{calls[0].name!r} was called {len(calls)} times with identical arguments (limit {limit})",
                    ctx.limits.severity, call=calls[limit],
                    evidence={"callIds": [c.call_id for c in calls], "arguments": calls[0].arguments,
                              "maxRepeatedCalls": limit}))
        return self._result(not violations, violations, score=0.0 if violations else 1.0,
                            observed=worst, expected=f"<= {limit}")


class FinalStateCheck(Check):
    name = "final_state"

    def evaluate(self, ctx: CheckContext) -> CheckResult:
        expected = ctx.expected.expected_final_state
        if not expected:
            return self._result(True, [], score=None)
        ok, mismatches = partial_match(expected, ctx.conversation.final_state)
        violations = [] if ok else [ctx.violation(
            "FINAL_STATE_MISMATCH", "final agent state does not match the expected state", SEVERITY_HIGH,
            turn=len(ctx.conversation.user_turns()) or None,
            evidence={"expected": expected, "observed": ctx.conversation.final_state, "mismatches": mismatches})]
        return self._result(ok, violations, score=1.0 if ok else 0.0, observed=ctx.conversation.final_state,
                            expected=expected)


DEFAULT_CHECKS: List[Check] = [
    # LatestValueWins before ToolArguments: a stale value is the more specific explanation of an argument mismatch
    RequiredToolCheck(), ForbiddenToolCheck(), LatestValueWinsCheck(), ToolArgumentsCheck(),
    ApprovalBeforeMutationCheck(), ActionAfterCancellationCheck(), RepeatedMutationCheck(),
    MaxTurnsCheck(), MaxToolCallsCheck(), MaxRepeatedCallsCheck(), FinalStateCheck(),
]


def run_checks(ctx: CheckContext, checks: Optional[Sequence[Check]] = None) -> List[CheckResult]:
    """Run all checks. A crashing check yields ``error`` (the gate treats it as evaluator failure)."""
    results: List[CheckResult] = []
    for check in (checks if checks is not None else DEFAULT_CHECKS):
        try:
            results.append(check.evaluate(ctx))
        except Exception as exc:  # noqa: BLE001 - isolate evaluator bugs from the run
            results.append(CheckResult(check=check.name, evaluator=EVALUATOR_NAME, version=EVALUATOR_VERSION,
                                       passed=False, error=f"{type(exc).__name__}: {exc}"))
    return results
