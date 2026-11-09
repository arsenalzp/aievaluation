"""Framework-neutral release gate for multi-turn evaluation.

This module owns the final PASS / WARN / FAIL verdict. It never talks to Phoenix or
DeepEval - it only consumes canonical violations, aggregated scores and semantic results.

Precedence (first matching tier decides the verdict and the root cause):

1. runner failure, or failure of a *required* evaluator            -> FAIL
2. critical (zero tolerance) violations above the configured limit  -> FAIL
3. deterministic threshold failure (HIGH violations, goal completion,
   tool selection / argument accuracy, regression)                   -> FAIL
4. required semantic metric below its threshold                      -> FAIL
5. optional evaluator failure, optional semantic metric below its
   threshold, MEDIUM/LOW violations                                  -> WARN
6. otherwise                                                         -> PASS

Semantic scores can only add failures/warnings; they never compensate for violations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .config import MultiTurnSettings
from .models import SEVERITY_CRITICAL, SEVERITY_HIGH, Violation

VERDICT_PASS = "PASS"
VERDICT_WARN = "WARN"
VERDICT_FAIL = "FAIL"

STAGE_EXECUTION = "Execution"
STAGE_EVALUATOR = "Evaluator"
STAGE_POLICY = "Policy"
STAGE_OUTCOME = "Outcome"
STAGE_TOOL_SELECTION = "Tool Selection"
STAGE_TOOL_INVOCATION = "Tool Invocation"
STAGE_REGRESSION = "Regression"
STAGE_CONVERSATION = "Conversation"


@dataclass
class SemanticAggregate:
    evaluator: str
    metric: str
    required: bool
    average: Optional[float]
    threshold: Optional[float]
    model: Optional[str] = None
    version: Optional[str] = None
    per_case: Dict[str, float] = field(default_factory=dict)
    # Evaluator version/model changed vs. baseline etc. are handled by the caller.


@dataclass
class EvaluatorFailure:
    evaluator: str
    required: bool
    message: str
    case_id: Optional[str] = None


@dataclass
class GateInput:
    settings: MultiTurnSettings
    violations: List[Violation] = field(default_factory=list)
    aggregates: Dict[str, Optional[float]] = field(default_factory=dict)
    semantic: List[SemanticAggregate] = field(default_factory=list)
    evaluator_failures: List[EvaluatorFailure] = field(default_factory=list)
    regression_failures: List[Dict[str, Any]] = field(default_factory=list)
    runner_error: Optional[str] = None
    # case id -> artifact URI of the first conversation that supports a finding
    conversation_uri: Callable[[str], Optional[str]] = lambda _case_id: None
    violations_uri: Optional[str] = None
    scores_uri: Optional[str] = None
    total_cases: int = 0


@dataclass
class GateDecision:
    verdict: str
    summary: str
    root_cause: Dict[str, Any]
    findings: List[Dict[str, Any]] = field(default_factory=list)  # every failure/warning, in precedence order

    def to_dict(self) -> Dict[str, Any]:
        return {"verdict": self.verdict, "summary": self.summary, "rootCause": self.root_cause,
                "findings": self.findings}


def _rc(stage: str, **kw: Any) -> Dict[str, Any]:
    return {"stage": stage, **{k: v for k, v in kw.items() if v is not None}}


def _fmt(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _violation_finding(v: Violation, count: int, limit: int, uri: Optional[str], stage: str = STAGE_POLICY) -> Dict[str, Any]:
    return {
        "level": VERDICT_FAIL,
        **_rc(stage, evaluator=v.evaluator, code=v.code, metric=f"{v.severity.lower()}_violations",
              observed=str(count), required=f"<= {limit}", caseId=v.case_id, turnIndex=v.turn_index,
              artifactUri=uri, details=v.message),
    }


def decide(inp: GateInput) -> GateDecision:
    s = inp.settings
    fails: List[Dict[str, Any]] = []
    warns: List[Dict[str, Any]] = []

    def case_uri(case_id: Optional[str]) -> Optional[str]:
        return (inp.conversation_uri(case_id) if case_id else None) or inp.violations_uri or inp.scores_uri

    # 1. runner / required evaluator failure
    if inp.runner_error:
        fails.append(
            {
                "level": VERDICT_FAIL, **_rc(
                    STAGE_EXECUTION,
                    evaluator="runner",
                    code="RUNNER_FAILURE",
                    details=inp.runner_error, artifactUri=inp.scores_uri
                )
            }
        )
    for f in inp.evaluator_failures:
        entry = {"level": VERDICT_FAIL if f.required else VERDICT_WARN,
                 **_rc(STAGE_EVALUATOR, evaluator=f.evaluator, code="EVALUATOR_FAILURE", caseId=f.case_id,
                       details=f.message, required="evaluator must complete" if f.required else None,
                       observed="failed", artifactUri=case_uri(f.case_id))}
        (fails if f.required else warns).append(entry)

    # 2. critical violations (zero tolerance by default)
    crit = [v for v in inp.violations if v.severity == SEVERITY_CRITICAL]
    if len(crit) > s.max_critical_violations:
        first = crit[0]
        fails.append(_violation_finding(first, len(crit), s.max_critical_violations,
                                        case_uri(first.case_id)))

    # 3. deterministic thresholds
    high = [v for v in inp.violations if v.severity == SEVERITY_HIGH]
    if len(high) > s.max_high_violations:
        first = high[0]
        fails.append(_violation_finding(first, len(high), s.max_high_violations, case_uri(first.case_id)))

    def threshold_min(metric_key: str, label: str, stage: str, minimum: Optional[float], code: str) -> None:
        value = inp.aggregates.get(metric_key)
        if minimum is None or value is None:
            return
        if value < minimum:
            fails.append({"level": VERDICT_FAIL, **_rc(
                stage, evaluator="trajectory-policy", code=code, metric=label, observed=_fmt(value),
                required=f">= {_fmt(minimum)}", artifactUri=inp.scores_uri,
                details=f"{label}={_fmt(value)} < {_fmt(minimum)}")})

    threshold_min("goal_completion", "goalCompletion", STAGE_OUTCOME, s.min_goal_completion, "GOAL_COMPLETION_BELOW_MIN")
    threshold_min("tool_selection_accuracy", "toolSelectionAccuracy", STAGE_TOOL_SELECTION,
                  s.min_selection_accuracy, "TOOL_SELECTION_BELOW_MIN")
    threshold_min("tool_argument_accuracy", "toolArgumentAccuracy", STAGE_TOOL_INVOCATION,
                  s.min_argument_accuracy, "TOOL_ARGUMENTS_BELOW_MIN")
    for rf in inp.regression_failures:
        # semantic regressions of optional evaluators only warn
        (fails if rf.get("required", True) else warns).append({"level": VERDICT_FAIL if rf.get("required", True) else VERDICT_WARN, **_rc(
            STAGE_REGRESSION, evaluator="regression-policy", code="REGRESSION_EXCEEDED_MAX_DROP",
            metric=rf["metric"], observed=_fmt(rf["current"]), required=f">= {_fmt(rf['baseline'] - rf['maxDrop'])}",
            artifactUri=inp.scores_uri,
            details=f"{rf['metric']} dropped {_fmt(-rf['drop'])} vs baseline {_fmt(rf['baseline'])} (maxDrop={rf['maxDrop']})")})

    # 4 / 5. semantic metrics
    for agg in inp.semantic:
        if agg.threshold is None or agg.average is None:
            continue
        if agg.average < agg.threshold:
            worst = min(agg.per_case, key=agg.per_case.get) if agg.per_case else None
            entry = {"level": VERDICT_FAIL if agg.required else VERDICT_WARN, **_rc(
                STAGE_CONVERSATION, evaluator=agg.evaluator, metric=agg.metric, code="SEMANTIC_BELOW_THRESHOLD",
                observed=_fmt(agg.average), required=f">= {_fmt(agg.threshold)}", caseId=worst,
                artifactUri=case_uri(worst),
                details=f"{agg.metric}={_fmt(agg.average)} < {_fmt(agg.threshold)}"
                        + ("" if agg.required else " (optional evaluator)"))}
            (fails if agg.required else warns).append(entry)

    for v in inp.violations:
        if v.severity not in (SEVERITY_CRITICAL, SEVERITY_HIGH):
            warns.append({"level": VERDICT_WARN, **_rc(
                STAGE_POLICY, evaluator=v.evaluator, code=v.code, metric=f"{v.severity.lower()}_violations",
                observed="1", caseId=v.case_id, turnIndex=v.turn_index, artifactUri=case_uri(v.case_id),
                details=v.message)})

    findings = fails + warns
    notes = f"cases={inp.total_cases}; violations={len(inp.violations)} (critical={len(crit)}, high={len(high)})"
    if fails:
        first = fails[0]
        summary = f"{first['stage']} failed: {first.get('details') or first.get('code')}; {notes}"
        return GateDecision(VERDICT_FAIL, summary, _strip(first), findings)
    if warns:
        first = warns[0]
        summary = f"Passed with warnings ({len(warns)}): {first.get('details') or first.get('code')}; {notes}"
        return GateDecision(VERDICT_WARN, summary, _strip(first), findings)
    return GateDecision(VERDICT_PASS, f"All required checks passed; {notes}", {"stage": "None"}, findings)


def _strip(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in entry.items() if k != "level"}
