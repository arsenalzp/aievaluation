"""Orchestrates a scripted multi-turn evaluation run (framework neutral).

Flow: execute scripted sessions against the target -> resolve trace observations (inline /
Phoenix) -> deterministic checks -> optional semantic evaluators -> gate -> artifacts.
"""
from __future__ import annotations

import datetime as _dt
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import gate as gatemod
from .artifacts import (
    ArtifactLayout, Uploader, redact_conversation, write_conversations, write_details, write_json_artifact,
    write_violations,
)
from .checks import (
    EVALUATOR_VERSION, Check, CheckContext, Limits, derive_events, run_checks, _tighter,
)
from .config import MultiTurnSettings
from .dataset import MultiTurnCase
from .models import (
    SCHEMA_VERSION, SEVERITY_CRITICAL, SEVERITY_HIGH, SEVERITY_MEDIUM, Conversation, ToolCall, Turn, Violation,
)
from .phoenix_adapter import (
    SOURCE_INLINE, SOURCE_PHOENIX, ObservedTurn, PhoenixFetchError, PhoenixRestFetcher,
    normalize_openinference_spans, resolve_turn_tool_calls,
)
from .results import CaseResult, RunResult, TrialResult, mean
from .semantic import ProviderFactory, run_semantic_evaluators
from .target import TargetAdapter, TargetError, TurnRequest

RUNNER_NAME = "evalrun-multiturn"
RUNNER_VERSION = "1.0.0"


def session_id_for(case: MultiTurnCase, evalrun_name: str, trial: int) -> str:
    base = case.session_id or f"evalrun-{evalrun_name}-{case.case_id}"
    return base if trial == 1 else f"{base}-t{trial}"


def _runner_violation(case: MultiTurnCase, session_id: str, trial: int, code: str, message: str, severity: str,
                      turn: Optional[int] = None, trace_id: Optional[str] = None,
                      evidence: Optional[Dict[str, Any]] = None) -> Violation:
    return Violation(code=code, message=message, severity=severity, case_id=case.case_id, session_id=session_id,
                     turn_index=turn, trace_id=trace_id, evaluator="runner", evidence=evidence or {}, trial=trial)


class _Executed:
    """A conversation as executed against the target, before trace resolution and checks."""

    def __init__(self, case: MultiTurnCase, trial: int, conversation: Conversation):
        self.case = case
        self.trial = trial
        self.conversation = conversation
        self.inline_calls: Dict[int, Optional[List[ToolCall]]] = {}
        self.violations: List[Violation] = []
        self.target_error: Optional[str] = None


def execute_conversation(case: MultiTurnCase, trial: int, session_id: str, adapter: TargetAdapter,
                         settings: MultiTurnSettings) -> _Executed:
    """Replay the scripted user turns. The runner forwards only ``sessionId`` and the scripted input;
    it never carries or repairs the agent's business state."""
    max_turns = _tighter(settings.max_turns, case.expected.max_turns)
    scripted = len(case.turns)
    to_run = case.turns[:max_turns] if max_turns else case.turns

    conv = Conversation(case_id=case.case_id, session_id=session_id,
                        metadata={"scriptedTurns": scripted, "trial": trial, "targetAdapter": adapter.name})
    ex = _Executed(case, trial, conv)

    for scripted_turn in to_run:
        idx = scripted_turn.index
        conv.turns.append(Turn(index=idx, role="user", content=scripted_turn.content,
                               metadata={"markers": scripted_turn.markers}))
        try:
            resp = adapter.send_turn(TurnRequest(session_id=session_id, input=scripted_turn.content, turn=idx),
                                     case_id=case.case_id)
        except TargetError as exc:
            ex.target_error = exc.message
            conv.termination_reason = "TARGET_ERROR"
            ex.violations.append(_runner_violation(
                case, session_id, trial, "TARGET_ERROR", f"target failed in turn {idx}: {exc.message}",
                SEVERITY_HIGH, turn=idx, evidence={"error": exc.message}))
            break
        if resp.session_id is not None and resp.session_id != session_id:
            ex.violations.append(_runner_violation(
                case, session_id, trial, "SESSION_ID_MISMATCH",
                f"target answered for session {resp.session_id!r} instead of {session_id!r}", SEVERITY_HIGH,
                turn=idx, trace_id=resp.trace_id, evidence={"sent": session_id, "received": resp.session_id}))
        conv.turns.append(Turn(
            index=idx, role="assistant", content=resp.output, trace_id=resp.trace_id,
            tool_calls=[], metadata={
                "state": resp.state, "parentTraceId": resp.parent_trace_id,
                "latencyMs": resp.metadata.get("latencyMs"), "responseMetadata": resp.metadata,
            }))
        ex.inline_calls[idx] = resp.tool_calls if resp.has_inline_trace else None
        conv.final_state = resp.state

    executed = len([t for t in conv.turns if t.role == "user"])
    if conv.termination_reason == "COMPLETED" and executed < scripted:
        conv.termination_reason = "MAX_TURNS"
    conv.events = [e for e in derive_events(case) if e.turn_index <= executed]
    return ex


def _resolve_traces(executions: List[_Executed], settings: MultiTurnSettings,
                    fetcher: Optional[PhoenixRestFetcher], run_started: str,
                    sleep: Callable[[float], None], retries: int, retry_delay: float
                    ) -> Tuple[Dict[Tuple[str, int], ObservedTurn], List[gatemod.EvaluatorFailure]]:
    """Fetch Phoenix spans once for the whole run. Returns observations keyed by (session_id, turn)."""
    failures: List[gatemod.EvaluatorFailure] = []
    if settings.trace_source == SOURCE_INLINE:
        return {}, failures
    if fetcher is None:
        raise RuntimeError(f"evaluation.trace.source={settings.trace_source} requires a Phoenix fetcher")

    trace_to_turn = {a.trace_id: a.index for ex in executions for a in ex.conversation.assistant_turns() if a.trace_id}
    wanted = set(trace_to_turn)
    observed: Dict[Tuple[str, int], ObservedTurn] = {}
    last_error: Optional[str] = None
    for attempt in range(max(1, retries)):
        try:
            spans = fetcher.fetch_spans(start_time=run_started)
            sessions = normalize_openinference_spans(spans, trace_to_turn)
            observed = {(sid, t.turn_index): t for sid, turns in sessions.items() for t in turns
                        if t.trace_id in wanted and t.turn_index is not None}
            last_error = None
        except PhoenixFetchError as exc:
            last_error = str(exc)
        if last_error is None and wanted <= {t.trace_id for t in observed.values()}:
            break
        if attempt + 1 < max(1, retries):
            sleep(retry_delay)  # spans are exported asynchronously; give the exporter time
    if last_error:
        # In `phoenix` mode the run cannot be evaluated; in `auto` mode inline observations are used.
        failures.append(gatemod.EvaluatorFailure("phoenix", required=settings.trace_source == SOURCE_PHOENIX,
                                                 message=last_error))
    return observed, failures


def evaluate_run(
    *,
    settings: MultiTurnSettings,
    cases: Sequence[MultiTurnCase],
    adapter: TargetAdapter,
    uploader: Uploader,
    layout: ArtifactLayout,
    run_info: Dict[str, Any],
    baseline: Optional[Dict[str, Any]] = None,
    default_model: Optional[str] = None,
    phoenix_fetcher: Optional[PhoenixRestFetcher] = None,
    provider_factory: Optional[ProviderFactory] = None,
    checks: Optional[Sequence[Check]] = None,
    sleep: Callable[[float], None] = time.sleep,
    trace_retries: int = 3,
    trace_retry_delay: float = 2.0,
) -> RunResult:
    run_started = _dt.datetime.now(_dt.timezone.utc).isoformat()
    evalrun = run_info.get("evalrun", "evalrun")

    # 1. execute all scripted sessions
    executions: List[_Executed] = []
    for case in cases:
        for trial in range(1, settings.trials + 1):
            executions.append(execute_conversation(case, trial, session_id_for(case, evalrun, trial), adapter, settings))

    # 2. trace observations from Phoenix (once for the whole run)
    phoenix_obs, evaluator_failures = _resolve_traces(
        executions, settings, phoenix_fetcher, run_started, sleep, trace_retries, trace_retry_delay)

    builtin = settings.builtin_evaluator()
    limits = Limits(settings.max_turns, settings.max_tool_calls, settings.max_repeated_calls, settings.limits_action)
    results: Dict[str, CaseResult] = {c.case_id: CaseResult(c.case_id) for c in cases}

    for ex in executions:
        conv, case = ex.conversation, ex.case
        violations = list(ex.violations)
        for a in conv.assistant_turns():
            res = resolve_turn_tool_calls(settings.trace_source, ex.inline_calls.get(a.index),
                                          phoenix_obs.get((conv.session_id, a.index)))
            for c in res.tool_calls:
                c.turn_index = a.index
            a.tool_calls = res.tool_calls
            a.metadata["traceSource"] = res.source
            if res.missing:
                violations.append(_runner_violation(
                    case, conv.session_id, ex.trial, "TRACE_MISSING",
                    f"no tool-call observations for turn {a.index} (trace source: {settings.trace_source})",
                    SEVERITY_HIGH if settings.trace_source == SOURCE_PHOENIX else SEVERITY_MEDIUM,
                    turn=a.index, trace_id=a.trace_id, evidence={"traceSource": settings.trace_source}))
            if res.conflict:
                violations.append(_runner_violation(
                    case, conv.session_id, ex.trial, "TRACE_SOURCE_CONFLICT",
                    f"inline and Phoenix tool calls differ in turn {a.index}; Phoenix observation used",
                    SEVERITY_MEDIUM, turn=a.index, trace_id=a.trace_id, evidence=res.conflict))
            # parent trace: prefer what the target reported, else what Phoenix derived
            obs = phoenix_obs.get((conv.session_id, a.index))
            if obs and not a.metadata.get("parentTraceId"):
                a.metadata["parentTraceId"] = obs.parent_trace_id

        ctx = CheckContext(case=case, conversation=conv, limits=limits, trial=ex.trial)
        check_results = run_checks(ctx, checks)
        for cr in check_results:
            violations.extend(cr.violations)
            if cr.error:
                evaluator_failures.append(gatemod.EvaluatorFailure(
                    builtin.name, builtin.required, f"check {cr.check} crashed: {cr.error}", case.case_id))

        blocking = [v for v in violations if v.severity in (SEVERITY_CRITICAL, SEVERITY_HIGH)]
        check_error = any(cr.error for cr in check_results)
        trial = TrialResult(
            case_id=case.case_id, trial=ex.trial, conversation=conv, checks=check_results, violations=violations,
            goal_completed=not blocking and not check_error and not ex.target_error,
            tool_selection=next((c.score for c in check_results if c.check == "required_tool"), None),
            tool_arguments=next((c.score for c in check_results if c.check == "tool_arguments"), None),
            error=ex.target_error,
        )
        trial.status = "ERROR" if (ex.target_error or check_error) else ("PASSED" if trial.goal_completed else "FAILED")
        if conv.termination_reason == "COMPLETED" and any(v.severity == SEVERITY_CRITICAL for v in violations):
            conv.termination_reason = "POLICY_VIOLATION"
        results[case.case_id].trials.append(trial)

    case_results = [results[c.case_id] for c in cases]

    # 3. optional semantic evaluation (first trial of each case)
    if settings.semantic_evaluators():
        for cr in case_results:
            t = cr.first_trial()
            if t.status == "ERROR" and t.error:
                continue
            t.semantic = run_semantic_evaluators(t.conversation, settings, default_model=default_model,
                                                 provider_factory=provider_factory, redact=redact_conversation)
            for sr in t.semantic:
                if sr.error:
                    evaluator_failures.append(gatemod.EvaluatorFailure(sr.evaluator, sr.required, sr.error, cr.case_id))

    # 4. aggregates, regression
    all_trials = [t for cr in case_results for t in cr.trials]
    all_violations = [v for t in all_trials for v in t.violations]
    aggregates: Dict[str, Optional[float]] = {
        "goal_completion": mean([1.0 if cr.pass_all else 0.0 for cr in case_results]),
        "tool_selection_accuracy": mean([t.tool_selection for t in all_trials]),
        "tool_argument_accuracy": mean([t.tool_arguments for t in all_trials]),
    }
    semantic_aggs = _semantic_aggregates(case_results, settings)
    regression, regression_failures = _regression(aggregates, semantic_aggs, baseline, settings)

    # 5. write artifacts (conversations first: the gate refers to their URIs; gate_summary last)
    write_conversations(case_results, layout, uploader)
    write_violations(case_results, layout, uploader)
    write_details(case_results, layout, uploader)
    uri_by_case = {cr.case_id: next((t.conversation_uri for t in cr.trials if t.violations), cr.first_trial().conversation_uri)
                   for cr in case_results}

    decision = gatemod.decide(gatemod.GateInput(
        settings=settings, violations=all_violations, aggregates=aggregates, semantic=semantic_aggs,
        evaluator_failures=evaluator_failures, regression_failures=regression_failures,
        conversation_uri=lambda cid: uri_by_case.get(cid),
        violations_uri=layout.uri(layout.violations_key), scores_uri=layout.uri(layout.scores_key),
        total_cases=len(case_results)))

    artifacts = layout.artifact_uris()
    scores = _scores_payload(run_info, settings, case_results, aggregates, semantic_aggs, regression, builtin.name)
    write_json_artifact(uploader, layout.scores_key, scores)
    write_json_artifact(uploader, layout.manifest_key, _manifest(run_info, settings, case_results, layout, decision))
    write_json_artifact(uploader, layout.gate_summary_key, {
        "schemaVersion": "evalrun.gate_summary/v1",
        "evaluationType": "agent",
        "interactionMode": "MultiTurn",
        "verdict": decision.verdict,
        "summary": decision.summary,
        "rootCause": decision.root_cause,
        "findings": decision.findings,
        "artifacts": artifacts,
    })
    return RunResult(case_results, aggregates, decision.verdict, decision.summary, decision.root_cause,
                     decision.findings, artifacts, regression)


def _semantic_aggregates(cases: List[CaseResult], settings: MultiTurnSettings) -> List[gatemod.SemanticAggregate]:
    out: List[gatemod.SemanticAggregate] = []
    for ev in settings.semantic_evaluators():
        for metric in settings.semantic_metrics():
            per_case: Dict[str, float] = {}
            model = version = None
            for cr in cases:
                for sr in cr.first_trial().semantic:
                    if sr.evaluator != ev.name:
                        continue
                    for s in sr.scores:
                        if s.metric == metric and s.score is not None:
                            per_case[cr.case_id] = s.score
                            model, version = s.model or model, s.version or version
            out.append(gatemod.SemanticAggregate(
                evaluator=ev.name, metric=metric, required=ev.required,
                average=mean(list(per_case.values())), threshold=settings.semantic_threshold(metric),
                model=model, version=version, per_case=per_case))
    return out


_REGRESSION_KEYS = ("goal_completion", "tool_selection_accuracy", "tool_argument_accuracy")


def _regression(aggregates: Dict[str, Optional[float]], semantic: List[gatemod.SemanticAggregate],
                baseline: Optional[Dict[str, Any]], settings: MultiTurnSettings
                ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    max_drop = settings.regression_max_drop
    info: Dict[str, Any] = {"baselineUsed": bool(baseline), "maxDrop": max_drop, "drops": {}, "failed": False,
                            "notComparable": []}
    failures: List[Dict[str, Any]] = []
    if not baseline or max_drop is None:
        return info, failures
    base_agg = baseline.get("aggregatedScores") or {}
    for key in _REGRESSION_KEYS:
        cur, old = aggregates.get(key), base_agg.get(key)
        if cur is None or not isinstance(old, (int, float)):
            continue
        drop = cur - old
        info["drops"][key] = round(drop, 6)
        if drop < -max_drop:
            failures.append({"metric": key, "current": cur, "baseline": float(old), "drop": drop,
                             "maxDrop": max_drop, "required": True})
    base_sem = baseline.get("semantic") or {}
    for agg in semantic:
        b = base_sem.get(agg.metric)
        if agg.average is None or not isinstance(b, dict) or not isinstance(b.get("average"), (int, float)):
            continue
        # Judge scores are only comparable if the judge (model + evaluator version) is unchanged.
        if b.get("model") != agg.model or b.get("evaluatorVersion") != agg.version:
            info["notComparable"].append({"metric": agg.metric, "reason": "judge model/version differs from baseline",
                                          "baseline": {"model": b.get("model"), "version": b.get("evaluatorVersion")},
                                          "current": {"model": agg.model, "version": agg.version}})
            continue
        drop = agg.average - b["average"]
        info["drops"][agg.metric] = round(drop, 6)
        if drop < -max_drop:
            failures.append({"metric": agg.metric, "current": agg.average, "baseline": float(b["average"]),
                             "drop": drop, "maxDrop": max_drop, "required": agg.required})
    info["failed"] = any(f["required"] for f in failures)
    return info, failures


def _scores_payload(run_info: Dict[str, Any], settings: MultiTurnSettings, cases: List[CaseResult],
                    aggregates: Dict[str, Optional[float]], semantic: List[gatemod.SemanticAggregate],
                    regression: Dict[str, Any], builtin_name: str) -> Dict[str, Any]:
    all_trials = [t for cr in cases for t in cr.trials]
    violations = [v for t in all_trials for v in t.violations]
    by_sev: Dict[str, int] = {}
    for v in violations:
        by_sev[v.severity] = by_sev.get(v.severity, 0) + 1
    return {
        "schemaVersion": "evalrun.scores/v1",
        "evaluationType": "agent",
        "interactionMode": "MultiTurn",
        "run": {"evalrun": run_info.get("evalrun"), "evalsuite": run_info.get("evalsuite"),
                "gitSha": run_info.get("gitSha"), "prNumber": run_info.get("prNumber"), "trials": settings.trials},
        "engine": {"name": RUNNER_NAME, "version": RUNNER_VERSION,
                   "evaluators": [{"name": builtin_name, "provider": "builtin", "version": EVALUATOR_VERSION}] +
                                 [{"name": e.name, "provider": e.provider, "required": e.required, "metrics": settings.semantic_metrics(),
                                   "model": e.model, "samples": e.samples} for e in settings.semantic_evaluators()]},
        "aggregatedScores": {
            "total_cases": len(cases),
            "passed_cases": sum(1 for c in cases if c.pass_all),
            **aggregates,
            # names shared with the single-turn schema so baselines/dashboards keep working
            "tool_selection_avg": aggregates.get("tool_selection_accuracy"),
            "tool_invocation_avg": aggregates.get("tool_argument_accuracy"),
            "policy_violations_total": sum(by_sev.get(s, 0) for s in ("CRITICAL", "HIGH")),
            "violations_by_severity": by_sev,
        },
        "deterministic": {c.case_id: {t.trial: {ch.check: {"passed": ch.passed, "score": ch.score}
                                                for ch in t.checks} for t in c.trials} for c in cases},
        "semantic": {a.metric: {"evaluator": a.evaluator, "average": a.average, "threshold": a.threshold,
                                "required": a.required, "model": a.model, "evaluatorVersion": a.version,
                                "perCase": a.per_case} for a in semantic},
        "regression": regression,
        "perCase": [{"case_id": c.case_id, "passed": c.pass_all, "pass_rate": c.pass_rate,
                     "trials": len(c.trials), "violations": len(c.violations()),
                     "tool_selection": mean([t.tool_selection for t in c.trials]),
                     "tool_invocation": mean([t.tool_arguments for t in c.trials])} for c in cases],
    }


def _manifest(run_info: Dict[str, Any], settings: MultiTurnSettings, cases: List[CaseResult],
              layout: ArtifactLayout, decision: gatemod.GateDecision) -> Dict[str, Any]:
    return {
        "schemaVersion": SCHEMA_VERSION,
        "evalrun": run_info.get("evalrun"),
        "evalsuite": run_info.get("evalsuite"),
        "interactionMode": "MultiTurn",
        "runner": {"name": RUNNER_NAME, "version": RUNNER_VERSION},
        "createdAt": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "verdict": decision.verdict,
        "trials": settings.trials,
        "traceSource": settings.trace_source,
        "artifacts": {k: v for k, v in layout.artifact_uris().items()},
        "conversations": [{"caseId": t.case_id, "trial": t.trial, "sessionId": t.conversation.session_id,
                           "status": t.status, "uri": t.conversation_uri} for c in cases for t in c.trials],
        "counts": {"cases": len(cases), "trials": sum(len(c.trials) for c in cases),
                   "violations": sum(len(c.violations()) for c in cases)},
    }
