"""Multi-turn settings parsed from the rendered ``suite.json``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .phoenix_adapter import SOURCE_INLINE, TRACE_SOURCES

INTERACTION_SINGLE = "SingleTurn"
INTERACTION_MULTI = "MultiTurn"

PROVIDER_BUILTIN = "builtin"
PROVIDER_DEEPEVAL = "deepeval"
TYPE_DETERMINISTIC = "deterministic"
TYPE_MULTI_TURN = "multiTurn"

SEMANTIC_METRICS = ("knowledgeRetention", "conversationCompleteness", "turnRelevancy")
BUILTIN_EVALUATOR_NAME = "trajectory-policy"

LIMITS_FAIL = "Fail"
LIMITS_WARN = "Warn"


class ConfigError(ValueError):
    pass


@dataclass
class EvaluatorConfig:
    name: str
    provider: str = PROVIDER_BUILTIN
    type: str = TYPE_DETERMINISTIC
    required: bool = False
    metrics: List[str] = field(default_factory=list)
    model: Optional[str] = None
    samples: int = 1


@dataclass
class MultiTurnSettings:
    interaction_mode: str = INTERACTION_SINGLE
    trials: int = 1
    trace_source: str = SOURCE_INLINE
    phoenix_endpoint: Optional[str] = None
    phoenix_project: str = "default"
    evaluators: List[EvaluatorConfig] = field(default_factory=list)

    max_turns: Optional[int] = None
    max_tool_calls: Optional[int] = None
    max_repeated_calls: Optional[int] = None
    limits_action: str = LIMITS_FAIL

    min_goal_completion: Optional[float] = None
    min_selection_accuracy: Optional[float] = None
    min_argument_accuracy: Optional[float] = None
    min_knowledge_retention: Optional[float] = None
    min_conversation_completeness: Optional[float] = None
    min_turn_relevancy: Optional[float] = None

    max_critical_violations: int = 0
    max_high_violations: int = 0
    regression_max_drop: Optional[float] = None
    timeout_seconds: int = 60

    @property
    def is_multi_turn(self) -> bool:
        return self.interaction_mode == INTERACTION_MULTI

    def semantic_threshold(self, metric: str) -> Optional[float]:
        return {
            "knowledgeRetention": self.min_knowledge_retention,
            "conversationCompleteness": self.min_conversation_completeness,
            "turnRelevancy": self.min_turn_relevancy,
        }.get(metric)

    def builtin_evaluator(self) -> EvaluatorConfig:
        for ev in self.evaluators:
            if ev.provider == PROVIDER_BUILTIN:
                return ev
        # Deterministic checks always run; with no explicit entry they are required.
        return EvaluatorConfig(name=BUILTIN_EVALUATOR_NAME, required=True)

    def semantic_evaluators(self) -> List[EvaluatorConfig]:
        return [e for e in self.evaluators if e.provider != PROVIDER_BUILTIN]


def get_interaction_mode(suite: Dict[str, Any]) -> str:
    evaluation = suite.get("evaluation") or {}
    mode = evaluation.get("interactionMode") or INTERACTION_SINGLE
    if mode not in (INTERACTION_SINGLE, INTERACTION_MULTI):
        raise ConfigError(f"evaluation.interactionMode must be {INTERACTION_SINGLE} or {INTERACTION_MULTI}, got {mode!r}")
    return mode


def _num(d: Dict[str, Any], key: str, path: str, lo: float, hi: Optional[float] = None) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{path}.{key} must be a number, got {v!r}")
    if v < lo or (hi is not None and v > hi):
        raise ConfigError(f"{path}.{key}={v} is out of range [{lo}, {hi if hi is not None else 'inf'}]")
    return float(v)


def _int(d: Dict[str, Any], key: str, path: str, lo: int) -> Optional[int]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ConfigError(f"{path}.{key} must be an integer >= {lo}, got {v!r}")
    return v


def parse_multiturn_settings(suite: Dict[str, Any]) -> MultiTurnSettings:
    evaluation = suite.get("evaluation") or {}
    agent = ((suite.get("metrics") or {}).get("agent")) or {}
    outcome = agent.get("outcome") or {}
    conversation = agent.get("conversation") or {}
    tools = agent.get("tools") or {}
    policy = agent.get("policy") or {}
    execution = suite.get("execution") or {}
    regression = suite.get("regressionPolicy") or {}

    s = MultiTurnSettings(interaction_mode=get_interaction_mode(suite))

    trials = _int(evaluation, "trials", "evaluation", 1)
    s.trials = trials if trials is not None else 1
    if s.trials > 10:
        raise ConfigError("evaluation.trials must be <= 10")

    trace = evaluation.get("trace") or {}
    s.trace_source = trace.get("source") or SOURCE_INLINE
    if s.trace_source not in TRACE_SOURCES:
        raise ConfigError(f"evaluation.trace.source must be one of {TRACE_SOURCES}, got {s.trace_source!r}")
    phoenix = trace.get("phoenix") or {}
    s.phoenix_endpoint = phoenix.get("endpoint") or None
    s.phoenix_project = phoenix.get("project") or "default"
    if s.trace_source != SOURCE_INLINE and not s.phoenix_endpoint:
        raise ConfigError(f"evaluation.trace.phoenix.endpoint is required when evaluation.trace.source={s.trace_source}")

    s.min_goal_completion = _num(outcome, "minGoalCompletion", "metrics.agent.outcome", 0, 1)
    s.max_turns = _int(conversation, "maxTurns", "metrics.agent.conversation", 1)
    s.min_knowledge_retention = _num(conversation, "minKnowledgeRetention", "metrics.agent.conversation", 0, 1)
    s.min_conversation_completeness = _num(conversation, "minConversationCompleteness", "metrics.agent.conversation", 0, 1)
    s.min_turn_relevancy = _num(conversation, "minTurnRelevancy", "metrics.agent.conversation", 0, 1)

    sel = _num(tools, "minSelectionAccuracy", "metrics.agent.tools", 0, 1)
    if sel is None:  # backward compatible flat field of the single-turn schema
        sel = _num(agent, "minToolSelection", "metrics.agent", 0, 1)
    s.min_selection_accuracy = sel
    s.min_argument_accuracy = _num(tools, "minArgumentAccuracy", "metrics.agent.tools", 0, 1)
    s.max_tool_calls = _int(tools, "maxToolCalls", "metrics.agent.tools", 1)
    s.max_repeated_calls = _int(tools, "maxRepeatedCalls", "metrics.agent.tools", 1)

    crit = _int(policy, "maxCriticalViolations", "metrics.agent.policy", 0)
    high = _int(policy, "maxHighViolations", "metrics.agent.policy", 0)
    s.max_critical_violations = crit if crit is not None else 0
    s.max_high_violations = high if high is not None else 0
    s.limits_action = policy.get("limitsAction") or LIMITS_FAIL
    if s.limits_action not in (LIMITS_FAIL, LIMITS_WARN):
        raise ConfigError(f"metrics.agent.policy.limitsAction must be {LIMITS_FAIL} or {LIMITS_WARN}")

    s.regression_max_drop = _num(regression, "maxDrop", "regressionPolicy", 0, 1)
    ts = execution.get("timeoutSeconds")
    s.timeout_seconds = int(ts) if isinstance(ts, (int, float)) and ts > 0 else 60

    seen = set()
    for i, raw in enumerate(suite.get("evaluators") or []):
        p = f"evaluators[{i}]"
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ConfigError(f"{p}.name is required")
        if raw["name"] in seen:
            raise ConfigError(f"{p}.name {raw['name']!r} is duplicated")
        seen.add(raw["name"])
        provider = raw.get("provider") or PROVIDER_BUILTIN
        if provider not in (PROVIDER_BUILTIN, PROVIDER_DEEPEVAL):
            raise ConfigError(f"{p}.provider must be {PROVIDER_BUILTIN} or {PROVIDER_DEEPEVAL}, got {provider!r}")
        etype = raw.get("type") or (TYPE_DETERMINISTIC if provider == PROVIDER_BUILTIN else TYPE_MULTI_TURN)
        if provider == PROVIDER_BUILTIN and etype != TYPE_DETERMINISTIC:
            raise ConfigError(f"{p}: builtin evaluators are deterministic")
        metrics = list(raw.get("metrics") or [])
        if provider != PROVIDER_BUILTIN and not metrics:
            metrics = list(SEMANTIC_METRICS)
        for m in metrics:
            if m not in SEMANTIC_METRICS:
                raise ConfigError(f"{p}.metrics: unknown metric {m!r}; supported: {list(SEMANTIC_METRICS)}")
        samples = _int(raw, "samples", p, 1) or 1
        if samples > 5:
            raise ConfigError(f"{p}.samples must be <= 5")
        s.evaluators.append(EvaluatorConfig(
            name=raw["name"], provider=provider, type=etype, required=bool(raw.get("required", False)),
            metrics=metrics, model=raw.get("model") or None, samples=samples,
        ))
    return s
