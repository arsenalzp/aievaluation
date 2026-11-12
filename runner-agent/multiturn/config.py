"""Multi-turn settings parsed from the rendered ``suite.json``."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .phoenix_adapter import SOURCE_INLINE, TRACE_SOURCES

INTERACTION_SINGLE = "SingleTurn"
INTERACTION_MULTI = "MultiTurn"

PROVIDER_BUILTIN = "builtin"
PROVIDER_DEEPEVAL = "deepeval"
TYPE_DETERMINISTIC = "deterministic"
TYPE_MULTI_TURN = "multiTurn"

SEMANTIC_METRICS = ("knowledgeRetention", "conversationCompleteness", "turnRelevancy")
# The single source of the semantic metrics: a metric is evaluated iff its threshold is set here.
SEMANTIC_THRESHOLD_FIELDS = {
    "knowledgeRetention": "minKnowledgeRetention",
    "conversationCompleteness": "minConversationCompleteness",
    "turnRelevancy": "minTurnRelevancy",
}
EVALUATOR_FIELDS = ("name", "provider", "type", "required", "model", "samples")
# Flat single-turn fields the multi-turn runner does not read. No synonyms or fallbacks: every metric
# has exactly one field.
SINGLE_TURN_ONLY_AGENT_FIELDS = {
    "minToolSelection": "use metrics.agent.tools.minSelectionAccuracy",
    "minToolInvocation": "use metrics.agent.tools.minArgumentAccuracy",
    "minToolExecution": "multi-turn tool errors are reported as violations (metrics.agent.policy)",
    "maxPolicyViolations": "use metrics.agent.policy.maxCriticalViolations / maxHighViolations",
    "minToolResponseHandling": "multi-turn suites have no tool-response judge",
    "minTaskCompletion": "use metrics.agent.outcome.minGoalCompletion",
}
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

    def semantic_metrics(self) -> List[str]:
        """Semantic metrics to evaluate: exactly those with a threshold in metrics.agent.conversation."""
        return [m for m in SEMANTIC_METRICS if self.semantic_threshold(m) is not None]

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


def section(parent: Dict[str, Any], key: str, path: str, allowed: Tuple[str, ...]) -> Dict[str, Any]:
    """Return ``parent[key]`` as an object (``{}`` when absent) after rejecting unknown fields.

    Rule: a configured field that nothing reads is an error, never a silent no-op.
    """
    value = parent.get(key)
    full = f"{path}.{key}" if path else key
    if value is None:
        return {}
    reject_unknown(value, allowed, full)
    return value


def reject_unknown(d: Any, allowed: Tuple[str, ...], path: str) -> None:
    if not isinstance(d, dict):
        raise ConfigError(f"{path} must be an object, got {type(d).__name__}")
    unknown = sorted(set(d) - set(allowed))
    if unknown:
        raise ConfigError(f"{path}: unknown field(s) {unknown}; supported: {sorted(allowed)}")


def number_field(d: Dict[str, Any], key: str, path: str, lo: float, hi: Optional[float] = None) -> Optional[float]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{path}.{key} must be a number, got {v!r}")
    if v < lo or (hi is not None and v > hi):
        raise ConfigError(f"{path}.{key}={v} is out of range [{lo}, {hi if hi is not None else 'inf'}]")
    return float(v)


def int_field(d: Dict[str, Any], key: str, path: str, lo: int) -> Optional[int]:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ConfigError(f"{path}.{key} must be an integer >= {lo}, got {v!r}")
    return v


def parse_multiturn_settings(suite: Dict[str, Any]) -> MultiTurnSettings:
    # Every key the controller can render is listed; anything else is rejected (nothing is silently ignored).
    reject_unknown(suite, ("metrics", "regressionPolicy", "execution", "evaluation", "evaluators"), "suite")
    evaluation = section(suite, "evaluation", "", ("interactionMode", "trials", "trace"))
    metrics = section(suite, "metrics", "", ("agent",))
    agent = section(metrics, "agent", "metrics",
                    ("outcome", "conversation", "tools", "policy", *SINGLE_TURN_ONLY_AGENT_FIELDS))
    outcome = section(agent, "outcome", "metrics.agent", ("minGoalCompletion",))
    conversation = section(agent, "conversation", "metrics.agent", ("maxTurns", *SEMANTIC_THRESHOLD_FIELDS.values()))
    tools = section(agent, "tools", "metrics.agent",
                    ("minSelectionAccuracy", "minArgumentAccuracy", "maxToolCalls", "maxRepeatedCalls"))
    policy = section(agent, "policy", "metrics.agent", ("maxCriticalViolations", "maxHighViolations", "limitsAction"))
    execution = section(suite, "execution", "", ("mode", "timeoutSeconds"))
    regression = section(suite, "regressionPolicy", "", ("maxDrop",))

    s = MultiTurnSettings(interaction_mode=get_interaction_mode(suite))

    trials = int_field(evaluation, "trials", "evaluation", 1)
    s.trials = trials if trials is not None else 1
    if s.trials > 10:
        raise ConfigError("evaluation.trials must be <= 10")

    trace = section(evaluation, "trace", "evaluation", ("source", "phoenix"))
    s.trace_source = trace.get("source") or SOURCE_INLINE
    if s.trace_source not in TRACE_SOURCES:
        raise ConfigError(f"evaluation.trace.source must be one of {TRACE_SOURCES}, got {s.trace_source!r}")
    phoenix = section(trace, "phoenix", "evaluation.trace", ("endpoint", "project"))
    s.phoenix_endpoint = phoenix.get("endpoint") or None
    s.phoenix_project = phoenix.get("project") or "default"
    if s.trace_source != SOURCE_INLINE and not s.phoenix_endpoint:
        raise ConfigError(f"evaluation.trace.phoenix.endpoint is required when evaluation.trace.source={s.trace_source}")

    s.min_goal_completion = number_field(outcome, "minGoalCompletion", "metrics.agent.outcome", 0, 1)
    s.max_turns = int_field(conversation, "maxTurns", "metrics.agent.conversation", 1)
    s.min_knowledge_retention = number_field(conversation, "minKnowledgeRetention", "metrics.agent.conversation", 0, 1)
    s.min_conversation_completeness = number_field(conversation, "minConversationCompleteness", "metrics.agent.conversation", 0, 1)
    s.min_turn_relevancy = number_field(conversation, "minTurnRelevancy", "metrics.agent.conversation", 0, 1)

    s.min_selection_accuracy = number_field(tools, "minSelectionAccuracy", "metrics.agent.tools", 0, 1)
    s.min_argument_accuracy = number_field(tools, "minArgumentAccuracy", "metrics.agent.tools", 0, 1)
    s.max_tool_calls = int_field(tools, "maxToolCalls", "metrics.agent.tools", 1)
    s.max_repeated_calls = int_field(tools, "maxRepeatedCalls", "metrics.agent.tools", 1)

    crit = int_field(policy, "maxCriticalViolations", "metrics.agent.policy", 0)
    high = int_field(policy, "maxHighViolations", "metrics.agent.policy", 0)
    s.max_critical_violations = crit if crit is not None else 0
    s.max_high_violations = high if high is not None else 0
    s.limits_action = policy.get("limitsAction") or LIMITS_FAIL
    if s.limits_action not in (LIMITS_FAIL, LIMITS_WARN):
        raise ConfigError(f"metrics.agent.policy.limitsAction must be {LIMITS_FAIL} or {LIMITS_WARN}")

    s.regression_max_drop = number_field(regression, "maxDrop", "regressionPolicy", 0, 1)
    ts = int_field(execution, "timeoutSeconds", "execution", 1)
    s.timeout_seconds = ts if ts is not None else 60

    if s.is_multi_turn:
        for key, hint in SINGLE_TURN_ONLY_AGENT_FIELDS.items():
            if agent.get(key) is not None:
                raise ConfigError(f"metrics.agent.{key} applies to single-turn agent suites; {hint}")

    seen = set()
    by_provider: Dict[str, int] = {}
    evaluators = suite.get("evaluators") or []
    if not isinstance(evaluators, list):
        raise ConfigError(f"evaluators must be a list, got {type(evaluators).__name__}")
    for i, raw in enumerate(evaluators):
        p = f"evaluators[{i}]"
        if not isinstance(raw, dict) or not raw.get("name"):
            raise ConfigError(f"{p}.name is required")
        unknown = sorted(set(raw) - set(EVALUATOR_FIELDS))
        if unknown:
            hint = (" (semantic metrics are selected by their thresholds in metrics.agent.conversation)"
                    if "metrics" in unknown else "")
            raise ConfigError(f"{p}: unknown field(s) {unknown}; supported: {list(EVALUATOR_FIELDS)}{hint}")
        if raw["name"] in seen:
            raise ConfigError(f"{p}.name {raw['name']!r} is duplicated")
        seen.add(raw["name"])
        provider = raw.get("provider") or PROVIDER_BUILTIN
        if provider not in (PROVIDER_BUILTIN, PROVIDER_DEEPEVAL):
            raise ConfigError(f"{p}.provider must be {PROVIDER_BUILTIN} or {PROVIDER_DEEPEVAL}, got {provider!r}")
        if provider in by_provider:
            raise ConfigError(f"{p}: only one evaluator with provider {provider} is allowed "
                              f"(evaluators[{by_provider[provider]}] is one already)")
        by_provider[provider] = i
        etype = raw.get("type") or (TYPE_DETERMINISTIC if provider == PROVIDER_BUILTIN else TYPE_MULTI_TURN)
        required = raw.get("required")
        if required is not None and not isinstance(required, bool):
            raise ConfigError(f"{p}.required must be a boolean, got {required!r}")
        if provider == PROVIDER_BUILTIN:
            if etype != TYPE_DETERMINISTIC:
                raise ConfigError(f"{p}: builtin evaluators are deterministic")
            if required is False:
                raise ConfigError(f"{p}: the builtin evaluator is always required; remove required: false")
            if raw.get("model") is not None or raw.get("samples") is not None:
                raise ConfigError(f"{p}: model and samples apply only to provider {PROVIDER_DEEPEVAL}; "
                                  "the builtin evaluator uses no judge")
            s.evaluators.append(EvaluatorConfig(name=raw["name"], provider=provider, type=etype, required=True))
            continue
        if etype != TYPE_MULTI_TURN:
            raise ConfigError(f"{p}: provider {provider} only supports type {TYPE_MULTI_TURN}")
        samples = int_field(raw, "samples", p, 1) or 1
        if samples > 5:
            raise ConfigError(f"{p}.samples must be <= 5")
        s.evaluators.append(EvaluatorConfig(
            name=raw["name"], provider=provider, type=etype, required=bool(required),
            model=raw.get("model") or None, samples=samples,
        ))

    if s.evaluators and not s.is_multi_turn:
        raise ConfigError("evaluators is only supported with evaluation.interactionMode: MultiTurn")
    thresholds = [SEMANTIC_THRESHOLD_FIELDS[m] for m in s.semantic_metrics()]
    if thresholds and PROVIDER_DEEPEVAL not in by_provider:
        raise ConfigError(f"metrics.agent.conversation.{', '.join(thresholds)} is set but no evaluator computes it; "
                          f"add an evaluators entry with provider: {PROVIDER_DEEPEVAL} or remove the threshold")
    if not thresholds and PROVIDER_DEEPEVAL in by_provider:
        raise ConfigError(f"evaluators[{by_provider[PROVIDER_DEEPEVAL]}] (provider {PROVIDER_DEEPEVAL}) has nothing to "
                          f"evaluate; set at least one of metrics.agent.conversation."
                          f"{', '.join(SEMANTIC_THRESHOLD_FIELDS.values())} or remove the evaluator")
    return s
