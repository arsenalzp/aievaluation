"""Provider abstraction for optional semantic (LLM-judged) conversation metrics."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

from .config import EvaluatorConfig, MultiTurnSettings
from .models import Conversation, SemanticResult, SemanticScore


class SemanticProvider(ABC):
    """Scores a conversation on semantic metrics. Implementations must not leak vendor types."""

    name: str = "abstract"

    @property
    @abstractmethod
    def version(self) -> str:
        """Version of the underlying evaluator library (recorded in artifacts)."""

    @abstractmethod
    def evaluate(self, conversation: Conversation, metrics: List[str],
                 thresholds: Dict[str, Optional[float]], model: Optional[str], samples: int) -> List[SemanticScore]:
        """Return one :class:`SemanticScore` per metric. Raise on provider-level failure."""


ProviderFactory = Callable[[EvaluatorConfig], SemanticProvider]
_PROVIDERS: Dict[str, ProviderFactory] = {}


def register_semantic_provider(name: str, factory: ProviderFactory) -> None:
    _PROVIDERS[name] = factory


def get_semantic_provider(config: EvaluatorConfig, default_model: Optional[str] = None) -> SemanticProvider:
    try:
        factory = _PROVIDERS[config.provider]
    except KeyError:
        raise RuntimeError(f"no semantic provider registered for {config.provider!r}; "
                           f"available: {sorted(_PROVIDERS)}") from None
    return factory(config)


def run_semantic_evaluators(
    conversation: Conversation,
    settings: MultiTurnSettings,
    *,
    default_model: Optional[str] = None,
    provider_factory: Optional[ProviderFactory] = None,
    redact: Optional[Callable[[Conversation], Conversation]] = None,
) -> List[SemanticResult]:
    """Run every configured semantic evaluator; failures are captured, never raised.

    The conversation is passed through ``redact`` first so no secrets reach an external LLM.
    """
    results: List[SemanticResult] = []
    for cfg in settings.semantic_evaluators():
        result = SemanticResult(evaluator=cfg.name, provider=cfg.provider, required=cfg.required)
        try:
            provider = provider_factory(cfg) if provider_factory else get_semantic_provider(cfg)
            safe = redact(conversation) if redact else conversation
            metrics = settings.semantic_metrics()  # single source: thresholds in metrics.agent.conversation
            thresholds = {m: settings.semantic_threshold(m) for m in metrics}
            scores = provider.evaluate(safe, metrics, thresholds, cfg.model or default_model, cfg.samples)
            for s in scores:
                s.evaluator = cfg.name  # the configured evaluator name, not the library name
                s.version = s.version or provider.version
                if s.threshold is None:
                    s.threshold = thresholds.get(s.metric)
            result.scores = scores
            errors = [f"{s.metric}: {s.error}" for s in scores if s.error]
            if errors:
                result.error = "; ".join(errors)
        except Exception as exc:  # noqa: BLE001 - optional evaluators must not break deterministic evaluation
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results
