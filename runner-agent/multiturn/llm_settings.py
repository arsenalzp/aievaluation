"""LLM connection settings of a run (EvalRun.spec.llm) and the start-of-run capability probe.

The controller renders EvalRun.spec.llm into the env contract:

``LLM_PROVIDER``     provider protocol; only ``openai`` (the OpenAI API or any OpenAI-compatible server)
``OPENAI_MODEL``     judge model
``OPENAI_EMBEDDER``  embedding model (RAG only; the agent runner rejects it)
``OPENAI_BASE_URL``  optional base URL of an OpenAI-compatible server
``OPENAI_API_KEY``   optional with a base URL (server without authentication), required without one

This module is stdlib only; the probe imports the OpenAI SDK lazily.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from .artifacts import redact
from .config import ConfigError

PROVIDER_OPENAI = "openai"
LLM_PROVIDERS = (PROVIDER_OPENAI,)

# The OpenAI SDK refuses to build a client without an API key. A server without authentication ignores the
# Authorization header, so this placeholder is sent only when spec.llm.url is set and no key is configured.
NO_API_KEY = "no-api-key"

LLM_ENV = ("LLM_PROVIDER", "OPENAI_MODEL", "OPENAI_EMBEDDER", "OPENAI_BASE_URL", "OPENAI_API_KEY")


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    @property
    def client_api_key(self) -> str:
        return self.api_key or NO_API_KEY

    @property
    def endpoint(self) -> str:
        """Where the judge runs, for logs and artifacts (never contains the key)."""
        return self.base_url or f"{self.provider} public API"

    def client_kwargs(self) -> dict:
        """Keyword arguments for an OpenAI SDK client (``OpenAI(**kwargs)`` / ``AsyncOpenAI(**kwargs)``)."""
        kwargs = {"api_key": self.client_api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return kwargs


def llm_settings_from_env(env: Mapping[str, str], *, required: bool, user: str) -> Optional[LLMSettings]:
    """Read and validate the LLM env contract.

    ``required`` says whether the run calls an LLM judge; ``user`` names what calls it (or why nothing does)
    for the error messages. A configured LLM that nothing uses is an error, like any unused configuration.
    """
    def get(name: str) -> Optional[str]:
        value = env.get(name)
        return value.strip() if isinstance(value, str) and value.strip() else None

    provider, model = get("LLM_PROVIDER"), get("OPENAI_MODEL")
    base_url, api_key, embedder = get("OPENAI_BASE_URL"), get("OPENAI_API_KEY"), get("OPENAI_EMBEDDER")

    if not required:
        if provider or model or embedder:
            raise ConfigError(f"an LLM is configured (EvalRun.spec.llm) but nothing uses it: {user} calls no LLM judge; "
                              "remove spec.llm")
        return None
    if provider is None:
        raise ConfigError(f"{user} calls an LLM judge but no LLM is configured: set EvalRun.spec.llm "
                          "(env LLM_PROVIDER, OPENAI_MODEL and OPENAI_API_KEY or OPENAI_BASE_URL)")
    if provider not in LLM_PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER (EvalRun.spec.llm.provider) must be one of {list(LLM_PROVIDERS)}, got {provider!r}")
    if model is None:
        raise ConfigError("OPENAI_MODEL (EvalRun.spec.llm.model) is required")
    if embedder is not None:
        raise ConfigError("OPENAI_EMBEDDER (EvalRun.spec.llm.embedder) is set but nothing uses it: "
                          "only RAG suites use an embedding model; remove spec.llm.embedder")
    if base_url is not None and not base_url.startswith(("http://", "https://")):
        raise ConfigError(f"OPENAI_BASE_URL (EvalRun.spec.llm.url) must start with http:// or https://, got {base_url!r}")
    if base_url is None and api_key is None:
        raise ConfigError(f"OPENAI_API_KEY (EvalRun.spec.llm.apiKeySecretRef) is required without "
                          f"OPENAI_BASE_URL (spec.llm.url): the public {provider} API needs an API key")
    return LLMSettings(provider=provider, model=model, base_url=base_url, api_key=api_key)


# A probe gets the settings and raises on failure. Injected in tests (no network).
LLMProbe = Callable[[LLMSettings], None]


def probe_llm(settings: LLMSettings, client_factory: Optional[Callable[..., Any]] = None) -> None:
    """Fail fast with an actionable ConfigError when the judge LLM cannot answer.

    One short chat completion checks the URL, the API key and the model name before any case is evaluated,
    instead of a judge error in every case. Library-specific features (tool calling, JSON modes) are not
    probed: their failures surface as evaluator errors of the first case.
    """
    if client_factory is None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - the runner image installs openai
            raise ConfigError(f"the openai package is required for provider {settings.provider}: {exc}") from exc
        client_factory = OpenAI
    try:
        client = client_factory(max_retries=1, timeout=60, **settings.client_kwargs())
        client.chat.completions.create(model=settings.model,
                                       messages=[{"role": "user", "content": "Reply with the single word OK."}])
    except Exception as exc:  # noqa: BLE001 - every failure is a configuration problem of the run
        message = str(exc).replace(settings.api_key, "<redacted>") if settings.api_key else str(exc)
        raise ConfigError(f"the judge LLM is not usable: model {settings.model!r} at {settings.endpoint} "
                          f"(EvalRun.spec.llm) failed a test request: {type(exc).__name__}: {redact(message)[:500]}") from exc
