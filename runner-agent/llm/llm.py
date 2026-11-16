from phoenix.evals import LLM

from multiturn.llm_settings import PROVIDER_OPENAI, LLMSettings


def build_phoenix_llm(settings: LLMSettings) -> LLM:
    """EvalRun.spec.llm -> phoenix-evals judge LLM.

    Provider openai: the OpenAI API, or any OpenAI-compatible server when a base URL is set.
    Building the LLM makes no request.
    """
    if settings.provider != PROVIDER_OPENAI:
        raise ValueError(f"the single-turn agent judges do not support LLM provider {settings.provider!r}")
    return LLM(provider="openai", model=settings.model, **settings.client_kwargs())
