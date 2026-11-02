"""Glue between the Job environment (runner-agent.py) and :func:`multiturn.runner.evaluate_run`."""
from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from . import deepeval_provider  # noqa: F401  (registers the optional "deepeval" provider; imports lazily)
from .artifacts import ArtifactLayout, Uploader
from .config import parse_multiturn_settings
from .dataset import parse_multiturn_dataset
from .phoenix_adapter import PhoenixRestFetcher, SOURCE_INLINE
from .results import RunResult
from .runner import evaluate_run
from .semantic import ProviderFactory
from .target import create_target_adapter


def run_multiturn_job(
    *,
    suite: Dict[str, Any],
    env: Mapping[str, str],
    dataset_path: str,
    bucket: str,
    gate_summary_key: str,
    scores_key: str,
    details_key: str,
    target_endpoint: str,
    uploader: Uploader,
    baseline: Optional[Dict[str, Any]] = None,
    llm_model: Optional[str] = None,
    post: Optional[Callable[..., Any]] = None,
    get: Optional[Callable[..., Any]] = None,
    provider_factory: Optional[ProviderFactory] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> RunResult:
    settings = parse_multiturn_settings(suite)
    cases = parse_multiturn_dataset(dataset_path)
    if not cases:
        raise RuntimeError("dataset is empty")

    adapter = create_target_adapter("stateless", endpoint=target_endpoint,
                                    timeout_seconds=settings.timeout_seconds, post=post)
    fetcher = None
    if settings.trace_source != SOURCE_INLINE:
        fetcher = PhoenixRestFetcher(settings.phoenix_endpoint or "", settings.phoenix_project, get=get)

    layout = ArtifactLayout.from_env(bucket, gate_summary_key, scores_key, details_key, env)
    run_info = {
        "evalrun": env.get("EVALRUN_NAME", ""),
        "evalsuite": env.get("EVALSUITE_NAME", ""),
        "gitSha": env.get("GIT_SHA") or None,
        "prNumber": env.get("PR_NUMBER") or None,
    }
    kwargs: Dict[str, Any] = {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    result = evaluate_run(settings=settings, cases=cases, adapter=adapter, uploader=uploader, layout=layout,
                          run_info=run_info, baseline=baseline, default_model=llm_model, phoenix_fetcher=fetcher,
                          provider_factory=provider_factory, **kwargs)
    print(f"multi-turn evaluation finished: verdict={result.verdict} cases={len(result.cases)} :: {result.summary}")
    return result
