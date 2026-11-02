"""Stable JSON canonicalisation and partial argument matching."""
from __future__ import annotations

import json
from typing import Any, List, Tuple


def normalize_value(value: Any) -> Any:
    """Normalise a JSON-like value so that semantically equal values compare equal.

    * dict key order is irrelevant (handled by sort_keys at dump time)
    * keys whose value is ``None`` are dropped (missing == null)
    * surrounding whitespace in strings is stripped
    * floats with an integral value become ints (``1.0`` == ``1``)
    * tuples / sets become lists (sets are sorted by their canonical form)
    """
    if isinstance(value, dict):
        return {str(k): normalize_value(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((normalize_value(v) for v in value), key=canonical_json)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        return value.strip()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        normalize_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def normalize_tool_name(name: str) -> str:
    return (name or "").strip()


def tool_call_key(name: str, arguments: Any) -> str:
    """Identity of a tool call: normalised name + canonical JSON arguments."""
    return f"{normalize_tool_name(name)}|{canonical_json(arguments or {})}"


def values_equal(expected: Any, observed: Any) -> bool:
    """Equality used for expected-vs-observed values.

    Strings compare case-insensitively (``"London"`` == ``"london"``); everything
    else compares by canonical JSON.
    """
    exp = normalize_value(expected)
    obs = normalize_value(observed)
    if isinstance(exp, str) and isinstance(obs, str):
        return exp.casefold() == obs.casefold()
    return canonical_json(exp) == canonical_json(obs)


_MISSING = object()


def partial_match(expected: Any, observed: Any, path: str = "") -> Tuple[bool, List[dict]]:
    """Check that ``expected`` is a subset of ``observed``.

    Dicts are matched key by key (extra observed keys are fine), lists must have the
    same length and match element-wise, scalars use :func:`values_equal`.
    Returns ``(matched, mismatches)`` where each mismatch is
    ``{"path", "expected", "observed"}``.
    """
    mismatches: List[dict] = []
    _partial(expected, observed, path or "$", mismatches)
    return (not mismatches), mismatches


def _partial(expected: Any, observed: Any, path: str, out: List[dict]) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict):
            out.append({"path": path, "expected": expected, "observed": None if observed is _MISSING else observed})
            return
        for key, exp_val in expected.items():
            if key not in observed:
                out.append({"path": f"{path}.{key}", "expected": exp_val, "observed": None, "missing": True})
                continue
            _partial(exp_val, observed[key], f"{path}.{key}", out)
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(observed, (list, tuple)) or len(expected) != len(observed):
            out.append({"path": path, "expected": list(expected), "observed": observed})
            return
        for i, (e, o) in enumerate(zip(expected, observed)):
            _partial(e, o, f"{path}[{i}]", out)
        return
    if not values_equal(expected, observed):
        out.append({"path": path, "expected": expected, "observed": observed})
