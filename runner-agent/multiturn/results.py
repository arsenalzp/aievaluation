"""Result containers shared by the runner, the gate inputs and the artifact writers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .models import CheckResult, Conversation, SemanticResult, Violation


@dataclass
class TrialResult:
    case_id: str
    trial: int
    conversation: Conversation
    checks: List[CheckResult] = field(default_factory=list)
    violations: List[Violation] = field(default_factory=list)
    semantic: List[SemanticResult] = field(default_factory=list)
    goal_completed: bool = False
    tool_selection: Optional[float] = None
    tool_arguments: Optional[float] = None
    status: str = "FAILED"  # PASSED | FAILED | ERROR
    error: Optional[str] = None
    conversation_key: str = ""
    conversation_uri: str = ""

    def check(self, name: str) -> Optional[CheckResult]:
        return next((c for c in self.checks if c.check == name), None)


@dataclass
class CaseResult:
    case_id: str
    trials: List[TrialResult] = field(default_factory=list)

    @property
    def pass_all(self) -> bool:
        return bool(self.trials) and all(t.goal_completed for t in self.trials)

    @property
    def pass_rate(self) -> float:
        return sum(1 for t in self.trials if t.goal_completed) / len(self.trials) if self.trials else 0.0

    def violations(self) -> List[Violation]:
        return [v for t in self.trials for v in t.violations]

    def first_trial(self) -> TrialResult:
        return self.trials[0]


def mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


@dataclass
class RunResult:
    cases: List[CaseResult]
    aggregates: Dict[str, Optional[float]]
    verdict: str
    summary: str
    root_cause: Dict[str, Any]
    findings: List[Dict[str, Any]]
    artifacts: Dict[str, str]
    regression: Dict[str, Any]
