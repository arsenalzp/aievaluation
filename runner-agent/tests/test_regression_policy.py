"""apply_regression_policy of the single-turn agent runner (previously crashed with AttributeError on any baseline)."""
import json

from test_runner_agent_main import SINGLE_LINE, run_main, runner  # noqa: F401  (fixture re-use)


def agg(runner, **over):
    base = dict(total_cases=2, passed_cases=2, tool_selection_avg=0.9, tool_invocation_avg=0.9, tool_execution_avg=1.0,
                policy_violations_total=0, latency_ms_avg=10.0, tool_response_handling_avg=0.9, task_completion_avg=0.9)
    base.update(over)
    return runner.AgentAggregatedScores(**base)


def cfg(runner, max_drop=0.03):
    c = type("Cfg", (), {})()
    c.regression_max_drop = max_drop
    return c


def baseline_from(runner, a):
    """A baseline is a previous scores.json: aggregatedScores = asdict(AgentAggregatedScores)."""
    from dataclasses import asdict
    return {"aggregatedScores": asdict(a)}


def test_no_baseline(runner):
    r = runner.apply_regression_policy(cfg(runner), agg(runner), None)
    assert (r.baseline_used, r.failed, r.drops) == (False, False, {})


def test_within_max_drop_passes_and_reports_all_compared_metrics(runner):
    r = runner.apply_regression_policy(cfg(runner), agg(runner, task_completion_avg=0.88), baseline_from(runner, agg(runner)))
    assert r.baseline_used and not r.failed and r.failed_metrics == []
    assert set(r.drops) == {"tool_selection_avg", "tool_invocation_avg", "tool_execution_avg",
                            "tool_response_handling_avg", "task_completion_avg"}
    assert r.drops["task_completion_avg"] == -0.02


def test_drop_beyond_max_drop_fails_and_names_the_metric(runner):
    r = runner.apply_regression_policy(cfg(runner), agg(runner, tool_selection_avg=0.8, task_completion_avg=0.5),
                                       baseline_from(runner, agg(runner)))
    assert r.failed and r.failed_metrics == ["tool_selection_avg", "task_completion_avg"]


def test_improvement_never_fails(runner):
    r = runner.apply_regression_policy(cfg(runner), agg(runner, task_completion_avg=1.0), baseline_from(runner, agg(runner, task_completion_avg=0.5)))
    assert not r.failed and r.drops["task_completion_avg"] == 0.5


def test_metric_missing_in_either_run_is_skipped(runner):
    old = {"aggregatedScores": {"tool_selection_avg": 0.9}}  # e.g. baseline from before task completion existed
    r = runner.apply_regression_policy(cfg(runner), agg(runner, tool_invocation_avg=None), old)
    assert set(r.drops) == {"tool_selection_avg"} and not r.failed


def test_without_max_drop_drops_are_reported_but_never_fail(runner):
    r = runner.apply_regression_policy(cfg(runner, max_drop=None), agg(runner, task_completion_avg=0.0), baseline_from(runner, agg(runner)))
    assert not r.failed and r.drops["task_completion_avg"] == -0.9


def test_end_to_end_baseline_regression_fails_the_gate_with_metric_root_cause(runner, monkeypatch):
    class Agent:
        def __call__(self, url, json=None, headers=None, timeout=None):
            class R:
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"output": "done", "trace": {"traceId": "t", "toolCalls": [{"name": "search", "arguments": {}}]}}
            return R()

    class FakeAdapter:
        def __init__(self, model, llm): pass
        def score_records(self, records):
            return [runner.AgentPerCaseScore(case_id=r.case_id, tool_selection=0.5, tool_invocation=1.0,
                                             tool_execution=1.0, passed=True) for r in records], {"metricNames": []}

    monkeypatch.setattr(runner.requests, "post", Agent())
    monkeypatch.setattr(runner, "PhoenixAgentAdapter", FakeAdapter)
    store = runner.setup({"metrics": {"agent": {"maxPolicyViolations": 0}}, "regressionPolicy": {"maxDrop": 0.03}}, [SINGLE_LINE])
    monkeypatch.setattr(runner, "load_optional_baseline",
                        lambda cfg: {"aggregatedScores": {"tool_selection_avg": 0.9, "tool_invocation_avg": 1.0}})
    assert run_main(runner) == 0  # used to exit 1 with AttributeError
    gate = store["runs/run1/gate_summary.json"]
    assert gate["verdict"] == "FAIL"
    assert gate["rootCause"]["stage"] == "Regression" and gate["rootCause"]["metric"] == "tool_selection_avg"
    assert "tool_selection_avg drop=-0.4000" in gate["rootCause"]["details"]
    reg = store["runs/run1/scores.json"]["regression"]
    assert reg["baseline_used"] and reg["failed_metrics"] == ["tool_selection_avg"] and reg["drops"]["tool_invocation_avg"] == 0.0
