"""main() of runner-agent.py: multi-turn dispatch, dataset/mode guard, single-turn backward compatibility."""
import asyncio
import importlib.util
import json
import os
import shutil

import pytest

from fake_travel_agent import FakeTravelAgent
from conftest import suite_dict, travel_case  # noqa: F401

RUNNER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "runner-agent.py")

MULTI_LINE = json.dumps({
    "caseId": "travel-001", "type": "multiTurn",
    "turns": [{"role": "user", "content": "Find me a flight from NYC to Paris"},
              {"role": "user", "content": "I want to leave on June 5"},
              {"role": "user", "content": "Actually, change the destination to London"}],
    "expectedBehavior": {"requiredTools": ["search_flights"],
                         "expectedToolCalls": [{"name": "search_flights", "turn": 3, "arguments": {"destination": "London"}}]}})
SINGLE_LINE = json.dumps({"caseId": "st-1", "input": "hello", "expectedTools": [{"name": "search"}]})


@pytest.fixture
def runner(monkeypatch, tmp_path):
    pytest.importorskip("phoenix.evals", reason="runner-agent.py needs the agent runner dependencies (make agent-runner-install-dev)")
    env = {"OPENAI_MODEL": "gpt-test", "OPENAI_API_KEY": "sk-test", "EVALRUN_NAME": "run1", "EVALSUITE_NAME": "suite1",
           "TARGET_ENDPOINT": "http://agent/turn", "S3_ENDPOINT": "http://s3", "DATASET_S3_BUCKET": "bucket",
           "S3_ACCESS_KEY": "a", "S3_SECRET_KEY": "b", "DATASET_S3_KEY": "ds.jsonl", "S3_BASELINE_KEY": "baselines/s.json",
           "GATE_SUMMARY_KEY": "runs/run1/gate_summary.json", "SCORES_KEY": "runs/run1/scores.json",
           "DETAILS_KEY": "runs/run1/details.jsonl", "S3_RUN_PREFIX": "runs/run1", "WORK_DIR": str(tmp_path),
           "DATASET_FILE": str(tmp_path / "dataset"), "SUITE_CONFIG": str(tmp_path / "suite.json")}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    spec = importlib.util.spec_from_file_location("runner_agent_under_test", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    store = {}
    monkeypatch.setattr(mod, "upload_json", lambda cfg, key, payload: store.__setitem__(key, json.loads(json.dumps(payload))) or {})
    monkeypatch.setattr(mod, "upload_text", lambda cfg, key, text, content_type="text/plain": store.__setitem__(key, text) or {})
    monkeypatch.setattr(mod, "load_optional_baseline", lambda cfg: None)

    def setup(suite, dataset_lines):
        (tmp_path / "suite.json").write_text(json.dumps(suite))
        src = tmp_path / "src.jsonl"
        src.write_text("\n".join(dataset_lines) + "\n")
        monkeypatch.setattr(mod, "download_dataset", lambda cfg: shutil.copy(src, cfg.dataset_file))
        return store

    mod.setup, mod.store = setup, store
    return mod


def run_main(mod):
    return asyncio.run(mod.main())


def test_multi_turn_suite_runs_the_multi_turn_path(runner, monkeypatch):
    agent = FakeTravelAgent()
    monkeypatch.setattr(runner.requests, "post", agent.post)
    store = runner.setup(suite_dict(), [MULTI_LINE])
    assert run_main(runner) == 0
    assert set(store) == {"runs/run1/gate_summary.json", "runs/run1/scores.json", "runs/run1/details.jsonl",
                          "runs/run1/manifest.json", "runs/run1/violations.jsonl",
                          "runs/run1/conversations/travel-001.json"}
    gate = store["runs/run1/gate_summary.json"]
    assert gate["verdict"] == "PASS" and gate["interactionMode"] == "MultiTurn"
    # same URI form as the RAG runner: {S3_ENDPOINT}/buckets/{bucket}/objects/{key}
    assert gate["artifacts"]["gateSummaryUri"] == "http://s3/buckets/bucket/objects/runs/run1/gate_summary.json"
    assert gate["artifacts"]["violationsUri"] == "http://s3/buckets/bucket/objects/runs/run1/violations.jsonl"
    assert gate["rootCause"] == {"stage": "None"}
    assert {r["sessionId"] for r in agent.requests} == {"evalrun-run1-travel-001"}


def test_failing_multi_turn_run_still_exits_zero_and_reports_fail_in_the_gate(runner, monkeypatch):
    monkeypatch.setattr(runner.requests, "post", FakeTravelAgent(["stale_destination"]).post)
    store = runner.setup(suite_dict(), [MULTI_LINE])
    assert run_main(runner) == 0
    gate = store["runs/run1/gate_summary.json"]
    assert gate["verdict"] == "FAIL" and gate["rootCause"]["caseId"] == "travel-001"


def test_malformed_multi_turn_dataset_publishes_a_dataset_failure(runner):
    bad = json.dumps({"caseId": "broken-1", "type": "multiTurn", "turns": [{"role": "assistant", "content": "x"}]})
    store = runner.setup(suite_dict(), [bad])
    assert run_main(runner) == 1
    gate = store["runs/run1/gate_summary.json"]
    assert gate["verdict"] == "FAIL"
    assert gate["rootCause"]["stage"] == "Dataset" and gate["rootCause"]["caseId"] == "broken-1"
    assert gate["artifacts"]["gateSummaryUri"] == "http://s3/buckets/bucket/objects/runs/run1/gate_summary.json"
    assert "turns[0].role" in gate["rootCause"]["details"]


def test_multi_turn_case_with_single_turn_suite_is_a_clear_error(runner):
    store = runner.setup({"metrics": {"agent": {"minToolSelection": 0.9}}}, [MULTI_LINE])
    assert run_main(runner) == 1
    rc = store["runs/run1/gate_summary.json"]["rootCause"]
    assert rc["stage"] == "Dataset" and "interactionMode" in rc["details"] and rc["caseId"] == "travel-001"


def test_single_turn_case_with_multi_turn_suite_is_a_clear_error(runner):
    store = runner.setup(suite_dict(), [SINGLE_LINE])
    assert run_main(runner) == 1
    rc = store["runs/run1/gate_summary.json"]["rootCause"]
    assert rc["stage"] == "Dataset" and rc["caseId"] == "st-1"


def test_invalid_multi_turn_configuration_is_reported_as_configuration_error(runner):
    store = runner.setup(suite_dict(**{"evaluation.trials": 99}), [MULTI_LINE])
    assert run_main(runner) == 1
    assert store["runs/run1/gate_summary.json"]["rootCause"]["stage"] == "Configuration"


def test_single_turn_flow_is_unchanged(runner, monkeypatch):
    """Existing suites/datasets keep the old pipeline and produce exactly the old three artifacts."""
    class FakeSingleTurnAgent:
        def __call__(self, url, json=None, headers=None, timeout=None):
            class R:
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"output": "done", "trace": {"traceId": "t", "toolCalls": [{"name": "search", "arguments": {}}]}}
            return R()

    class FakeAdapter:
        def __init__(self, model, llm): pass
        def score_records(self, records):
            return [runner.AgentPerCaseScore(case_id=r.case_id, tool_selection=1.0, tool_invocation=1.0,
                                             tool_execution=1.0, passed=True) for r in records], {"metricNames": []}

    monkeypatch.setattr(runner.requests, "post", FakeSingleTurnAgent())
    monkeypatch.setattr(runner, "PhoenixAgentAdapter", FakeAdapter)
    legacy_suite = {"metrics": {"agent": {"minToolSelection": 0.9, "minToolInvocation": 0.9, "minToolExecution": 0.85,
                                          "maxPolicyViolations": 0}},
                    "regressionPolicy": {"maxDrop": 0.03}, "execution": {"timeoutSeconds": 30}}
    store = runner.setup(legacy_suite, [SINGLE_LINE])
    assert run_main(runner) == 0
    assert set(store) == {"runs/run1/gate_summary.json", "runs/run1/scores.json", "runs/run1/details.jsonl"}
    gate = store["runs/run1/gate_summary.json"]
    assert gate["verdict"] == "PASS" and gate["schemaVersion"] == "evalrun.gate_summary/v1"
    assert "interactionMode" not in gate
    assert gate["artifacts"] == {
        "gateSummaryUri": "http://s3/buckets/bucket/objects/runs/run1/gate_summary.json",
        "scoresUri": "http://s3/buckets/bucket/objects/runs/run1/scores.json",
        "detailsUri": "http://s3/buckets/bucket/objects/runs/run1/details.jsonl",
    }


def test_max_policy_violations_is_read_from_agent_metrics(runner, monkeypatch):
    runner.setup({"metrics": {"agent": {"maxPolicyViolations": 2}}}, [SINGLE_LINE])
    assert runner.build_runner_config().thresholds.max_policy_violations == 2
    runner.setup({"metrics": {"policy": {"maxPolicyViolations": 3}}}, [SINGLE_LINE])  # legacy location
    assert runner.build_runner_config().thresholds.max_policy_violations == 3


DATASET = os.path.join(os.path.dirname(RUNNER_PATH), "..", "examples", "dataset-agent-single-turn.jsonl")


def test_single_turn_example_dataset_parses_and_every_case_has_expected_output(runner):
    cases = runner.parse_dataset_jsonl(DATASET)
    assert len(cases) == 11
    assert all(c.expected_output for c in cases)  # reference answer of a *correct* agent, for task completion
    assert all(c.expected_tools for c in cases)


def test_runner_requests_tool_outputs_by_default_and_redacts_them_in_details(runner, monkeypatch):
    sent = {}

    class Agent:
        def __call__(self, url, json=None, headers=None, timeout=None):
            sent.update(json)

            class R:
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"output": "done", "trace": {"traceId": "t", "toolCalls": [
                        {"name": "search", "arguments": {}, "status": "OK",
                         "output": {"flights": [1], "api_key": "sk-abcdefghijklmnopqrstuvwx"}}]}}
            return R()

    class FakeAdapter:
        def __init__(self, model, llm): pass
        def score_records(self, records):
            return [runner.AgentPerCaseScore(case_id=r.case_id, tool_selection=1.0, tool_invocation=1.0,
                                             tool_execution=1.0, passed=True) for r in records], {"metricNames": []}

    monkeypatch.setattr(runner.requests, "post", Agent())
    monkeypatch.setattr(runner, "PhoenixAgentAdapter", FakeAdapter)
    store = runner.setup({"metrics": {"agent": {"minToolSelection": 0.9}}}, [SINGLE_LINE])
    assert run_main(runner) == 0
    assert sent["trace"]["includeToolOutputs"] is True
    row = json.loads(store["runs/run1/details.jsonl"].splitlines()[0])
    out = row["toolsCalled"][0]["output"]
    assert out["flights"] == [1] and out["api_key"] == "[REDACTED]"
    assert "sk-abcdefghijklmnopqrstuvwx" not in store["runs/run1/details.jsonl"]
