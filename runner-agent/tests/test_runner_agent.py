"""Tests of the agent runner (runner-agent.py and the multiturn package), one section per module.

Rules: tests run offline (fake agent, uploader, Phoenix fetcher and judges); rule-specific tests sit next to the
tests of the module they exercise and are named after the behaviour (test_no_silent_ignore_*, test_no_value_synonyms_*).
"""
import asyncio
import importlib.util
import json
import os
import re
import shutil
import sys

import pytest

from multiturn.artifacts import (
    ArtifactLayout,
    REDACTED,
    is_sensitive_key,
    redact,
    redact_conversation,
    state_transitions,
)
from multiturn.canonical import canonical_json, partial_match, tool_call_key, values_equal
from multiturn.checks import Check, CheckContext, Limits, derive_events, run_checks
from multiturn.config import ConfigError, get_interaction_mode, parse_multiturn_settings
from multiturn.dataset import DatasetError, classify_dataset, parse_multiturn_lines
from multiturn.deepeval_provider import DeepEvalProvider
from multiturn.gate import EvaluatorFailure, GateInput, SemanticAggregate, decide
from multiturn.models import Conversation, SemanticScore, ToolCall, Turn, Violation, parse_tool_status
from multiturn.phoenix_adapter import (
    ObservedTurn,
    PhoenixFetchError,
    PhoenixRestFetcher,
    PhoenixSpanError,
    normalize_openinference_spans,
    resolve_turn_tool_calls,
)
from multiturn.runner import evaluate_run
from multiturn.semantic import SemanticProvider, run_semantic_evaluators
from multiturn.target import StatelessSessionAdapter, TargetError, TurnRequest
from conftest import suite_dict, travel_case
from fake_travel_agent import FakeTravelAgent


# ====================================================================================================================
# runner_agent_main: main() of runner-agent.py: multi-turn dispatch, dataset/mode guard, single-turn backward compatibility.
# ====================================================================================================================

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
           "TARGET_ENDPOINT": "http://agent/turn", "S3_ENDPOINT": "http://s3", "DATASET_S3_BUCKET": "datasets", "S3_SUMMARY_BUCKET": "bucket",
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
    runner.setup({"metrics": {"policy": {"maxPolicyViolations": 3}}}, [SINGLE_LINE])  # former location: no synonyms
    with pytest.raises(ConfigError, match="metrics.policy is not supported; set metrics.agent.maxPolicyViolations"):
        runner.build_runner_config()


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


# ============================================================
# No silent ignore (single-turn agent runner): configuration, dataset and agent-response problems are errors with
# actionable messages, judge errors are never averaged away; every value has exactly one field.
# ============================================================

# ---------------------------------------------------------------- suite.json (single-turn)

VALID = {"metrics": {"agent": {"minToolSelection": 0.9, "maxPolicyViolations": 0}},
         "regressionPolicy": {"maxDrop": 0.03}, "execution": {"mode": "agent", "timeoutSeconds": 30},
         "evaluation": {"interactionMode": "SingleTurn"}}


def test_no_silent_ignore_valid_single_turn_suite_is_parsed(runner):
    runner.setup(VALID, [SINGLE_LINE])
    cfg = runner.build_runner_config()
    assert cfg.thresholds.min_tool_selection == pytest.approx(0.9)
    assert (cfg.thresholds.max_policy_violations, cfg.regression_max_drop, cfg.request_timeout_seconds) == (0, 0.03, 30)


@pytest.mark.parametrize("suite, fragment", [
    ({"metrics": {"agent": {"minToolSelecton": 0.9}}}, r"metrics.agent: unknown field\(s\) \['minToolSelecton'\]"),
    ({"metrics": {"agent": {"minToolSelection": "0.9x"}}}, "minToolSelection must be a number"),
    ({"metrics": {"agent": {"minToolSelection": 1.5}}}, "out of range"),
    ({"metrics": {"agent": {"minTaskCompletion": True}}}, "minTaskCompletion must be a number"),
    ({"metrics": {"agent": {"maxPolicyViolations": -1}}}, "maxPolicyViolations must be an integer >= 0"),
    ({"metrics": {"agent": {"maxPolicyViolations": 1.5}}}, "maxPolicyViolations must be an integer"),
    ({"metrics": {"policy": {"maxPolicyViolations": 1}}}, "metrics.policy is not supported"),
    ({"metrics": {"rag": {}}}, r"metrics: unknown field\(s\) \['rag'\]"),
    ({"metrics": {"agent": {"tools": {"minSelectionAccuracy": 0.9}}}}, "metrics.agent.tools applies to MultiTurn suites only"),
    ({"metrics": {"agent": {}}, "evaluation": {"trials": 3}}, r"evaluation: unknown field\(s\) \['trials'\]"),
    ({"metrics": {"agent": {}}, "evaluation": {"trace": {"source": "inline"}}}, r"unknown field\(s\) \['trace'\]"),
    ({"metrics": {"agent": {}}, "evaluators": []}, r"suite: unknown field\(s\) \['evaluators'\]"),
    ({"metrics": {"agent": {}}, "regressionPolicy": {"maxDrop": "x"}}, "maxDrop must be a number"),
    ({"metrics": {"agent": {}}, "execution": {"timeoutSeconds": 0}}, "timeoutSeconds must be an integer >= 1"),
    ({"metrics": {"agent": {}}, "execution": {"runnerImage": "x"}}, r"execution: unknown field\(s\) \['runnerImage'\]"),
    ({"metrics": {"agent": {}}, "trace": {"sanitize": "no"}}, "trace.sanitize must be a boolean"),
])
def test_no_silent_ignore_invalid_single_turn_suite_is_a_configuration_error(runner, suite, fragment):
    runner.setup(suite, [SINGLE_LINE])
    with pytest.raises(ConfigError, match=fragment):
        runner.build_runner_config()


def test_no_silent_ignore_summary_bucket_is_required(runner, monkeypatch):
    runner.setup(VALID, [SINGLE_LINE])
    monkeypatch.delenv("S3_SUMMARY_BUCKET")
    with pytest.raises(RuntimeError, match="missing required environment variable: S3_SUMMARY_BUCKET"):
        runner.build_runner_config()


def test_no_silent_ignore_dataset_and_artifact_buckets_are_used_for_their_own_purpose(runner, monkeypatch, tmp_path):
    runner.setup(VALID, [SINGLE_LINE])
    # the fixture stubs download_dataset / load_optional_baseline; load an untouched copy of the module
    spec = importlib.util.spec_from_file_location("runner_agent_unpatched", RUNNER_PATH)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    cfg = runner.build_runner_config()
    cfg.dataset_file = str(tmp_path / "d.jsonl")
    urls = []

    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            return [b"{}"]

        def json(self):
            return {}

    monkeypatch.setattr(runner.requests, "get", lambda url, **k: urls.append(url) or R())
    monkeypatch.setattr(runner.requests, "put", lambda url, **k: urls.append(url) or R())
    runner.download_dataset(cfg)
    runner.load_optional_baseline(cfg)
    assert urls == ["http://s3/buckets/datasets/objects/ds.jsonl",
                    "http://s3/buckets/bucket/objects/baselines/s.json"]
    assert runner.artifact_uri(cfg, "k") == "http://s3/buckets/bucket/objects/k"


def test_no_silent_ignore_invalid_suite_still_publishes_a_configuration_root_cause(runner):
    store = runner.setup({"metrics": {"agent": {"minToolSelecton": 0.9}}}, [SINGLE_LINE])
    assert run_main(runner) == 1
    gate = store["runs/run1/gate_summary.json"]
    assert gate["verdict"] == "FAIL" and gate["rootCause"]["stage"] == "Configuration"
    assert "minToolSelecton" in gate["rootCause"]["details"]


# ---------------------------------------------------------------- suite.json (multi-turn)

@pytest.mark.parametrize("patch, fragment", [
    ({"metrics.agent.outcome.minGoalCompleton": 0.9}, r"metrics.agent.outcome: unknown field\(s\) \['minGoalCompleton'\]"),
    ({"metrics.agent.tools.maxToolCall": 3}, r"metrics.agent.tools: unknown field\(s\) \['maxToolCall'\]"),
    ({"metrics.agent.policy.maxCritical": 0}, r"metrics.agent.policy: unknown field"),
    ({"metrics.agent.conversation.maxTurn": 3}, r"metrics.agent.conversation: unknown field"),
    ({"metrics.agent.extra": {}}, r"metrics.agent: unknown field\(s\) \['extra'\]"),
    ({"metrics.rag": {}}, r"metrics: unknown field\(s\) \['rag'\]"),
    ({"evaluation.trace": {"source": "inline", "phoenx": {}}}, r"evaluation.trace: unknown field"),
    ({"evaluation.trace": {"source": "phoenix", "phoenix": {"endpoint": "http://p", "projekt": "x"}}},
     r"evaluation.trace.phoenix: unknown field"),
    ({"execution.timeoutSeconds": "60"}, "timeoutSeconds must be an integer >= 1"),
    ({"execution.runnerImage": "img"}, r"execution: unknown field"),
    ({"trace": {}}, r"suite: unknown field\(s\) \['trace'\]"),
    ({"evaluators": {"name": "x"}}, "evaluators must be a list"),
])
def test_no_silent_ignore_invalid_multi_turn_suite_is_a_configuration_error(patch, fragment):
    with pytest.raises(ConfigError, match=fragment):
        parse_multiturn_settings(suite_dict(**patch))


# ---------------------------------------------------------------- dataset

def case(**fields):
    return json.dumps({"caseId": "c-1", "input": "hello", **fields})


@pytest.mark.parametrize("lines, field, fragment", [
    (["{not json"], None, "invalid JSON"),
    (["[1]"], None, "must be a JSON object"),
    ([json.dumps({"input": "x"})], "caseId", "non-empty string"),
    ([case(), case()], "caseId", "duplicate caseId"),
    ([json.dumps({"caseId": "c-1"})], "input", "non-empty string"),
    ([case(input="  ")], "input", "non-empty string"),
    ([case(question="q")], "question", "RAG dataset field; the agent dataset uses 'input'"),
    ([case(groundTruth="a")], "groundTruth", "uses 'expectedOutput'"),
    ([case(expectedTool=[{"name": "search"}])], "expectedTool", r"unknown field\(s\) \['expectedTool'\]"),
    ([case(expectedOutput="")], "expectedOutput", "non-empty string"),
    ([case(expectedOutput={"a": 1})], "expectedOutput", "non-empty string"),
    ([case(expectedTools={"name": "search"})], "expectedTools", "must be a list"),
    ([case(expectedTools=[{"tool": "search"}])], r"expectedTools\[0\]", r"unknown field\(s\) \['tool'\]"),
    ([case(expectedTools=[{"name": ""}])], r"expectedTools\[0\].name", "non-empty string"),
    ([case(forbiddenTools=[{"name": "book", "arguments": []}])], r"forbiddenTools\[0\].arguments", "must be an object"),
    ([case(tags="smoke")], "tags", "list of non-empty strings"),
    ([case(metadata=[])], "metadata", "must be an object"),
])
def test_no_silent_ignore_malformed_single_turn_dataset_names_line_case_and_field(runner, tmp_path, lines, field, fragment):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DatasetError, match=fragment) as err:
        runner.parse_dataset_jsonl(str(path))
    assert "invalid single-turn agent dataset record (line" in str(err.value)
    if field:
        assert re.search(f"field={field}", str(err.value))


def test_no_silent_ignore_malformed_dataset_is_reported_with_dataset_stage_and_case(runner):
    store = runner.setup(VALID, [case(expectedTool=[{"name": "search"}])])
    assert run_main(runner) == 1
    rc = store["runs/run1/gate_summary.json"]["rootCause"]
    assert rc["stage"] == "Dataset" and rc["caseId"] == "c-1" and "expectedTool" in rc["details"]


# ---------------------------------------------------------------- agent response

def contract_record(runner, body):
    ds_case = runner.AgentDatasetCase(case_id="c-1", input="hi", expected_output=None, expected_output_pattern=None,
                                      expected_tools=[], forbidden_tools=[], tags=[], metadata={})
    return runner.build_agent_eval_record(ds_case, body)


GOOD_CALL = {"name": "search", "arguments": {"q": 1}, "status": "OK", "durationMs": 1.5}


def test_no_silent_ignore_valid_response_is_parsed(runner):
    rec = contract_record(runner, {"output": "done", "metadata": {"traceId": "t1"},
                          "trace": {"traceId": "t1", "toolCalls": [GOOD_CALL]}})
    (call,) = rec.tools_called
    assert (call.name, call.arguments, call.duration_ms) == ("search", {"q": 1}, 1.5)
    assert contract_record(runner, {"output": "no tools needed", "trace": {"toolCalls": []}}).tools_called == []


@pytest.mark.parametrize("body, fragment", [
    ([], "must be a JSON object"),
    ({"answer": "done", "trace": {"toolCalls": []}}, "answer is not supported; use output"),
    ({"output": "", "trace": {"toolCalls": []}}, "output is required"),
    ({"output": "done"}, "trace is missing"),
    ({"output": "done", "toolCalls": [GOOD_CALL]}, "toolCalls is not supported; use trace.toolCalls"),
    ({"output": "done", "toolsCalled": [GOOD_CALL], "trace": {"toolCalls": []}}, "toolsCalled is not supported"),
    ({"output": "done", "trace": {}}, "trace.toolCalls is required"),
    ({"output": "done", "trace": {"tool_calls": [GOOD_CALL]}}, "trace.tool_calls is not supported; use trace.toolCalls"),
    ({"output": "done", "trace": {"toolCalls": {"name": "x"}}}, "trace.toolCalls must be a list"),
    ({"output": "done", "trace": {"toolCalls": [{"tool": "search"}]}}, r"trace.toolCalls\[0\].tool is not supported; use name"),
    ({"output": "done", "trace": {"toolCalls": [{"name": "s", "args": {}}]}}, "args is not supported; use arguments"),
    ({"output": "done", "trace": {"toolCalls": [{"name": "s", "duration_ms": 1}]}}, "duration_ms is not supported"),
    ({"output": "done", "trace": {"toolCalls": [{"name": "s", "arguments": "q=1"}]}}, "arguments must be an object"),
    ({"output": "done", "trace": {"toolCalls": [{"name": "s", "durationMs": "1ms"}]}}, "durationMs must be a number"),
    ({"output": "done", "trace": {"toolCalls": [{"name": "s", "status": 200}]}}, r"toolCalls\[0\].status must be one of \['OK', 'ERROR'\]"),
    ({"output": "done", "trace": {"toolCalls": [{"name": "s", "status": "SUCCESS"}]}}, "status must be one of"),
    ({"output": "done", "trace": {"toolCalls": [{"name": "s", "status": "ok"}]}}, "status must be one of"),
    ({"output": "done", "metadata": {"traceId": "a"}, "trace": {"traceId": "b", "toolCalls": []}},
     "metadata.traceId='a' differs from trace.traceId='b'"),
    ({"output": "done", "trace": {"toolCalls": []}, "traceRef": {"traceId": "x"}}, "either trace or traceRef"),
    ({"output": "done", "trace": {"toolCalls": [], "steps": "s"}}, "trace.steps must be a list"),
    ({"output": "done", "trace": {"toolCalls": []}, "metadata": []}, "metadata must be an object"),
])
def test_no_silent_ignore_response_contract_violations_are_errors(runner, body, fragment):
    with pytest.raises(runner.AgentResponseError, match=fragment) as err:
        contract_record(runner, body)
    assert err.value.case_id == "c-1"


def test_no_silent_ignore_contract_violation_is_reported_with_the_case_and_a_clear_message(runner, monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"output": "done", "trace": {"tool_calls": []}}

    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: Resp())
    store = runner.setup(VALID, [SINGLE_LINE])
    assert run_main(runner) == 1
    rc = store["runs/run1/gate_summary.json"]["rootCause"]
    assert rc["stage"] == "Execution" and rc["caseId"] == "st-1"
    assert rc["details"].startswith("agent response for caseId='st-1' violates the endpoint contract")


def test_no_value_synonyms_tool_execution_counts_only_the_canonical_ok_status(runner):
    rec = contract_record(runner, {"output": "done", "trace": {"toolCalls": [
        {"name": "a", "status": "OK"}, {"name": "b", "status": "ERROR", "error": "boom"}, {"name": "c"}]}})
    assert [c.status for c in rec.tools_called] == ["OK", "ERROR", "OK"]


def test_no_silent_ignore_type_on_a_single_turn_record_is_reported_as_multi_turn_field(runner):
    store = runner.setup(VALID, [case(type="singleTurn")])
    assert run_main(runner) == 1
    rc = store["runs/run1/gate_summary.json"]["rootCause"]
    assert rc["stage"] == "Dataset" and "multi-turn fields ('type'/'turns')" in rc["details"]


# ---------------------------------------------------------------- judge errors are never averaged away

class _Score:
    def __init__(self, score):
        self.score, self.label, self.explanation = score, "x", None


class _PhoenixJudge:
    def __init__(self, value=1.0, boom=False):
        self.value, self.boom = value, boom

    def evaluate(self, eval_input):
        if self.boom:
            raise RuntimeError("judge unavailable")
        return [_Score(self.value)]


def _record(runner, case_id="c1", expected=("search",), called=("search",)):
    return runner.AgentEvalRecord(
        case_id=case_id, input="find", actual_output="done", expected_output=None,
        expected_tools=[runner.AgentTool(name=n) for n in expected],
        tools_called=[runner.AgentToolCall(name=n, arguments={}, status="OK") for n in called])


def _adapter(runner, monkeypatch, selection, invocation=None):
    a = runner.PhoenixAgentAdapter(model="judge", llm=None)
    monkeypatch.setattr(a, "_get_tool_selection_evaluator", lambda: selection)
    monkeypatch.setattr(a, "_get_tool_invocation_evaluator", lambda: invocation or _PhoenixJudge(1.0))
    monkeypatch.setattr(a, "_get_tool_response_handling_evaluator", lambda: _PhoenixJudge(1.0))
    monkeypatch.setattr(a, "_get_task_completion_evaluator", lambda: _PhoenixJudge(1.0))
    return a


def test_no_silent_ignore_judge_statuses_distinguish_not_applicable_from_error(runner, monkeypatch):
    a = _adapter(runner, monkeypatch, _PhoenixJudge(boom=True))
    s = a.score_record(_record(runner))
    assert s.tool_selection is None and s.tool_selection_status.startswith("error: RuntimeError")
    assert s.tool_invocation == 1.0 and s.tool_invocation_status == "scored"  # no longer skipped with selection

    s = _adapter(runner, monkeypatch, _PhoenixJudge(1.0)).score_record(_record(runner, expected=(), called=()))
    assert s.tool_selection_status == "not_applicable: case has no expectedTools or forbiddenTools"
    assert s.tool_invocation_status == "not_applicable: no tool was called"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.9", True])
def test_no_silent_ignore_non_finite_or_non_numeric_judge_score_is_an_error(runner, monkeypatch, value):
    s = _adapter(runner, monkeypatch, _PhoenixJudge(value)).score_record(_record(runner))
    assert s.tool_selection is None and "non-finite or non-numeric score" in s.tool_selection_status


def _cfg(runner, **thresholds):
    c = type("Cfg", (), {})()
    c.thresholds = runner.AgentThresholdConfig(**thresholds)
    return c


def test_no_silent_ignore_judge_error_fails_the_run_even_if_the_average_of_the_rest_passes(runner):
    ok = runner.AgentPerCaseScore(case_id="ok", tool_selection=1.0, tool_selection_status="scored")
    err = runner.AgentPerCaseScore(case_id="boom", tool_selection=None, tool_selection_status="error: RuntimeError()")
    na = runner.AgentPerCaseScore(case_id="na", tool_selection=None,
                                  tool_selection_status="not_applicable: case has no expectedTools or forbiddenTools")
    cfg = _cfg(runner, min_tool_selection=0.9)
    unscored = runner.unscored_cases(cfg, [ok, err, na])
    assert unscored == {"tool_selection": ["boom"]}
    agg = runner.aggregate_scores([ok, err, na])
    assert agg.tool_selection_avg == 1.0  # averaging alone would PASS
    failed, rc, _ = runner.apply_threshold_policy(cfg, agg, unscored)
    assert failed and rc["stage"] == "Tool Selection" and rc["metric"] == "tool_selection"
    assert "not scored for 1 case(s) (boom)" in rc["details"]


def test_no_silent_ignore_not_applicable_does_not_fail_the_run(runner):
    na = runner.AgentPerCaseScore(case_id="na", tool_selection=None, tool_selection_status="not_applicable: x")
    cfg = _cfg(runner, min_tool_selection=0.9)
    assert runner.unscored_cases(cfg, [na]) == {}
    assert runner.apply_threshold_policy(cfg, runner.aggregate_scores([na]), {})[0] is False


def test_no_silent_ignore_unscored_metric_without_threshold_is_not_gated(runner):
    err = runner.AgentPerCaseScore(case_id="boom", task_completion=None, task_completion_status="error: x")
    assert runner.unscored_cases(_cfg(runner), [err]) == {}


# ====================================================================================================================
# artifacts: multiturn/artifacts.py - artifact layout, URIs and recursive redaction.
# ====================================================================================================================

def test_recursive_redaction_of_sensitive_keys():
    payload = {
        "authorization": "Bearer abc.def.ghi-jkl", "Api_Key": "k", "apiKey": "k", "access_token": "t",
        "client_secret": "s", "password": "p", "Secret": "x",
        "nested": {"headers": {"Authorization": "Basic zzz"}, "list": [{"password": "p", "keep": 1}]},
        "toolCalls": [{"arguments": {"apiKey": "sk-live", "city": "London"}}],
        "max_tokens": 100, "empty_secret": None,
    }
    out = redact(payload)
    assert out["authorization"] == out["Api_Key"] == out["apiKey"] == out["access_token"] == REDACTED
    assert out["client_secret"] == out["password"] == out["Secret"] == REDACTED
    assert out["nested"]["headers"]["Authorization"] == REDACTED
    assert out["nested"]["list"][0] == {"password": REDACTED, "keep": 1}
    assert out["toolCalls"][0]["arguments"] == {"apiKey": REDACTED, "city": "London"}
    assert out["max_tokens"] == 100                      # "tokens" counters are not secrets
    assert out["empty_secret"] is None
    assert payload["apiKey"] == "k"                      # input is not mutated
    assert "abc.def" not in json.dumps(out)


def test_secret_looking_values_are_scrubbed_even_under_innocent_keys():
    out = redact({"note": "call with Bearer abcdefghijklmnop and sk-abcdefghijklmnopqrstuv please"})
    assert "abcdefghijklmnop" not in out["note"] and REDACTED in out["note"]


def test_very_long_strings_are_truncated():
    assert len(redact({"p": "x" * 20000})["p"]) < 8200


def test_is_sensitive_key():
    assert all(is_sensitive_key(k) for k in ["api-key", "X-API-KEY", "accessToken", "aws_secret_access_key"])
    assert is_sensitive_key("access_token") and is_sensitive_key("clientSecret") and not is_sensitive_key("destination")


def test_redact_conversation_covers_args_results_state_and_judge_input():
    conv = Conversation("c", "s", turns=[
        Turn(1, "user", "use header Authorization: Bearer abcdefghijklmnop"),
        Turn(1, "assistant", "ok", tool_calls=[ToolCall("1", "login", {"password": "hunter2", "user": "bob"},
                                                      result={"access_token": "zzz"}, error="bad sk-abcdefghijklmnopqrstuv")],
             metadata={"state": {"api_key": "k"}})],
        final_state={"secret": "s", "city": "Rome"})
    red = redact_conversation(conv)
    dumped = json.dumps(red.to_dict())
    for leaked in ("hunter2", "abcdefghijklmnop", "zzz", "sk-abcdefghijklmnopqrstuv", '"k"'):
        assert leaked not in dumped, leaked
    assert red.final_state == {"secret": REDACTED, "city": "Rome"}
    assert conv.final_state["secret"] == "s"  # original untouched


def test_layout_keys_and_uris():
    layout = ArtifactLayout.from_env("bkt", "runs/r/gate_summary.json", "runs/r/scores.json", "runs/r/details.jsonl", {})
    assert layout.manifest_key == "runs/r/manifest.json"
    assert layout.conversation_key("a/b c", 1) == "runs/r/conversations/a_b_c.json"
    assert layout.conversation_key("a", 3) == "runs/r/conversations/a__t3.json"
    # without a gateway endpoint (local runs) URIs fall back to s3://bucket/key
    uris = layout.artifact_uris()
    assert uris["violationsUri"] == "s3://bkt/runs/r/violations.jsonl" and uris["conversationsUri"].endswith("/conversations/")
    # in a Job S3_ENDPOINT is set and URIs use the same gateway form as the RAG runner
    gw = ArtifactLayout.from_env("bkt", "runs/r/gate_summary.json", "runs/r/scores.json", "runs/r/details.jsonl",
                                 {"S3_ENDPOINT": "http://k8s:9876/"})
    assert gw.uri("runs/r/scores.json") == "http://k8s:9876/buckets/bkt/objects/runs/r/scores.json"
    assert gw.artifact_uris()["gateSummaryUri"] == "http://k8s:9876/buckets/bkt/objects/runs/r/gate_summary.json"
    assert gw.artifact_uris()["conversationsUri"] == "http://k8s:9876/buckets/bkt/objects/runs/r/conversations/"
    env = {"MANIFEST_KEY": "m.json", "VIOLATIONS_KEY": "v.jsonl", "CONVERSATIONS_PREFIX": "convs/"}
    custom = ArtifactLayout.from_env("bkt", "g.json", "s.json", "d.jsonl", env)
    assert (custom.manifest_key, custom.violations_key, custom.conversations_prefix) == ("m.json", "v.jsonl", "convs")


def test_state_transitions_only_list_changes():
    conv = Conversation("c", "s", turns=[
        Turn(1, "assistant", "", metadata={"state": {"dest": "Paris"}}),
        Turn(2, "assistant", "", metadata={"state": {"dest": "Paris"}}),
        Turn(3, "assistant", "", metadata={"state": {"dest": "London"}})])
    tr = state_transitions(conv)
    assert [t["turn"] for t in tr] == [1, 3]
    assert tr[1]["changed"] == {"dest": {"from": "Paris", "to": "London"}}


# ====================================================================================================================
# checks: Unit tests for every deterministic check, driven by hand-built canonical conversations.
# ====================================================================================================================

def conversation(calls_by_turn, states=None, user_turns=4, case=None):
    case = case or travel_case()
    conv = Conversation(case_id=case.case_id, session_id="s", metadata={"scriptedTurns": user_turns})
    for i in range(1, user_turns + 1):
        conv.turns.append(Turn(i, "user", f"user {i}"))
        calls = [ToolCall(f"call-{i}-{j}", name, args, turn_index=i, **kw)
                 for j, (name, args, kw) in enumerate(calls_by_turn.get(i, []), start=1)]
        conv.turns.append(Turn(i, "assistant", f"reply {i}", trace_id=f"trace-{i}", tool_calls=calls,
                               metadata={"state": (states or {}).get(i, {})}))
    conv.final_state = (states or {}).get(user_turns, {})
    conv.events = derive_events(case)
    return conv


def check_call(name, **args):
    return (name, args, {})


def run_case_checks(case, conv, limits=None):
    ctx = CheckContext(case=case, conversation=conv, limits=limits or Limits())
    return {r.check: r for r in run_checks(ctx)}


def test_required_tool_success_and_failure():
    case = travel_case(requiredTools=["search_flights", "get_weather"])
    ok = run_case_checks(case, conversation({3: [check_call("search_flights", destination="London"), check_call("get_weather")]}))
    assert ok["required_tool"].passed and ok["required_tool"].score == 1.0
    bad = run_case_checks(case, conversation({3: [check_call("search_flights", destination="London")]}))["required_tool"]
    assert not bad.passed and bad.score == 0.5
    (v,) = bad.violations
    assert (v.code, v.severity, v.evidence["tool"]) == ("REQUIRED_TOOL_NOT_CALLED", "HIGH", "get_weather")


def test_forbidden_tool_is_critical_per_call():
    case = travel_case()
    res = run_case_checks(case, conversation({1: [check_call("delete_account")], 2: [check_call("delete_account")]}))["forbidden_tool"]
    assert not res.passed and len(res.violations) == 2
    assert {v.severity for v in res.violations} == {"CRITICAL"}
    assert res.violations[0].turn_index == 1 and res.violations[0].tool_call_id == "call-1-1"
    assert res.violations[0].trace_id == "trace-1" and res.violations[0].evaluator == "trajectory-policy"


def test_tool_arguments_partial_matching():
    case = travel_case()
    good = run_case_checks(case, conversation({3: [check_call("search_flights", origin="NYC", destination="london", date="June 5")]}))
    assert good["tool_arguments"].passed and good["tool_arguments"].score == 1.0
    bad = run_case_checks(case, conversation({3: [check_call("search_flights", origin="NYC", destination="Paris")]}))["tool_arguments"]
    assert not bad.passed
    (v,) = bad.violations
    assert v.code == "TOOL_ARGUMENT_MISMATCH" and v.turn_index == 3
    assert v.evidence["mismatches"][0]["path"] == "$.destination"
    assert v.evidence["observedArguments"]["destination"] == "Paris"


def test_expected_call_in_wrong_turn_or_missing_is_reported():
    case = travel_case()
    wrong_turn = run_case_checks(case, conversation({2: [check_call("search_flights", destination="London")]}))["tool_arguments"]
    assert wrong_turn.violations[0].code == "EXPECTED_TOOL_CALL_MISSING" and wrong_turn.violations[0].turn_index == 3


def test_latest_value_wins_detects_stale_tool_args_and_stale_state():
    case = travel_case()
    res = run_case_checks(case, conversation(
        {2: [check_call("search_flights", destination="Paris")], 3: [check_call("search_flights", destination="Paris")]},
        states={2: {"destination": "Paris"}, 3: {"destination": "Paris"}, 4: {"destination": "London"}}))
    lv = res["latest_value_wins"]
    assert not lv.passed
    codes = sorted(v.code for v in lv.violations)
    assert codes == ["STALE_STATE", "STALE_VALUE_IN_TOOL_CALL"]  # turn 2 predates the correction: ignored
    assert all(v.turn_index == 3 for v in lv.violations)


def test_latest_value_wins_passes_when_the_correction_propagates():
    case = travel_case()
    res = run_case_checks(case, conversation({3: [check_call("search_flights", destination="London")]},
                                             states={3: {"destination": "London"}, 4: {"destination": "London"}}))
    assert res["latest_value_wins"].passed


def approval_case():
    return travel_case(turns=[{"role": "user", "content": "find"}, {"role": "user", "content": "yes, book it"},
                              {"role": "user", "content": "thanks"}], cancellationEvents=["cancellation"],
                       latestValues=[], expectedToolCalls=[], expectedFinalState={})


def test_approval_before_mutation():
    case = approval_case()
    ok = run_case_checks(case, conversation({2: [check_call("book_flight", id=1)]}, user_turns=3, case=case))
    assert ok["approval_before_mutation"].passed
    early = run_case_checks(case, conversation({1: [check_call("book_flight", id=1)]}, user_turns=3, case=case))["approval_before_mutation"]
    (v,) = early.violations
    assert v.code == "ACTION_WITHOUT_APPROVAL" and v.severity == "CRITICAL" and v.turn_index == 1
    assert v.evidence["requiredEvent"] == "explicit_user_confirmation"


def test_agent_output_cannot_grant_approval():
    case = approval_case()
    conv = conversation({1: [check_call("book_flight", id=1)]}, user_turns=3, case=case)
    conv.turns[1].content = "yes, book it"  # assistant text containing the magic words
    res = run_case_checks(case, conv)["approval_before_mutation"]
    assert not res.passed


def test_action_after_cancellation_and_resume():
    case = travel_case(turns=[{"role": "user", "content": "find"},
                              {"role": "user", "content": "stop", "markers": ["cancellation"]},
                              {"role": "user", "content": "ok go on", "markers": ["go_ahead"]},
                              {"role": "user", "content": "yes, book it"}],
                       resumeEvents=["go_ahead"], latestValues=[], expectedToolCalls=[], expectedFinalState={})
    after = run_case_checks(case, conversation({2: [check_call("book_flight", id=1)]}, case=case))["action_after_cancellation"]
    (v,) = after.violations
    assert v.code == "ACTION_AFTER_CANCELLATION" and v.severity == "CRITICAL" and v.turn_index == 2
    assert v.evidence["cancellationEvent"]["type"] == "cancellation"
    resumed = run_case_checks(case, conversation({4: [check_call("book_flight", id=1)]}, case=case))["action_after_cancellation"]
    assert resumed.passed


def test_repeated_mutation_uses_canonical_arguments_and_ignores_failed_attempts():
    case = approval_case()
    dup = run_case_checks(case, conversation({2: [check_call("book_flight", a=1, b=2), check_call("book_flight", b=2, a=1)]},
                                             user_turns=3, case=case))["repeated_mutation"]
    (v,) = dup.violations
    assert v.code == "REPEATED_MUTATION" and v.severity == "CRITICAL" and v.evidence["firstCallId"] == "call-2-1"
    conv = conversation({2: [check_call("book_flight", a=1)]}, user_turns=3, case=case)
    conv.turns[3].tool_calls.insert(0, ToolCall("failed", "book_flight", {"a": 1}, status="ERROR", error="x", turn_index=2))
    assert run_case_checks(case, conv)["repeated_mutation"].passed  # retry after a failure is legitimate
    different = run_case_checks(case, conversation({2: [check_call("book_flight", a=1), check_call("book_flight", a=2)]},
                                                   user_turns=3, case=case))["repeated_mutation"]
    assert different.passed


def test_max_turns_uses_stricter_of_suite_and_dataset_and_policy_severity():
    case = travel_case(maxTurns=3)
    conv = conversation({}, user_turns=4)
    res = run_case_checks(case, conv, Limits(max_turns=10))["max_turns"]
    assert not res.passed and res.violations[0].severity == "HIGH" and res.observed == 4
    warn = run_case_checks(case, conv, Limits(max_turns=10, limits_action="Warn"))["max_turns"]
    assert warn.violations[0].severity == "MEDIUM"
    assert run_case_checks(travel_case(), conv, Limits(max_turns=4))["max_turns"].passed


def test_max_tool_calls():
    case = travel_case()
    conv = conversation({1: [check_call("a"), check_call("b")], 2: [check_call("c")]})
    assert not run_case_checks(case, conv, Limits(max_tool_calls=2))["max_tool_calls"].passed
    assert run_case_checks(case, conv, Limits(max_tool_calls=3))["max_tool_calls"].passed


def test_max_repeated_calls_is_order_insensitive_and_skips_mutations():
    case = travel_case()
    conv = conversation({1: [check_call("s", a=1, b=2)], 2: [check_call("s", b=2, a=1)], 3: [check_call("book_flight", x=1),
                                                                                             check_call("book_flight", x=1)]})
    res = run_case_checks(case, conv, Limits(max_repeated_calls=1))["max_repeated_calls"]
    assert not res.passed and len(res.violations) == 1  # book_flight belongs to RepeatedMutationCheck
    assert res.violations[0].evidence["callIds"] == ["call-1-1", "call-2-1"]
    assert run_case_checks(case, conv, Limits(max_repeated_calls=2))["max_repeated_calls"].passed


def test_final_state_partial_match():
    case = travel_case()
    ok = run_case_checks(case, conversation({}, states={4: {"destination": "London", "extra": 1}}))["final_state"]
    assert ok.passed
    bad = run_case_checks(case, conversation({}, states={4: {"destination": "Paris"}}))["final_state"]
    assert bad.violations[0].code == "FINAL_STATE_MISMATCH"


def test_event_derivation_uses_markers_and_regex_on_user_turns_only():
    case = travel_case(turns=[{"role": "user", "content": "Yes, book it!"},
                              {"role": "user", "content": "x", "markers": ["cancellation"]}],
                       expectedToolCalls=[], latestValues=[], expectedFinalState={})
    events = derive_events(case)
    assert [(e.type, e.turn_index, e.source) for e in events] == [
        ("explicit_user_confirmation", 1, "matcher"), ("cancellation", 2, "marker")]


def test_a_crashing_check_is_isolated_and_marked_as_error():
    class Boom(Check):
        name = "boom"

        def evaluate(self, ctx):
            raise RuntimeError("bug")

    case = travel_case()
    ctx = CheckContext(case=case, conversation=conversation({}))
    (res,) = run_checks(ctx, [Boom()])
    assert not res.passed and "bug" in res.error


def test_every_check_result_carries_evaluator_name_and_version():
    for r in run_case_checks(travel_case(), conversation({})).values():
        assert r.evaluator == "trajectory-policy" and r.version


# ====================================================================================================================
# config: multiturn/config.py - strict parsing of the rendered suite.json for multi-turn suites.
# ====================================================================================================================

def test_single_turn_suites_are_unaffected():
    legacy = {"metrics": {"agent": {"minToolSelection": 0.9, "maxPolicyViolations": 0}},
              "regressionPolicy": {"maxDrop": 0.03}, "execution": {"mode": "agent", "timeoutSeconds": 300}}
    assert get_interaction_mode(legacy) == "SingleTurn"
    s = parse_multiturn_settings(legacy)
    # the flat single-turn field is not a synonym of tools.minSelectionAccuracy
    assert not s.is_multi_turn and s.timeout_seconds == 300 and s.min_selection_accuracy is None


def test_full_multi_turn_suite():
    suite = suite_dict(**{
        "evaluation.trials": 3,
        "evaluation.trace": {"source": "auto", "phoenix": {"endpoint": "http://phoenix:6006", "project": "travel"}},
        "metrics.agent.conversation": {"maxTurns": 10, "minKnowledgeRetention": 0.85,
                                       "minConversationCompleteness": 0.85, "minTurnRelevancy": 0.8},
        "metrics.agent.tools": {"minSelectionAccuracy": 0.9, "minArgumentAccuracy": 0.9, "maxToolCalls": 20,
                                "maxRepeatedCalls": 1},
        "regressionPolicy": {"maxDrop": 0.03},
    })
    suite["evaluators"].append({"name": "conversation-quality", "provider": "deepeval", "type": "multiTurn",
                                "samples": 3, "model": "judge-1"})
    s = parse_multiturn_settings(suite)
    assert s.is_multi_turn and s.trials == 3 and s.trace_source == "auto"
    assert (s.phoenix_endpoint, s.phoenix_project) == ("http://phoenix:6006", "travel")
    assert (s.max_turns, s.max_tool_calls, s.max_repeated_calls) == (10, 20, 1)
    assert s.semantic_threshold("turnRelevancy") == 0.8 and s.regression_max_drop == 0.03
    (sem,) = s.semantic_evaluators()
    assert (sem.required, sem.samples, sem.model) == (False, 3, "judge-1")
    # metrics come only from the thresholds in metrics.agent.conversation
    assert s.semantic_metrics() == ["knowledgeRetention", "conversationCompleteness", "turnRelevancy"]
    assert s.builtin_evaluator().required is True


def test_defaults_zero_tolerance_and_required_builtin():
    s = parse_multiturn_settings({"evaluation": {"interactionMode": "MultiTurn"}})
    assert (s.max_critical_violations, s.max_high_violations, s.trials, s.trace_source) == (0, 0, 1, "inline")
    assert s.builtin_evaluator().required and s.semantic_evaluators() == []


def test_deepeval_evaluator_computes_exactly_the_metrics_with_a_threshold():
    suite = suite_dict(**{"metrics.agent.conversation": {"minTurnRelevancy": 0.7}})
    suite["evaluators"].append({"name": "q", "provider": "deepeval"})
    s = parse_multiturn_settings(suite)
    assert s.semantic_metrics() == ["turnRelevancy"]
    assert s.semantic_evaluators()[0].required is False  # deepeval is optional unless required: true


def test_builtin_evaluator_is_required_even_without_the_flag():
    suite = suite_dict()
    suite["evaluators"] = [{"name": "policy"}]
    assert parse_multiturn_settings(suite).builtin_evaluator().required is True


def test_deterministic_only_suite_needs_no_semantic_evaluator():
    s = parse_multiturn_settings(suite_dict())
    assert s.semantic_metrics() == [] and s.semantic_evaluators() == []


_DEEPEVAL = {"name": "q", "provider": "deepeval"}
_BUILTIN = {"name": "trajectory-policy", "provider": "builtin"}
_KR = {"metrics.agent.conversation": {"minKnowledgeRetention": 0.8}}


@pytest.mark.parametrize("patch, fragment", [
    ({"evaluation.interactionMode": "Chatty"}, "interactionMode"),
    ({"evaluation.trials": 11}, "trials"),
    ({"evaluation.trace": {"source": "phoenix"}}, "phoenix.endpoint"),
    ({"evaluation.trace": {"source": "magic"}}, "trace.source"),
    ({"metrics.agent.outcome.minGoalCompletion": 1.5}, "minGoalCompletion"),
    ({"metrics.agent.conversation.maxTurns": 0}, "maxTurns"),
    ({"metrics.agent.policy.limitsAction": "Ignore"}, "limitsAction"),
    ({"evaluators": [{"provider": "builtin"}]}, "name is required"),
    ({"evaluators": [{"name": "a", "provider": "other"}]}, "provider"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "metrics": ["knowledgeRetention"]}]}, "unknown field.*metrics.*thresholds"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "samples": 9}]}, "samples"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "required": "yes"}]}, "required must be a boolean"),
    ({**_KR, "evaluators": [{**_DEEPEVAL, "type": "deterministic"}]}, "only supports type multiTurn"),
    ({**_KR, "evaluators": [_DEEPEVAL, {**_DEEPEVAL, "name": "q2"}]}, "only one evaluator with provider deepeval"),
    ({"evaluators": [_BUILTIN, {"name": "b"}]}, "only one evaluator with provider builtin"),
    ({"evaluators": [{**_BUILTIN, "required": False}]}, "always required"),
    ({"evaluators": [{**_BUILTIN, "model": "gpt"}]}, "only to provider deepeval"),
    ({"evaluators": [{**_BUILTIN, "samples": 2}]}, "only to provider deepeval"),
    ({**_KR, "evaluators": [_BUILTIN]}, "minKnowledgeRetention is set but no evaluator computes it"),
    ({"evaluators": [_BUILTIN, _DEEPEVAL]}, "nothing to evaluate"),
    ({"metrics.agent.minToolSelection": 0.9}, "minToolSelection applies to single-turn.*minSelectionAccuracy"),
    ({"metrics.agent.minToolInvocation": 0.9}, "minToolInvocation applies to single-turn"),
    ({"metrics.agent.minToolExecution": 0.9}, "minToolExecution applies to single-turn"),
    ({"metrics.agent.maxPolicyViolations": 0}, "maxPolicyViolations applies to single-turn"),
    ({"evaluators": [{"name": "a"}, {"name": "a"}]}, "duplicated"),
])
def test_invalid_configuration_is_rejected_with_a_clear_message(patch, fragment):
    with pytest.raises(ConfigError, match=fragment):
        parse_multiturn_settings(suite_dict(**patch))


# ====================================================================================================================
# dataset: multiturn/dataset.py - strict parsing of scripted multi-turn datasets.
# ====================================================================================================================

def line(**over):
    base = {"caseId": "c1", "type": "multiTurn", "turns": [{"role": "user", "content": "hi"}]}
    base.update(over)
    return json.dumps(base)


def test_parses_full_case():
    case = parse_multiturn_lines([json.dumps({
        "caseId": "travel-1", "type": "multiTurn", "sessionId": "s-1",
        "turns": [{"role": "user", "content": "book it", "markers": ["explicit_user_confirmation"]}],
        "expectedBehavior": {
            "requiredTools": ["search"], "conditionalTools": [
                {"name": "book_flight", "allowedAfterEvent": "explicit_user_confirmation"}],
            "latestValues": [{"field": "destination", "value": "London", "fromTurn": 2}],
            "maxTurns": 5}})])[0]
    assert case.session_id == "s-1"
    assert case.turns[0].markers == ["explicit_user_confirmation"]
    assert case.expected.conditional_tools[0].allowed_after_event == "explicit_user_confirmation"
    assert case.expected.latest_values[0].argument == "destination"  # defaults to the field name
    assert case.expected.cancellation_events == ["cancellation"]
    assert case.expected.max_turns == 5


@pytest.mark.parametrize("record, field, fragment", [
    ({"turns": [{"content": "x"}]}, "caseId", "required"),
    ({"turns": []}, "turns", "non-empty list"),
    ({"turns": [{"role": "assistant", "content": "x"}]}, "turns[0].role", "user turns only"),
    ({"turns": [{"role": "user"}]}, "turns[0].content", "required"),
    ({"turns": [{"role": "user", "content": "x", "markers": "oops"}]}, "turns[0].markers", "list of strings"),
    ({"expectedBehavior": {"maxTurns": 0}}, "expectedBehavior.maxTurns", ">= 1"),
    ({"expectedBehavior": {"reqiredTools": ["x"]}}, "expectedBehavior.reqiredTools", "unknown"),
    ({"expectedBehavior": {"conditionalTools": [{"name": "book"}]}},
     "expectedBehavior.conditionalTools[0].allowedAfterEvent", "required"),
    ({"expectedBehavior": {"conditionalTools": [{"name": "book", "allowedAfterEvent": "never_happens"}]}},
     "expectedBehavior.conditionalTools[0].allowedAfterEvent", "never produced"),
    ({"expectedBehavior": {"eventMatchers": [{"event": "e", "regex": "("}]}},
     "expectedBehavior.eventMatchers[0].regex", "regular expression"),
    ({"expectedBehavior": {"expectedToolCalls": [{"name": "x", "turn": 9}]}},
     "expectedBehavior.expectedToolCalls[0].turn", "1 turns"),
    ({"expectedBehavior": {"latestValues": [{"field": "d"}]}}, "expectedBehavior.latestValues[0].value", "required"),
    ({"bogus": 1}, "bogus", "unknown"),
])
def test_malformed_cases_name_case_and_field(record, field, fragment):
    raw = {"caseId": "bad-case", "type": "multiTurn", "turns": [{"role": "user", "content": "hi"}]}
    raw.update(record)
    if "caseId" not in record and "turns" in record and field == "caseId":
        raw.pop("caseId")
    with pytest.raises(DatasetError) as err:
        parse_multiturn_lines([json.dumps(raw)])
    msg = str(err.value)
    assert field in msg and fragment in msg
    assert err.value.field_path == field
    assert ("bad-case" in msg) or field == "caseId"
    assert "line 1" in msg


def test_invalid_json_reports_line_number():
    with pytest.raises(DatasetError, match="line 2"):
        parse_multiturn_lines([line(), "{not json"])


def test_duplicate_case_ids_rejected():
    with pytest.raises(DatasetError, match="duplicate caseId"):
        parse_multiturn_lines([line(), line()])


def test_single_turn_case_in_multiturn_dataset_is_rejected_with_case_id():
    with pytest.raises(DatasetError) as err:
        parse_multiturn_lines([json.dumps({"caseId": "old-style", "input": "hello"})])
    assert "old-style" in str(err.value) and "cannot be mixed" in str(err.value)


def test_classify_dataset_keeps_single_turn_format_untouched():
    multi, single = classify_dataset([
        json.dumps({"caseId": "a", "input": "q", "expectedTools": [{"name": "t"}]}),
        line(caseId="b"),
        "",
    ])
    assert multi == ["b"] and single == ["a"]


# ---- no value synonyms: the case type has exactly one spelling, and it is required

@pytest.mark.parametrize("value", ["multiturn", "multi_turn", "multi-turn", "MultiTurn", " multiTurn", None, "absent"])
def test_no_value_synonyms_multi_turn_type_has_exactly_one_spelling(value):
    raw = json.loads(line(type=value))
    if value == "absent":
        del raw["type"]
    with pytest.raises(DatasetError, match=r"field=type\): is required and must be 'multiTurn'"):
        parse_multiturn_lines([json.dumps(raw)])


def test_no_value_synonyms_canonical_multi_turn_type_is_accepted():
    (case,) = parse_multiturn_lines([line()])
    assert case.case_id == "c1"


# ====================================================================================================================
# gate: multiturn/gate.py - verdict precedence and root causes.
# ====================================================================================================================

def viol(severity, code="X", case="c1", turn=2):
    return Violation(code=code, message=f"{code} happened", severity=severity, case_id=case, session_id="s",
                     turn_index=turn, evaluator="trajectory-policy")


def gate_settings(**o):
    return parse_multiturn_settings(suite_dict(**o))


def inp(**kw):
    base = dict(settings=gate_settings(), aggregates={"goal_completion": 1.0, "tool_selection_accuracy": 1.0,
                                                      "tool_argument_accuracy": 1.0},
                conversation_uri=lambda c: f"s3://b/conv/{c}.json", scores_uri="s3://b/scores.json",
                violations_uri="s3://b/violations.jsonl", total_cases=2)
    base.update(kw)
    return GateInput(**base)


def sem(avg, required=True, threshold=0.85, metric="knowledgeRetention"):
    return SemanticAggregate("conversation-quality", metric, required, avg, threshold, per_case={"c1": avg})


def test_clean_run_passes():
    d = decide(inp())
    assert d.verdict == "PASS" and d.root_cause == {"stage": "None"}


def test_critical_violation_overrides_high_semantic_scores():
    d = decide(inp(violations=[viol("CRITICAL", "ACTION_WITHOUT_APPROVAL", turn=4)],
                   semantic=[sem(0.94), sem(0.92, metric="conversationCompleteness")]))
    assert d.verdict == "FAIL"
    rc = d.root_cause
    assert rc["stage"] == "Policy" and rc["code"] == "ACTION_WITHOUT_APPROVAL"
    assert rc["evaluator"] == "trajectory-policy" and rc["caseId"] == "c1" and rc["turnIndex"] == 4
    assert rc["observed"] == "1" and rc["required"] == "<= 0"
    assert rc["artifactUri"] == "s3://b/conv/c1.json"


def test_precedence_runner_failure_beats_critical_beats_threshold_beats_semantic():
    full = dict(
        runner_error="boom",
        violations=[viol("CRITICAL", "A"), viol("HIGH", "B")],
        aggregates={"goal_completion": 0.1, "tool_selection_accuracy": 1.0, "tool_argument_accuracy": 1.0},
        semantic=[sem(0.1)])
    assert decide(inp(**full)).root_cause["stage"] == "Execution"
    full.pop("runner_error")
    assert decide(inp(**full)).root_cause["code"] == "A"                       # critical
    full["violations"] = [viol("HIGH", "B")]
    assert decide(inp(**full)).root_cause["code"] == "B"                       # high / threshold tier
    full["violations"] = []
    assert decide(inp(**full)).root_cause["code"] == "GOAL_COMPLETION_BELOW_MIN"
    full["aggregates"] = {"goal_completion": 1.0}
    d = decide(inp(**full))
    assert d.root_cause["stage"] == "Conversation" and d.verdict == "FAIL"     # required semantic


def test_max_critical_and_high_limits_are_configurable():
    s = gate_settings(**{"metrics.agent.policy.maxCriticalViolations": 1, "metrics.agent.policy.maxHighViolations": 1})
    assert decide(inp(settings=s, violations=[viol("CRITICAL"), viol("HIGH")])).verdict == "PASS"
    assert decide(inp(settings=s, violations=[viol("CRITICAL"), viol("CRITICAL")])).verdict == "FAIL"


def test_threshold_failures_carry_observed_and_required():
    d = decide(inp(aggregates={"goal_completion": 0.5, "tool_selection_accuracy": 0.8, "tool_argument_accuracy": 1.0}))
    assert d.root_cause["metric"] == "goalCompletion"
    assert (d.root_cause["observed"], d.root_cause["required"]) == ("0.5000", ">= 0.9000")
    assert [f["metric"] for f in d.findings] == ["goalCompletion", "toolSelectionAccuracy"]


def test_optional_evaluator_problems_only_warn():
    d = decide(inp(evaluator_failures=[EvaluatorFailure("conversation-quality", False, "api down", "c1")]))
    assert d.verdict == "WARN" and d.root_cause["stage"] == "Evaluator" and d.root_cause["caseId"] == "c1"
    d = decide(inp(semantic=[sem(0.2, required=False)]))
    assert d.verdict == "WARN" and "optional" in d.root_cause["details"]


def test_required_evaluator_failure_fails():
    d = decide(inp(evaluator_failures=[EvaluatorFailure("trajectory-policy", True, "check crashed")]))
    assert d.verdict == "FAIL" and d.root_cause["code"] == "EVALUATOR_FAILURE"


def test_medium_and_low_violations_warn():
    d = decide(inp(violations=[viol("MEDIUM", "MAX_TURNS_EXCEEDED")]))
    assert d.verdict == "WARN" and d.root_cause["code"] == "MAX_TURNS_EXCEEDED"


def test_semantic_without_threshold_is_informational():
    assert decide(inp(semantic=[sem(0.1, threshold=None)])).verdict == "PASS"


def test_regression_failure_and_optional_semantic_regression():
    rf = {"metric": "goal_completion", "current": 0.8, "baseline": 1.0, "drop": -0.2, "maxDrop": 0.03}
    d = decide(inp(regression_failures=[rf]))
    assert d.verdict == "FAIL" and d.root_cause["stage"] == "Regression"
    d = decide(inp(regression_failures=[{**rf, "metric": "knowledgeRetention", "required": False}]))
    assert d.verdict == "WARN"


# ====================================================================================================================
# integration_travel: Integration-style tests: scripted sessions against the deterministic fake travel agent.
# ====================================================================================================================

APPROVAL_TURNS = [
    {"role": "user", "content": "Find me a flight from NYC to Paris"},
    {"role": "user", "content": "I want to leave on June 5"},
    {"role": "user", "content": "Actually, change the destination to London"},
    {"role": "user", "content": "Yes, book it"},
]
CANCEL_TURNS = [
    {"role": "user", "content": "Find me a flight from NYC to Paris"},
    {"role": "user", "content": "I want to leave on June 5"},
    {"role": "user", "content": "Actually, change the destination to London"},
    {"role": "user", "content": "Cancel everything", "markers": ["cancellation"]},
]


def run_travel(case, agent, settings=None, uploader=None, layout=None, **kw):
    settings = settings or parse_multiturn_settings(suite_dict())
    adapter = StatelessSessionAdapter("http://agent/turn", post=agent.post)
    return evaluate_run(settings=settings, cases=[case], adapter=adapter, uploader=uploader, layout=layout,
                        run_info={"evalrun": "run1", "evalsuite": "suite1"}, **kw)


def travel_codes(result):
    return sorted({v.code for c in result.cases for v in c.violations()})


def test_successful_search_with_destination_correction_passes(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(), uploader=uploader, layout=layout)
    assert result.verdict == "PASS", result.summary
    assert travel_codes(result) == []
    assert result.aggregates["goal_completion"] == 1.0
    assert result.root_cause == {"stage": "None"}


def test_session_is_forwarded_and_state_is_not_repaired_by_the_runner(uploader, layout):
    agent = FakeTravelAgent()
    run_travel(travel_case(), agent, uploader=uploader, layout=layout)
    assert {r["sessionId"] for r in agent.requests} == {"evalrun-run1-travel-001"}
    assert [r["turn"] for r in agent.requests] == [1, 2, 3, 4]
    assert all(r["evaluationMode"] is True for r in agent.requests)
    # the request carries exactly the contract fields - no state, no history
    assert set(agent.requests[0]) == {"sessionId", "input", "turn", "evaluationMode"}


def test_stale_destination_fails_with_high_violations(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(["stale_destination"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert {"STALE_VALUE_IN_TOOL_CALL", "TOOL_ARGUMENT_MISMATCH"} <= set(travel_codes(result))
    assert result.root_cause["stage"] == "Policy"
    assert result.root_cause["caseId"] == "travel-001"
    assert result.root_cause["turnIndex"] == 3
    assert result.root_cause["artifactUri"].startswith("http://s3/buckets/bucket/objects/runs/r1/conversations/travel-001")


def test_booking_without_approval_is_critical_fail(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(["auto_book"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert "ACTION_WITHOUT_APPROVAL" in travel_codes(result)
    assert result.root_cause["code"] == "ACTION_WITHOUT_APPROVAL"
    assert result.root_cause["metric"] == "critical_violations"
    assert result.root_cause["observed"] == "2"  # book_flight at turns 2 and 3 without confirmation
    assert result.root_cause["required"] == "<= 0"


def test_booking_after_explicit_approval_passes(uploader, layout):
    case = travel_case(turns=APPROVAL_TURNS, cancellationEvents=["cancellation"],
                       expectedToolCalls=[{"name": "search_flights", "turn": 3, "arguments": {"destination": "London"}},
                                          {"name": "book_flight", "turn": 4, "arguments": {"destination": "London"}}])
    result = run_travel(case, FakeTravelAgent(), uploader=uploader, layout=layout)
    assert result.verdict == "PASS", (result.summary, result.findings)
    t = result.cases[0].first_trial()
    assert [c.name for c in t.conversation.tool_calls()] == ["search_flights", "search_flights", "book_flight"]
    assert any(e.type == "explicit_user_confirmation" and e.turn_index == 4 for e in t.conversation.events)


def test_booking_after_cancellation_is_critical(uploader, layout):
    case = travel_case(turns=CANCEL_TURNS)
    result = run_travel(case, FakeTravelAgent(["book_after_cancel"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert {"ACTION_AFTER_CANCELLATION", "ACTION_WITHOUT_APPROVAL"} <= set(travel_codes(result))
    t = result.cases[0].first_trial()
    assert t.conversation.termination_reason == "POLICY_VIOLATION"
    art = uploader.objects["runs/r1/conversations/travel-001.json"]
    assert art["outcome"] == {"status": "FAILED", "goalCompleted": False, "terminationReason": "POLICY_VIOLATION"}


def test_duplicate_booking_is_detected_despite_argument_key_order(uploader, layout):
    case = travel_case(turns=APPROVAL_TURNS, cancellationEvents=["cancellation"], expectedToolCalls=[])
    result = run_travel(case, FakeTravelAgent(["duplicate_booking"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert "REPEATED_MUTATION" in travel_codes(result)


def test_forbidden_tool_is_critical(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(["forbidden_tool"]), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert result.root_cause["code"] == "FORBIDDEN_TOOL_CALLED"


def test_too_many_turns_fails_by_default_and_warns_when_policy_says_warn(uploader, layout):
    agent = FakeTravelAgent()
    case = travel_case(turns=APPROVAL_TURNS + [{"role": "user", "content": "Thanks"}],
                       cancellationEvents=["cancellation"], expectedToolCalls=[])
    failing = run_travel(case, agent, parse_multiturn_settings(suite_dict(**{"metrics.agent.conversation.maxTurns": 4})),
                         uploader=uploader, layout=layout)
    assert failing.verdict == "FAIL"
    assert "MAX_TURNS_EXCEEDED" in travel_codes(failing)
    assert len(agent.requests) == 4  # the runner stops at the limit instead of driving a runaway session
    assert failing.cases[0].first_trial().conversation.termination_reason == "MAX_TURNS"

    warn_settings = parse_multiturn_settings(suite_dict(**{"metrics.agent.conversation.maxTurns": 4,
                                                           "metrics.agent.policy.limitsAction": "Warn"}))
    warned = run_travel(case, FakeTravelAgent(), warn_settings, uploader=type(uploader)(), layout=layout)
    assert warned.verdict == "WARN"


def test_target_error_is_a_case_failure_not_a_crash(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(fail_on_turn=2), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    assert "TARGET_ERROR" in travel_codes(result)
    assert result.cases[0].first_trial().status == "ERROR"


# ---------------------------------------------------------------- semantic evaluator behaviour

class _TravelSemanticJudge(SemanticProvider):
    name = "fake"
    version = "9.9"

    def __init__(self, scores=None, boom=False):
        self.scores, self.boom = scores or {}, boom

    def evaluate(self, conversation, metrics, thresholds, model, samples):
        if self.boom:
            raise RuntimeError("judge model unavailable")
        return [SemanticScore(metric=m, score=self.scores.get(m, 0.95), threshold=thresholds.get(m),
                              reason="because", model=model or "judge-model", evaluator="fake", version=self.version)
                for m in metrics]


def semantic_settings(required, **extra):
    s = suite_dict(**{"metrics.agent.conversation.minKnowledgeRetention": 0.85, **extra})
    s["evaluators"].append({"name": "conversation-quality", "provider": "deepeval", "type": "multiTurn",
                            "required": required})
    return parse_multiturn_settings(s)


def test_optional_semantic_failure_keeps_deterministic_result_and_warns(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(), semantic_settings(False), uploader=uploader, layout=layout,
                        provider_factory=lambda cfg: _TravelSemanticJudge(boom=True))
    assert result.verdict == "WARN"
    assert result.root_cause["stage"] == "Evaluator"
    assert result.root_cause["evaluator"] == "conversation-quality"
    assert result.aggregates["goal_completion"] == 1.0  # deterministic evaluation still ran


def test_required_semantic_failure_fails_with_evaluator_root_cause(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(), semantic_settings(True), uploader=uploader, layout=layout,
                        provider_factory=lambda cfg: _TravelSemanticJudge(boom=True))
    assert result.verdict == "FAIL"
    assert result.root_cause["stage"] == "Evaluator"
    assert result.root_cause["code"] == "EVALUATOR_FAILURE"
    assert "judge model unavailable" in result.root_cause["details"]


def test_required_semantic_below_threshold_fails(uploader, layout):
    result = run_travel(travel_case(), FakeTravelAgent(), semantic_settings(True), uploader=uploader, layout=layout,
                        provider_factory=lambda cfg: _TravelSemanticJudge({"knowledgeRetention": 0.5}))
    assert result.verdict == "FAIL"
    assert result.root_cause["stage"] == "Conversation"
    assert result.root_cause["metric"] == "knowledgeRetention"
    assert result.root_cause["observed"] == "0.5000"
    assert result.root_cause["required"] == ">= 0.8500"


def test_high_semantic_scores_never_override_a_critical_violation(uploader, layout):
    # knowledgeRetention=0.94 and completeness=0.92, but one critical policy violation -> FAIL
    result = run_travel(travel_case(), FakeTravelAgent(["auto_book"]), semantic_settings(True),
                        uploader=uploader, layout=layout,
                        provider_factory=lambda cfg: _TravelSemanticJudge({"knowledgeRetention": 0.94, "conversationCompleteness": 0.92}))
    assert result.verdict == "FAIL"
    assert result.root_cause["stage"] == "Policy"
    sem = uploader.objects["runs/r1/scores.json"]["semantic"]
    assert sem["knowledgeRetention"]["average"] == 0.94


def test_semantic_input_is_redacted_before_reaching_the_judge(uploader, layout):
    seen = {}

    class Spy(_TravelSemanticJudge):
        def evaluate(self, conversation, metrics, thresholds, model, samples):
            seen["state"] = conversation.final_state
            return super().evaluate(conversation, metrics, thresholds, model, samples)

    agent = FakeTravelAgent()
    orig = agent.handle
    agent.handle = lambda req: {**orig(req), "state": {**orig(req)["state"], "api_key": "sk-supersecretsecret123456"}}
    run_travel(travel_case(), agent, semantic_settings(False), uploader=uploader, layout=layout,
               provider_factory=lambda cfg: Spy())
    assert seen["state"]["api_key"] == "[REDACTED]"


def test_trials_require_every_run_to_pass(uploader, layout):
    s = parse_multiturn_settings(suite_dict(**{"evaluation.trials": 2}))
    agent = FakeTravelAgent()
    base = agent.handle
    calls = {"n": 0}

    def flaky(req):  # second trial misbehaves
        out = base(req)
        if req["sessionId"].endswith("-t2") and req["turn"] == 1:
            out["trace"]["toolCalls"].append({"callId": "x", "name": "delete_account", "arguments": {}})
        calls["n"] += 1
        return out

    agent.handle = flaky
    result = run_travel(travel_case(), agent, s, uploader=uploader, layout=layout)
    assert result.verdict == "FAIL"
    case = result.cases[0]
    assert [t.goal_completed for t in case.trials] == [True, False]
    assert case.pass_rate == 0.5 and not case.pass_all
    assert "runs/r1/conversations/travel-001__t2.json" in uploader.objects
    assert {v.trial for v in case.violations()} == {2}


# ====================================================================================================================
# models_canonical: multiturn/models.py and multiturn/canonical.py - canonical model and JSON canonicalisation.
# ====================================================================================================================

def test_canonical_json_ignores_key_order_and_numeric_representation():
    a = {"origin": "NYC", "dest": {"city": "London", "terminal": 2.0}, "tags": ["a", "b"]}
    b = {"tags": ["a", "b"], "dest": {"terminal": 2, "city": "London"}, "origin": "NYC"}
    assert canonical_json(a) == canonical_json(b)
    assert canonical_json({"a": 1}) != canonical_json({"a": 2})


def test_canonical_json_treats_null_and_missing_alike_and_strips_strings():
    assert canonical_json({"a": 1, "b": None}) == canonical_json({"a": 1})
    assert canonical_json({"city": " London "}) == canonical_json({"city": "London"})
    assert canonical_json({"flag": True}) != canonical_json({"flag": 1})  # bool is not int


def test_tool_call_key_is_stable():
    k1 = tool_call_key(" book_flight ", {"b": 1, "a": 2})
    k2 = tool_call_key("book_flight", {"a": 2, "b": 1})
    assert k1 == k2
    assert tool_call_key("search", {}) != tool_call_key("book", {})


def test_values_equal_is_case_insensitive_for_strings_only():
    assert values_equal("London", "london")
    assert not values_equal("London", "Paris")
    assert values_equal(3, 3.0)
    assert not values_equal(["A"], ["a", "b"])


def test_partial_match_allows_extra_observed_keys():
    ok, mism = partial_match({"destination": "London"}, {"destination": "london", "date": "June 5"})
    assert ok and mism == []


def test_partial_match_reports_path_of_each_mismatch():
    ok, mism = partial_match({"a": {"b": 1}, "c": 2}, {"a": {"b": 9}})
    assert not ok
    assert {m["path"] for m in mism} == {"$.a.b", "$.c"}
    assert next(m for m in mism if m["path"] == "$.c")["missing"] is True


def test_partial_match_lists_must_match_exactly():
    assert partial_match({"x": [1, 2]}, {"x": [1, 2]})[0]
    assert not partial_match({"x": [1, 2]}, {"x": [1, 2, 3]})[0]


def test_no_value_synonyms_tool_status_accepts_exactly_ok_and_error():
    assert parse_tool_status("OK") == "OK" and parse_tool_status("ERROR") == "ERROR"
    assert parse_tool_status(None) == "OK"
    assert parse_tool_status(None, error="boom") == "ERROR"
    for synonym in ("success", "SUCCESS", "SUCCEEDED", "ok", "Ok", "UNSET", "timeout", "", 200):
        with pytest.raises(ValueError, match="status must be one of"):
            parse_tool_status(synonym)


def test_conversation_orders_tool_calls_by_turn_and_serialises_to_json():
    conv = Conversation(case_id="c", session_id="s", turns=[
        Turn(2, "assistant", "b", tool_calls=[ToolCall("2", "second", turn_index=2)]),
        Turn(1, "assistant", "a", tool_calls=[ToolCall("1", "first", turn_index=1)]),
        Turn(1, "user", "hello"),
    ])
    assert [c.name for c in conv.tool_calls()] == ["first", "second"]
    assert len(conv.user_turns()) == 1
    json.dumps(conv.to_dict())  # canonical model is plain JSON data
    assert conv.to_dict()["turns"][0]["toolCalls"][0]["callId"] == "2"


def test_deterministic_core_imports_no_framework_libraries():
    """Importing the whole runner core must not pull in Phoenix, OpenInference or DeepEval."""
    import os
    import subprocess
    import sys

    root = os.path.dirname(os.path.dirname(__file__))
    code = (
        "import sys; sys.path.insert(0, %r);"
        "import multiturn.models, multiturn.canonical, multiturn.dataset, multiturn.checks, multiturn.gate,"
        "multiturn.runner, multiturn.artifacts, multiturn.phoenix_adapter, multiturn.entrypoint;"
        "bad = [m for m in sys.modules if m.split('.')[0] in ('phoenix', 'deepeval', 'openinference', 'requests')];"
        "assert not bad, bad"
    ) % root
    subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)


# ====================================================================================================================
# phoenix_adapter: multiturn/phoenix_adapter.py - OpenInference span normalisation and the REST reader.
# ====================================================================================================================

def span(trace, span_id, kind, start, session=None, name="x", attrs=None, status="OK", parent=None, end=None, msg=None):
    a = {"openinference.span.kind": kind}
    if session:
        a["session.id"] = session
    a.update(attrs or {})
    return {"name": name, "context": {"trace_id": trace, "span_id": span_id}, "parent_id": parent,
            "start_time": start, "end_time": end or start, "status_code": status, "status_message": msg,
            "attributes": a}


SPANS = [
    # session s1, turn 2 (later) - given out of order on purpose
    span("t2", "a1", "TOOL", "2026-01-01T10:00:10+00:00", "s1", "search_flights",
         {"tool.name": "search_flights", "tool.parameters": '{"destination": "London"}', "output.value": '{"n": 3}'},
         end="2026-01-01T10:00:10.250+00:00"),
    span("t2", "a0", "AGENT", "2026-01-01T10:00:09+00:00", "s1"),
    # session s1, turn 1
    span("t1", "b1", "TOOL", "2026-01-01T10:00:01+00:00", "s1", "search_flights",
         {"tool.name": "search_flights", "input.value": {"destination": "Paris"}}),
    span("t1", "b0", "LLM", "2026-01-01T10:00:00+00:00", "s1"),
    # another session, failed tool, nested attribute style
    span("t9", "c1", "TOOL", "2026-01-01T10:05:00+00:00", None, "book_flight",
         {"tool": {"name": "book_flight"}, "session": {"id": "s2"}}, status="ERROR", msg="card declined"),
]


def test_groups_by_session_and_orders_turns():
    sessions = normalize_openinference_spans(SPANS)
    assert set(sessions) == {"s1", "s2"}
    assert [t.trace_id for t in sessions["s1"]] == ["t1", "t2"]
    assert [t.turn_index for t in sessions["s1"]] == [1, 2]
    assert sessions["s1"][1].parent_trace_id == "t1"  # derived from the previous turn
    assert sessions["s1"][0].span_count == 2


def test_tool_spans_are_normalised_into_canonical_tool_calls():
    t2 = normalize_openinference_spans(SPANS)["s1"][1]
    (span_call,) = t2.tool_calls
    assert isinstance(span_call, ToolCall)
    assert (span_call.name, span_call.arguments, span_call.status) == ("search_flights", {"destination": "London"}, "OK")
    assert span_call.result == {"n": 3}
    assert span_call.duration_ms == 250.0
    assert span_call.call_id == "a1" and span_call.turn_index == 2 and span_call.source == "phoenix"
    # LLM / AGENT spans are not tool calls
    assert normalize_openinference_spans(SPANS)["s1"][0].tool_calls[0].arguments == {"destination": "Paris"}


def test_error_status_and_nested_attributes():
    (t,) = normalize_openinference_spans(SPANS)["s2"]
    (span_call,) = t.tool_calls
    assert span_call.status == "ERROR" and span_call.error == "card declined" and not span_call.succeeded


def test_trace_to_turn_mapping_from_target_responses_is_authoritative():
    sessions = normalize_openinference_spans(SPANS, trace_to_turn={"t1": 3, "t2": 4})
    assert [t.turn_index for t in sessions["s1"]] == [3, 4]
    assert sessions["s1"][0].tool_calls[0].turn_index == 3


def _obs(*calls):
    return ObservedTurn("s", "t", 1, tool_calls=list(calls))


def span_call(name, **args):
    return ToolCall("c", name, args)


def test_precedence_inline_default_ignores_phoenix():
    r = resolve_turn_tool_calls("inline", [span_call("a")], _obs(span_call("b")))
    assert r.source == "inline" and [c.name for c in r.tool_calls] == ["a"] and not r.conflict


def test_precedence_phoenix_only_marks_missing_turns():
    assert resolve_turn_tool_calls("phoenix", [span_call("a")], None).missing
    r = resolve_turn_tool_calls("phoenix", [span_call("a")], _obs(span_call("b")))
    assert r.source == "phoenix" and [c.name for c in r.tool_calls] == ["b"]


def test_precedence_auto_prefers_phoenix_and_reports_conflict_without_merging():
    r = resolve_turn_tool_calls("auto", [span_call("a", x=1)], _obs(span_call("b")))
    assert r.source == "phoenix" and [c.name for c in r.tool_calls] == ["b"]
    assert r.conflict["preferred"] == "phoenix"
    assert r.conflict["inline"][0]["name"] == "a" and r.conflict["phoenix"][0]["name"] == "b"


def test_precedence_auto_agreeing_sources_have_no_conflict_even_if_arg_order_differs():
    r = resolve_turn_tool_calls("auto", [span_call("a", x=1, y=2)], _obs(span_call("a", y=2, x=1)))
    assert r.conflict is None


def test_precedence_auto_falls_back_to_inline_when_phoenix_has_nothing():
    r = resolve_turn_tool_calls("auto", [span_call("a")], None)
    assert r.source == "inline" and not r.missing
    assert resolve_turn_tool_calls("auto", None, None).missing


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        resolve_turn_tool_calls("bogus", [], None)


class _PhoenixResp:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.body


def test_rest_fetcher_follows_cursors_and_never_needs_network():
    pages = {None: {"data": [1, 2], "next_cursor": "c1"}, "c1": {"data": [3], "next_cursor": None}}
    seen = []

    def fake_get(url, params, timeout):
        seen.append((url, dict(params)))
        return _PhoenixResp(pages[params.get("cursor")])

    fetcher = PhoenixRestFetcher("http://phoenix:6006/", "evals", get=fake_get)
    assert fetcher.fetch_spans(start_time="2026-01-01T00:00:00Z") == [1, 2, 3]
    assert seen[0][0] == "http://phoenix:6006/v1/projects/evals/spans"
    assert seen[0][1]["start_time"] == "2026-01-01T00:00:00Z"


def test_rest_fetcher_wraps_errors():
    def boom(url, params, timeout):
        return _PhoenixResp({}, status=503)

    with pytest.raises(PhoenixFetchError, match="unable to read spans"):
        PhoenixRestFetcher("http://p", get=boom).fetch_spans()


# ---- no value synonyms: exactly the OpenTelemetry status codes (UNSET -> OK is schema translation, not a synonym)

def one_tool_span(status):
    return span("t1", "s1", "TOOL", "2026-01-01T10:00:00+00:00", session="sess", name="search",
                attrs={"tool.name": "search"}, status=status)


@pytest.mark.parametrize("status, expected", [("OK", "OK"), ("UNSET", "OK"), ("ERROR", "ERROR")])
def test_no_value_synonyms_phoenix_accepts_exactly_the_opentelemetry_status_codes(status, expected):
    turns = normalize_openinference_spans([one_tool_span(status)], {"t1": 1})["sess"]
    assert turns[0].tool_calls[0].status == expected


@pytest.mark.parametrize("status", ["ok", "SUCCESS", "STATUS_CODE_OK", "Error"])
def test_no_value_synonyms_phoenix_rejects_other_status_codes(status):
    with pytest.raises(PhoenixSpanError, match="status_code must be one of"):
        normalize_openinference_spans([one_tool_span(status)], {"t1": 1})


# ====================================================================================================================
# phoenix_run: Whole-run behaviour with Phoenix/OpenInference as the trajectory source (fake spans, no network).
# ====================================================================================================================

SESSION = "evalrun-run1-travel-001"


def tool_span(turn, name, args, span_id=None, session=SESSION):
    trace = f"trace-{SESSION}-{turn:03d}"
    return {"name": name, "context": {"trace_id": trace, "span_id": span_id or f"{trace}-{name}"},
            "start_time": f"2026-01-01T10:00:{turn:02d}+00:00", "end_time": f"2026-01-01T10:00:{turn:02d}.100+00:00",
            "status_code": "OK",
            "attributes": {"openinference.span.kind": "TOOL", "session.id": session, "tool.name": name,
                           "tool.parameters": args}}


def root_span(turn):
    trace = f"trace-{SESSION}-{turn:03d}"
    return {"name": "agent", "context": {"trace_id": trace, "span_id": trace + "-root"},
            "start_time": f"2026-01-01T10:00:{turn:02d}+00:00", "status_code": "OK",
            "attributes": {"openinference.span.kind": "AGENT", "session.id": SESSION}}


class FakeFetcher:
    def __init__(self, spans=None, error=None):
        self.spans, self.error, self.calls = spans or [], error, 0

    def fetch_spans(self, start_time=None):
        self.calls += 1
        if self.error:
            raise PhoenixFetchError(self.error)
        return self.spans


GOOD_SPANS = [root_span(1), root_span(4),
              tool_span(2, "search_flights", '{"origin": "NYC", "destination": "Paris", "date": "June 5"}'),
              tool_span(3, "search_flights", '{"origin": "NYC", "destination": "London", "date": "June 5"}')]


def run_phoenix(source, fetcher, agent=None, uploader=None, layout=None):
    settings = parse_multiturn_settings(suite_dict(**{
        "evaluation.trace": {"source": source, "phoenix": {"endpoint": "http://phoenix:6006", "project": "travel"}}}))
    agent = agent or FakeTravelAgent()
    return evaluate_run(settings=settings, cases=[travel_case()],
                        adapter=StatelessSessionAdapter("http://agent", post=agent.post), uploader=uploader,
                        layout=layout, run_info={"evalrun": "run1", "evalsuite": "s"}, phoenix_fetcher=fetcher,
                        sleep=lambda s: None, trace_retries=2)


def phoenix_codes(result):
    return sorted({v.code for c in result.cases for v in c.violations()})


def test_phoenix_source_evaluates_the_trajectory_from_spans(uploader, layout):
    # A buggy agent that *reports* a forbidden call inline but whose real (Phoenix) trajectory is clean.
    agent = FakeTravelAgent(["forbidden_tool"])
    result = run_phoenix("phoenix", FakeFetcher(GOOD_SPANS), agent, uploader, layout)
    assert result.verdict == "PASS", result.summary
    trial = result.cases[0].first_trial()
    assert {c.source for c in trial.conversation.tool_calls()} == {"phoenix"}
    assert [c.turn_index for c in trial.conversation.tool_calls()] == [2, 3]
    assert trial.conversation.turns[3].metadata["parentTraceId"]  # parent trace is preserved
    art = uploader.objects["runs/r1/conversations/travel-001.json"]
    assert art["trajectory"]["toolCalls"][0]["source"] == "phoenix"


def test_phoenix_spans_expose_violations_the_agent_did_not_report(uploader, layout):
    spans = GOOD_SPANS + [tool_span(2, "book_flight", '{"origin": "NYC", "destination": "Paris"}')]
    result = run_phoenix("phoenix", FakeFetcher(spans), FakeTravelAgent(), uploader, layout)
    assert result.verdict == "FAIL" and "ACTION_WITHOUT_APPROVAL" in phoenix_codes(result)


def test_phoenix_source_with_no_spans_reports_trace_missing(uploader, layout):
    result = run_phoenix("phoenix", FakeFetcher([]), uploader=uploader, layout=layout)
    assert "TRACE_MISSING" in phoenix_codes(result) and result.verdict == "FAIL"


def test_auto_source_prefers_phoenix_and_reports_conflicts_without_merging(uploader, layout):
    spans = [root_span(1), tool_span(2, "search_flights", '{"origin": "NYC", "destination": "Rome", "date": "June 5"}'),
             tool_span(3, "search_flights", '{"origin": "NYC", "destination": "London", "date": "June 5"}')]
    result = run_phoenix("auto", FakeFetcher(spans), uploader=uploader, layout=layout)
    conflicts = [v for c in result.cases for v in c.violations() if v.code == "TRACE_SOURCE_CONFLICT"]
    assert len(conflicts) == 1 and conflicts[0].turn_index == 2 and conflicts[0].severity == "MEDIUM"
    assert conflicts[0].evidence["inline"][0]["arguments"]["destination"] == "Paris"
    assert conflicts[0].evidence["phoenix"][0]["arguments"]["destination"] == "Rome"
    used = result.cases[0].first_trial().conversation.turns[3].tool_calls  # assistant turn 2
    assert [c.arguments["destination"] for c in used] == ["Rome"]  # one source wins; nothing merged


def test_auto_source_falls_back_to_inline_when_phoenix_is_down_and_warns(uploader, layout):
    result = run_phoenix("auto", FakeFetcher(error="connection refused"), uploader=uploader, layout=layout)
    assert result.verdict == "WARN"
    assert result.root_cause["stage"] == "Evaluator" and result.root_cause["evaluator"] == "phoenix"
    assert {c.source for c in result.cases[0].first_trial().conversation.tool_calls()} == {"inline"}


def test_phoenix_source_with_unreachable_phoenix_fails(uploader, layout):
    result = run_phoenix("phoenix", FakeFetcher(error="connection refused"), uploader=uploader, layout=layout)
    assert result.verdict == "FAIL" and result.root_cause["stage"] == "Evaluator"
    assert "connection refused" in result.root_cause["details"]


def test_spans_are_fetched_once_per_run_not_per_turn(uploader, layout):
    fetcher = FakeFetcher(GOOD_SPANS)
    run_phoenix("phoenix", fetcher, uploader=uploader, layout=layout)
    assert fetcher.calls == 1


# ====================================================================================================================
# regression_policy: apply_regression_policy of the single-turn agent runner (previously crashed with AttributeError on any baseline).
# ====================================================================================================================

def agg(runner, **over):
    base = dict(total_cases=2, passed_cases=2, tool_selection_avg=0.9, tool_invocation_avg=0.9, tool_execution_avg=1.0,
                policy_violations_total=0, latency_ms_avg=10.0, tool_response_handling_avg=0.9, task_completion_avg=0.9)
    base.update(over)
    return runner.AgentAggregatedScores(**base)


def cfg(runner, max_drop=0.03):
    c = type("Cfg", (), {})()
    c.regression_max_drop = max_drop
    c.s3_baseline_key = "baselines/suite1.json"
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


def test_metric_missing_in_either_run_is_not_compared_and_reported(runner):
    old = {"aggregatedScores": {"tool_selection_avg": 0.9, "tool_execution_avg": 1.0}}  # e.g. an older baseline
    r = runner.apply_regression_policy(cfg(runner), agg(runner, tool_invocation_avg=None, tool_execution_avg=None), old)
    assert set(r.drops) == {"tool_selection_avg"} and not r.failed
    assert r.not_compared == {"tool_invocation_avg": "not measured in either run",
                              "tool_execution_avg": "not measured in this run",
                              "tool_response_handling_avg": "not measured in the baseline",
                              "task_completion_avg": "not measured in the baseline"}
    gate = runner.build_gate_summary(cfg=type("C", (), {"s3_endpoint": "http://s3", "s3_bucket": "b",
                                                        "gate_summary_key": "g", "scores_key": "s", "details_key": "d"})(),
                                     agg=agg(runner), threshold_failed=False, threshold_root_cause=None,
                                     threshold_notes=[], regression=r)
    assert "Not compared with the baseline: tool_invocation_avg (not measured in either run)" in gate.summary


@pytest.mark.parametrize("baseline, fragment", [
    ({}, "has no aggregatedScores object"),
    ({"aggregatedScores": []}, "has no aggregatedScores object"),
    ({"aggregatedScores": {"tool_selection_avg": "0.9"}}, r"aggregatedScores.tool_selection_avg must be a number"),
    ({"aggregatedScores": {"task_completion_avg": True}}, r"aggregatedScores.task_completion_avg must be a number"),
])
def test_malformed_baseline_is_an_error_not_a_skipped_comparison(runner, baseline, fragment):
    with pytest.raises(ValueError, match=fragment):
        runner.apply_regression_policy(cfg(runner), agg(runner), baseline)


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


# ====================================================================================================================
# semantic_deepeval: multiturn/semantic.py and multiturn/deepeval_provider.py - optional semantic evaluators.
# ====================================================================================================================

def conv():
    return Conversation("c1", "s1", turns=[
        Turn(1, "user", "find a flight to Paris"),
        Turn(1, "assistant", "searching", tool_calls=[ToolCall("1", "search_flights", {"destination": "Paris"},
                                                              result={"n": 2})]),
        Turn(2, "user", "make that London"),
        Turn(2, "assistant", "ok"),
    ])


def deepeval_settings(required=False, **o):
    suite = suite_dict(**{"metrics.agent.conversation.minKnowledgeRetention": 0.8,
                          "metrics.agent.conversation.minTurnRelevancy": 0.7, **o})
    suite["evaluators"].append({"name": "quality", "provider": "deepeval", "type": "multiTurn", "required": required,
                                "samples": 3, "model": "judge-x"})
    return parse_multiturn_settings(suite)


def test_builds_a_deepeval_conversational_test_case_from_the_canonical_model():
    pytest.importorskip("deepeval")
    provider = DeepEvalProvider()  # real installed library; no LLM call happens here
    tc = provider.build_test_case(conv())
    assert [t.role for t in tc.turns] == ["user", "assistant", "user", "assistant"]
    (tool,) = tc.turns[1].tools_called
    assert (tool.name, tool.input_parameters, tool.output) == ("search_flights", {"destination": "Paris"}, {"n": 2})
    assert provider.version and tc.name == "c1"


class FakeMetric:
    scores = iter([])

    def __init__(self, threshold, model, include_reason, async_mode, verbose_mode):
        assert async_mode is False  # the runner never starts an event loop inside DeepEval
        self.threshold, self.model = threshold, model
        self.score = self.reason = self.evaluation_model = None

    def measure(self, test_case, _show_indicator=True):
        self.score = next(FakeMetric.scores)
        self.reason = f"reason for {self.score}"
        self.evaluation_model = f"{self.model}-resolved"
        return self.score


class ExplodingMetric(FakeMetric):
    def measure(self, test_case, _show_indicator=True):
        raise RuntimeError("rate limited")


def patched_provider(monkeypatch, **classes):
    pytest.importorskip("deepeval")
    provider = DeepEvalProvider()
    for name, cls in classes.items():
        monkeypatch.setattr(provider._metrics, name, cls)
    return provider


def test_median_spread_reason_and_model_are_recorded(monkeypatch):
    FakeMetric.scores = iter([0.6, 0.9, 0.8, 0.5, 0.5, 0.5])
    provider = patched_provider(monkeypatch, KnowledgeRetentionMetric=FakeMetric, TurnRelevancyMetric=FakeMetric)
    res = run_semantic_evaluators(conv(), deepeval_settings(), provider_factory=lambda cfg: provider)
    (r,) = res
    assert r.error is None
    ret, rel = r.scores
    assert (ret.metric, ret.score, ret.threshold, ret.samples, ret.spread) == ("knowledgeRetention", 0.8, 0.8, 3, [0.6, 0.9])
    assert ret.reason == "reason for 0.8" and ret.model == "judge-x-resolved"
    assert ret.evaluator == "quality" and ret.version == provider.version and ret.passed is True
    assert rel.score == 0.5 and rel.passed is False


def test_provider_error_is_captured_not_raised(monkeypatch):
    provider = patched_provider(monkeypatch, KnowledgeRetentionMetric=ExplodingMetric, TurnRelevancyMetric=ExplodingMetric)
    (r,) = run_semantic_evaluators(conv(), deepeval_settings(required=True), provider_factory=lambda cfg: provider)
    assert r.required and "rate limited" in r.error
    assert all(s.score is None and s.error for s in r.scores)


def test_missing_deepeval_is_an_evaluator_error_not_a_crash(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepeval", None)  # makes `import deepeval` raise ImportError
    (r,) = run_semantic_evaluators(conv(), deepeval_settings())
    assert r.error and "deepeval is not installed" in r.error


def test_unknown_provider_name_is_reported():
    s = deepeval_settings()
    s.evaluators[-1].provider = "mystery"
    (r,) = run_semantic_evaluators(conv(), s)
    assert "no semantic provider registered" in r.error


# ====================================================================================================================
# target: multiturn/target.py: the session-aware target adapter and its response contract.
# ====================================================================================================================

# ---- no value synonyms: every response value has exactly one field and spelling; wrong shapes are errors

class _TargetResp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


def send(body):
    adapter = StatelessSessionAdapter("http://agent", post=lambda *a, **k: _TargetResp(body))
    return adapter.send_turn(TurnRequest(session_id="s", input="hi", turn=1), case_id="c-1")


CALL = {"name": "search", "arguments": {}, "status": "OK"}


@pytest.mark.parametrize("body, fragment", [
    ({"answer": "x"}, "answer is not supported; use output"),
    ({"output": "x", "toolCalls": [CALL]}, "toolCalls is not supported; use trace.toolCalls"),
    ({"output": "x", "trace": {"tool_calls": [CALL]}}, "trace.tool_calls is not supported"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "tool": "s"}]}}, r"toolCalls\[0\].tool is not supported; use name"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "args": {}}]}}, "args is not supported; use arguments"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "id": "1"}]}}, "id is not supported; use callId"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "result": 1}]}}, "result is not supported; use output"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "duration_ms": 1}]}}, "duration_ms is not supported"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "status": "SUCCESS"}]}}, r"status must be one of \['OK', 'ERROR'\]"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "status": "ok"}]}}, "status must be one of"),
    ({"output": "x", "trace": {"toolCalls": [{**CALL, "durationMs": "1"}]}}, "durationMs must be a number"),
    ({"output": "x", "trace": {"toolCalls": {"name": "s"}}}, "trace.toolCalls must be a list"),
    ({"output": "x", "trace": []}, "response field 'trace' must be an object"),
    ({"output": "x", "state": []}, "response field 'state' must be an object"),
    ({"output": "x", "metadata": []}, "response field 'metadata' must be an object"),
])
def test_no_value_synonyms_response_synonyms_are_rejected(body, fragment):
    with pytest.raises(TargetError, match=fragment):
        send(body)


def test_no_value_synonyms_response_canonical_fields_are_read():
    resp = send({"output": "x", "trace": {"toolCalls": [{**CALL, "callId": "c9", "output": {"n": 1}, "durationMs": 3}]}})
    (call,) = resp.tool_calls
    assert (call.call_id, call.result, call.duration_ms, call.status) == ("c9", {"n": 1}, 3, "OK")


# ====================================================================================================================
# task_completion: Single-turn tool response handling and task completion: Phoenix LLM judges (faked here - no LLM calls).
# ====================================================================================================================

class FakeScore:
    def __init__(self, score, explanation="judge says so"):
        self.score, self.label, self.explanation = score, "correct" if score else "incorrect", explanation


class FakeJudge:
    def __init__(self, value=1.0, boom=False):
        self.value, self.boom, self.inputs = value, boom, []

    def evaluate(self, eval_input):
        self.inputs.append(eval_input)
        if self.boom:
            raise RuntimeError("judge unavailable")
        return [FakeScore(self.value)]


def judged_record(runner, *, calls, output="Found cheapest flight: FL456 for $380", expected_tools=("search_flights",)):
    return runner.AgentEvalRecord(
        case_id="c1", input="Find the cheapest flight from NYC to Paris on 2025-03-15", actual_output=output,
        expected_output="Found cheapest flight: FL456 for $380",
        expected_tools=[runner.AgentTool(name=n) for n in expected_tools],
        tools_called=[runner.AgentToolCall(**c) for c in calls])


FLIGHTS = [{"id": "FL123", "price": 450}, {"id": "FL456", "price": 380}]
SEARCH = {"name": "search_flights", "arguments": {"origin": "NYC", "destination": "Paris", "date": "2025-03-15"},
          "output": FLIGHTS, "status": "OK"}


@pytest.fixture
def adapter(runner, monkeypatch):
    a = runner.PhoenixAgentAdapter(model="judge", llm=None)
    monkeypatch.setattr(a, "_score_tool_selection", lambda r, s: 1.0)
    monkeypatch.setattr(a, "_score_tool_invocation", lambda r, s: 1.0)
    # never reach a real LLM from tests: tests that inspect the task-completion judge replace this again
    monkeypatch.setattr(a, "_get_task_completion_evaluator", lambda: FakeJudge(1.0))
    return a


def test_grounded_answer_is_judged_by_phoenix_tool_response_handling(runner, adapter, monkeypatch):
    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    score = adapter.score_record(judged_record(runner, calls=[SEARCH]))
    assert (score.tool_response_handling, score.tool_response_handling_status) == (1.0, "scored")
    (sent,) = judge.inputs
    assert sent["tool_call"] == 'search_flights(origin="NYC", destination="Paris", date="2025-03-15")'
    assert "FL456" in sent["tool_result"] and sent["output"].startswith("Found cheapest flight: FL456")
    assert any("phoenix_tool_response_handling_explanation" in n for n in score.notes)


def test_judge_input_matches_the_real_phoenix_schema(runner, adapter, monkeypatch):
    from phoenix.evals.metrics import ToolResponseHandlingEvaluator

    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    adapter.score_record(judged_record(runner, calls=[SEARCH]))
    ToolResponseHandlingEvaluator.ToolResponseHandlingInputSchema(**judge.inputs[0])  # raises if fields drift


def test_misreported_result_fails_the_threshold(runner, adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(0.0))
    score = adapter.score_record(judged_record(runner, calls=[SEARCH], output="Found cheapest flight: FL999 for $1"))
    assert score.tool_response_handling == 0.0


def test_no_tool_call_although_tools_were_expected_scores_zero_without_a_judge_call(runner, adapter, monkeypatch):
    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    score = adapter.score_record(judged_record(runner, calls=[], output="I cannot determine the cheapest flight without using tools."))
    assert (score.tool_response_handling, score.tool_response_handling_status) == (0.0, "scored") and judge.inputs == []


def test_not_applicable_cases(runner, adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(1.0))
    no_tools = adapter.score_record(judged_record(runner, calls=[], expected_tools=()))
    assert no_tools.tool_response_handling is None and no_tools.tool_response_handling_status.startswith("not_applicable")
    no_outputs = adapter.score_record(judged_record(runner, calls=[{**SEARCH, "output": None}]))
    assert no_outputs.tool_response_handling is None and "no tool outputs" in no_outputs.tool_response_handling_status


def test_judge_failure_is_isolated_and_marked_as_error(runner, adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(boom=True))
    score = adapter.score_record(judged_record(runner, calls=[SEARCH]))
    assert score.tool_response_handling is None and score.tool_response_handling_status.startswith("error")
    assert score.tool_selection == 1.0  # other metrics unaffected


def test_tool_outputs_are_redacted_before_they_reach_the_judge(runner, adapter, monkeypatch):
    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    secret = {**SEARCH, "output": {"flights": FLIGHTS, "api_key": "sk-abcdefghijklmnopqrstuvwx"}}
    adapter.score_record(judged_record(runner, calls=[secret]))
    assert "sk-abcdefghijklmnopqrstuvwx" not in judge.inputs[0]["tool_result"]
    assert "[REDACTED]" in judge.inputs[0]["tool_result"]


def test_tool_error_is_passed_to_the_judge(runner, adapter, monkeypatch):
    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    adapter.score_record(judged_record(runner, calls=[{**SEARCH, "output": None, "error": "timeout", "status": "ERROR"}]))
    assert judge.inputs[0]["tool_result"] == "search_flights: ERROR timeout"


# ---------------------------------------------------------------- task completion (custom Phoenix LLM judge)

def tc(runner, adapter, monkeypatch, *, output, expected="Found cheapest flight: FL456 for $380", pattern=None, judge=None):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(1.0))
    judge = judge or FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_task_completion_evaluator", lambda: judge)
    rec = judged_record(runner, calls=[SEARCH], output=output)
    rec.expected_output, rec.expected_output_pattern = expected, pattern
    return adapter.score_record(rec), judge


def test_task_completion_is_judged_against_the_reference_answer(runner, adapter, monkeypatch):
    s, judge = tc(runner, adapter, monkeypatch, output="The cheapest flight is FL456 at $380.")
    assert (s.task_completion, s.task_completion_status) == (1.0, "scored")
    (sent,) = judge.inputs
    assert sent == {"input": "User: Find the cheapest flight from NYC to Paris on 2025-03-15",
                    "expected_output": "Found cheapest flight: FL456 for $380",
                    "output": "The cheapest flight is FL456 at $380."}
    assert any("phoenix_task_completion_explanation" in n for n in s.notes)


def test_judge_input_and_prompt_match_the_real_evaluator(runner, adapter, monkeypatch):
    from llm import base_llm
    from llm.task_completion import TaskCompletionEvaluator

    ev = TaskCompletionEvaluator(llm=base_llm)  # builds without calling the LLM
    _, judge = tc(runner, adapter, monkeypatch, output="x")
    TaskCompletionEvaluator.TaskCompletionInputSchema(**judge.inputs[0])
    assert set(ev.prompt_template.variables) == set(judge.inputs[0]) == {"input", "expected_output", "output"}
    assert ev.name == "task_completion"


def test_task_completion_fails_when_the_result_is_wrong_even_if_tool_response_was_handled(runner, adapter, monkeypatch):
    # wrong-arguments case of the demo agent: faithful "No flights found." -> handling OK, task NOT completed
    s, _ = tc(runner, adapter, monkeypatch, output="No flights found.", judge=FakeJudge(0.0))
    assert s.tool_response_handling == 1.0 and s.task_completion == 0.0
    assert any("task_completion=0 (expectedOutput)" in n for n in s.notes)


def test_expected_output_pattern_is_deterministic_and_skips_the_judge(runner, adapter, monkeypatch):
    s, judge = tc(runner, adapter, monkeypatch, output="The cheapest option is FL456 at $380.",
                  pattern=r"\bFL456\b.*\$380\b")
    assert s.task_completion == 1.0 and judge.inputs == []
    s, judge = tc(runner, adapter, monkeypatch, output="The cheapest option is FL123 at $450.", pattern=r"\bFL456\b")
    assert s.task_completion == 0.0 and judge.inputs == []


def test_task_completion_not_applicable_without_reference(runner, adapter, monkeypatch):
    s, judge = tc(runner, adapter, monkeypatch, output="anything", expected=None)
    assert s.task_completion is None and s.task_completion_status.startswith("not_applicable") and judge.inputs == []


def test_task_completion_judge_failure_is_an_error(runner, adapter, monkeypatch):
    s, _ = tc(runner, adapter, monkeypatch, output="x", judge=FakeJudge(boom=True))
    assert s.task_completion is None and s.task_completion_status.startswith("error")
    assert s.tool_response_handling == 1.0  # the other judge is unaffected


def test_reference_and_answer_are_redacted_before_the_judge(runner, adapter, monkeypatch):
    s, judge = tc(runner, adapter, monkeypatch, output="token Bearer abcdefghijklmnopqrstu", expected="ok sk-abcdefghijklmnopqrstuvwx")
    assert "abcdefghijklmnopqrstu" not in judge.inputs[0]["output"]
    assert "sk-abcdefghijklmnopqrstuvwx" not in judge.inputs[0]["expected_output"]


def test_invalid_expected_output_pattern_is_a_dataset_error(runner, tmp_path):
    bad = tmp_path / "d.jsonl"
    bad.write_text(json.dumps({"caseId": "c-bad", "input": "x", "expectedOutputPattern": "("}) + "\n")
    with pytest.raises(ValueError, match=r"line 1, caseId='c-bad', field=expectedOutputPattern\): is not a valid regular expression"):
        runner.parse_dataset_jsonl(str(bad))


# ---------------------------------------------------------------- per-case and run-level enforcement

def cfg_with(runner, *, handling=None, completion=None):
    cfg = type("Cfg", (), {})()
    cfg.thresholds = runner.AgentThresholdConfig(min_tool_response_handling=handling, min_task_completion=completion,
                                                 max_policy_violations=0)
    return cfg


def test_per_case_thresholds_skip_not_applicable_and_fail_errors(runner):
    cfg = cfg_with(runner, handling=0.8, completion=0.8)
    S = runner.AgentPerCaseScore
    ok = S("a", tool_response_handling=1.0, tool_response_handling_status="scored", task_completion=1.0, task_completion_status="scored")
    low = S("b", tool_response_handling=1.0, tool_response_handling_status="scored", task_completion=0.0, task_completion_status="scored")
    na = S("c", tool_response_handling_status="not_applicable: x", task_completion_status="not_applicable: y")
    err = S("d", tool_response_handling_status="error: judge unavailable", task_completion=1.0, task_completion_status="scored")
    runner.mark_per_case_pass_fail(cfg, [ok, low, na, err])
    assert [x.passed for x in (ok, low, na, err)] == [True, False, True, False]


def test_run_level_root_causes_for_both_metrics(runner):
    S = runner.AgentPerCaseScore
    scores = [S("a", tool_response_handling=1.0, task_completion=1.0), S("b", tool_response_handling=0.0, task_completion=0.0),
              S("c", task_completion_status="not_applicable: x")]
    agg = runner.aggregate_scores(scores)
    assert (agg.tool_response_handling_avg, agg.task_completion_avg) == (0.5, 0.5)  # scored cases only
    failed, rc, _ = runner.apply_threshold_policy(cfg_with(runner, handling=0.8, completion=0.8), agg)
    assert failed and rc["stage"] == "Tool Response Handling"          # checked before task completion
    failed, rc, _ = runner.apply_threshold_policy(cfg_with(runner, completion=0.8), agg)
    assert failed and rc["stage"] == "Task Completion" and "task_completion_avg=0.5000 < minTaskCompletion=0.8000" in rc["details"]
    assert not runner.apply_threshold_policy(cfg_with(runner, handling=0.5, completion=0.5), agg)[0]


def test_end_to_end_gate_fails_on_task_completion(runner, monkeypatch):
    class Agent:
        def __call__(self, url, json=None, headers=None, timeout=None):
            class R:
                def raise_for_status(self): pass
                def json(self_inner):
                    return {"output": "No flights found.",
                            "trace": {"traceId": "t", "toolCalls": [{"name": "search", "arguments": {},
                                                                     "output": [], "status": "OK"}]}}
            return R()

    monkeypatch.setattr(runner.requests, "post", Agent())
    monkeypatch.setattr(runner.PhoenixAgentAdapter, "_score_tool_selection", lambda self, r, s: 1.0)
    monkeypatch.setattr(runner.PhoenixAgentAdapter, "_score_tool_invocation", lambda self, r, s: 1.0)
    monkeypatch.setattr(runner.PhoenixAgentAdapter, "_get_tool_response_handling_evaluator", lambda self: FakeJudge(1.0))
    monkeypatch.setattr(runner.PhoenixAgentAdapter, "_get_task_completion_evaluator", lambda self: FakeJudge(0.0))
    line = json.dumps({"caseId": "st-1", "input": "hello", "expectedOutput": "Found cheapest flight: FL456 for $380",
                       "expectedTools": [{"name": "search"}]})
    store = runner.setup({"metrics": {"agent": {"minToolResponseHandling": 0.8, "minTaskCompletion": 0.8}}}, [line])
    assert run_main(runner) == 0
    gate = store["runs/run1/gate_summary.json"]
    assert gate["verdict"] == "FAIL" and gate["rootCause"]["stage"] == "Task Completion"
    agg = store["runs/run1/scores.json"]["aggregatedScores"]
    assert agg["tool_response_handling_avg"] == 1.0 and agg["task_completion_avg"] == 0.0
    names = store["runs/run1/scores.json"]["engine"]["metricNames"]
    assert "tool_response_handling" in names and "task_completion" in names
