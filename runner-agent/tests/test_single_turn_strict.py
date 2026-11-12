"""Rule 6 for the single-turn agent runner: nothing is silently ignored.

Configuration, dataset and agent-response problems are errors with actionable messages; every value has exactly
one field (no synonyms, no legacy locations).
"""
import importlib.util
import json
import re

import pytest

from multiturn.config import ConfigError, parse_multiturn_settings
from multiturn.dataset import DatasetError

from conftest import suite_dict
from test_runner_agent_main import RUNNER_PATH, SINGLE_LINE, run_main, runner  # noqa: F401 - pytest fixture

# ---------------------------------------------------------------- suite.json (single-turn)

VALID = {"metrics": {"agent": {"minToolSelection": 0.9, "maxPolicyViolations": 0}},
         "regressionPolicy": {"maxDrop": 0.03}, "execution": {"mode": "agent", "timeoutSeconds": 30},
         "evaluation": {"interactionMode": "SingleTurn"}}


def test_valid_single_turn_suite_is_parsed(runner):
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
def test_invalid_single_turn_suite_is_a_configuration_error(runner, suite, fragment):
    runner.setup(suite, [SINGLE_LINE])
    with pytest.raises(ConfigError, match=fragment):
        runner.build_runner_config()


def test_summary_bucket_is_required(runner, monkeypatch):
    runner.setup(VALID, [SINGLE_LINE])
    monkeypatch.delenv("S3_SUMMARY_BUCKET")
    with pytest.raises(RuntimeError, match="missing required environment variable: S3_SUMMARY_BUCKET"):
        runner.build_runner_config()


def test_dataset_and_artifact_buckets_are_used_for_their_own_purpose(runner, monkeypatch, tmp_path):
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


def test_invalid_suite_still_publishes_a_configuration_root_cause(runner):
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
def test_invalid_multi_turn_suite_is_a_configuration_error(patch, fragment):
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
def test_malformed_single_turn_dataset_names_line_case_and_field(runner, tmp_path, lines, field, fragment):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(DatasetError, match=fragment) as err:
        runner.parse_dataset_jsonl(str(path))
    assert "invalid single-turn agent dataset record (line" in str(err.value)
    if field:
        assert re.search(f"field={field}", str(err.value))


def test_malformed_dataset_is_reported_with_dataset_stage_and_case(runner):
    store = runner.setup(VALID, [case(expectedTool=[{"name": "search"}])])
    assert run_main(runner) == 1
    rc = store["runs/run1/gate_summary.json"]["rootCause"]
    assert rc["stage"] == "Dataset" and rc["caseId"] == "c-1" and "expectedTool" in rc["details"]


# ---------------------------------------------------------------- agent response

def record(runner, body):
    ds_case = runner.AgentDatasetCase(case_id="c-1", input="hi", expected_output=None, expected_output_pattern=None,
                                      expected_tools=[], forbidden_tools=[], tags=[], metadata={})
    return runner.build_agent_eval_record(ds_case, body)


GOOD_CALL = {"name": "search", "arguments": {"q": 1}, "status": "OK", "durationMs": 1.5}


def test_valid_response_is_parsed(runner):
    rec = record(runner, {"output": "done", "metadata": {"traceId": "t1"},
                          "trace": {"traceId": "t1", "toolCalls": [GOOD_CALL]}})
    (call,) = rec.tools_called
    assert (call.name, call.arguments, call.duration_ms) == ("search", {"q": 1}, 1.5)
    assert record(runner, {"output": "no tools needed", "trace": {"toolCalls": []}}).tools_called == []


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
def test_response_contract_violations_are_errors(runner, body, fragment):
    with pytest.raises(runner.AgentResponseError, match=fragment) as err:
        record(runner, body)
    assert err.value.case_id == "c-1"


def test_contract_violation_is_reported_with_the_case_and_a_clear_message(runner, monkeypatch):
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


def test_tool_execution_counts_only_the_canonical_ok_status(runner):
    rec = record(runner, {"output": "done", "trace": {"toolCalls": [
        {"name": "a", "status": "OK"}, {"name": "b", "status": "ERROR", "error": "boom"}, {"name": "c"}]}})
    assert [c.status for c in rec.tools_called] == ["OK", "ERROR", "OK"]


def test_type_on_a_single_turn_record_is_reported_as_multi_turn_field(runner):
    store = runner.setup(VALID, [case(type="singleTurn")])
    assert run_main(runner) == 1
    rc = store["runs/run1/gate_summary.json"]["rootCause"]
    assert rc["stage"] == "Dataset" and "multi-turn fields ('type'/'turns')" in rc["details"]


# ---------------------------------------------------------------- judge errors are never averaged away

class _Score:
    def __init__(self, score):
        self.score, self.label, self.explanation = score, "x", None


class _Judge:
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
    monkeypatch.setattr(a, "_get_tool_invocation_evaluator", lambda: invocation or _Judge(1.0))
    monkeypatch.setattr(a, "_get_tool_response_handling_evaluator", lambda: _Judge(1.0))
    monkeypatch.setattr(a, "_get_task_completion_evaluator", lambda: _Judge(1.0))
    return a


def test_judge_statuses_distinguish_not_applicable_from_error(runner, monkeypatch):
    a = _adapter(runner, monkeypatch, _Judge(boom=True))
    s = a.score_record(_record(runner))
    assert s.tool_selection is None and s.tool_selection_status.startswith("error: RuntimeError")
    assert s.tool_invocation == 1.0 and s.tool_invocation_status == "scored"  # no longer skipped with selection

    s = _adapter(runner, monkeypatch, _Judge(1.0)).score_record(_record(runner, expected=(), called=()))
    assert s.tool_selection_status == "not_applicable: case has no expectedTools or forbiddenTools"
    assert s.tool_invocation_status == "not_applicable: no tool was called"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.9", True])
def test_non_finite_or_non_numeric_judge_score_is_an_error(runner, monkeypatch, value):
    s = _adapter(runner, monkeypatch, _Judge(value)).score_record(_record(runner))
    assert s.tool_selection is None and "non-finite or non-numeric score" in s.tool_selection_status


def _cfg(runner, **thresholds):
    c = type("Cfg", (), {})()
    c.thresholds = runner.AgentThresholdConfig(**thresholds)
    return c


def test_judge_error_fails_the_run_even_if_the_average_of_the_rest_passes(runner):
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


def test_not_applicable_does_not_fail_the_run(runner):
    na = runner.AgentPerCaseScore(case_id="na", tool_selection=None, tool_selection_status="not_applicable: x")
    cfg = _cfg(runner, min_tool_selection=0.9)
    assert runner.unscored_cases(cfg, [na]) == {}
    assert runner.apply_threshold_policy(cfg, runner.aggregate_scores([na]), {})[0] is False


def test_unscored_metric_without_threshold_is_not_gated(runner):
    err = runner.AgentPerCaseScore(case_id="boom", task_completion=None, task_completion_status="error: x")
    assert runner.unscored_cases(_cfg(runner), [err]) == {}
