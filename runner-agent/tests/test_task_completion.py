"""Single-turn tool response handling and task completion: Phoenix LLM judges (faked here - no LLM calls)."""
import json

import pytest

from test_runner_agent_main import run_main, runner  # noqa: F401  (fixture re-use)


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


def record(runner, *, calls, output="Found cheapest flight: FL456 for $380", expected_tools=("search_flights",)):
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
    score = adapter.score_record(record(runner, calls=[SEARCH]))
    assert (score.tool_response_handling, score.tool_response_handling_status) == (1.0, "scored")
    (sent,) = judge.inputs
    assert sent["tool_call"] == 'search_flights(origin="NYC", destination="Paris", date="2025-03-15")'
    assert "FL456" in sent["tool_result"] and sent["output"].startswith("Found cheapest flight: FL456")
    assert any("phoenix_tool_response_handling_explanation" in n for n in score.notes)


def test_judge_input_matches_the_real_phoenix_schema(runner, adapter, monkeypatch):
    from phoenix.evals.metrics import ToolResponseHandlingEvaluator

    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    adapter.score_record(record(runner, calls=[SEARCH]))
    ToolResponseHandlingEvaluator.ToolResponseHandlingInputSchema(**judge.inputs[0])  # raises if fields drift


def test_misreported_result_fails_the_threshold(runner, adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(0.0))
    score = adapter.score_record(record(runner, calls=[SEARCH], output="Found cheapest flight: FL999 for $1"))
    assert score.tool_response_handling == 0.0


def test_no_tool_call_although_tools_were_expected_scores_zero_without_a_judge_call(runner, adapter, monkeypatch):
    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    score = adapter.score_record(record(runner, calls=[], output="I cannot determine the cheapest flight without using tools."))
    assert (score.tool_response_handling, score.tool_response_handling_status) == (0.0, "scored") and judge.inputs == []


def test_not_applicable_cases(runner, adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(1.0))
    no_tools = adapter.score_record(record(runner, calls=[], expected_tools=()))
    assert no_tools.tool_response_handling is None and no_tools.tool_response_handling_status.startswith("not_applicable")
    no_outputs = adapter.score_record(record(runner, calls=[{**SEARCH, "output": None}]))
    assert no_outputs.tool_response_handling is None and "no tool outputs" in no_outputs.tool_response_handling_status


def test_judge_failure_is_isolated_and_marked_as_error(runner, adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(boom=True))
    score = adapter.score_record(record(runner, calls=[SEARCH]))
    assert score.tool_response_handling is None and score.tool_response_handling_status.startswith("error")
    assert score.tool_selection == 1.0  # other metrics unaffected


def test_tool_outputs_are_redacted_before_they_reach_the_judge(runner, adapter, monkeypatch):
    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    secret = {**SEARCH, "output": {"flights": FLIGHTS, "api_key": "sk-abcdefghijklmnopqrstuvwx"}}
    adapter.score_record(record(runner, calls=[secret]))
    assert "sk-abcdefghijklmnopqrstuvwx" not in judge.inputs[0]["tool_result"]
    assert "[REDACTED]" in judge.inputs[0]["tool_result"]


def test_tool_error_is_passed_to_the_judge(runner, adapter, monkeypatch):
    judge = FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: judge)
    adapter.score_record(record(runner, calls=[{**SEARCH, "output": None, "error": "timeout", "status": "ERROR"}]))
    assert judge.inputs[0]["tool_result"] == "search_flights: ERROR timeout"


# ---------------------------------------------------------------- task completion (custom Phoenix LLM judge)

def tc(runner, adapter, monkeypatch, *, output, expected="Found cheapest flight: FL456 for $380", pattern=None, judge=None):
    monkeypatch.setattr(adapter, "_get_tool_response_handling_evaluator", lambda: FakeJudge(1.0))
    judge = judge or FakeJudge(1.0)
    monkeypatch.setattr(adapter, "_get_task_completion_evaluator", lambda: judge)
    rec = record(runner, calls=[SEARCH], output=output)
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
    with pytest.raises(ValueError, match=r"caseId='c-bad' field=expectedOutputPattern"):
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
