"""Rule 6 for the RAG runner: nothing is silently ignored, every value has exactly one spelling."""
import json
import math

import pytest

import runner
from runner import AggregatedScores, DatasetCase, PerCaseScore, ThresholdConfig

ENV = {
    "EVALRUN_NAME": "run-1", "EVALSUITE_NAME": "suite-1", "TARGET_ENDPOINT": "http://target/eval",
    "S3_ENDPOINT": "http://s3", "DATASET_S3_BUCKET": "datasets", "S3_SUMMARY_BUCKET": "bucket", "DATASET_S3_KEY": "ds.jsonl",
    "S3_ACCESS_KEY": "a", "S3_SECRET_KEY": "b", "S3_BASELINE_KEY": "baselines/suite-1.json",
    "GATE_SUMMARY_KEY": "runs/run-1/gate_summary.json", "SCORES_KEY": "runs/run-1/scores.json",
    "DETAILS_KEY": "runs/run-1/details.jsonl", "OPENAI_MODEL": "m", "OPENAI_EMBEDDER": "e", "OPENAI_API_KEY": "k",
}
VALID = {"metrics": {"rag": {"retrieval": {"minContextRecall": 0.7}, "generation": {"minFaithfulness": 0.8}}},
         "regressionPolicy": {"maxDrop": 0.05}, "execution": {"mode": "rag", "timeoutSeconds": 30}}


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("WORK_DIR", str(tmp_path))

    def write_suite(suite):
        path = tmp_path / "suite.json"
        path.write_text(json.dumps(suite))
        monkeypatch.setenv("SUITE_CONFIG", str(path))
    write_suite(VALID)
    return write_suite


# ---------------------------------------------------------------- suite.json

def test_summary_bucket_is_required(env, monkeypatch):
    monkeypatch.delenv("S3_SUMMARY_BUCKET")
    with pytest.raises(RuntimeError, match="missing required environment variable: S3_SUMMARY_BUCKET"):
        runner.build_runner_config()


def test_dataset_and_artifact_buckets_are_used_for_their_own_purpose(env, monkeypatch, tmp_path):
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

    monkeypatch.setattr(runner.requests, "get", lambda url: urls.append(url) or R())
    monkeypatch.setattr(runner.requests, "put", lambda url, **k: urls.append(url) or R())
    runner.download_dataset(cfg)
    runner.load_optional_baseline(cfg)
    runner.upload_json(cfg, cfg.gate_summary_key, {})
    assert urls == ["http://s3/buckets/datasets/objects/ds.jsonl",
                    "http://s3/buckets/bucket/objects/baselines/suite-1.json",
                    "http://s3/buckets/bucket/objects/runs/run-1/gate_summary.json"]


def test_valid_suite(env):
    cfg = runner.build_runner_config()
    assert (cfg.thresholds.min_context_recall, cfg.thresholds.min_faithfulness) == (0.7, 0.8)
    assert (cfg.regression_max_drop, cfg.request_timeout_seconds) == (0.05, 30)


@pytest.mark.parametrize("suite, fragment", [
    ({"metrics": {"rag": {"generation": {"minRelevancy:": 0.5}}}}, r"metrics.rag.generation: unknown field\(s\) \['minRelevancy:'\]"),
    ({"metrics": {"rag": {"retrieval": {"minContextRecall": "0.7"}}}}, r"minContextRecall must be a number in \[0, 1\]"),
    ({"metrics": {"rag": {"retrieval": {"minContextRecall": 1.2}}}}, r"must be a number in \[0, 1\]"),
    ({"metrics": {"rag": {"generation": {"minFaithfulness": True}}}}, "must be a number"),
    ({"metrics": {"rag": {"trustworthiness": {"maxStaleChunkRatio": 0.1}}}}, r"metrics.rag: unknown field\(s\) \['trustworthiness'\]"),
    ({"metrics": {"agent": {"minToolSelection": 0.9}}}, r"metrics: unknown field\(s\) \['agent'\]"),
    ({"metrics": {"rag": {}}, "evaluators": []}, r"suite: unknown field\(s\) \['evaluators'\]"),
    ({"metrics": {"rag": {}}, "evaluation": {"trials": 2}}, r"evaluation: unknown field\(s\) \['trials'\]"),
    ({"metrics": {"rag": {}}, "evaluation": {"interactionMode": "MultiTurn"}}, "must be SingleTurn"),
    ({"metrics": {"rag": {}}, "execution": {"timeoutSeconds": 0}}, "timeoutSeconds must be an integer >= 1"),
    ({"metrics": {"rag": {}}, "execution": {"timeoutSeconds": "30"}}, "timeoutSeconds must be an integer"),
    ({"metrics": {"rag": {}}, "regressionPolicy": {"maxDrop": "x"}}, "maxDrop must be a number"),
])
def test_invalid_suite_is_a_configuration_error(env, suite, fragment):
    env(suite)
    with pytest.raises(runner.ConfigError, match=fragment):
        runner.build_runner_config()


async def test_invalid_suite_still_publishes_a_configuration_root_cause(env, monkeypatch):
    env({"metrics": {"rag": {"generation": {"minRelevancy:": 0.5}}}})
    published = {}
    monkeypatch.setattr(runner, "publish_gate_summary", lambda cfg, gate: published.update(gate=gate))
    assert await runner.main() == 1
    rc = published["gate"].rootCause
    assert published["gate"].verdict == "FAIL" and rc["stage"] == "Configuration" and "minRelevancy:" in rc["details"]


# ---------------------------------------------------------------- dataset

def rec(**fields):
    return json.dumps({"caseId": "c-1", "question": "q", "groundTruth": "gt", **fields})


@pytest.mark.parametrize("lines, fragment", [
    (["{bad"], "invalid JSON"),
    (["[]"], "must be a JSON object"),
    ([json.dumps({"caseId": "c-1", "question": "q"})], r"field=groundTruth\): is required"),
    ([rec(groundTruth="")], r"field=groundTruth\): is required"),
    ([rec(input="x")], r"field=input\): is an agent dataset field; the RAG dataset uses 'question'"),
    ([rec(expectedOutput="x")], "uses 'groundTruth'"),
    ([rec(ground_truth="x")], r"unknown field\(s\) \['ground_truth'\]"),
    ([rec(), rec()], "duplicate caseId"),
    ([rec(tags="rag")], "list of non-empty strings"),
    ([rec(metadata=[1])], r"field=metadata\): must be an object"),
])
def test_malformed_dataset_is_a_dataset_error(tmp_path, lines, fragment):
    path = tmp_path / "d.jsonl"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(runner.DatasetError, match=fragment):
        runner.parse_dataset_jsonl(str(path))


# ---------------------------------------------------------------- target response

class _Resp:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self.body


@pytest.fixture
def cfg(env):
    return runner.build_runner_config()


def respond(monkeypatch, cfg, body):
    monkeypatch.setattr(runner.requests, "post", lambda *a, **k: _Resp(body))
    return runner.call_target_endpoint(cfg, DatasetCase(case_id="c-1", question="q", ground_truth="gt"))


@pytest.mark.parametrize("body, fragment", [
    ([], "must be a JSON object"),
    ({"answer": "a"}, r"contexts is required \(send \[\] when nothing was retrieved\)"),
    ({"answer": "a", "contexts": {}}, "contexts must be a list"),
    ({"answer": "a", "contexts": [{"id": "d"}]}, r"contexts\[0\].text is required"),
    ({"answer": "a", "contexts": [{"content": "x", "text": "t"}]}, r"unknown field\(s\) \['content'\]"),
    ({"answer": "a", "contexts": [{"text": "t", "score": "0.9"}]}, r"contexts\[0\].score must be a number, got '0.9'"),
    ({"answer": "a", "contexts": [{"text": "t", "source": 1}]}, "source must be a string"),
    ({"answer": "a", "contexts": [], "metadata": []}, "metadata must be an object"),
    ({"output": "a", "contexts": []}, "answer is required"),
])
def test_response_contract_violations(monkeypatch, cfg, body, fragment):
    with pytest.raises(runner.TargetResponseError, match=fragment) as err:
        respond(monkeypatch, cfg, body)
    assert err.value.case_id == "c-1"


def test_empty_contexts_list_is_valid(monkeypatch, cfg):
    assert respond(monkeypatch, cfg, {"answer": "a", "contexts": []}).contexts == []


# ---------------------------------------------------------------- scores

def test_nan_and_inf_scores_are_unscored_not_passing():
    assert runner.finite_or_none(float("nan")) is None
    assert runner.finite_or_none(float("inf")) is None
    assert runner.finite_or_none(0.5) == 0.5
    assert not (math.nan < 0.8)  # why: a NaN would otherwise pass every threshold


def th(**kw):
    c = type("Cfg", (), {})()
    c.thresholds = ThresholdConfig(**kw)
    return c


def agg(**over):
    base = dict(total_cases=2, passed_cases=2, safety_passed=True, vulnerability_type="", faithfulness_avg=0.9,
                answer_relevancy_avg=0.9, answer_correctness_avg=0.9, context_precision_avg=0.9,
                context_recall_avg=0.9, latency_ms_avg=1.0)
    base.update(over)
    return AggregatedScores(**base)


def test_unscored_case_fails_the_case_and_the_run():
    cfg = th(min_faithfulness=0.8)
    per_case = [PerCaseScore(case_id="ok", faithfulness=0.95, safety_passed=True),
                PerCaseScore(case_id="judge-error", faithfulness=None, safety_passed=True)]
    runner.mark_per_case_pass_fail(cfg, per_case)
    assert [x.passed for x in per_case] == [True, False]
    assert "faithfulness was not scored by the evaluator but minFaithfulness is set" in per_case[1].notes

    unscored = runner.unscored_cases(cfg, per_case)
    assert unscored == {"faithfulness": ["judge-error"]}
    # the average of the scored case alone (0.95) would pass
    failed, rc, _ = runner.apply_threshold_policy(cfg, runner.aggregate_scores(per_case), unscored)
    assert failed and rc["stage"] == "Generation" and rc["metric"] == "faithfulness"
    assert "not scored for 1 case(s) (judge-error)" in rc["details"]


def test_metric_scored_for_no_case_fails_the_run():
    failed, rc, _ = runner.apply_threshold_policy(th(min_correctness=0.7), agg(answer_correctness_avg=None))
    assert failed and rc["metric"] == "correctness" and "not scored for any case" in rc["details"]


def test_metric_without_threshold_is_not_gated():
    assert runner.unscored_cases(th(), [PerCaseScore(case_id="c", safety_passed=True)]) == {}
    assert runner.apply_threshold_policy(th(), agg(answer_correctness_avg=None))[0] is False


def test_safety_root_cause_is_not_overwritten_by_retrieval():
    failed, rc, notes = runner.apply_threshold_policy(
        th(min_context_recall=0.9), agg(safety_passed=False, vulnerability_type="pii_leakage", context_recall_avg=0.1))
    assert failed and rc["stage"] == "Safety" and len(notes) == 1


def test_relevancy_details_name_the_right_threshold():
    _, rc, _ = runner.apply_threshold_policy(th(min_relevancy=0.8), agg(answer_relevancy_avg=0.5))
    assert rc["details"] == "answer_relevancy_avg=0.5000 < minRelevancy=0.8000"


def test_missing_safety_verdict_fails_closed():
    per_case = [PerCaseScore(case_id="a", safety_passed=True), PerCaseScore(case_id="b", safety_passed=None)]
    assert runner.aggregate_scores(per_case).safety_passed is False


# ---------------------------------------------------------------- regression

def rcfg():
    c = type("Cfg", (), {})()
    c.regression_max_drop, c.s3_baseline_key = 0.05, "baselines/suite-1.json"
    return c


def test_metric_missing_in_a_run_is_reported_not_skipped():
    baseline = {"aggregatedScores": {"faithfulness_avg": 0.9, "answer_relevancy_avg": 0.9}}
    r = runner.apply_regression_policy(rcfg(), agg(answer_relevancy_avg=None, context_recall_avg=None), baseline)
    assert set(r.drops) == {"faithfulness_avg"}
    assert r.not_compared == {"answer_relevancy_avg": "not measured in this run",
                              "answer_correctness_avg": "not measured in the baseline",
                              "context_precision_avg": "not measured in the baseline",
                              "context_recall_avg": "not measured in either run"}


@pytest.mark.parametrize("baseline, fragment", [
    ({}, "has no aggregatedScores object"),
    ({"aggregatedScores": None}, "has no aggregatedScores object"),
    ({"aggregatedScores": {"faithfulness_avg": "0.9"}}, "faithfulness_avg must be a number"),
])
def test_malformed_baseline_is_an_error(baseline, fragment):
    with pytest.raises(ValueError, match=fragment):
        runner.apply_regression_policy(rcfg(), agg(), baseline)
