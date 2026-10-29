
import json
import pytest

import runner

from runner import (
    RunnerConfig,
    ThresholdConfig,
    DatasetCase,
    RetrievedContext,
    EvalRecord,
    PerCaseScore,
    AggregatedScores,
    RegressionResult,
    GateSummary,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def suite_config_file(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps({
            "metrics": {
                "rag": {
                    "retrieval": {
                        "minContextRecall": 0.7,
                        "minContextPrecision": 0.6,
                    },
                    "generation": {
                        "minFaithfulness": 0.8,
                        "minCorrectness": 0.75,
                        "minRelevancy:": 0.5,
                    },
                    "trustworthiness": {
                        "maxStaleChunkRatio": 0.1,
                        "maxUnknownOwnerRatio": 0.2,
                    },
                },
            },
            "regressionPolicy": {
                "maxDrop": 0.05,
            },
            "execution": {
                "timeoutSeconds": 30,
            },
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def cfg(tmp_path):
    return RunnerConfig(
        evalrun_name="evalrun-1",
        evalsuite_name="suite-1",
        git_sha="abc123",
        pr_number="42",

        target_endpoint="http://target.local/evaluate",
        request_timeout_seconds=30,

        suite_config_path=str(tmp_path / "suite.json"),
        work_dir=str(tmp_path),
        dataset_file=str(tmp_path / "dataset.jsonl"),

        s3_endpoint="http://s3.local",
        s3_bucket="eval-bucket",
        s3_access_key="access",
        s3_secret_key="secret",

        dataset_s3_bucket="dataset-bucket",
        dataset_s3_key="datasets/test.jsonl",
        dataset_s3_version_id=None,

        s3_baseline_key="baseline.json",
        gate_summary_key="artifacts/gate-summary.json",
        scores_key="artifacts/scores.json",
        details_key="artifacts/details.jsonl",

        llm_model="gpt-test",
        llm_embedder="embed-test",
        llm_api_key="test-key",

        thresholds=ThresholdConfig(
            min_context_recall=0.7,
            min_context_precision=0.6,
            min_faithfulness=0.8,
            min_correctness=0.75,
            min_relevancy=0.75,
            max_stale_chunk_ratio=None,
            max_unknown_owner_ratio=None,
        ),
        regression_max_drop=0.05,
    )


# ============================================================
# Small helpers / config
# ============================================================

def test_getenv_required_returns_value(monkeypatch):
    monkeypatch.setenv("TEST_ENV", "value")

    assert runner.getenv_required("TEST_ENV") == "value"


def test_getenv_required_raises_when_missing(monkeypatch):
    monkeypatch.delenv("TEST_ENV", raising=False)

    with pytest.raises(RuntimeError, match="missing required environment variable"):
        runner.getenv_required("TEST_ENV")


def test_getenv_optional_returns_default(monkeypatch):
    monkeypatch.delenv("OPTIONAL_ENV", raising=False)

    assert runner.getenv_optional("OPTIONAL_ENV", "default") == "default"


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1.23", 1.23),
        (5, 5.0),
        (None, None),
        ("bad", None),
        ("", None),
    ],
)
def test_safe_float(value, expected):
    assert runner.safe_float(value) == expected


def test_mean_ignores_none_values():
    assert runner.mean([1.0, None, 3.0]) == 2.0


def test_mean_returns_none_for_empty_or_all_none():
    assert runner.mean([]) is None
    assert runner.mean([None, None]) is None


def test_build_runner_config(monkeypatch, suite_config_file):
    monkeypatch.setenv("SUITE_CONFIG", str(suite_config_file))
    monkeypatch.setenv("EVALRUN_NAME", "evalrun-1")
    monkeypatch.setenv("EVALSUITE_NAME", "suite-1")
    monkeypatch.setenv("TARGET_ENDPOINT", "http://target.local/evaluate")

    monkeypatch.setenv("S3_ENDPOINT", "http://s3.local")
    monkeypatch.setenv("DATASET_S3_BUCKET", "dataset-bucket")
    monkeypatch.setenv("DATASET_S3_KEY", "datasets/test.jsonl")
    monkeypatch.setenv("S3_ACCESS_KEY", "access")
    monkeypatch.setenv("S3_SECRET_KEY", "secret")

    monkeypatch.setenv("S3_BASELINE_KEY", "baseline.json")
    monkeypatch.setenv("GATE_SUMMARY_KEY", "gate-summary.json")
    monkeypatch.setenv("SCORES_KEY", "scores.json")
    monkeypatch.setenv("DETAILS_KEY", "details.jsonl")

    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    monkeypatch.setenv("OPENAI_EMBEDDER", "embed-test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    cfg = runner.build_runner_config()

    assert cfg.evalrun_name == "evalrun-1"
    assert cfg.evalsuite_name == "suite-1"
    assert cfg.target_endpoint == "http://target.local/evaluate"
    assert cfg.request_timeout_seconds == 30

    assert cfg.thresholds.min_context_recall == 0.7
    assert cfg.thresholds.min_context_precision == 0.6
    assert cfg.thresholds.min_faithfulness == 0.8
    assert cfg.thresholds.min_correctness == 0.75
    assert cfg.regression_max_drop == 0.05

    assert cfg.llm_model == "gpt-test"
    assert cfg.llm_embedder == "embed-test"
    assert cfg.llm_api_key == "test-key"


# ============================================================
# Dataset parsing
# ============================================================

def test_parse_dataset_jsonl_valid_file(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"caseId":"case-1","question":"What is RAG?","groundTruth":"Retrieval augmented generation","tags":["rag"],"metadata":{"owner":"team-a"}}\n'
        '{"caseId":"case-2","question":"What is Kubernetes?"}\n',
        encoding="utf-8",
    )

    cases = runner.parse_dataset_jsonl(str(dataset))

    assert len(cases) == 2

    assert cases[0].case_id == "case-1"
    assert cases[0].question == "What is RAG?"
    assert cases[0].ground_truth == "Retrieval augmented generation"
    assert cases[0].tags == ["rag"]
    assert cases[0].metadata == {"owner": "team-a"}

    assert cases[1].case_id == "case-2"
    assert cases[1].question == "What is Kubernetes?"
    assert cases[1].ground_truth is None
    assert cases[1].tags == []
    assert cases[1].metadata == {}


def test_parse_dataset_jsonl_skips_empty_lines(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '\n{"caseId":"case-1","question":"Q1"}\n\n',
        encoding="utf-8",
    )

    cases = runner.parse_dataset_jsonl(str(dataset))

    assert len(cases) == 1
    assert cases[0].case_id == "case-1"


def test_parse_dataset_jsonl_missing_case_id_raises(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"question":"Q1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing caseId"):
        runner.parse_dataset_jsonl(str(dataset))


def test_parse_dataset_jsonl_missing_question_raises(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"caseId":"case-1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing question"):
        runner.parse_dataset_jsonl(str(dataset))


# ============================================================
# S3-like client functions
# ============================================================

class FakeResponse:
    def __init__(self, payload=None, status_ok=True, content_chunks=None, status_code=200):
        self.payload = payload
        self.status_ok = status_ok
        self.content_chunks = content_chunks or []
        self.status_code = status_code

    def raise_for_status(self):
        if not self.status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self.payload

    def iter_content(self, chunk_size=128):
        yield from self.content_chunks


def test_download_dataset_writes_file(monkeypatch, cfg, tmp_path):
    cfg.dataset_file = str(tmp_path / "downloaded.jsonl")

    captured = {}

    def fake_get(url):
        captured["url"] = url
        return FakeResponse(
            content_chunks=[
                b'{"caseId":"case-1",',
                b'"question":"Q1"}\n',
            ]
        )

    monkeypatch.setattr(runner.requests, "get", fake_get)

    result_path = runner.download_dataset(cfg)

    assert result_path == cfg.dataset_file
    assert captured["url"] == "http://s3.local/buckets/dataset-bucket/objects/datasets/test.jsonl"

    content = (tmp_path / "downloaded.jsonl").read_text(encoding="utf-8")
    assert content == '{"caseId":"case-1","question":"Q1"}\n'


def test_load_optional_baseline_success(monkeypatch, cfg):
    def fake_get(url):
        assert url == "http://s3.local/buckets/eval-bucket/objects/baseline.json"
        return FakeResponse(payload={
            "aggregatedScores": {
                "faithfulness_avg": 0.9,
            }
        })

    monkeypatch.setattr(runner.requests, "get", fake_get)

    baseline = runner.load_optional_baseline(cfg)

    assert baseline["aggregatedScores"]["faithfulness_avg"] == 0.9


def test_upload_json_sends_json_payload(monkeypatch, cfg):
    captured = {}

    def fake_put(url, data, headers):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(runner.requests, "put", fake_put)

    result = runner.upload_json(cfg, "scores.json", {"metric": 0.9})

    assert result == {"ok": True}
    assert captured["url"] == "http://s3.local/buckets/eval-bucket/objects/scores.json"
    assert captured["headers"]["Content-Type"] == "application/json"

    body = json.loads(captured["data"].decode("utf-8"))
    assert body == {"metric": 0.9}


def test_upload_text_sends_text_payload(monkeypatch, cfg):
    captured = {}

    def fake_put(url, data, headers):
        captured["url"] = url
        captured["data"] = data
        captured["headers"] = headers
        return FakeResponse(payload={"ok": True})

    monkeypatch.setattr(runner.requests, "put", fake_put)

    result = runner.upload_text(
        cfg,
        "details.jsonl",
        '{"caseId":"case-1"}\n',
        content_type="application/x-ndjson",
    )

    assert result == {"ok": True}
    assert captured["url"] == "http://s3.local/buckets/eval-bucket/objects/details.jsonl"
    assert captured["data"] == b'{"caseId":"case-1"}\n'
    assert captured["headers"]["Content-Type"] == "application/x-ndjson; charset=utf-8"


# ============================================================
# Target endpoint client
# ============================================================

def test_call_target_endpoint_success(monkeypatch, cfg):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout

        return FakeResponse(payload={
            "answer": "RAG means retrieval augmented generation.",
            "contexts": [
                {
                    "id": "doc-1",
                    "text": "RAG retrieves documents before generation.",
                    "score": "0.95",
                    "source": "docs",
                }
            ],
            "metadata": {
                "model": "test-model",
            },
        })

    monkeypatch.setattr(runner.requests, "post", fake_post)

    case = DatasetCase(
        case_id="case-1",
        question="What is RAG?",
        ground_truth="Retrieval augmented generation",
    )

    record = runner.call_target_endpoint(cfg, case)

    assert captured["url"] == cfg.target_endpoint
    assert captured["json"] == {"question": "What is RAG?"}
    assert captured["timeout"] == cfg.request_timeout_seconds

    assert record.case_id == "case-1"
    assert record.question == "What is RAG?"
    assert record.answer == "RAG means retrieval augmented generation."
    assert record.ground_truth == "Retrieval augmented generation"

    assert len(record.contexts) == 1
    assert record.contexts[0].id == "doc-1"
    assert record.contexts[0].text == "RAG retrieves documents before generation."
    assert record.contexts[0].score == 0.95
    assert record.contexts[0].source == "docs"

    assert record.metadata["model"] == "test-model"
    assert "latencyMs" in record.metadata


def test_call_target_endpoint_generates_context_id_when_missing(monkeypatch, cfg):
    def fake_post(*args, **kwargs):
        return FakeResponse(payload={
            "answer": "Answer",
            "contexts": [
                {
                    "text": "Context text",
                }
            ],
        })

    monkeypatch.setattr(runner.requests, "post", fake_post)

    case = DatasetCase(case_id="case-1", question="Q")
    record = runner.call_target_endpoint(cfg, case)

    assert record.contexts[0].id == "ctx-0"
    assert record.contexts[0].text == "Context text"


def test_call_target_endpoint_empty_answer_raises(monkeypatch, cfg):
    def fake_post(*args, **kwargs):
        return FakeResponse(payload={
            "answer": "",
            "contexts": [],
        })

    monkeypatch.setattr(runner.requests, "post", fake_post)

    case = DatasetCase(case_id="case-1", question="Q")

    with pytest.raises(RuntimeError, match="empty answer"):
        runner.call_target_endpoint(cfg, case)


# ============================================================
# RagasAdapter lightweight unit tests
# ============================================================

def test_ragas_adapter_build_hf_dataset():
    records = [
        EvalRecord(
            case_id="case-1",
            question="Q1",
            answer="A1",
            ground_truth="GT1",
            contexts=[
                RetrievedContext(id="ctx-1", text="Context 1"),
                RetrievedContext(id="ctx-2", text="Context 2"),
            ],
            metadata={},
        )
    ]

    dataset = runner.RagasAdapter._build_hf_dataset(records)

    assert len(dataset.samples) == 1
    sample = dataset.samples[0]
    assert sample.user_input == "Q1"
    assert sample.response == "A1"
    assert sample.retrieved_contexts == ["Context 1", "Context 2"]


def test_ragas_adapter_choose_metrics_without_ground_truth():
    records = [
        EvalRecord(
            case_id="case-1",
            question="Q1",
            answer="A1",
            ground_truth=None,
            contexts=[],
        )
    ]

    _, metric_names = runner.RagasAdapter._choose_metrics(records)

    assert metric_names == ["faithfulness", "answer_relevancy", "answer_correctness"]


def test_ragas_adapter_choose_metrics_with_ground_truth():
    records = [
        EvalRecord(
            case_id="case-1",
            question="Q1",
            answer="A1",
            ground_truth="GT1",
            contexts=[],
        )
    ]

    _, metric_names = runner.RagasAdapter._choose_metrics(records)

    assert "faithfulness" in metric_names
    assert "answer_relevancy" in metric_names
    assert "context_precision" in metric_names
    assert "context_recall" in metric_names
    assert "answer_correctness" in metric_names


# ============================================================
# Aggregation / analyzer
# ============================================================

def test_aggregate_scores_calculates_averages():
    per_case = [
        PerCaseScore(
            case_id="case-1",
            faithfulness=0.8,
            answer_relevancy=0.7,
            context_precision=0.6,
            context_recall=0.9,
            latency_ms=100,
            passed=True,
        ),
        PerCaseScore(
            case_id="case-2",
            faithfulness=1.0,
            answer_relevancy=0.9,
            context_precision=None,
            context_recall=0.7,
            latency_ms=300,
            passed=False,
        ),
    ]

    agg = runner.aggregate_scores(per_case)

    assert agg.total_cases == 2
    assert agg.passed_cases == 1
    assert agg.faithfulness_avg == 0.9
    assert agg.answer_relevancy_avg == 0.8
    assert agg.context_precision_avg == 0.6
    assert agg.context_recall_avg == 0.8
    assert agg.latency_ms_avg == 200


def test_aggregate_scores_empty_list():
    agg = runner.aggregate_scores([])

    assert agg.total_cases == 0
    assert agg.passed_cases == 0
    assert agg.faithfulness_avg is None
    assert agg.answer_relevancy_avg is None
    assert agg.latency_ms_avg is None


def test_mark_per_case_pass_fail_marks_failure(cfg):
    per_case = [
        PerCaseScore(
            case_id="case-1",
            faithfulness=0.7,
            answer_relevancy=0.8,
            context_precision=0.8,
            context_recall=0.8,
        )
    ]

    runner.mark_per_case_pass_fail(cfg, per_case)

    assert per_case[0].passed is False
    assert any("faithfulness" in note for note in per_case[0].notes)


def test_mark_per_case_pass_fail_passes_when_all_metrics_above_thresholds(cfg):
    per_case = [
        PerCaseScore(
            case_id="case-1",
            faithfulness=0.9,
            answer_relevancy=0.8,
            context_precision=0.8,
            context_recall=0.8,
            safety_passed=True
        )
    ]

    runner.mark_per_case_pass_fail(cfg, per_case)

    assert per_case[0].passed is True
    assert per_case[0].notes == []


def test_mark_per_case_pass_fail_uses_answer_relevancy_as_correctness_fallback(cfg):
    per_case = [
        PerCaseScore(
            case_id="case-1",
            faithfulness=0.9,
            answer_relevancy=0.4,
            answer_correctness=None,
            context_precision=0.9,
            context_recall=0.9,
            safety_passed=True
        )
    ]

    runner.mark_per_case_pass_fail(cfg, per_case)

    assert per_case[0].passed is False
    assert any("answer_relevancy" in note for note in per_case[0].notes)


def test_apply_threshold_policy_fails_on_context_recall_first(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=1,
        faithfulness_avg=0.9,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.5,
        latency_ms_avg=100,
    )

    failed, root_cause, notes = runner.apply_threshold_policy(cfg, agg)

    assert failed is True
    assert root_cause["stage"] == "Retrieval"
    assert root_cause["metric"] == "context_recall"
    assert len(notes) == 1


def test_apply_threshold_policy_fails_on_context_precision_when_recall_ok(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=1,
        faithfulness_avg=0.9,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.5,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    failed, root_cause, notes = runner.apply_threshold_policy(cfg, agg)

    assert failed is True
    assert root_cause["stage"] == "Retrieval"
    assert root_cause["metric"] == "context_precision"
    assert len(notes) == 1


def test_apply_threshold_policy_fails_on_faithfulness_when_retrieval_ok(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=1,
        faithfulness_avg=0.6,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    failed, root_cause, notes = runner.apply_threshold_policy(cfg, agg)

    assert failed is True
    assert root_cause["stage"] == "Generation"
    assert root_cause["metric"] == "faithfulness"
    assert len(notes) == 1


def test_apply_threshold_policy_fails_on_answer_relevancy_fallback(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=1,
        faithfulness_avg=0.9,
        answer_relevancy_avg=0.6,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    failed, root_cause, notes = runner.apply_threshold_policy(cfg, agg)

    assert failed is True
    assert root_cause["stage"] == "Generation"
    assert root_cause["metric"] == "relevancy"
    assert len(notes) == 1


def test_apply_threshold_policy_passes(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=2,
        faithfulness_avg=0.9,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    failed, root_cause, notes = runner.apply_threshold_policy(cfg, agg)

    assert failed is False
    assert root_cause == {"stage": "None"}
    assert notes == []


def test_apply_regression_policy_without_baseline(cfg):
    agg = AggregatedScores(
        total_cases=1,
        passed_cases=1,
        faithfulness_avg=0.9,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    regression = runner.apply_regression_policy(cfg, agg, None)

    assert regression.baseline_used is False
    assert regression.failed is False
    assert regression.drops == {}


def test_apply_regression_policy_fails_when_drop_exceeds_limit(cfg):
    agg = AggregatedScores(
        total_cases=1,
        passed_cases=1,
        faithfulness_avg=0.80,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    baseline = {
        "aggregatedScores": {
            "faithfulness_avg": 0.90,
        }
    }

    regression = runner.apply_regression_policy(cfg, agg, baseline)

    assert regression.baseline_used is True
    assert regression.failed is True
    assert regression.drops["faithfulness_avg"] == pytest.approx(-0.10)


def test_apply_regression_policy_passes_when_drop_within_limit(cfg):
    agg = AggregatedScores(
        total_cases=1,
        passed_cases=1,
        faithfulness_avg=0.86,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    baseline = {
        "aggregatedScores": {
            "faithfulness_avg": 0.90,
        }
    }

    regression = runner.apply_regression_policy(cfg, agg, baseline)

    assert regression.baseline_used is True
    assert regression.failed is False
    assert regression.drops["faithfulness_avg"] == pytest.approx(-0.04)


def test_build_gate_summary_pass(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=2,
        faithfulness_avg=0.9,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    regression = RegressionResult(
        baseline_used=False,
        max_drop=0.05,
        drops={},
        failed=False,
    )

    gate = runner.build_gate_summary(
        cfg=cfg,
        agg=agg,
        threshold_failed=False,
        threshold_root_cause={"stage": "None"},
        threshold_notes=[],
        regression=regression,
    )

    assert gate.verdict == "PASS"
    assert gate.rootCause == {"stage": "None"}
    assert "No threshold or regression failures detected" in gate.summary
    assert "No baseline available" in gate.summary
    assert "total_cases=2" in gate.summary
    assert "passed_cases=2" in gate.summary

    # artifacts = json.loads(gate.artifacts)
    assert gate.artifacts["gateSummaryUri"] == "s3://eval-bucket/artifacts/gate-summary.json"
    assert gate.artifacts["scoresUri"] == "s3://eval-bucket/artifacts/scores.json"
    assert gate.artifacts["detailsUri"] == "s3://eval-bucket/artifacts/details.jsonl"


def test_build_gate_summary_fail_from_threshold(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=1,
        faithfulness_avg=0.6,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    regression = RegressionResult(
        baseline_used=False,
        max_drop=0.05,
        drops={},
        failed=False,
    )

    gate = runner.build_gate_summary(
        cfg=cfg,
        agg=agg,
        threshold_failed=True,
        threshold_root_cause={
            "stage": "Generation",
            "metric": "faithfulness",
            "details": "faithfulness_avg=0.6000 < minFaithfulness=0.8000",
        },
        threshold_notes=[
            "faithfulness_avg=0.6000 < minFaithfulness=0.8000",
        ],
        regression=regression,
    )

    assert gate.verdict == "FAIL"
    assert gate.rootCause["stage"] == "Generation"
    assert "Threshold policy failed" in gate.summary


def test_build_gate_summary_fail_from_regression(cfg):
    agg = AggregatedScores(
        total_cases=2,
        passed_cases=2,
        faithfulness_avg=0.8,
        answer_relevancy_avg=0.9,
        answer_correctness_avg=None,
        context_precision_avg=0.9,
        context_recall_avg=0.9,
        latency_ms_avg=100,
    )

    regression = RegressionResult(
        baseline_used=True,
        max_drop=0.05,
        drops={"faithfulness_avg": -0.1},
        failed=True,
    )

    gate = runner.build_gate_summary(
        cfg=cfg,
        agg=agg,
        threshold_failed=False,
        threshold_root_cause={"stage": "None"},
        threshold_notes=[],
        regression=regression,
    )

    assert gate.verdict == "FAIL"
    assert gate.rootCause["stage"] == "Regression"
    assert "Regression policy failed" in gate.summary
    assert "Baseline comparison used" in gate.summary


# ============================================================
# Publishing
# ============================================================

def test_publish_details_jsonl(monkeypatch, cfg):
    captured = {}

    def fake_upload_text(cfg_arg, key, text, content_type):
        captured["cfg"] = cfg_arg
        captured["key"] = key
        captured["text"] = text
        captured["content_type"] = content_type

    monkeypatch.setattr(runner, "upload_text", fake_upload_text)

    records = [
        EvalRecord(
            case_id="case-1",
            question="Q",
            answer="A",
            ground_truth="GT",
            contexts=[
                RetrievedContext(id="ctx-1", text="Context"),
            ],
            metadata={"latencyMs": 100},
        )
    ]

    scores = [
        PerCaseScore(
            case_id="case-1",
            faithfulness=0.9,
            answer_relevancy=0.8,
            passed=True,
        )
    ]

    runner.publish_details_jsonl(cfg, records, scores)

    assert captured["cfg"] is cfg
    assert captured["key"] == cfg.details_key
    assert captured["content_type"] == "application/x-ndjson"

    lines = captured["text"].strip().splitlines()
    assert len(lines) == 1

    row = json.loads(lines[0])
    assert row["caseId"] == "case-1"
    assert row["question"] == "Q"
    assert row["answer"] == "A"
    assert row["groundTruth"] == "GT"
    assert row["contexts"][0]["id"] == "ctx-1"
    assert row["score"]["faithfulness"] == 0.9


def test_publish_scores_json(monkeypatch, cfg):
    captured = {}

    def fake_upload_json(cfg_arg, key, payload):
        captured["cfg"] = cfg_arg
        captured["key"] = key
        captured["payload"] = payload

    monkeypatch.setattr(runner, "upload_json", fake_upload_json)

    agg = AggregatedScores(
        total_cases=1,
        passed_cases=1,
        faithfulness_avg=0.9,
        answer_relevancy_avg=0.8,
        answer_correctness_avg=None,
        context_precision_avg=None,
        context_recall_avg=None,
        latency_ms_avg=100,
    )

    regression = RegressionResult(
        baseline_used=False,
        max_drop=0.05,
        drops={},
        failed=False,
    )

    per_case = [
        PerCaseScore(
            case_id="case-1",
            faithfulness=0.9,
            answer_relevancy=0.8,
            passed=True,
        )
    ]

    runner.publish_scores_json(
        cfg=cfg,
        agg=agg,
        regression=regression,
        per_case=per_case,
        ragas_meta={"metricNames": ["faithfulness", "answer_relevancy"]},
    )

    assert captured["cfg"] is cfg
    assert captured["key"] == cfg.scores_key

    payload = captured["payload"]
    assert payload["run"]["evalrun"] == "evalrun-1"
    assert payload["run"]["evalsuite"] == "suite-1"
    assert payload["engine"]["name"] == "ragas"
    assert payload["engine"]["metricNames"] == ["faithfulness", "answer_relevancy"]
    assert payload["aggregatedScores"]["total_cases"] == 1
    assert payload["perCase"][0]["case_id"] == "case-1"


def test_publish_gate_summary(monkeypatch, cfg):
    captured = {}

    def fake_upload_json(cfg_arg, key, payload):
        captured["cfg"] = cfg_arg
        captured["key"] = key
        captured["payload"] = payload

    monkeypatch.setattr(runner, "upload_json", fake_upload_json)

    gate = GateSummary(
        verdict="PASS",
        summary="ok",
        rootCause={"stage": "None"},
        artifacts="{}",
    )

    runner.publish_gate_summary(cfg, gate)

    assert captured["cfg"] is cfg
    assert captured["key"] == cfg.gate_summary_key
    assert captured["payload"]["verdict"] == "PASS"
    assert captured["payload"]["summary"] == "ok"


# ============================================================
# Main flow
# ============================================================

@pytest.mark.asyncio
async def test_main_success_flow(monkeypatch, cfg, tmp_path):
    uploaded = {
        "details": False,
        "scores": False,
        "gate": False,
    }

    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        '{"caseId":"case-1","question":"Q1","groundTruth":"GT1"}\n',
        encoding="utf-8",
    )
    cfg.dataset_file = str(dataset_path)

    monkeypatch.setattr(runner, "build_runner_config", lambda: cfg)
    monkeypatch.setattr(runner, "download_dataset", lambda cfg_arg: str(dataset_path))
    monkeypatch.setattr(runner, "load_optional_baseline", lambda cfg_arg: None)

    def fake_call_target_endpoint(cfg_arg, case):
        return EvalRecord(
            case_id=case.case_id,
            question=case.question,
            answer="A1",
            ground_truth=case.ground_truth,
            contexts=[
                RetrievedContext(id="ctx-1", text="Context 1"),
            ],
            metadata={"latencyMs": 100},
        )

    monkeypatch.setattr(runner, "call_target_endpoint", fake_call_target_endpoint)

    class FakeRagasAdapter:
        def __init__(self, model_name, embedder_name, api_key):
            self.model_name = model_name
            self.embedder_name = embedder_name
            self.api_key = api_key

        async def score_records(self, records):
            assert len(records) == 1
            assert records[0].case_id == "case-1"

            return [
                PerCaseScore(
                    case_id="case-1",
                    faithfulness=0.9,
                    answer_relevancy=0.9,
                    context_precision=0.9,
                    context_recall=0.9,
                    latency_ms=100,
                    safety_passed=True
                )
            ], {
                "metricNames": ["faithfulness", "answer_relevancy"],
            }

    monkeypatch.setattr(runner, "RagasAdapter", FakeRagasAdapter)

    def fake_publish_details_jsonl(cfg_arg, records, scores):
        uploaded["details"] = True
        assert len(records) == 1
        assert len(scores) == 1

    def fake_publish_scores_json(cfg_arg, agg, regression, per_case, ragas_meta):
        uploaded["scores"] = True
        assert agg.total_cases == 1
        assert agg.passed_cases == 1
        assert regression.failed is False
        assert ragas_meta["metricNames"] == ["faithfulness", "answer_relevancy"]

    def fake_publish_gate_summary(cfg_arg, gate):
        uploaded["gate"] = True
        assert gate.verdict == "PASS"

    monkeypatch.setattr(runner, "publish_details_jsonl", fake_publish_details_jsonl)
    monkeypatch.setattr(runner, "publish_scores_json", fake_publish_scores_json)
    monkeypatch.setattr(runner, "publish_gate_summary", fake_publish_gate_summary)

    exit_code = await runner.main()

    assert exit_code == 0
    assert uploaded == {
        "details": True,
        "scores": True,
        "gate": True,
    }
