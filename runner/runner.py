import json
import os
import sys
import time
import traceback
import asyncio
from pydantic import BaseModel
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import requests

from openai import AsyncOpenAI
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, AnswerCorrectness
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
    context_precision,
    context_recall,
)
from ragas.llms import llm_factory
from ragas.embeddings import OpenAIEmbeddings
from ragas import experiment, EvaluationDataset, SingleTurnSample
from sqlalchemy.engine import row

# answer_correctness may not exist in some ragas versions
# try:
#     from ragas.metrics import answer_correctness
# except Exception:
#     answer_correctness = None


# ============================================================
# Data models
# ============================================================

class SingleTurnExtension(SingleTurnSample):
  case_id: str
  ground_truth: str
  
# Define experiment result structure
class ExperimentResult(BaseModel):
  case_id: str
  faithfulness: float  
  answer_relevancy: float
  answer_correctness: float


@dataclass
class DatasetCase:
    case_id: str
    question: str
    ground_truth: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedContext:
    id: str
    text: str
    score: Optional[float] = None
    source: Optional[str] = None


@dataclass
class EvalRecord:
    case_id: str
    question: str
    answer: str
    ground_truth: Optional[str]
    contexts: List[RetrievedContext] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerCaseScore:
    case_id: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    answer_correctness: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    latency_ms: Optional[float] = None
    passed: bool = False
    notes: List[str] = field(default_factory=list)


@dataclass
class AggregatedScores:
    total_cases: int
    passed_cases: int

    faithfulness_avg: Optional[float]
    answer_relevancy_avg: Optional[float]
    answer_correctness_avg: Optional[float]
    context_precision_avg: Optional[float]
    context_recall_avg: Optional[float]

    latency_ms_avg: Optional[float]


@dataclass
class RegressionResult:
    baseline_used: bool
    max_drop: Optional[float]
    drops: Dict[str, float] = field(default_factory=dict)
    failed: bool = False


@dataclass
class GateSummary:
    verdict: str
    summary: str
    rootCause: Dict[str, Any]
    artifacts: Dict[str, str]


# ============================================================
# Helpers
# ============================================================

def getenv_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"missing required environment variable: {name}")
    return value


def getenv_optional(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.getenv(name, default)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def now_epoch_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def mean(values: List[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


# ============================================================
# Config loading
# ============================================================

@dataclass
class ThresholdConfig:
    min_context_recall: Optional[float] = None
    min_context_precision: Optional[float] = None
    min_faithfulness: Optional[float] = None
    min_relevancy: Optional[float] = None
    min_correctness: Optional[float] = None
    max_stale_chunk_ratio: Optional[float] = None
    max_unknown_owner_ratio: Optional[float] = None


@dataclass
class RunnerConfig:
    evalrun_name: str
    evalsuite_name: str
    git_sha: Optional[str]
    pr_number: Optional[str]

    target_endpoint: str
    request_timeout_seconds: int

    suite_config_path: str
    work_dir: str
    dataset_file: str

    s3_endpoint: str
    s3_bucket: str
    s3_access_key: str
    s3_secret_key: str

    dataset_s3_bucket: Optional[str]
    dataset_s3_key: str
    dataset_s3_version_id: Optional[str]

    s3_baseline_key: str
    gate_summary_key: str
    scores_key: str
    details_key: str

    llm_model: str
    llm_embedder: str
    llm_api_key: str

    thresholds: ThresholdConfig
    regression_max_drop: Optional[float] = None


def load_suite_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_runner_config() -> RunnerConfig:
    suite_config_path = getenv_required("SUITE_CONFIG")
    suite = load_suite_config(suite_config_path)

    metrics = suite.get("metrics", {})
    rag_metrics = metrics.get("rag", {})
    retrieval = rag_metrics.get("retrieval", {})
    generation = rag_metrics.get("generation", {})
    trust = rag_metrics.get("trustworthiness", {})
    regression = suite.get("regressionPolicy", {})
    execution = suite.get("execution", {})

    thresholds = ThresholdConfig(
        min_context_recall=safe_float(retrieval.get("minContextRecall", None)),
        min_context_precision=safe_float(retrieval.get("minContextPrecision", None)),
        min_faithfulness=safe_float(generation.get("minFaithfulness", None)),
        min_relevancy = safe_float(generation.get("minRelevancy", None)),
        min_correctness=safe_float(generation.get("minCorrectness", None)),
        max_stale_chunk_ratio=safe_float(trust.get("maxStaleChunkRatio", None)),
        max_unknown_owner_ratio=safe_float(trust.get("maxUnknownOwnerRatio", None)),
    )

    timeout_s = int(execution.get("timeoutSeconds", 60))

    return RunnerConfig(
        evalrun_name=getenv_required("EVALRUN_NAME"),
        evalsuite_name=getenv_required("EVALSUITE_NAME"),
        git_sha=getenv_optional("GIT_SHA"),
        pr_number=getenv_optional("PR_NUMBER"),

        target_endpoint=getenv_required("TARGET_ENDPOINT"),
        request_timeout_seconds=timeout_s,

        suite_config_path=suite_config_path,
        work_dir=getenv_optional("WORK_DIR", "/work"),
        dataset_file=getenv_optional("DATASET_FILE", "/work/dataset"),

        s3_endpoint=getenv_required("S3_ENDPOINT"),
        s3_bucket=getenv_required("DATASET_S3_BUCKET"),
        s3_access_key=getenv_required("S3_ACCESS_KEY"),
        s3_secret_key=getenv_required("S3_SECRET_KEY"),

        dataset_s3_bucket=getenv_optional("DATASET_S3_BUCKET"),
        dataset_s3_key=getenv_required("DATASET_S3_KEY"),
        dataset_s3_version_id=getenv_optional("DATASET_S3_VERSION_ID"),

        s3_baseline_key=getenv_required("S3_BASELINE_KEY"),
        gate_summary_key=getenv_required("GATE_SUMMARY_KEY"),
        scores_key=getenv_required("SCORES_KEY"),
        details_key=getenv_required("DETAILS_KEY"),

        thresholds=thresholds,
        regression_max_drop=safe_float(regression.get("maxDrop")),

        llm_model=getenv_required("OPENAI_MODEL"),
        llm_embedder=getenv_required("OPENAI_EMBEDDER"),
        llm_api_key=getenv_required("OPENAI_API_KEY")
    )


# ============================================================
# S3 client
# ============================================================

def download_dataset(cfg: RunnerConfig) -> str:
    bucket = cfg.dataset_s3_bucket or cfg.s3_bucket
    key = cfg.dataset_s3_key

    ensure_dir(os.path.dirname(cfg.dataset_file))
    resp = requests.get(f"{cfg.s3_endpoint}/buckets/{bucket}/objects/{key}")
    try:
        resp.raise_for_status()
    except Exception:
        print(f"Unable to call GET request to {cfg.s3_endpoint} during downloading a dataset")
        raise
    
    with open(cfg.dataset_file, "wb") as fd:
        for chunk in resp.iter_content(chunk_size=128):
          fd.write(chunk)

    return cfg.dataset_file


def load_optional_baseline(cfg: RunnerConfig) -> Optional[Dict[str, Any]]:
    resp = requests.get(f"{cfg.s3_endpoint}/buckets/{cfg.s3_bucket}/objects/{cfg.s3_baseline_key}")
    
    if getattr(resp, "status_code", None) == 404:
        return None
      
    try:
        resp.raise_for_status()
    except:
        print(f"Unable to call GET request to {cfg.s3_endpoint} during loading an optional baseline")
        raise
    return resp.json()

def upload_json(cfg: RunnerConfig, key: str, payload: Dict[str, Any]) -> None:
    def safe_serialize(obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        if isinstance(obj, float) and (obj != obj or obj == float('inf') or obj == float('-inf')):
            return None # NaN и Inf в null
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
    body = json.dumps(payload, ensure_ascii=False, indent=2, default=safe_serialize).encode("utf-8")
    
    resp = requests.put(
        f"{cfg.s3_endpoint}/buckets/{cfg.s3_bucket}/objects/{key}", 
        data=body,
        headers={"Content-Type": "application/json"}
    )

    try:
        resp.raise_for_status()
    except Exception:
        print(f"Unable to do PUT request to {cfg.s3_endpoint} during uploading a json content")
        raise
    return resp.json()



def upload_text(cfg: RunnerConfig, key: str, text: str, content_type: str = "text/plain") -> None:
    headers = {"Content-Type": f"{content_type}; charset=utf-8"}

    resp = requests.put(
        f"{cfg.s3_endpoint}/buckets/{cfg.s3_bucket}/objects/{key}",
        data=text.encode("utf-8"), headers=headers
    )
    
    try:
        resp.raise_for_status()
    except:
        print(f"Unable to do PUT request to {cfg.s3_endpoint} during uploading a text content")
        raise
    return resp.json()



# ============================================================
# Dataset parsing
# ============================================================

def parse_dataset_jsonl(path: str) -> List[DatasetCase]:
    cases: List[DatasetCase] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            case_id = raw.get("caseId")
            question = raw.get("question")
            ground_truth = raw.get("groundTruth")

            if not case_id:
                raise ValueError(f"dataset line {line_no}: missing caseId")
            if not question:
                raise ValueError(f"dataset line {line_no}: missing question")

            cases.append(
                DatasetCase(
                    case_id=case_id,
                    question=question,
                    ground_truth=ground_truth,
                    tags=raw.get("tags", []),
                    metadata=raw.get("metadata", {}),
                )
            )

    return cases


# ============================================================
# Target endpoint client
# ============================================================

def call_target_endpoint(cfg: RunnerConfig, case: DatasetCase) -> EvalRecord:
    headers = {
        "Content-Type": "application/json",
    }

    payload = {
        "question": case.question,
    }

    started_ms = now_epoch_ms()
    resp = requests.post(
        cfg.target_endpoint,
        headers=headers,
        json=payload,
        timeout=cfg.request_timeout_seconds,
    )
    elapsed_ms = now_epoch_ms() - started_ms

    resp.raise_for_status()
    data = resp.json()

    answer = data.get("answer")
    if not answer:
        raise RuntimeError(f"target endpoint returned empty answer for case={case.case_id}")

    contexts_raw = data.get("contexts", []) or []
    contexts: List[RetrievedContext] = []
    for idx, item in enumerate(contexts_raw):
        contexts.append(
            RetrievedContext(
                id=item.get("id") or f"ctx-{idx}",
                text=item.get("text", ""),
                score=safe_float(item.get("score")),
                source=item.get("source"),
            )
        )

    metadata = data.get("metadata", {}) or {}
    if "latencyMs" not in metadata:
        metadata["latencyMs"] = elapsed_ms

    return EvalRecord(
        case_id=case.case_id,
        question=case.question,
        answer=answer,
        ground_truth=case.ground_truth,
        contexts=contexts,
        metadata=metadata,
    )


# ============================================================
# Ragas adapter
# ============================================================

class RagasAdapter:
    """
    Adapter layer around ragas so runner core stays stable.
    """

    def __init__(self, model_name, embedder_name, api_key) -> None:
        self._model_name = model_name
        self._embedder_name = embedder_name
        self._openai_api_key = api_key
        pass
      
    @staticmethod
    def _build_hf_dataset(records: List[EvalRecord]) -> EvaluationDataset:
        """
        Build dataset in a format that ragas expects.

        Ragas generally wants:
        - question
        - answer
        - contexts: list[list[str]]
        - ground_truth (or ground_truths in some flows)

        We keep ground_truth singular here for the common 0.1.x path.
        """
        
        '''
        Use ragas v0.4 dataset
        '''
        samples = []
        for record in records:
            samples.append(
                SingleTurnExtension(
                    case_id=record.case_id,
                    user_input=record.question,
                    response=record.answer,
                    retrieved_contexts = [context.text for context in record.contexts],
                    ground_truth=record.ground_truth
                )
            )
        return EvaluationDataset(samples=samples)

    @staticmethod
    def _choose_metrics(records: List[EvalRecord]) -> Tuple[List[Any], List[str]]:
        has_ground_truth = any(bool(r.ground_truth) for r in records)

        metric_defs: List[Tuple[str, Any]] = [
            ("faithfulness", faithfulness),
            ("answer_relevancy", answer_relevancy),
            ("answer_correctness", answer_correctness)
        ]

        if has_ground_truth:
            metric_defs.append(("context_precision", context_precision))
            metric_defs.append(("context_recall", context_recall))

        metrics = [m for _, m in metric_defs]
        metric_names = [n for n, _ in metric_defs]
        return metrics, metric_names

    async def score_records(self, records: List[EvalRecord]) -> Tuple[List[PerCaseScore], Dict[str, Any]]:
        metric_names = ["faithfulness", "answer_relevancy", "answer_correctness"]
        generation_config = {
          "max_completion_tokens": 8192, 
          "max_tokens": 8192
        }
        openai_async_client = AsyncOpenAI(api_key=self._openai_api_key)
        ragas_llm = llm_factory(
          model=self._model_name,
          client=openai_async_client,
          **generation_config
        )
        
        embedder = OpenAIEmbeddings(
            client=openai_async_client,
            model=self._embedder_name,
        )

        @experiment(ExperimentResult, backend="inmemory")
        async def run_evaluation(row):
          answer_correctness = AnswerCorrectness(llm=ragas_llm, embeddings=embedder)
          faithfulness = Faithfulness(llm=ragas_llm)
          answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=embedder)
          
          correctness_result = await answer_correctness.ascore(
            user_input=row.user_input,
            response=row.response,
            reference=row.ground_truth
          )
          faith_result = await faithfulness.ascore(       
            response=row.response,
            retrieved_contexts=row.retrieved_contexts,
            user_input=row.user_input
          )
          relevancy_result = await answer_relevancy.ascore(       
            user_input=row.user_input,
            response=row.response
          )
          
          return ExperimentResult(
            case_id=row.case_id,
            faithfulness=faith_result.value,
            answer_relevancy=relevancy_result.value,
            answer_correctness=correctness_result.value
          )
        
        if not records:
          return [], {"metricNames": []}

        dataset = self._build_hf_dataset(records)
        result = await run_evaluation.arun(dataset)
        
        df = result.to_pandas()
        per_case_scores: List[PerCaseScore] = []

        for idx, row in df.iterrows():
          latency_ms = safe_float(records[idx].metadata.get("latencyMs"))

          pcs = PerCaseScore(
            case_id=row.get("case_id"),
            faithfulness=safe_float(row.get("faithfulness")),
            answer_relevancy=safe_float(row.get("answer_relevancy")),
            answer_correctness=safe_float(row.get("answer_correctness")) if "answer_correctness" in row else None,
            context_precision=safe_float(row.get("context_precision")) if "context_precision" in row else None,
            context_recall=safe_float(row.get("context_recall")) if "context_recall" in row else None,
            latency_ms=latency_ms,
            passed=False,
            notes=[],
          )

          per_case_scores.append(pcs)

        return per_case_scores, {
          "metricNames": metric_names,
        }


# ============================================================
# Aggregation / analyzer
# ============================================================

def aggregate_scores(per_case: List[PerCaseScore]) -> AggregatedScores:
    return AggregatedScores(
        total_cases=len(per_case),
        passed_cases=sum(1 for x in per_case if x.passed),

        faithfulness_avg=mean([x.faithfulness for x in per_case]),
        answer_relevancy_avg=mean([x.answer_relevancy for x in per_case]),
        answer_correctness_avg=mean([x.answer_correctness for x in per_case]),
        context_precision_avg=mean([x.context_precision for x in per_case]),
        context_recall_avg=mean([x.context_recall for x in per_case]),

        latency_ms_avg=mean([x.latency_ms for x in per_case]),
    )


def mark_per_case_pass_fail(cfg: RunnerConfig, per_case: List[PerCaseScore]) -> None:
    """
    Simple per-case pass/fail marking based on available thresholds.
    Final verdict is still computed at run level.
    """
    for score in per_case:
        notes: List[str] = []
        passed = True

        if cfg.thresholds.min_faithfulness is not None and score.faithfulness is not None:
            if score.faithfulness < cfg.thresholds.min_faithfulness:
                passed = False
                notes.append(
                    f"faithfulness={score.faithfulness:.4f} < minFaithfulness={cfg.thresholds.min_faithfulness:.4f}"
                )


        if cfg.thresholds.min_correctness is not None and score.answer_correctness is not None:
            if score.answer_correctness < cfg.thresholds.min_correctness:
                passed = False
                notes.append(
                    f"answer_correctness={score.answer_correctness:.4f} < minCorrectness={cfg.thresholds.min_correctness:.4f}"
                )
                
        if cfg.thresholds.min_relevancy is not None and score.answer_relevancy is not None:
          if score.answer_relevancy < cfg.thresholds.min_relevancy:
            passed = False
            notes.append(
                    f"answer_relevancy={score.answer_relevancy:.4f} < minRelevancy={cfg.thresholds.min_relevancy:.4f}"
                )
          
        if cfg.thresholds.min_context_precision is not None and score.context_precision is not None:
            if score.context_precision < cfg.thresholds.min_context_precision:
                passed = False
                notes.append(
                    f"context_precision={score.context_precision:.4f} < minContextPrecision={cfg.thresholds.min_context_precision:.4f}"
                )

        if cfg.thresholds.min_context_recall is not None and score.context_recall is not None:
            if score.context_recall < cfg.thresholds.min_context_recall:
                passed = False
                notes.append(
                    f"context_recall={score.context_recall:.4f} < minContextRecall={cfg.thresholds.min_context_recall:.4f}"
                )

        score.passed = passed
        score.notes.extend(notes)


def apply_threshold_policy(cfg: RunnerConfig, agg: AggregatedScores) -> Tuple[bool, Dict[str, Any], List[str]]:
    failed = False
    notes: List[str] = []
    root_cause = {"stage": "None"}

    # Retrieval stage checks
    if cfg.thresholds.min_context_recall is not None and agg.context_recall_avg is not None:
      if agg.context_recall_avg < cfg.thresholds.min_context_recall:
        failed = True
        root_cause = {
          "stage": "Retrieval",
          "metric": "context_recall",
          "details": f"context_recall_avg={agg.context_recall_avg:.4f} < minContextRecall={cfg.thresholds.min_context_recall:.4f}",
        }
        notes.append(root_cause["details"])

    if not failed and cfg.thresholds.min_context_precision is not None and agg.context_precision_avg is not None:
      if agg.context_precision_avg < cfg.thresholds.min_context_precision:
        failed = True
        root_cause = {
          "stage": "Retrieval",
          "metric": "context_precision",
          "details": f"context_precision_avg={agg.context_precision_avg:.4f} < minContextPrecision={cfg.thresholds.min_context_precision:.4f}",
        }
        notes.append(root_cause["details"])

    # Generation stage checks
    if not failed and cfg.thresholds.min_faithfulness is not None and agg.faithfulness_avg is not None:
      if agg.faithfulness_avg < cfg.thresholds.min_faithfulness:
        failed = True
        root_cause = {
          "stage": "Generation",
          "metric": "faithfulness",
          "details": f"faithfulness_avg={agg.faithfulness_avg:.4f} < minFaithfulness={cfg.thresholds.min_faithfulness:.4f}",
        }
        notes.append(root_cause["details"])


    if not failed and cfg.thresholds.min_correctness is not None and agg.answer_correctness_avg is not None:
      if agg.answer_correctness_avg < cfg.thresholds.min_correctness:
        failed = True
        root_cause = {
          "stage": "Generation",
          "metric": "correctness",
          "details": f"answer_correctness_avg={agg.answer_correctness_avg:.4f} < minCorrectness={cfg.thresholds.min_correctness:.4f}",
        }
        notes.append(root_cause["details"])
      
    if not failed and cfg.thresholds.min_relevancy is not None and agg.answer_relevancy_avg is not None:
      if agg.answer_relevancy_avg < cfg.thresholds.min_relevancy:
        failed = True
        root_cause = {
          "stage": "Generation",
          "metric": "relevancy",
          "details": f"answer_relevancy_avg={agg.answer_relevancy_avg:.4f} < minFaithfulness={cfg.thresholds.min_relevancy:.4f}",
        }
        notes.append(root_cause["details"])

    return failed, root_cause, notes


def apply_regression_policy(
    cfg: RunnerConfig,
    agg: AggregatedScores,
    baseline: Optional[Dict[str, Any]],
) -> RegressionResult:
    if not baseline:
        return RegressionResult(
            baseline_used=False,
            max_drop=cfg.regression_max_drop,
            drops={},
            failed=False,
        )

    max_drop = cfg.regression_max_drop
    drops: Dict[str, float] = {}
    failed = False

    baseline_scores = baseline.get("aggregatedScores", {})

    metric_pairs = {
        "faithfulness_avg": agg.faithfulness_avg,
        "answer_relevancy_avg": agg.answer_relevancy_avg,
        "answer_correctness_avg": agg.answer_correctness_avg,
        "context_precision_avg": agg.context_precision_avg,
        "context_recall_avg": agg.context_recall_avg,
    }

    for metric_name, current_value in metric_pairs.items():
        baseline_value = safe_float(baseline_scores.get(metric_name))
        if baseline_value is None or current_value is None:
            continue

        drop = current_value - baseline_value
        drops[metric_name] = drop
        if max_drop is not None and drop < -max_drop:
            failed = True

    return RegressionResult(
        baseline_used=True,
        max_drop=max_drop,
        drops=drops,
        failed=failed,
    )


def build_gate_summary(
    cfg: RunnerConfig,
    agg: AggregatedScores,
    threshold_failed: bool,
    threshold_root_cause: Dict[str, Any],
    threshold_notes: List[str],
    regression: RegressionResult,
) -> GateSummary:
    verdict = "PASS"
    root_cause = {"stage": "None"}
    summary_parts: List[str] = []

    if threshold_failed:
        verdict = "FAIL"
        root_cause = threshold_root_cause
        summary_parts.append("Threshold policy failed")

    if regression.failed:
        verdict = "FAIL"
        if root_cause.get("stage") == "None":
            root_cause = {
                "stage": "Regression",
                "metric": "baseline",
                "details": f"metric regression exceeded maxDrop={regression.max_drop}",
            }
        summary_parts.append("Regression policy failed")

    if not summary_parts:
        summary_parts.append("No threshold or regression failures detected")

    if threshold_notes:
        summary_parts.extend(threshold_notes)

    if regression.baseline_used:
        summary_parts.append("Baseline comparison used")
    else:
        summary_parts.append("No baseline available")

    summary_parts.append(f"total_cases={agg.total_cases}")
    summary_parts.append(f"passed_cases={agg.passed_cases}")

    artifacts = {
        "gateSummaryUri": f"s3://{cfg.s3_bucket}/{cfg.gate_summary_key}",
        "scoresUri": f"s3://{cfg.s3_bucket}/{cfg.scores_key}",
        "detailsUri": f"s3://{cfg.s3_bucket}/{cfg.details_key}",
    }

    return GateSummary(
        verdict=verdict,
        summary="; ".join(summary_parts),
        rootCause=root_cause,
        artifacts=artifacts
    )


# ============================================================
# Publishing
# ============================================================

def publish_details_jsonl(cfg: RunnerConfig, records: List[EvalRecord], scores: List[PerCaseScore]) -> None:
    score_by_case = {x.case_id: x for x in scores}
    lines: List[str] = []

    for record in records:
        score = score_by_case.get(record.case_id)
        row = {
            "caseId": record.case_id,
            "question": record.question,
            "groundTruth": record.ground_truth,
            "answer": record.answer,
            "contexts": [asdict(c) for c in record.contexts],
            "metadata": record.metadata,
            "score": asdict(score) if score else None,
        }
        lines.append(json.dumps(row, ensure_ascii=False))

    payload = "\n".join(lines) + "\n"
    upload_text(cfg, cfg.details_key, payload, content_type="application/x-ndjson")


def publish_scores_json(
    cfg: RunnerConfig,
    agg: AggregatedScores,
    regression: RegressionResult,
    per_case: List[PerCaseScore],
    ragas_meta: Dict[str, Any],
) -> None:
    payload = {
        "run": {
            "evalrun": cfg.evalrun_name,
            "evalsuite": cfg.evalsuite_name,
            "gitSha": cfg.git_sha,
            "prNumber": cfg.pr_number,
        },
        "engine": {
            "name": "ragas",
            "metricNames": ragas_meta.get("metricNames", []),
        },
        "aggregatedScores": asdict(agg),
        "regression": asdict(regression),
        "perCase": [asdict(x) for x in per_case],
    }
    upload_json(cfg, cfg.scores_key, payload)


def publish_gate_summary(cfg: RunnerConfig, gate: GateSummary) -> None:
    payload = asdict(gate)
    upload_json(cfg, cfg.gate_summary_key, payload)


# ============================================================
# Main
# ============================================================

async def main() -> int:
    try:
        cfg = build_runner_config()
        ensure_dir(cfg.work_dir)

        # 1) Download dataset
        dataset_path = download_dataset(cfg)

        # 2) Parse dataset
        cases = parse_dataset_jsonl(dataset_path)
        if not cases:
            raise RuntimeError("dataset is empty")

        # 3) Load optional baseline
        baseline = load_optional_baseline(cfg)

        # 4) Call target endpoint and collect eval records
        records: List[EvalRecord] = []
        for case in cases:
            record = call_target_endpoint(cfg, case)
            records.append(record)

        # 5) Evaluate with Ragas
        ragas_adapter = RagasAdapter(
            model_name=cfg.llm_model, 
            embedder_name=cfg.llm_embedder, 
            api_key=cfg.llm_api_key
        )
        per_case_scores, ragas_meta = await ragas_adapter.score_records(records)

        # 6) Per-case pass/fail markers
        mark_per_case_pass_fail(cfg, per_case_scores)

        # 7) Aggregate
        agg = aggregate_scores(per_case_scores)

        # 8) Threshold policy
        threshold_failed, threshold_root_cause, threshold_notes = apply_threshold_policy(cfg, agg)

        # 9) Regression policy
        regression = apply_regression_policy(cfg, agg, baseline)

        # 10) Gate summary
        gate = build_gate_summary(
            cfg=cfg,
            agg=agg,
            threshold_failed=threshold_failed,
            threshold_root_cause=threshold_root_cause,
            threshold_notes=threshold_notes,
            regression=regression,
        )

        # 11) Publish artifacts
        publish_details_jsonl(cfg, records, per_case_scores)
        publish_scores_json(cfg, agg, regression, per_case_scores, ragas_meta)
        publish_gate_summary(cfg, gate)

        # Technical execution completed successfully
        return 0

    except Exception as e:
        # Best-effort publish of execution failure summary
        try:
            cfg = build_runner_config()

            gate = GateSummary(
                verdict="FAIL",
                summary=f"Execution failure: {str(e)}",
                rootCause={
                    "stage": "Execution",
                    "details": traceback.format_exc()[-3000:],
                },
                artifacts={
                    "gateSummaryUri": f"s3://{cfg.s3_bucket}/{cfg.gate_summary_key}",
                    "scoresUri": f"s3://{cfg.s3_bucket}/{cfg.scores_key}",
                    "detailsUri": f"s3://{cfg.s3_bucket}/{cfg.details_key}",
                },
            )
            publish_gate_summary(cfg, gate)
        except Exception:
            pass

        print("runner failed:", str(e), file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))