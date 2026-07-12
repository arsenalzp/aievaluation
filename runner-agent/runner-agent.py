import json
import os
import sys
import traceback
import asyncio
import time
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from phoenix.evals import evaluate_dataframe
from phoenix.evals.metrics import (
  ToolInvocationEvaluator, 
  ToolSelectionEvaluator, 
  CorrectnessEvaluator,
  HallucinationEvaluator
)
import requests
from llm import base_llm

# ============================================================
# Data models
# ============================================================

@dataclass
class AgentTool:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentToolCall:
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    status: Optional[str] = None


@dataclass
class AgentTrace:
    trace_id: Optional[str]
    source: str
    framework: Optional[str]
    tool_calls: List[AgentToolCall] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentDatasetCase:
    case_id: str
    input: str
    expected_output: Optional[str] = None
    expected_tools: List[AgentTool] = field(default_factory=list)
    forbidden_tools: List[AgentTool] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentEvalRecord:
    case_id: str
    input: str
    actual_output: str
    expected_output: Optional[str]
    expected_tools: List[AgentTool] = field(default_factory=list)
    forbidden_tools: List[AgentTool] = field(default_factory=list)
    tools_called: List[AgentToolCall] = field(default_factory=list)
    trace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerCaseScore:
    case_id: str
    policy_violations: int = 0
    latency_ms: Optional[float] = None
    passed: bool = False
    notes: List[str] = field(default_factory=list)
    
    tool_selection: Optional[float] = None
    tool_invocation: Optional[float] = None
    tool_execution: Optional[float] = None

@dataclass
class AgentAggregatedScores:
    total_cases: int
    passed_cases: int

    tool_selection_avg: Optional[float]
    tool_invocation_avg: Optional[float]
    tool_execution_avg: Optional[float]

    policy_violations_total: int
    latency_ms_avg: Optional[float]


@dataclass
class RegressionResult:
    baseline_used: bool
    max_drop: Optional[float]
    drops: Dict[str, float] = field(default_factory=dict)
    failed: bool = False


@dataclass
class GateSummary:
    schemaVersion: str
    evaluationType: str
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


def getenv_optional(name: str, default: Optional[str] = None) -> Optional: 
  return os.getenv(name, default)


def safe_float(value: Any) -> Optional:
    try:
        if value is None:
            return None
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any, default: Optional[int] = None) -> Optional:
    try:
        if value is None:
            return default
        if value == "":
            return default
        return int(value)
    except Exception:
        return default


def now_epoch_ms() -> int:
    return int(time.time() * 1000)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def mean(values: List[Optional[float]]) -> Optional:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def normalize_name(value: str) -> str:
    return value.strip()


def tool_key(tool: AgentTool | AgentToolCall) -> str:
    return normalize_name(tool.name)


# ============================================================
# Config loading
# ============================================================

@dataclass
class AgentThresholdConfig:
    min_tool_selection: Optional[float] = None
    min_tool_execution: Optional[float] = None
    min_tool_invocation: Optional[float] = None
    max_policy_violations: Optional[int] = 0


@dataclass
class TraceConfig:
    mode: str = "inline"
    include_tool_outputs: bool = False
    sanitize: bool = True


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
    llm_api_key: str

    thresholds: AgentThresholdConfig
    trace: TraceConfig
    regression_max_drop: Optional[float] = None


def load_suite_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_runner_config() -> RunnerConfig:
    suite_config_path = getenv_required("SUITE_CONFIG")
    suite = load_suite_config(suite_config_path)

    metrics = suite.get("metrics", {})
    agent_metrics = metrics.get("agent", {})
    policy_metrics = metrics.get("policy", {})
    regression = suite.get("regressionPolicy", {})
    execution = suite.get("execution", {})
    trace_cfg = suite.get("trace", {})

    thresholds = AgentThresholdConfig(
        min_tool_selection=safe_float(agent_metrics.get("minToolSelection")),
        min_tool_execution=safe_float(agent_metrics.get("minToolExecution")),
        min_tool_invocation=safe_float(agent_metrics.get("minToolInvocation")),
        max_policy_violations=safe_int(policy_metrics.get("maxPolicyViolations"), default=0),
    )
    
    print(f"thresholds={thresholds}")

    trace = TraceConfig(
        mode=trace_cfg.get("mode", "inline"),
        include_tool_outputs=bool(trace_cfg.get("includeToolOutputs", False)),
        sanitize=bool(trace_cfg.get("sanitize", True)),
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

        llm_model=getenv_required("OPENAI_MODEL"),
        llm_api_key=getenv_required("OPENAI_API_KEY"),

        thresholds=thresholds,
        trace=trace,
        regression_max_drop=safe_float(regression.get("maxDrop")),
    )


# ============================================================
# S3-like client
# ============================================================

def download_dataset(cfg: RunnerConfig) -> str:
    print("--downloading dataset--")
    bucket = cfg.dataset_s3_bucket or cfg.s3_bucket
    key = cfg.dataset_s3_key

    ensure_dir(os.path.dirname(cfg.dataset_file))

    resp = requests.get(f"{cfg.s3_endpoint}/buckets/{bucket}/objects/{key}")

    try:
        resp.raise_for_status()
    except Exception:
        print(f"Unable to call GET request to {cfg.s3_endpoint} during downloading dataset")
        raise

    with open(cfg.dataset_file, "wb") as fd:
        for chunk in resp.iter_content(chunk_size=128):
            fd.write(chunk)

    return cfg.dataset_file


def load_optional_baseline(cfg: RunnerConfig) -> Optional[Dict[str, Any]]:
    print("--downloading optional dataset--")
    resp = requests.get(f"{cfg.s3_endpoint}/buckets/{cfg.s3_bucket}/objects/{cfg.s3_baseline_key}")

    if getattr(resp, "status_code", None) == 404:
        return None

    try:
        resp.raise_for_status()
    except Exception:
        print(f"Unable to call GET request to {cfg.s3_endpoint} during loading optional baseline")
        raise

    return resp.json()


def upload_json(cfg: RunnerConfig, key: str, payload: Dict[str, Any]) -> Any:
    print("--uploading JSON--")
    def safe_serialize(obj):
        if isinstance(obj, bytes):
            return obj.decode("utf-8", errors="ignore")
        if isinstance(obj, float) and (obj != obj or obj == float("inf") or obj == float("-inf")):
            return None
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    body = json.dumps(payload, ensure_ascii=False, indent=2, default=safe_serialize).encode("utf-8")

    resp = requests.put(
        f"{cfg.s3_endpoint}/buckets/{cfg.s3_bucket}/objects/{key}",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    try:
        resp.raise_for_status()
    except Exception:
        print(f"Unable to do PUT request to {cfg.s3_endpoint} during uploading json content")
        raise

    return resp.json()


def upload_text(cfg: RunnerConfig, key: str, text: str, content_type: str = "text/plain") -> Any:
    print("--uploading text--")
    headers = {"Content-Type": f"{content_type}; charset=utf-8"}

    resp = requests.put(
        f"{cfg.s3_endpoint}/buckets/{cfg.s3_bucket}/objects/{key}",
        data=text.encode("utf-8"),
        headers=headers,
    )

    try:
        resp.raise_for_status()
    except Exception:
        print(f"Unable to do PUT request to {cfg.s3_endpoint} during uploading text content")
        raise

    return resp.json()


# ============================================================
# Dataset parsing
# ============================================================

def parse_tool(raw: Dict[str, Any]) -> AgentTool:
    name = raw.get("name")
    if not name:
        raise ValueError("tool is missing name")

    return AgentTool(
        name=name,
        arguments=raw.get("arguments", {}) or {},
    )


def parse_dataset_jsonl(path: str) -> Listcases:
    cases: List[AgentDatasetCase] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            raw = json.loads(line)

            case_id = raw.get("caseId")
            input_text = raw.get("input") or raw.get("question")

            if not case_id:
                raise ValueError(f"dataset line {line_no}: missing caseId")
            if not input_text:
                raise ValueError(f"dataset line {line_no}: missing input")

            expected_tools = [
                parse_tool(x)
                for x in raw.get("expectedTools", []) or []
            ]

            forbidden_tools = [
                parse_tool(x)
                for x in raw.get("forbiddenTools", []) or []
            ]

            cases.append(
                AgentDatasetCase(
                    case_id=case_id,
                    input=input_text,
                    expected_output=raw.get("expectedOutput") or raw.get("groundTruth"),
                    expected_tools=expected_tools,
                    forbidden_tools=forbidden_tools,
                    tags=raw.get("tags", []),
                    metadata=raw.get("metadata", {}),
                )
            )

    return cases


# ============================================================
# Agent response / trace normalization
# ============================================================

def parse_tool_call(raw: Dict[str, Any]) -> AgentToolCall:
    print("--parsing tool call--")
    name = raw.get("name") or raw.get("tool")
    if not name:
        raise ValueError("tool call is missing name")

    status = raw.get("status")
    error = raw.get("error")

    if not status:
        status = "ERROR" if error else "OK"

    return AgentToolCall(
        name=name,
        arguments=raw.get("arguments", {}) or raw.get("args", {}) or {},
        output=raw.get("output"),
        duration_ms=safe_float(raw.get("durationMs") or raw.get("duration_ms")),
        error=error,
        status=status,
    )


def normalize_inline_trace(data: Dict[str, Any]) -> AgentTrace:
    metadata = data.get("metadata", {}) or {}
    trace = data.get("trace", {}) or {}

    tool_calls_raw = (
        trace.get("toolCalls")
        or trace.get("tool_calls")
        or data.get("toolCalls")
        or data.get("toolsCalled")
        or []
    )

    tool_calls = [
        parse_tool_call(x)
        for x in tool_calls_raw
    ]

    return AgentTrace(
        trace_id=trace.get("traceId") or metadata.get("traceId"),
        source=trace.get("source") or "inline",
        framework=trace.get("framework") or metadata.get("framework"),
        tool_calls=tool_calls,
        steps=trace.get("steps", []) or [],
        raw=trace,
    )


def resolve_trace_ref(trace_ref: Dict[str, Any]) -> AgentTrace:
    """
    Production extension point.

    In MVP we do not fetch traces from LangSmith/Phoenix/OTel backend.
    The contract is supported, but actual traceRef resolution should be added
    as a separate adapter.
    """
    trace_id = trace_ref.get("traceId")

    raise RuntimeError(
        f"traceRef resolution is not supported in MVP; traceId={trace_id}"
    )


def resolve_agent_trace(data: Dict[str, Any]) -> AgentTrace:
    if data.get("trace"):
        return normalize_inline_trace(data)

    if data.get("traceRef"):
        return resolve_trace_ref(data["traceRef"])

    metadata = data.get("metadata", {}) or {}

    return AgentTrace(
        trace_id=metadata.get("traceId"),
        source="none",
        framework=metadata.get("framework"),
        tool_calls=[],
        steps=[],
        raw={},
    )


# ============================================================
# Target endpoint client
# ============================================================

def call_agent_endpoint_raw(cfg: RunnerConfig, case: AgentDatasetCase) -> Dict[str, Any]:
    print("--calling an agent endpoint--")
    headers = {"Content-Type": "application/json"}

    payload = {
        "input": case.input,
        "evaluationMode": True,
        "trace": {
            "mode": cfg.trace.mode,
            "includeToolOutputs": cfg.trace.include_tool_outputs,
            "sanitize": cfg.trace.sanitize,
        },
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

    metadata = data.get("metadata", {}) or {}
    if "latencyMs" not in metadata:
        metadata["latencyMs"] = elapsed_ms
    data["metadata"] = metadata

    return data


def build_agent_eval_record(case: AgentDatasetCase, response_data: Dict[str, Any]) -> AgentEvalRecord:
    output = response_data.get("output") or response_data.get("answer")
    if not output:
        raise RuntimeError(f"agent endpoint returned empty output for case={case.case_id}")

    trace = resolve_agent_trace(response_data)

    metadata = response_data.get("metadata", {}) or {}
    if trace.trace_id and "traceId" not in metadata:
        metadata["traceId"] = trace.trace_id

    return AgentEvalRecord(
        case_id=case.case_id,
        input=case.input,
        actual_output=output,
        expected_output=case.expected_output,
        expected_tools=case.expected_tools,
        forbidden_tools=case.forbidden_tools,
        tools_called=trace.tool_calls,
        trace={
            "traceId": trace.trace_id,
            "source": trace.source,
            "framework": trace.framework,
            "toolCalls": [asdict(x) for x in trace.tool_calls],
            "steps": trace.steps,
            "raw": trace.raw,
        },
        metadata=metadata,
    )


def call_agent_endpoint(cfg: RunnerConfig, case: AgentDatasetCase) -> AgentEvalRecord:
    response_data = call_agent_endpoint_raw(cfg, case)
    return build_agent_eval_record(case, response_data)

# ============================================================
# Phoenix adapter
# ============================================================

class PhoenixAgentAdapter:
    """
    Adapter around Arize Phoenix agent metrics.

    The runner core stays stable:

        AgentEvalRecord -> AgentPerCaseScore

    MVP evaluation model:

    - tool_selection:
        Phoenix ToolSelectionEvaluator.
        Checks whether the agent selected the right tool.

    - tool_invocation:
        Phoenix ToolInvocationEvaluator.
        Checks whether selected tools were invoked with correct arguments.

    - tool_execution:
        Deterministic runtime check based on status/error.

    tool_response_handling is intentionally skipped in MVP.
    """

    def __init__(self, model: str, llm: LLM) -> None:
        self._model = model
        self._llm = llm
        self._tool_selection_evaluator = None
        self._tool_invocation_evaluator = None

    # ------------------------------------------------------------
    # Phoenix lazy imports
    # ------------------------------------------------------------

    def _get_llm(self):
        return self._llm

    def _get_tool_selection_evaluator(self):
        if self._tool_selection_evaluator is None:
            self._tool_selection_evaluator = ToolSelectionEvaluator(
                llm=self._get_llm()
            )

        return self._tool_selection_evaluator

    def _get_tool_invocation_evaluator(self):
        if self._tool_invocation_evaluator is None:
            self._tool_invocation_evaluator = ToolInvocationEvaluator(
                llm=self._get_llm()
            )

        return self._tool_invocation_evaluator

    # ------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------

    @staticmethod
    def _tool_name(tool: AgentTool | AgentToolCall) -> str:
        return (tool.name or "").strip()

    @staticmethod
    def _format_tool_call(tool: AgentTool | AgentToolCall) -> str:
        name = (tool.name or "").strip()

        if not name:
            return "<unknown_tool>"

        arguments = getattr(tool, "arguments", None) or {}

        if not arguments:
            return name

        args = ", ".join(
            f'{key}="{value}"'
            for key, value in arguments.items()
        )

        return f"{name}({args})"

    @classmethod
    def _format_tool_selection(
        cls,
        tools: List[AgentTool | AgentToolCall],
    ) -> str:
        calls = [
            cls._format_tool_call(tool)
            for tool in tools
            if cls._tool_name(tool)
        ]

        return "\n".join(calls) if calls else "No tools selected"
    
    @classmethod
    def _format_available_tools_for_selection(
        cls,
        record: AgentEvalRecord,
    ) -> str:
      """
      Phoenix ToolSelectionEvaluator should see only tool names/descriptions.
    
      It must not see expected argument values, otherwise argument errors
      can incorrectly affect tool_selection.
      """

      tool_names: Dict[str, str] = {}
    
      for tool in record.expected_tools:
          name = cls._tool_name(tool)
    
          if name:
              tool_names[name] = f"{name}: Available tool for this task."
    
      # Do not add forbiddenTools here.
      # Forbidden tools are policy constraints, not normal available tools.
      # They are handled by count_policy_violations(record).
    
      for tool in record.tools_called:
          name = cls._tool_name(tool)
    
          if name and name not in tool_names:
              tool_names[name] = f"{name}: Tool observed in agent trace."
    
      if not tool_names:
          return "No tools available."
    
      return "\n".join(tool_names.values())
    @staticmethod
    def _infer_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"

        if isinstance(value, int) and not isinstance(value, bool):
            return "integer"

        if isinstance(value, float):
            return "number"

        if isinstance(value, list):
            return "array"

        if isinstance(value, dict):
            return "object"

        return "string"
    
    @classmethod
    def _format_tool_names_only(
      cls,
      tools: List[AgentTool | AgentToolCall],
    ) -> str:
      """
      Used only for ToolSelectionEvaluator.
    
      Arguments are intentionally omitted.
      """
    
      names = [
          cls._tool_name(tool)
          for tool in tools
          if cls._tool_name(tool)
      ]
    
      return "\n".join(names) if names else "No tools selected"
    
    @classmethod
    def _format_available_tools_for_invocation(
      cls,
      record: AgentEvalRecord,
    ) -> str:
      """
      Phoenix ToolInvocationEvaluator needs argument-level contract.
    
      Since the current dataset has no separate tool schema, expectedTools.arguments
      are used as the MVP argument contract.
      """
    
      lines: List[str] = []
    
      for tool in record.expected_tools:
          name = cls._tool_name(tool)
    
          if not name:
              continue
    
          arguments = tool.arguments or {}
    
          if not arguments:
              lines.append(f"{name}: Available tool.")
              continue
    
          arg_lines: List[str] = []
    
          for arg_name, arg_value in arguments.items():
              arg_type = cls._infer_type(arg_value)
    
              arg_lines.append(
                  f'  - {arg_name} (required, {arg_type}): '
                  f'expected value "{arg_value}"'
              )
    
          lines.append(
              f"{name}: Available tool expected for this task.\n"
              + "\n".join(arg_lines)
          )
    
      if not lines:
          return "No tools available."
    
      return "\n".join(lines)

    @classmethod
    def _format_available_tools(cls, record: AgentEvalRecord) -> str:
        """
        Builds a Phoenix-compatible human-readable tool list
        from the existing dataset and actual trace.

        No dataset format change is required.

        Important:
        - expected_tools provides expected argument names/values;
        - forbidden_tools are included as available tool names but marked as forbidden;
        - actual tools are included if they were not present in expected/forbidden
          so Phoenix can identify hallucinated or unknown selections.
        """

        tools_by_name: Dict[str, Dict[str, Any]] = {}

        for tool in record.expected_tools:
            name = cls._tool_name(tool)

            if not name:
                continue

            tools_by_name[name] = {
                "name": name,
                "arguments": tool.arguments or {},
                "kind": "expected",
            }

        for tool in record.forbidden_tools:
            name = cls._tool_name(tool)

            if not name:
                continue

            tools_by_name.setdefault(
                name,
                {
                    "name": name,
                    "arguments": tool.arguments or {},
                    "kind": "forbidden",
                },
            )

            tools_by_name[name]["kind"] = "forbidden"

        for tool in record.tools_called:
            name = cls._tool_name(tool)

            if not name:
                continue

            tools_by_name.setdefault(
                name,
                {
                    "name": name,
                    "arguments": tool.arguments or {},
                    "kind": "observed",
                },
            )

        if not tools_by_name:
            return "No tools available."

        lines: List[str] = []

        for item in tools_by_name.values():
            name = item["name"]
            arguments = item.get("arguments") or {}
            kind = item.get("kind")

            if kind == "forbidden":
                prefix = f"{name}: Forbidden tool. Must not be used."
            elif kind == "expected":
                prefix = f"{name}: Available tool expected for this task."
            else:
                prefix = f"{name}: Observed tool from agent trace."

            if not arguments:
                lines.append(prefix)
                continue

            arg_lines = []

            for arg_name, arg_value in arguments.items():
                arg_type = cls._infer_type(arg_value)

                arg_lines.append(
                    f'  - {arg_name} (required, {arg_type}): expected value "{arg_value}"'
                )

            lines.append(prefix + "\n" + "\n".join(arg_lines))

        return "\n".join(lines)

    @staticmethod
    def _first_score_value(scores: Any) -> Optional[float]:
        if not scores:
            return None

        first = scores[0]

        return safe_float(getattr(first, "score", None))

    @staticmethod
    def _first_score_label(scores: Any) -> Optional[str]:
        if not scores:
            return None

        first = scores[0]

        label = getattr(first, "label", None)

        return str(label) if label is not None else None

    @staticmethod
    def _first_score_explanation(scores: Any) -> Optional[str]:
        if not scores:
            return None

        first = scores[0]

        explanation = getattr(first, "explanation", None)

        return str(explanation) if explanation else None

    # ------------------------------------------------------------
    # Metric 1: Phoenix Tool Selection
    # ------------------------------------------------------------

    def _score_tool_selection(
        self,
        record: AgentEvalRecord,
        score: AgentPerCaseScore,
    ) -> Optional[float]:
        if not record.expected_tools and not record.forbidden_tools:
            return None

        evaluator = self._get_tool_selection_evaluator()

        eval_input = {
          "input": f"User: {record.input}",
          "available_tools": self._format_available_tools_for_selection(record),
          "tool_selection": self._format_tool_names_only(record.tools_called),
        }

        scores = evaluator.evaluate(eval_input)

        value = self._first_score_value(scores)
        label = self._first_score_label(scores)
        explanation = self._first_score_explanation(scores)

        if explanation:
            score.notes.append(
                f"phoenix_tool_selection_explanation={explanation}"
            )

        return value

    # ------------------------------------------------------------
    # Metric 2: Phoenix Tool Invocation
    # ------------------------------------------------------------

    def _score_tool_invocation(
        self,
        record: AgentEvalRecord,
        score: AgentPerCaseScore,
    ) -> Optional[float]:
        if not record.tools_called:
            return None

        evaluator = self._get_tool_invocation_evaluator()

        # eval_input = {
        #     "input": f"User: {record.input}",
        #     "available_tools": self._format_available_tools(record),
        #     "tool_selection": self._format_tool_selection(record.tools_called),
        # }
        eval_input = {
          "input": f"User: {record.input}",
          "available_tools": self._format_available_tools_for_invocation(record),
          "tool_selection": self._format_tool_selection(record.tools_called),
        }
        
        scores = evaluator.evaluate(eval_input)

        value = self._first_score_value(scores)
        label = self._first_score_label(scores)
        explanation = self._first_score_explanation(scores)

        if explanation:
            score.notes.append(
                f"phoenix_tool_invocation_explanation={explanation}"
            )

        return value

    # ------------------------------------------------------------
    # Metric 3: Tool Execution
    # ------------------------------------------------------------

    @staticmethod
    def _score_tool_execution(record: AgentEvalRecord) -> Optional[float]:
        """
        Runtime execution is not an LLM judge task.

        It is evaluated deterministically from the current agent output:

        - status in OK/SUCCESS/SUCCEEDED
        - error is empty
        """

        executable_calls = [
            tool for tool in record.tools_called
            if (tool.name or "").strip()
        ]

        if not executable_calls:
            return None

        successful = 0

        for tool in executable_calls:
            status = (tool.status or "").strip().upper()
            error = tool.error

            if status in {"OK", "SUCCESS", "SUCCEEDED"} and not error:
                successful += 1

        return successful / len(executable_calls)

    # ------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------

    @staticmethod
    def _append_tool_execution_notes(
        record: AgentEvalRecord,
        score: AgentPerCaseScore,
    ) -> None:
        for tool in record.tools_called:
            score.notes.append(
                "tool_execution="
                f"name={tool.name},"
                f"status={tool.status},"
                f"error={tool.error},"
                f"durationMs={tool.duration_ms}"
            )

    @staticmethod
    def _compute_passed(score: AgentPerCaseScore) -> bool:
        metric_values = [
            score.tool_selection,
            score.tool_invocation,
            score.tool_execution,
        ]

        available_metric_values = [
            value for value in metric_values
            if value is not None
        ]

        if available_metric_values:
            return (
                all(value >= 0.5 for value in available_metric_values)
                and score.policy_violations == 0
            )

        return score.policy_violations == 0

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def score_record(self, record: AgentEvalRecord) -> AgentPerCaseScore:
        score = AgentPerCaseScore(
            case_id=record.case_id,
            latency_ms=safe_float(record.metadata.get("latencyMs")),
        )

        try:
            # --------------------------------------------------------
            # 1. Phoenix Tool Selection
            # --------------------------------------------------------
            score.tool_selection = self._score_tool_selection(record, score)

            # --------------------------------------------------------
            # 2. Phoenix Tool Invocation
            # --------------------------------------------------------
            score.tool_invocation = self._score_tool_invocation(record, score)


        except Exception as exc:
            score.notes.append(f"phoenix_evals_failed={repr(exc)}")

            score.tool_selection = None
            score.tool_invocation = None
            score.tool_correctness = None
            score.argument_correctness = None

        # ------------------------------------------------------------
        # 3. Deterministic Tool Execution
        # ------------------------------------------------------------
        score.tool_execution = self._score_tool_execution(record)

        self._append_tool_execution_notes(record, score)

        # ------------------------------------------------------------
        # 4. Policy violations
        # ------------------------------------------------------------
        score.policy_violations = count_policy_violations(record)

        # ------------------------------------------------------------
        # 5. Task completion intentionally skipped in MVP
        # ------------------------------------------------------------
        score.task_completion = None
        score.notes.append("task_completion_skipped=mvp_no_tool_response_handling")

        # ------------------------------------------------------------
        # 6. Final pass/fail
        # ------------------------------------------------------------
        score.passed = self._compute_passed(score)

        return score

    def score_records(
        self,
        records: List[AgentEvalRecord],
    ) -> Tuple[List[AgentPerCaseScore], Dict[str, Any]]:
        scores = [self.score_record(record) for record in records]

        return scores, {
            "metricNames": [
                "tool_selection",
                "tool_invocation",
                "tool_execution",
                "policy_violations",
            ]
        }


# ============================================================
# Deterministic agent checks
# ============================================================

def compute_argument_correctness(record: AgentEvalRecord) -> Optional:
    """
    Deterministic MVP check.

    For each expected tool with expected arguments:
    - find matching called tool by name
    - compare expected argument values
    """

    expected_with_args = [
        tool for tool in record.expected_tools
        if tool.arguments
    ]

    if not expected_with_args:
        return None

    calls_by_name: Dict[str, List[AgentToolCall]] = {}
    for call in record.tools_called:
        calls_by_name.setdefault(tool_key(call), []).append(call)

    total = 0
    matched = 0

    for expected in expected_with_args:
        candidates = calls_by_name.get(tool_key(expected), [])

        if not candidates:
            total += len(expected.arguments)
            continue

        actual = candidates[0]

        for arg_name, expected_value in expected.arguments.items():
            total += 1
            actual_value = actual.arguments.get(arg_name)

            if actual_value == expected_value:
                matched += 1

    if total == 0:
        return None

    return matched / total


def count_policy_violations(record: AgentEvalRecord) -> int:
    forbidden = {tool_key(tool) for tool in record.forbidden_tools}
    if not forbidden:
        return 0

    count = 0

    for call in record.tools_called:
        if tool_key(call) in forbidden:
            count += 1

    return count


# ============================================================
# Aggregation / analyzer
# ============================================================

def aggregate_scores(per_case: List[AgentPerCaseScore]) -> AgentAggregatedScores:
    return AgentAggregatedScores(
        total_cases=len(per_case),
        passed_cases=sum(1 for x in per_case if x.passed),

        tool_selection_avg=mean([x.tool_selection for x in per_case]),
        tool_invocation_avg=mean([x.tool_invocation for x in per_case]),
        tool_execution_avg=mean([x.tool_execution for x in per_case]),

        policy_violations_total=sum(x.policy_violations for x in per_case),
        latency_ms_avg=mean([x.latency_ms for x in per_case]),
    )


def mark_per_case_pass_fail(cfg: RunnerConfig, per_case: List[AgentPerCaseScore]) -> None:
    for score in per_case:
        notes: List[str] = []
        passed = True
        print(f"cfg.thresholds.min_tool_selection={cfg.thresholds.min_tool_selection}, score.tool_selection={score.tool_selection}")
        print(f"cfg.thresholds.min_tool_invocation={cfg.thresholds.min_tool_invocation}, score.tool_invocation={score.tool_invocation}")
        print(f"cfg.thresholds.min_tool_execution={cfg.thresholds.min_tool_execution}, score.tool_execution={score.tool_execution}")
        print(f"cfg.thresholds.max_policy_violations={cfg.thresholds.max_policy_violations}, score.policy_violations={score.policy_violations}")
        
        if cfg.thresholds.min_tool_selection is not None:
          if score.tool_selection is not None:
            if score.tool_selection < cfg.thresholds.min_tool_selection:
                passed = False
                notes.append(
                    f"tool_selection={score.tool_selection:.4f} < minToolSelection={cfg.thresholds.min_tool_selection:.4f}"
                )
          else:
            passed = False
            notes.append(
              f"min_tool_selection thresholds was assigned, but assessment score was not received from the evaluator"
            )

        if cfg.thresholds.min_tool_invocation is not None: 
          if score.tool_invocation is not None:
            if score.tool_invocation < cfg.thresholds.min_tool_invocation:
                passed = False
                notes.append(
                    f"tool_invocation={score.tool_invocation:.4f} < minToolInvocation={cfg.thresholds.min_tool_invocation:.4f}"
                )
          else:
            passed = False
            notes.append(
              f"min_tool_invocation thresholds was assigned, but assessment score was not received from the evaluator"
            )
            
        if cfg.thresholds.min_tool_execution is not None: 
          if score.tool_execution is not None:
            if score.tool_execution < cfg.thresholds.min_tool_execution:
                passed = False
                notes.append(
                    f"tool_execution={score.tool_execution:.4f} < minToolExecution={cfg.thresholds.min_tool_execution:.4f}"
                )
          else:
            passed = False
            notes.append(
              f"min_tool_execution thresholds was assigned, but assessment score was not received from the evaluator"
            )
            
        if cfg.thresholds.max_policy_violations is not None:
          if score.policy_violations is not None:
            if score.policy_violations > cfg.thresholds.max_policy_violations:
                passed = False
                notes.append(
                    f"policy_violations={score.policy_violations} > maxPolicyViolations={cfg.thresholds.max_policy_violations}"
                )
          else:
            passed = False
            notes.append(
              f"policy_violations thresholds was assigned, but assessment score was not received from the evaluator"
            )

        score.passed = passed
        score.notes.extend(notes)


def apply_threshold_policy(
    cfg: RunnerConfig,
    agg: AgentAggregatedScores,
) -> Tuple[bool, Dict[str, Any], List[str]]:
    failed = False
    notes: List[str] = []
    root_cause = {"stage": "None"}

    if cfg.thresholds.max_policy_violations is not None:
        if agg.policy_violations_total > cfg.thresholds.max_policy_violations:
            failed = True
            root_cause = {
                "stage": "Policy",
                "metric": "policy_violations",
                "details": f"policy_violations_total={agg.policy_violations_total} > maxPolicyViolations={cfg.thresholds.max_policy_violations}",
            }
            notes.append(root_cause["details"])

    if not failed and cfg.thresholds.min_tool_selection is not None and agg.tool_selection_avg is not None:
        if agg.tool_selection_avg < cfg.thresholds.min_tool_selection:
            failed = True
            root_cause = {
                "stage": "Tool Selection",
                "metric": "tool_selection",
                "details": f"tool_selection_avg={agg.tool_selection_avg:.4f} < minToolSelection={cfg.thresholds.min_tool_selection:.4f}",
            }
            notes.append(root_cause["details"])

    if not failed and cfg.thresholds.min_tool_execution is not None and agg.tool_execution_avg is not None:
        if agg.tool_execution_avg < cfg.thresholds.min_tool_execution:
            failed = True
            root_cause = {
                "stage": "Tool Execution",
                "metric": "tool_execution",
                "details": f"tool_execution_avg={agg.tool_execution_avg:.4f} < minToolExecution={cfg.thresholds.min_tool_execution:.4f}",
            }
            notes.append(root_cause["details"])

    if not failed and cfg.thresholds.min_tool_invocation is not None and agg.tool_invocation_avg is not None:
        if agg.tool_invocation_avg < cfg.thresholds.min_tool_invocation:
            failed = True
            root_cause = {
                "stage": "Tool Invocation",
                "metric": "tool_invocation",
                "details": f"tool_invocation_avg={agg.tool_invocation_avg:.4f} < minToolInvocation={cfg.thresholds.min_tool_invocation:.4f}",
            }
            notes.append(root_cause["details"])

    return failed, root_cause, notes


def apply_regression_policy(
    cfg: RunnerConfig,
    agg: AgentAggregatedScores,
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
        "tool_correctness_avg": agg.tool_correctness_avg,
        "task_completion_avg": agg.task_completion_avg,
        "argument_correctness_avg": agg.argument_correctness_avg,
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
    agg: AgentAggregatedScores,
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
        schemaVersion="evalrun.gate_summary/v1",
        evaluationType="agent",
        verdict=verdict,
        summary="; ".join(summary_parts),
        rootCause=root_cause,
        artifacts=artifacts,
    )


# ============================================================
# Publishing
# ============================================================

def publish_details_jsonl(
    cfg: RunnerConfig,
    records: List[AgentEvalRecord],
    scores: List[AgentPerCaseScore],
) -> None:
    print("--publishing details jsonl--")
    score_by_case = {x.case_id: x for x in scores}
    lines: List[str] = []

    for record in records:
        score = score_by_case.get(record.case_id)

        row = {
            "schemaVersion": "evalrun.agent_details/v1",
            "caseId": record.case_id,
            "input": record.input,
            "expectedOutput": record.expected_output,
            "actualOutput": record.actual_output,
            "expectedTools": [asdict(x) for x in record.expected_tools],
            "forbiddenTools": [asdict(x) for x in record.forbidden_tools],
            "toolsCalled": [asdict(x) for x in record.tools_called],
            "trace": record.trace,
            "metadata": record.metadata,
            "score": asdict(score) if score else None,
        }

        lines.append(json.dumps(row, ensure_ascii=False))

    payload = "\n".join(lines) + "\n"
    upload_text(cfg, cfg.details_key, payload, content_type="application/x-ndjson")


def publish_scores_json(
    cfg: RunnerConfig,
    agg: AgentAggregatedScores,
    regression: RegressionResult,
    per_case: List[AgentPerCaseScore],
    deepeval_meta: Dict[str, Any],
) -> None:
    print("--publishing scores jsonl--")
    payload = {
        "schemaVersion": "evalrun.scores/v1",
        "evaluationType": "agent",
        "run": {
            "evalrun": cfg.evalrun_name,
            "evalsuite": cfg.evalsuite_name,
            "gitSha": cfg.git_sha,
            "prNumber": cfg.pr_number,
        },
        "engine": {
            "name": "deepeval-agent",
            "metricNames": deepeval_meta.get("metricNames", []),
        },
        "aggregatedScores": asdict(agg),
        "regression": asdict(regression),
        "perCase": [asdict(x) for x in per_case],
    }

    upload_json(cfg, cfg.scores_key, payload)


def publish_gate_summary(cfg: RunnerConfig, gate: GateSummary) -> None:
    print("--publishing gate summary--")
    upload_json(cfg, cfg.gate_summary_key, asdict(gate))


# ============================================================
# Main
# ============================================================

async def main() -> int:
    try:
        cfg = build_runner_config()
        ensure_dir(cfg.work_dir)

        dataset_path = download_dataset(cfg)

        cases = parse_dataset_jsonl(dataset_path)
        if not cases:
            raise RuntimeError("dataset is empty")

        baseline = load_optional_baseline(cfg)

        records: List[AgentEvalRecord] = []
        for case in cases:
            record = call_agent_endpoint(cfg, case)
            records.append(record)
        
        adapter = PhoenixAgentAdapter(model=cfg.llm_model, llm=base_llm)
        per_case_scores, deepeval_meta = adapter.score_records(records)

        mark_per_case_pass_fail(cfg, per_case_scores)

        agg = aggregate_scores(per_case_scores)

        threshold_failed, threshold_root_cause, threshold_notes = apply_threshold_policy(cfg, agg)

        regression = apply_regression_policy(cfg, agg, baseline)

        gate = build_gate_summary(
            cfg=cfg,
            agg=agg,
            threshold_failed=threshold_failed,
            threshold_root_cause=threshold_root_cause,
            threshold_notes=threshold_notes,
            regression=regression,
        )

        publish_details_jsonl(cfg, records, per_case_scores)
        publish_scores_json(cfg, agg, regression, per_case_scores, deepeval_meta)
        publish_gate_summary(cfg, gate)

        # Technical execution completed successfully.
        # Quality failure is represented by gate.verdict=FAIL, not by process exit code.
        return 0

    except Exception as e:
        try:
            cfg = build_runner_config()

            gate = GateSummary(
                schemaVersion="evalrun.gate_summary/v1",
                evaluationType="agent",
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

        print("agent runner failed:", str(e), file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))