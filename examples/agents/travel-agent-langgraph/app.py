"""Deterministic single-step travel agent (LangGraph, no LLM) used to demo and test EvalRun agent evaluation.

Every scenario is selected by keywords in the input, so results are repeatable and cost no tokens.
Endpoint: POST /agent/run  (EvalRun single-turn agent contract, see README "Target Endpoint Contract").

Tool outputs: in evaluation mode the trace contains each tool call's ``output`` unless the request
explicitly sets ``trace.includeToolOutputs: false``. EvalRun needs them to judge whether the final answer
is grounded in what the tools returned (task completion).
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Union, TypedDict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph, START, END

FLIGHT_FIXTURES: Dict[tuple[str, str, str], List[Dict[str, Any]]] = {
  ("NYC", "Paris", "2025-03-15"): [
      {"id": "FL123", "price": 450, "origin": "NYC", "destination": "Paris", "date": "2025-03-15"},
      {"id": "FL456", "price": 380, "origin": "NYC", "destination": "Paris", "date": "2025-03-15"},
  ],
  ("Prague", "Berlin", "2025-04-10"): [
      {"id": "PRG-BER-1", "price": 220, "origin": "Prague", "destination": "Berlin", "date": "2025-04-10"},
      {"id": "PRG-BER-2", "price": 180, "origin": "Prague", "destination": "Berlin", "date": "2025-04-10"},
  ],
  ("London", "Tokyo", "2025-05-20"): [
      {"id": "LON-TYO-1", "price": 980, "origin": "London", "destination": "Tokyo", "date": "2025-05-20"},
      {"id": "LON-TYO-2", "price": 870, "origin": "London", "destination": "Tokyo", "date": "2025-05-20"},
  ],
  ("NYC", "Rome", "2025-06-01"): [
      {"id": "NYC-ROM-1", "price": 510, "origin": "NYC", "destination": "Rome", "date": "2025-06-01"},
      {"id": "NYC-ROM-2", "price": 430, "origin": "NYC", "destination": "Rome", "date": "2025-06-01"},
  ],
  ("Madrid", "Oslo", "2025-08-01"): [],
  # Key matches what infer_flight_args() returns ("Noway"), so a *correct* search finds these flights.
  # (Previously keyed ("NOWAY", "NOWAY", ...), which a correct search_flights call could never hit.)
  ("Noway", "Noway", "2025-09-09"): [
      {"id": "NOWAY-NOWAY-1", "price": 221, "origin": "Noway", "destination": "Noway", "date": "2025-09-09"},
      {"id": "NOWAY-NOWAY-2", "price": 183, "origin": "Noway", "destination": "Noway", "date": "2025-09-09"},
  ],
}


# Helper function for selecting a route from user's input
def infer_flight_args(user_input: str) -> Dict[str, str]:
    """
    Deterministic route extraction for EvalRun demo cases.

    This is intentionally simple and predictable.
    It is not meant to be a production NLU parser.
    """

    normalized = user_input.lower()
    if "noway" in normalized:
        return {"origin": "Noway", "destination": "Noway", "date": "2025-09-09"}

    if "prague" in normalized and "berlin" in normalized:
        return {"origin": "Prague", "destination": "Berlin", "date": "2025-04-10"}

    if "london" in normalized and "tokyo" in normalized:
        return {"origin": "London", "destination": "Tokyo", "date": "2025-05-20"}

    if "nyc" in normalized and "rome" in normalized:
        return {"origin": "NYC", "destination": "Rome", "date": "2025-06-01"}

    if "madrid" in normalized and "oslo" in normalized:
        return {"origin": "Madrid", "destination": "Oslo", "date": "2025-08-01"}

    return {"origin": "NYC", "destination": "Paris", "date": "2025-03-15"}


# ============================================================
# API models
# ============================================================


class TraceOptions(BaseModel):
    mode: str = Field(default="inline")
    # None = not specified by the caller: tool outputs are included in evaluation mode.
    includeToolOutputs: Optional[bool] = Field(default=None)
    sanitize: bool = Field(default=True)


class AgentRequest(BaseModel):
    input: str = Field(..., description="User request for the travel agent")

    evaluationMode: bool = Field(
        default=False,
        description="When true, the response includes trace data for evaluation",
    )

    trace: Union[bool, TraceOptions] = Field(
        default=False,
        description="False, true, or trace options object",
    )


class AgentResponse(BaseModel):
    output: str
    trace: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any]


# ============================================================
# LangGraph state
# ============================================================

class AgentState(TypedDict, total=False):
    input: str
    output: str
    tool_calls: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]


# ============================================================
# LangChain tools
# ============================================================

@tool
def search_flights(origin: str, destination: str, date: str) -> List[Dict[str, Any]]:
    """Search available flights for a given origin, destination, and date."""

    return FLIGHT_FIXTURES.get((origin, destination, date), [])


@tool
def book_flight(flight_id: str) -> Dict[str, Any]:
    """Book a flight by flight id. Use only after explicit user confirmation."""

    return {"bookingId": "BK-999", "flightId": flight_id, "status": "booked"}


@tool
def search_policy(topic: str) -> Dict[str, Any]:
    """Search internal travel policy by topic."""

    policies = {
        "booking": "Flights must not be booked without explicit user confirmation.",
        "refund": "Refunds are available within 30 days for eligible tickets.",
    }

    return {"topic": topic, "text": policies.get(topic, "No policy found.")}


TOOLS: List[BaseTool] = [search_flights, book_flight, search_policy]

TOOLS_BY_NAME: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}


# ============================================================
# Graph nodes
# ============================================================
def plan_node(state: AgentState) -> Dict[str, Any]:
    """
    Deterministic planner for EvalRun demo.

    Scenarios:
    - default request            -> correct search_flights call
    - "wrong destination"        -> wrong tool argument
    - "wrong date"               -> wrong date argument
    - "without confirmation"     -> forbidden book_flight call
    - "noway"                    -> wrong tool selection (books instead of searching)
    - "policy and find"          -> two-tool call: search_policy + search_flights
    - "policy" / "confirmation"  -> policy lookup
    - "answer without tools"     -> no tool call
    """

    user_input = state["input"]
    normalized = user_input.lower()
    flight_args = infer_flight_args(user_input)

    steps = list(state.get("steps", []))
    tool_calls: List[Dict[str, Any]] = []

    if "book" in normalized and "without confirmation" in normalized:
        tool_calls = [
            {"name": "search_flights", "arguments": flight_args},
            {"name": "book_flight", "arguments": {"flight_id": "FL456"}},
        ]
        steps.append({"node": "plan", "description": "Planned a forbidden booking action for policy-failure demo.",
                      "scenario": "policy_failure"})

    elif "noway" in normalized:
        tool_calls = [{"name": "book_flight", "arguments": {"flight_id": "NOWAY-NOWAY-2"}}]
        steps.append({"node": "plan", "description": "Planned hidden wrong tool selection for ToolSelection demo.",
                      "scenario": "hidden_tool_selection_failure"})

    elif "policy and find" in normalized or "check booking policy and find" in normalized:
        tool_calls = [
            {"name": "search_policy", "arguments": {"topic": "booking"}},
            {"name": "search_flights", "arguments": flight_args},
        ]
        steps.append({"node": "plan", "description": "Planned policy lookup and flight search for multi-tool demo.",
                      "scenario": "two_tools_policy_and_search"})

    elif "wrong destination" in normalized:
        wrong_args = dict(flight_args)
        wrong_args["destination"] = "Berlin"
        tool_calls = [{"name": "search_flights", "arguments": wrong_args}]
        steps.append({"node": "plan", "description": "Planned search_flights with wrong destination for ToolInvocation demo.",
                      "scenario": "tool_invocation_wrong_destination"})

    elif "wrong date" in normalized:
        wrong_args = dict(flight_args)
        wrong_args["date"] = "2025-03-16"
        tool_calls = [{"name": "search_flights", "arguments": wrong_args}]
        steps.append({"node": "plan", "description": "Planned search_flights with wrong date for ToolInvocation demo.",
                      "scenario": "tool_invocation_wrong_date"})

    elif "policy" in normalized or "confirmation" in normalized:
        tool_calls = [{"name": "search_policy", "arguments": {"topic": "booking"}}]
        steps.append({"node": "plan", "description": "Planned policy lookup before taking action.",
                      "scenario": "policy_lookup"})

    elif "answer without tools" in normalized:
        tool_calls = []
        steps.append({"node": "plan", "description": "Planned no tool calls for outcome-failure demo.",
                      "scenario": "no_tools_failure"})

    else:
        tool_calls = [{"name": "search_flights", "arguments": flight_args}]
        steps.append({"node": "plan", "description": "Planned search_flights with expected arguments.",
                      "scenario": "happy_path"})

    return {"tool_calls": tool_calls, "steps": steps}


def tools_node(state: AgentState) -> Dict[str, Any]:
    """
    Execute planned LangChain tools.

    This node creates normalized tool execution records.
    EvalRun agent runner reads these records as trace.toolCalls.
    """

    tool_results: List[Dict[str, Any]] = []
    steps = list(state.get("steps", []))

    for index, planned_call in enumerate(state.get("tool_calls", []), start=1):
        tool_name = planned_call["name"]
        arguments = planned_call.get("arguments", {})

        started = time.time()
        output = None
        error = None
        status = "OK"

        try:
            selected_tool = TOOLS_BY_NAME[tool_name]
            output = selected_tool.invoke(arguments)

        except Exception as exc:
            error = repr(exc)
            status = "ERROR"

        duration_ms = round((time.time() - started) * 1000, 3)

        tool_results.append({
            "callId": f"call-{index}",
            "name": tool_name,
            "arguments": arguments,
            "output": output,
            "error": error,
            "status": status,
            "durationMs": duration_ms,
        })

    steps.append({
        "node": "tools",
        "description": "Executed planned LangChain tools.",
        "toolNames": [result["name"] for result in tool_results],
    })

    return {"tool_results": tool_results, "steps": steps}


def answer_node(state: AgentState) -> Dict[str, Any]:
    """
    Build final answer from tool results.

    The answer is deterministic to keep EvalRun demo repeatable.
    """

    tool_results = state.get("tool_results", [])
    steps = list(state.get("steps", []))

    def first(name: str) -> Dict[str, Any]:
        return next(result for result in tool_results if result["name"] == name)

    def has(name: str) -> bool:
        return any(result["name"] == name for result in tool_results)

    if not tool_results:
        output = "I cannot determine the cheapest flight without using tools."

    elif has("book_flight"):
        booking = first("book_flight").get("output") or {}
        output = f"Booked cheapest flight: {booking.get('flightId')} with booking id {booking.get('bookingId')}"

    elif has("search_policy") and has("search_flights"):
        policy = first("search_policy").get("output") or {}
        flights = first("search_flights").get("output") or []
        if flights:
            cheapest = min(flights, key=lambda flight: flight["price"])
            output = f"{policy.get('text', 'No policy found.')} Cheapest flight: {cheapest['id']} for ${cheapest['price']}"
        else:
            output = f"{policy.get('text', 'No policy found.')} No flights found."

    elif has("search_policy"):
        policy = first("search_policy").get("output") or {}
        output = policy.get("text", "No policy found.")

    elif has("search_flights"):
        flights = first("search_flights").get("output") or []
        if flights:
            cheapest = min(flights, key=lambda flight: flight["price"])
            output = f"Found cheapest flight: {cheapest['id']} for ${cheapest['price']}"
        else:
            output = "No flights found."

    else:
        output = "No supported tool result was produced."

    steps.append({"node": "answer", "description": "Generated final response from tool results.", "output": output})

    return {"output": output, "steps": steps}


# ============================================================
# Graph definition
# ============================================================

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("plan", plan_node)
    graph.add_node("tools", tools_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "tools")
    graph.add_edge("tools", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


AGENT_GRAPH = build_agent_graph()


# ============================================================
# Trace helpers
# ============================================================

def normalize_trace_options(req: AgentRequest) -> TraceOptions:
    if req.evaluationMode:
        if isinstance(req.trace, TraceOptions):
            return req.trace
        return TraceOptions(mode="inline")

    if req.trace is True:
        return TraceOptions(mode="inline")

    if isinstance(req.trace, TraceOptions):
        return req.trace

    return TraceOptions(mode="none")


def should_include_trace(req: AgentRequest, trace_options: TraceOptions) -> bool:
    if req.evaluationMode or req.trace is True or isinstance(req.trace, TraceOptions):
        return trace_options.mode == "inline"
    return False


def should_include_tool_outputs(req: AgentRequest, trace_options: TraceOptions) -> bool:
    """Explicit request wins; otherwise tool outputs are part of the evaluation trace."""
    if trace_options.includeToolOutputs is not None:
        return trace_options.includeToolOutputs
    return req.evaluationMode


def sanitize_tool_result_for_trace(tool_result: Dict[str, Any], include_outputs: bool) -> Dict[str, Any]:
    """
    Produce trace-safe tool call record.

    EvalRun runner reads:
    - callId, tool name, arguments, duration, status, error  (trajectory checks)
    - output                                                 (task completion: is the answer grounded in tool results)
    """

    sanitized = {
        "callId": tool_result.get("callId"),
        "name": tool_result.get("name"),
        "arguments": tool_result.get("arguments", {}),
        "durationMs": tool_result.get("durationMs"),
        "status": tool_result.get("status", "OK"),
        "error": tool_result.get("error"),
    }

    if include_outputs:
        sanitized["output"] = tool_result.get("output")

    return sanitized


def build_trace_payload(trace_id: str, final_state: AgentState, include_outputs: bool) -> Dict[str, Any]:
    tool_results = final_state.get("tool_results", [])
    steps = final_state.get("steps", [])

    return {
        "traceId": trace_id,
        "source": "inline",
        "framework": "langgraph",
        "toolOutputsIncluded": include_outputs,
        "toolCalls": [sanitize_tool_result_for_trace(result, include_outputs) for result in tool_results],
        "steps": steps,
    }


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(title="EvalRun LangGraph Travel Agent Demo", version="0.5.0")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "agent": "travel-agent-langgraph-demo", "frameworks": ["langchain", "langgraph"]}


@app.post("/agent/run", response_model=AgentResponse)
def run_agent(req: AgentRequest):
    started = time.time()
    trace_id = str(uuid.uuid4())

    trace_options = normalize_trace_options(req)

    initial_state: AgentState = {"input": req.input, "output": "", "tool_calls": [], "tool_results": [], "steps": []}

    final_state = AGENT_GRAPH.invoke(initial_state)

    latency_ms = round((time.time() - started) * 1000, 3)

    include_trace = should_include_trace(req, trace_options)

    trace_payload = None
    if include_trace:
        trace_payload = build_trace_payload(
            trace_id=trace_id,
            final_state=final_state,
            include_outputs=should_include_tool_outputs(req, trace_options),
        )

    return AgentResponse(
        output=final_state["output"],
        trace=trace_payload,
        metadata={
            "traceId": trace_id,
            "latencyMs": latency_ms,
            "agent": "travel-agent-langgraph-demo",
            "mode": "evaluation" if include_trace else "normal",
            "framework": "langgraph",
        },
    )
