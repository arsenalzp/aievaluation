"""Deterministic in-memory travel agent used by the multi-turn tests and examples.

It keeps per-session memory (the *agent* owns the state, the runner never repairs it) and can be
told to misbehave through ``flags`` so every policy scenario is reproducible without any LLM,
Phoenix, DeepEval or object storage.

Use it in-process (``FakeTravelAgent().post``) or as a tiny HTTP server::

    python tests/fake_travel_agent.py --port 8080 [--flags stale_destination,auto_book]
"""
from __future__ import annotations

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterable, List, Optional

KNOWN_FLAGS = {
    "stale_destination",  # keeps searching the first destination after the user changed it
    "auto_book",          # books right after searching, without any user confirmation
    "book_after_cancel",  # books even though the user said "don't book" / "cancel"
    "duplicate_booking",  # calls book_flight twice for one confirmation
    "forbidden_tool",     # calls delete_account
    "forget_state",       # reports an empty state (loses context)
}


class FakeResponse:
    def __init__(self, payload: Dict[str, Any], status: int = 200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Dict[str, Any]:
        return self._payload


class FakeTravelAgent:
    def __init__(self, flags: Iterable[str] = (), fail_on_turn: Optional[int] = None):
        unknown = set(flags) - KNOWN_FLAGS
        if unknown:
            raise ValueError(f"unknown flags {sorted(unknown)}; known: {sorted(KNOWN_FLAGS)}")
        self.flags = set(flags)
        self.fail_on_turn = fail_on_turn
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.requests: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ transport
    def post(self, url: str, json: Dict[str, Any], headers: Optional[Dict[str, str]] = None,
             timeout: Optional[float] = None) -> FakeResponse:  # noqa: A002 - mirrors requests.post
        self.requests.append(json)
        if self.fail_on_turn is not None and json["turn"] == self.fail_on_turn:
            return FakeResponse({"error": "boom"}, status=500)
        return FakeResponse(self.handle(json))

    # ------------------------------------------------------------------ behaviour
    def handle(self, req: Dict[str, Any]) -> Dict[str, Any]:
        sid, text, turn = req["sessionId"], req["input"], req["turn"]
        st = self.sessions.setdefault(sid, {"origin": None, "destination": None, "first_destination": None,
                                            "date": None, "confirmed": False, "hold": False, "booked": 0,
                                            "last_trace": None})
        low = text.lower()
        calls: List[Dict[str, Any]] = []

        def tool(name: str, args: Dict[str, Any]) -> None:
            calls.append({"callId": f"call-{turn}-{len(calls) + 1}", "name": name, "arguments": args,
                          "status": "OK", "error": None, "durationMs": 12})

        changed = False
        if m := re.search(r"from (\w+) to (\w+)", text, re.I):
            st["origin"], st["destination"] = m.group(1).upper() if len(m.group(1)) <= 3 else m.group(1).title(), m.group(2).title()
            st["first_destination"] = st["first_destination"] or st["destination"]
            changed = True
        if m := re.search(r"(?:change|switch) (?:the )?destination to (\w+)", text, re.I):
            st["destination"] = m.group(1).title()
            changed = True
        if m := re.search(r"\b(?:on|date is) ([A-Z][a-z]+ \d{1,2})", text):
            st["date"] = m.group(1)
            changed = True
        if re.search(r"don't book|do not book|cancel", low):
            st["hold"] = True
        wants_booking = bool(re.search(r"\byes\b.*\bbook\b", low))
        if wants_booking:
            st["confirmed"] = True
            st["hold"] = False

        output = "Understood."
        ready = st["origin"] and st["destination"] and st["date"]
        if changed and not st["date"]:
            output = "Sure - which date do you want to travel?"
        elif changed and ready:
            dest = st["first_destination"] if "stale_destination" in self.flags else st["destination"]
            tool("search_flights", {"origin": st["origin"], "destination": dest, "date": st["date"]})
            output = f"I found flights from {st['origin']} to {dest} on {st['date']}."
            if "auto_book" in self.flags:
                tool("book_flight", {"origin": st["origin"], "destination": dest, "date": st["date"]})
                st["booked"] += 1
        if wants_booking and ready:
            args = {"origin": st["origin"], "destination": st["destination"], "date": st["date"]}
            tool("book_flight", args)
            if "duplicate_booking" in self.flags:
                tool("book_flight", dict(reversed(list(args.items()))))  # same call, different key order
            st["booked"] += 1
            output = "Your flight is booked."
        elif st["hold"] and "book_after_cancel" in self.flags and ready:
            tool("book_flight", {"origin": st["origin"], "destination": st["destination"], "date": st["date"]})
            st["booked"] += 1
            output = "I booked it anyway."
        elif st["hold"]:
            output = "Okay, I will not book anything yet."
        if "forbidden_tool" in self.flags and turn == 1:
            tool("delete_account", {"user": "demo"})

        prev, trace_id = st["last_trace"], f"trace-{sid}-{turn:03d}"
        st["last_trace"] = trace_id
        state = {} if "forget_state" in self.flags else {
            "origin": st["origin"], "destination": st["destination"], "date": st["date"],
            "confirmed": st["confirmed"], "booked": st["booked"] > 0}
        return {"sessionId": sid, "output": output, "state": state,
                "trace": {"traceId": trace_id, "parentTraceId": prev, "toolCalls": calls},
                "metadata": {"latencyMs": 5}}


def serve(agent: FakeTravelAgent, port: int) -> HTTPServer:  # pragma: no cover - manual use
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = json.dumps(agent.handle(body)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_: Any) -> None:
            return

    return HTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--flags", default="", help=f"comma separated: {sorted(KNOWN_FLAGS)}")
    a = ap.parse_args()
    srv = serve(FakeTravelAgent([f for f in a.flags.split(",") if f]), a.port)
    print(f"fake travel agent on http://127.0.0.1:{a.port}")
    srv.serve_forever()
