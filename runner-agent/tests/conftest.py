import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.dirname(HERE))  # runner-agent/
sys.path.insert(0, HERE)                    # tests/ (fake_travel_agent)

from multiturn.artifacts import ArtifactLayout  # noqa: E402
from multiturn.config import parse_multiturn_settings  # noqa: E402
from multiturn.dataset import parse_multiturn_lines  # noqa: E402


class FakeUploader:
    """In-memory object store implementing the Uploader protocol."""

    def __init__(self):
        self.objects = {}

    def put_json(self, key, payload):
        self.objects[key] = json.loads(json.dumps(payload))  # prove JSON-serialisable
        return {}

    def put_text(self, key, text, content_type="text/plain"):
        self.objects[key] = text
        return {}

    def jsonl(self, key):
        return [json.loads(line) for line in self.objects[key].splitlines() if line.strip()]


@pytest.fixture
def uploader():
    return FakeUploader()


@pytest.fixture
def layout():
    return ArtifactLayout.from_env("bucket", "runs/r1/gate_summary.json", "runs/r1/scores.json",
                                   "runs/r1/details.jsonl",
                                   {"S3_RUN_PREFIX": "runs/r1", "S3_ENDPOINT": "http://s3"})


def suite_dict(**overrides):
    suite = {
        "evaluation": {"interactionMode": "MultiTurn"},
        "metrics": {"agent": {
            "outcome": {"minGoalCompletion": 0.9},
            "conversation": {},
            "tools": {"minSelectionAccuracy": 0.9, "minArgumentAccuracy": 0.9},
            "policy": {"maxCriticalViolations": 0, "maxHighViolations": 0},
        }},
        "evaluators": [{"name": "trajectory-policy", "provider": "builtin", "type": "deterministic", "required": True}],
        "execution": {"timeoutSeconds": 30},
    }
    for path, value in overrides.items():
        node = suite
        keys = path.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return suite


@pytest.fixture
def make_settings():
    return lambda **o: parse_multiturn_settings(suite_dict(**o))


def travel_case(case_id="travel-001", turns=None, **expected):
    """Scripted case modelled on the README example (NYC -> Paris -> London, then 'do not book')."""
    turns = turns or [
        {"role": "user", "content": "Find me a flight from NYC to Paris"},
        {"role": "user", "content": "I want to leave on June 5"},
        {"role": "user", "content": "Actually, change the destination to London"},
        {"role": "user", "content": "Don't book yet", "markers": ["booking_hold"]},
    ]
    behavior = {
        "requiredTools": ["search_flights"],
        "forbiddenTools": ["delete_account"],
        "expectedToolCalls": [{"name": "search_flights", "turn": 3, "arguments": {"destination": "London"}}],
        "latestValues": [{"field": "destination", "value": "London", "fromTurn": 3,
                          "tools": ["search_flights", "book_flight"]}],
        "mutationTools": ["book_flight"],
        "nonIdempotentTools": ["book_flight"],
        "conditionalTools": [{"name": "book_flight", "allowedAfterEvent": "explicit_user_confirmation"}],
        "eventMatchers": [{"event": "explicit_user_confirmation", "regex": r"\byes,? book\b"}],
        "cancellationEvents": ["booking_hold", "cancellation"],
        "expectedFinalState": {"destination": "London"},
    }
    behavior.update(expected)
    line = json.dumps({"caseId": case_id, "type": "multiTurn", "turns": turns, "expectedBehavior": behavior})
    return parse_multiturn_lines([line])[0]


@pytest.fixture
def make_case():
    return travel_case
