import json

from multiturn.artifacts import REDACTED, ArtifactLayout, is_sensitive_key, redact, redact_conversation, state_transitions
from multiturn.models import Conversation, ToolCall, Turn


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
