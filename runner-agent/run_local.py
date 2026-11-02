#!/usr/bin/env python3
"""Run a multi-turn evaluation locally, without Kubernetes or an object store.

Artifacts are written to a directory instead of the object-store gateway; everything else (dataset
parsing, session-aware target calls, deterministic checks, optional DeepEval, gate) is the code the
Job runs.

    python tests/fake_travel_agent.py --port 8080 &          # or point --target at your own agent
    python run_local.py --dataset ../examples/dataset-agent-multiturn.jsonl \
        --suite ../examples/multiturn-local/suite.json --target http://127.0.0.1:8080 --no-semantic

Exit code: 0 for PASS/WARN, 1 for FAIL (the Kubernetes Job always exits 0 and reports the verdict in
gate_summary.json instead).
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multiturn.config import PROVIDER_BUILTIN  # noqa: E402
from multiturn.entrypoint import run_multiturn_job  # noqa: E402


class DirectoryUploader:
    def __init__(self, root: str):
        self.root = root

    def _path(self, key: str) -> str:
        path = os.path.join(self.root, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def put_json(self, key, payload):
        with open(self._path(key), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return {}

    def put_text(self, key, text, content_type="text/plain"):
        with open(self._path(key), "w", encoding="utf-8") as fh:
            fh.write(text)
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="multi-turn JSONL dataset")
    ap.add_argument("--suite", required=True, help="rendered suite.json (see examples/multiturn-local/suite.json)")
    ap.add_argument("--target", required=True, help="agent endpoint implementing the session turn contract")
    ap.add_argument("--out", default=None, help="artifact directory (default: a temp dir)")
    ap.add_argument("--run-name", default="local-run")
    ap.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5-mini"), help="default judge model")
    ap.add_argument("--no-semantic", action="store_true", help="skip optional semantic (DeepEval) evaluators")
    args = ap.parse_args()

    with open(args.suite, encoding="utf-8") as fh:
        suite = json.load(fh)
    if args.no_semantic:
        suite["evaluators"] = [e for e in suite.get("evaluators", []) if e.get("provider", PROVIDER_BUILTIN) == PROVIDER_BUILTIN]

    out = args.out or tempfile.mkdtemp(prefix="evalrun-local-")
    prefix = f"runs/{args.run_name}"
    env = {"EVALRUN_NAME": args.run_name, "EVALSUITE_NAME": "local-suite", "S3_RUN_PREFIX": prefix}
    result = run_multiturn_job(
        suite=suite, env=env, dataset_path=args.dataset, bucket="local",
        gate_summary_key=f"{prefix}/gate_summary.json", scores_key=f"{prefix}/scores.json",
        details_key=f"{prefix}/details.jsonl", target_endpoint=args.target,
        uploader=DirectoryUploader(out), llm_model=args.model,
    )
    print(f"\nverdict : {result.verdict}\nsummary : {result.summary}")
    if result.root_cause.get("stage") != "None":
        print("rootCause:", json.dumps(result.root_cause, indent=2))
    print(f"artifacts: {os.path.join(out, prefix)}")
    return 1 if result.verdict == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
