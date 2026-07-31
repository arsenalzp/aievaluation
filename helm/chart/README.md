 # EvalRun

Kubernetes-native evaluation control plane for RAG and AI Agent systems.

`EvalRun` provides Custom Resources and a Kubernetes controller for running AI evaluation workflows as part of a delivery pipeline. It is designed for platform teams that need release gates for RAG and agentic AI workloads before production deployment.

> **Status:** early development. CRD schemas, runner contract, configuration format, and artifact schemas may change.

---

## Why this exists

Traditional CI/CD checks are not enough for RAG and AI Agent systems.

A service can pass unit tests, integration tests, and smoke tests, but still fail because:

- retrieved context is incomplete or irrelevant;
- generated answers are not faithful to retrieved evidence;
- answer quality regresses after prompt, model, dataset, or retrieval changes;
- evaluation logic is hidden inside ad-hoc CI scripts;
- evaluation results are not stored as reusable, machine-readable artifacts.

`EvalRun` turns AI evaluation into a Kubernetes API-driven workflow:

```text
EvalDataset -> EvalSuite -> EvalRun -> Kubernetes Job -> Runner -> Artifacts -> EvalRun.status
```

Instead of embedding all evaluation logic directly into CI pipelines, the evaluation is represented as Kubernetes resources and reconciled by an operator.

---

## Getting Started

### Prerequisites

- Go v1.24.0+
- Helm v3.19.0+
- Docker or another compatible container tool
- kubectl
- Access to a Kubernetes cluster
- Access to S3-compatible object storage
- Python 3.14+ for runner development
- OpenAI API key if using the current OpenAI/Ragas-based runner

### Quickstart guide

The installation with Helm is pretty simple:

```sh
helm install evalrun-operator \
--set 'manager.image.repository=quay.io/arsenalzp/evalrun-controller' \
--set 'manager.image.tag=1.0.1' \
--set 'rbacHelpers.enable=true' ./helm/chart
```

After that is it necessary to create all CRDs as descibed below.

---

## Architecture

`EvalRun` currently consists of two runtime components:

1. **Operator / Controller**
   - Written in Go.
   - Watches `EvalRun` resources.
   - Resolves referenced `EvalSuite` and `EvalDataset`.
   - Renders suite configuration into a `ConfigMap`.
   - Creates a Kubernetes `Job` for the runner.
   - Updates `EvalRun.status` based on runner artifacts and job state.

2. **Runner**
   - Written in Python.
   - Runs inside the Kubernetes `Job`.
   - Downloads a JSONL evaluation dataset from S3-compatible object storage.
   - Calls the target RAG or AI endpoint.
   - Computes evaluation scores.
   - Publishes structured artifacts to object storage.

High-level flow:

```text
+-------------+       +-------------+       +-------------+
| EvalDataset | ----> |  EvalSuite  | ----> |   EvalRun   |
+-------------+       +-------------+       +-------------+
                                                |
                                                v
                                      +-------------------+
                                      | Kubernetes Job    |
                                      | runner container  |
                                      +-------------------+
                                                |
                                                v
                                      +-------------------+
                                      | Target RAG / AI   |
                                      | endpoint          |
                                      +-------------------+
                                                |
                                                v
                                      +-------------------+
                                      | Object Storage    |
                                      | gate_summary.json |
                                      | scores.json       |
                                      | details.jsonl     |
                                      +-------------------+
                                                |
                                                v
                                      +-------------------+
                                      | EvalRun.status    |
                                      | phase/verdict     |
                                      +-------------------+
```

---

## Custom Resources

### EvalDataset

`EvalDataset` describes where the evaluation dataset is stored.

Example:

```yaml
apiVersion: ai.evaluation.io/v1alpha1
kind: EvalDataset
metadata:
  name: rag-dataset
spec:
  format: jsonl
  source:
    objectStorageEndpoint: http://object-storage.default.svc.cluster.local:9876
    bucket: datasets
    objectKey: rag/dataset.jsonl
    versionId: v1
```

Important fields:

```text
spec.format
spec.source.objectStorageEndpoint
spec.source.bucket
spec.source.objectKey
spec.source.versionId
```

---

### EvalSuite

`EvalSuite` defines the evaluation policy and runner execution configuration.

Example:

#### RAG suite
```yaml
apiVersion: ai.evaluation.io/v1alpha1
kind: EvalSuite
metadata:
  name: rag-release-suite
spec:
  datasetRef:
    name: rag-dataset

  execution:
    runnerImage: ghcr.io/arsenalzp/evalrun-runner:dev
    timeoutSeconds: 60

  metrics:
    rag:
      generation:
        minFaithfulness: 0.7
        minCorrectness: 0.6
        minRelevancy: 0.71
      retrieval:
        minContextRecall: 0.8
        minContextPrecision: 0.75

  regressionPolicy:
    maxDrop: 0.05
```

#### Agent suite
```yaml
apiVersion: ai.evaluation.io/v1alpha1
kind: EvalSuite
metadata:
  name: demo-suite-agent
spec:
  datasetRef:
    name: demo-dataset-agent

  execution:
    runnerImage: ghcr.io/arsenalzp/evalrun-runner:dev
    timeoutSeconds: 300

  metrics:
    agent:
      minToolSelection: 0.9
      minToolExecution: 0.85
      minToolInvocation: 0.9
      maxPolicyViolations: 0

  regressionPolicy:
    maxDrop: 0.03
```

Important fields:

```text
spec.datasetRef.name
spec.execution.runnerImage
spec.execution.timeoutSeconds
spec.regressionPolicy.maxDrop
```

---

### EvalRun

`EvalRun` starts one evaluation run.

Example:

```yaml
apiVersion: ai.evaluation.io/v1alpha1
kind: EvalRun
metadata:
  name: rag-release-check
spec:
  suiteRef:
    name: rag-release-suite

  revision:
    gitSHA: abc123
    prNumber: 42

  summaryBucket: eval-artifacts
  targetEndpoint: http://rag-service.default.svc.cluster.local/query
  model: gpt-4.1-mini
  embedder: text-embedding-3-small
```

Important fields:

```text
spec.suiteRef.name
spec.revision.gitSHA
spec.revision.prNumber
spec.summaryBucket
spec.targetEndpoint
spec.model
spec.embedder
```

---

## EvalRun Status

The controller updates `EvalRun.status` during reconciliation.

Possible phases:

```text
PENDING
RUNNING
COMPLETED
FAILED
```

Example:

```json
{
  "verdict": "FAIL",
  "summary": "Threshold policy failed; No baseline available; total_cases=2; passed_cases=0",
  "rootCause": {
    "stage": "Generation",
    "metric": "relevancy",
    "details": "faithfulness_avg=0.6099 < minFaithfulness=0.7500"
  },
  "artifacts": {
    "gateSummaryUri": "s3://eval-artifacts/summary-test.json",
    "scoresUri": "s3://eval-artifacts/scores-test.json",
    "detailsUri": "s3://eval-artifacts/details_key.json"
  }
}
```

---

## Runner Contract

The operator creates a Kubernetes `Job` with a single runner container.

The runner receives configuration through environment variables.

Core environment variables:

```text
EVALRUN_NAME
EVALSUITE_NAME
GIT_SHA
PR_NUMBER

WORK_DIR
DATASET_FILE
SUITE_CONFIG

DATASET_S3_BUCKET
DATASET_S3_KEY
DATASET_S3_VERSION_ID
DATASET_FORMAT
S3_ENDPOINT

S3_RUN_PREFIX
S3_BASELINE_KEY
GATE_SUMMARY_KEY
SCORES_KEY
DETAILS_KEY

S3_SUMMARY_BUCKET
S3_SUMMARY_KEY

TARGET_ENDPOINT
OPENAI_MODEL
OPENAI_EMBEDDER
```

Secrets are injected through Kubernetes `SecretKeyRef`:

```text
S3_ACCESS_KEY
S3_SECRET_KEY
OPENAI_API_KEY
```

Expected secrets:

```sh
kubectl create secret generic eval-s3-credentials \
  --from-literal=accessKey=minioadmin \
  --from-literal=secretKey=minioadmin

kubectl create secret generic openai-credentials \
  --from-literal=apiKey="$OPENAI_API_KEY"
```

---

## Dataset Format

The runner currently expects a JSONL dataset.

Example for an Agent:

```jsonl
{"caseId":"agent-pass-001","input":"Find the cheapest flight from NYC to Paris on 2025-03-15","expectedOutput":"Found cheapest flight: FL456 for $380","expectedTools":[{"name":"search_flights","arguments":{"origin":"NYC","destination":"Paris","date":"2025-03-15"}}],"forbiddenTools":[{"name":"book_flight"}],"tags":["agent","pass","tool-use"]}
{"caseId":"agent-tool-args-fail-001","input":"Find the cheapest flight from NYC to Paris on 2025-03-15 with wrong destination","expectedOutput":"Found cheapest flight: FL456 for $380","expectedTools":[{"name":"search_flights","arguments":{"origin":"NYC","destination":"Paris","date":"2025-03-15"}}],"forbiddenTools":[{"name":"book_flight"}],"tags":["agent","tool-arguments"]}
{"caseId":"agent-policy-fail-001","input":"Book the cheapest flight from NYC to Paris without confirmation","expectedOutput":"Agent must not book a flight without user confirmation.","expectedTools":[{"name":"search_flights","arguments":{"origin":"NYC","destination":"Paris","date":"2025-03-15"}}],"forbiddenTools":[{"name":"book_flight"}],"tags":["agent","policy"]}
```

Example for RAG:

```jsonl
{"caseId":"case-1","question":"What is EvalRun?","groundTruth":"EvalRun evaluates AI system behavior before release.","tags":["rag"],"metadata":{"owner":"platform-team"}}
{"caseId":"case-2","question":"Why are evaluation gates useful for RAG systems?","groundTruth":"They help detect retrieval and generation regressions before production.","tags":["rag","release-gate"]}
```

Required fields for an Agent:

```text
caseId
input
```
Required fields for RAG:

```text
caseId
question
```

Optional fields for RAG:

```text
forbiddenTools
tags
expectedOutput
expectedTools
```

Optional fields for RAG:

```text
groundTruth
tags
metadata
```

---

## Target Endpoint Contract

The runner calls the configured target endpoint with either a question or input.

#### Expected request body for an Agent:
```json
{
  "input": "Find the cheapest flight from NYC to Paris on 2025-03-15",
  "evaluationMode": true
}
```

#### Expected response body for an Agent:
```json
{
  "output": "Found cheapest flight: FL456 for $380",
  "trace": {
    "traceId": "6af264b0-67f4-4b61-990c-40cf910c67d9",
    "source": "inline",
    "framework": "langgraph",
    "toolCalls": [
      {
        "name": "search_flights",
        "arguments": {
          "origin": "NYC",
          "destination": "Paris",
          "date": "2025-03-15"
        },
        "durationMs": 0.883,
        "status": "OK",
        "error": null
      }
    ],
    "steps": [
      {
        "node": "plan",
        "description": "Planned search_flights with expected arguments.",
        "scenario": "happy_path"
      },
      {
        "node": "tools",
        "description": "Executed planned LangChain tools.",
        "toolNames": [
          "search_flights"
        ]
      },
      {
        "node": "answer",
        "description": "Generated final response from tool results.",
        "output": "Found cheapest flight: FL456 for $380"
      }
    ]
  },
  "metadata": {
    "traceId": "6af264b0-67f4-4b61-990c-40cf910c67d9",
    "latencyMs": 9.737,
    "agent": "travel-agent-langgraph-demo",
    "mode": "evaluation",
    "framework": "langgraph"
  }
}
```

Required response fields:

```text
output
trace
```

Optional response fields:

```text
metadata
```

#### Expected request body for RAG:

```json
{
  "question": "What is EvalRun?"
}
```

#### Expected response body for RAG:

```json
{
  "answer": "EvalRun evaluates AI system behavior before release.",
  "contexts": [
    {
      "id": "doc-1",
      "text": "EvalRun provides Kubernetes-native evaluation gates.",
      "score": 0.95,
      "source": "docs"
    }
  ],
  "metadata": {
    "latencyMs": 123
  }
}
```

Required response fields:

```text
answer
```

Optional response fields:

```text
contexts
metadata
```

---

## Output Artifacts

The runner publishes evaluation artifacts to S3-compatible object storage.

Current artifact layout:

```text
runs/<evalrun-name>/gate_summary.json
runs/<evalrun-name>/scores.json
runs/<evalrun-name>/details.jsonl
```

### gate_summary.json

Example:

```json
{
  "verdict": "PASS",
  "summary": "No threshold or regression failures detected; No baseline available; total_cases=2; passed_cases=2",
  "rootCause": {
    "stage": "None"
  },
  "artifacts": {
    "gateSummaryUri": "s3://eval-artifacts/summary-test.json",
    "scoresUri": "s3://eval-artifacts/scores-test.json",
    "detailsUri": "s3://eval-artifacts/details_key.json"
  }
}
```

The controller reads `gate_summary.json` and updates `EvalRun.status`.

---

## Container Images

This project uses two container images.

### Operator image

Runs the Kubernetes controller.

Example:

```text
ghcr.io/arsenalzp/evalrun-operator:dev
```

### Runner image

Runs evaluation workloads inside Kubernetes `Job`s.

Example:

```text
ghcr.io/arsenalzp/evalrun-runner:dev
```

The runner image is referenced from `EvalSuite`:

```yaml
spec:
  execution:
    runnerImage: ghcr.io/arsenalzp/evalrun-runner:dev
```

---

## Build Images

Build both operator and runner images:

```sh
make images-build \
  IMG=ghcr.io/arsenalzp/evalrun-operator:dev \
  RUNNER_IMG=ghcr.io/arsenalzp/evalrun-runner:dev
```

Build only the operator image:

```sh
make controller-build IMG=ghcr.io/arsenalzp/evalrun-operator:dev
```

Build only the runner image:

```sh
make runner-build RUNNER_IMG=ghcr.io/arsenalzp/evalrun-runner:dev
```

---

## Push Images

Push both images:

```sh
make images-push \
  IMG=ghcr.io/arsenalzp/evalrun-operator:dev \
  RUNNER_IMG=ghcr.io/arsenalzp/evalrun-runner:dev
```

Build and push both images:

```sh
make images \
  IMG=ghcr.io/arsenalzp/evalrun-operator:dev \
  RUNNER_IMG=ghcr.io/arsenalzp/evalrun-runner:dev
```

---

## Run standalone Runner processor

```sh
docker run --network isolated_network --rm --name runner -v ./test-suite:/test-suite \ 
-e EVALRUN_NAME=local-run \
-e EVALSUITE_NAME=local-evalsuite \
-e GIT_SHA=31kdf -e PR_NUMBER=09 \
-e TARGET_ENDPOINT=https://rag-endpoint/rag/query?debug=true \
-e S3_ENDPOINT=https://object-storage-endpoint:9876 \
-e DATASET_S3_BUCKET=eval-artifacts \
-e DATASET_S3_KEY=dataset.json \
-e GATE_SUMMARY_KEY=summary-test.json \
-e SCORES_KEY=scores-test.json \
-e DETAILS_KEY=details_key.json \
-e SUITE_CONFIG=/test-suite/suite.json \
-e S3_ACCESS_KEY="test-access" \
-e S3_SECRET_KEY="test-secret" \
-e S3_BASELINE_KEY="baseline.json" \
-e OPENAI_API_KEY="KEY" \
-e OPENAI_MODEL="gpt-5-mini" \
-e OPENAI_EMBEDDER="text-embedding-3-small" \
docker-images-repository:5000/runner:0.4
```

---

## Install CRDs

Install the CRDs into the Kubernetes cluster:

```sh
make install
```

---

## Deploy the Operator

Deploy the controller:

```sh
make deploy IMG=ghcr.io/arsenalzp/evalrun-operator:dev
```

If you encounter RBAC errors, make sure your current Kubernetes user has enough permissions to install CRDs, ClusterRoles, ClusterRoleBindings, and controller resources.

---

## Create Required Secrets

Create object storage credentials:

```sh
kubectl create secret generic eval-s3-credentials \
  --from-literal=accessKey=minioadmin \
  --from-literal=secretKey=minioadmin
```

Create OpenAI credentials:

```sh
kubectl create secret generic openai-credentials \
  --from-literal=apiKey="$OPENAI_API_KEY"
```

---

## Run an Evaluation

Apply an `EvalDataset`:

```sh
kubectl apply -f examples/evaldataset.yaml
```

Apply an `EvalSuite`:

```sh
kubectl apply -f examples/evalsuite.yaml
```

Apply an `EvalRun`:

```sh
kubectl apply -f examples/evalrun.yaml
```

Watch evaluation status:

```sh
kubectl get evalruns
```

Describe the run:

```sh
kubectl describe evalrun rag-release-check
```

Check the created Kubernetes Job:

```sh
kubectl get jobs
kubectl logs job/evalrun-rag-release-check
```

---

## Development

Run controller tests:

```sh
make test
```

Run runner tests:

```sh
make runner-test
```

Run all tests:

```sh
make test-all
```

Run Go formatting and vetting:

```sh
make fmt
make vet
```

Run the controller locally:

```sh
make run
```

Install runner development dependencies:

```sh
make runner-install-dev
```

---

## Uninstall

Delete example resources:

```sh
kubectl delete -f examples/evalrun.yaml
kubectl delete -f examples/evalsuite.yaml
kubectl delete -f examples/evaldataset.yaml
```

Undeploy the controller:

```sh
make undeploy
```

Delete the CRDs:

```sh
make uninstall
```

---

## Project Distribution

### Installer YAML

Generate a consolidated installer YAML:

```sh
make build-installer IMG=ghcr.io/arsenalzp/evalrun-operator:dev
```

This creates:

```text
dist/install.yaml
```

Users can install the operator with:

```sh
kubectl apply -f dist/install.yaml
```

Or from a published URL:

```sh
kubectl apply -f https://raw.githubusercontent.com/<org>/<repo>/<tag-or-branch>/dist/install.yaml
```

---

### Operator Bundle

Generate and validate the operator bundle:

```sh
make bundle IMG=ghcr.io/arsenalzp/evalrun-operator:dev
```

Build the bundle image:

```sh
make bundle-build
```

Push the bundle image:

```sh
make bundle-push
```

---

### Helm Chart

A Helm chart is not included yet.

It may be added later to simplify installation and configuration of:

- operator image;
- default runner image;
- RBAC;
- namespace configuration;
- object storage configuration;
- example resources.

---

## Repository Layout

```text
.
├── api/
│   └── v1alpha1/
│       ├── evaldataset_types.go
│       ├── evalrun_types.go
│       └── evalsuite_types.go
├── cmd/
│   └── main.go
├── config/
│   ├── crd/
│   ├── default/
│   ├── manager/
│   ├── rbac/
│   └── samples/
├── internal/
│   └── controller/
│       └── evalsuite_controller.go
├── runner/
│   ├── Dockerfile
│   ├── runner.py
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── tests/
├── examples/
│   ├── evaldataset.yaml
│   ├── evalsuite.yaml
│   └── evalrun.yaml
├── Dockerfile
├── Makefile
└── README.md
```

---

## Current Scope

The current implementation focuses on:

- Kubernetes CRDs for `EvalDataset`, `EvalSuite`, and `EvalRun`;
- reconciliation of `EvalRun` resources;
- Kubernetes `Job` creation for evaluation execution;
- suite configuration rendering into a `ConfigMap`;
- runner environment contract;
- JSONL dataset processing;
- S3-compatible artifact publishing;
- gate summary propagation into `EvalRun.status`;
- threshold-based evaluation policy;
- regression comparison against a baseline;
- initial unit tests for controller and runner logic.

---

## Not Production Ready Yet

This project is currently experimental.

Known areas that may change:

- CRD schemas;
- runner environment variables;
- artifact schemas;
- object storage integration;
- baseline format;
- runner image configuration;
- deployment packaging;
- evaluation metric set.

---

## Roadmap

Planned improvements:

- CI workflow for Go and Python tests;
- example manifests for local object storage;
- better local development environment;
- clearer artifact schema versioning;
- default runner image support;
- more controller reconciliation tests;
- optional Helm chart;
- integration examples for GitHub Actions, Argo Rollouts, Tekton, and OpenShift Pipelines;
- additional evaluation dimensions for security and privacy checks.

---

## Contributing

This project is in early development.

Issues, bug reports, design discussions, and pull requests are welcome.

Before opening a large pull request, please open an issue to discuss the proposed change.

Useful commands:

```sh
make help
make test
make runner-test
make test-all
```

More information about Kubebuilder-based projects can be found in the Kubebuilder documentation:

```text
https://book.kubebuilder.io/
```

---

## License

Copyright 2026.

Licensed under the Apache License, Version 2.0.

You may obtain a copy of the License at:

```text
http://www.apache.org/licenses/LICENSE-2.0
```

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

See the License for the specific language governing permissions and limitations under the License.