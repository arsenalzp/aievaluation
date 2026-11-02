/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	aieval "github.com/arsenalzp/aievaluation/api/v1alpha1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	ptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/yaml"
)

// ============================================================
// Test helpers
// ============================================================

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()

	scheme := runtime.NewScheme()

	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add corev1 scheme: %v", err)
	}

	if err := batchv1.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add batchv1 scheme: %v", err)
	}

	if err := aieval.AddToScheme(scheme); err != nil {
		t.Fatalf("failed to add aieval scheme: %v", err)
	}

	return scheme
}

func newTestReconciler(t *testing.T, objs ...client.Object) *EvalRunReconciler {
	t.Helper()

	scheme := testScheme(t)

	c := fakeclient.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		WithStatusSubresource(&aieval.EvalRun{}).
		Build()

	return &EvalRunReconciler{
		Client: c,
		Scheme: scheme,
		Queue: workqueue.NewTypedRateLimitingQueue(
			workqueue.DefaultTypedControllerRateLimiter[string](),
		),
	}
}

func testEvalRun(name string) *aieval.EvalRun {
	return &aieval.EvalRun{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: aieval.EvalRunSpec{
			SuiteRef: &v1.LocalObjectReference{
				Name: "suite-1",
			},
			Revision: aieval.Revision{
				GitSHA:   ptr.To("abc123"),
				PRnumber: ptr.To[int32](42),
			},
			SummaryBucket:  "eval-artifacts",
			TargetEndpoint: "http://rag.default.svc.cluster.local/query",
			Model:          "gpt-4.1-mini",
			Embedder:       "text-embedding-3-small",
		},
	}
}

func testEvalSuite(name string) *aieval.EvalSuite {
	minRecall := float32(0.7)
	minPrecision := float32(0.6)
	minFaithfulness := float32(0.8)
	minCorrectness := float32(0.75)
	maxStaleChunkRatio := float32(0.1)
	maxUnknownOwnerRatio := float32(0.2)
	maxDrop := float32(0.05)
	timeoutSeconds := int64(60)

	return &aieval.EvalSuite{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: aieval.EvalSuiteSpec{
			DatasetRef: &v1.LocalObjectReference{
				Name: "dataset-1",
			},
			Metrics: aieval.Metrics{
				Rag: &aieval.Rag{
					Retrieval: aieval.Retrieval{
						MinContextRecall:    &minRecall,
						MinContextPrecision: &minPrecision,
					},
					Generation: aieval.Generation{
						MinFaithfulness: &minFaithfulness,
						MinCorrectness:  &minCorrectness,
					},
					Trustworthiness: aieval.Trustworthiness{
						MaxStaleChunkRatio:   &maxStaleChunkRatio,
						MaxUnknownOwnerRatio: &maxUnknownOwnerRatio,
					},
				},
			},
			RegressionPolicy: aieval.RegressionPolicy{
				MaxDrop: &maxDrop,
			},
			Execution: aieval.Execution{
				RunnerImage:    "ghcr.io/arsenalzp/evalrun-runner:test",
				TimeoutSeconds: &timeoutSeconds,
			},
		},
	}
}

func testEvalDataset(name string) *aieval.EvalDataset {
	return &aieval.EvalDataset{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Spec: aieval.EvalDatasetSpec{
			Format: "jsonl",
			Source: aieval.DatasetObjectSource{
				ObjectStorageEndpoint: "http://object-storage.default.svc.cluster.local:9876",
				Bucket:                "datasets",
				ObjectKey:             "rag/dataset.jsonl",
				VersionId:             "v1",
			},
		},
	}
}

func reconcileRequest(name string) ctrl.Request {
	return ctrl.Request{
		NamespacedName: types.NamespacedName{
			Name:      name,
			Namespace: "default",
		},
	}
}

func envMap(env []corev1.EnvVar) map[string]corev1.EnvVar {
	result := map[string]corev1.EnvVar{}

	for _, e := range env {
		result[e.Name] = e
	}

	return result
}

// ============================================================
// Pure unit tests
// ============================================================

func TestDesiredJobName(t *testing.T) {
	run := testEvalRun("run-1")

	got := desiredJobName(run)

	if got != "evalrun-run-1" {
		t.Fatalf("expected job name evalrun-run-1, got %s", got)
	}
}

func TestJobTerminalStateComplete(t *testing.T) {
	job := &batchv1.Job{
		Status: batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{
				{
					Type:   batchv1.JobComplete,
					Status: corev1.ConditionTrue,
				},
			},
		},
	}

	completed, failed, reason, msg := jobTerminalState(job)

	if !completed {
		t.Fatalf("expected completed=true")
	}

	if failed {
		t.Fatalf("expected failed=false")
	}

	if reason != "Complete" {
		t.Fatalf("expected reason Complete, got %s", reason)
	}

	if msg != "" {
		t.Fatalf("expected empty message, got %s", msg)
	}
}

func TestJobTerminalStateFailed(t *testing.T) {
	job := &batchv1.Job{
		Status: batchv1.JobStatus{
			Conditions: []batchv1.JobCondition{
				{
					Type:    batchv1.JobFailed,
					Status:  corev1.ConditionTrue,
					Reason:  "BackoffLimitExceeded",
					Message: "Job has reached the specified backoff limit",
				},
			},
		},
	}

	completed, failed, reason, msg := jobTerminalState(job)

	if completed {
		t.Fatalf("expected completed=false")
	}

	if !failed {
		t.Fatalf("expected failed=true")
	}

	if reason != "BackoffLimitExceeded" {
		t.Fatalf("unexpected reason: %s", reason)
	}

	if msg != "Job has reached the specified backoff limit" {
		t.Fatalf("unexpected message: %s", msg)
	}
}

func TestRenderSuiteConfigJSON(t *testing.T) {
	suite := testEvalSuite("suite-1")

	data, err := renderSuiteConfigJSON(suite)
	if err != nil {
		t.Fatalf("renderSuiteConfigJSON failed: %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("rendered suite config is not valid JSON: %v", err)
	}

	metrics := decoded["metrics"].(map[string]any)["rag"].(map[string]any)
	retrieval := metrics["retrieval"].(map[string]any)
	generation := metrics["generation"].(map[string]any)
	trustworthiness := metrics["trustworthiness"].(map[string]any)
	regressionPolicy := decoded["regressionPolicy"].(map[string]any)
	execution := decoded["execution"].(map[string]any)

	if retrieval["minContextRecall"] != float64(0.7) {
		t.Fatalf("unexpected minContextRecall: %v", retrieval["minContextRecall"])
	}

	if retrieval["minContextPrecision"] != float64(0.6) {
		t.Fatalf("unexpected minContextPrecision: %v", retrieval["minContextPrecision"])
	}

	if generation["minFaithfulness"] != float64(0.8) {
		t.Fatalf("unexpected minFaithfulness: %v", generation["minFaithfulness"])
	}

	if generation["minCorrectness"] != float64(0.75) {
		t.Fatalf("unexpected minCorrectness: %v", generation["minCorrectness"])
	}

	if trustworthiness["maxStaleChunkRatio"] != float64(0.1) {
		t.Fatalf("unexpected maxStaleChunkRatio: %v", trustworthiness["maxStaleChunkRatio"])
	}

	if trustworthiness["maxUnknownOwnerRatio"] != float64(0.2) {
		t.Fatalf("unexpected maxUnknownOwnerRatio: %v", trustworthiness["maxUnknownOwnerRatio"])
	}

	if regressionPolicy["maxDrop"] != float64(0.05) {
		t.Fatalf("unexpected maxDrop: %v", regressionPolicy["maxDrop"])
	}

	if execution["timeoutSeconds"] != float64(60) {
		t.Fatalf("unexpected timeoutSeconds: %v", execution["timeoutSeconds"])
	}
}

// ============================================================
// Job contract tests
// ============================================================

func TestBuildEvalJobEnvironmentContract(t *testing.T) {
	run := testEvalRun("run-1")
	suite := testEvalSuite("suite-1")
	dataset := testEvalDataset("dataset-1")

	run.Status.SuiteSpecHash = hashSuiteSpec(suite)

	job := buildEvalJob(run, suite, dataset, "evalrun-run-1", "evalsuite-suite-1-cfg")

	if job.Name != "evalrun-run-1" {
		t.Fatalf("unexpected job name: %s", job.Name)
	}

	if job.Namespace != "default" {
		t.Fatalf("unexpected namespace: %s", job.Namespace)
	}

	if job.Spec.BackoffLimit == nil || *job.Spec.BackoffLimit != 0 {
		t.Fatalf("expected BackoffLimit=0")
	}

	if job.Spec.Template.Spec.RestartPolicy != corev1.RestartPolicyNever {
		t.Fatalf("expected RestartPolicyNever")
	}

	if len(job.Spec.Template.Spec.Containers) != 1 {
		t.Fatalf("expected exactly one container")
	}

	container := job.Spec.Template.Spec.Containers[0]

	if container.Name != "runner" {
		t.Fatalf("unexpected container name: %s", container.Name)
	}

	if container.Image != "ghcr.io/arsenalzp/evalrun-runner:test" {
		t.Fatalf("unexpected runner image: %s", container.Image)
	}

	env := envMap(container.Env)

	expectedEnvValues := map[string]string{
		"EVALRUN_NAME":          "run-1",
		"EVALSUITE_NAME":        "suite-1",
		"GIT_SHA":               "abc123",
		"PR_NUMBER":             "42",
		"WORK_DIR":              "/work",
		"DATASET_FILE":          "/work/dataset",
		"SUITE_CONFIG":          "/config/suite.json",
		"DATASET_S3_BUCKET":     "datasets",
		"DATASET_S3_KEY":        "rag/dataset.jsonl",
		"DATASET_S3_VERSION_ID": "v1",
		"DATASET_FORMAT":        "jsonl",
		"S3_ENDPOINT":           "http://object-storage.default.svc.cluster.local:9876",
		"S3_RUN_PREFIX":         "runs/run-1",
		"S3_BASELINE_KEY":       "baselines/suite-1.json",
		"GATE_SUMMARY_KEY":      "runs/run-1/gate_summary.json",
		"SCORES_KEY":            "runs/run-1/scores.json",
		"DETAILS_KEY":           "runs/run-1/details.jsonl",
		"S3_SUMMARY_BUCKET":     "eval-artifacts",
		"S3_SUMMARY_KEY":        "runs/run-1/gate_summary.json",
		"TARGET_ENDPOINT":       "http://rag.default.svc.cluster.local/query",
		"OPENAI_MODEL":          "gpt-4.1-mini",
		"OPENAI_EMBEDDER":       "text-embedding-3-small",
	}

	for name, expected := range expectedEnvValues {
		got, ok := env[name]
		if !ok {
			t.Fatalf("missing env var %s", name)
		}

		if got.Value != expected {
			t.Fatalf("env %s: expected %q, got %q", name, expected, got.Value)
		}
	}

	secretEnvNames := []string{
		"S3_ACCESS_KEY",
		"S3_SECRET_KEY",
		"OPENAI_API_KEY",
	}

	for _, name := range secretEnvNames {
		got, ok := env[name]
		if !ok {
			t.Fatalf("missing secret env var %s", name)
		}

		if got.ValueFrom == nil || got.ValueFrom.SecretKeyRef == nil {
			t.Fatalf("env %s should use SecretKeyRef", name)
		}
	}

	if env["S3_ACCESS_KEY"].ValueFrom.SecretKeyRef.Name != "eval-s3-credentials" {
		t.Fatalf("S3_ACCESS_KEY should come from eval-s3-credentials")
	}

	if env["S3_ACCESS_KEY"].ValueFrom.SecretKeyRef.Key != "accessKey" {
		t.Fatalf("S3_ACCESS_KEY should use key accessKey")
	}

	if env["S3_SECRET_KEY"].ValueFrom.SecretKeyRef.Name != "eval-s3-credentials" {
		t.Fatalf("S3_SECRET_KEY should come from eval-s3-credentials")
	}

	if env["S3_SECRET_KEY"].ValueFrom.SecretKeyRef.Key != "secretKey" {
		t.Fatalf("S3_SECRET_KEY should use key secretKey")
	}

	if env["OPENAI_API_KEY"].ValueFrom.SecretKeyRef.Name != "openai-credentials" {
		t.Fatalf("OPENAI_API_KEY should come from openai-credentials")
	}

	if env["OPENAI_API_KEY"].ValueFrom.SecretKeyRef.Key != "apiKey" {
		t.Fatalf("OPENAI_API_KEY should use key apiKey")
	}

	if len(job.Spec.Template.Spec.Volumes) != 2 {
		t.Fatalf("expected workdir and suite-config volumes")
	}

	labels := job.Labels

	if labels["app.kubernetes.io/name"] != "eval-gate" {
		t.Fatalf("missing app label")
	}

	if labels["ai.evaluation.io/evalrun"] != "run-1" {
		t.Fatalf("missing evalrun label")
	}

	if labels["ai.evaluation.io/evalsuite"] != "suite-1" {
		t.Fatalf("missing evalsuite label")
	}

	if labels["ai.evaluation.io/suite-hash"] != run.Status.SuiteSpecHash {
		t.Fatalf("missing suite hash label")
	}
}

// ============================================================
// Reconcile tests
// ============================================================

func TestReconcileCreatesJobAndConfigMapForNewEvalRun(t *testing.T) {
	ctx := context.Background()

	run := testEvalRun("run-1")
	suite := testEvalSuite("suite-1")
	dataset := testEvalDataset("dataset-1")

	r := newTestReconciler(t, run, suite, dataset)

	result, err := r.Reconcile(ctx, reconcileRequest("run-1"))
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	if result.RequeueAfter != 5*time.Second {
		t.Fatalf("expected RequeueAfter=5s, got %s", result.RequeueAfter)
	}

	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatalf("failed to get updated EvalRun: %v", err)
	}

	expectedHash := hashSuiteSpec(suite)

	if updated.Status.Phase != aieval.PhaseRunning {
		t.Fatalf("expected phase RUNNING, got %s", updated.Status.Phase)
	}

	if updated.Status.JobRef.Name != "evalrun-run-1" {
		t.Fatalf("unexpected job ref: %s", updated.Status.JobRef.Name)
	}

	if updated.Status.SuiteSpecHash != expectedHash {
		t.Fatalf("unexpected suite hash: %s", updated.Status.SuiteSpecHash)
	}

	if updated.Status.StartedAt.IsZero() {
		t.Fatalf("expected StartedAt to be set")
	}

	var job batchv1.Job
	if err := r.Get(ctx, types.NamespacedName{Name: "evalrun-run-1", Namespace: "default"}, &job); err != nil {
		t.Fatalf("expected Job to be created: %v", err)
	}

	if len(job.OwnerReferences) != 1 {
		t.Fatalf("expected one owner reference")
	}

	if job.OwnerReferences[0].Name != "run-1" {
		t.Fatalf("unexpected owner reference: %s", job.OwnerReferences[0].Name)
	}

	container := job.Spec.Template.Spec.Containers[0]
	if container.Image != suite.Spec.Execution.RunnerImage {
		t.Fatalf("unexpected runner image: %s", container.Image)
	}

	cmName := suiteConfigMapName(suite.Name, expectedHash)

	var cm corev1.ConfigMap
	if err := r.Get(ctx, types.NamespacedName{Name: cmName, Namespace: "default"}, &cm); err != nil {
		t.Fatalf("expected ConfigMap to be created: %v", err)
	}

	if cm.Data["suite.json"] == "" {
		t.Fatalf("expected suite.json in ConfigMap")
	}

	if cm.Labels["ai.evaluation.io/evalsuite"] != "suite-1" {
		t.Fatalf("unexpected ConfigMap evalsuite label")
	}

	if cm.Labels["ai.evaluation.io/suite-hash"] != expectedHash {
		t.Fatalf("unexpected ConfigMap suite hash label")
	}
}

func TestReconcileDoesNotCreateDuplicateJob(t *testing.T) {
	ctx := context.Background()

	run := testEvalRun("run-1")
	suite := testEvalSuite("suite-1")
	dataset := testEvalDataset("dataset-1")

	r := newTestReconciler(t, run, suite, dataset)

	if _, err := r.Reconcile(ctx, reconcileRequest("run-1")); err != nil {
		t.Fatalf("first reconcile failed: %v", err)
	}

	if _, err := r.Reconcile(ctx, reconcileRequest("run-1")); err != nil {
		t.Fatalf("second reconcile failed: %v", err)
	}

	var jobs batchv1.JobList
	if err := r.List(ctx, &jobs, client.InNamespace("default")); err != nil {
		t.Fatalf("failed to list jobs: %v", err)
	}

	if len(jobs.Items) != 1 {
		t.Fatalf("expected exactly one Job, got %d", len(jobs.Items))
	}

	if jobs.Items[0].Name != "evalrun-run-1" {
		t.Fatalf("unexpected job name: %s", jobs.Items[0].Name)
	}
}

func TestReconcileMarksEvalRunFailedWhenJobFailed(t *testing.T) {
	ctx := context.Background()

	run := testEvalRun("run-1")
	suite := testEvalSuite("suite-1")
	dataset := testEvalDataset("dataset-1")

	run.Status.Phase = aieval.PhaseRunning
	run.Status.SuiteSpecHash = hashSuiteSpec(suite)
	run.Status.JobRef = v1.LocalObjectReference{Name: "evalrun-run-1"}

	job := buildEvalJob(run, suite, dataset, "evalrun-run-1", suiteConfigMapName(suite.Name, run.Status.SuiteSpecHash))
	job.Status.Conditions = []batchv1.JobCondition{
		{
			Type:    batchv1.JobFailed,
			Status:  corev1.ConditionTrue,
			Reason:  "BackoffLimitExceeded",
			Message: "Job has reached the specified backoff limit",
		},
	}

	r := newTestReconciler(t, run, suite, dataset, job)
	r.GateFetcher = func(context.Context, *aieval.EvalRun, string) (*aieval.GateSummary, error) {
		return nil, fmt.Errorf("gate_summary not published")
	}

	_, err := r.Reconcile(ctx, reconcileRequest("run-1"))
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatalf("failed to get updated EvalRun: %v", err)
	}

	if updated.Status.Phase != aieval.PhaseFailed {
		t.Fatalf("expected phase FAILED, got %s", updated.Status.Phase)
	}

	if updated.Status.Verdict != "FAIL" {
		t.Fatalf("expected verdict FAIL, got %s", updated.Status.Verdict)
	}

	if updated.Status.RootCause.Stage != "Execution" {
		t.Fatalf("expected root cause stage Execution, got %s", updated.Status.RootCause.Stage)
	}

	if updated.Status.Summary == "" {
		t.Fatalf("expected failure summary to be set")
	}

	if updated.Status.CompletedAt.IsZero() {
		t.Fatalf("expected CompletedAt to be set")
	}
}

func TestReconcileSetsPendingWhenDatasetObjectKeyMissing(t *testing.T) {
	ctx := context.Background()

	run := testEvalRun("run-1")
	suite := testEvalSuite("suite-1")
	dataset := testEvalDataset("dataset-1")
	dataset.Spec.Source.ObjectKey = ""

	r := newTestReconciler(t, run, suite, dataset)

	result, err := r.Reconcile(ctx, reconcileRequest("run-1"))
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	if result.RequeueAfter != 30*time.Second {
		t.Fatalf("expected RequeueAfter=30s, got %s", result.RequeueAfter)
	}

	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatalf("failed to get updated EvalRun: %v", err)
	}

	if updated.Status.Phase != aieval.PhasePending {
		t.Fatalf("expected phase PENDING, got %s", updated.Status.Phase)
	}

	cond := meta.FindStatusCondition(updated.Status.Conditions, "DatasetObjectKeyMissing")
	if cond == nil {
		t.Fatalf("expected DatasetObjectKeyMissing condition")
	}

	if cond.Reason != "Pending" {
		t.Fatalf("expected condition reason Pending, got %s", cond.Reason)
	}
}

func TestReconcileSetsPendingWhenEvalSuiteMissing(t *testing.T) {
	ctx := context.Background()

	run := testEvalRun("run-1")

	r := newTestReconciler(t, run)

	result, err := r.Reconcile(ctx, reconcileRequest("run-1"))
	if err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	if result.RequeueAfter != 30*time.Second {
		t.Fatalf("expected RequeueAfter=30s, got %s", result.RequeueAfter)
	}

	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatalf("failed to get updated EvalRun: %v", err)
	}

	if updated.Status.Phase != aieval.PhasePending {
		t.Fatalf("expected phase PENDING, got %s", updated.Status.Phase)
	}

	cond := meta.FindStatusCondition(updated.Status.Conditions, "SuiteNotFound")
	if cond == nil {
		t.Fatalf("expected SuiteNotFound condition")
	}
}

// ============================================================
// Multi-turn evaluation
// ============================================================

func legacyAgentSuite(name string) *aieval.EvalSuite {
	minSel, minInv, minExec := float32(0.9), float32(0.9), float32(0.85)
	maxViol := int32(0)
	maxDrop := float32(0.03)
	timeout := int64(300)

	return &aieval.EvalSuite{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "default"},
		Spec: aieval.EvalSuiteSpec{
			DatasetRef: &v1.LocalObjectReference{Name: "dataset-1"},
			Metrics: aieval.Metrics{Agent: &aieval.Agent{
				MinToolSelection: &minSel, MinToolInvocation: &minInv, MinToolExecution: &minExec,
				MaxPolicyViolations: &maxViol,
			}},
			RegressionPolicy: aieval.RegressionPolicy{MaxDrop: &maxDrop},
			Execution:        aieval.Execution{RunnerImage: "ghcr.io/arsenalzp/evalrun-runner:test", TimeoutSeconds: &timeout},
		},
	}
}

func multiTurnSuite(name string) *aieval.EvalSuite {
	suite := legacyAgentSuite(name)
	suite.Spec.Execution.Mode = "agent" // the task type is execution.mode
	f32 := func(v float32) *float32 { return &v }
	suite.Spec.Evaluation = &aieval.Evaluation{
		InteractionMode: "MultiTurn", Trials: ptr.To[int32](2),
		Trace: &aieval.TraceSpec{Source: "auto", Phoenix: &aieval.PhoenixTrace{Endpoint: "http://phoenix:6006", Project: "travel"}},
	}
	suite.Spec.Metrics.Agent.Outcome = &aieval.AgentOutcome{MinGoalCompletion: f32(0.9)}
	suite.Spec.Metrics.Agent.Conversation = &aieval.AgentConversation{
		MaxTurns: ptr.To[int32](10), MinKnowledgeRetention: f32(0.85),
		MinConversationCompleteness: f32(0.85), MinTurnRelevancy: f32(0.8),
	}
	suite.Spec.Metrics.Agent.Tools = &aieval.AgentTools{
		MinSelectionAccuracy: f32(0.9), MinArgumentAccuracy: f32(0.9),
		MaxToolCalls: ptr.To[int32](20), MaxRepeatedCalls: ptr.To[int32](1),
	}
	suite.Spec.Metrics.Agent.Policy = &aieval.AgentPolicy{
		MaxCriticalViolations: ptr.To[int32](0), MaxHighViolations: ptr.To[int32](0), LimitsAction: "Fail",
	}
	suite.Spec.Evaluators = []aieval.Evaluator{
		{Name: "trajectory-policy", Provider: "builtin", Type: "deterministic", Required: true},
		{Name: "conversation-quality", Provider: "deepeval", Type: "multiTurn", Required: false,
			Metrics: []string{"knowledgeRetention", "conversationCompleteness", "turnRelevancy"}, Samples: ptr.To[int32](3)},
	}
	return suite
}

func TestLegacySuitesRenderAndHashAreUnchangedByMultiTurnFields(t *testing.T) {
	suite := legacyAgentSuite("suite-1")

	// Spec JSON (input of the suite hash) must not gain any key for unset multi-turn fields.
	spec, err := json.Marshal(suite.Spec)
	if err != nil {
		t.Fatal(err)
	}
	const wantSpec = `{"datasetRef":{"name":"dataset-1"},"metrics":{"agent":{"minToolSelection":0.9,"minToolInvocation":0.9,` +
		`"minToolExecution":0.85,"maxPolicyViolations":0}},"regressionPolicy":{"maxDrop":0.03},` +
		`"execution":{"runnerImage":"ghcr.io/arsenalzp/evalrun-runner:test","timeoutSeconds":300}}`
	if string(spec) != wantSpec {
		t.Fatalf("legacy spec JSON changed (suite hash would change):\n got: %s\nwant: %s", spec, wantSpec)
	}

	data, err := renderSuiteConfigJSON(suite)
	if err != nil {
		t.Fatal(err)
	}
	var got any
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatal(err)
	}
	const wantCfg = `{"metrics":{"agent":{"minToolSelection":0.9,"minToolInvocation":0.9,"minToolExecution":0.85,` +
		`"maxPolicyViolations":0}},"regressionPolicy":{"maxDrop":0.03},"execution":{"timeoutSeconds":300}}`
	var want any
	if err := json.Unmarshal([]byte(wantCfg), &want); err != nil {
		t.Fatal(err)
	}
	gotJSON, _ := json.Marshal(got)
	wantJSON, _ := json.Marshal(want)
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("legacy agent suite.json changed:\n got: %s\nwant: %s", gotJSON, wantJSON)
	}
}

func TestRenderSuiteConfigJSONForMultiTurn(t *testing.T) {
	data, err := renderSuiteConfigJSON(multiTurnSuite("suite-mt"))
	if err != nil {
		t.Fatalf("render failed: %v", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	eval := cfg["evaluation"].(map[string]any)
	if eval["interactionMode"] != "MultiTurn" || eval["trials"] != float64(2) || eval["type"] != nil {
		t.Fatalf("unexpected evaluation: %v", eval)
	}
	trace := eval["trace"].(map[string]any)
	phoenix := trace["phoenix"].(map[string]any)
	if trace["source"] != "auto" || phoenix["endpoint"] != "http://phoenix:6006" || phoenix["project"] != "travel" {
		t.Fatalf("unexpected trace: %v", trace)
	}

	agent := cfg["metrics"].(map[string]any)["agent"].(map[string]any)
	// legacy flat fields are still rendered
	if agent["minToolSelection"] != 0.9 || agent["maxPolicyViolations"] != float64(0) {
		t.Fatalf("legacy agent fields lost: %v", agent)
	}
	conv := agent["conversation"].(map[string]any)
	if conv["maxTurns"] != float64(10) || conv["minKnowledgeRetention"] != 0.85 || conv["minTurnRelevancy"] != 0.8 {
		t.Fatalf("unexpected conversation: %v", conv)
	}
	tools := agent["tools"].(map[string]any)
	if tools["maxToolCalls"] != float64(20) || tools["maxRepeatedCalls"] != float64(1) || tools["minArgumentAccuracy"] != 0.9 {
		t.Fatalf("unexpected tools: %v", tools)
	}
	policy := agent["policy"].(map[string]any)
	if policy["maxCriticalViolations"] != float64(0) || policy["maxHighViolations"] != float64(0) || policy["limitsAction"] != "Fail" {
		t.Fatalf("unexpected policy: %v", policy)
	}
	if agent["outcome"].(map[string]any)["minGoalCompletion"] != 0.9 {
		t.Fatalf("unexpected outcome: %v", agent["outcome"])
	}

	evaluators := cfg["evaluators"].([]any)
	if len(evaluators) != 2 {
		t.Fatalf("expected 2 evaluators, got %d", len(evaluators))
	}
	semantic := evaluators[1].(map[string]any)
	if semantic["name"] != "conversation-quality" || semantic["provider"] != "deepeval" || semantic["samples"] != float64(3) {
		t.Fatalf("unexpected semantic evaluator: %v", semantic)
	}
	if semantic["required"] != nil && semantic["required"] != false {
		t.Fatalf("optional evaluator must not be required: %v", semantic)
	}
	if got := len(semantic["metrics"].([]any)); got != 3 {
		t.Fatalf("expected 3 semantic metrics, got %d", got)
	}
	if evaluators[0].(map[string]any)["required"] != true {
		t.Fatalf("builtin evaluator should be required: %v", evaluators[0])
	}
}

func TestRenderedMultiTurnConfigMatchesWhatTheRunnerParses(t *testing.T) {
	// The runner reads exactly these paths (runner-agent/multiturn/config.py).
	data, _ := renderSuiteConfigJSON(multiTurnSuite("suite-mt"))
	for _, path := range []string{
		`"evaluation":{"interactionMode":"MultiTurn"`,
		`"outcome":{"minGoalCompletion"`,
		`"conversation":{"maxTurns"`,
		`"tools":{"minSelectionAccuracy"`,
		`"policy":{"maxCriticalViolations"`,
		`"evaluators":[{"name":"trajectory-policy"`,
	} {
		compact := strings.NewReplacer(" ", "", "\n", "").Replace(string(data))
		if !strings.Contains(compact, path) {
			t.Fatalf("rendered config lacks %s\n%s", path, data)
		}
	}
}

func TestBuildEvalJobProvidesMultiTurnArtifactKeys(t *testing.T) {
	run := testEvalRun("run-1")
	suite := multiTurnSuite("suite-mt")
	run.Status.SuiteSpecHash = hashSuiteSpec(suite)

	job := buildEvalJob(run, suite, testEvalDataset("dataset-1"), "evalrun-run-1", "cfg")
	env := envMap(job.Spec.Template.Spec.Containers[0].Env)

	want := map[string]string{
		"MANIFEST_KEY":         "runs/run-1/manifest.json",
		"VIOLATIONS_KEY":       "runs/run-1/violations.jsonl",
		"CONVERSATIONS_PREFIX": "runs/run-1/conversations",
		// pre-existing contract is untouched
		"GATE_SUMMARY_KEY": "runs/run-1/gate_summary.json",
		"SCORES_KEY":       "runs/run-1/scores.json",
		"DETAILS_KEY":      "runs/run-1/details.jsonl",
		"S3_RUN_PREFIX":    "runs/run-1",
		"TARGET_ENDPOINT":  "http://rag.default.svc.cluster.local/query",
		"SUITE_CONFIG":     "/config/suite.json",
	}
	for name, value := range want {
		if env[name].Value != value {
			t.Fatalf("env %s: expected %q, got %q", name, value, env[name].Value)
		}
	}
	// no additional Secret is required for multi-turn: judge LLM calls reuse OPENAI_API_KEY
	for name, e := range env {
		if e.ValueFrom != nil && name != "S3_ACCESS_KEY" && name != "S3_SECRET_KEY" && name != "OPENAI_API_KEY" {
			t.Fatalf("unexpected secret env var %s", name)
		}
	}
}

func TestReconcileCreatesJobAndConfigMapForMultiTurnEvalRun(t *testing.T) {
	ctx := context.Background()
	run, suite, dataset := testEvalRun("run-1"), multiTurnSuite("suite-1"), testEvalDataset("dataset-1")
	r := newTestReconciler(t, run, suite, dataset)

	if _, err := r.Reconcile(ctx, reconcileRequest("run-1")); err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}

	var job batchv1.Job
	if err := r.Get(ctx, types.NamespacedName{Name: "evalrun-run-1", Namespace: "default"}, &job); err != nil {
		t.Fatalf("expected Job: %v", err)
	}
	var cm corev1.ConfigMap
	cmName := suiteConfigMapName(suite.Name, hashSuiteSpec(suite))
	if err := r.Get(ctx, types.NamespacedName{Name: cmName, Namespace: "default"}, &cm); err != nil {
		t.Fatalf("expected ConfigMap: %v", err)
	}
	var cfg map[string]any
	if err := json.Unmarshal([]byte(cm.Data["suite.json"]), &cfg); err != nil {
		t.Fatalf("suite.json invalid: %v", err)
	}
	if cfg["evaluation"].(map[string]any)["interactionMode"] != "MultiTurn" {
		t.Fatalf("runner would not see MultiTurn: %v", cfg["evaluation"])
	}

	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatal(err)
	}
	if updated.Status.Phase != aieval.PhaseRunning {
		t.Fatalf("expected RUNNING, got %s", updated.Status.Phase)
	}
}

func TestValidateSuite(t *testing.T) {
	cases := []struct {
		name    string
		mutate  func(s *aieval.EvalSuite)
		wantErr string
	}{
		{"legacy suite is valid", func(s *aieval.EvalSuite) { s.Spec.Evaluation, s.Spec.Evaluators = nil, nil }, ""},
		{"multi-turn suite is valid", func(s *aieval.EvalSuite) {}, ""},
		{"evaluators need MultiTurn", func(s *aieval.EvalSuite) { s.Spec.Evaluation = nil }, "only supported with"},
		{"mode rag cannot be multi-turn", func(s *aieval.EvalSuite) { s.Spec.Execution.Mode = "rag" }, "only supported for agent"},
		{"rag metrics without mode cannot be multi-turn", func(s *aieval.EvalSuite) {
			s.Spec.Execution.Mode, s.Spec.Metrics = "", aieval.Metrics{Rag: &aieval.Rag{}}
		}, "only supported for agent"},
		{"agent mode with only rag metrics is inconsistent", func(s *aieval.EvalSuite) {
			s.Spec.Execution.Mode, s.Spec.Metrics = "agent", aieval.Metrics{Rag: &aieval.Rag{}}
		}, "mode is agent"},
		{"mode may be omitted when metrics.agent is set (as in existing agent suites)", func(s *aieval.EvalSuite) { s.Spec.Execution.Mode = "" }, ""},
		{"phoenix source needs endpoint", func(s *aieval.EvalSuite) { s.Spec.Evaluation.Trace.Phoenix.Endpoint = "" }, "phoenix.endpoint"},
		{"duplicate evaluator", func(s *aieval.EvalSuite) { s.Spec.Evaluators[1].Name = "trajectory-policy" }, "duplicate"},
		{"deepeval is not deterministic", func(s *aieval.EvalSuite) { s.Spec.Evaluators[1].Type = "deterministic" }, "deepeval only supports"},
		{"builtin is not multiTurn", func(s *aieval.EvalSuite) { s.Spec.Evaluators[0].Type = "multiTurn" }, "builtin only supports"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			suite := multiTurnSuite("s")
			tc.mutate(suite)
			err := validateSuite(suite)
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("unexpected error: %v", err)
			case tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)):
				t.Fatalf("expected error containing %q, got %v", tc.wantErr, err)
			}
		})
	}
}

func TestReconcileSetsPendingForInvalidMultiTurnSuite(t *testing.T) {
	ctx := context.Background()
	suite := multiTurnSuite("suite-1")
	suite.Spec.Evaluation.Trace.Phoenix.Endpoint = ""
	r := newTestReconciler(t, testEvalRun("run-1"), suite, testEvalDataset("dataset-1"))

	if _, err := r.Reconcile(ctx, reconcileRequest("run-1")); err != nil {
		t.Fatal(err)
	}
	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatal(err)
	}
	cond := meta.FindStatusCondition(updated.Status.Conditions, "SuiteInvalid")
	if updated.Status.Phase != aieval.PhasePending || cond == nil || !strings.Contains(cond.Message, "phoenix.endpoint") {
		t.Fatalf("expected Pending/SuiteInvalid, got phase=%s cond=%v", updated.Status.Phase, cond)
	}
	var jobs batchv1.JobList
	_ = r.List(ctx, &jobs, client.InNamespace("default"))
	if len(jobs.Items) != 0 {
		t.Fatalf("no Job must be created for an invalid suite")
	}
}

// failingGateSummaryJSON is a gate_summary.json exactly as the multi-turn runner writes it.
const failingGateSummaryJSON = `{
  "schemaVersion": "evalrun.gate_summary/v1",
  "evaluationType": "agent",
  "interactionMode": "MultiTurn",
  "verdict": "FAIL",
  "summary": "Policy failed: 'book_flight' was called without the required event 'explicit_user_confirmation'",
  "rootCause": {
    "stage": "Policy",
    "evaluator": "trajectory-policy",
    "code": "ACTION_WITHOUT_APPROVAL",
    "metric": "critical_violations",
    "observed": "1",
    "required": "<= 0",
    "caseId": "travel-approval-001",
    "turnIndex": 4,
    "artifactUri": "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/conversations/travel-approval-001.json",
    "details": "'book_flight' was called without the required event 'explicit_user_confirmation'"
  },
  "findings": [{"level": "FAIL", "stage": "Policy"}],
  "artifacts": {
    "gateSummaryUri": "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/gate_summary.json",
    "scoresUri": "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/scores.json",
    "detailsUri": "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/details.jsonl",
    "manifestUri": "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/manifest.json",
    "violationsUri": "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/violations.jsonl",
    "conversationsUri": "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/conversations/"
  }
}`

func completedJob(run *aieval.EvalRun, suite *aieval.EvalSuite, dataset *aieval.EvalDataset) *batchv1.Job {
	job := buildEvalJob(run, suite, dataset, "evalrun-"+run.Name, suiteConfigMapName(suite.Name, run.Status.SuiteSpecHash))
	job.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobComplete, Status: corev1.ConditionTrue}}
	return job
}

func reconcileFinishedRun(t *testing.T, gateJSON string) aieval.EvalRun {
	t.Helper()
	ctx := context.Background()
	run, suite, dataset := testEvalRun("run-1"), multiTurnSuite("suite-1"), testEvalDataset("dataset-1")
	run.Status.Phase = aieval.PhaseRunning
	run.Status.SuiteSpecHash = hashSuiteSpec(suite)
	run.Status.JobRef = v1.LocalObjectReference{Name: "evalrun-run-1"}

	r := newTestReconciler(t, run, suite, dataset, completedJob(run, suite, dataset))
	r.GateFetcher = func(_ context.Context, got *aieval.EvalRun, endpoint string) (*aieval.GateSummary, error) {
		if got.Name != "run-1" || endpoint != dataset.Spec.Source.ObjectStorageEndpoint {
			t.Fatalf("gate fetched for unexpected run/endpoint: %s %s", got.Name, endpoint)
		}
		var gate aieval.GateSummary
		if err := json.Unmarshal([]byte(gateJSON), &gate); err != nil {
			return nil, err
		}
		return &gate, nil
	}
	if _, err := r.Reconcile(ctx, reconcileRequest("run-1")); err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatal(err)
	}
	return updated
}

func TestGateSummaryWithCriticalViolationIsPropagatedToStatus(t *testing.T) {
	updated := reconcileFinishedRun(t, failingGateSummaryJSON)

	if updated.Status.Phase != aieval.PhaseCompleted || updated.Status.Verdict != "FAIL" {
		t.Fatalf("expected COMPLETED/FAIL, got %s/%s", updated.Status.Phase, updated.Status.Verdict)
	}
	rc := updated.Status.RootCause
	if rc.Stage != "Policy" || rc.Code != "ACTION_WITHOUT_APPROVAL" || rc.Evaluator != "trajectory-policy" ||
		rc.CaseID != "travel-approval-001" || rc.Metric != "critical_violations" ||
		rc.Observed != "1" || rc.Required != "<= 0" {
		t.Fatalf("root cause not preserved: %+v", rc)
	}
	if rc.TurnIndex == nil || *rc.TurnIndex != 4 {
		t.Fatalf("turn index not preserved: %v", rc.TurnIndex)
	}
	if rc.ArtifactURI != "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/conversations/travel-approval-001.json" {
		t.Fatalf("artifact uri not preserved: %s", rc.ArtifactURI)
	}
	if !strings.Contains(updated.Status.Summary, "without the required event") {
		t.Fatalf("summary not propagated: %q", updated.Status.Summary)
	}
	if cond := meta.FindStatusCondition(updated.Status.Conditions, "GateWarning"); cond != nil {
		t.Fatalf("FAIL verdict must not set GateWarning")
	}
}

func TestGateSummaryArtifactURIsArePropagatedToStatus(t *testing.T) {
	updated := reconcileFinishedRun(t, failingGateSummaryJSON)
	a := updated.Status.Artifacts

	want := map[string]string{
		a.GateSummaryUri:   "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/gate_summary.json",
		a.ScoresUri:        "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/scores.json",
		a.DetailsUri:       "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/details.jsonl",
		a.ManifestUri:      "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/manifest.json",
		a.ViolationsUri:    "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/violations.jsonl",
		a.ConversationsUri: "http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/conversations/",
	}
	for got, expected := range want {
		if got != expected {
			t.Fatalf("artifact uri: expected %q, got %q (all: %+v)", expected, got, a)
		}
	}
}

func TestWarnVerdictIsPropagatedWithGateWarningCondition(t *testing.T) {
	warn := `{"verdict":"WARN","summary":"Passed with warnings (1): judge unavailable",
	  "rootCause":{"stage":"Evaluator","evaluator":"conversation-quality","code":"EVALUATOR_FAILURE","details":"judge unavailable"},
	  "artifacts":{"gateSummaryUri":"http://k8s:9876/buckets/b/objects/g.json"}}`
	updated := reconcileFinishedRun(t, warn)

	if updated.Status.Verdict != "WARN" || updated.Status.Phase != aieval.PhaseCompleted {
		t.Fatalf("expected COMPLETED/WARN, got %s/%s", updated.Status.Phase, updated.Status.Verdict)
	}
	if updated.Status.RootCause.Evaluator != "conversation-quality" {
		t.Fatalf("unexpected root cause: %+v", updated.Status.RootCause)
	}
	cond := meta.FindStatusCondition(updated.Status.Conditions, "GateWarning")
	if cond == nil || cond.Status != metav1.ConditionTrue {
		t.Fatalf("expected GateWarning=True, got %v", cond)
	}
}

func TestSingleTurnGateSummaryStillPropagates(t *testing.T) {
	old := `{"schemaVersion":"evalrun.gate_summary/v1","evaluationType":"agent","verdict":"PASS","summary":"ok",
	  "rootCause":{"stage":"None"},
	  "artifacts":{"gateSummaryUri":"http://k8s:9876/buckets/b/objects/g.json","scoresUri":"http://k8s:9876/buckets/b/objects/s.json","detailsUri":"http://k8s:9876/buckets/b/objects/d.jsonl"}}`
	updated := reconcileFinishedRun(t, old)
	if updated.Status.Verdict != "PASS" || updated.Status.RootCause.Stage != "None" || updated.Status.Artifacts.ManifestUri != "" {
		t.Fatalf("unexpected status for single-turn gate: %+v", updated.Status)
	}
}

// The example manifests must stay valid against the Go API types (catches schema drift in docs).
func TestMultiTurnExampleManifestsDecode(t *testing.T) {
	read := func(file string, into any) {
		raw, err := os.ReadFile(filepath.Join("..", "..", "examples", file))
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		if err := yaml.UnmarshalStrict(raw, into); err != nil {
			t.Fatalf("%s does not match the API types: %v", file, err)
		}
	}
	var suite aieval.EvalSuite
	read("evalsuite-agent-multiturn.yaml", &suite)
	if !isMultiTurn(&suite) || suite.Spec.Metrics.Agent == nil || suite.Spec.Metrics.Agent.Conversation == nil {
		t.Fatalf("example suite is not a multi-turn agent suite: %+v", suite.Spec)
	}
	if err := validateSuite(&suite); err != nil {
		t.Fatalf("example suite is invalid: %v", err)
	}
	rendered, err := renderSuiteConfigJSON(&suite)
	if err != nil {
		t.Fatalf("example suite does not render: %v", err)
	}
	// examples/multiturn-local/suite.json is what the controller would mount for this suite; keep them in sync.
	local, err := os.ReadFile(filepath.Join("..", "..", "examples", "multiturn-local", "suite.json"))
	if err != nil {
		t.Fatal(err)
	}
	var a, b any
	if err := json.Unmarshal(rendered, &a); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(local, &b); err != nil {
		t.Fatalf("local suite.json invalid: %v", err)
	}
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	if string(aj) != string(bj) {
		t.Fatalf("examples/multiturn-local/suite.json drifted from the rendered example suite:\n rendered: %s\n local:    %s", aj, bj)
	}

	var dataset aieval.EvalDataset
	read("evaldataset-agent-multiturn.yaml", &dataset)
	var run aieval.EvalRun
	read("evalrun-agent-multiturn.yaml", &run)
	if run.Spec.SuiteRef.Name != suite.Name || suite.Spec.DatasetRef.Name != dataset.Name {
		t.Fatalf("example manifests reference each other incorrectly: run->%s suite->%s",
			run.Spec.SuiteRef.Name, suite.Spec.DatasetRef.Name)
	}
}

func TestFailedJobKeepsTheRunnersPublishedRootCause(t *testing.T) {
	ctx := context.Background()
	run, suite, dataset := testEvalRun("run-1"), multiTurnSuite("suite-1"), testEvalDataset("dataset-1")
	run.Status.Phase = aieval.PhaseRunning
	run.Status.SuiteSpecHash = hashSuiteSpec(suite)
	run.Status.JobRef = v1.LocalObjectReference{Name: "evalrun-run-1"}

	job := buildEvalJob(run, suite, dataset, "evalrun-run-1", suiteConfigMapName(suite.Name, run.Status.SuiteSpecHash))
	job.Status.Conditions = []batchv1.JobCondition{{Type: batchv1.JobFailed, Status: corev1.ConditionTrue, Reason: "BackoffLimitExceeded", Message: "failed"}}

	// The runner exits non-zero for dataset/configuration errors but publishes a precise FAIL gate first.
	published := `{"verdict":"FAIL","summary":"Execution failure: bad dataset",
	  "rootCause":{"stage":"Dataset","caseId":"broken-1","details":"field=turns[0].role"},
	  "artifacts":{"gateSummaryUri":"http://k8s:9876/buckets/eval-artifacts/objects/runs/run-1/gate_summary.json"}}`
	r := newTestReconciler(t, run, suite, dataset, job)
	r.GateFetcher = func(context.Context, *aieval.EvalRun, string) (*aieval.GateSummary, error) {
		var g aieval.GateSummary
		return &g, json.Unmarshal([]byte(published), &g)
	}
	if _, err := r.Reconcile(ctx, reconcileRequest("run-1")); err != nil {
		t.Fatalf("reconcile failed: %v", err)
	}
	var updated aieval.EvalRun
	if err := r.Get(ctx, types.NamespacedName{Name: "run-1", Namespace: "default"}, &updated); err != nil {
		t.Fatal(err)
	}
	if updated.Status.Phase != aieval.PhaseFailed || updated.Status.Verdict != "FAIL" {
		t.Fatalf("expected FAILED/FAIL, got %s/%s", updated.Status.Phase, updated.Status.Verdict)
	}
	if rc := updated.Status.RootCause; rc.Stage != "Dataset" || rc.CaseID != "broken-1" || rc.Details != "field=turns[0].role" {
		t.Fatalf("runner root cause lost: %+v", rc)
	}
	if updated.Status.Artifacts.GateSummaryUri == "" {
		t.Fatalf("artifact URI lost")
	}
}
