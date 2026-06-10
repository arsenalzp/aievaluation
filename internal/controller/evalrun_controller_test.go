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

	metrics := decoded["metrics"].(map[string]any)
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
