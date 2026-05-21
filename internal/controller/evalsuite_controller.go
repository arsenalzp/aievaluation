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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"time"

	aieval "github.com/arsenalzp/aievaluation/api/v1alpha1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type suiteConfigJSON struct {
	Metrics struct {
		Retrieval struct {
			MinContextRecall    *float32 `json:"minContextRecall,omitempty"`
			MinContextPrecision *float32 `json:"minContextPrecision,omitempty"`
		} `json:"retrieval,omitempty"`
		Generation struct {
			MinFaithfulness *float32 `json:"minFaithfulness,omitempty"`
			MinCorrectness  *float32 `json:"minCorrectness,omitempty"`
		} `json:"generation,omitempty"`
		Trustworthiness struct {
			MaxStaleChunkRatio   *float32 `json:"maxStaleChunkRatio,omitempty"`
			MaxUnknownOwnerRatio *float32 `json:"maxUnknownOwnerRatio,omitempty"`
		} `json:"trustworthiness,omitempty"`
	} `json:"metrics,omitempty"`

	RegressionPolicy struct {
		MaxDrop *float32 `json:"maxDrop,omitempty"`
	} `json:"regressionPolicy,omitempty"`

	Execution struct {
		Mode           *string `json:"mode,omitempty"`
		TimeoutSeconds *int64  `json:"timeoutSeconds,omitempty"`
	} `json:"execution,omitempty"`
}

// EvalSuiteReconciler reconciles a EvalSuite object
type EvalSuiteReconciler struct {
	client.Client
	Scheme              *runtime.Scheme
	RequeueWhileRunning time.Duration
}

// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evalruns,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evalruns/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evalsuites,verbs=get;list;watch
// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evaldatasets,verbs=get;list;watch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
func (r *EvalSuiteReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	l := log.FromContext(ctx).WithValues("evalrun", req.NamespacedName)

	// 1) Fetch EvalRun
	var run aieval.EvalRun
	if err := r.Get(ctx, req.NamespacedName, &run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	// If being deleted, we can optionally do finalization logic. For MVP, do nothing.
	if !run.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// 2) Load EvalSuite (policy/dependency)
	suiteName := run.Spec.SuiteRef
	if suiteName.Name == "" {
		// Mark status and stop requeueing aggressively
		return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "", "SuiteRefMissing",
			"EvalRun.spec.suiteRef is empty", 30*time.Second)
	}

	var suite aieval.EvalSuite
	if err := r.Get(ctx, types.NamespacedName{Name: suiteName.Name, Namespace: run.Namespace}, &suite); err != nil {
		if apierrors.IsNotFound(err) {
			return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "", "SuiteNotFound",
				fmt.Sprintf("EvalSuite %q not found", suiteName), 30*time.Second)
		}
		return ctrl.Result{}, err
	}

	// 2.1) Load EvalDataset referenced by EvalSuite
	if suite.Spec.DatasetRef == nil || suite.Spec.DatasetRef.Name == "" {
		return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "", "DatasetRefMissing",
			"EvalSuite.spec.datasetRef is empty (must reference EvalDataset)", 30*time.Second)
	}

	var dataset aieval.EvalDataset
	if err := r.Get(ctx,
		types.NamespacedName{Name: suite.Spec.DatasetRef.Name, Namespace: run.Namespace},
		&dataset,
	); err != nil {
		if apierrors.IsNotFound(err) {
			return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "", "EvalDatasetNotFound",
				fmt.Sprintf("EvalDataset %q not found", suite.Spec.DatasetRef.Name), 30*time.Second)
		}
		return ctrl.Result{}, err
	}

	// Validate that dataset points to PVC
	if dataset.Spec.Source.SourceRef == nil || dataset.Spec.Source.SourceRef.Name == "" {
		return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "", "DatasetPVCMissing",
			fmt.Sprintf("EvalDataset %q spec.source.sourceRef.name must be PVC name", dataset.Name), 30*time.Second)
	}

	if dataset.Spec.Source.FileName == nil || *dataset.Spec.Source.FileName == "" {
		return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "", "DatasetFileMissing",
			fmt.Sprintf("EvalDataset %q: spec.source.fileName must be file path inside PVC", dataset.Name), 30*time.Second)
	}

	// 3) Snapshot suiteSpecHash into EvalRun.status (prevents policy drift mid-run)
	suiteHash := hashSuiteSpec(&suite)
	if run.Status.SuiteSpecHash == "" {
		patch := client.MergeFrom(run.DeepCopy())
		run.Status.SuiteSpecHash = suiteHash
		if err := r.Status().Patch(ctx, &run, patch); err != nil {
			return ctrl.Result{}, err
		}
		l.Info("snapshotted suite spec hash", "hash", suiteHash)
	} else if run.Status.SuiteSpecHash != suiteHash {
		// MVP: do not change running evaluation; just record a warning condition.
		// You can choose stricter behavior later.
		_ = r.setCondition(ctx, &run, "SuiteChanged", corev1.ConditionTrue,
			"SuiteSpecUpdated", "EvalSuite spec changed after EvalRun started")
	}

	// 4) Ensure Job exists (owned by EvalRun)
	jobName := desiredJobName(&run)
	var job batchv1.Job
	err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: run.Namespace}, &job)
	if apierrors.IsNotFound(err) {

		// Before creating of Job it is necessary to create ConfigMap resource
		// and render EvalSuite resource in that ConfigMap.
		suiteCfgCM, err := r.ensureSuiteConfigMap(ctx, &suite, suiteHash)
		if err != nil {
			return ctrl.Result{}, err
		}

		// Create Job
		newJob := buildEvalJob(&run, &suite, &dataset, jobName, "eval-artifacts-pvc", suiteCfgCM)

		// Set owner reference (this is your preferred pattern)
		if err := controllerutil.SetControllerReference(&run, newJob, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}

		if err := r.Create(ctx, newJob); err != nil {
			return ctrl.Result{}, err
		}

		// Update EvalRun status -> Running + jobRef
		patch := client.MergeFrom(run.DeepCopy())
		run.Status.Phase = aieval.PhaseRunning
		run.Status.JobRef = v1.LocalObjectReference{Name: jobName}
		run.Status.StartedAt = metav1.Now()
		if err := r.Status().Patch(ctx, &run, patch); err != nil {
			return ctrl.Result{}, err
		}

		l.Info("created eval Job", "job", jobName)
		return ctrl.Result{RequeueAfter: r.runningRequeue()}, nil
	}
	if err != nil {
		return ctrl.Result{}, err
	}

	// 5) Observe Job completion/failure
	completed, failed, reason, msg := jobTerminalState(&job)
	if !completed && !failed {
		// Still running
		if run.Status.Phase != aieval.PhaseRunning {
			patch := client.MergeFrom(run.DeepCopy())
			run.Status.Phase = aieval.PhaseRunning
			_ = r.Status().Patch(ctx, &run, patch)
		}
		return ctrl.Result{RequeueAfter: r.runningRequeue()}, nil
	}

	// 6) Finalize: read gate_summary.json (or equivalent) and set EvalRun.status
	// IMPORTANT: Job Succeeded does NOT mean quality PASS; quality verdict is in gate_summary.
	var gate aieval.GateSummary
	if completed {
		// TODO: implement artifact retrieval
		// Example: gate, err = r.fetchGateSummaryFromS3(ctx, &suite, &run)
		// For MVP: you can store summary in a small object store or as a ConfigMap created by the job.
		var err error
		gate, err = r.fetchGateSummary(ctx, &suite, &run)
		if err != nil {
			// If artifact missing, mark infra failure
			return r.setStatusAndRequeue(ctx, &run, aieval.PhaseCompleted, "FAIL", "ArtifactMissing",
				fmt.Sprintf("Job succeeded but gate_summary not found: %v", err), 60*time.Second)
		}
	} else {
		// job failed at infrastructure level
		gate = aieval.GateSummary{
			Verdict: "FAIL",
			Summary: fmt.Sprintf("Execution job failed: %s: %s", reason, msg),
			RootCause: aieval.RootCause{
				Stage: "Execution",
			},
		}
	}

	patch := client.MergeFrom(run.DeepCopy())
	run.Status.Phase = aieval.PhaseCompleted
	run.Status.CompletedAt = metav1.Now()
	run.Status.Verdict = gate.Verdict
	run.Status.Summary = gate.Summary
	run.Status.RootCause = gate.RootCause
	run.Status.Artifacts = gate.Artifacts
	if failed {
		run.Status.ExecutionFailure = &aieval.ExecutionFailure{Reason: reason, Message: msg}
	}
	if err := r.Status().Patch(ctx, &run, patch); err != nil {
		return ctrl.Result{}, err
	}

	l.Info("finalized EvalRun", "verdict", run.Status.Verdict, "stage", run.Status.RootCause.Stage)

	// 7) Optional cleanup (MVP: keep job for debugging; later add TTL / cleanup policy)
	return ctrl.Result{}, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *EvalSuiteReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&aieval.EvalRun{}).
		Owns(&batchv1.Job{}).
		Named("evalrun").
		Complete(r)
}

func (r *EvalSuiteReconciler) setStatusAndRequeue(ctx context.Context, run *aieval.EvalRun, phase aieval.Phase, verdict, condType, msg string, after time.Duration) (ctrl.Result, error) {
	patch := client.MergeFrom(run.DeepCopy())
	run.Status.Phase = phase
	if verdict != "" {
		run.Status.Verdict = verdict
	}
	run.Status.Summary = msg
	if err := r.Status().Patch(ctx, run, patch); err != nil {
		return ctrl.Result{}, err
	}
	_ = r.setCondition(ctx, run, condType, corev1.ConditionTrue, "Error", msg)
	return ctrl.Result{RequeueAfter: after}, nil
}

func (r *EvalSuiteReconciler) setCondition(ctx context.Context, run *aieval.EvalRun, condType string, status corev1.ConditionStatus, reason, msg string) error {
	patch := client.MergeFrom(run.DeepCopy())
	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               condType,
		Status:             metav1.ConditionStatus(status),
		Reason:             reason,
		Message:            msg,
		LastTransitionTime: metav1.Now(),
	})
	return r.Status().Patch(ctx, run, patch)
}

func (r *EvalSuiteReconciler) runningRequeue() time.Duration {
	if r.RequeueWhileRunning <= 0 {
		return 10 * time.Second
	}
	return r.RequeueWhileRunning
}

func jobTerminalState(job *batchv1.Job) (completed bool, failed bool, reason string, message string) {
	for _, c := range job.Status.Conditions {
		if c.Type == batchv1.JobComplete && c.Status == corev1.ConditionTrue {
			return true, false, "Complete", ""
		}
		if c.Type == batchv1.JobFailed && c.Status == corev1.ConditionTrue {
			return false, true, c.Reason, c.Message
		}
	}
	return false, false, "", ""
}

func desiredJobName(run *aieval.EvalRun) string {
	// Deterministic name makes idempotency trivial
	return fmt.Sprintf("evalrun-%s", run.Name)
}

func hashSuiteSpec(suite *aieval.EvalSuite) string {
	// You can hash only the fields you care about; JSON marshal is also fine.
	// Keep it stable.
	h := sha256.Sum256([]byte(fmt.Sprintf("%v", suite.Spec)))
	return hex.EncodeToString(h[:8]) // short hash ok
}

func buildEvalJob(
	run *aieval.EvalRun,
	suite *aieval.EvalSuite,
	dataset *aieval.EvalDataset,
	jobName string,
	artifactsPVC string, // "eval-artifacts-pvc"
	suiteCfgCM string, // ensured CM name
) *batchv1.Job {

	labels := map[string]string{
		"app.kubernetes.io/name":      "eval-gate",
		"ai.evaluation.io/evalrun":    run.Name,
		"ai.evaluation.io/evalsuite":  suite.Name,
		"ai.evaluation.io/suite-hash": run.Status.SuiteSpecHash,
	}

	datasetPVC := dataset.Spec.Source.SourceRef.Name
	datasetFile := *dataset.Spec.Source.FileName

	mountedDatasetFilePath := "/datasets/dataset" // FILE PATH (important)
	runDir := fmt.Sprintf("/artifacts/runs/%s", run.Name)
	baselineFile := fmt.Sprintf("/artifacts/baselines/%s.json", suite.Name)

	backoff := int32(0)

	runnerImage := ptr.Deref(suite.Spec.Execution.RunnerImage, "")

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: run.Namespace,
			Labels:    labels,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoff,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						{
							Name:  "runner",
							Image: runnerImage,
							Env: []corev1.EnvVar{
								{Name: "EVALRUN_NAME", Value: run.Name},
								{Name: "EVALSUITE_NAME", Value: suite.Name},
								{Name: "GIT_SHA", Value: ptr.Deref(run.Spec.Revision.GitSHA, "")},
								{Name: "PR_NUMBER", Value: fmt.Sprintf("%d", ptr.Deref(run.Spec.Revision.PRnumber, 0))},

								{Name: "DATASET_FILE", Value: mountedDatasetFilePath},
								{Name: "ARTIFACT_DIR", Value: "/artifacts"},
								{Name: "RUN_DIR", Value: runDir},
								{Name: "BASELINE_FILE", Value: baselineFile},

								// suite config from ConfigMap as a file
								{Name: "SUITE_CONFIG", Value: "/config/suite.json"},
							},
							VolumeMounts: []corev1.VolumeMount{
								// artifacts directory on shared PVC
								{Name: "artifacts", MountPath: "/artifacts"},

								// dataset single FILE via subPath (file inside PVC)
								{
									Name:      "dataset",
									MountPath: mountedDatasetFilePath,
									SubPath:   datasetFile,
									ReadOnly:  true,
								},

								// suite config
								{Name: "suite-config", MountPath: "/config", ReadOnly: true},
							},
						},
					},
					Volumes: []corev1.Volume{
						{
							Name: "artifacts",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: artifactsPVC,
									ReadOnly:  false,
								},
							},
						},
						{
							Name: "dataset",
							VolumeSource: corev1.VolumeSource{
								PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
									ClaimName: datasetPVC,
									ReadOnly:  true,
								},
							},
						},
						{
							Name: "suite-config",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{Name: suiteCfgCM},
									Items: []corev1.KeyToPath{
										{Key: "suite.json", Path: "suite.json"},
									},
								},
							},
						},
					},
				},
			},
		},
	}
}

// A render from EvalSuite.Spec to suite.json
func renderSuiteConfigJSON(suite *aieval.EvalSuite) ([]byte, error) {
	var cfg suiteConfigJSON

	// Metrics (у тебя они embedded в suite.Spec)
	cfg.Metrics.Retrieval.MinContextRecall = suite.Spec.Metrics.MinContextRecall
	cfg.Metrics.Retrieval.MinContextPrecision = suite.Spec.Metrics.MinContextPrecision

	cfg.Metrics.Generation.MinFaithfulness = suite.Spec.Metrics.MinFaithfulness
	cfg.Metrics.Generation.MinCorrectness = suite.Spec.Metrics.MinCorrectness

	cfg.Metrics.Trustworthiness.MaxStaleChunkRatio = suite.Spec.Metrics.MaxStaleChunkRatio
	cfg.Metrics.Trustworthiness.MaxUnknownOwnerRatio = suite.Spec.Metrics.MaxUnknownOwnerRatio

	// RegressionPolicy
	cfg.RegressionPolicy.MaxDrop = suite.Spec.RegressionPolicy.MaxDrop

	// Execution
	cfg.Execution.Mode = suite.Spec.Execution.Mode
	cfg.Execution.TimeoutSeconds = suite.Spec.Execution.TimeoutSeconds

	return json.MarshalIndent(cfg, "", "  ")
}

func suiteConfigMapName(suiteName, suiteHash string) string {
	short := suiteHash
	if len(short) > 8 {
		short = short[:8]
	}
	return fmt.Sprintf("evalsuite-%s-%s-cfg", suiteName, short)
}

func (r *EvalSuiteReconciler) ensureSuiteConfigMap(ctx context.Context, suite *aieval.EvalSuite, suiteHash string) (string, error) {
	name := suiteConfigMapName(suite.Name, suiteHash)

	payload, err := renderSuiteConfigJSON(suite)
	if err != nil {
		return "", err
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: suite.Namespace,
		},
	}

	_, err = controllerutil.CreateOrUpdate(ctx, r.Client, cm, func() error {
		if cm.Data == nil {
			cm.Data = map[string]string{}
		}
		cm.Data["suite.json"] = string(payload)

		if cm.Labels == nil {
			cm.Labels = map[string]string{}
		}
		cm.Labels["ai.evaluation.io/evalsuite"] = suite.Name
		cm.Labels["ai.evaluation.io/suite-hash"] = suiteHash

		return nil
	})
	if err != nil {
		return "", err
	}
	return name, nil
}

// fetchGateSummary is intentionally a placeholder.
// In real MVP you can:
//   - read a ConfigMap created by job,
//   - read from MinIO/S3,
//   - or parse from a known artifact file on a shared PVC.
//
// Keep operator responsible for status updates (runner pod stays low-privileged).
func (r *EvalSuiteReconciler) fetchGateSummary(ctx context.Context, suite *aieval.EvalSuite, run *aieval.EvalRun) (aieval.GateSummary, error) {
	return aieval.GateSummary{}, fmt.Errorf("not implemented")
}
