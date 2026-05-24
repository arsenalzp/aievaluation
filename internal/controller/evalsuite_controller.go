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
	objectstorage "github.com/metorial/object-storage/clients/go"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	meta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	ptr "k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

const (
	artifactsPVCName = "eval-artifacts-pvc"
	artifactsMount   = "/artifacts"
	maxRetries       = 3
)

type gateSummaryJSON struct {
	Verdict   string `json:"verdict"`
	Summary   string `json:"summary"`
	RootCause struct {
		Stage   string `json:"stage,omitempty"`
		Metric  string `json:"metric,omitempty"`
		Details string `json:"details,omitempty"`
	} `json:"rootCause,omitempty"`
	Artifacts any `json:"artifacts,omitempty"`
}

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
		Mode           string `json:"mode,omitempty"`
		TimeoutSeconds *int64 `json:"timeoutSeconds,omitempty"`
	} `json:"execution,omitempty"`
}

// EvalSuiteReconciler reconciles a EvalSuite object
type EvalSuiteReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	Objectstorage *objectstorage.Client
	ObjectBuket   string
	Queue         workqueue.TypedRateLimitingInterface[string]
}

// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evalruns,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evalruns/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evalsuites,verbs=get;list;watch
// +kubebuilder:rbac:groups=ai.evaluation.io,resources=evaldatasets,verbs=get;list;watch
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=persistentvolumeclaims,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups="",resources=pods/log,verbs=get
func (r *EvalSuiteReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	l := log.FromContext(ctx).WithValues("evalrun", req.NamespacedName)

	// 1) Fetch EvalRun
	var run aieval.EvalRun
	if err := r.Get(ctx, req.NamespacedName, &run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !run.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	// 2) Load EvalSuite (policy/dependency)
	suiteName := run.Spec.SuiteRef
	if suiteName.Name == "" {
		return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "SuiteRefMissing", "Pending", "EvalRun.spec.suiteRef is empty", 30*time.Second)
	}

	var suite aieval.EvalSuite
	if err := r.Get(ctx, types.NamespacedName{Name: suiteName.Name, Namespace: run.Namespace}, &suite); err != nil {
		if apierrors.IsNotFound(err) {
			return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "SuiteNotFound", "Pending", fmt.Sprintf("EvalSuite %q not found", suiteName), 30*time.Second)
		}
		return ctrl.Result{}, err
	}

	// 2.1) Load EvalDataset referenced by EvalSuite
	if suite.Spec.DatasetRef == nil || suite.Spec.DatasetRef.Name == "" {
		return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "DatasetRefMissing", "Pending", "EvalSuite.spec.datasetRef is empty (must reference EvalDataset)", 30*time.Second)
	}

	var dataset aieval.EvalDataset
	if err := r.Get(ctx, types.NamespacedName{Name: suite.Spec.DatasetRef.Name, Namespace: run.Namespace}, &dataset); err != nil {
		if apierrors.IsNotFound(err) {
			return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "EvalDatasetNotFound", "Pending", fmt.Sprintf("EvalDataset %q not found", suite.Spec.DatasetRef.Name), 30*time.Second)
		}
		return ctrl.Result{}, err
	}

	if dataset.Spec.Source.ObjectKey == "" || dataset.Spec.Source.ObjectKey == "" {
		return r.setStatusAndRequeue(ctx, &run, aieval.PhasePending, "DatasetObjectKeyMissing", "Pending", fmt.Sprintf("EvalDataset %s: spec.source.objectKey must be set", dataset.Name), 30*time.Second)
	}

	// 3) Snapshot suiteSpecHash (Skipped comment block for brevity)
	suiteHash := hashSuiteSpec(&suite)
	if run.Status.SuiteSpecHash == "" {
		patch := client.MergeFrom(run.DeepCopy())
		run.Status.SuiteSpecHash = suiteHash
		if err := r.Status().Patch(ctx, &run, patch); err != nil {
			return ctrl.Result{}, err
		}
		l.Info("snapshotted suite spec hash", "hash", suiteHash)
	} else if run.Status.SuiteSpecHash != suiteHash {
		// TODO: do not change running evaluation; just record a warning condition.
		_ = r.setCondition(ctx, &run, "SuiteChanged", metav1.ConditionTrue,
			"SuiteSpecUpdated", "EvalSuite spec changed after EvalRun started")
	}

	// 4) Ensure Job exists
	jobName := desiredJobName(&run)
	var job batchv1.Job
	err := r.Get(ctx, types.NamespacedName{Name: jobName, Namespace: run.Namespace}, &job)
	if apierrors.IsNotFound(err) {
		suiteCfgCM, err := r.ensureSuiteConfigMap(ctx, &suite, suiteHash)
		if err != nil {
			return ctrl.Result{}, err
		}

		newJob := buildEvalJob(&run, &suite, &dataset, jobName, suiteCfgCM)
		if err := controllerutil.SetControllerReference(&run, newJob, r.Scheme); err != nil {
			return ctrl.Result{}, err
		}

		if err := r.Create(ctx, newJob); err != nil {
			return ctrl.Result{}, err
		}

		patch := client.MergeFrom(run.DeepCopy())
		run.Status.Phase = aieval.PhaseRunning
		run.Status.JobRef = v1.LocalObjectReference{Name: jobName}
		run.Status.StartedAt = metav1.Now()
		if err := r.Status().Patch(ctx, &run, patch); err != nil {
			return ctrl.Result{}, err
		}

		l.Info("created eval Job", "job", jobName)
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}
	if err != nil {
		return ctrl.Result{}, err
	}

	// 5) Observe Job completion/failure
	completed, failed, reason, msg := jobTerminalState(&job)
	if !completed && !failed {
		if run.Status.Phase != aieval.PhaseRunning {
			patch := client.MergeFrom(run.DeepCopy())
			run.Status.Phase = aieval.PhaseRunning
			_ = r.Status().Patch(ctx, &run, patch)
		}
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}

	// 6) Finalize
	if completed {
		if r.Queue.NumRequeues(run.Name) >= maxRetries && run.Status.Phase == aieval.PhaseRunning {
			run.Status.Phase = aieval.PhaseFailed
			run.Status.Verdict = "FAIL"
			r.Queue.Forget(run.Name)
			l.Info("finalized EvalRun due to max retries", "verdict", run.Status.Verdict)
			err := r.setCondition(ctx, &run, "ArtifactMissing", metav1.ConditionTrue, "MaximumRetriesReached", "Unable to get gate_summary")
			if err != nil {
				return ctrl.Result{}, reconcile.TerminalError(err)
			}
			return ctrl.Result{}, nil
		}

		if run.Status.Phase == aieval.PhaseRunning {
			gate, err := r.fetchGateSummary(ctx, &suite, &run)
			if err != nil {
				r.Queue.AddRateLimited(run.Name)
				return r.setStatusAndRequeue(ctx, &run, aieval.PhaseRunning, "ArtifactMissing", "Running", fmt.Sprintf("Job succeeded but gate_summary not found: %v", err), 10*time.Second)
			}

			// Successfully completed
			if err := r.completeEvalRun(ctx, &run, aieval.PhaseCompleted, gate); err != nil {
				return ctrl.Result{}, err
			}
			l.Info("finalized EvalRun", "verdict", run.Status.Verdict, "stage", run.Status.RootCause.Stage)
			r.Queue.Forget(run.Name)
		}
	} else {
		// Job failed
		gate := &aieval.GateSummary{
			Verdict: "FAIL",
			Summary: fmt.Sprintf("Execution job failed: %s: %s", reason, msg),
			RootCause: aieval.RootCause{
				Stage: "Execution",
			},
		}
		if err := r.completeEvalRun(ctx, &run, aieval.PhaseFailed, gate); err != nil {
			return ctrl.Result{}, err
		}
	}

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

func (r *EvalSuiteReconciler) completeEvalRun(ctx context.Context, run *aieval.EvalRun, phase aieval.Phase, gate *aieval.GateSummary) error {
	run.Status.Phase = phase
	run.Status.CompletedAt = metav1.Now()
	run.Status.Verdict = gate.Verdict
	run.Status.Summary = gate.Summary
	run.Status.RootCause = gate.RootCause
	run.Status.Artifacts = gate.Artifacts
	return r.Status().Update(ctx, run)
}

func (r *EvalSuiteReconciler) setStatusAndRequeue(ctx context.Context, run *aieval.EvalRun, phase aieval.Phase, condType, reason, msg string, after time.Duration) (ctrl.Result, error) {
	patch := client.MergeFrom(run.DeepCopy())
	run.Status.Phase = phase
	run.Status.Summary = msg
	if err := r.Status().Patch(ctx, run, patch); err != nil {
		return ctrl.Result{}, err
	}

	if condType != "" {
		_ = r.setCondition(ctx, run, condType, metav1.ConditionTrue, reason, msg)
	}
	return ctrl.Result{RequeueAfter: after}, nil
}

func (r *EvalSuiteReconciler) setCondition(ctx context.Context, run *aieval.EvalRun, condType string, status metav1.ConditionStatus, reason, msg string) error {
	meta.SetStatusCondition(&run.Status.Conditions, metav1.Condition{
		Type:               condType,
		Status:             status,
		Reason:             reason,
		Message:            msg,
		LastTransitionTime: metav1.Now(),
	})
	return r.Status().Update(ctx, run)
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
	data, _ := json.Marshal(suite.Spec)
	h := sha256.Sum256([]byte(data))
	return hex.EncodeToString(h[:8]) // short hash ok
}

func buildEvalJob(
	run *aieval.EvalRun,
	suite *aieval.EvalSuite,
	dataset *aieval.EvalDataset,
	jobName string,
	suiteCfgCM string,
) *batchv1.Job {
	labels := map[string]string{
		"app.kubernetes.io/name":      "eval-gate",
		"ai.evaluation.io/evalrun":    run.Name,
		"ai.evaluation.io/evalsuite":  suite.Name,
		"ai.evaluation.io/suite-hash": run.Status.SuiteSpecHash,
	}

	backoff := int32(0)
	runnerImage := suite.Spec.Execution.RunnerImage

	runPrefix := fmt.Sprintf("runs/%s", run.Name)
	baselineKey := fmt.Sprintf("baselines/%s.json", suite.Name)

	datasetBucket := ""
	if dataset.Spec.Source.Bucket != "" {
		datasetBucket = dataset.Spec.Source.Bucket
	}

	datasetObjectKey := dataset.Spec.Source.ObjectKey
	datasetVersionID := dataset.Spec.Source.VersionId

	env := []corev1.EnvVar{
		{Name: "EVALRUN_NAME", Value: run.Name},
		{Name: "EVALSUITE_NAME", Value: suite.Name},
		{Name: "GIT_SHA", Value: ptr.Deref(run.Spec.Revision.GitSHA, "")},
		{Name: "PR_NUMBER", Value: fmt.Sprintf("%d", ptr.Deref(run.Spec.Revision.PRnumber, 0))},

		// local paths
		{Name: "WORK_DIR", Value: "/work"},
		{Name: "DATASET_FILE", Value: "/work/dataset"},
		{Name: "SUITE_CONFIG", Value: "/config/suite.json"},

		// dataset location in S3
		{Name: "DATASET_S3_BUCKET", Value: datasetBucket},
		{Name: "DATASET_S3_KEY", Value: datasetObjectKey},
		{Name: "DATASET_S3_VERSION_ID", Value: datasetVersionID},
		{Name: "DATASET_FORMAT", Value: dataset.Spec.Format},
		{Name: "S3_ENDPOINT", Value: dataset.Spec.Source.ObjectStorageEndpoint},

		// artifacts / baseline in S3
		{Name: "S3_RUN_PREFIX", Value: runPrefix},
		{Name: "S3_BASELINE_KEY", Value: baselineKey},
		{Name: "GATE_SUMMARY_KEY", Value: fmt.Sprintf("%s/gate_summary.json", runPrefix)},
		{Name: "SCORES_KEY", Value: fmt.Sprintf("%s/scores.json", runPrefix)},
		{Name: "DETAILS_KEY", Value: fmt.Sprintf("%s/details.jsonl", runPrefix)},
		{Name: "S3_SUMMARY_BUCKET", Value: run.Spec.SummaryBucket},
		{Name: "S3_SUMMARY_KEY", Value: fmt.Sprintf("runs/%s/gate_summary.json", run.Name)},
	}

	// shared S3 credentials/config from Secret
	env = append(env,
		corev1.EnvVar{
			Name: "S3_ACCESS_KEY",
			ValueFrom: &corev1.EnvVarSource{
				SecretKeyRef: &corev1.SecretKeySelector{
					LocalObjectReference: corev1.LocalObjectReference{Name: "eval-s3-credentials"},
					Key:                  "accessKey",
				},
			},
		},
		corev1.EnvVar{
			Name: "S3_SECRET_KEY",
			ValueFrom: &corev1.EnvVarSource{
				SecretKeyRef: &corev1.SecretKeySelector{
					LocalObjectReference: corev1.LocalObjectReference{Name: "eval-s3-credentials"},
					Key:                  "secretKey",
				},
			},
		},
	)

	container := corev1.Container{
		Name:  "runner",
		Image: runnerImage,
		Env:   env,
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      "workdir",
				MountPath: "/work",
			},
			{
				Name:      "suite-config",
				MountPath: "/config",
				ReadOnly:  true,
			},
		},
	}

	mode := ""
	if suite.Spec.Execution.Mode != "" {
		mode = suite.Spec.Execution.Mode
	}

	if mode == "dummy" {
		container.Image = "nginx:stable-alpine3.23-perl"
		container.Command = []string{"sh", "-c"}
		container.Args = []string{
			`set -e
			echo "creating fake gate_summary in object bucket"
			curl -H "Content-Type: application/json" -X PUT http://k8s:9876/buckets/${SUMMARY_BUCKET}/objects/${SUMMARY_KEY} --data '{
"verdict": "PASS",
"summary": "Fake gate_summary uploaded successfully for integration test",
"rootCause": {
  "stage": "None"
},
"artifacts": "{\"gateSummaryUri\":\"s3://${SUMMARY_BUCKET}/${SUMMARY_KEY}\",\"scoresUri\":\"s3://${SUMMARY_BUCKET}/${SCORES_KEY}\"}"
}'`,
		}
	}

	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: run.Namespace,
			Labels:    labels,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoff,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{
						container,
					},
					Volumes: []corev1.Volume{
						{
							Name: "workdir",
							VolumeSource: corev1.VolumeSource{
								EmptyDir: &corev1.EmptyDirVolumeSource{},
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

	// Metrics
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
	// TODO: rewrite to either delete CM when a Job finished or create CM with a name related to Job name
	// to avoid multiple CM objects in ETCD
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

func (r *EvalSuiteReconciler) buildGateReaderPod(run *aieval.EvalRun) *corev1.Pod {
	podName := fmt.Sprintf("gate-reader-%s", run.Name)

	gatePath := fmt.Sprintf("%s/runs/%s/gate_summary.json", artifactsMount, run.Name)
	cmd := fmt.Sprintf("set -e; cat %s", gatePath)

	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: run.Namespace,
			Labels: map[string]string{
				"app.kubernetes.io/name":   "eval-gate-reader",
				"ai.evaluation.io/evalrun": run.Name,
			},
		},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{
				{
					Name:    "reader",
					Image:   "alpine:3",
					Command: []string{"sh", "-c", cmd},
					VolumeMounts: []corev1.VolumeMount{
						{Name: "artifacts", MountPath: artifactsMount, ReadOnly: true},
					},
				},
			},
			Volumes: []corev1.Volume{
				{
					Name: "artifacts",
					VolumeSource: corev1.VolumeSource{
						PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
							ClaimName: artifactsPVCName,
							ReadOnly:  true,
						},
					},
				},
			},
		},
	}
}

func (r *EvalSuiteReconciler) fetchGateSummary(ctx context.Context, suite *aieval.EvalSuite, run *aieval.EvalRun) (*aieval.GateSummary, error) {
	key := fmt.Sprintf("runs/%s/gate_summary.json", run.Name)
	obj, err := r.Objectstorage.GetObject(run.Spec.SummaryBucket, key)
	if err != nil {
		return nil, fmt.Errorf("unable to get gate_summary object: %s", err)
	}

	var gate aieval.GateSummary
	if err := json.Unmarshal(obj.Data, &gate); err != nil {
		return nil, fmt.Errorf("failed to parse gate_summary json: %w; raw=%s", err, string(obj.Data))
	}

	return &gate, nil
}
