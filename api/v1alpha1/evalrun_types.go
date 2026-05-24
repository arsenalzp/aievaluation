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

package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	PhaseCompleted = "COMPLETED"
	PhaseFailed    = "FAILED"
	PhaseRunning   = "RUNNING"
	PhasePending   = "PENDING"
)

type Phase = string

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.
type Revision struct {
	GitSHA   *string `json:"gitSHA,omitempty"`
	PRnumber *int32  `json:"prNumber,omitempty"`
}

type RootCause struct {
	Stage string `json:"stage,omitempty"`
}

// EvalRunSpec defines the desired state of EvalRun.
type EvalRunSpec struct {
	// +kubebuilder:validation:Required
	SuiteRef *v1.LocalObjectReference `json:"suiteRef,omitempty"`
	// +kubebuilder:validation:Required
	Revision `json:"revision,omitempty"`
	// +kubebuilder:validation:Required
	SummaryBucket string `json:"summaryBucket,omitempty"`
	// RAG endpoint
	// +kubebuilder:validation:Required
	TargetEndpoint string `json:"targetEndpoint,omitempty"`
}

type ExecutionFailure struct {
	Reason  string `json:"reason,omitempty"`
	Message string `json:"message,omitempty"`
}

type GateSummary struct {
	Verdict   string    `json:"verdict,omitempty"`
	Summary   string    `json:"summary,omitempty"`
	RootCause RootCause `json:"rootCause,omitempty"`
	Artifacts string    `json:"artifacts,omitempty"`
}

// EvalRunStatus defines the observed state of EvalRun.
type EvalRunStatus struct {
	Phase            `json:"phase,omitempty"`
	Conditions       []metav1.Condition      `json:"conditions,omitempty"`
	Verdict          string                  `json:"verdict,omitempty"`
	Summary          string                  `json:"summary,omitempty"`
	StartedAt        metav1.Time             `json:"startedAt,omitempty"`
	CompletedAt      metav1.Time             `json:"completedAt,omitempty"`
	JobRef           v1.LocalObjectReference `json:"jobRef,omitempty"`
	SuiteSpecHash    string                  `json:"suiteSpecHash,omitempty"`
	RootCause        `json:"rootCause,omitempty"`
	Artifacts        string            `json:"artifacts,omitempty"`
	ExecutionFailure *ExecutionFailure `json:"executionFailure,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// EvalRun is the Schema for the evalruns API.
type EvalRun struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   EvalRunSpec   `json:"spec,omitempty"`
	Status EvalRunStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// EvalRunList contains a list of EvalRun.
type EvalRunList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []EvalRun `json:"items"`
}

func init() {
	SchemeBuilder.Register(&EvalRun{}, &EvalRunList{})
}
