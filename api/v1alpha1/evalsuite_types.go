package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.
type Trustworthiness struct {
	// +kubebuilder:validation:Optional
	MaxStaleChunkRatio *float32 `json:"maxStaleChunkRatio,omitempty"`
	// +kubebuilder:validation:Optional
	MaxUnknownOwnerRatio *float32 `json:"maxUnknownOwnerRatio,omitempty"`
}

type Generation struct {
	// +kubebuilder:validation:Optional
	MinFaithfulness *float32 `json:"minFaithfulness,omitempty"`
	// +kubebuilder:validation:Optional
	MinCorrectness *float32 `json:"minCorrectness,omitempty"`
}

type Retrieval struct {
	// +kubebuilder:validation:Optional
	MinContextRecall *float32 `json:"minContextRecall,omitempty"`
	// +kubebuilder:validation:Optional
	MinContextPrecision *float32 `json:"minContextPrecision,omitempty"`
}

type Agent struct {
	MinToolCorrectness     *float32 `json:"minToolCorrectness,omitempty"`
	MinTaskCompletion      *float32 `json:"minTaskCompletion,omitempty"`
	MinArgumentCorrectness *float32 `json:"minArgumentCorrectness,omitempty"`
	MaxPolicyViolations    *float32 `json:"MaxPolicyViolations,omitempty"`
}

type Rag struct {
	Retrieval       `json:"retrieval,omitempty"`
	Generation      `json:"generation,omitempty"`
	Trustworthiness `json:"trustworthiness,omitempty"`
}

type Metrics struct {
	// +kubebuilder:validation:Optional
	Agent `json:"agent,omitempty"`
	// +kubebuilder:validation:Optional
	Rag `json:"rag,omitempty"`
}

type RegressionPolicy struct {
	// +kubebuilder:validation:Optional
	MaxDrop *float32 `json:"maxDrop,omitempty"`
}

type Execution struct {
	// +kubebuilder:validation:Enum=agent;rag;dummy
	Mode           string `json:"mode,omitempty"`
	RunnerImage    string `json:"runnerImage,omitempty"`
	TimeoutSeconds *int64 `json:"timeoutSeconds,omitempty"`
}

// EvalSuiteSpec defines the desired state of EvalSuite.
type EvalSuiteSpec struct {
	// DatasetRef references EvalDataset resource (not ConfigMap).
	DatasetRef       *v1.LocalObjectReference `json:"datasetRef,omitempty"`
	Metrics          `json:"metrics,omitempty"`
	RegressionPolicy `json:"regressionPolicy,omitempty"`
	Execution        `json:"execution,omitempty"`
}

// EvalSuiteStatus defines the observed state of EvalSuite.
type EvalSuiteStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// EvalSuite is the Schema for the evalsuites API.
type EvalSuite struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   EvalSuiteSpec   `json:"spec,omitempty"`
	Status EvalSuiteStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// EvalSuiteList contains a list of EvalSuite.
type EvalSuiteList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []EvalSuite `json:"items"`
}

func init() {
	SchemeBuilder.Register(&EvalSuite{}, &EvalSuiteList{})
}
