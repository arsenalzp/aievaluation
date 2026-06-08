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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Source of dataset
type DatasetObjectSource struct {
	// +kubebuilder:validation:Required
	ObjectStorageEndpoint string `json:"objectStorageEndpoint,omitempty"`
	// +kubebuilder:validation:Required
	Bucket string `json:"bucket,omitempty"`
	// +kubebuilder:validation:Required
	ObjectKey string `json:"objectKey,omitempty"`
	VersionId string `json:"versionId,omitempty"`
}

// EvalDatasetSpec defines the desired state of EvalDataset.
type EvalDatasetSpec struct {
	Source DatasetObjectSource `json:"source,omitempty"`
	Format string              `json:"format,omitempty"`
}

// EvalDatasetStatus defines the observed state of EvalDataset.
type EvalDatasetStatus struct {
	// INSERT ADDITIONAL STATUS FIELD - define observed state of cluster
	// Important: Run "make" to regenerate code after modifying this file
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status

// EvalDataset is the Schema for the evaldatasets API.
type EvalDataset struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   EvalDatasetSpec   `json:"spec,omitempty"`
	Status EvalDatasetStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// EvalDatasetList contains a list of EvalDataset.
type EvalDatasetList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []EvalDataset `json:"items"`
}

func init() {
	SchemeBuilder.Register(&EvalDataset{}, &EvalDatasetList{})
}
