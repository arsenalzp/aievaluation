package v1alpha1

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// EDIT THIS FILE!  THIS IS SCAFFOLDING FOR YOU TO OWN!
// NOTE: json tags are required.  Any new fields you add must have json tags for the fields to be serialized.
type Generation struct {
	// +kubebuilder:validation:Optional
	MinFaithfulness *float32 `json:"minFaithfulness,omitempty"`

	// +kubebuilder:validation:Optional
	MinCorrectness *float32 `json:"minCorrectness,omitempty"`

	// +kubebuilder:validation:Optional
	MinRelevancy *float32 `json:"minRelevancy,omitempty"`
}

type Retrieval struct {
	// +kubebuilder:validation:Optional
	MinContextRecall *float32 `json:"minContextRecall,omitempty"`

	// +kubebuilder:validation:Optional
	MinContextPrecision *float32 `json:"minContextPrecision,omitempty"`
}

type Agent struct {
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinToolSelection *float32 `json:"minToolSelection,omitempty"`

	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinToolInvocation *float32 `json:"minToolInvocation,omitempty"`

	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinToolExecution *float32 `json:"minToolExecution,omitempty"`

	// Minimum average tool response handling of single-turn agent cases (Phoenix ToolResponseHandlingEvaluator,
	// LLM judge): is the final answer grounded in what the tools returned (no invented or misreported data,
	// tool errors handled)? Needs tool outputs in the trace. Cases without tool calls/outputs are not scored.
	// Single-turn only.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinToolResponseHandling *float32 `json:"minToolResponseHandling,omitempty"`

	// Minimum average task completion of single-turn agent cases: did the user get the expected result?
	// A Phoenix LLM judge (custom ClassificationEvaluator) compares the final answer with the case's expectedOutput;
	// a case with expectedOutputPattern is checked with a regular expression instead (no LLM).
	// Cases without either are not scored.
	// Single-turn only; multi-turn suites use outcome.minGoalCompletion.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinTaskCompletion *float32 `json:"minTaskCompletion,omitempty"`

	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	MaxPolicyViolations *int32 `json:"maxPolicyViolations,omitempty"`

	// Outcome, Conversation, Tools and Policy configure multi-turn evaluation
	// (spec.evaluation.interactionMode: MultiTurn). They are additive: the flat fields above keep working.

	// +kubebuilder:validation:Optional
	Outcome *AgentOutcome `json:"outcome,omitempty"`
	// +kubebuilder:validation:Optional
	Conversation *AgentConversation `json:"conversation,omitempty"`
	// +kubebuilder:validation:Optional
	Tools *AgentTools `json:"tools,omitempty"`
	// +kubebuilder:validation:Optional
	Policy *AgentPolicy `json:"policy,omitempty"`
}

// AgentOutcome holds goal-level thresholds.
type AgentOutcome struct {
	// Minimum share of cases (every trial) that complete without blocking violations.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinGoalCompletion *float32 `json:"minGoalCompletion,omitempty"`
}

// AgentConversation holds conversation-level limits and semantic (LLM-judged) thresholds.
type AgentConversation struct {
	// Maximum number of user turns per session. The runner stops at the limit and reports MAX_TURNS_EXCEEDED.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	MaxTurns *int32 `json:"maxTurns,omitempty"`

	// Semantic thresholds. Each one that is set is computed by the (single) provider deepeval evaluator in
	// spec.evaluators, which is then mandatory; setting none of them while a deepeval evaluator exists is an error.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinKnowledgeRetention *float32 `json:"minKnowledgeRetention,omitempty"`
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinConversationCompleteness *float32 `json:"minConversationCompleteness,omitempty"`
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinTurnRelevancy *float32 `json:"minTurnRelevancy,omitempty"`
}

// AgentTools holds deterministic tool-use thresholds and limits for a whole session.
type AgentTools struct {
	// Share of required tools that were called (deterministic).
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinSelectionAccuracy *float32 `json:"minSelectionAccuracy,omitempty"`
	// Share of expected tool calls whose (partial) expected arguments matched.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	MinArgumentAccuracy *float32 `json:"minArgumentAccuracy,omitempty"`
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	MaxToolCalls *int32 `json:"maxToolCalls,omitempty"`
	// Maximum number of identical calls (same tool and canonical JSON arguments) per session.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	MaxRepeatedCalls *int32 `json:"maxRepeatedCalls,omitempty"`
}

// AgentPolicy configures the zero-tolerance policy gate.
type AgentPolicy struct {
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=0
	MaxCriticalViolations *int32 `json:"maxCriticalViolations,omitempty"`
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:default=0
	MaxHighViolations *int32 `json:"maxHighViolations,omitempty"`
	// What happens when maxTurns / maxToolCalls / maxRepeatedCalls are exceeded:
	// Fail records a HIGH violation, Warn a MEDIUM one (WARN verdict).
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Enum=Fail;Warn
	// +kubebuilder:default=Fail
	LimitsAction string `json:"limitsAction,omitempty"`
}

type Rag struct {
	// +kubebuilder:validation:Optional
	Retrieval `json:"retrieval,omitempty"`

	// +kubebuilder:validation:Optional
	Generation `json:"generation,omitempty"`
}

type Metrics struct {
	// +kubebuilder:validation:Optional
	*Agent `json:"agent,omitempty"`
	// +kubebuilder:validation:Optional
	*Rag `json:"rag,omitempty"`
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

// PhoenixTrace locates the Phoenix server that stores the agent's OpenInference traces.
type PhoenixTrace struct {
	// Phoenix base URL, e.g. http://phoenix.observability.svc:6006 (no authentication support yet).
	// +kubebuilder:validation:Optional
	Endpoint string `json:"endpoint,omitempty"`
	// Phoenix project name holding the agent's spans.
	// +kubebuilder:validation:Optional
	// +kubebuilder:default=default
	Project string `json:"project,omitempty"`
}

// TraceSpec selects where tool-call observations come from.
type TraceSpec struct {
	// inline: tool calls reported in the agent response (default).
	// phoenix: Phoenix/OpenInference spans only.
	// auto: Phoenix when spans exist for the turn's trace, otherwise inline; conflicts are reported, never merged.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Enum=inline;phoenix;auto
	// +kubebuilder:default=inline
	Source string `json:"source,omitempty"`
	// +kubebuilder:validation:Optional
	Phoenix *PhoenixTrace `json:"phoenix,omitempty"`
}

// Evaluation selects how interactions are driven. The task type itself (agent / rag) is defined by
// spec.execution.mode, or inferred from spec.metrics when the mode is not set.
type Evaluation struct {
	// SingleTurn (default) keeps the existing one-request-per-case behaviour.
	// MultiTurn replays scripted multi-turn conversations in a session.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Enum=SingleTurn;MultiTurn
	// +kubebuilder:default=SingleTurn
	InteractionMode string `json:"interactionMode,omitempty"`
	// Number of times every scripted conversation is replayed; a case passes only if all trials pass (pass^k).
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=10
	Trials *int32 `json:"trials,omitempty"`
	// +kubebuilder:validation:Optional
	Trace *TraceSpec `json:"trace,omitempty"`
}

// Evaluator configures *how* one evaluator of a multi-turn evaluation runs. *What* is evaluated and the
// thresholds live only in spec.metrics.agent: a provider deepeval evaluator computes exactly the semantic
// metrics that have a threshold in spec.metrics.agent.conversation.
// The built-in deterministic evaluator always runs and is always required; at most one evaluator per provider.
type Evaluator struct {
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Enum=builtin;deepeval
	// +kubebuilder:default=builtin
	Provider string `json:"provider,omitempty"`
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Enum=deterministic;multiTurn
	Type string `json:"type,omitempty"`
	// Provider deepeval: when true, an evaluator failure or a semantic metric below its threshold fails the run;
	// when false or omitted, it turns the verdict into WARN at most.
	// Provider builtin: always required; only omitted or true is accepted.
	// +kubebuilder:validation:Optional
	Required *bool `json:"required,omitempty"`
	// Judge model (provider deepeval only); defaults to the EvalRun model.
	// +kubebuilder:validation:Optional
	Model string `json:"model,omitempty"`
	// Judge runs per conversation; the median is used (provider deepeval only).
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=5
	Samples *int32 `json:"samples,omitempty"`
}

// EvalSuiteSpec defines the desired state of EvalSuite.
type EvalSuiteSpec struct {
	// DatasetRef references EvalDataset resource (not ConfigMap).
	DatasetRef       *v1.LocalObjectReference `json:"datasetRef,omitempty"`
	Metrics          `json:"metrics,omitempty"`
	RegressionPolicy `json:"regressionPolicy,omitempty"`
	Execution        `json:"execution,omitempty"`

	// Evaluation is optional; omitting it keeps the single-turn behaviour of existing suites.
	// +kubebuilder:validation:Optional
	Evaluation *Evaluation `json:"evaluation,omitempty"`
	// Evaluators is only used by multi-turn evaluation.
	// +kubebuilder:validation:Optional
	Evaluators []Evaluator `json:"evaluators,omitempty"`
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
