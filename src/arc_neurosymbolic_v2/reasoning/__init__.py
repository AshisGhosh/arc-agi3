"""
Reasoning Engine for Neurosymbolic v2

Components:
- Goal Inference: Discover goals from observation
- Hidden State Detection: Infer unobservable variables
- Causal Rule Induction: Distinguish correlation from causation
- Belief State: Track uncertainty over hidden variables
"""

from .causal_induction import (
    CausalHypothesis,
    CausalRuleInductor,
    CounterfactualReasoner,
    InterventionExperiment,
    InterventionResult,
)
from .goal_inference import (
    GoalFeatureExtractor,
    GoalHypothesis,
    GoalInferenceModule,
    GoalProximityPredictor,
)
from .hidden_state import (
    BeliefState,
    CooldownHypothesis,
    CounterHypothesis,
    HiddenStateDetector,
    HiddenVariableHypothesis,
    PredictionDiscrepancy,
    StateMachineHypothesis,
)

__all__ = [
    # Goal Inference
    "GoalInferenceModule",
    "GoalHypothesis",
    "GoalFeatureExtractor",
    "GoalProximityPredictor",
    # Hidden State
    "HiddenStateDetector",
    "HiddenVariableHypothesis",
    "CounterHypothesis",
    "CooldownHypothesis",
    "StateMachineHypothesis",
    "BeliefState",
    "PredictionDiscrepancy",
    # Causal Induction
    "CausalRuleInductor",
    "CausalHypothesis",
    "InterventionExperiment",
    "InterventionResult",
    "CounterfactualReasoner",
]
