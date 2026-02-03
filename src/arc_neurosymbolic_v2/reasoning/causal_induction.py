"""
Causal Rule Induction Module

Distinguish correlation from causation using:
- Intervention-based testing (do-calculus principles)
- Controlled experiments via targeted actions
- Counterfactual reasoning
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Set,
)

if TYPE_CHECKING:
    from arcengine import GameAction

    from ..perception.symbolic_state import SymbolicState


@dataclass
class Transition:
    """Represents a state transition."""

    prev_state: SymbolicState
    action: GameAction
    next_state: SymbolicState
    effects: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class CausalHypothesis:
    """
    A hypothesis that action A causes effect E under conditions C.

    Validated through intervention testing.
    """

    cause: str  # The action/event
    effect: str  # The observed effect
    conditions: FrozenSet[str]  # Required conditions
    confidence: float  # 0.0 to 1.0
    interventions_run: int  # Number of controlled tests
    support_count: int  # Observations supporting
    refute_count: int  # Observations refuting

    @property
    def is_confident(self) -> bool:
        """Check if hypothesis is confidently confirmed."""
        return self.confidence >= 0.8 and self.interventions_run >= 5


@dataclass
class InterventionExperiment:
    """A controlled experiment to test a causal hypothesis."""

    hypothesis_description: str
    intervention_action: str  # Action to test
    target_effect: str  # Effect we're checking for
    required_conditions: List[str]  # Context conditions
    control_action: str = "NO_OP"  # Alternative action for control
    n_trials: int = 10  # Trials per condition


@dataclass
class InterventionResult:
    """Result of running an intervention experiment."""

    experiment: InterventionExperiment
    intervention_with_effect: int = 0
    intervention_without_effect: int = 0
    control_with_effect: int = 0
    control_without_effect: int = 0

    @property
    def causal_effect(self) -> float:
        """
        Compute causal effect: P(E | do(A)) - P(E | do(not A))

        Positive = A causes E
        Zero = no causal relationship
        Negative = A prevents E
        """
        n_intervention = self.intervention_with_effect + self.intervention_without_effect
        n_control = self.control_with_effect + self.control_without_effect

        if n_intervention == 0 or n_control == 0:
            return 0.0

        p_effect_given_intervention = self.intervention_with_effect / n_intervention
        p_effect_given_control = self.control_with_effect / n_control

        return p_effect_given_intervention - p_effect_given_control


class EnvironmentInterface(Protocol):
    """Protocol for environment interaction during experiments."""

    def reset_to_controlled_state(self) -> SymbolicState:
        """Reset environment to a controlled starting state."""
        ...

    def step(self, action: GameAction) -> SymbolicState:
        """Execute action and return next state."""
        ...


class CausalRuleInductor:
    """
    Distinguish correlation from causation using intervention-based testing.

    Workflow:
    1. Observe transitions and track correlations
    2. Generate intervention experiments for strong correlations
    3. Run controlled experiments to test causation
    4. Build confirmed causal rules
    """

    def __init__(self):
        self.observations: List[Transition] = []
        # Track correlations: action -> effect -> {occur: n, total: n}
        self.correlations: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"occur": 0, "total": 0})
        )
        self.causal_hypotheses: List[CausalHypothesis] = []
        self.intervention_queue: List[InterventionExperiment] = []

    def observe(self, transition: Transition) -> None:
        """Record an observation for correlation analysis."""
        self.observations.append(transition)
        self._update_correlations(transition)

    def _update_correlations(self, transition: Transition) -> None:
        """Track co-occurrence of actions and effects."""
        action = transition.action.name
        effects = transition.effects

        # Track when action leads to each effect
        for effect in effects:
            self.correlations[action][effect]["occur"] += 1

        # Track total occurrences of this action
        for effect_data in self.correlations[action].values():
            effect_data["total"] += 1

    def get_correlation(self, action: str, effect: str) -> float:
        """Get correlation strength between action and effect."""
        if action not in self.correlations:
            return 0.0

        if effect not in self.correlations[action]:
            return 0.0

        data = self.correlations[action][effect]
        if data["total"] == 0:
            return 0.0

        return data["occur"] / data["total"]

    def generate_intervention_experiments(
        self, correlation_threshold: float = 0.6
    ) -> List[InterventionExperiment]:
        """
        For strong correlations, design experiments to test causation.

        Only tests correlations that haven't been confirmed/refuted yet.
        """
        experiments = []

        for action, effects in self.correlations.items():
            for effect, counts in effects.items():
                if counts["total"] < 3:
                    continue  # Not enough data

                correlation = counts["occur"] / counts["total"]

                if correlation > correlation_threshold:
                    # Check if we already have a hypothesis for this
                    existing = any(
                        h.cause == action and h.effect == effect for h in self.causal_hypotheses
                    )

                    if not existing:
                        # Design controlled experiment
                        confounders = self._identify_confounders(action, effect)

                        experiments.append(
                            InterventionExperiment(
                                hypothesis_description=f"{action} causes {effect}",
                                intervention_action=action,
                                target_effect=effect,
                                required_conditions=confounders,
                            )
                        )

        return experiments

    def _identify_confounders(self, action: str, effect: str) -> List[str]:
        """Identify potential confounding variables to control."""
        confounders = []

        # Find other actions that also correlate with the effect
        for other_action, effects in self.correlations.items():
            if other_action == action:
                continue

            if effect in effects:
                other_corr = self.get_correlation(other_action, effect)
                if other_corr > 0.3:
                    confounders.append(other_action)

        return confounders[:5]  # Limit to manageable number

    def run_intervention(
        self,
        experiment: InterventionExperiment,
        env: EnvironmentInterface,
    ) -> InterventionResult:
        """
        Execute intervention experiment to test causal hypothesis.

        Compares P(effect | do(action)) to P(effect | do(control)).
        """
        result = InterventionResult(experiment=experiment)

        for _ in range(experiment.n_trials):
            # INTERVENTION: do(action)
            env.reset_to_controlled_state()

            # Create action object (simplified)
            from arcengine import GameAction

            intervention_action = getattr(
                GameAction, experiment.intervention_action, GameAction.RESET
            )
            next_state = env.step(intervention_action)

            # Check for effect
            effects = self._extract_effects_from_state(next_state)
            if experiment.target_effect in effects:
                result.intervention_with_effect += 1
            else:
                result.intervention_without_effect += 1

            # CONTROL: do(not action)
            env.reset_to_controlled_state()

            control_action = getattr(GameAction, experiment.control_action, GameAction.RESET)
            next_state = env.step(control_action)

            effects = self._extract_effects_from_state(next_state)
            if experiment.target_effect in effects:
                result.control_with_effect += 1
            else:
                result.control_without_effect += 1

        return result

    def _extract_effects_from_state(self, state: SymbolicState) -> Set[str]:
        """Extract observable effects from a state."""
        # Simplified - in practice would compare to previous state
        effects: Set[str] = set()

        for obj in state.objects:
            effects.add(f"has_{obj.object_type.name}")

        if state.agent:
            effects.add(f"agent_at_{state.agent.position.x}_{state.agent.position.y}")

        return effects

    def evaluate_causation(self, result: InterventionResult) -> CausalHypothesis:
        """
        Evaluate intervention results to determine causal relationship.

        Uses causal effect magnitude and sample size for confidence.
        """
        causal_effect = result.causal_effect
        n = (
            result.intervention_with_effect
            + result.intervention_without_effect
            + result.control_with_effect
            + result.control_without_effect
        )

        # Confidence based on effect size and sample size
        # Larger effects and more samples = higher confidence
        if n < 4:
            confidence = 0.0
        else:
            confidence = min(1.0, abs(causal_effect) * math.sqrt(n) / 5)

        # Only consider causal if effect is positive and significant
        if causal_effect <= 0.1:
            confidence = 0.0

        return CausalHypothesis(
            cause=result.experiment.intervention_action,
            effect=result.experiment.target_effect,
            conditions=frozenset(result.experiment.required_conditions),
            confidence=confidence,
            interventions_run=n,
            support_count=result.intervention_with_effect,
            refute_count=result.control_with_effect,
        )

    def add_hypothesis(self, hypothesis: CausalHypothesis) -> None:
        """Add or update a causal hypothesis."""
        # Remove existing hypothesis for same cause-effect pair
        self.causal_hypotheses = [
            h
            for h in self.causal_hypotheses
            if not (h.cause == hypothesis.cause and h.effect == hypothesis.effect)
        ]

        if hypothesis.confidence > 0.1:  # Only keep non-trivial hypotheses
            self.causal_hypotheses.append(hypothesis)

    def get_confident_rules(self, threshold: float = 0.7) -> List[CausalHypothesis]:
        """Get causal rules with confidence above threshold."""
        return [h for h in self.causal_hypotheses if h.confidence >= threshold]

    def get_rule_for_effect(self, effect: str) -> Optional[CausalHypothesis]:
        """Get the most confident rule that produces a given effect."""
        matching = [h for h in self.causal_hypotheses if h.effect == effect]

        if not matching:
            return None

        return max(matching, key=lambda h: h.confidence)


class CounterfactualReasoner:
    """
    Reason about "what would have happened if..."

    Uses a world model to simulate alternative scenarios.
    """

    def __init__(self, world_model: Optional[Any] = None):
        self.world_model = world_model

    def counterfactual(
        self,
        actual_trajectory: List[Transition],
        intervention_point: int,
        alternative_action: GameAction,
    ) -> List[SymbolicState]:
        """
        Given actual trajectory, compute what would have happened
        if we had taken a different action at intervention_point.
        """
        if self.world_model is None:
            # Can't simulate without world model
            return []

        if intervention_point >= len(actual_trajectory):
            return []

        # Get state just before intervention
        state = actual_trajectory[intervention_point].prev_state

        # Simulate alternative action
        alt_state = self.world_model.predict(state, alternative_action)

        # Continue simulation with original actions
        simulated = [alt_state]

        for t in range(intervention_point + 1, len(actual_trajectory)):
            original_action = actual_trajectory[t].action
            alt_state = self.world_model.predict(alt_state, original_action)
            simulated.append(alt_state)

        return simulated

    def necessary_cause(
        self,
        trajectory: List[Transition],
        target_effect: str,
        candidate_cause_idx: int,
    ) -> float:
        """
        Is the action at candidate_cause_idx a NECESSARY cause of target_effect?

        Necessary: Without the cause, effect would not have occurred.

        Returns: 0.0 (not necessary) to 1.0 (definitely necessary)
        """
        if self.world_model is None:
            return 0.5  # Unknown

        # Find when effect occurred
        effect_idx = None
        for i, t in enumerate(trajectory):
            if target_effect in t.effects:
                effect_idx = i
                break

        if effect_idx is None or effect_idx <= candidate_cause_idx:
            return 0.0  # Effect didn't occur or occurred before candidate

        # Counterfactual: what if we did nothing instead?
        from arcengine import GameAction

        alt_trajectory = self.counterfactual(trajectory, candidate_cause_idx, GameAction.RESET)

        # Did effect still occur in counterfactual?
        effect_in_counterfactual = any(
            target_effect in self._effects_from_transition(alt_trajectory, i)
            for i in range(len(alt_trajectory) - 1)
        )

        # Necessary cause: effect occurs in actual, not in counterfactual
        return 0.0 if effect_in_counterfactual else 1.0

    def _effects_from_transition(self, states: List[SymbolicState], idx: int) -> Set[str]:
        """Extract effects between consecutive states."""
        if idx + 1 >= len(states):
            return set()

        # Simplified effect extraction
        prev = states[idx]
        next_ = states[idx + 1]

        effects: Set[str] = set()

        # Object changes
        prev_types = {obj.object_type.name for obj in prev.objects}
        next_types = {obj.object_type.name for obj in next_.objects}

        for t in next_types - prev_types:
            effects.add(f"appeared_{t}")

        for t in prev_types - next_types:
            effects.add(f"disappeared_{t}")

        return effects

    def sufficient_cause(
        self,
        state: SymbolicState,
        action: GameAction,
        target_effect: str,
        n_samples: int = 20,
    ) -> float:
        """
        Is this action a SUFFICIENT cause of the target_effect?

        Sufficient: The cause alone is enough to produce the effect.

        Tests by simulating action from various similar states.

        Returns: 0.0 (not sufficient) to 1.0 (definitely sufficient)
        """
        if self.world_model is None:
            return 0.5  # Unknown

        # Generate variations of current state
        similar_states = self._generate_similar_states(state, n_samples)

        effect_count = 0
        for s in similar_states:
            next_state = self.world_model.predict(s, action)
            effects = self._extract_effects(s, next_state)

            if target_effect in effects:
                effect_count += 1

        return effect_count / len(similar_states) if similar_states else 0.0

    def _generate_similar_states(self, state: SymbolicState, n: int) -> List[SymbolicState]:
        """Generate states similar to input with small variations."""
        # Simplified - in practice would make small perturbations
        return [state] * n

    def _extract_effects(self, prev_state: SymbolicState, next_state: SymbolicState) -> Set[str]:
        """Extract effects between states."""
        effects: Set[str] = set()

        prev_ids = {obj.object_id for obj in prev_state.objects}
        next_ids = {obj.object_id for obj in next_state.objects}

        for _ in next_ids - prev_ids:
            effects.add("object_appeared")

        for _ in prev_ids - next_ids:
            effects.add("object_disappeared")

        if prev_state.agent and next_state.agent:
            if prev_state.agent.position != next_state.agent.position:
                effects.add("agent_moved")

        return effects
