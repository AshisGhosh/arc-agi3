"""
Hidden State Detection Module

Mechanisms to infer unobservable variables:
- Detect when observed effects don't match predicted effects
- Hypothesize latent variables that explain discrepancies
- Track belief state over hidden variables using Bayesian inference
"""

from __future__ import annotations

import math
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
    Tuple,
)

if TYPE_CHECKING:
    from arcengine import GameAction

    from ..perception.symbolic_state import SymbolicState


@dataclass
class PredictionDiscrepancy:
    """A mismatch between predicted and actual effects."""

    predicted_effects: Set[str]
    actual_effects: Set[str]
    missing_effects: Set[str]  # Predicted but didn't happen
    unexpected_effects: Set[str]  # Happened but not predicted
    context_description: str
    action_name: str
    frame_index: int

    @property
    def is_significant(self) -> bool:
        """Check if discrepancy is significant enough to warrant investigation."""
        return len(self.missing_effects) > 0 or len(self.unexpected_effects) > 0


class HiddenVariableHypothesis(Protocol):
    """Protocol for hidden variable hypotheses."""

    @property
    def variable_name(self) -> str:
        """Unique name for this hidden variable."""
        ...

    def get_prior(self) -> Dict[Any, float]:
        """Get prior distribution over possible values."""
        ...

    def likelihood(
        self,
        prev_state: SymbolicState,
        action: GameAction,
        next_state: SymbolicState,
        observed_effects: Set[str],
    ) -> float:
        """P(observation | hidden_var = current_belief)"""
        ...

    def update_estimate(self, observation_likelihood: float) -> None:
        """Update internal state based on new observation."""
        ...


@dataclass
class CounterHypothesis:
    """
    Hypothesis: There's a hidden counter incremented by some action.

    Example: A door opens after collecting 3 keys, but key count isn't shown.
    """

    trigger_action: str
    threshold: int
    effect_description: str
    max_value: int = 10
    current_estimate: int = 0
    confidence: float = 0.5

    @property
    def variable_name(self) -> str:
        return f"counter_{self.trigger_action}"

    def get_prior(self) -> Dict[int, float]:
        """Uniform prior over counter values."""
        return {i: 1.0 / (self.max_value + 1) for i in range(self.max_value + 1)}

    def likelihood(
        self,
        prev_state: SymbolicState,
        action: GameAction,
        next_state: SymbolicState,
        observed_effects: Set[str],
    ) -> float:
        """Compute likelihood of observation given counter hypothesis."""
        effect_occurred = self.effect_description in observed_effects
        action_matches = action.name == self.trigger_action

        if effect_occurred:
            # Effect occurred - counter was likely at threshold
            if self.current_estimate >= self.threshold:
                return 0.9
            else:
                return 0.1
        else:
            # Effect didn't occur
            if action_matches:
                # Action happened but no effect - counter below threshold
                if self.current_estimate < self.threshold:
                    return 0.8
                else:
                    return 0.2
            return 0.5  # Uninformative

    def update_estimate(self, action_name: str) -> None:
        """Update counter estimate based on action."""
        if action_name == self.trigger_action:
            self.current_estimate = min(self.current_estimate + 1, self.max_value)


@dataclass
class CooldownHypothesis:
    """
    Hypothesis: An effect only happens if enough time has passed since last trigger.

    Example: A special ability that can only be used every 5 frames.
    """

    trigger_action: str
    cooldown_frames: int
    effect_description: str
    frames_since_trigger: int = 0
    confidence: float = 0.5

    @property
    def variable_name(self) -> str:
        return f"cooldown_{self.trigger_action}"

    def get_prior(self) -> Dict[int, float]:
        """Prior over frames since last trigger."""
        return {i: 1.0 / (self.cooldown_frames + 10) for i in range(self.cooldown_frames + 10)}

    def likelihood(
        self,
        prev_state: SymbolicState,
        action: GameAction,
        next_state: SymbolicState,
        observed_effects: Set[str],
    ) -> float:
        """Compute likelihood based on cooldown hypothesis."""
        effect_occurred = self.effect_description in observed_effects
        action_matches = action.name == self.trigger_action

        if action_matches:
            if effect_occurred:
                # Effect happened - cooldown was ready
                if self.frames_since_trigger >= self.cooldown_frames:
                    return 0.9
                else:
                    return 0.1
            else:
                # Action but no effect - cooldown not ready
                if self.frames_since_trigger < self.cooldown_frames:
                    return 0.8
                else:
                    return 0.2

        return 0.5

    def update_estimate(self, action_name: str) -> None:
        """Update cooldown timer."""
        if action_name == self.trigger_action:
            self.frames_since_trigger = 0
        else:
            self.frames_since_trigger += 1


@dataclass
class StateMachineHypothesis:
    """
    Hypothesis: Environment has hidden discrete states with transitions.

    Example: A puzzle that requires actions in a specific sequence.
    """

    states: List[str]
    transitions: Dict[Tuple[str, str], str]  # (state, action) -> next_state
    effect_map: Dict[str, Set[str]]  # state -> possible effects
    current_state: str = ""
    confidence: float = 0.5

    def __post_init__(self):
        if not self.current_state and self.states:
            self.current_state = self.states[0]

    @property
    def variable_name(self) -> str:
        return "hidden_state_machine"

    def get_prior(self) -> Dict[str, float]:
        """Uniform prior over states."""
        return {s: 1.0 / len(self.states) for s in self.states}

    def likelihood(
        self,
        prev_state: SymbolicState,
        action: GameAction,
        next_state: SymbolicState,
        observed_effects: Set[str],
    ) -> float:
        """Compute likelihood based on state machine hypothesis."""
        expected_effects = self.effect_map.get(self.current_state, set())

        # Check how well observed effects match expected
        if not expected_effects and not observed_effects:
            return 0.5

        if expected_effects:
            match_ratio = len(expected_effects & observed_effects) / len(expected_effects)
            return match_ratio * 0.8 + 0.1

        return 0.3

    def update_estimate(self, action_name: str) -> None:
        """Update state based on transition function."""
        key = (self.current_state, action_name)
        if key in self.transitions:
            self.current_state = self.transitions[key]


@dataclass
class BeliefState:
    """
    Probability distribution over possible hidden variable values.

    Tracks uncertainty about each hypothesized hidden variable
    and updates beliefs based on observations using Bayes' rule.
    """

    distributions: Dict[str, Dict[Any, float]] = field(default_factory=dict)

    def update(
        self,
        var_name: str,
        value_likelihoods: Dict[Any, float],
    ) -> None:
        """
        Bayesian update of belief about hidden variable.

        Args:
            var_name: Name of the hidden variable
            value_likelihoods: P(observation | value) for each possible value
        """
        if var_name not in self.distributions:
            # Initialize with uniform prior
            self.distributions[var_name] = {
                v: 1.0 / len(value_likelihoods) for v in value_likelihoods
            }

        dist = self.distributions[var_name]

        # Bayesian update: P(value | obs) proportional to P(obs | value) * P(value)
        for value in dist:
            if value in value_likelihoods:
                dist[value] *= value_likelihoods[value]

        # Normalize
        total = sum(dist.values())
        if total > 0:
            for value in dist:
                dist[value] /= total

    def most_likely(self, var_name: str) -> Any:
        """Get most likely value for a hidden variable."""
        if var_name not in self.distributions:
            return None

        dist = self.distributions[var_name]
        if not dist:
            return None

        return max(dist.items(), key=lambda x: x[1])[0]

    def entropy(self, var_name: str) -> float:
        """Measure uncertainty about hidden variable (higher = more uncertain)."""
        if var_name not in self.distributions:
            return float("inf")

        dist = self.distributions[var_name]
        return -sum(p * math.log(p + 1e-10) for p in dist.values() if p > 0)

    def confidence(self, var_name: str) -> float:
        """Confidence in most likely value (1 - normalized entropy)."""
        if var_name not in self.distributions:
            return 0.0

        dist = self.distributions[var_name]
        if len(dist) <= 1:
            return 1.0

        max_entropy = math.log(len(dist))
        current_entropy = self.entropy(var_name)

        if max_entropy == 0:
            return 1.0

        return 1.0 - (current_entropy / max_entropy)


class LearnedEffectPredictor:
    """
    Predicts effects of actions based on learned patterns.

    Tracks action -> effect correlations and uses them to predict
    what should happen, enabling discrepancy detection.
    """

    def __init__(self):
        # Track observed (action, context_features) -> effects
        self.effect_history: Dict[str, List[Set[str]]] = {}
        self.min_observations = 3

    def observe(
        self,
        action_name: str,
        context_features: FrozenSet[str],
        effects: Set[str],
    ) -> None:
        """Record an observation of action effects."""
        key = f"{action_name}:{','.join(sorted(context_features)[:5])}"

        if key not in self.effect_history:
            self.effect_history[key] = []

        self.effect_history[key].append(effects)

    def predict(
        self,
        action_name: str,
        context_features: FrozenSet[str],
    ) -> Set[str]:
        """Predict effects of an action in given context."""
        key = f"{action_name}:{','.join(sorted(context_features)[:5])}"

        if key not in self.effect_history:
            return set()

        observations = self.effect_history[key]

        if len(observations) < self.min_observations:
            return set()

        # Return effects that occurred in >50% of observations
        effect_counts: Dict[str, int] = {}
        for obs in observations:
            for effect in obs:
                effect_counts[effect] = effect_counts.get(effect, 0) + 1

        threshold = len(observations) * 0.5
        return {e for e, c in effect_counts.items() if c >= threshold}


class HiddenStateDetector:
    """
    Detect when observed effects don't match predictions,
    suggesting hidden state variables.

    Main workflow:
    1. Observe transitions and predict effects
    2. Detect discrepancies between predicted and actual
    3. Generate hypotheses about hidden variables
    4. Track belief state over hypothesized variables
    """

    def __init__(self):
        self.effect_predictor = LearnedEffectPredictor()
        self.discrepancy_history: List[PredictionDiscrepancy] = []
        self.hidden_var_hypotheses: List[HiddenVariableHypothesis] = []
        self.belief_state = BeliefState()
        self.frame_count: int = 0

    def observe_transition(
        self,
        prev_state: SymbolicState,
        action: GameAction,
        next_state: SymbolicState,
    ) -> Optional[PredictionDiscrepancy]:
        """
        Check if transition matches predictions.

        Returns discrepancy if prediction failed, None otherwise.
        """
        self.frame_count += 1

        # Extract context features
        context = self._extract_context_features(prev_state)

        # What did we predict?
        predicted = self.effect_predictor.predict(action.name, context)

        # What actually happened?
        actual = self._extract_effects(prev_state, next_state)

        # Record observation
        self.effect_predictor.observe(action.name, context, actual)

        # Check for discrepancy
        missing = predicted - actual
        unexpected = actual - predicted

        if missing or unexpected:
            discrepancy = PredictionDiscrepancy(
                predicted_effects=predicted,
                actual_effects=actual,
                missing_effects=missing,
                unexpected_effects=unexpected,
                context_description=str(context),
                action_name=action.name,
                frame_index=self.frame_count,
            )
            self.discrepancy_history.append(discrepancy)
            self._generate_hidden_var_hypotheses(discrepancy, action)
            return discrepancy

        return None

    def _extract_context_features(self, state: SymbolicState) -> FrozenSet[str]:
        """Extract context features that might affect action outcomes."""
        features: Set[str] = set()

        # Object proximity features
        if state.agent:
            for obj in state.objects:
                dist = state.agent.position.manhattan_distance(obj.position)
                if dist <= 2:
                    features.add(f"near_{obj.object_type.name}")

        # Object count features
        for obj in state.objects:
            features.add(f"has_{obj.object_type.name}")

        return frozenset(features)

    def _extract_effects(
        self,
        prev_state: SymbolicState,
        next_state: SymbolicState,
    ) -> Set[str]:
        """Extract effects (changes) between states."""
        effects: Set[str] = set()

        # Object appearance/disappearance
        prev_ids = {obj.object_id for obj in prev_state.objects}
        next_ids = {obj.object_id for obj in next_state.objects}

        appeared = next_ids - prev_ids
        disappeared = prev_ids - next_ids

        for obj_id in appeared:
            obj = next((o for o in next_state.objects if o.object_id == obj_id), None)
            if obj:
                effects.add(f"appeared_{obj.object_type.name}")

        for obj_id in disappeared:
            obj = next((o for o in prev_state.objects if o.object_id == obj_id), None)
            if obj:
                effects.add(f"disappeared_{obj.object_type.name}")

        # Agent movement
        if prev_state.agent and next_state.agent:
            if prev_state.agent.position != next_state.agent.position:
                effects.add("agent_moved")

        # Color changes
        for prev_obj in prev_state.objects:
            next_obj = next(
                (o for o in next_state.objects if o.object_id == prev_obj.object_id),
                None,
            )
            if next_obj and prev_obj.color != next_obj.color:
                effects.add(f"color_changed_{prev_obj.object_type.name}")

        return effects

    def _generate_hidden_var_hypotheses(
        self,
        discrepancy: PredictionDiscrepancy,
        action: GameAction,
    ) -> None:
        """Generate hypotheses about hidden variables that explain discrepancy."""

        # Pattern 1: Counter variable
        # Unexpected effect might mean a counter reached threshold
        if discrepancy.unexpected_effects:
            for effect in discrepancy.unexpected_effects:
                # Check if we've seen this action multiple times without effect
                similar_discrepancies = [
                    d
                    for d in self.discrepancy_history
                    if d.action_name == action.name and effect in d.missing_effects
                ]

                if len(similar_discrepancies) >= 2:
                    # This action sometimes has effect, sometimes doesn't
                    # Hypothesis: there's a counter
                    hypothesis = CounterHypothesis(
                        trigger_action=action.name,
                        threshold=len(similar_discrepancies),
                        effect_description=effect,
                    )

                    if not any(
                        h.variable_name == hypothesis.variable_name
                        for h in self.hidden_var_hypotheses
                    ):
                        self.hidden_var_hypotheses.append(hypothesis)

        # Pattern 2: Cooldown
        # Effect didn't happen when it should have
        if discrepancy.missing_effects:
            for effect in discrepancy.missing_effects:
                # Check timing of last success
                last_success = None
                for d in reversed(self.discrepancy_history):
                    if d.action_name == action.name and effect in d.actual_effects:
                        last_success = d
                        break

                if last_success:
                    frames_diff = self.frame_count - last_success.frame_index
                    if frames_diff < 10:  # Recent success but now failure
                        hypothesis = CooldownHypothesis(
                            trigger_action=action.name,
                            cooldown_frames=frames_diff + 1,
                            effect_description=effect,
                        )

                        if not any(
                            h.variable_name == hypothesis.variable_name
                            for h in self.hidden_var_hypotheses
                        ):
                            self.hidden_var_hypotheses.append(hypothesis)

    def update_belief_state(
        self,
        prev_state: SymbolicState,
        action: GameAction,
        next_state: SymbolicState,
    ) -> None:
        """Update probability distribution over hidden variables."""
        effects = self._extract_effects(prev_state, next_state)

        for hypothesis in self.hidden_var_hypotheses:
            likelihood = hypothesis.likelihood(prev_state, action, next_state, effects)

            # Update belief state
            self.belief_state.update(
                hypothesis.variable_name,
                {v: likelihood for v in hypothesis.get_prior()},
            )

            # Update hypothesis internal state
            hypothesis.update_estimate(action.name)

    def get_high_confidence_hypotheses(
        self, threshold: float = 0.7
    ) -> List[HiddenVariableHypothesis]:
        """Get hypotheses with confidence above threshold."""
        return [
            h
            for h in self.hidden_var_hypotheses
            if self.belief_state.confidence(h.variable_name) >= threshold
        ]

    def reset(self) -> None:
        """Reset detector state for new episode."""
        self.frame_count = 0
        # Keep learned effect predictor but reset hypotheses
        self.hidden_var_hypotheses = []
        self.belief_state = BeliefState()
