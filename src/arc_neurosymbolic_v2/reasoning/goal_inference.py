"""
Goal Inference Module

Discovers what "winning" looks like from observations:
- Contrastive learning between success/failure states
- Feature extraction from terminal states
- Predictive coding for goal proximity estimation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

if TYPE_CHECKING:
    from arcengine import GameAction

    from ..perception.symbolic_state import SymbolicState


@dataclass(frozen=True)
class GoalHypothesis:
    """
    A hypothesis about what the goal state looks like.

    Built from observing successful level completions and
    contrasting with failure states.
    """

    description: str
    features: FrozenSet[str]
    confidence: float
    evidence_count: int

    def matches(self, state_features: FrozenSet[str]) -> float:
        """
        Calculate how well a state matches this goal hypothesis.

        Returns: 0.0 (no match) to 1.0 (perfect match)
        """
        if not self.features:
            return 0.0

        matching = len(self.features & state_features)
        return matching / len(self.features)


class GoalFeatureExtractor:
    """
    Extract goal-relevant features from symbolic states.

    Features are boolean indicators that might distinguish
    goal states from non-goal states.
    """

    def extract(self, state: SymbolicState) -> FrozenSet[str]:
        """Extract boolean features from state."""
        features: Set[str] = set()

        # Object-based features
        if self._all_collectibles_collected(state):
            features.add("no_collectibles_remaining")

        if self._agent_at_goal(state):
            features.add("agent_at_goal_position")

        # Count-based features
        for obj_type in ["collectible", "trigger", "obstacle", "goal"]:
            count = self._count_by_type(state, obj_type)
            features.add(f"{obj_type}_count_{count}")

        # Spatial features
        if self._agent_in_corner(state):
            features.add("agent_in_corner")

        if self._all_objects_aligned(state):
            features.add("objects_aligned")

        # State-change features (require history)
        # These would be populated by the GoalInferenceModule

        return frozenset(features)

    def _all_collectibles_collected(self, state: SymbolicState) -> bool:
        """Check if no collectibles remain."""
        return self._count_by_type(state, "collectible") == 0

    def _agent_at_goal(self, state: SymbolicState) -> bool:
        """Check if agent is at a goal object."""
        if not hasattr(state, "agent") or state.agent is None:
            return False

        for obj in state.objects:
            if obj.object_type.name.lower() == "goal":
                if state.agent.position.manhattan_distance(obj.position) <= 1:
                    return True
        return False

    def _count_by_type(self, state: SymbolicState, type_name: str) -> int:
        """Count objects of a specific type."""
        count = 0
        for obj in state.objects:
            if obj.object_type.name.lower() == type_name:
                count += 1
        return count

    def _agent_in_corner(self, state: SymbolicState) -> bool:
        """Check if agent is in a corner."""
        if not hasattr(state, "agent") or state.agent is None:
            return False

        pos = state.agent.position
        grid_size = 30  # Default ARC grid size

        return (pos.x <= 2 or pos.x >= grid_size - 2) and (pos.y <= 2 or pos.y >= grid_size - 2)

    def _all_objects_aligned(self, state: SymbolicState) -> bool:
        """Check if all objects are aligned (same row or column)."""
        if len(state.objects) < 2:
            return True

        xs = [obj.position.x for obj in state.objects]
        ys = [obj.position.y for obj in state.objects]

        return len(set(xs)) == 1 or len(set(ys)) == 1


class GoalInferenceModule:
    """
    Discover goals through state observation and contrastive analysis.

    Core strategy:
    1. Track success states (level completions)
    2. Track failure states (game overs)
    3. Find features common to success but absent in failure
    4. Build hypotheses about what constitutes goal achievement
    """

    def __init__(self):
        self.success_states: List[SymbolicState] = []
        self.failure_states: List[SymbolicState] = []
        self.terminal_states: List[SymbolicState] = []
        self.hypotheses: List[GoalHypothesis] = []
        self.feature_extractor = GoalFeatureExtractor()

        # State-change tracking
        self.pre_terminal_states: List[Tuple[SymbolicState, SymbolicState]] = []

    def observe_transition(
        self,
        prev_state: SymbolicState,
        action: GameAction,
        next_state: SymbolicState,
        level_completed: bool,
        game_over: bool,
    ) -> None:
        """
        Update goal hypotheses based on observed transition.

        Args:
            prev_state: State before action
            action: Action taken
            next_state: State after action
            level_completed: Whether this transition completed a level
            game_over: Whether this transition caused game over
        """
        if level_completed:
            self.success_states.append(prev_state)
            self.terminal_states.append(prev_state)
            self.pre_terminal_states.append((prev_state, next_state))
            self._update_hypotheses_on_success(prev_state)

        elif game_over:
            self.failure_states.append(prev_state)
            self._update_hypotheses_on_failure(prev_state)

    def _update_hypotheses_on_success(self, state: SymbolicState) -> None:
        """When we see a success, strengthen matching hypotheses."""
        features = self.feature_extractor.extract(state)

        # Update existing hypotheses
        updated_hypotheses: List[GoalHypothesis] = []
        for h in self.hypotheses:
            if h.features.issubset(features):
                # Hypothesis matches - strengthen
                updated_hypotheses.append(
                    GoalHypothesis(
                        description=h.description,
                        features=h.features,
                        confidence=min(1.0, h.confidence + 0.1),
                        evidence_count=h.evidence_count + 1,
                    )
                )
            else:
                # Doesn't match success - weaken
                updated_hypotheses.append(
                    GoalHypothesis(
                        description=h.description,
                        features=h.features,
                        confidence=max(0.0, h.confidence - 0.05),
                        evidence_count=h.evidence_count,
                    )
                )

        self.hypotheses = updated_hypotheses

        # Generate new hypotheses from common features
        if len(self.success_states) >= 2:
            common = self._find_common_features(self.success_states)
            self._generate_hypotheses_from_features(common)

    def _update_hypotheses_on_failure(self, state: SymbolicState) -> None:
        """When we see a failure, weaken hypotheses with those features."""
        features = self.feature_extractor.extract(state)

        updated_hypotheses: List[GoalHypothesis] = []
        for h in self.hypotheses:
            overlap = len(h.features & features)
            if overlap > 0:
                # Features present in failure - weaken
                updated_hypotheses.append(
                    GoalHypothesis(
                        description=h.description,
                        features=h.features,
                        confidence=max(0.0, h.confidence - 0.1 * overlap / len(h.features)),
                        evidence_count=h.evidence_count,
                    )
                )
            else:
                updated_hypotheses.append(h)

        self.hypotheses = updated_hypotheses

    def _find_common_features(self, states: List[SymbolicState]) -> Set[str]:
        """Find features present in all success states but not failure states."""
        if not states:
            return set()

        success_features = [set(self.feature_extractor.extract(s)) for s in states]

        # Intersection of all success state features
        common = set.intersection(*success_features) if success_features else set()

        # Remove features also present in failure states
        if self.failure_states:
            failure_features = set.union(
                *[set(self.feature_extractor.extract(s)) for s in self.failure_states]
            )
            common = common - failure_features

        return common

    def _generate_hypotheses_from_features(self, features: Set[str]) -> None:
        """Generate new hypotheses from a set of features."""
        if not features:
            return

        # Check if we already have a hypothesis with these exact features
        existing_features = {h.features for h in self.hypotheses}
        frozen = frozenset(features)

        if frozen in existing_features:
            return

        # Create new hypothesis
        description = self._describe_features(features)
        new_hypothesis = GoalHypothesis(
            description=description,
            features=frozen,
            confidence=0.5,  # Start with moderate confidence
            evidence_count=len(self.success_states),
        )

        self.hypotheses.append(new_hypothesis)

        # Prune weak hypotheses
        self.hypotheses = [h for h in self.hypotheses if h.confidence > 0.1]

        # Keep only top 10 hypotheses
        self.hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        self.hypotheses = self.hypotheses[:10]

    def _describe_features(self, features: Set[str]) -> str:
        """Generate human-readable description of features."""
        descriptions = []

        for f in sorted(features):
            if f == "no_collectibles_remaining":
                descriptions.append("collect all items")
            elif f == "agent_at_goal_position":
                descriptions.append("reach goal")
            elif f.startswith("collectible_count_"):
                count = f.split("_")[-1]
                if count == "0":
                    descriptions.append("no collectibles left")
            elif f == "agent_in_corner":
                descriptions.append("agent in corner")
            elif f == "objects_aligned":
                descriptions.append("objects aligned")
            else:
                descriptions.append(f)

        return " AND ".join(descriptions) if descriptions else "unknown goal"

    def get_goal_predicate(self) -> Callable[[SymbolicState], float]:
        """
        Return function that estimates goal probability for a state.

        Returns callable that takes SymbolicState and returns 0.0-1.0.
        """
        if not self.hypotheses:
            return lambda s: 0.0

        def goal_probability(state: SymbolicState) -> float:
            features = set(self.feature_extractor.extract(state))

            total_confidence = sum(h.confidence for h in self.hypotheses)
            if total_confidence == 0:
                return 0.0

            score = sum(h.confidence * h.matches(frozenset(features)) for h in self.hypotheses)
            return score / total_confidence

        return goal_probability

    def get_best_hypothesis(self) -> Optional[GoalHypothesis]:
        """Return the most confident goal hypothesis."""
        if not self.hypotheses:
            return None

        return max(self.hypotheses, key=lambda h: h.confidence)


class GoalProximityPredictor:
    """
    Use predictive coding to estimate how close current state is to goal.

    Core idea: Learn to predict goal states, then use prediction error
    as a measure of goal proximity (low error = close to goal).

    Note: This requires PyTorch for the neural components.
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.goal_embeddings: List[List[float]] = []

        # Lazy import to avoid dependency if not used
        self._encoder = None
        self._predictor = None

    def _init_networks(self) -> None:
        """Initialize neural networks if not already done."""
        if self._encoder is not None:
            return

        try:
            import torch.nn as nn

            class SimpleEncoder(nn.Module):
                def __init__(self, output_dim: int):
                    super().__init__()
                    self.fc = nn.Sequential(
                        nn.Linear(1024, 256),
                        nn.ReLU(),
                        nn.Linear(256, output_dim),
                    )

                def forward(self, x):
                    return self.fc(x.view(x.size(0), -1))

            self._encoder = SimpleEncoder(self.embedding_dim)
            self._predictor = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.ReLU(),
                nn.Linear(self.embedding_dim, self.embedding_dim),
            )

        except ImportError:
            # PyTorch not available - use fallback
            pass

    def train_on_goal_state(self, state: SymbolicState) -> None:
        """Train predictor to encode goal states in consistent embedding."""
        self._init_networks()

        if self._encoder is None:
            # Fallback: store simple feature vector
            features = self._simple_features(state)
            self.goal_embeddings.append(features)
            return

        tensor = self._state_to_tensor(state)
        embedding = self._encoder(tensor.unsqueeze(0))
        self.goal_embeddings.append(embedding.detach().squeeze().tolist())

    def estimate_goal_proximity(self, state: SymbolicState) -> float:
        """
        Returns 0.0 (far from goal) to 1.0 (at goal).
        """
        if not self.goal_embeddings:
            return 0.0

        self._init_networks()

        if self._encoder is None:
            # Fallback: use feature similarity
            features = self._simple_features(state)
            similarities = [
                self._cosine_similarity(features, goal_emb) for goal_emb in self.goal_embeddings
            ]
            return max(similarities) if similarities else 0.0

        import torch
        import torch.nn.functional as F

        tensor = self._state_to_tensor(state)
        embedding = self._encoder(tensor.unsqueeze(0))

        # Compare to goal centroid
        centroid = torch.tensor(self.goal_embeddings).mean(dim=0)
        distance = F.mse_loss(embedding.squeeze(), centroid, reduction="sum").item()

        # Convert distance to proximity
        return 1.0 / (1.0 + distance)

    def _state_to_tensor(self, state: SymbolicState):
        """Convert state to tensor."""
        import torch

        # Flatten grid to 1D tensor
        grid = state.to_grid()
        return torch.tensor(grid, dtype=torch.float32).flatten()

    def _simple_features(self, state: SymbolicState) -> List[float]:
        """Extract simple feature vector for fallback mode."""
        features = []

        # Object counts by type
        type_counts = {}
        for obj in state.objects:
            t = obj.object_type.name
            type_counts[t] = type_counts.get(t, 0) + 1

        for t in ["AGENT", "GOAL", "COLLECTIBLE", "OBSTACLE", "TRIGGER"]:
            features.append(float(type_counts.get(t, 0)))

        # Agent position (normalized)
        if state.agent:
            features.append(state.agent.position.x / 30.0)
            features.append(state.agent.position.y / 30.0)
        else:
            features.extend([0.0, 0.0])

        return features

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute cosine similarity between vectors."""
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)
