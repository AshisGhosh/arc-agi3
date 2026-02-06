"""
Exploration Policies - Random (baseline) and Learned (treatment).

The learned version can be A/B tested against random.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .belief_state import BeliefState
from .navigation import AStarNavigator, find_nearest, positions_of_color


class ExplorationStrategy(Enum):
    """High-level exploration strategies."""
    RANDOM_ACTION = "random_action"
    VISIT_UNVISITED = "visit_unvisited"
    TEST_NEW_COLOR = "test_new_color"
    REVISIT_UNCERTAIN = "revisit_uncertain"
    EXPLOIT_KNOWLEDGE = "exploit_knowledge"


@dataclass
class ExplorationDecision:
    """Decision from exploration policy."""
    strategy: ExplorationStrategy
    target_position: Optional[tuple[int, int]] = None
    target_color: Optional[int] = None
    action: Optional[int] = None
    confidence: float = 0.5


class ExplorationPolicy(ABC):
    """Base class for exploration policies."""

    @abstractmethod
    def decide(
        self,
        belief_state: BeliefState,
        current_frame: np.ndarray,
        available_actions: list[int],
    ) -> ExplorationDecision:
        """Decide what to do next."""
        pass


class RandomExplorationPolicy(ExplorationPolicy):
    """
    Random exploration - baseline for A/B testing.

    Just picks random actions or random targets.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def decide(
        self,
        belief_state: BeliefState,
        current_frame: np.ndarray,
        available_actions: list[int],
    ) -> ExplorationDecision:
        """Pick a random action."""
        # 50% chance: random action
        # 50% chance: random target position
        if self.rng.random() < 0.5:
            action = self.rng.choice(available_actions)
            return ExplorationDecision(
                strategy=ExplorationStrategy.RANDOM_ACTION,
                action=action,
                confidence=0.0,
            )
        else:
            # Pick random non-visited position
            h, w = current_frame.shape
            unvisited = []
            for y in range(0, h, 5):  # Sample every 5 pixels
                for x in range(0, w, 5):
                    if (x, y) not in belief_state.positions_visited:
                        unvisited.append((x, y))

            if unvisited:
                target = unvisited[self.rng.randint(len(unvisited))]
                return ExplorationDecision(
                    strategy=ExplorationStrategy.VISIT_UNVISITED,
                    target_position=target,
                    confidence=0.0,
                )
            else:
                action = self.rng.choice(available_actions)
                return ExplorationDecision(
                    strategy=ExplorationStrategy.RANDOM_ACTION,
                    action=action,
                    confidence=0.0,
                )


class SystematicExplorationPolicy(ExplorationPolicy):
    """
    Systematic exploration - smarter baseline.

    Prioritizes testing untested colors and visiting unvisited areas.
    """

    def __init__(self):
        self.navigator = AStarNavigator(step_size=5)

    def decide(
        self,
        belief_state: BeliefState,
        current_frame: np.ndarray,
        available_actions: list[int],
    ) -> ExplorationDecision:
        """Decide based on simple heuristics."""
        h, w = current_frame.shape

        # Priority 1: If we haven't identified player, try movement actions
        if not belief_state.player_identified:
            return ExplorationDecision(
                strategy=ExplorationStrategy.RANDOM_ACTION,
                action=np.random.choice([1, 2, 3, 4]),  # Movement actions
                confidence=0.3,
            )

        # Priority 2: Test untested colors
        all_colors = set(np.unique(current_frame)) - {0}  # Exclude background
        untested = all_colors - belief_state.colors_tested

        if untested and belief_state.player_position:
            # Find nearest position with untested color
            for color in untested:
                positions = positions_of_color(current_frame, color)
                if positions:
                    target = find_nearest(belief_state.player_position, positions)
                    if target:
                        return ExplorationDecision(
                            strategy=ExplorationStrategy.TEST_NEW_COLOR,
                            target_position=target,
                            target_color=color,
                            confidence=0.6,
                        )

        # Priority 3: Visit unvisited areas
        if belief_state.player_position:
            unvisited = []
            for y in range(0, h, 10):
                for x in range(0, w, 10):
                    if (x, y) not in belief_state.positions_visited:
                        unvisited.append((x, y))

            if unvisited:
                target = find_nearest(belief_state.player_position, set(unvisited))
                if target:
                    return ExplorationDecision(
                        strategy=ExplorationStrategy.VISIT_UNVISITED,
                        target_position=target,
                        confidence=0.4,
                    )

        # Fallback: random action
        return ExplorationDecision(
            strategy=ExplorationStrategy.RANDOM_ACTION,
            action=np.random.choice(available_actions),
            confidence=0.1,
        )


class LearnedExplorationPolicy(ExplorationPolicy):
    """
    Learned exploration policy - treatment for A/B testing.

    Uses a small neural network to decide exploration strategy
    based on abstract features of the belief state.
    """

    def __init__(self, model: Optional[nn.Module] = None):
        self.model = model or self._create_default_model()
        self.navigator = AStarNavigator(step_size=5)

    def _create_default_model(self) -> nn.Module:
        """Create default exploration policy network."""
        return ExplorationPolicyNetwork()

    def decide(
        self,
        belief_state: BeliefState,
        current_frame: np.ndarray,
        available_actions: list[int],
    ) -> ExplorationDecision:
        """Use learned policy to decide."""
        # Extract abstract features
        features = self._extract_features(belief_state, current_frame)

        # Forward pass
        with torch.no_grad():
            features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.model(features_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            strategy_idx = torch.argmax(probs).item()
            confidence = probs[strategy_idx].item()

        strategy = list(ExplorationStrategy)[strategy_idx]

        # Convert strategy to concrete decision
        return self._strategy_to_decision(
            strategy, belief_state, current_frame, available_actions, confidence
        )

    def _extract_features(
        self,
        belief_state: BeliefState,
        current_frame: np.ndarray,
    ) -> list[float]:
        """Extract abstract features (not pixel-level)."""
        h, w = current_frame.shape
        all_colors = set(np.unique(current_frame)) - {0}

        features = [
            # Player knowledge
            float(belief_state.player_identified),
            float(belief_state.player_position is not None),

            # Exploration progress
            len(belief_state.colors_tested) / max(len(all_colors), 1),
            len(belief_state.positions_visited) / (h * w / 25),  # Normalized by grid cells

            # Uncertainty
            belief_state.get_uncertainty_score(),

            # Knowledge counts
            len(belief_state.get_confident_blockers()) / max(len(all_colors), 1),
            len(belief_state.get_confident_collectibles()) / max(len(all_colors), 1),

            # Actions taken
            min(belief_state.total_actions / 100, 1.0),

            # Levels completed
            float(belief_state.levels_completed > 0),
        ]

        return features

    def _strategy_to_decision(
        self,
        strategy: ExplorationStrategy,
        belief_state: BeliefState,
        current_frame: np.ndarray,
        available_actions: list[int],
        confidence: float,
    ) -> ExplorationDecision:
        """Convert high-level strategy to concrete decision."""
        h, w = current_frame.shape

        if strategy == ExplorationStrategy.RANDOM_ACTION:
            return ExplorationDecision(
                strategy=strategy,
                action=np.random.choice(available_actions),
                confidence=confidence,
            )

        elif strategy == ExplorationStrategy.TEST_NEW_COLOR:
            all_colors = set(np.unique(current_frame)) - {0}
            untested = all_colors - belief_state.colors_tested

            if untested and belief_state.player_position:
                for color in untested:
                    positions = positions_of_color(current_frame, color)
                    if positions:
                        target = find_nearest(belief_state.player_position, positions)
                        if target:
                            return ExplorationDecision(
                                strategy=strategy,
                                target_position=target,
                                target_color=color,
                                confidence=confidence,
                            )

        elif strategy == ExplorationStrategy.VISIT_UNVISITED:
            if belief_state.player_position:
                unvisited = set()
                for y in range(0, h, 10):
                    for x in range(0, w, 10):
                        if (x, y) not in belief_state.positions_visited:
                            unvisited.add((x, y))

                if unvisited:
                    target = find_nearest(belief_state.player_position, unvisited)
                    if target:
                        return ExplorationDecision(
                            strategy=strategy,
                            target_position=target,
                            confidence=confidence,
                        )

        # Fallback
        return ExplorationDecision(
            strategy=ExplorationStrategy.RANDOM_ACTION,
            action=np.random.choice(available_actions),
            confidence=confidence * 0.5,
        )


class ExplorationPolicyNetwork(nn.Module):
    """
    Small network for exploration policy.

    Input: Abstract belief state features (9 dims)
    Output: Strategy probabilities (5 strategies)
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 32, num_strategies: int = 5):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_strategies),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def test_exploration_policies():
    """Test exploration policies."""
    from .belief_state import BeliefState

    # Create test data
    belief_state = BeliefState()
    belief_state.player_identified = True
    belief_state.player_position = (32, 32)
    belief_state.colors_tested = {1, 2}

    frame = np.zeros((64, 64), dtype=np.int32)
    frame[10:20, 10:20] = 3  # Some color region
    frame[30:35, 30:35] = 4  # Another region

    available_actions = [1, 2, 3, 4]

    # Test random policy
    print("Random Policy:")
    random_policy = RandomExplorationPolicy(seed=42)
    for i in range(3):
        decision = random_policy.decide(belief_state, frame, available_actions)
        print(f"  Decision {i+1}: {decision.strategy.value}, action={decision.action}")

    # Test systematic policy
    print("\nSystematic Policy:")
    systematic_policy = SystematicExplorationPolicy()
    decision = systematic_policy.decide(belief_state, frame, available_actions)
    print(f"  Decision: {decision.strategy.value}, target={decision.target_position}")

    # Test learned policy (untrained)
    print("\nLearned Policy (untrained):")
    learned_policy = LearnedExplorationPolicy()
    decision = learned_policy.decide(belief_state, frame, available_actions)
    print(f"  Decision: {decision.strategy.value}, confidence={decision.confidence:.2f}")


if __name__ == "__main__":
    test_exploration_policies()
