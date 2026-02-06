"""
Action Selector - Decides what action to take.

Combines belief state, exploration policy, and navigation.
Priority: Evidence > Simple Rules > Exploration
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .belief_state import BeliefState
from .exploration import (
    ExplorationDecision,
    ExplorationPolicy,
    ExplorationStrategy,
    RandomExplorationPolicy,
    SystematicExplorationPolicy,
)
from .navigation import AStarNavigator, find_nearest, positions_of_color


@dataclass
class ActionDecision:
    """Final action decision with reasoning."""
    action: int
    reasoning: str
    confidence: float
    source: str  # "evidence", "rule", "exploration"


class ActionSelector:
    """
    Selects actions based on beliefs and exploration.

    No ML in the selection logic itself - just combines
    belief state with navigation and exploration policy.
    """

    def __init__(
        self,
        exploration_policy: Optional[ExplorationPolicy] = None,
        step_size: int = 5,
    ):
        self.exploration_policy = exploration_policy or SystematicExplorationPolicy()
        self.navigator = AStarNavigator(step_size=step_size)
        self.step_size = step_size

    def select_action(
        self,
        belief_state: BeliefState,
        current_frame: np.ndarray,
        available_actions: list[int] = None,
    ) -> ActionDecision:
        """
        Select the next action.

        Priority:
        1. If timer critical, rush to nearest target
        2. If goal is known with high confidence, go to goal
        3. If collectibles known, collect them
        4. Otherwise, explore
        """
        if available_actions is None:
            available_actions = [0, 1, 2, 3, 4, 5, 6]  # Default ARC actions

        h, w = current_frame.shape
        grid_size = (w, h)

        # Check timer urgency
        timer_urgent = belief_state.timer.is_urgent
        timer_critical = belief_state.timer.is_critical

        # If stuck, force exploration in a different direction
        if self._is_stuck(belief_state):
            # Find which directions we've been trying (and failing)
            recent = belief_state.recent_observations[-10:]
            recent_actions = [obs.action_taken for obs in recent if obs.action_taken in [1, 2, 3, 4]]

            # Count actions
            from collections import Counter
            action_counts = Counter(recent_actions)

            # Choose the least-tried direction
            all_dirs = [1, 2, 3, 4]
            least_tried = min(all_dirs, key=lambda a: action_counts.get(a, 0))

            return ActionDecision(
                action=least_tried,
                reasoning=f"Stuck! Trying opposite direction ({least_tried})",
                confidence=0.3,
                source="exploration",
            )

        # Get obstacle positions (colors we know block movement)
        obstacles = set()
        for color in belief_state.get_confident_blockers(threshold=0.6):
            obstacles.update(positions_of_color(current_frame, color))

        # TIMER CRITICAL: Rush to any non-blocker colored region
        if timer_critical and belief_state.player_position:
            # Try any color that might be a goal or collectible
            for color in range(1, 10):  # Try colors 1-9
                if color == belief_state.player_color:
                    continue
                if color == belief_state.timer.timer_color:
                    continue  # Skip timer
                color_belief = belief_state.color_beliefs.get(color)
                if color_belief and color_belief.is_blocker >= 0.7:
                    continue  # Skip known blockers
                positions = positions_of_color(current_frame, color)
                if positions:
                    target = find_nearest(belief_state.player_position, positions)
                    action = self._navigate_to(
                        belief_state.player_position, target, obstacles, grid_size
                    )
                    if action:
                        return ActionDecision(
                            action=action,
                            reasoning=f"TIMER CRITICAL: rushing to color {color}",
                            confidence=0.5,
                            source="evidence",
                        )

        # Priority 1: Go to goal if known
        # Lower threshold when timer is urgent
        goal_threshold = 0.3 if timer_urgent else 0.5
        goal_colors = [
            color for color, belief in belief_state.color_beliefs.items()
            if belief.is_goal >= goal_threshold
        ]
        if goal_colors and belief_state.player_position:
            for color in goal_colors:
                goal_positions = positions_of_color(current_frame, color)
                if goal_positions:
                    target = find_nearest(belief_state.player_position, goal_positions)
                    action = self._navigate_to(
                        belief_state.player_position, target, obstacles, grid_size
                    )
                    if action:
                        urgency = " [URGENT]" if timer_urgent else ""
                        return ActionDecision(
                            action=action,
                            reasoning=f"Going to goal (color {color}){urgency}",
                            confidence=0.8,
                            source="evidence",
                        )

        # Priority 2: Collect known collectibles
        # Lower threshold when timer is urgent
        collectible_threshold = 0.4 if timer_urgent else 0.6
        collectible_colors = belief_state.get_confident_collectibles(threshold=collectible_threshold)
        if collectible_colors and belief_state.player_position:
            for color in collectible_colors:
                collectible_positions = positions_of_color(current_frame, color)
                if collectible_positions:
                    target = find_nearest(belief_state.player_position, collectible_positions)
                    action = self._navigate_to(
                        belief_state.player_position, target, obstacles, grid_size
                    )
                    if action:
                        return ActionDecision(
                            action=action,
                            reasoning=f"Collecting item (color {color})",
                            confidence=0.7,
                            source="rule",
                        )

        # Priority 3: Explore
        explore_decision = self.exploration_policy.decide(
            belief_state, current_frame, available_actions
        )

        # If exploration gives a target position, navigate to it
        if explore_decision.target_position and belief_state.player_position:
            action = self._navigate_to(
                belief_state.player_position,
                explore_decision.target_position,
                obstacles,
                grid_size,
            )
            if action:
                return ActionDecision(
                    action=action,
                    reasoning=f"Exploring: {explore_decision.strategy.value}",
                    confidence=explore_decision.confidence,
                    source="exploration",
                )

        # If exploration gives a direct action, use it
        if explore_decision.action is not None:
            return ActionDecision(
                action=explore_decision.action,
                reasoning=f"Exploring: {explore_decision.strategy.value}",
                confidence=explore_decision.confidence,
                source="exploration",
            )

        # Fallback: random movement action
        action = np.random.choice([1, 2, 3, 4])
        return ActionDecision(
            action=action,
            reasoning="Fallback: random movement",
            confidence=0.1,
            source="exploration",
        )

    def _is_stuck(self, belief_state: BeliefState) -> bool:
        """Check if agent seems stuck (same position for many actions)."""
        if len(belief_state.recent_observations) < 10:
            return False

        # Count how many recent observations had blocked movement
        recent = belief_state.recent_observations[-10:]
        blocked_count = sum(1 for obs in recent if obs.movement_blocked)

        # If most recent moves were blocked, we're stuck
        if blocked_count >= 7:
            return True

        # Check if position hasn't changed much
        recent_positions = [
            obs.player_new_pos for obs in recent
            if obs.player_new_pos
        ]
        if len(recent_positions) < 5:
            # Not enough position data - check if we've been trying to move
            move_attempts = sum(1 for obs in recent if obs.action_taken in [1, 2, 3, 4])
            if move_attempts >= 8 and blocked_count >= 3:
                return True
            return False

        # If all recent positions are very close, we're stuck
        unique_positions = set(recent_positions)
        return len(unique_positions) <= 2

    def _navigate_to(
        self,
        current: tuple[int, int],
        target: tuple[int, int],
        obstacles: set[tuple[int, int]],
        grid_size: tuple[int, int],
    ) -> Optional[int]:
        """Get action to navigate toward target."""
        return self.navigator.get_next_action(current, target, obstacles, grid_size)


class SimpleRuleGeneralizer:
    """
    Simple rule generalization without LLM.

    Rule: Same color = same behavior
    """

    @staticmethod
    def infer_rules(belief_state: BeliefState) -> list[str]:
        """Generate simple rules from beliefs."""
        rules = []

        for color, belief in belief_state.color_beliefs.items():
            if belief.is_blocker >= 0.7:
                rules.append(f"Color {color} blocks movement")
            if belief.is_collectible >= 0.7:
                rules.append(f"Color {color} is collectible")
            if belief.is_goal >= 0.5:
                rules.append(f"Color {color} may be the goal")

        return rules

    @staticmethod
    def predict_color_behavior(
        belief_state: BeliefState,
        color: int,
    ) -> dict[str, float]:
        """Predict behavior of a color based on evidence."""
        if color in belief_state.color_beliefs:
            belief = belief_state.color_beliefs[color]
            return {
                "blocker": belief.is_blocker,
                "collectible": belief.is_collectible,
                "goal": belief.is_goal,
            }
        else:
            # Unknown color - neutral priors
            return {
                "blocker": 0.3,  # Most things aren't blockers
                "collectible": 0.2,
                "goal": 0.1,
            }


def test_action_selector():
    """Test action selector."""
    from .belief_state import BeliefState

    # Create test scenario
    belief_state = BeliefState()
    belief_state.player_identified = True
    belief_state.player_position = (32, 32)

    # Add some beliefs
    belief_state.get_color_belief(5).times_blocked_movement = 3
    belief_state.get_color_belief(4).times_disappeared_on_touch = 2

    # Create frame
    frame = np.zeros((64, 64), dtype=np.int32)
    frame[10:20, 10:20] = 4  # Collectible
    frame[40:50, 40:50] = 5  # Blocker

    # Test action selection
    selector = ActionSelector()
    decision = selector.select_action(belief_state, frame)

    print(f"Action: {decision.action}")
    print(f"Reasoning: {decision.reasoning}")
    print(f"Confidence: {decision.confidence:.2f}")
    print(f"Source: {decision.source}")

    # Test rule generalization
    rules = SimpleRuleGeneralizer.infer_rules(belief_state)
    print(f"\nInferred rules:")
    for rule in rules:
        print(f"  - {rule}")


if __name__ == "__main__":
    test_action_selector()
