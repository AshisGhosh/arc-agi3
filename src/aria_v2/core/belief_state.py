"""
Belief State - Evidence-based tracking of game understanding.

No ML here - just counting observations as evidence.
"""

from dataclasses import dataclass, field
from typing import Optional

from .observation_tracker import Observation


@dataclass
class ColorBelief:
    """What we believe about a specific color."""
    color: int

    # Evidence counts
    times_blocked_movement: int = 0
    times_walked_through: int = 0
    times_disappeared_on_touch: int = 0
    times_touched_without_disappearing: int = 0
    times_triggered_level_complete: int = 0

    @property
    def is_blocker(self) -> float:
        """Confidence that this color blocks movement."""
        total = self.times_blocked_movement + self.times_walked_through
        if total == 0:
            return 0.5  # Unknown
        return self.times_blocked_movement / total

    @property
    def is_collectible(self) -> float:
        """Confidence that this color is collectible."""
        total = self.times_disappeared_on_touch + self.times_touched_without_disappearing
        if total == 0:
            return 0.5  # Unknown
        return self.times_disappeared_on_touch / total

    @property
    def is_goal(self) -> float:
        """Confidence that this color is the goal."""
        if self.times_triggered_level_complete > 0:
            return min(1.0, self.times_triggered_level_complete * 0.5)
        return 0.0

    @property
    def total_observations(self) -> int:
        return (self.times_blocked_movement +
                self.times_walked_through +
                self.times_disappeared_on_touch +
                self.times_touched_without_disappearing +
                self.times_triggered_level_complete)


@dataclass
class PositionBelief:
    """What we believe about a specific position."""
    x: int
    y: int

    visited: bool = False
    times_visited: int = 0
    blocked_from_here: list[int] = field(default_factory=list)  # Directions blocked
    item_collected_here: bool = False


@dataclass
class TimerState:
    """Track game timer (Color 8 countdown)."""
    timer_color: int = 8  # Color that represents the timer
    current_size: int = 12  # Current number of timer pixels (starts full)
    initial_size: int = 12  # Starting timer size
    last_reset_action: int = 0  # Action when timer last reset
    resets_remaining: int = 2  # Timer typically resets 2 times before game over
    initialized: bool = False  # Whether we've seen the timer in game yet

    @property
    def time_fraction(self) -> float:
        """Fraction of time remaining (1.0 = full, 0.0 = empty)."""
        if not self.initialized or self.initial_size == 0:
            return 1.0  # Assume full if not yet seen
        return self.current_size / self.initial_size

    @property
    def is_urgent(self) -> bool:
        """Is time running low?"""
        return self.initialized and self.time_fraction < 0.4

    @property
    def is_critical(self) -> bool:
        """Is time almost out?"""
        return self.initialized and self.time_fraction < 0.2


@dataclass
class BeliefState:
    """
    Complete belief state about the current game.

    Updated incrementally from observations.
    """

    # Player knowledge
    player_identified: bool = False
    player_color: Optional[int] = None
    player_position: Optional[tuple[int, int]] = None

    # Color beliefs (what we've learned about each color)
    color_beliefs: dict[int, ColorBelief] = field(default_factory=dict)

    # Position beliefs (what we've learned about each position)
    position_beliefs: dict[tuple[int, int], PositionBelief] = field(default_factory=dict)

    # Game progress
    levels_completed: int = 0
    total_actions: int = 0
    total_observations: int = 0

    # Timer tracking
    timer: TimerState = field(default_factory=TimerState)

    # Exploration tracking
    colors_tested: set[int] = field(default_factory=set)
    positions_visited: set[tuple[int, int]] = field(default_factory=set)

    # Recent history (for LLM context if needed)
    recent_observations: list[Observation] = field(default_factory=list)
    max_recent: int = 20

    def get_color_belief(self, color: int) -> ColorBelief:
        """Get or create belief for a color."""
        if color not in self.color_beliefs:
            self.color_beliefs[color] = ColorBelief(color=color)
        return self.color_beliefs[color]

    def get_position_belief(self, x: int, y: int) -> PositionBelief:
        """Get or create belief for a position."""
        pos = (x, y)
        if pos not in self.position_beliefs:
            self.position_beliefs[pos] = PositionBelief(x=x, y=y)
        return self.position_beliefs[pos]

    def update_timer(self, frame):
        """Update timer state from current frame."""
        import numpy as np
        timer_color = self.timer.timer_color
        timer_pixels = int(np.sum(frame == timer_color))

        # Initialize timer size on first observation with timer pixels
        if not self.timer.initialized and timer_pixels > 0:
            self.timer.initial_size = timer_pixels
            self.timer.current_size = timer_pixels
            self.timer.initialized = True
            return

        if not self.timer.initialized:
            return  # No timer in this game

        # Track timer changes
        if timer_pixels < self.timer.current_size:
            # Timer decreasing
            self.timer.current_size = timer_pixels
        elif timer_pixels > self.timer.current_size and self.timer.current_size == 0:
            # Timer reset (went from 0 to something)
            self.timer.current_size = timer_pixels
            self.timer.last_reset_action = self.total_actions
            if self.timer.resets_remaining > 0:
                self.timer.resets_remaining -= 1
        else:
            self.timer.current_size = timer_pixels

    def update(self, obs: Observation):
        """Update beliefs based on new observation."""
        self.total_observations += 1
        self.total_actions += 1

        # Track recent observations
        self.recent_observations.append(obs)
        if len(self.recent_observations) > self.max_recent:
            self.recent_observations.pop(0)

        # Update player knowledge
        if obs.player_moved:
            self.player_identified = True
            self.player_position = obs.player_new_pos
            if obs.player_new_pos:
                self.positions_visited.add(obs.player_new_pos)

        # Update from blocked movement
        if obs.movement_blocked and obs.blocked_by_color is not None:
            color_belief = self.get_color_belief(obs.blocked_by_color)
            color_belief.times_blocked_movement += 1
            self.colors_tested.add(obs.blocked_by_color)

        # Update from successful movement through a color
        if obs.player_moved and obs.player_new_pos:
            # We successfully moved, so whatever was at new position is walkable
            # (This is implicit - we'd need frame data to know the color)
            pass

        # Update from region disappearance (collectible?)
        if obs.region_disappeared and obs.disappeared_region:
            color = obs.disappeared_region.old_color
            color_belief = self.get_color_belief(color)
            color_belief.times_disappeared_on_touch += 1
            self.colors_tested.add(color)

        # Update from level completion
        if obs.level_completed:
            self.levels_completed += 1
            # If we know player position, mark that color as goal-related
            if self.player_position and obs.player_new_pos:
                # The thing we touched might be the goal
                # (Would need more info to know what color)
                pass

    def get_confident_blockers(self, threshold: float = 0.7) -> list[int]:
        """Get colors we're confident are blockers."""
        return [
            color for color, belief in self.color_beliefs.items()
            if belief.is_blocker >= threshold and belief.total_observations >= 2
        ]

    def get_confident_collectibles(self, threshold: float = 0.7) -> list[int]:
        """Get colors we're confident are collectibles."""
        return [
            color for color, belief in self.color_beliefs.items()
            if belief.is_collectible >= threshold and belief.total_observations >= 1
        ]

    def get_untested_colors(self, all_colors: set[int]) -> set[int]:
        """Get colors we haven't tested yet."""
        return all_colors - self.colors_tested - {0}  # Exclude background

    def get_exploration_coverage(self, grid_size: tuple[int, int]) -> float:
        """What fraction of the grid have we visited?"""
        total_cells = grid_size[0] * grid_size[1]
        return len(self.positions_visited) / total_cells

    def get_uncertainty_score(self) -> float:
        """How uncertain are we overall? Higher = more uncertain."""
        if not self.color_beliefs:
            return 1.0  # Completely uncertain

        # Average confidence across beliefs
        confidences = []
        for belief in self.color_beliefs.values():
            if belief.total_observations > 0:
                # Take max confidence for this color
                max_conf = max(belief.is_blocker, belief.is_collectible, belief.is_goal)
                confidences.append(max_conf)

        if not confidences:
            return 1.0

        avg_confidence = sum(confidences) / len(confidences)
        return 1.0 - avg_confidence

    def to_summary(self) -> str:
        """Generate human-readable summary."""
        timer_status = "OK"
        if self.timer.is_critical:
            timer_status = "CRITICAL"
        elif self.timer.is_urgent:
            timer_status = "URGENT"

        lines = [
            f"=== Belief State (actions={self.total_actions}) ===",
            f"Player: identified={self.player_identified}, pos={self.player_position}, color={self.player_color}",
            f"Timer: {self.timer.current_size}/{self.timer.initial_size} ({self.timer.time_fraction:.0%}) [{timer_status}]",
            f"Levels completed: {self.levels_completed}",
            f"Colors tested: {len(self.colors_tested)}",
            f"Positions visited: {len(self.positions_visited)}",
            "",
            "Color beliefs:",
        ]

        for color, belief in sorted(self.color_beliefs.items()):
            if belief.total_observations > 0:
                lines.append(
                    f"  Color {color}: blocker={belief.is_blocker:.2f}, "
                    f"collectible={belief.is_collectible:.2f}, "
                    f"goal={belief.is_goal:.2f} (n={belief.total_observations})"
                )

        return "\n".join(lines)


def test_belief_state():
    """Test the belief state."""
    from .observation_tracker import Observation, RegionChange

    belief = BeliefState()

    # Simulate: player moved
    obs1 = Observation(
        frame_id=1,
        action_taken=4,  # right
        pixel_changes=[],
        region_changes=[],
        player_moved=True,
        player_old_pos=(5, 5),
        player_new_pos=(6, 5),
    )
    belief.update(obs1)
    print(f"After move: player_identified={belief.player_identified}")

    # Simulate: blocked by grey (color 5)
    obs2 = Observation(
        frame_id=2,
        action_taken=4,  # right
        pixel_changes=[],
        region_changes=[],
        movement_blocked=True,
        blocked_by_color=5,
    )
    belief.update(obs2)
    print(f"After block: grey is_blocker={belief.get_color_belief(5).is_blocker:.2f}")

    # Simulate: collected yellow (color 4)
    obs3 = Observation(
        frame_id=3,
        action_taken=4,
        pixel_changes=[],
        region_changes=[],
        region_disappeared=True,
        disappeared_region=RegionChange(
            positions=[(10, 10)],
            old_color=4,
            new_color=0,
        ),
    )
    belief.update(obs3)
    print(f"After collect: yellow is_collectible={belief.get_color_belief(4).is_collectible:.2f}")

    print("\n" + belief.to_summary())


if __name__ == "__main__":
    test_belief_state()
