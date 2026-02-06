"""
Goal Induction - Discover what the objective is.

Key insight: Goals can be inferred from:
1. Structure - What looks like a target?
2. Behavior - What do successful agents try to reach?
3. Change - What's different when levels complete?
4. Hypothesis testing - Try goals and see what works

This is the hard part of abstract learning.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict, Counter
from enum import Enum
import numpy as np


class GoalType(Enum):
    """Types of goals we might discover."""
    REACH_POSITION = "reach_position"      # Get to a specific location
    REACH_COLOR = "reach_color"            # Touch a specific color
    CLEAR_COLOR = "clear_color"            # Make a color disappear
    FILL_PATTERN = "fill_pattern"          # Fill in a shape
    MATCH_PATTERN = "match_pattern"        # Match controllable to target
    COUNTER_ZERO = "counter_zero"          # Get a counter to zero
    UNKNOWN = "unknown"


@dataclass
class GoalHypothesis:
    """A hypothesis about what the goal might be."""
    goal_type: GoalType
    confidence: float  # 0-1
    evidence: int  # Number of supporting observations

    # Goal-specific parameters
    target_position: Optional[tuple[int, int]] = None
    target_color: Optional[int] = None
    target_pattern: Optional[set[tuple[int, int]]] = None

    def describe(self) -> str:
        if self.goal_type == GoalType.REACH_POSITION:
            return f"Reach position {self.target_position}"
        elif self.goal_type == GoalType.REACH_COLOR:
            return f"Touch/overlap Color {self.target_color}"
        elif self.goal_type == GoalType.CLEAR_COLOR:
            return f"Clear all Color {self.target_color}"
        elif self.goal_type == GoalType.MATCH_PATTERN:
            return f"Match controllable to Color {self.target_color} pattern"
        elif self.goal_type == GoalType.FILL_PATTERN:
            return f"Fill pattern at {len(self.target_pattern or [])} positions"
        else:
            return f"Unknown goal type"

    def check(self, frame: np.ndarray, controllable_color: int) -> bool:
        """Check if goal is achieved in current frame."""
        if self.goal_type == GoalType.REACH_POSITION:
            ctrl_positions = set(zip(*np.where(frame == controllable_color)))
            if self.target_position:
                # Check if controllable overlaps target position (within tolerance)
                ty, tx = self.target_position
                for cy, cx in ctrl_positions:
                    if abs(cy - ty) <= 5 and abs(cx - tx) <= 5:
                        return True
            return False

        elif self.goal_type == GoalType.REACH_COLOR:
            if self.target_color is None:
                return False
            ctrl_positions = set(zip(*np.where(frame == controllable_color)))
            target_positions = set(zip(*np.where(frame == self.target_color)))
            # Check if controllable is adjacent to or overlapping target
            for cy, cx in ctrl_positions:
                for ty, tx in target_positions:
                    if abs(cy - ty) <= 5 and abs(cx - tx) <= 5:
                        return True
            return False

        elif self.goal_type == GoalType.CLEAR_COLOR:
            if self.target_color is None:
                return False
            return np.sum(frame == self.target_color) == 0

        elif self.goal_type == GoalType.MATCH_PATTERN:
            # Check if controllable exactly overlaps target pattern
            if self.target_color is None:
                return False
            ctrl_positions = set(zip(*np.where(frame == controllable_color)))
            target_positions = set(zip(*np.where(frame == self.target_color)))
            # Patterns match if they have similar shape/size
            return len(ctrl_positions) > 0 and ctrl_positions == target_positions

        return False


class StructuralAnalyzer:
    """
    Analyze game structure to find potential goals.

    Goals often have distinctive structural properties:
    - Small isolated regions (targets)
    - Unique colors (special objects)
    - Symmetric patterns (puzzles)
    - Edge positions (exits)
    """

    def find_potential_goals(
        self,
        frame: np.ndarray,
        controllable_color: Optional[int] = None,
    ) -> list[GoalHypothesis]:
        """Find potential goal regions based on structure."""
        hypotheses = []
        h, w = frame.shape

        # Find connected components for each color
        color_components = {}
        for color in range(16):
            positions = list(zip(*np.where(frame == color)))
            if not positions:
                continue

            components = self._find_connected_components(positions)
            color_components[color] = components

        # Analyze each color and its components
        for color, components in color_components.items():
            if color == controllable_color:
                continue
            if color == 0:  # Skip background
                continue

            # Analyze each connected component separately
            for i, comp_positions in enumerate(components):
                count = len(comp_positions)
                if count < 3:  # Too small
                    continue

                ys = [p[0] for p in comp_positions]
                xs = [p[1] for p in comp_positions]
                center = (sum(ys) // count, sum(xs) // count)
                compactness = self._compute_compactness(set(comp_positions))

                # Hypothesis 1: Small compact regions are targets
                if 3 <= count <= 50 and compactness > 0.3:
                    hypotheses.append(GoalHypothesis(
                        goal_type=GoalType.REACH_COLOR,
                        confidence=0.6,
                        evidence=1,
                        target_color=color,
                        target_position=center,
                    ))

                # Hypothesis 2: Regions near controllable might be immediate targets
                if controllable_color and controllable_color in color_components:
                    for ctrl_comp in color_components[controllable_color]:
                        ctrl_ys = [p[0] for p in ctrl_comp]
                        ctrl_xs = [p[1] for p in ctrl_comp]
                        ctrl_center = (sum(ctrl_ys) // len(ctrl_comp), sum(ctrl_xs) // len(ctrl_comp))

                        # Check if this component is close to controllable
                        dist = abs(center[0] - ctrl_center[0]) + abs(center[1] - ctrl_center[1])
                        if dist < 20:  # Close by
                            hypotheses.append(GoalHypothesis(
                                goal_type=GoalType.REACH_COLOR,
                                confidence=0.8,  # High confidence for nearby targets
                                evidence=2,
                                target_color=color,
                                target_position=center,
                            ))

        # Hypothesis 3: Patterns similar to controllable might need matching
        if controllable_color and controllable_color in color_components:
            ctrl_sizes = [len(comp) for comp in color_components[controllable_color]]
            ctrl_total = sum(ctrl_sizes)

            for color, components in color_components.items():
                if color == controllable_color:
                    continue

                for comp in components:
                    # Similar size might mean pattern matching
                    if ctrl_total > 0 and 0.5 <= len(comp) / ctrl_total <= 2.0:
                        ys = [p[0] for p in comp]
                        xs = [p[1] for p in comp]
                        hypotheses.append(GoalHypothesis(
                            goal_type=GoalType.MATCH_PATTERN,
                            confidence=0.5,
                            evidence=1,
                            target_color=color,
                            target_pattern=set(comp),
                            target_position=(sum(ys)//len(ys), sum(xs)//len(xs)),
                        ))

        return hypotheses

    def _find_connected_components(self, positions: list) -> list[list]:
        """Find connected components in a list of positions."""
        from collections import deque

        pos_set = set(positions)
        visited = set()
        components = []

        for start in positions:
            if start in visited:
                continue

            component = []
            queue = deque([start])
            while queue:
                pos = queue.popleft()
                if pos in visited or pos not in pos_set:
                    continue
                visited.add(pos)
                component.append(pos)
                y, x = pos
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    neighbor = (y+dy, x+dx)
                    if neighbor in pos_set and neighbor not in visited:
                        queue.append(neighbor)

            if component:
                components.append(component)

        return components

    def _compute_compactness(self, positions: set) -> float:
        """Compute how compact a region is (1.0 = perfect square)."""
        if not positions:
            return 0.0

        ys = [p[0] for p in positions]
        xs = [p[1] for p in positions]

        height = max(ys) - min(ys) + 1
        width = max(xs) - min(xs) + 1
        bounding_area = height * width

        if bounding_area == 0:
            return 0.0

        return len(positions) / bounding_area


class BehavioralAnalyzer:
    """
    Analyze behavior to infer goals.

    Key insight: Agents move TOWARD goals.
    By tracking movement direction over time, we can infer targets.
    """

    def __init__(self):
        self.position_history: list[tuple[int, int]] = []
        self.action_history: list[int] = []
        self.direction_votes: dict[tuple[int, int], int] = defaultdict(int)

    def observe(self, position: tuple[int, int], action: int):
        """Record agent position and action."""
        self.position_history.append(position)
        self.action_history.append(action)

        # Vote for direction based on action
        direction = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))
        if direction != (0, 0):
            self.direction_votes[direction] += 1

    def infer_target_direction(self) -> Optional[tuple[int, int]]:
        """Infer which direction the agent is trying to go."""
        if not self.direction_votes:
            return None

        # Most common direction
        return max(self.direction_votes.items(), key=lambda x: x[1])[0]

    def infer_target_position(self, frame_shape: tuple[int, int]) -> Optional[tuple[int, int]]:
        """Infer target position from movement patterns."""
        if len(self.position_history) < 5:
            return None

        # Look at net movement over time
        start = self.position_history[0]
        end = self.position_history[-1]

        dy = end[0] - start[0]
        dx = end[1] - start[1]

        # Extrapolate to edge
        h, w = frame_shape

        if dy > 0:  # Moving down
            target_y = h - 10
        elif dy < 0:  # Moving up
            target_y = 10
        else:
            target_y = end[0]

        if dx > 0:  # Moving right
            target_x = w - 10
        elif dx < 0:  # Moving left
            target_x = 10
        else:
            target_x = end[1]

        return (target_y, target_x)


class HypothesisTester:
    """
    Test goal hypotheses through experimentation.

    Strategy:
    1. Generate hypotheses
    2. Plan actions to test each hypothesis
    3. Execute and observe
    4. Update confidence based on results
    """

    def __init__(self):
        self.hypotheses: list[GoalHypothesis] = []
        self.tested: dict[str, int] = {}  # hypothesis_key -> times_tested
        self.successes: dict[str, int] = {}  # hypothesis_key -> times_succeeded

    def add_hypotheses(self, new_hypotheses: list[GoalHypothesis]):
        """Add new hypotheses to test."""
        for h in new_hypotheses:
            key = self._hypothesis_key(h)
            if key not in self.tested:
                self.hypotheses.append(h)
                self.tested[key] = 0
                self.successes[key] = 0

    def _hypothesis_key(self, h: GoalHypothesis) -> str:
        """Create unique key for hypothesis."""
        return f"{h.goal_type.value}:{h.target_color}:{h.target_position}"

    def get_best_hypothesis(self) -> Optional[GoalHypothesis]:
        """Get the most promising hypothesis to pursue."""
        if not self.hypotheses:
            return None

        # Prioritize by confidence and evidence
        def score(h):
            key = self._hypothesis_key(h)
            tests = self.tested.get(key, 0)
            successes = self.successes.get(key, 0)

            # UCB-like exploration bonus
            exploration = 1.0 / (1 + tests) if tests > 0 else 1.0
            exploitation = successes / tests if tests > 0 else h.confidence

            return exploitation + 0.5 * exploration

        return max(self.hypotheses, key=score)

    def record_result(self, hypothesis: GoalHypothesis, success: bool):
        """Record the result of testing a hypothesis."""
        key = self._hypothesis_key(hypothesis)
        self.tested[key] = self.tested.get(key, 0) + 1
        if success:
            self.successes[key] = self.successes.get(key, 0) + 1
            hypothesis.confidence = min(1.0, hypothesis.confidence + 0.2)
            hypothesis.evidence += 1
        else:
            hypothesis.confidence = max(0.0, hypothesis.confidence - 0.1)


class GoalInducer:
    """
    Main goal induction system.

    Combines structural analysis, behavioral analysis, and hypothesis testing
    to discover game objectives.
    """

    def __init__(self):
        self.structural = StructuralAnalyzer()
        self.behavioral = BehavioralAnalyzer()
        self.tester = HypothesisTester()

        self.controllable_color: Optional[int] = None
        self.current_goal: Optional[GoalHypothesis] = None
        self.frames_since_progress = 0
        self.last_position: Optional[tuple[int, int]] = None
        self.stuck_counter = 0
        self.excluded_colors: set[int] = set()  # Colors that are NOT goals

    def set_controllable(self, color: int):
        """Set the controllable object color."""
        self.controllable_color = color

    def exclude_color(self, color: int, reason: str = ""):
        """Mark a color as NOT a goal (e.g., counters, timers)."""
        self.excluded_colors.add(color)
        # Remove any hypotheses targeting this color
        self.tester.hypotheses = [
            h for h in self.tester.hypotheses
            if h.target_color != color
        ]

    def analyze_frame(self, frame: np.ndarray) -> list[GoalHypothesis]:
        """Analyze frame structure to generate goal hypotheses."""
        hypotheses = self.structural.find_potential_goals(frame, self.controllable_color)

        # Filter out excluded colors
        hypotheses = [
            h for h in hypotheses
            if h.target_color not in self.excluded_colors
        ]

        self.tester.add_hypotheses(hypotheses)
        return hypotheses

    def observe_action(self, frame: np.ndarray, action: int):
        """Observe agent action for behavioral analysis."""
        if self.controllable_color is not None:
            positions = list(zip(*np.where(frame == self.controllable_color)))
            if positions:
                center_y = sum(p[0] for p in positions) // len(positions)
                center_x = sum(p[1] for p in positions) // len(positions)
                self.behavioral.observe((center_y, center_x), action)

    def observe_level_complete(self, prev_frame: np.ndarray, curr_frame: np.ndarray):
        """Analyze what happened when level completed."""
        # Strong evidence for current goal hypothesis
        if self.current_goal:
            self.tester.record_result(self.current_goal, success=True)
            self.current_goal.confidence = 1.0
            self.current_goal.evidence += 10

        # Analyze frame differences
        for color in range(16):
            prev_count = np.sum(prev_frame == color)
            curr_count = np.sum(curr_frame == color)

            # Color disappeared - might have been the goal
            if prev_count > 0 and curr_count == 0:
                self.tester.add_hypotheses([GoalHypothesis(
                    goal_type=GoalType.CLEAR_COLOR,
                    confidence=0.9,
                    evidence=10,
                    target_color=color,
                )])

    def get_goal(self) -> Optional[GoalHypothesis]:
        """Get the current best goal hypothesis."""
        # If we have a high-confidence goal, stick with it
        if self.current_goal and self.current_goal.confidence >= 0.7:
            return self.current_goal

        # Otherwise, get best hypothesis
        best = self.tester.get_best_hypothesis()
        if best and best.confidence >= 0.3:
            self.current_goal = best
            return best

        return None

    def get_action_toward_goal(
        self,
        frame: np.ndarray,
        goal: GoalHypothesis,
    ) -> Optional[int]:
        """Get action to move toward goal."""
        if self.controllable_color is None:
            return None

        # Find controllable position
        ctrl_positions = list(zip(*np.where(frame == self.controllable_color)))
        if not ctrl_positions:
            return None

        ctrl_y = sum(p[0] for p in ctrl_positions) // len(ctrl_positions)
        ctrl_x = sum(p[1] for p in ctrl_positions) // len(ctrl_positions)
        current_pos = (ctrl_y, ctrl_x)

        # Detect if stuck
        if self.last_position == current_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.frames_since_progress = 0
        self.last_position = current_pos

        # If stuck for too long, mark current goal as failed and try next
        if self.stuck_counter > 5:
            self.tester.record_result(goal, success=False)
            self.current_goal = None
            self.stuck_counter = 0
            return None  # Signal to try next hypothesis

        # Find target position
        target_y, target_x = None, None

        if goal.target_position:
            target_y, target_x = int(goal.target_position[0]), int(goal.target_position[1])
        elif goal.target_color is not None:
            target_positions = list(zip(*np.where(frame == goal.target_color)))
            if target_positions:
                target_y = sum(p[0] for p in target_positions) // len(target_positions)
                target_x = sum(p[1] for p in target_positions) // len(target_positions)

        if target_y is None or target_x is None:
            return None

        # Choose action to move toward target
        dy = target_y - ctrl_y
        dx = target_x - ctrl_x

        # If very close, might have reached goal
        if abs(dy) <= 5 and abs(dx) <= 5:
            self.frames_since_progress = 0
            # Mark as potential success (we reached the target)
            self.tester.record_result(goal, success=True)
            return None  # At goal

        if abs(dy) > abs(dx):
            return 2 if dy > 0 else 1  # DOWN or UP
        else:
            return 4 if dx > 0 else 3  # RIGHT or LEFT

    def next_hypothesis(self) -> Optional[GoalHypothesis]:
        """Get next untested or promising hypothesis."""
        # Sort by: untested first, then by confidence
        candidates = sorted(
            self.tester.hypotheses,
            key=lambda h: (self.tester.tested.get(self.tester._hypothesis_key(h), 0), -h.confidence)
        )

        for h in candidates:
            if h.target_color not in self.excluded_colors:
                self.current_goal = h
                return h

        return None

    def describe(self) -> str:
        """Describe current goal understanding."""
        lines = ["=== Goal Induction ==="]

        if self.current_goal:
            lines.append(f"Current goal: {self.current_goal.describe()}")
            lines.append(f"  Confidence: {self.current_goal.confidence:.0%}")
            lines.append(f"  Evidence: {self.current_goal.evidence}")
        else:
            lines.append("No goal identified yet")

        lines.append("")
        lines.append("All hypotheses:")
        for h in sorted(self.tester.hypotheses, key=lambda x: -x.confidence)[:5]:
            key = self.tester._hypothesis_key(h)
            tests = self.tester.tested.get(key, 0)
            lines.append(f"  [{h.confidence:.0%}] {h.describe()} (tested {tests}x)")

        return "\n".join(lines)


def test_goal_induction():
    """Test goal induction on synthetic data."""
    import numpy as np

    # Create synthetic frame with clear structure
    frame = np.zeros((64, 64), dtype=np.int32)
    frame[:, :] = 3  # Floor
    frame[0:5, :] = 4  # Top wall
    frame[59:64, :] = 4  # Bottom wall
    frame[:, 0:5] = 4  # Left wall
    frame[:, 59:64] = 4  # Right wall

    # Controllable object (color 12)
    frame[30:32, 30:35] = 12

    # Potential target (color 9) - small isolated region
    frame[10:13, 50:53] = 9

    # Test goal inducer
    inducer = GoalInducer()
    inducer.set_controllable(12)

    hypotheses = inducer.analyze_frame(frame)
    print("Generated hypotheses:")
    for h in hypotheses:
        print(f"  {h.describe()} (confidence: {h.confidence:.0%})")

    goal = inducer.get_goal()
    if goal:
        print(f"\nBest goal: {goal.describe()}")
        action = inducer.get_action_toward_goal(frame, goal)
        print(f"Suggested action: {action}")

    print("\n" + inducer.describe())


if __name__ == "__main__":
    test_goal_induction()
