"""
Abstract Game Learner - Discovers game rules from observation.

No assumptions about game type. Learns everything from scratch:
1. What do actions do?
2. What are the objects?
3. What are the rules?
4. What is the goal?

Key insight: Observe (state, action, next_state) triples and infer rules.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict
from enum import Enum
import numpy as np


class RuleType(Enum):
    """Types of rules we can learn."""
    MOVE = "move"           # Object moves in response to action
    TRANSFORM = "transform"  # Object changes color
    DISAPPEAR = "disappear"  # Object disappears
    APPEAR = "appear"        # Object appears
    BLOCKED = "blocked"      # Movement blocked by something
    COUNTER = "counter"      # Something counts down/up
    WIN = "win"             # Triggers level completion
    LOSE = "lose"           # Triggers game over


@dataclass
class GameObject:
    """A detected object in the game."""
    object_id: int
    color: int
    positions: set[tuple[int, int]]

    @property
    def center(self) -> tuple[int, int]:
        if not self.positions:
            return (0, 0)
        ys = [p[0] for p in self.positions]
        xs = [p[1] for p in self.positions]
        return (sum(ys) // len(ys), sum(xs) // len(xs))

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """Return (min_y, max_y, min_x, max_x)."""
        if not self.positions:
            return (0, 0, 0, 0)
        ys = [p[0] for p in self.positions]
        xs = [p[1] for p in self.positions]
        return (min(ys), max(ys), min(xs), max(xs))

    @property
    def size(self) -> int:
        return len(self.positions)


@dataclass
class LearnedRule:
    """A rule learned from observation."""
    rule_type: RuleType
    confidence: float  # 0-1, how confident we are
    observations: int  # How many times we've seen this

    # Rule-specific parameters
    action: Optional[int] = None  # Which action triggers this
    subject_color: Optional[int] = None  # What color is affected
    direction: Optional[tuple[int, int]] = None  # (dy, dx) for movement
    magnitude: Optional[int] = None  # How much (e.g., step size)
    condition_color: Optional[int] = None  # Color that blocks/triggers

    def describe(self) -> str:
        """Human-readable description."""
        if self.rule_type == RuleType.MOVE:
            dir_name = {(-1, 0): "UP", (1, 0): "DOWN", (0, -1): "LEFT", (0, 1): "RIGHT"}
            d = dir_name.get(self.direction, str(self.direction))
            return f"Action {self.action} moves Color {self.subject_color} {d} by {self.magnitude}"
        elif self.rule_type == RuleType.BLOCKED:
            return f"Color {self.subject_color} blocked by Color {self.condition_color}"
        elif self.rule_type == RuleType.COUNTER:
            return f"Color {self.subject_color} decreases by {self.magnitude} each action"
        elif self.rule_type == RuleType.WIN:
            return f"Win condition: {self.subject_color}"
        elif self.rule_type == RuleType.LOSE:
            return f"Lose when Color {self.subject_color} reaches 0"
        else:
            return f"{self.rule_type.value}: color={self.subject_color}"


@dataclass
class ActionEffect:
    """Observed effect of an action."""
    action: int

    # Object movements
    movements: dict[int, tuple[int, int]] = field(default_factory=dict)  # color -> (dy, dx)

    # Size changes (counters)
    size_changes: dict[int, int] = field(default_factory=dict)  # color -> delta

    # Blocked movements
    blocked: dict[int, int] = field(default_factory=dict)  # color -> blocking_color

    # State changes
    level_completed: bool = False
    game_over: bool = False


class ObjectDetector:
    """Detect and track objects in frames."""

    def detect_objects(self, frame: np.ndarray) -> list[GameObject]:
        """Find all distinct objects (connected components of same color)."""
        objects = []
        visited = set()
        object_id = 0

        h, w = frame.shape

        for y in range(h):
            for x in range(w):
                if (y, x) in visited:
                    continue

                color = int(frame[y, x])

                # Flood fill to find connected component
                positions = self._flood_fill(frame, y, x, color, visited)

                if len(positions) > 0:  # Ignore background (color 0) or very small
                    objects.append(GameObject(
                        object_id=object_id,
                        color=color,
                        positions=positions,
                    ))
                    object_id += 1

        return objects

    def _flood_fill(
        self,
        frame: np.ndarray,
        start_y: int,
        start_x: int,
        color: int,
        visited: set,
    ) -> set[tuple[int, int]]:
        """Flood fill to find connected component."""
        h, w = frame.shape
        positions = set()
        stack = [(start_y, start_x)]

        while stack:
            y, x = stack.pop()

            if (y, x) in visited:
                continue
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            if frame[y, x] != color:
                continue

            visited.add((y, x))
            positions.add((y, x))

            # Add neighbors
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

        return positions

    def track_objects(
        self,
        prev_objects: list[GameObject],
        curr_objects: list[GameObject],
    ) -> dict[int, int]:
        """Match objects between frames. Returns {prev_id: curr_id}."""
        matches = {}

        # Simple matching by color and proximity
        for prev_obj in prev_objects:
            best_match = None
            best_distance = float('inf')

            for curr_obj in curr_objects:
                if curr_obj.color != prev_obj.color:
                    continue
                if curr_obj.object_id in matches.values():
                    continue

                # Calculate distance between centers
                py, px = prev_obj.center
                cy, cx = curr_obj.center
                dist = abs(py - cy) + abs(px - cx)

                if dist < best_distance:
                    best_distance = dist
                    best_match = curr_obj.object_id

            if best_match is not None:
                matches[prev_obj.object_id] = best_match

        return matches


class RuleInducer:
    """Infer rules from observed state transitions."""

    def __init__(self):
        self.detector = ObjectDetector()
        self.learned_rules: list[LearnedRule] = []
        self.observations: list[tuple[np.ndarray, int, np.ndarray, bool, bool]] = []

        # Track patterns
        self.action_effects: dict[int, list[ActionEffect]] = defaultdict(list)
        self.color_movements: dict[int, dict[int, list[tuple[int, int]]]] = defaultdict(lambda: defaultdict(list))
        self.color_size_history: dict[int, list[int]] = defaultdict(list)

    def observe(
        self,
        prev_frame: np.ndarray,
        action: int,
        curr_frame: np.ndarray,
        level_completed: bool = False,
        game_over: bool = False,
    ):
        """Record an observation and update rule hypotheses."""
        self.observations.append((prev_frame.copy(), action, curr_frame.copy(), level_completed, game_over))

        # Detect objects in both frames
        prev_objects = self.detector.detect_objects(prev_frame)
        curr_objects = self.detector.detect_objects(curr_frame)

        # Track objects
        matches = self.detector.track_objects(prev_objects, curr_objects)

        # Build action effect
        effect = ActionEffect(action=action, level_completed=level_completed, game_over=game_over)

        # Analyze movements
        prev_by_color = {obj.color: obj for obj in prev_objects}
        curr_by_color = {obj.color: obj for obj in curr_objects}

        for color in set(prev_by_color.keys()) | set(curr_by_color.keys()):
            prev_obj = prev_by_color.get(color)
            curr_obj = curr_by_color.get(color)

            if prev_obj and curr_obj:
                # Check for movement
                py, px = prev_obj.center
                cy, cx = curr_obj.center
                dy, dx = cy - py, cx - px

                if dy != 0 or dx != 0:
                    effect.movements[color] = (dy, dx)
                    self.color_movements[color][action].append((dy, dx))
                elif action in [1, 2, 3, 4]:
                    # Movement action but no movement - might be blocked
                    effect.blocked[color] = self._find_blocking_color(curr_frame, curr_obj, action)

                # Check for size change (counters)
                size_delta = curr_obj.size - prev_obj.size
                if size_delta != 0:
                    effect.size_changes[color] = size_delta

            # Track size history for counter detection
            if curr_obj:
                self.color_size_history[color].append(curr_obj.size)

        self.action_effects[action].append(effect)

        # Update rules based on new observation
        self._update_rules()

    def _find_blocking_color(self, frame: np.ndarray, obj: GameObject, action: int) -> Optional[int]:
        """Find what color might be blocking movement."""
        dy, dx = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}.get(action, (0, 0))

        # Check pixels in the movement direction
        h, w = frame.shape
        for y, x in obj.positions:
            check_y = y + dy * 5  # Assume ~5 pixel steps
            check_x = x + dx * 5
            if 0 <= check_y < h and 0 <= check_x < w:
                blocking = int(frame[check_y, check_x])
                if blocking != obj.color:
                    return blocking

        return None

    def _update_rules(self):
        """Update learned rules based on all observations."""
        self.learned_rules = []

        # Rule 1: Movement rules - "Action X moves Color Y in direction Z"
        for color, action_movements in self.color_movements.items():
            for action, movements in action_movements.items():
                if len(movements) >= 2:  # Need multiple observations
                    # Find most common movement
                    from collections import Counter
                    common = Counter(movements).most_common(1)[0]
                    movement, count = common

                    if count >= 2 and (movement[0] != 0 or movement[1] != 0):
                        magnitude = max(abs(movement[0]), abs(movement[1]))
                        direction = (
                            movement[0] // magnitude if movement[0] != 0 else 0,
                            movement[1] // magnitude if movement[1] != 0 else 0,
                        )

                        self.learned_rules.append(LearnedRule(
                            rule_type=RuleType.MOVE,
                            confidence=count / len(movements),
                            observations=count,
                            action=action,
                            subject_color=color,
                            direction=direction,
                            magnitude=magnitude,
                        ))

        # Rule 2: Counter rules - "Color X decreases by Y each action"
        for color, sizes in self.color_size_history.items():
            if len(sizes) >= 3:
                # Check if consistently decreasing
                deltas = [sizes[i+1] - sizes[i] for i in range(len(sizes)-1)]
                if deltas and all(d == deltas[0] for d in deltas) and deltas[0] != 0:
                    self.learned_rules.append(LearnedRule(
                        rule_type=RuleType.COUNTER,
                        confidence=1.0,
                        observations=len(deltas),
                        subject_color=color,
                        magnitude=deltas[0],
                    ))

        # Rule 3: Blocking rules
        for action, effects in self.action_effects.items():
            blocking_counts = defaultdict(lambda: defaultdict(int))
            for effect in effects:
                for color, blocker in effect.blocked.items():
                    if blocker is not None:
                        blocking_counts[color][blocker] += 1

            for color, blockers in blocking_counts.items():
                for blocker, count in blockers.items():
                    if count >= 2:
                        self.learned_rules.append(LearnedRule(
                            rule_type=RuleType.BLOCKED,
                            confidence=count / len(effects),
                            observations=count,
                            subject_color=color,
                            condition_color=blocker,
                        ))

        # Rule 4: Win/Lose conditions
        for prev_frame, action, curr_frame, level_completed, game_over in self.observations:
            if level_completed:
                # What was special about this transition?
                self.learned_rules.append(LearnedRule(
                    rule_type=RuleType.WIN,
                    confidence=0.5,
                    observations=1,
                ))

            if game_over:
                # Check if any counter hit 0
                for color, sizes in self.color_size_history.items():
                    if sizes and sizes[-1] == 0:
                        self.learned_rules.append(LearnedRule(
                            rule_type=RuleType.LOSE,
                            confidence=0.8,
                            observations=1,
                            subject_color=color,
                        ))

    def get_rules(self) -> list[LearnedRule]:
        """Get all learned rules, sorted by confidence."""
        return sorted(self.learned_rules, key=lambda r: -r.confidence)

    def describe_rules(self) -> str:
        """Human-readable description of learned rules."""
        lines = ["=== Learned Rules ==="]
        for rule in self.get_rules():
            lines.append(f"  [{rule.confidence:.0%}] {rule.describe()}")
        return "\n".join(lines)

    def predict_effect(self, frame: np.ndarray, action: int) -> dict:
        """Predict what will happen if we take this action."""
        predictions = {
            "movements": {},
            "blocked": set(),
            "counter_changes": {},
        }

        for rule in self.learned_rules:
            if rule.rule_type == RuleType.MOVE and rule.action == action:
                if rule.confidence >= 0.5:
                    predictions["movements"][rule.subject_color] = (
                        rule.direction[0] * rule.magnitude,
                        rule.direction[1] * rule.magnitude,
                    )

            elif rule.rule_type == RuleType.COUNTER:
                predictions["counter_changes"][rule.subject_color] = rule.magnitude

        return predictions


class GoalDiscoverer:
    """
    Discover what the goal of the game might be.

    Strategies:
    1. Look for patterns when levels complete
    2. Identify "special" regions (small unique colors)
    3. Track what changes correlate with score increases
    """

    def __init__(self):
        self.win_observations: list[tuple[np.ndarray, np.ndarray]] = []
        self.hypotheses: list[dict] = []

    def observe_win(self, prev_frame: np.ndarray, curr_frame: np.ndarray):
        """Record what the state looked like when a level was completed."""
        self.win_observations.append((prev_frame.copy(), curr_frame.copy()))
        self._update_hypotheses()

    def _update_hypotheses(self):
        """Generate hypotheses about win conditions."""
        self.hypotheses = []

        for prev, curr in self.win_observations:
            # Hypothesis 1: Some color disappeared completely
            for color in range(16):
                prev_count = np.sum(prev == color)
                curr_count = np.sum(curr == color)
                if prev_count > 0 and curr_count == 0:
                    self.hypotheses.append({
                        "type": "color_cleared",
                        "color": color,
                        "confidence": 0.7,
                    })

            # Hypothesis 2: Moveable object reached a position
            # (Would need more info)

    def get_goal_regions(self, frame: np.ndarray) -> list[tuple[int, set]]:
        """Identify potential goal regions (small unique colors)."""
        goals = []
        for color in range(16):
            positions = set(zip(*np.where(frame == color)))
            count = len(positions)
            # Small isolated regions might be goals
            if 0 < count < 50:
                goals.append((color, positions))
        return goals


class WorldModelSimulator:
    """
    Simulate future states using learned rules.

    This allows planning without actually taking actions.
    """

    def __init__(self, rule_inducer: RuleInducer):
        self.rule_inducer = rule_inducer
        self.detector = ObjectDetector()

    def simulate(self, frame: np.ndarray, action: int) -> np.ndarray:
        """Predict the next frame after taking an action."""
        # This is a simplified simulation
        # In reality, we'd need more sophisticated physics

        next_frame = frame.copy()

        # Apply movement rules
        for rule in self.rule_inducer.learned_rules:
            if rule.rule_type == RuleType.MOVE and rule.action == action:
                if rule.confidence >= 0.5:
                    # Find objects of this color
                    positions = set(zip(*np.where(frame == rule.subject_color)))
                    if positions:
                        # Calculate new positions
                        dy = rule.direction[0] * rule.magnitude
                        dx = rule.direction[1] * rule.magnitude

                        # Check if movement is blocked
                        blocked = False
                        new_positions = set()
                        for y, x in positions:
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < frame.shape[0] and 0 <= nx < frame.shape[1]:
                                # Check what's at the new position
                                target_color = frame[ny, nx]
                                if target_color != rule.subject_color and target_color != 0:
                                    # Check if this color blocks
                                    for block_rule in self.rule_inducer.learned_rules:
                                        if (block_rule.rule_type == RuleType.BLOCKED and
                                            block_rule.subject_color == rule.subject_color and
                                            block_rule.condition_color == target_color):
                                            blocked = True
                                            break
                                new_positions.add((ny, nx))
                            else:
                                blocked = True

                        if not blocked:
                            # Clear old positions
                            for y, x in positions:
                                next_frame[y, x] = 0  # Or background color
                            # Set new positions
                            for y, x in new_positions:
                                next_frame[y, x] = rule.subject_color

        return next_frame

    def plan(
        self,
        start_frame: np.ndarray,
        goal_check: Callable[[np.ndarray], bool],
        max_depth: int = 20,
    ) -> Optional[list[int]]:
        """
        Plan a sequence of actions to reach a goal.

        Uses breadth-first search with world model simulation.
        """
        from collections import deque

        # BFS
        queue = deque([(start_frame, [])])
        visited = set()

        while queue and len(visited) < 1000:
            frame, actions = queue.popleft()

            # Hash frame for visited check
            frame_hash = hash(frame.tobytes())
            if frame_hash in visited:
                continue
            visited.add(frame_hash)

            # Check if goal reached
            if goal_check(frame):
                return actions

            # Explore actions
            if len(actions) < max_depth:
                for action in [1, 2, 3, 4]:
                    next_frame = self.simulate(frame, action)
                    queue.append((next_frame, actions + [action]))

        return None


class AbstractGameAgent:
    """
    Agent that learns game rules abstractly.

    No assumptions about game type - discovers everything through play.

    Phases:
    1. Exploration: Learn what actions do
    2. Hypothesis: Generate hypotheses about goals
    3. Planning: Use world model to plan action sequences
    4. Exploitation: Execute plans
    """

    def __init__(self, exploration_budget: int = 16):
        self.rule_inducer = RuleInducer()
        self.goal_discoverer = GoalDiscoverer()
        self.world_model: Optional[WorldModelSimulator] = None

        self.prev_frame: Optional[np.ndarray] = None
        self.prev_action: Optional[int] = None
        self.action_count = 0

        # Phase control
        self.exploration_budget = exploration_budget
        self.exploration_actions_remaining = exploration_budget
        self.exploration_sequence = self._create_exploration_sequence()
        self.exploration_idx = 0

        # Learned state
        self.controllable_color: Optional[int] = None
        self.current_plan: list[int] = []

    def _create_exploration_sequence(self) -> list[int]:
        """Create a sequence of actions that explores each direction."""
        # Try each action multiple times to learn consistent effects
        sequence = []
        for _ in range(4):  # 4 repetitions
            for action in [1, 2, 3, 4]:
                sequence.append(action)
        return sequence

    def reset(self):
        """Reset for new episode."""
        self.rule_inducer = RuleInducer()
        self.goal_discoverer = GoalDiscoverer()
        self.world_model = None
        self.prev_frame = None
        self.prev_action = None
        self.action_count = 0
        self.exploration_actions_remaining = self.exploration_budget
        self.exploration_idx = 0
        self.controllable_color = None
        self.current_plan = []

    def act(
        self,
        frame: np.ndarray,
        level_completed: bool = False,
        game_over: bool = False,
    ) -> int:
        """Choose action based on learned rules."""
        self.action_count += 1

        # Record observation from previous action
        if self.prev_frame is not None and self.prev_action is not None:
            self.rule_inducer.observe(
                self.prev_frame, self.prev_action, frame,
                level_completed, game_over
            )

            if level_completed:
                self.goal_discoverer.observe_win(self.prev_frame, frame)

        # Phase 1: Exploration - learn what actions do
        if self.exploration_actions_remaining > 0:
            self.exploration_actions_remaining -= 1
            action = self.exploration_sequence[self.exploration_idx % len(self.exploration_sequence)]
            self.exploration_idx += 1
        else:
            # Build world model after exploration is complete
            # We wait one extra action to ensure all observations are recorded
            if self.world_model is None:
                self.world_model = WorldModelSimulator(self.rule_inducer)
                self._identify_controllable()
                # Debug output
                print(f"[AbstractAgent] Identified controllable: Color {self.controllable_color}")

            # Phase 2: Use learned rules to choose action
            action = self._choose_action(frame)

        # Store for next observation
        self.prev_frame = frame.copy()
        self.prev_action = action

        return action

    def _identify_controllable(self):
        """Identify which color is the controllable object."""
        # Find color that moves consistently with different actions
        movement_counts = {}
        for rule in self.rule_inducer.learned_rules:
            if rule.rule_type == RuleType.MOVE:
                color = rule.subject_color
                movement_counts[color] = movement_counts.get(color, 0) + 1

        # Color with most movement rules is likely controllable
        if movement_counts:
            best_color, best_count = max(movement_counts.items(), key=lambda x: x[1])
            # Only consider controllable if it responds to multiple different actions
            if best_count >= 2:
                self.controllable_color = best_color
            print(f"[AbstractAgent] Movement counts: {movement_counts}")

    def _choose_action(self, frame: np.ndarray) -> int:
        """Choose action using learned rules."""
        # If we have a plan, follow it
        if self.current_plan:
            return self.current_plan.pop(0)

        # Strategy: Systematic exploration of reachable positions
        # Since we don't know the goal, try to visit every position

        if self.controllable_color is not None:
            ctrl_positions = list(zip(*np.where(frame == self.controllable_color)))
            if ctrl_positions:
                ctrl_y = sum(p[0] for p in ctrl_positions) // len(ctrl_positions)
                ctrl_x = sum(p[1] for p in ctrl_positions) // len(ctrl_positions)

                # Track visited positions
                if not hasattr(self, 'visited_positions'):
                    self.visited_positions = set()
                    self.stuck_counter = 0
                    self.last_position = None

                current_pos = (ctrl_y // 5, ctrl_x // 5)  # Quantize to grid
                self.visited_positions.add(current_pos)

                # Detect if stuck
                if self.last_position == current_pos:
                    self.stuck_counter += 1
                else:
                    self.stuck_counter = 0
                self.last_position = current_pos

                # If stuck, try random action
                if self.stuck_counter > 3:
                    self.stuck_counter = 0
                    return np.random.choice([1, 2, 3, 4])

                # Try to find goal regions (small isolated colors)
                goals = self.goal_discoverer.get_goal_regions(frame)

                # Prioritize reaching small colored regions that aren't blockers
                for color, positions in sorted(goals, key=lambda x: len(x[1])):
                    if color == self.controllable_color:
                        continue
                    # Check if this color is known to block
                    is_blocker = False
                    for rule in self.rule_inducer.learned_rules:
                        if (rule.rule_type == RuleType.BLOCKED and
                            rule.condition_color == color):
                            is_blocker = True
                            break
                    if is_blocker:
                        continue

                    # Try to reach this region
                    goal_y = sum(p[0] for p in positions) // len(positions)
                    goal_x = sum(p[1] for p in positions) // len(positions)

                    dy = goal_y - ctrl_y
                    dx = goal_x - ctrl_x

                    # Check if we can move in the desired direction
                    if abs(dy) > 3 or abs(dx) > 3:
                        if abs(dy) > abs(dx):
                            action = 2 if dy > 0 else 1
                        else:
                            action = 4 if dx > 0 else 3
                        return action

                # Exploration: try to reach unvisited positions
                # Check all 4 directions for unvisited positions
                directions = [
                    (1, (ctrl_y - 5) // 5, ctrl_x // 5),  # UP
                    (2, (ctrl_y + 5) // 5, ctrl_x // 5),  # DOWN
                    (3, ctrl_y // 5, (ctrl_x - 5) // 5),  # LEFT
                    (4, ctrl_y // 5, (ctrl_x + 5) // 5),  # RIGHT
                ]

                for action, ny, nx in directions:
                    if (ny, nx) not in self.visited_positions:
                        return action

        # Fallback: cycle through actions
        return (self.action_count % 4) + 1

    def observe_effect(
        self,
        prev_frame: np.ndarray,
        action: int,
        curr_frame: np.ndarray,
        level_completed: bool = False,
        game_over: bool = False,
    ):
        """Observe the effect of an action (call after act + env.step)."""
        self.rule_inducer.observe(prev_frame, action, curr_frame, level_completed, game_over)

    def get_learned_rules(self) -> str:
        """Get description of learned rules."""
        return self.rule_inducer.describe_rules()


def test_abstract_learner():
    """Test the abstract learner on synthetic data."""
    learner = RuleInducer()

    # Simulate observations of a block moving
    for i in range(10):
        # Create frames where a block moves down with action 2
        prev_frame = np.zeros((64, 64), dtype=np.int32)
        prev_frame[20+i*5:25+i*5, 30:35] = 12  # Block

        curr_frame = np.zeros((64, 64), dtype=np.int32)
        curr_frame[25+i*5:30+i*5, 30:35] = 12  # Block moved down

        learner.observe(prev_frame, action=2, curr_frame=curr_frame)

    print(learner.describe_rules())


if __name__ == "__main__":
    test_abstract_learner()
