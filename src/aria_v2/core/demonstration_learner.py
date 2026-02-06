"""
Demonstration Learning - Learn goals from human gameplay.

Key insight: When humans succeed at a game, their trajectories reveal:
1. What they were trying to achieve (goal)
2. How they navigated obstacles (strategy)
3. What mattered vs. what was ignored (relevance)

This module analyzes recorded trajectories to bootstrap goal understanding.
"""

from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict, Counter
from enum import Enum
import numpy as np
import json
from pathlib import Path


@dataclass
class TrajectoryFrame:
    """Single frame in a recorded trajectory."""
    frame_id: int
    frame: np.ndarray  # [H, W] grid
    action_taken: Optional[int]  # Action after this frame
    timestamp: float = 0.0

    # Computed properties (filled during analysis)
    player_position: Optional[tuple[int, int]] = None
    player_color: Optional[int] = None


@dataclass
class Trajectory:
    """Complete recorded trajectory (episode)."""
    game_id: str
    frames: list[TrajectoryFrame]
    success: bool  # Did the episode complete successfully?
    total_reward: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def length(self) -> int:
        return len(self.frames)

    @property
    def actions(self) -> list[int]:
        return [f.action_taken for f in self.frames if f.action_taken is not None]


@dataclass
class DemonstrationInsight:
    """An insight derived from demonstrations."""
    insight_type: str  # "goal", "strategy", "obstacle", "irrelevant"
    description: str
    confidence: float
    evidence_count: int
    target_color: Optional[int] = None
    target_position: Optional[tuple[int, int]] = None


class LevelTransition:
    """Captures the state change at a level completion boundary."""
    level_number: int
    frame_before: np.ndarray
    frame_after: np.ndarray
    action_at_transition: Optional[int]
    colors_disappeared: list[int]
    colors_changed: list[tuple[int, int, int]]  # (color, count_before, count_after)
    player_position_at_completion: Optional[tuple[int, int]]

    def __init__(self, level_number, frame_before, frame_after, action_at_transition=None):
        self.level_number = level_number
        self.frame_before = frame_before
        self.frame_after = frame_after
        self.action_at_transition = action_at_transition
        self.colors_disappeared = []
        self.colors_changed = []
        self.player_position_at_completion = None


class TrajectoryAnalyzer:
    """
    Analyzes individual trajectories to extract patterns.
    """

    def __init__(self):
        self.step_size = 5  # Movement step size in ARC games

    def identify_player(self, trajectory: Trajectory) -> Optional[int]:
        """Identify which color is the player by finding what responds to actions."""
        if len(trajectory.frames) < 2:
            return None

        # Track color movements
        color_movements = defaultdict(int)

        for i in range(len(trajectory.frames) - 1):
            prev_frame = trajectory.frames[i].frame
            curr_frame = trajectory.frames[i + 1].frame
            action = trajectory.frames[i].action_taken

            if action not in [1, 2, 3, 4]:  # Movement actions
                continue

            # Check each color for movement
            for color in range(1, 16):
                prev_positions = set(zip(*np.where(prev_frame == color)))
                curr_positions = set(zip(*np.where(curr_frame == color)))

                if not prev_positions or not curr_positions:
                    continue

                # Compute center movement
                prev_center = (
                    sum(p[0] for p in prev_positions) // len(prev_positions),
                    sum(p[1] for p in prev_positions) // len(prev_positions)
                )
                curr_center = (
                    sum(p[0] for p in curr_positions) // len(curr_positions),
                    sum(p[1] for p in curr_positions) // len(curr_positions)
                )

                dy = curr_center[0] - prev_center[0]
                dx = curr_center[1] - prev_center[1]

                # Check if movement matches action
                expected = {1: (-self.step_size, 0), 2: (self.step_size, 0),
                           3: (0, -self.step_size), 4: (0, self.step_size)}

                if action in expected:
                    exp_dy, exp_dx = expected[action]
                    if abs(dy - exp_dy) <= 2 and abs(dx - exp_dx) <= 2:
                        color_movements[color] += 1

        if not color_movements:
            return None

        # Return color with most consistent movements
        return max(color_movements.items(), key=lambda x: x[1])[0]

    def find_goal_from_success(
        self,
        trajectory: Trajectory,
        player_color: int
    ) -> list[DemonstrationInsight]:
        """
        Analyze what happened at the end of a successful trajectory.
        """
        if not trajectory.success or len(trajectory.frames) < 2:
            return []

        insights = []

        # Compare final frames
        final_frame = trajectory.frames[-1].frame
        second_last_frame = trajectory.frames[-2].frame

        # What changed in the final action?
        for color in range(1, 16):
            if color == player_color:
                continue

            prev_count = np.sum(second_last_frame == color)
            curr_count = np.sum(final_frame == color)

            # Color disappeared - likely the goal
            if prev_count > 0 and curr_count == 0:
                insights.append(DemonstrationInsight(
                    insight_type="goal",
                    description=f"Color {color} disappeared when level completed",
                    confidence=0.9,
                    evidence_count=1,
                    target_color=color,
                ))

            # Color changed significantly
            elif abs(prev_count - curr_count) > prev_count * 0.3:
                insights.append(DemonstrationInsight(
                    insight_type="goal",
                    description=f"Color {color} changed significantly ({prev_count} -> {curr_count})",
                    confidence=0.7,
                    evidence_count=1,
                    target_color=color,
                ))

        # Check player final position
        player_positions = list(zip(*np.where(final_frame == player_color)))
        if player_positions:
            center_y = sum(p[0] for p in player_positions) // len(player_positions)
            center_x = sum(p[1] for p in player_positions) // len(player_positions)

            # Check what colors player was near
            for color in range(1, 16):
                if color == player_color:
                    continue

                color_positions = list(zip(*np.where(final_frame == color)))
                if not color_positions:
                    continue

                # Check proximity
                for cy, cx in color_positions:
                    if abs(cy - center_y) <= 5 and abs(cx - center_x) <= 5:
                        insights.append(DemonstrationInsight(
                            insight_type="goal",
                            description=f"Player reached Color {color} at level completion",
                            confidence=0.85,
                            evidence_count=1,
                            target_color=color,
                            target_position=(center_y, center_x),
                        ))
                        break

        return insights

    def find_approach_patterns(
        self,
        trajectory: Trajectory,
        player_color: int
    ) -> list[DemonstrationInsight]:
        """
        Analyze what the player was moving toward.

        Filters out large background colors (>15% of pixels) that are
        unlikely to be goals.
        """
        if len(trajectory.frames) < 5:
            return []

        # Track player positions over time
        player_positions = []
        for frame_data in trajectory.frames:
            positions = list(zip(*np.where(frame_data.frame == player_color)))
            if positions:
                center = (
                    sum(p[0] for p in positions) // len(positions),
                    sum(p[1] for p in positions) // len(positions)
                )
                player_positions.append(center)

        if len(player_positions) < 5:
            return []

        # Compute net direction
        start = player_positions[0]
        end = player_positions[-1]

        net_dy = end[0] - start[0]
        net_dx = end[1] - start[1]

        insights = []

        # What colors are in the direction of movement?
        final_frame = trajectory.frames[-1].frame
        total_pixels = final_frame.shape[0] * final_frame.shape[1]

        for color in range(1, 16):
            if color == player_color:
                continue

            color_positions = list(zip(*np.where(final_frame == color)))
            if not color_positions:
                continue

            # Skip large background/wall colors (>15% of pixels)
            if len(color_positions) / total_pixels > 0.15:
                continue

            # Compute center of this color
            color_center = (
                sum(p[0] for p in color_positions) // len(color_positions),
                sum(p[1] for p in color_positions) // len(color_positions)
            )

            # Is movement toward this color?
            dir_to_color = (color_center[0] - start[0], color_center[1] - start[1])

            # Dot product > 0 means moving toward
            if net_dy * dir_to_color[0] + net_dx * dir_to_color[1] > 0:
                # Compute distance change
                start_dist = abs(color_center[0] - start[0]) + abs(color_center[1] - start[1])
                end_dist = abs(color_center[0] - end[0]) + abs(color_center[1] - end[1])

                if end_dist < start_dist:
                    insights.append(DemonstrationInsight(
                        insight_type="goal",
                        description=f"Player moved toward Color {color} (dist: {start_dist} -> {end_dist})",
                        confidence=0.6,
                        evidence_count=1,
                        target_color=color,
                        target_position=color_center,
                    ))

        return insights

    def find_level_transitions(
        self,
        trajectory: Trajectory,
        player_color: int
    ) -> list[LevelTransition]:
        """
        Find level completion boundaries in a trajectory.

        Level completions cause significant frame changes (new layout).
        Detect these by comparing consecutive frames for large pixel diffs.
        """
        transitions = []
        level_num = 0

        for i in range(len(trajectory.frames) - 1):
            prev = trajectory.frames[i].frame
            curr = trajectory.frames[i + 1].frame
            action = trajectory.frames[i].action_taken

            # Large pixel diff = level transition
            diff_count = np.sum(prev != curr)
            total_pixels = prev.shape[0] * prev.shape[1]
            diff_ratio = diff_count / total_pixels

            if diff_ratio > 0.3:  # >30% of pixels changed = new level
                level_num += 1
                transition = LevelTransition(
                    level_number=level_num,
                    frame_before=prev,
                    frame_after=curr,
                    action_at_transition=action,
                )

                # Analyze what changed
                for color in range(1, 16):
                    prev_count = int(np.sum(prev == color))
                    curr_count = int(np.sum(curr == color))

                    if prev_count > 0 and curr_count == 0:
                        transition.colors_disappeared.append(color)
                    if abs(prev_count - curr_count) > max(prev_count, 1) * 0.2:
                        transition.colors_changed.append((color, prev_count, curr_count))

                # Find player position right before transition
                positions = list(zip(*np.where(prev == player_color)))
                if positions:
                    center_y = sum(p[0] for p in positions) // len(positions)
                    center_x = sum(p[1] for p in positions) // len(positions)
                    transition.player_position_at_completion = (center_y, center_x)

                transitions.append(transition)

        return transitions

    def find_goal_from_transitions(
        self,
        transitions: list[LevelTransition],
        player_color: int
    ) -> list[DemonstrationInsight]:
        """
        Analyze level transitions to infer goal patterns.

        Key insight: What's consistent across ALL level completions
        is likely part of the goal. What varies is level-specific.
        """
        if not transitions:
            return []

        insights = []

        # Track what happens consistently at transitions
        player_near_colors = defaultdict(int)  # color -> count of transitions

        for transition in transitions:
            frame = transition.frame_before
            pos = transition.player_position_at_completion
            if not pos:
                continue

            py, px = pos
            total_pixels = frame.shape[0] * frame.shape[1]

            # What colors were near the player at each completion?
            for color in range(1, 16):
                if color == player_color:
                    continue
                color_positions = list(zip(*np.where(frame == color)))
                if not color_positions:
                    continue

                # Skip background/wall colors (>15% of pixels)
                if len(color_positions) / total_pixels > 0.15:
                    continue

                # Check proximity
                min_dist = float('inf')
                for cy, cx in color_positions:
                    dist = abs(cy - py) + abs(cx - px)
                    if dist < min_dist:
                        min_dist = dist

                if min_dist <= 10:  # Within 10 pixels
                    player_near_colors[color] += 1

        num_transitions = len(transitions)

        # Colors consistently near player at completion are strong goal candidates
        for color, count in player_near_colors.items():
            consistency = count / num_transitions
            if consistency >= 0.5:  # Present at >=50% of completions
                insights.append(DemonstrationInsight(
                    insight_type="goal",
                    description=f"Player near Color {color} at {count}/{num_transitions} level completions ({consistency:.0%})",
                    confidence=min(0.95, 0.5 + consistency * 0.4),
                    evidence_count=count,
                    target_color=color,
                ))

        # Colors that consistently disappear at transitions
        disappear_counts = defaultdict(int)
        for transition in transitions:
            for color in transition.colors_disappeared:
                if color != player_color:
                    disappear_counts[color] += 1

        for color, count in disappear_counts.items():
            consistency = count / num_transitions
            if consistency >= 0.3:
                insights.append(DemonstrationInsight(
                    insight_type="goal",
                    description=f"Color {color} disappeared at {count}/{num_transitions} level completions",
                    confidence=min(0.9, 0.4 + consistency * 0.5),
                    evidence_count=count,
                    target_color=color,
                ))

        return insights

    def find_avoided_colors(
        self,
        trajectory: Trajectory,
        player_color: int
    ) -> list[DemonstrationInsight]:
        """
        Find colors that the player avoided (likely obstacles).
        """
        insights = []

        # Track colors player never overlapped with
        all_colors = set()
        overlapped_colors = set()

        for frame_data in trajectory.frames:
            frame = frame_data.frame

            # Colors present
            for color in range(1, 16):
                if np.sum(frame == color) > 0:
                    all_colors.add(color)

            # Colors player overlapped
            player_positions = set(zip(*np.where(frame == player_color)))
            for color in all_colors:
                if color == player_color:
                    continue
                color_positions = set(zip(*np.where(frame == color)))

                # Check overlap
                for py, px in player_positions:
                    for cy, cx in color_positions:
                        if abs(py - cy) <= 3 and abs(px - cx) <= 3:
                            overlapped_colors.add(color)

        # Colors that were never overlapped might be obstacles
        avoided = all_colors - overlapped_colors - {player_color, 0}

        for color in avoided:
            insights.append(DemonstrationInsight(
                insight_type="obstacle",
                description=f"Player avoided Color {color} throughout trajectory",
                confidence=0.5,
                evidence_count=1,
                target_color=color,
            ))

        return insights


class DemonstrationLearner:
    """
    Main demonstration learning system.

    Aggregates insights from multiple trajectories to build
    robust understanding of game goals.
    """

    def __init__(self):
        self.analyzer = TrajectoryAnalyzer()
        self.trajectories: list[Trajectory] = []
        self.insights: dict[str, list[DemonstrationInsight]] = defaultdict(list)

        # Aggregated statistics
        self.goal_votes: dict[int, float] = defaultdict(float)  # color -> score
        self.obstacle_votes: dict[int, float] = defaultdict(float)
        self.player_color: Optional[int] = None

    def add_trajectory(self, trajectory: Trajectory):
        """Add a trajectory for analysis."""
        self.trajectories.append(trajectory)
        self._analyze_trajectory(trajectory)

    def load_from_jsonl(self, jsonl_path: str) -> Optional[Trajectory]:
        """
        Load a trajectory from a JSONL human recording file.

        JSONL format: Each line has {"data": {"frame": [[...]], "action_input": {"id": N}, "state": "...", "score": N}}
        """
        path = Path(jsonl_path)
        if not path.exists():
            print(f"  Warning: {jsonl_path} not found")
            return None

        # Extract game_id from filename
        name = path.stem
        game_id = name.split("-")[0] if "-" in name else "unknown"

        frames = []
        prev_score = 0
        final_state = None
        frame_idx = 0

        with open(path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                data = entry.get("data", {})
                if not data:
                    continue

                frame_data = data.get("frame")
                if not frame_data or len(frame_data) == 0:
                    continue

                obs = np.array(frame_data[0], dtype=np.int32)

                action_input = data.get("action_input")
                action = action_input["id"] if action_input and "id" in action_input else None

                state = data.get("state")
                if state:
                    final_state = state

                frames.append(TrajectoryFrame(
                    frame_id=frame_idx,
                    frame=obs,
                    action_taken=action,
                ))
                frame_idx += 1

        if len(frames) < 2:
            return None

        success = final_state == "WIN"

        trajectory = Trajectory(
            game_id=game_id,
            frames=frames,
            success=success,
        )

        self.add_trajectory(trajectory)
        return trajectory

    def load_from_jsonl_folder(self, folder_path: str) -> int:
        """
        Load all JSONL demos from a folder (recursively).

        Returns number of trajectories loaded.
        """
        folder = Path(folder_path)
        count = 0
        for jsonl_path in sorted(folder.rglob("*.jsonl")):
            traj = self.load_from_jsonl(str(jsonl_path))
            if traj:
                count += 1
        return count

    def _find_background_colors(self, trajectory: Trajectory) -> set[int]:
        """Identify colors that are likely background/walls (large, static)."""
        if not trajectory.frames:
            return set()

        # Sample a few frames to check color sizes
        sample_indices = [0, len(trajectory.frames) // 2, -1]
        large_colors = defaultdict(int)

        for idx in sample_indices:
            frame = trajectory.frames[idx].frame
            total_pixels = frame.shape[0] * frame.shape[1]

            for color in range(1, 16):
                count = int(np.sum(frame == color))
                if count / total_pixels > 0.12:  # >12% of pixels
                    large_colors[color] += 1

        # Colors that are large in all sampled frames are background
        return {c for c, count in large_colors.items() if count >= 2}

    def _analyze_trajectory(self, trajectory: Trajectory):
        """Analyze a single trajectory and update insights."""
        # Identify player
        player = self.analyzer.identify_player(trajectory)
        if player:
            self.player_color = player

            # Annotate frames
            for frame_data in trajectory.frames:
                frame_data.player_color = player
                positions = list(zip(*np.where(frame_data.frame == player)))
                if positions:
                    center = (
                        sum(p[0] for p in positions) // len(positions),
                        sum(p[1] for p in positions) // len(positions)
                    )
                    frame_data.player_position = center

        else:
            return  # Can't analyze without player

        # Extract insights
        all_insights = []

        if trajectory.success:
            goal_insights = self.analyzer.find_goal_from_success(trajectory, player)
            all_insights.extend(goal_insights)

        # Level transition analysis (most powerful signal)
        transitions = self.analyzer.find_level_transitions(trajectory, player)
        if transitions:
            transition_insights = self.analyzer.find_goal_from_transitions(transitions, player)
            all_insights.extend(transition_insights)

        approach_insights = self.analyzer.find_approach_patterns(trajectory, player)
        all_insights.extend(approach_insights)

        avoided_insights = self.analyzer.find_avoided_colors(trajectory, player)
        all_insights.extend(avoided_insights)

        # Filter out background colors from goal insights
        bg_colors = self._find_background_colors(trajectory)

        # Aggregate insights
        for insight in all_insights:
            # Skip goal insights for background colors
            if insight.insight_type == "goal" and insight.target_color in bg_colors:
                continue

            self.insights[insight.insight_type].append(insight)

            if insight.target_color is not None:
                if insight.insight_type == "goal":
                    self.goal_votes[insight.target_color] += insight.confidence
                elif insight.insight_type == "obstacle":
                    self.obstacle_votes[insight.target_color] += insight.confidence

    def get_likely_goals(self) -> list[tuple[int, float]]:
        """Get most likely goal colors with confidence scores."""
        if not self.goal_votes:
            return []

        # Normalize by total votes
        total = sum(self.goal_votes.values())
        if total == 0:
            return []

        return sorted(
            [(color, score / total) for color, score in self.goal_votes.items()],
            key=lambda x: -x[1]
        )

    def get_likely_obstacles(self) -> list[tuple[int, float]]:
        """Get most likely obstacle colors."""
        if not self.obstacle_votes:
            return []

        total = sum(self.obstacle_votes.values())
        if total == 0:
            return []

        return sorted(
            [(color, score / total) for color, score in self.obstacle_votes.items()],
            key=lambda x: -x[1]
        )

    def get_goal_insights(self) -> list[DemonstrationInsight]:
        """Get all goal-related insights, sorted by confidence."""
        return sorted(
            self.insights.get("goal", []),
            key=lambda x: -x.confidence
        )

    def export_for_goal_inducer(self) -> dict:
        """
        Export learned information in format usable by GoalInducer.

        Returns:
            {
                "player_color": int,
                "goal_colors": [(color, confidence), ...],
                "obstacle_colors": [(color, confidence), ...],
                "target_positions": [(y, x), ...],
            }
        """
        result = {
            "player_color": self.player_color,
            "goal_colors": self.get_likely_goals(),
            "obstacle_colors": self.get_likely_obstacles(),
            "target_positions": [],
        }

        # Collect target positions from high-confidence insights
        for insight in self.insights.get("goal", []):
            if insight.target_position and insight.confidence >= 0.7:
                result["target_positions"].append(insight.target_position)

        return result

    def describe(self) -> str:
        """Human-readable summary of learned information."""
        lines = ["=== Demonstration Learning Summary ==="]
        lines.append(f"Trajectories analyzed: {len(self.trajectories)}")
        lines.append(f"Successful trajectories: {sum(1 for t in self.trajectories if t.success)}")
        lines.append(f"Total frames: {sum(t.length for t in self.trajectories)}")
        lines.append(f"Player color: {self.player_color}")
        lines.append("")

        lines.append("Likely goals (ranked):")
        for color, confidence in self.get_likely_goals()[:5]:
            lines.append(f"  Color {color}: {confidence:.0%}")

        lines.append("")
        lines.append("Likely obstacles:")
        for color, confidence in self.get_likely_obstacles()[:5]:
            lines.append(f"  Color {color}: {confidence:.0%}")

        lines.append("")
        lines.append(f"Key insights ({len(self.get_goal_insights())} total):")
        for insight in self.get_goal_insights()[:8]:
            lines.append(f"  [{insight.confidence:.0%}] {insight.description}")

        return "\n".join(lines)


class TrajectoryRecorder:
    """
    Records trajectories during gameplay for later analysis.
    """

    def __init__(self, game_id: str, save_dir: Optional[Path] = None):
        self.game_id = game_id
        self.save_dir = save_dir or Path("trajectories")
        self.frames: list[TrajectoryFrame] = []
        self.frame_count = 0
        self.recording = True

    def record_frame(self, frame: np.ndarray, action: Optional[int] = None):
        """Record a frame and the action taken after it."""
        if not self.recording:
            return

        frame_data = TrajectoryFrame(
            frame_id=self.frame_count,
            frame=frame.copy(),
            action_taken=action,
        )
        self.frames.append(frame_data)
        self.frame_count += 1

    def mark_success(self):
        """Mark the trajectory as successful."""
        self.success = True

    def finish(self, success: bool = False, reward: float = 0.0) -> Trajectory:
        """Finish recording and return the trajectory."""
        self.recording = False

        trajectory = Trajectory(
            game_id=self.game_id,
            frames=self.frames,
            success=success,
            total_reward=reward,
        )

        return trajectory

    def save(self, trajectory: Trajectory, filename: Optional[str] = None):
        """Save trajectory to disk."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            filename = f"{self.game_id}_{len(trajectory.frames)}frames.npz"

        filepath = self.save_dir / filename

        # Save frames as numpy array
        frames_array = np.stack([f.frame for f in trajectory.frames])
        actions_array = np.array([f.action_taken or -1 for f in trajectory.frames])

        np.savez(
            filepath,
            frames=frames_array,
            actions=actions_array,
            success=trajectory.success,
            reward=trajectory.total_reward,
            game_id=trajectory.game_id,
        )

        return filepath

    @staticmethod
    def load(filepath: Path) -> Trajectory:
        """Load trajectory from disk."""
        data = np.load(filepath, allow_pickle=True)

        frames = []
        for i, (frame, action) in enumerate(zip(data["frames"], data["actions"])):
            frames.append(TrajectoryFrame(
                frame_id=i,
                frame=frame,
                action_taken=action if action >= 0 else None,
            ))

        return Trajectory(
            game_id=str(data["game_id"]),
            frames=frames,
            success=bool(data["success"]),
            total_reward=float(data["reward"]),
        )


def integrate_with_goal_inducer(
    learner: DemonstrationLearner,
    inducer: "GoalInducer"
):
    """
    Transfer knowledge from DemonstrationLearner to GoalInducer.
    """
    from .goal_induction import GoalHypothesis, GoalType

    export = learner.export_for_goal_inducer()

    # Set player color
    if export["player_color"]:
        inducer.set_controllable(export["player_color"])

    # Mark obstacles as excluded
    for color, confidence in export["obstacle_colors"]:
        if confidence >= 0.5:
            inducer.exclude_color(color, reason="avoided in demonstrations")

    # Add goal hypotheses from demonstrations
    # confidence here is a relative proportion (0-1 summing to 1)
    # Convert to absolute confidence for hypothesis testing
    for color, proportion in export["goal_colors"]:
        if proportion >= 0.05:  # At least 5% of votes
            # Scale proportion to confidence: top candidate gets high confidence
            confidence = min(0.95, proportion * 3)

            # Find target position for this color
            target_pos = None
            for insight in learner.get_goal_insights():
                if insight.target_color == color and insight.target_position:
                    target_pos = insight.target_position
                    break

            hypothesis = GoalHypothesis(
                goal_type=GoalType.REACH_COLOR,
                confidence=confidence,
                evidence=int(proportion * 20),
                target_color=color,
                target_position=target_pos,
            )
            inducer.tester.add_hypotheses([hypothesis])


def test_demonstration_learner_synthetic():
    """Test demonstration learning with synthetic data."""
    print("=== Testing Demonstration Learner (Synthetic) ===\n")

    # Create synthetic successful trajectory
    # Scenario: Player (color 12) moves from (30,30) to reach goal (color 9) at (10,50)

    frames = []
    player_positions = [(30, 30), (25, 30), (20, 30), (15, 35), (10, 40), (10, 45), (10, 50)]
    actions = [1, 1, 1, 4, 4, 4, 0]  # UP, UP, UP, RIGHT, RIGHT, RIGHT, NOOP

    for i, (py, px) in enumerate(player_positions):
        frame = np.zeros((64, 64), dtype=np.int32)

        # Walls (color 4)
        frame[:5, :] = 4
        frame[59:, :] = 4
        frame[:, :5] = 4
        frame[:, 59:] = 4

        # Player (color 12)
        frame[py:py+5, px:px+5] = 12

        # Goal (color 9) - disappears in final frame
        if i < len(player_positions) - 1:
            frame[8:13, 48:53] = 9

        # Obstacle (color 5)
        frame[20:30, 45:50] = 5

        frames.append(TrajectoryFrame(
            frame_id=i,
            frame=frame,
            action_taken=actions[i] if i < len(actions) else None,
        ))

    trajectory = Trajectory(
        game_id="test_game",
        frames=frames,
        success=True,
        total_reward=1.0,
    )

    # Analyze trajectory
    learner = DemonstrationLearner()
    learner.add_trajectory(trajectory)

    print(learner.describe())

    print("\n--- Export for GoalInducer ---")
    export = learner.export_for_goal_inducer()
    for key, value in export.items():
        print(f"  {key}: {value}")


def test_demonstration_learner_real():
    """Test demonstration learning on real JSONL human demos."""
    print("=== Testing Demonstration Learner (Real Data) ===\n")

    demo_folder = Path("videos/ARC-AGI-3 Human Performance/ls20")
    if not demo_folder.exists():
        print(f"Demo folder not found: {demo_folder}")
        print("Skipping real data test")
        return

    learner = DemonstrationLearner()
    count = learner.load_from_jsonl_folder(str(demo_folder))
    print(f"\nLoaded {count} trajectories\n")

    print(learner.describe())

    print("\n--- Export for GoalInducer ---")
    export = learner.export_for_goal_inducer()
    for key, value in export.items():
        print(f"  {key}: {value}")

    # Integration test
    print("\n--- GoalInducer Integration ---")
    from .goal_induction import GoalInducer
    inducer = GoalInducer()
    integrate_with_goal_inducer(learner, inducer)

    print(f"Controllable: Color {inducer.controllable_color}")
    print(f"Excluded: {inducer.excluded_colors}")
    print(f"Hypotheses: {len(inducer.tester.hypotheses)}")
    for h in sorted(inducer.tester.hypotheses, key=lambda x: -x.confidence)[:5]:
        print(f"  [{h.confidence:.0%}] {h.describe()}")


if __name__ == "__main__":
    import sys
    if "--real" in sys.argv:
        test_demonstration_learner_real()
    else:
        test_demonstration_learner_synthetic()
