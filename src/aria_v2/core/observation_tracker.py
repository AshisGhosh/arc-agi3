"""
Observation Tracker - Ground truth from pixel differences.

No ML here - just tracking what changed between frames.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PixelChange:
    """A single pixel that changed."""
    x: int
    y: int
    old_color: int
    new_color: int


@dataclass
class RegionChange:
    """A contiguous region that changed."""
    positions: list[tuple[int, int]]
    old_color: int
    new_color: int

    @property
    def center(self) -> tuple[int, int]:
        xs = [p[0] for p in self.positions]
        ys = [p[1] for p in self.positions]
        return (sum(xs) // len(xs), sum(ys) // len(ys))

    @property
    def size(self) -> int:
        return len(self.positions)


@dataclass
class Observation:
    """What we observed from one frame transition."""
    frame_id: int
    action_taken: int  # 0=noop, 1=up, 2=down, 3=left, 4=right, etc.

    # Raw changes
    pixel_changes: list[PixelChange]
    region_changes: list[RegionChange]

    # Derived observations
    player_moved: bool = False
    player_old_pos: Optional[tuple[int, int]] = None
    player_new_pos: Optional[tuple[int, int]] = None

    movement_blocked: bool = False
    blocked_by_color: Optional[int] = None

    region_disappeared: bool = False
    disappeared_region: Optional[RegionChange] = None

    region_appeared: bool = False
    appeared_region: Optional[RegionChange] = None

    level_completed: bool = False
    game_over: bool = False

    @property
    def num_changes(self) -> int:
        return len(self.pixel_changes)

    @property
    def is_significant(self) -> bool:
        """Did something meaningful happen?"""
        return (self.player_moved or
                self.movement_blocked or
                self.region_disappeared or
                self.level_completed)


class ObservationTracker:
    """
    Tracks changes between frames and produces Observations.

    Pure algorithmic - no ML.
    """

    def __init__(self):
        self.frame_count = 0
        self.prev_frame: Optional[np.ndarray] = None
        self.player_pos: Optional[tuple[int, int]] = None
        self.player_color: Optional[int] = None

    def reset(self):
        """Reset tracker for new episode."""
        self.frame_count = 0
        self.prev_frame = None
        self.player_pos = None
        self.player_color = None

    def observe(
        self,
        frame: np.ndarray,
        action: int,
        level_completed: bool = False,
        game_over: bool = False,
    ) -> Observation:
        """
        Process a new frame and return observation.

        Args:
            frame: Current frame [H, W] of color indices
            action: Action that was taken (0-7)
            level_completed: Signal from game
            game_over: Signal from game

        Returns:
            Observation with all detected changes
        """
        self.frame_count += 1

        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return Observation(
                frame_id=self.frame_count,
                action_taken=action,
                pixel_changes=[],
                region_changes=[],
                level_completed=level_completed,
                game_over=game_over,
            )

        # Find pixel changes
        pixel_changes = self._find_pixel_changes(self.prev_frame, frame)

        # Group into regions
        region_changes = self._group_into_regions(pixel_changes, self.prev_frame, frame)

        # Create observation
        obs = Observation(
            frame_id=self.frame_count,
            action_taken=action,
            pixel_changes=pixel_changes,
            region_changes=region_changes,
            level_completed=level_completed,
            game_over=game_over,
        )

        # Detect player movement
        self._detect_player_movement(obs, action, self.prev_frame, frame)

        # Detect region disappearance/appearance
        self._detect_region_changes(obs, region_changes)

        # Update state
        self.prev_frame = frame.copy()

        return obs

    def _find_pixel_changes(
        self,
        prev: np.ndarray,
        curr: np.ndarray
    ) -> list[PixelChange]:
        """Find all pixels that changed."""
        diff_mask = prev != curr
        changed_positions = np.argwhere(diff_mask)

        changes = []
        for pos in changed_positions:
            y, x = pos
            changes.append(PixelChange(
                x=int(x),
                y=int(y),
                old_color=int(prev[y, x]),
                new_color=int(curr[y, x]),
            ))

        return changes

    def _group_into_regions(
        self,
        pixel_changes: list[PixelChange],
        prev: np.ndarray,
        curr: np.ndarray,
    ) -> list[RegionChange]:
        """Group pixel changes into contiguous regions."""
        if not pixel_changes:
            return []

        # Build adjacency and group by color transition
        by_transition: dict[tuple[int, int], list[PixelChange]] = {}
        for pc in pixel_changes:
            key = (pc.old_color, pc.new_color)
            if key not in by_transition:
                by_transition[key] = []
            by_transition[key].append(pc)

        regions = []
        for (old_color, new_color), changes in by_transition.items():
            # Simple grouping: all changes with same transition = one region
            # (Could do proper connected components, but this is simpler)
            positions = [(c.x, c.y) for c in changes]
            regions.append(RegionChange(
                positions=positions,
                old_color=old_color,
                new_color=new_color,
            ))

        return regions

    def _detect_player_movement(
        self,
        obs: Observation,
        action: int,
        prev: np.ndarray,
        curr: np.ndarray,
    ):
        """Detect if player moved based on action and changes."""
        if action == 0:  # NOOP
            return

        # Expected movement direction
        dx, dy = 0, 0
        if action == 1:    # up
            dy = -1
        elif action == 2:  # down
            dy = 1
        elif action == 3:  # left
            dx = -1
        elif action == 4:  # right
            dx = 1
        else:
            return  # Not a movement action

        # Look for color swap pattern: A->B at old_pos, B->A at new_pos
        # This indicates something (player) moved from old to new position
        for rc1 in obs.region_changes:
            old_color = rc1.old_color
            new_color = rc1.new_color

            # Find a matching swap: same colors but reversed
            for rc2 in obs.region_changes:
                if rc2.old_color == new_color and rc2.new_color == old_color:
                    # Found a swap! Check if direction matches action
                    center1 = rc1.center
                    center2 = rc2.center

                    # Calculate movement direction
                    actual_dx = center1[0] - center2[0]
                    actual_dy = center1[1] - center2[1]

                    # Check if movement matches expected direction
                    if (np.sign(actual_dx) == np.sign(dx) or dx == 0) and \
                       (np.sign(actual_dy) == np.sign(dy) or dy == 0) and \
                       (abs(actual_dx) > 0 or abs(actual_dy) > 0):
                        # Player moved from rc2.center to rc1.center
                        obs.player_moved = True
                        obs.player_old_pos = center2
                        obs.player_new_pos = center1
                        self.player_pos = center1
                        # Player color is what appeared at new position
                        self.player_color = old_color
                        return

        # Fallback: look for any consistent region movement in expected direction
        if not obs.player_moved:
            for rc in obs.region_changes:
                if rc.size >= 5:  # Minimum player size
                    # Check if there's a matching disappearance
                    for rc2 in obs.region_changes:
                        if rc2.old_color == rc.new_color and rc2.size >= 5:
                            center1 = rc.center
                            center2 = rc2.center
                            actual_dx = center1[0] - center2[0]
                            actual_dy = center1[1] - center2[1]

                            if (np.sign(actual_dx) == np.sign(dx) or dx == 0) and \
                               (np.sign(actual_dy) == np.sign(dy) or dy == 0) and \
                               (abs(actual_dx) > 0 or abs(actual_dy) > 0):
                                obs.player_moved = True
                                obs.player_old_pos = center2
                                obs.player_new_pos = center1
                                self.player_pos = center1
                                self.player_color = rc.new_color
                                return

        # If no movement detected but movement action taken, might be blocked
        if not obs.player_moved and action in [1, 2, 3, 4]:
            if self.player_pos:
                # Check what's in the direction we tried to move
                step = 5  # Assume ~5 pixel steps
                check_x = self.player_pos[0] + dx * step
                check_y = self.player_pos[1] + dy * step
                if 0 <= check_x < curr.shape[1] and 0 <= check_y < curr.shape[0]:
                    blocking_color = curr[int(check_y), int(check_x)]
                    if blocking_color != self.player_color:
                        obs.movement_blocked = True
                        obs.blocked_by_color = int(blocking_color)

    def _detect_region_changes(self, obs: Observation, region_changes: list[RegionChange]):
        """Detect if regions appeared or disappeared."""
        for rc in region_changes:
            if rc.old_color != 0 and rc.new_color == 0:
                # Something non-background became background = disappeared
                obs.region_disappeared = True
                obs.disappeared_region = rc
            elif rc.old_color == 0 and rc.new_color != 0:
                # Background became something = appeared
                obs.region_appeared = True
                obs.appeared_region = rc


def test_observation_tracker():
    """Test the observation tracker."""
    tracker = ObservationTracker()

    # Create simple test frames
    frame1 = np.zeros((10, 10), dtype=np.int32)
    frame1[5, 5] = 1  # Blue pixel (player)

    frame2 = np.zeros((10, 10), dtype=np.int32)
    frame2[5, 6] = 1  # Player moved right

    # First observation (no previous frame)
    obs1 = tracker.observe(frame1, action=0)
    print(f"Obs 1: {obs1.num_changes} changes")

    # Second observation (player moved right)
    obs2 = tracker.observe(frame2, action=4)  # action 4 = right
    print(f"Obs 2: {obs2.num_changes} changes, player_moved={obs2.player_moved}")

    if obs2.player_moved:
        print(f"  Player: {obs2.player_old_pos} -> {obs2.player_new_pos}")


if __name__ == "__main__":
    test_observation_tracker()
