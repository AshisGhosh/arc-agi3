"""
Base class for synthetic games.

All synthetic games produce 64x64 frames with 16 colors (0-15),
support actions 0-7 matching ARC-AGI-3 format, and provide
ground truth labels for training the understanding model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class GameType(str, Enum):
    NAVIGATION = "navigation"
    CLICK_PUZZLE = "click_puzzle"
    COLLECTION = "collection"
    MIXED = "mixed"
    PUSH = "push"
    CONDITIONAL = "conditional"
    UNKNOWN = "unknown"


class EntityRole(str, Enum):
    PLAYER = "player"
    WALL = "wall"
    COLLECTIBLE = "collectible"
    BACKGROUND = "background"
    COUNTER = "counter"
    TARGET = "target"
    BUTTON = "button"
    OBSTACLE = "obstacle"


@dataclass
class EntityInfo:
    color: int
    role: EntityRole
    positions: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class ActionEffect:
    """Ground truth effect of an action."""
    action_id: int
    shift_dx: float = 0.0
    shift_dy: float = 0.0
    change_prob: float = 0.0  # probability this action changes the frame
    affected_color: int = -1  # color most affected by this action
    blocked_prob: float = 0.0  # probability of being blocked


@dataclass
class GroundTruth:
    """Ground truth labels for a game state."""
    game_type: GameType = GameType.UNKNOWN
    entities: dict[int, EntityInfo] = field(default_factory=dict)  # color → EntityInfo
    action_effects: dict[int, ActionEffect] = field(default_factory=dict)  # action_id → effect
    confidence: float = 0.0  # how much evidence is available (0=none, 1=fully determined)
    level_complete: bool = False
    game_over: bool = False


@dataclass
class GameConfig:
    """Configuration for generating a game instance."""
    seed: int = 0
    grid_size: int = 64  # always 64x64
    num_colors: int = 16  # always 16
    # Color assignments (set by augmentation or randomly)
    background_color: int = 0
    player_color: int = 12
    wall_color: int = 5
    collectible_color: int = 9
    counter_color: int = 3
    target_color: int = 14
    # Game parameters (vary per archetype)
    step_size: int = 8  # pixels per movement
    wall_density: float = 0.2  # fraction of grid cells that are walls
    num_collectibles: int = 5
    # Action mapping: which game action IDs correspond to which directions
    # Default: 1=up, 2=down, 3=left, 4=right, 5=action5
    action_up: int = 1
    action_down: int = 2
    action_left: int = 3
    action_right: int = 4


class SyntheticGame:
    """Base class for synthetic games.

    All games:
    - Produce 64x64 frames with values 0-15
    - Support actions 1-7 (subset varies per game)
    - Provide ground truth labels at any point
    - Are deterministic given the same seed
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.frame = np.zeros((64, 64), dtype=np.uint8)
        self.step_count = 0

    def reset(self) -> np.ndarray:
        """Reset to initial state. Returns 64x64 frame."""
        raise NotImplementedError

    def step(self, action: int, x: int = 0, y: int = 0) -> tuple[np.ndarray, bool, bool]:
        """Take an action.

        Args:
            action: Action ID (1-7)
            x, y: Click coordinates (for action 6)

        Returns:
            (next_frame, level_complete, game_over)
        """
        raise NotImplementedError

    def get_ground_truth(self) -> GroundTruth:
        """Return ground truth labels for current state."""
        raise NotImplementedError

    @property
    def available_actions(self) -> list[int]:
        """Which actions are available in this game."""
        raise NotImplementedError

    def render_rect(
        self,
        frame: np.ndarray,
        x: int, y: int,
        w: int, h: int,
        color: int,
    ) -> None:
        """Draw a filled rectangle on the frame."""
        y1 = max(0, y)
        y2 = min(64, y + h)
        x1 = max(0, x)
        x2 = min(64, x + w)
        if y1 < y2 and x1 < x2:
            frame[y1:y2, x1:x2] = color

    def render_border(
        self,
        frame: np.ndarray,
        color: int,
        thickness: int = 1,
    ) -> None:
        """Draw a border around the frame."""
        frame[:thickness, :] = color
        frame[-thickness:, :] = color
        frame[:, :thickness] = color
        frame[:, -thickness:] = color
