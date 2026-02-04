"""
Base interfaces for primitive environments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch


class Action(IntEnum):
    """
    Universal action space for all primitives.

    Actions 0-7: Discrete actions (navigation, interaction)
    Action 8: Click/Select at (x, y) - requires coordinates
    """
    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5
    CONFIRM = 6  # Submit/confirm selection
    CANCEL = 7   # Cancel/reset
    CLICK = 8    # Click at coordinate (requires x, y)


class PrimitiveFamily(IntEnum):
    """Categories of primitive skills."""
    NAVIGATION = 0
    CLICK = 1
    PATTERN = 2
    STATE_TRACKING = 3
    COMPOSITION = 4


@dataclass
class PrimitiveResult:
    """Result of taking a step in a primitive environment."""
    observation: torch.Tensor  # [H, W] or [C, H, W]
    reward: float
    done: bool
    success: bool  # Whether goal was achieved (for metrics)
    info: dict = field(default_factory=dict)


class PrimitiveEnv(ABC):
    """
    Base class for all primitive environments.

    Each primitive teaches a specific skill:
    - Navigation: Move agent to targets
    - Click: Select specific coordinates
    - Pattern: Match/find patterns
    - State Tracking: Remember and use state
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 50,
        seed: Optional[int] = None,
        deterministic: bool = False,
    ):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.base_seed = seed
        self.reset_count = 0
        self.step_count = 0

        # Set seed if deterministic
        if deterministic and seed is not None:
            torch.manual_seed(seed)

    @property
    @abstractmethod
    def family(self) -> PrimitiveFamily:
        """Return the primitive family this env belongs to."""
        pass

    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """Number of discrete actions (excluding click coordinates)."""
        pass

    @property
    def requires_coordinates(self) -> bool:
        """Whether this primitive uses click coordinates."""
        return False

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        pass

    @abstractmethod
    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> PrimitiveResult:
        """
        Take action and return result.

        Args:
            action: Discrete action (0-8)
            x: X coordinate for CLICK action
            y: Y coordinate for CLICK action
        """
        pass

    def _reseed(self):
        """Reseed RNG for non-deterministic mode."""
        if self.deterministic and self.base_seed is not None:
            # Deterministic: use predictable sequence
            seed = self.base_seed + self.reset_count * 7919
            torch.manual_seed(seed)
        # Non-deterministic: don't seed, use current RNG state

    def get_task_description(self) -> str:
        """Return human-readable description of current task."""
        return f"{self.__class__.__name__} on {self.grid_size}x{self.grid_size} grid"


@dataclass
class PrimitiveSpec:
    """Specification for generating a primitive task."""
    family: PrimitiveFamily
    difficulty: int = 1  # 1-5, higher = harder
    grid_size: int = 10
    max_steps: int = 50
    variant: str = "default"  # Specific variant within family
    seed: Optional[int] = None
