"""
Click/Selection primitives - predicting and selecting coordinates.

Variants:
- target: Click on highlighted target
- sequence: Click targets in order
- drag: Select start and end points
- region: Select rectangular region
"""

import random
from typing import Optional

import torch

from .base import Action, PrimitiveEnv, PrimitiveFamily, PrimitiveResult


class ClickEnv(PrimitiveEnv):
    """
    Click/Selection primitive environment.

    Agent must click on target cell(s) to complete task.
    """

    EMPTY = 0
    TARGET = 1      # Click target
    CLICKED = 2     # Already clicked
    HIGHLIGHT = 3   # Visual hint
    OBSTACLE = 4    # Cannot click
    SEQUENCE_1 = 5  # First in sequence
    SEQUENCE_2 = 6  # Second in sequence
    SEQUENCE_3 = 7  # Third in sequence

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 20,
        seed: Optional[int] = None,
        deterministic: bool = False,
        variant: str = "target",
        num_targets: int = 1,
        show_order: bool = True,  # For sequence variant
    ):
        super().__init__(grid_size, max_steps, seed, deterministic)
        self.variant = variant
        self.num_targets = num_targets
        self.show_order = show_order

        # State
        self.grid: Optional[torch.Tensor] = None
        self.targets: list[tuple[int, int]] = []
        self.clicked: list[tuple[int, int]] = []
        self.current_target_idx: int = 0

    @property
    def family(self) -> PrimitiveFamily:
        return PrimitiveFamily.CLICK

    @property
    def action_space_size(self) -> int:
        return 9  # 0-8, but CLICK (8) requires coordinates

    @property
    def requires_coordinates(self) -> bool:
        return True

    def reset(self) -> torch.Tensor:
        self._reseed()
        self.reset_count += 1
        self.step_count = 0

        # Initialize grid
        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Place targets
        self.targets = []
        self.clicked = []
        self.current_target_idx = 0

        for i in range(self.num_targets):
            pos = self._random_empty_pos()
            if pos:
                self.targets.append(pos)
                if self.variant == "sequence" and self.show_order:
                    # Show numbered targets
                    self.grid[pos] = self.SEQUENCE_1 + min(i, 2)
                else:
                    self.grid[pos] = self.TARGET

        return self.grid.clone()

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> PrimitiveResult:
        if self.grid is None:
            raise RuntimeError("Environment not reset")

        self.step_count += 1
        reward = -0.05  # Step penalty

        success = False
        click_correct = False

        if action == Action.CLICK and x is not None and y is not None:
            # Validate coordinates
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                click_pos = (y, x)  # Note: y is row, x is col

                if self.variant == "sequence":
                    # Must click in order
                    if self.current_target_idx < len(self.targets):
                        expected = self.targets[self.current_target_idx]
                        if click_pos == expected:
                            click_correct = True
                            self.clicked.append(click_pos)
                            self.grid[click_pos] = self.CLICKED
                            self.current_target_idx += 1
                            reward += 1.0
                        else:
                            reward -= 0.5  # Wrong target
                else:
                    # Any order
                    if click_pos in self.targets and click_pos not in self.clicked:
                        click_correct = True
                        self.clicked.append(click_pos)
                        self.grid[click_pos] = self.CLICKED
                        reward += 1.0
                    elif click_pos in self.clicked:
                        reward -= 0.2  # Already clicked
                    else:
                        reward -= 0.3  # Wrong cell
            else:
                reward -= 0.5  # Out of bounds

        # Check completion
        success = len(self.clicked) == len(self.targets)
        done = success or self.step_count >= self.max_steps

        if success:
            reward += 5.0  # Completion bonus

        return PrimitiveResult(
            observation=self.grid.clone(),
            reward=reward,
            done=done,
            success=success,
            info={
                "clicked": len(self.clicked),
                "total_targets": len(self.targets),
                "last_click_correct": click_correct,
                "steps": self.step_count,
            },
        )

    def _random_empty_pos(self) -> Optional[tuple[int, int]]:
        """Get random empty position."""
        for _ in range(100):
            y = random.randint(0, self.grid_size - 1)
            x = random.randint(0, self.grid_size - 1)
            if self.grid[y, x] == self.EMPTY:
                return (y, x)
        return None

    def get_task_description(self) -> str:
        if self.variant == "sequence":
            return f"Click {self.num_targets} targets in order"
        return f"Click {self.num_targets} target(s) on {self.grid_size}x{self.grid_size} grid"


class ClickPrimitiveGenerator:
    """Generate click primitive tasks with varying difficulty."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        difficulty: int = 1,
        deterministic: bool = False,
    ) -> ClickEnv:
        """
        Generate click task based on difficulty.

        Difficulty 1: Single visible target
        Difficulty 2: Multiple targets (any order)
        Difficulty 3: Multiple targets (sequence)
        Difficulty 4: More targets, sequence
        Difficulty 5: Many targets, sequence, larger grid
        """
        seed = self.rng.randint(0, 2**31)

        if difficulty == 1:
            return ClickEnv(
                grid_size=8,
                num_targets=1,
                max_steps=10,
                seed=seed,
                deterministic=deterministic,
                variant="target",
            )
        elif difficulty == 2:
            return ClickEnv(
                grid_size=10,
                num_targets=3,
                max_steps=15,
                seed=seed,
                deterministic=deterministic,
                variant="target",
            )
        elif difficulty == 3:
            return ClickEnv(
                grid_size=10,
                num_targets=3,
                max_steps=15,
                seed=seed,
                deterministic=deterministic,
                variant="sequence",
                show_order=True,
            )
        elif difficulty == 4:
            return ClickEnv(
                grid_size=12,
                num_targets=5,
                max_steps=20,
                seed=seed,
                deterministic=deterministic,
                variant="sequence",
                show_order=True,
            )
        else:  # difficulty >= 5
            return ClickEnv(
                grid_size=15,
                num_targets=7,
                max_steps=30,
                seed=seed,
                deterministic=deterministic,
                variant="sequence",
                show_order=False,  # No visual hints
            )
