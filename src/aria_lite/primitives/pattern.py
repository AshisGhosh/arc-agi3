"""
Pattern matching primitives - finding and matching patterns.

Variants:
- match: Find where template appears in grid
- difference: Find cells that differ between grids
- complete: Complete partial pattern
- cycle: Track color cycle state (n clicks = target color)
"""

import random
from typing import Optional

import torch

from .base import Action, PrimitiveEnv, PrimitiveFamily, PrimitiveResult


class PatternEnv(PrimitiveEnv):
    """
    Pattern matching primitive environment.

    Agent must find, match, or complete patterns.
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 20,
        seed: Optional[int] = None,
        deterministic: bool = False,
        variant: str = "match",
        pattern_size: int = 3,
        num_colors: int = 4,
    ):
        super().__init__(grid_size, max_steps, seed, deterministic)
        self.variant = variant
        self.pattern_size = pattern_size
        self.num_colors = num_colors

        # State
        self.grid: Optional[torch.Tensor] = None
        self.template: Optional[torch.Tensor] = None
        self.target_pos: Optional[tuple[int, int]] = None  # Where pattern is located
        self.found: bool = False

    @property
    def family(self) -> PrimitiveFamily:
        return PrimitiveFamily.PATTERN

    @property
    def action_space_size(self) -> int:
        return 9  # CLICK to indicate pattern location

    @property
    def requires_coordinates(self) -> bool:
        return True

    def reset(self) -> torch.Tensor:
        self._reseed()
        self.reset_count += 1
        self.step_count = 0
        self.found = False

        if self.variant == "match":
            return self._setup_match()
        elif self.variant == "difference":
            return self._setup_difference()
        elif self.variant == "complete":
            return self._setup_complete()
        else:  # cycle
            return self._setup_cycle()

    def _setup_match(self) -> torch.Tensor:
        """Set up pattern matching task."""
        # Create random background
        self.grid = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))

        # Create template pattern
        self.template = torch.randint(0, self.num_colors, (self.pattern_size, self.pattern_size))

        # Place template somewhere in grid
        max_pos = self.grid_size - self.pattern_size
        y = random.randint(0, max_pos)
        x = random.randint(0, max_pos)
        self.target_pos = (y, x)

        self.grid[y:y+self.pattern_size, x:x+self.pattern_size] = self.template

        # Return observation: grid with template shown separately
        # We'll use a 2-channel observation: [grid, template_indicator]
        # For simplicity, concatenate template info at bottom
        obs = torch.zeros(self.grid_size + self.pattern_size, self.grid_size, dtype=torch.long)
        obs[:self.grid_size, :] = self.grid
        obs[self.grid_size:self.grid_size+self.pattern_size, :self.pattern_size] = self.template

        return obs

    def _setup_difference(self) -> torch.Tensor:
        """Set up spot-the-difference task."""
        # Create base grid
        base = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))

        # Create copy with one cell different
        self.grid = base.clone()
        y = random.randint(0, self.grid_size - 1)
        x = random.randint(0, self.grid_size - 1)
        self.target_pos = (y, x)

        # Change one cell to different color
        old_color = self.grid[y, x].item()
        new_color = (old_color + 1) % self.num_colors
        self.grid[y, x] = new_color

        # Return both grids stacked
        obs = torch.zeros(self.grid_size * 2, self.grid_size, dtype=torch.long)
        obs[:self.grid_size, :] = base
        obs[self.grid_size:, :] = self.grid

        return obs

    def _setup_complete(self) -> torch.Tensor:
        """Set up pattern completion task."""
        # Create complete pattern
        complete = torch.randint(0, self.num_colors, (self.pattern_size, self.pattern_size))

        # Create incomplete version (one cell hidden)
        self.grid = complete.clone()
        y = random.randint(0, self.pattern_size - 1)
        x = random.randint(0, self.pattern_size - 1)
        self.target_pos = (y, x)
        self.expected_color = complete[y, x].item()
        self.grid[y, x] = self.num_colors  # Use extra value as "unknown"

        # Pad to standard grid size
        obs = torch.full((self.grid_size, self.grid_size), self.num_colors, dtype=torch.long)
        offset = (self.grid_size - self.pattern_size) // 2
        obs[offset:offset+self.pattern_size, offset:offset+self.pattern_size] = self.grid

        return obs

    def _setup_cycle(self) -> torch.Tensor:
        """Set up color cycle tracking task."""
        # Grid with single cell that needs to reach target color
        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Current color (0) and target color
        self.current_color = 0
        self.target_color = random.randint(1, self.num_colors - 1)
        self.target_pos = (self.grid_size // 2, self.grid_size // 2)

        # Mark target cell
        self.grid[self.target_pos] = self.current_color

        # Show target color indicator
        self.grid[0, 0] = self.target_color

        return self.grid.clone()

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> PrimitiveResult:
        if self.grid is None:
            raise RuntimeError("Environment not reset")

        self.step_count += 1
        reward = -0.05
        success = False

        if self.variant == "cycle":
            # Special handling for color cycle
            if action == Action.CLICK and x is not None and y is not None:
                click_pos = (y, x)
                if click_pos == self.target_pos:
                    # Cycle color forward
                    self.current_color = (self.current_color + 1) % self.num_colors
                    self.grid[self.target_pos] = self.current_color
                    reward += 0.1

                    if self.current_color == self.target_color:
                        success = True
                        reward += 5.0
        else:
            # Pattern matching / difference / complete
            if action == Action.CLICK and x is not None and y is not None:
                click_pos = (y, x)

                if self.variant == "match":
                    # Check if clicked position is top-left of pattern
                    if click_pos == self.target_pos:
                        success = True
                        reward += 5.0
                    else:
                        reward -= 0.5
                elif self.variant == "difference":
                    # Check if clicked the different cell
                    if click_pos == self.target_pos:
                        success = True
                        reward += 5.0
                    else:
                        reward -= 0.5
                elif self.variant == "complete":
                    # For complete, click should indicate the color
                    # We use x coordinate as color selection
                    if x == self.expected_color:
                        success = True
                        reward += 5.0
                    else:
                        reward -= 0.5

        done = success or self.step_count >= self.max_steps

        return PrimitiveResult(
            observation=self.grid.clone() if self.variant == "cycle" else self._get_obs(),
            reward=reward,
            done=done,
            success=success,
            info={
                "variant": self.variant,
                "steps": self.step_count,
            },
        )

    def _get_obs(self) -> torch.Tensor:
        """Get current observation."""
        if self.variant == "match":
            obs = torch.zeros(self.grid_size + self.pattern_size, self.grid_size, dtype=torch.long)
            obs[:self.grid_size, :] = self.grid
            obs[self.grid_size:self.grid_size+self.pattern_size, :self.pattern_size] = self.template
            return obs
        elif self.variant == "difference":
            # Already set up in reset
            return self.grid
        else:
            return self.grid.clone()

    def get_task_description(self) -> str:
        if self.variant == "match":
            return f"Find {self.pattern_size}x{self.pattern_size} pattern in grid"
        elif self.variant == "difference":
            return "Find the cell that differs between two grids"
        elif self.variant == "complete":
            return "Determine the missing color in the pattern"
        else:
            return f"Cycle cell to target color ({self.target_color})"


class PatternPrimitiveGenerator:
    """Generate pattern primitive tasks with varying difficulty."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        difficulty: int = 1,
        deterministic: bool = False,
    ) -> PatternEnv:
        """
        Generate pattern task based on difficulty.

        Difficulty 1: Small pattern match
        Difficulty 2: Larger pattern match
        Difficulty 3: Spot the difference
        Difficulty 4: Pattern completion
        Difficulty 5: Color cycling with more colors
        """
        seed = self.rng.randint(0, 2**31)
        variants = ["match", "difference", "complete", "cycle"]

        if difficulty == 1:
            return PatternEnv(
                grid_size=8,
                pattern_size=2,
                num_colors=3,
                max_steps=15,
                seed=seed,
                deterministic=deterministic,
                variant="match",
            )
        elif difficulty == 2:
            return PatternEnv(
                grid_size=10,
                pattern_size=3,
                num_colors=4,
                max_steps=20,
                seed=seed,
                deterministic=deterministic,
                variant="match",
            )
        elif difficulty == 3:
            return PatternEnv(
                grid_size=8,
                pattern_size=3,
                num_colors=4,
                max_steps=15,
                seed=seed,
                deterministic=deterministic,
                variant="difference",
            )
        elif difficulty == 4:
            return PatternEnv(
                grid_size=8,
                pattern_size=4,
                num_colors=4,
                max_steps=15,
                seed=seed,
                deterministic=deterministic,
                variant="complete",
            )
        else:  # difficulty >= 5
            variant = self.rng.choice(variants)
            return PatternEnv(
                grid_size=12,
                pattern_size=4,
                num_colors=6,
                max_steps=25,
                seed=seed,
                deterministic=deterministic,
                variant=variant,
            )
