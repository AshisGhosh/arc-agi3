"""
Composition primitives - combining multiple skills.

Variants:
- nav_then_click: Navigate to area, then click target
- pattern_then_act: Find pattern, then apply action
- conditional: Action depends on observed state
"""

import random
from typing import Optional

import torch

from .base import Action, PrimitiveEnv, PrimitiveFamily, PrimitiveResult


class CompositionEnv(PrimitiveEnv):
    """
    Composition primitive environment.

    Agent must combine multiple primitive skills.
    """

    EMPTY = 0
    AGENT = 1
    TARGET = 2
    WALL = 3
    GOAL_ZONE = 4
    CLICK_TARGET = 5

    def __init__(
        self,
        grid_size: int = 12,
        max_steps: int = 50,
        seed: Optional[int] = None,
        deterministic: bool = False,
        variant: str = "nav_then_click",
        num_obstacles: int = 5,
    ):
        super().__init__(grid_size, max_steps, seed, deterministic)
        self.variant = variant
        self.num_obstacles = num_obstacles

        # State
        self.grid: Optional[torch.Tensor] = None
        self.agent_pos: Optional[tuple[int, int]] = None
        self.phase: str = "navigate"  # navigate -> click -> done
        self.click_target: Optional[tuple[int, int]] = None
        self.in_zone: bool = False
        self.conditional_value: Optional[int] = None

    @property
    def family(self) -> PrimitiveFamily:
        return PrimitiveFamily.COMPOSITION

    @property
    def action_space_size(self) -> int:
        return 9

    @property
    def requires_coordinates(self) -> bool:
        return True

    def reset(self) -> torch.Tensor:
        self._reseed()
        self.reset_count += 1
        self.step_count = 0
        self.phase = "navigate"
        self.in_zone = False

        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        if self.variant == "nav_then_click":
            return self._setup_nav_then_click()
        elif self.variant == "pattern_then_act":
            return self._setup_pattern_then_act()
        else:  # conditional
            return self._setup_conditional()

    def _setup_nav_then_click(self) -> torch.Tensor:
        """Navigate to zone, then click target within zone."""
        # Create goal zone on right side
        zone_y = random.randint(2, self.grid_size - 4)
        zone_size = 3
        for dy in range(zone_size):
            for dx in range(zone_size):
                self.grid[zone_y + dy, self.grid_size - zone_size - 1 + dx] = self.GOAL_ZONE

        # Place click target inside zone
        self.click_target = (
            zone_y + random.randint(0, zone_size - 1),
            self.grid_size - zone_size - 1 + random.randint(0, zone_size - 1),
        )
        self.grid[self.click_target] = self.CLICK_TARGET

        # Place obstacles
        for _ in range(self.num_obstacles):
            pos = self._random_empty_pos()
            if pos:
                self.grid[pos] = self.WALL

        # Place agent on left side
        self.agent_pos = (random.randint(1, self.grid_size - 2), 1)
        self.grid[self.agent_pos] = self.AGENT

        return self.grid.clone()

    def _setup_pattern_then_act(self) -> torch.Tensor:
        """Find pattern, then click matching location."""
        # Create a simple pattern on left
        pattern_y, pattern_x = 2, 2
        pattern_color = random.randint(1, 3)
        self.grid[pattern_y:pattern_y+2, pattern_x:pattern_x+2] = pattern_color

        # Create matching pattern somewhere on right (this is the target)
        target_y = random.randint(2, self.grid_size - 4)
        target_x = random.randint(self.grid_size // 2, self.grid_size - 4)
        self.grid[target_y:target_y+2, target_x:target_x+2] = pattern_color
        self.click_target = (target_y, target_x)  # Top-left of matching pattern

        # Add distractor patterns with different colors
        for _ in range(3):
            dy = random.randint(0, self.grid_size - 3)
            dx = random.randint(self.grid_size // 2, self.grid_size - 3)
            distractor_color = (pattern_color % 3) + 1  # Different color
            if self.grid[dy, dx] == 0:
                self.grid[dy:dy+2, dx:dx+2] = distractor_color

        # No agent - just click task
        self.agent_pos = None
        self.phase = "click"

        return self.grid.clone()

    def _setup_conditional(self) -> torch.Tensor:
        """Action depends on shown indicator."""
        # Show indicator value at top
        self.conditional_value = random.randint(1, 3)
        self.grid[0, self.grid_size // 2] = self.conditional_value

        # Create click targets for each possible response
        # Target positions depend on conditional value
        target_positions = [
            (self.grid_size - 2, 2),
            (self.grid_size - 2, self.grid_size // 2),
            (self.grid_size - 2, self.grid_size - 3),
        ]

        for i, pos in enumerate(target_positions):
            self.grid[pos] = i + 1  # Different colors for different targets

        # Correct target is the one matching conditional value
        self.click_target = target_positions[self.conditional_value - 1]
        self.phase = "click"
        self.agent_pos = None

        return self.grid.clone()

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> PrimitiveResult:
        if self.grid is None:
            raise RuntimeError("Environment not reset")

        self.step_count += 1
        reward = -0.02
        success = False

        if self.variant == "nav_then_click":
            reward, success = self._step_nav_then_click(action, x, y)
        elif self.variant == "pattern_then_act":
            reward, success = self._step_pattern_then_act(action, x, y)
        else:  # conditional
            reward, success = self._step_conditional(action, x, y)

        done = success or self.step_count >= self.max_steps

        return PrimitiveResult(
            observation=self.grid.clone(),
            reward=reward,
            done=done,
            success=success,
            info={
                "variant": self.variant,
                "phase": self.phase,
                "steps": self.step_count,
            },
        )

    def _step_nav_then_click(self, action: int, x: Optional[int], y: Optional[int]) -> tuple[float, bool]:
        """Handle nav_then_click variant."""
        reward = -0.02
        success = False

        if self.phase == "navigate":
            # Handle movement
            moved = False
            dy, dx = 0, 0

            if action == Action.UP:
                dy, dx = -1, 0
                moved = True
            elif action == Action.DOWN:
                dy, dx = 1, 0
                moved = True
            elif action == Action.LEFT:
                dy, dx = 0, -1
                moved = True
            elif action == Action.RIGHT:
                dy, dx = 0, 1
                moved = True

            if moved and self.agent_pos:
                ay, ax = self.agent_pos
                new_y = ay + dy
                new_x = ax + dx

                # Check bounds and walls
                if 0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size:
                    cell = self.grid[new_y, new_x].item()
                    if cell != self.WALL:
                        # Move agent
                        self.grid[ay, ax] = self.EMPTY
                        self.agent_pos = (new_y, new_x)
                        if cell not in [self.GOAL_ZONE, self.CLICK_TARGET]:
                            self.grid[new_y, new_x] = self.AGENT

                        # Check if in goal zone
                        if cell in [self.GOAL_ZONE, self.CLICK_TARGET]:
                            self.in_zone = True
                            self.phase = "click"
                            reward += 1.0  # Reward for reaching zone
                    else:
                        reward -= 0.1

        elif self.phase == "click":
            if action == Action.CLICK and x is not None and y is not None:
                click_pos = (y, x)
                if click_pos == self.click_target:
                    success = True
                    reward = 5.0
                else:
                    reward = -0.3

        return reward, success

    def _step_pattern_then_act(self, action: int, x: Optional[int], y: Optional[int]) -> tuple[float, bool]:
        """Handle pattern_then_act variant."""
        reward = -0.02
        success = False

        if action == Action.CLICK and x is not None and y is not None:
            click_pos = (y, x)
            if click_pos == self.click_target:
                success = True
                reward = 5.0
            else:
                reward = -0.5

        return reward, success

    def _step_conditional(self, action: int, x: Optional[int], y: Optional[int]) -> tuple[float, bool]:
        """Handle conditional variant."""
        reward = -0.02
        success = False

        if action == Action.CLICK and x is not None and y is not None:
            click_pos = (y, x)
            if click_pos == self.click_target:
                success = True
                reward = 5.0
            else:
                reward = -0.5

        return reward, success

    def _random_empty_pos(self) -> Optional[tuple[int, int]]:
        """Get random empty position."""
        for _ in range(100):
            y = random.randint(0, self.grid_size - 1)
            x = random.randint(0, self.grid_size - 1)
            if self.grid[y, x] == self.EMPTY:
                return (y, x)
        return None

    def get_task_description(self) -> str:
        if self.variant == "nav_then_click":
            return "Navigate to zone, then click target"
        elif self.variant == "pattern_then_act":
            return "Find matching pattern and click it"
        else:
            return "Click target based on indicator value"


class CompositionGenerator:
    """Generate composition tasks with varying difficulty."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        difficulty: int = 1,
        deterministic: bool = False,
    ) -> CompositionEnv:
        """
        Generate composition task based on difficulty.

        Difficulty 1: Simple nav then click
        Difficulty 2: Nav with obstacles
        Difficulty 3: Pattern matching
        Difficulty 4: Conditional actions
        Difficulty 5: Complex compositions
        """
        seed = self.rng.randint(0, 2**31)

        if difficulty == 1:
            return CompositionEnv(
                grid_size=10,
                num_obstacles=0,
                max_steps=30,
                seed=seed,
                deterministic=deterministic,
                variant="nav_then_click",
            )
        elif difficulty == 2:
            return CompositionEnv(
                grid_size=12,
                num_obstacles=8,
                max_steps=40,
                seed=seed,
                deterministic=deterministic,
                variant="nav_then_click",
            )
        elif difficulty == 3:
            return CompositionEnv(
                grid_size=10,
                max_steps=20,
                seed=seed,
                deterministic=deterministic,
                variant="pattern_then_act",
            )
        elif difficulty == 4:
            return CompositionEnv(
                grid_size=10,
                max_steps=15,
                seed=seed,
                deterministic=deterministic,
                variant="conditional",
            )
        else:  # difficulty >= 5
            variant = self.rng.choice(["nav_then_click", "pattern_then_act", "conditional"])
            return CompositionEnv(
                grid_size=15,
                num_obstacles=12,
                max_steps=50,
                seed=seed,
                deterministic=deterministic,
                variant=variant,
            )
