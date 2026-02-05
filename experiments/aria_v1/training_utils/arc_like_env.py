"""
ARC-like Synthetic Environment

Creates tasks closer to actual ARC-AGI-3:
1. Pattern completion - fill in missing cells
2. Transformations - rotate, reflect, translate
3. Copy patterns - replicate with modifications
4. Color mapping - apply color transformations

Unlike navigation tasks, these require:
- Abstract pattern recognition
- Rule inference from examples
- Grid manipulation (not just movement)
"""

import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Callable

import torch


class ARCAction(IntEnum):
    """Actions for ARC-like tasks."""
    # Cursor movement
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    # Cell operations
    SET_COLOR_0 = 4
    SET_COLOR_1 = 5
    SET_COLOR_2 = 6
    SET_COLOR_3 = 7
    SET_COLOR_4 = 8
    SET_COLOR_5 = 9
    SET_COLOR_6 = 10
    SET_COLOR_7 = 11
    SET_COLOR_8 = 12
    SET_COLOR_9 = 13
    # Control
    SUBMIT = 14
    NOOP = 15


@dataclass
class ARCState:
    """State for ARC-like environment."""
    input_grid: torch.Tensor   # [H, W] - the input pattern
    output_grid: torch.Tensor  # [H, W] - current output (agent edits this)
    target_grid: torch.Tensor  # [H, W] - correct answer
    cursor_pos: tuple[int, int]  # Current cursor position
    step_count: int = 0


@dataclass
class ARCStepResult:
    """Result of taking a step."""
    observation: torch.Tensor  # Combined view: [2, H, W] or flattened
    reward: float
    done: bool
    info: dict


class ARCLikeEnv:
    """
    ARC-like environment for pattern manipulation tasks.

    The agent sees an input grid and must produce the correct output grid.
    """

    def __init__(
        self,
        grid_size: int = 10,
        task_type: str = "copy",
        max_steps: int = 100,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.task_type = task_type
        self.max_steps = max_steps

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.state: Optional[ARCState] = None
        self.num_actions = len(ARCAction)

    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        # Generate task based on type
        if self.task_type == "copy":
            input_grid, target_grid = self._generate_copy_task()
        elif self.task_type == "fill":
            input_grid, target_grid = self._generate_fill_task()
        elif self.task_type == "reflect":
            input_grid, target_grid = self._generate_reflect_task()
        elif self.task_type == "rotate":
            input_grid, target_grid = self._generate_rotate_task()
        elif self.task_type == "translate":
            input_grid, target_grid = self._generate_translate_task()
        elif self.task_type == "color_map":
            input_grid, target_grid = self._generate_color_map_task()
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # Start with empty output
        output_grid = torch.zeros_like(input_grid)

        self.state = ARCState(
            input_grid=input_grid,
            output_grid=output_grid,
            target_grid=target_grid,
            cursor_pos=(0, 0),
            step_count=0,
        )

        return self._get_observation()

    def step(self, action: int) -> ARCStepResult:
        """Take an action and return result."""
        assert self.state is not None

        self.state.step_count += 1
        reward = -0.01  # Small step penalty
        done = False
        info = {}

        y, x = self.state.cursor_pos
        H, W = self.state.output_grid.shape

        if action == ARCAction.UP:
            self.state.cursor_pos = (max(0, y - 1), x)
        elif action == ARCAction.DOWN:
            self.state.cursor_pos = (min(H - 1, y + 1), x)
        elif action == ARCAction.LEFT:
            self.state.cursor_pos = (y, max(0, x - 1))
        elif action == ARCAction.RIGHT:
            self.state.cursor_pos = (y, min(W - 1, x + 1))
        elif ARCAction.SET_COLOR_0 <= action <= ARCAction.SET_COLOR_9:
            color = action - ARCAction.SET_COLOR_0
            self.state.output_grid[y, x] = color
        elif action == ARCAction.SUBMIT:
            # Check if output matches target
            if torch.equal(self.state.output_grid, self.state.target_grid):
                reward = 10.0
                info["success"] = True
            else:
                # Partial credit based on accuracy
                correct = (self.state.output_grid == self.state.target_grid).float().mean()
                reward = correct.item() * 5.0 - 2.0  # -2 to +3
                info["success"] = False
                info["accuracy"] = correct.item()
            done = True
        # NOOP does nothing

        # Check max steps
        if self.state.step_count >= self.max_steps:
            done = True

        return ARCStepResult(
            observation=self._get_observation(),
            reward=reward,
            done=done,
            info=info,
        )

    def _get_observation(self) -> torch.Tensor:
        """Get current observation."""
        # Stack input and output grids
        # Shape: [2, H, W]
        obs = torch.stack([
            self.state.input_grid,
            self.state.output_grid,
        ])

        # Add cursor marker to output view
        y, x = self.state.cursor_pos
        # We could encode cursor in a separate channel, but for simplicity
        # we'll just return the stacked grids

        return obs

    def _generate_random_pattern(self, size: int, num_colors: int = 4, density: float = 0.3) -> torch.Tensor:
        """Generate a random pattern."""
        grid = torch.zeros(size, size, dtype=torch.long)
        num_cells = int(size * size * density)

        for _ in range(num_cells):
            y = random.randint(0, size - 1)
            x = random.randint(0, size - 1)
            color = random.randint(1, num_colors)
            grid[y, x] = color

        return grid

    def _generate_copy_task(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy the input pattern exactly."""
        pattern = self._generate_random_pattern(self.grid_size, num_colors=4, density=0.2)
        return pattern, pattern.clone()

    def _generate_fill_task(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill in missing cells of a pattern."""
        # Create a pattern with a clear structure
        size = self.grid_size
        full_pattern = torch.zeros(size, size, dtype=torch.long)

        # Create a simple repeating pattern
        color = random.randint(1, 4)
        step = random.randint(2, 3)

        for i in range(0, size, step):
            for j in range(0, size, step):
                full_pattern[i, j] = color

        # Remove some cells for the input
        input_pattern = full_pattern.clone()
        num_remove = random.randint(3, 6)

        for _ in range(num_remove):
            y = random.randrange(0, size, step)
            x = random.randrange(0, size, step)
            if y < size and x < size:
                input_pattern[y, x] = 0

        return input_pattern, full_pattern

    def _generate_reflect_task(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reflect pattern horizontally or vertically."""
        size = self.grid_size
        half = size // 2

        # Create pattern in left half
        pattern = torch.zeros(size, size, dtype=torch.long)

        for _ in range(random.randint(5, 10)):
            y = random.randint(0, size - 1)
            x = random.randint(0, half - 1)
            color = random.randint(1, 4)
            pattern[y, x] = color

        # Target: reflect to right half
        target = pattern.clone()
        for y in range(size):
            for x in range(half):
                target[y, size - 1 - x] = pattern[y, x]

        return pattern, target

    def _generate_rotate_task(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate pattern 90 degrees."""
        pattern = self._generate_random_pattern(self.grid_size, num_colors=4, density=0.2)

        # Rotate 90 degrees clockwise
        target = torch.rot90(pattern, k=-1)

        return pattern, target

    def _generate_translate_task(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate pattern by offset."""
        size = self.grid_size

        # Create small pattern in corner
        pattern = torch.zeros(size, size, dtype=torch.long)

        for _ in range(random.randint(3, 6)):
            y = random.randint(0, size // 3)
            x = random.randint(0, size // 3)
            color = random.randint(1, 4)
            pattern[y, x] = color

        # Target: translate to different position
        dy = random.randint(size // 3, 2 * size // 3)
        dx = random.randint(size // 3, 2 * size // 3)

        target = torch.zeros_like(pattern)
        for y in range(size):
            for x in range(size):
                if pattern[y, x] > 0:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < size and 0 <= nx < size:
                        target[ny, nx] = pattern[y, x]

        return pattern, target

    def _generate_color_map_task(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Map colors according to a rule."""
        pattern = self._generate_random_pattern(self.grid_size, num_colors=3, density=0.3)

        # Create color mapping (swap colors)
        mapping = {1: 2, 2: 3, 3: 1}

        target = pattern.clone()
        for old_color, new_color in mapping.items():
            target[pattern == old_color] = new_color

        return pattern, target


def get_arc_task_types() -> list[str]:
    """Get list of available ARC-like task types."""
    return ["copy", "fill", "reflect", "rotate", "translate", "color_map"]
