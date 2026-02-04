"""
Simplified ARC-like environment for tractable learning.

Key changes from full ARC:
1. Single-cell output per step (not cursor navigation)
2. Dense rewards (per-cell feedback)
3. Auto-submit when grid is filled
4. Start with very simple patterns
"""

import random
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class SimpleARCState:
    """State for simplified ARC."""
    input_grid: torch.Tensor   # [H, W]
    target_grid: torch.Tensor  # [H, W]
    output_grid: torch.Tensor  # [H, W] - what agent has produced
    current_cell: int          # Which cell to fill next (linear index)
    step_count: int = 0


class SimpleARCEnv:
    """
    Simplified ARC environment.

    Agent fills cells one at a time, left-to-right, top-to-bottom.
    Action = which color to set for current cell.
    Reward = +1 for correct, -1 for wrong.
    """

    def __init__(
        self,
        grid_size: int = 5,  # Smaller for tractability
        task_type: str = "copy",
        num_colors: int = 4,
        seed: Optional[int] = None,
        deterministic: bool = False,  # If True, use seed for reproducibility
    ):
        self.grid_size = grid_size
        self.task_type = task_type
        self.num_colors = num_colors
        self.num_cells = grid_size * grid_size
        self.num_actions = num_colors  # Just pick a color
        self.deterministic = deterministic
        self.base_seed = seed
        self.reset_count = 0

        # Only set global seed if deterministic mode
        if deterministic and seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.state: Optional[SimpleARCState] = None

    def reset(self) -> torch.Tensor:
        """Reset and return observation."""
        # For deterministic eval, reseed based on reset count
        if self.deterministic and self.base_seed is not None:
            seed = self.base_seed + self.reset_count * 7919  # Large prime
            random.seed(seed)
            torch.manual_seed(seed)
        self.reset_count += 1

        if self.task_type == "copy":
            input_grid, target_grid = self._gen_copy()
        elif self.task_type == "fill_single":
            input_grid, target_grid = self._gen_fill_single()
        elif self.task_type == "color_swap":
            input_grid, target_grid = self._gen_color_swap()
        elif self.task_type == "reflect_h":
            input_grid, target_grid = self._gen_reflect_h()
        else:
            input_grid, target_grid = self._gen_copy()

        self.state = SimpleARCState(
            input_grid=input_grid,
            target_grid=target_grid,
            output_grid=torch.zeros_like(input_grid),
            current_cell=0,
            step_count=0,
        )

        return self._get_obs()

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, dict]:
        """
        Take action (set color for current cell).

        Returns: (obs, reward, done, info)
        """
        assert self.state is not None

        # Get current cell position
        y = self.state.current_cell // self.grid_size
        x = self.state.current_cell % self.grid_size

        # Set the color
        self.state.output_grid[y, x] = action

        # Check if correct
        target_color = self.state.target_grid[y, x].item()
        if action == target_color:
            reward = 1.0
        else:
            reward = -0.5

        # Move to next cell
        self.state.current_cell += 1
        self.state.step_count += 1

        # Check if done
        done = self.state.current_cell >= self.num_cells

        info = {}
        if done:
            # Calculate final accuracy
            correct = (self.state.output_grid == self.state.target_grid).float().mean()
            info["accuracy"] = correct.item()
            info["success"] = correct.item() == 1.0

            # Bonus for perfect match
            if info["success"]:
                reward += 5.0

        return self._get_obs(), reward, done, info

    def _get_obs(self) -> torch.Tensor:
        """Get observation: [3, H, W] = input, target_so_far, mask_of_current"""
        H, W = self.grid_size, self.grid_size

        # Input grid
        input_obs = self.state.input_grid.clone()

        # What we've filled so far
        output_obs = self.state.output_grid.clone()

        # Mask showing current cell
        mask = torch.zeros(H, W, dtype=torch.long)
        if self.state.current_cell < self.num_cells:
            y = self.state.current_cell // W
            x = self.state.current_cell % W
            mask[y, x] = 1

        return torch.stack([input_obs, output_obs, mask])

    def _gen_copy(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Simple copy task."""
        grid = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))
        return grid, grid.clone()

    def _gen_fill_single(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fill with a single color."""
        color = random.randint(0, self.num_colors - 1)
        grid = torch.full((self.grid_size, self.grid_size), color, dtype=torch.long)
        return grid, grid.clone()

    def _gen_color_swap(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Swap two colors."""
        grid = torch.randint(0, min(2, self.num_colors), (self.grid_size, self.grid_size))
        target = grid.clone()
        target[grid == 0] = 1
        target[grid == 1] = 0
        return grid, target

    def _gen_reflect_h(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Horizontal reflection."""
        grid = torch.randint(0, self.num_colors, (self.grid_size, self.grid_size))
        target = torch.flip(grid, dims=[1])
        return grid, target
