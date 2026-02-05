"""
Navigation primitives - moving agents to targets.

Variants:
- direct: Move to visible target
- obstacles: Navigate around walls
- waypoints: Visit multiple targets in order
- timed: Reach target within step limit
"""

import random
from typing import Optional

import torch

from .base import Action, PrimitiveEnv, PrimitiveFamily, PrimitiveResult


class NavigationEnv(PrimitiveEnv):
    """
    Navigation primitive environment.

    Agent must navigate to goal position(s) on a grid.
    """

    EMPTY = 0
    WALL = 1
    AGENT = 2
    GOAL = 3
    VISITED = 4  # For waypoint tracking

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 50,
        seed: Optional[int] = None,
        deterministic: bool = False,
        variant: str = "direct",
        num_obstacles: int = 0,
        num_waypoints: int = 1,
    ):
        super().__init__(grid_size, max_steps, seed, deterministic)
        self.variant = variant
        self.num_obstacles = num_obstacles
        self.num_waypoints = num_waypoints

        # State
        self.grid: Optional[torch.Tensor] = None
        self.agent_pos: Optional[tuple[int, int]] = None
        self.goals: list[tuple[int, int]] = []
        self.visited_goals: set[tuple[int, int]] = set()

    @property
    def family(self) -> PrimitiveFamily:
        return PrimitiveFamily.NAVIGATION

    @property
    def action_space_size(self) -> int:
        return 5  # NOOP, UP, DOWN, LEFT, RIGHT

    def reset(self) -> torch.Tensor:
        self._reseed()
        self.reset_count += 1
        self.step_count = 0

        # Initialize grid
        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Place agent
        self.agent_pos = self._random_pos()
        self.grid[self.agent_pos] = self.AGENT

        # Place obstacles (walls)
        for _ in range(self.num_obstacles):
            pos = self._random_empty_pos()
            if pos:
                self.grid[pos] = self.WALL

        # Place goals
        self.goals = []
        self.visited_goals = set()
        for _ in range(self.num_waypoints):
            pos = self._random_empty_pos()
            if pos:
                self.goals.append(pos)
                self.grid[pos] = self.GOAL

        return self.grid.clone()

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> PrimitiveResult:
        if self.grid is None:
            raise RuntimeError("Environment not reset")

        self.step_count += 1
        reward = -0.01  # Small step penalty

        # Execute movement
        if action == Action.UP:
            reward += self._move(-1, 0)
        elif action == Action.DOWN:
            reward += self._move(1, 0)
        elif action == Action.LEFT:
            reward += self._move(0, -1)
        elif action == Action.RIGHT:
            reward += self._move(0, 1)

        # Check if agent reached any unvisited goal
        if self.agent_pos in self.goals and self.agent_pos not in self.visited_goals:
            self.visited_goals.add(self.agent_pos)
            reward += 1.0  # Reward for reaching goal
            # Mark as visited
            self.grid[self.agent_pos] = self.VISITED

        # Check completion
        success = len(self.visited_goals) == len(self.goals)
        done = success or self.step_count >= self.max_steps

        if success:
            reward += 5.0  # Bonus for completing all goals

        return PrimitiveResult(
            observation=self.grid.clone(),
            reward=reward,
            done=done,
            success=success,
            info={
                "goals_reached": len(self.visited_goals),
                "total_goals": len(self.goals),
                "steps": self.step_count,
            },
        )

    def _move(self, dy: int, dx: int) -> float:
        """Attempt to move agent, return reward delta."""
        assert self.agent_pos is not None
        y, x = self.agent_pos
        new_y, new_x = y + dy, x + dx

        # Check bounds
        if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
            return -0.1  # Penalty for hitting boundary

        # Check wall
        if self.grid[new_y, new_x] == self.WALL:
            return -0.1  # Penalty for hitting wall

        # Move agent
        self.grid[y, x] = self.EMPTY
        self.agent_pos = (new_y, new_x)

        # Don't overwrite goal marker until it's visited
        if self.agent_pos not in self.goals or self.agent_pos in self.visited_goals:
            self.grid[new_y, new_x] = self.AGENT
        else:
            self.grid[new_y, new_x] = self.AGENT  # Temporarily show agent

        return 0.0

    def _random_pos(self) -> tuple[int, int]:
        """Get random position on grid."""
        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        )

    def _random_empty_pos(self) -> Optional[tuple[int, int]]:
        """Get random empty position."""
        for _ in range(100):
            pos = self._random_pos()
            if self.grid[pos] == self.EMPTY:
                return pos
        return None

    def get_task_description(self) -> str:
        if self.num_waypoints == 1:
            return f"Navigate to goal on {self.grid_size}x{self.grid_size} grid"
        return f"Visit {self.num_waypoints} waypoints on {self.grid_size}x{self.grid_size} grid"


class NavigationPrimitiveGenerator:
    """Generate navigation primitive tasks with varying difficulty."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        difficulty: int = 1,
        deterministic: bool = False,
    ) -> NavigationEnv:
        """
        Generate navigation task based on difficulty.

        Difficulty 1: Direct path, no obstacles
        Difficulty 2: Few obstacles
        Difficulty 3: Many obstacles
        Difficulty 4: Multiple waypoints
        Difficulty 5: Many waypoints + obstacles
        """
        seed = self.rng.randint(0, 2**31) if not deterministic else self.rng.randint(0, 2**31)

        if difficulty == 1:
            return NavigationEnv(
                grid_size=8,
                num_obstacles=0,
                num_waypoints=1,
                max_steps=30,
                seed=seed,
                deterministic=deterministic,
                variant="direct",
            )
        elif difficulty == 2:
            return NavigationEnv(
                grid_size=10,
                num_obstacles=5,
                num_waypoints=1,
                max_steps=40,
                seed=seed,
                deterministic=deterministic,
                variant="obstacles",
            )
        elif difficulty == 3:
            return NavigationEnv(
                grid_size=12,
                num_obstacles=15,
                num_waypoints=1,
                max_steps=50,
                seed=seed,
                deterministic=deterministic,
                variant="obstacles",
            )
        elif difficulty == 4:
            return NavigationEnv(
                grid_size=10,
                num_obstacles=3,
                num_waypoints=3,
                max_steps=60,
                seed=seed,
                deterministic=deterministic,
                variant="waypoints",
            )
        else:  # difficulty >= 5
            return NavigationEnv(
                grid_size=15,
                num_obstacles=20,
                num_waypoints=5,
                max_steps=100,
                seed=seed,
                deterministic=deterministic,
                variant="waypoints",
            )
