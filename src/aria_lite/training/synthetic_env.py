"""
ARIA-Lite Synthetic Environment Generator

Procedural grid environment generator for training data augmentation.
Generates ARC-AGI-3-like puzzles with known mechanics and solutions.

Mechanics Library:
1. Navigation - Agent movement on grid
2. Collection - Pick up colored objects
3. Switches - Toggle states with interactions
4. Keys/Doors - Unlock mechanisms
5. Pushing - Move objects by contact
6. Patterns - Match/create color patterns
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import torch


class Action(IntEnum):
    """Available actions in the environment."""

    NOOP = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    INTERACT = 5
    PICK_UP = 6
    DROP = 7


class CellType(IntEnum):
    """Cell types in the grid."""

    EMPTY = 0
    WALL = 1
    AGENT = 2
    GOAL = 3
    KEY = 4
    DOOR = 5
    SWITCH = 6
    COLLECTIBLE = 7
    PUSHABLE = 8
    PATTERN = 9


@dataclass
class EnvState:
    """Current state of the environment."""

    grid: torch.Tensor  # [H, W] cell types
    agent_pos: tuple[int, int]
    inventory: list[int] = field(default_factory=list)
    switches_on: set[tuple[int, int]] = field(default_factory=set)
    collected: set[tuple[int, int]] = field(default_factory=set)
    step_count: int = 0


@dataclass
class StepResult:
    """Result of taking a step."""

    observation: torch.Tensor  # [H, W]
    reward: float
    done: bool
    info: dict


class SyntheticEnv:
    """
    Synthetic ARC-AGI-3-like environment.

    Supports multiple mechanics that can be composed together.
    """

    def __init__(
        self,
        grid_size: int = 16,
        mechanics: Optional[list[str]] = None,
        max_steps: int = 50,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.mechanics = mechanics or ["navigation"]
        self.max_steps = max_steps

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        self.state: Optional[EnvState] = None
        self.goal_condition: Optional[callable] = None

    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Place walls around border
        grid[0, :] = CellType.WALL
        grid[-1, :] = CellType.WALL
        grid[:, 0] = CellType.WALL
        grid[:, -1] = CellType.WALL

        # Place agent
        agent_pos = self._random_empty_pos(grid)
        grid[agent_pos] = CellType.AGENT

        # Initialize state
        self.state = EnvState(
            grid=grid,
            agent_pos=agent_pos,
        )

        # Set up mechanics
        self._setup_mechanics()

        return self._get_observation()

    def step(self, action: int) -> StepResult:
        """Take action and return result."""
        if self.state is None:
            raise RuntimeError("Environment not reset")

        self.state.step_count += 1
        reward = -0.01  # Small step penalty

        # Execute action
        if action == Action.UP:
            reward += self._move((-1, 0))
        elif action == Action.DOWN:
            reward += self._move((1, 0))
        elif action == Action.LEFT:
            reward += self._move((0, -1))
        elif action == Action.RIGHT:
            reward += self._move((0, 1))
        elif action == Action.INTERACT:
            reward += self._interact()
        elif action == Action.PICK_UP:
            reward += self._pick_up()
        elif action == Action.DROP:
            reward += self._drop()

        # Check win condition
        done = False
        if self.goal_condition and self.goal_condition(self.state):
            reward += 10.0
            done = True

        # Check max steps
        if self.state.step_count >= self.max_steps:
            done = True

        return StepResult(
            observation=self._get_observation(),
            reward=reward,
            done=done,
            info={
                "step": self.state.step_count,
                "inventory": self.state.inventory.copy(),
                "collected": len(self.state.collected),
            },
        )

    def _move(self, direction: tuple[int, int]) -> float:
        """Move agent in direction. Returns reward."""
        dy, dx = direction
        y, x = self.state.agent_pos
        new_y, new_x = y + dy, x + dx

        # Check bounds
        if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
            return -0.1

        target_cell = self.state.grid[new_y, new_x].item()

        # Check if blocked
        if target_cell == CellType.WALL:
            return -0.1

        if target_cell == CellType.DOOR and CellType.KEY not in self.state.inventory:
            return -0.1

        # Handle pushable objects
        if target_cell == CellType.PUSHABLE:
            push_reward = self._push((new_y, new_x), direction)
            if push_reward < 0:
                return push_reward

        # Move agent
        self.state.grid[y, x] = CellType.EMPTY
        self.state.grid[new_y, new_x] = CellType.AGENT
        self.state.agent_pos = (new_y, new_x)

        # Check for collectibles
        if target_cell == CellType.COLLECTIBLE:
            self.state.collected.add((new_y, new_x))
            return 1.0

        # Check for goal
        if target_cell == CellType.GOAL:
            return 5.0

        return 0.0

    def _push(self, pos: tuple[int, int], direction: tuple[int, int]) -> float:
        """Push object at pos in direction."""
        dy, dx = direction
        y, x = pos
        new_y, new_x = y + dy, x + dx

        # Check if push target is valid
        if not (0 <= new_y < self.grid_size and 0 <= new_x < self.grid_size):
            return -0.1

        if self.state.grid[new_y, new_x] != CellType.EMPTY:
            return -0.1

        # Move object
        self.state.grid[new_y, new_x] = CellType.PUSHABLE
        self.state.grid[y, x] = CellType.EMPTY

        return 0.1

    def _interact(self) -> float:
        """Interact with adjacent cell."""
        y, x = self.state.agent_pos

        # Check all adjacent cells
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                cell = self.state.grid[ny, nx].item()

                if cell == CellType.SWITCH:
                    if (ny, nx) in self.state.switches_on:
                        self.state.switches_on.remove((ny, nx))
                    else:
                        self.state.switches_on.add((ny, nx))
                    return 0.5

                if cell == CellType.DOOR and CellType.KEY in self.state.inventory:
                    self.state.grid[ny, nx] = CellType.EMPTY
                    self.state.inventory.remove(CellType.KEY)
                    return 1.0

        return -0.05

    def _pick_up(self) -> float:
        """Pick up item at current position."""
        y, x = self.state.agent_pos

        # Check adjacent cells for pickable items
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                cell = self.state.grid[ny, nx].item()

                if cell == CellType.KEY:
                    self.state.inventory.append(CellType.KEY)
                    self.state.grid[ny, nx] = CellType.EMPTY
                    return 1.0

        return -0.05

    def _drop(self) -> float:
        """Drop item from inventory."""
        if not self.state.inventory:
            return -0.05

        y, x = self.state.agent_pos

        # Find empty adjacent cell
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.grid_size and 0 <= nx < self.grid_size:
                if self.state.grid[ny, nx] == CellType.EMPTY:
                    item = self.state.inventory.pop()
                    self.state.grid[ny, nx] = item
                    return 0.0

        return -0.05

    def _get_observation(self) -> torch.Tensor:
        """Get current observation."""
        return self.state.grid.clone()

    def _random_empty_pos(self, grid: torch.Tensor) -> tuple[int, int]:
        """Get random empty position in grid."""
        empty = (grid == CellType.EMPTY).nonzero()
        if len(empty) == 0:
            # Fallback to center-ish position
            return (self.grid_size // 2, self.grid_size // 2)
        idx = random.randint(0, len(empty) - 1)
        return (empty[idx, 0].item(), empty[idx, 1].item())

    def _setup_mechanics(self):
        """Set up mechanics based on config."""
        for mechanic in self.mechanics:
            if mechanic == "navigation":
                self._add_goal()
            elif mechanic == "collection":
                self._add_collectibles()
            elif mechanic == "switches":
                self._add_switches()
            elif mechanic == "keys_doors":
                self._add_keys_doors()
            elif mechanic == "pushing":
                self._add_pushables()
            elif mechanic == "patterns":
                self._add_pattern()

        # Set goal condition based on mechanics
        self._set_goal_condition()

    def _add_goal(self):
        """Add goal position."""
        pos = self._random_empty_pos(self.state.grid)
        self.state.grid[pos] = CellType.GOAL

    def _add_collectibles(self, count: int = 3):
        """Add collectible items."""
        for _ in range(count):
            pos = self._random_empty_pos(self.state.grid)
            self.state.grid[pos] = CellType.COLLECTIBLE

    def _add_switches(self, count: int = 2):
        """Add switches."""
        for _ in range(count):
            pos = self._random_empty_pos(self.state.grid)
            self.state.grid[pos] = CellType.SWITCH

    def _add_keys_doors(self):
        """Add key-door pair."""
        key_pos = self._random_empty_pos(self.state.grid)
        self.state.grid[key_pos] = CellType.KEY

        door_pos = self._random_empty_pos(self.state.grid)
        self.state.grid[door_pos] = CellType.DOOR

    def _add_pushables(self, count: int = 2):
        """Add pushable objects."""
        for _ in range(count):
            pos = self._random_empty_pos(self.state.grid)
            self.state.grid[pos] = CellType.PUSHABLE

    def _add_pattern(self):
        """Add pattern matching area."""
        # Create a small pattern area
        py, px = self.grid_size // 2, self.grid_size // 2
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if 0 <= py + dy < self.grid_size and 0 <= px + dx < self.grid_size:
                    if self.state.grid[py + dy, px + dx] == CellType.EMPTY:
                        self.state.grid[py + dy, px + dx] = CellType.PATTERN

    def _set_goal_condition(self):
        """Set win condition based on mechanics."""
        conditions = []

        if "navigation" in self.mechanics:
            goal_pos = (self.state.grid == CellType.GOAL).nonzero()
            if len(goal_pos) > 0:
                gy, gx = goal_pos[0, 0].item(), goal_pos[0, 1].item()
                conditions.append(lambda s, gy=gy, gx=gx: s.agent_pos == (gy, gx))

        if "collection" in self.mechanics:
            num_collectibles = (self.state.grid == CellType.COLLECTIBLE).sum().item()
            conditions.append(lambda s, n=num_collectibles: len(s.collected) >= n)

        if "switches" in self.mechanics:
            num_switches = (self.state.grid == CellType.SWITCH).sum().item()
            conditions.append(lambda s, n=num_switches: len(s.switches_on) >= n)

        if conditions:
            self.goal_condition = lambda s: all(c(s) for c in conditions)
        else:
            self.goal_condition = lambda s: False


class SyntheticEnvGenerator:
    """
    Generator for diverse synthetic environments.

    Creates environments with random combinations of mechanics.
    """

    MECHANICS = [
        "navigation",
        "collection",
        "switches",
        "keys_doors",
        "pushing",
        "patterns",
    ]

    def __init__(
        self,
        min_grid_size: int = 8,
        max_grid_size: int = 32,
        min_mechanics: int = 1,
        max_mechanics: int = 3,
        min_steps: int = 20,
        max_steps: int = 100,
    ):
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.min_mechanics = min_mechanics
        self.max_mechanics = max_mechanics
        self.min_steps = min_steps
        self.max_steps = max_steps

    def generate(self, seed: Optional[int] = None) -> SyntheticEnv:
        """Generate a random synthetic environment."""
        if seed is not None:
            random.seed(seed)

        grid_size = random.randint(self.min_grid_size, self.max_grid_size)
        num_mechanics = random.randint(self.min_mechanics, self.max_mechanics)
        mechanics = random.sample(self.MECHANICS, num_mechanics)
        max_steps = random.randint(self.min_steps, self.max_steps)

        return SyntheticEnv(
            grid_size=grid_size,
            mechanics=mechanics,
            max_steps=max_steps,
            seed=seed,
        )

    def generate_batch(self, count: int) -> list[SyntheticEnv]:
        """Generate multiple environments."""
        return [self.generate(seed=i) for i in range(count)]


def collect_episode(
    env: SyntheticEnv,
    policy: Optional[callable] = None,
) -> tuple[list[torch.Tensor], list[int], list[float], list[bool]]:
    """
    Collect an episode from the environment.

    Args:
        env: Environment to collect from
        policy: Optional policy function (state -> action). Random if None.

    Returns:
        observations, actions, rewards, dones
    """
    observations = []
    actions = []
    rewards = []
    dones = []

    obs = env.reset()
    observations.append(obs)

    done = False
    while not done:
        if policy is not None:
            action = policy(obs)
        else:
            action = random.randint(0, 7)

        result = env.step(action)

        observations.append(result.observation)
        actions.append(action)
        rewards.append(result.reward)
        dones.append(result.done)

        obs = result.observation
        done = result.done

    return observations, actions, rewards, dones
