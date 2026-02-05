"""
State tracking primitives - remembering and applying state.

Variants:
- memory: Remember shown value, apply later
- counter: Track budget/resources
- multi_property: Track multiple properties (color AND shape)
- sequence: Remember sequence of events
"""

import random
from typing import Optional

import torch

from .base import Action, PrimitiveEnv, PrimitiveFamily, PrimitiveResult


class StateTrackingEnv(PrimitiveEnv):
    """
    State tracking primitive environment.

    Agent must remember and use information across steps.
    """

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 30,
        seed: Optional[int] = None,
        deterministic: bool = False,
        variant: str = "memory",
        num_values: int = 4,
        sequence_length: int = 3,
        budget: int = 5,
    ):
        super().__init__(grid_size, max_steps, seed, deterministic)
        self.variant = variant
        self.num_values = num_values
        self.sequence_length = sequence_length
        self.initial_budget = budget

        # State
        self.grid: Optional[torch.Tensor] = None
        self.phase: str = "show"  # show -> hide -> respond
        self.target_value: Optional[int] = None
        self.target_sequence: list[int] = []
        self.response_sequence: list[int] = []
        self.budget: int = budget
        self.properties: dict = {}  # For multi_property

    @property
    def family(self) -> PrimitiveFamily:
        return PrimitiveFamily.STATE_TRACKING

    @property
    def action_space_size(self) -> int:
        return 9  # CLICK with coordinate for selection

    @property
    def requires_coordinates(self) -> bool:
        return True

    def reset(self) -> torch.Tensor:
        self._reseed()
        self.reset_count += 1
        self.step_count = 0
        self.phase = "show"
        self.response_sequence = []
        self.budget = self.initial_budget

        if self.variant == "memory":
            return self._setup_memory()
        elif self.variant == "counter":
            return self._setup_counter()
        elif self.variant == "multi_property":
            return self._setup_multi_property()
        else:  # sequence
            return self._setup_sequence()

    def _setup_memory(self) -> torch.Tensor:
        """Show a value that agent must remember."""
        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Show target value prominently in center
        self.target_value = random.randint(0, self.num_values - 1)
        center = self.grid_size // 2
        self.grid[center-1:center+2, center-1:center+2] = self.target_value + 1  # +1 to avoid 0=empty

        return self.grid.clone()

    def _setup_counter(self) -> torch.Tensor:
        """Set up budget counting task."""
        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Show budget indicator at top
        self.grid[0, :self.budget] = 1  # Budget bar

        # Place target that costs actions to reach
        self.target_pos = (self.grid_size - 2, self.grid_size // 2)
        self.grid[self.target_pos] = 2  # Target

        # Place agent
        self.agent_pos = (1, self.grid_size // 2)
        self.grid[self.agent_pos] = 3  # Agent

        return self.grid.clone()

    def _setup_multi_property(self) -> torch.Tensor:
        """Set up multi-property tracking (color + position)."""
        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Show object with color and position
        self.properties = {
            "color": random.randint(1, self.num_values),
            "row": random.randint(0, self.grid_size - 1),
            "col": random.randint(0, self.grid_size - 1),
        }

        # Display object
        y, x = self.properties["row"], self.properties["col"]
        self.grid[y, x] = self.properties["color"]

        return self.grid.clone()

    def _setup_sequence(self) -> torch.Tensor:
        """Set up sequence memory task."""
        self.grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        # Generate target sequence
        self.target_sequence = [random.randint(0, self.num_values - 1) for _ in range(self.sequence_length)]
        self.current_show_idx = 0

        # Show first element of sequence
        self._show_sequence_element()

        return self.grid.clone()

    def _show_sequence_element(self):
        """Show current sequence element."""
        self.grid.fill_(0)
        if self.current_show_idx < len(self.target_sequence):
            value = self.target_sequence[self.current_show_idx]
            center = self.grid_size // 2
            self.grid[center, center] = value + 1

    def step(self, action: int, x: Optional[int] = None, y: Optional[int] = None) -> PrimitiveResult:
        if self.grid is None:
            raise RuntimeError("Environment not reset")

        self.step_count += 1
        reward = -0.02
        success = False

        if self.variant == "memory":
            reward, success = self._step_memory(action, x, y)
        elif self.variant == "counter":
            reward, success = self._step_counter(action, x, y)
        elif self.variant == "multi_property":
            reward, success = self._step_multi_property(action, x, y)
        else:  # sequence
            reward, success = self._step_sequence(action, x, y)

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
                "budget": self.budget if self.variant == "counter" else None,
            },
        )

    def _step_memory(self, action: int, x: Optional[int], y: Optional[int]) -> tuple[float, bool]:
        """Handle memory variant step."""
        reward = -0.02
        success = False

        if self.phase == "show":
            # Any action advances to hide phase
            if action != Action.NOOP:
                self.phase = "hide"
                self.grid.fill_(0)  # Hide the value

                # Show response options at bottom
                for i in range(self.num_values):
                    self.grid[self.grid_size - 1, i * 2] = i + 1

        elif self.phase == "hide":
            # Agent must click the remembered value
            if action == Action.CLICK and x is not None and y is not None:
                # Check if clicked correct option
                clicked_value = x // 2
                if clicked_value == self.target_value:
                    success = True
                    reward = 5.0
                else:
                    reward = -1.0

        return reward, success

    def _step_counter(self, action: int, x: Optional[int], y: Optional[int]) -> tuple[float, bool]:
        """Handle counter variant step."""
        reward = -0.02
        success = False

        if self.budget > 0:
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

            if moved:
                self.budget -= 1
                # Update budget display
                self.grid[0, :] = 0
                self.grid[0, :self.budget] = 1

                # Move agent
                y, x = self.agent_pos
                new_y = max(1, min(self.grid_size - 1, y + dy))
                new_x = max(0, min(self.grid_size - 1, x + dx))

                self.grid[y, x] = 0
                self.agent_pos = (new_y, new_x)
                self.grid[new_y, new_x] = 3

                # Check if reached target
                if self.agent_pos == self.target_pos:
                    success = True
                    reward = 5.0 + self.budget  # Bonus for remaining budget

        if self.budget == 0 and not success:
            reward = -2.0  # Ran out of budget

        return reward, success

    def _step_multi_property(self, action: int, x: Optional[int], y: Optional[int]) -> tuple[float, bool]:
        """Handle multi-property variant step."""
        reward = -0.02
        success = False

        if self.phase == "show":
            # Advance to question phase
            if action != Action.NOOP:
                self.phase = "question"
                self.grid.fill_(0)

                # Ask about color - show color options
                for i in range(1, self.num_values + 1):
                    self.grid[self.grid_size - 1, i - 1] = i

        elif self.phase == "question":
            if action == Action.CLICK and x is not None:
                # Clicking indicates color answer
                if x + 1 == self.properties["color"]:
                    reward = 2.0
                    self.phase = "position"
                    self.grid.fill_(0)
                    # Now show position grid
                else:
                    reward = -0.5

        elif self.phase == "position":
            if action == Action.CLICK and x is not None and y is not None:
                # Check position answer
                if y == self.properties["row"] and x == self.properties["col"]:
                    success = True
                    reward = 5.0
                else:
                    reward = -0.5

        return reward, success

    def _step_sequence(self, action: int, x: Optional[int], y: Optional[int]) -> tuple[float, bool]:
        """Handle sequence variant step."""
        reward = -0.02
        success = False

        if self.phase == "show":
            # Advance through sequence showing
            if action != Action.NOOP:
                self.current_show_idx += 1
                if self.current_show_idx >= len(self.target_sequence):
                    self.phase = "respond"
                    self.grid.fill_(0)
                    # Show response options
                    for i in range(self.num_values):
                        self.grid[self.grid_size - 1, i * 2] = i + 1
                else:
                    self._show_sequence_element()

        elif self.phase == "respond":
            if action == Action.CLICK and x is not None:
                clicked_value = x // 2
                expected = self.target_sequence[len(self.response_sequence)]

                if clicked_value == expected:
                    reward = 1.0
                    self.response_sequence.append(clicked_value)

                    if len(self.response_sequence) == len(self.target_sequence):
                        success = True
                        reward = 5.0
                else:
                    reward = -1.0  # Wrong sequence

        return reward, success

    def get_task_description(self) -> str:
        if self.variant == "memory":
            return "Remember the shown value and select it later"
        elif self.variant == "counter":
            return f"Reach target with {self.initial_budget} moves"
        elif self.variant == "multi_property":
            return "Remember both color and position"
        else:
            return f"Repeat the {self.sequence_length}-item sequence"


class StateTrackingGenerator:
    """Generate state tracking tasks with varying difficulty."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def generate(
        self,
        difficulty: int = 1,
        deterministic: bool = False,
    ) -> StateTrackingEnv:
        """
        Generate state tracking task based on difficulty.

        Difficulty 1: Simple memory (4 values)
        Difficulty 2: More values to remember
        Difficulty 3: Counter/budget tracking
        Difficulty 4: Multi-property
        Difficulty 5: Sequence memory
        """
        seed = self.rng.randint(0, 2**31)

        if difficulty == 1:
            return StateTrackingEnv(
                grid_size=8,
                num_values=4,
                max_steps=15,
                seed=seed,
                deterministic=deterministic,
                variant="memory",
            )
        elif difficulty == 2:
            return StateTrackingEnv(
                grid_size=10,
                num_values=6,
                max_steps=20,
                seed=seed,
                deterministic=deterministic,
                variant="memory",
            )
        elif difficulty == 3:
            return StateTrackingEnv(
                grid_size=10,
                budget=8,
                max_steps=20,
                seed=seed,
                deterministic=deterministic,
                variant="counter",
            )
        elif difficulty == 4:
            return StateTrackingEnv(
                grid_size=10,
                num_values=5,
                max_steps=25,
                seed=seed,
                deterministic=deterministic,
                variant="multi_property",
            )
        else:  # difficulty >= 5
            return StateTrackingEnv(
                grid_size=10,
                num_values=5,
                sequence_length=5,
                max_steps=30,
                seed=seed,
                deterministic=deterministic,
                variant="sequence",
            )
