"""
Click Puzzle synthetic game.

Click on cells to toggle their colors. Goal: match a target pattern.

Teaches: click-target identification, state-toggle mechanics, pattern goals.
"""

from __future__ import annotations

import numpy as np

from .base import (
    ActionEffect,
    EntityInfo,
    EntityRole,
    GameConfig,
    GameType,
    GroundTruth,
    SyntheticGame,
)


class ClickPuzzleGame(SyntheticGame):
    """Click puzzle: click cells to toggle colors toward target pattern."""

    def __init__(self, config: GameConfig, puzzle_type: str = "toggle"):
        """
        Args:
            config: Game configuration
            puzzle_type: "toggle" (click toggles between 2 colors),
                        "cycle" (click cycles through N colors),
                        "lights_out" (click toggles neighbors too)
        """
        super().__init__(config)
        self.puzzle_type = puzzle_type
        self.cell_size = config.step_size  # size of each clickable cell
        self.grid_w = 64 // self.cell_size
        self.grid_h = 64 // self.cell_size
        self.grid: np.ndarray = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        self.target: np.ndarray = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        self.color_a = config.background_color  # "off" color
        self.color_b = config.target_color  # "on" color
        self.cycle_colors: list[int] = []
        self._level_complete = False

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self._level_complete = False
        c = self.config
        rng = self.rng

        if self.puzzle_type == "cycle":
            # Use 3-4 colors for cycling
            num_cycle = rng.randint(3, 5)
            all_colors = list(range(16))
            rng.shuffle(all_colors)
            self.cycle_colors = all_colors[:num_cycle]
            self.color_a = self.cycle_colors[0]
            self.color_b = self.cycle_colors[-1]
        else:
            self.color_a = c.background_color
            self.color_b = c.target_color

        # Generate target pattern
        if self.puzzle_type == "lights_out":
            # Target is all same color
            self.target[:] = self.color_a
            # Start state: random presses from solved state
            self.grid[:] = self.color_a
            num_presses = rng.randint(3, min(self.grid_w * self.grid_h, 10))
            for _ in range(num_presses):
                gx = rng.randint(self.grid_w)
                gy = rng.randint(self.grid_h)
                self._toggle_lights_out(gx, gy)
        else:
            # Random target pattern
            if self.puzzle_type == "cycle":
                self.target = rng.choice(
                    self.cycle_colors, size=(self.grid_h, self.grid_w)
                ).astype(np.uint8)
            else:
                self.target = rng.choice(
                    [self.color_a, self.color_b], size=(self.grid_h, self.grid_w)
                ).astype(np.uint8)

            # Start state: randomize grid (different from target)
            if self.puzzle_type == "cycle":
                self.grid = rng.choice(
                    self.cycle_colors, size=(self.grid_h, self.grid_w)
                ).astype(np.uint8)
            else:
                self.grid = rng.choice(
                    [self.color_a, self.color_b], size=(self.grid_h, self.grid_w)
                ).astype(np.uint8)

            # Ensure at least some cells differ
            if np.array_equal(self.grid, self.target):
                gy, gx = rng.randint(self.grid_h), rng.randint(self.grid_w)
                self.grid[gy, gx] = self.color_b if self.grid[gy, gx] == self.color_a else self.color_a

        self._render()
        return self.frame.copy()

    def step(self, action: int, x: int = 0, y: int = 0) -> tuple[np.ndarray, bool, bool]:
        self.step_count += 1

        if action == 6:
            # Click action
            gx = min(x // self.cell_size, self.grid_w - 1)
            gy = min(y // self.cell_size, self.grid_h - 1)

            if self.puzzle_type == "toggle":
                self.grid[gy, gx] = self.color_b if self.grid[gy, gx] == self.color_a else self.color_a
            elif self.puzzle_type == "cycle":
                current = int(self.grid[gy, gx])
                try:
                    idx = self.cycle_colors.index(current)
                    next_idx = (idx + 1) % len(self.cycle_colors)
                except ValueError:
                    next_idx = 0
                self.grid[gy, gx] = self.cycle_colors[next_idx]
            elif self.puzzle_type == "lights_out":
                self._toggle_lights_out(gx, gy)

        # Check win
        if np.array_equal(self.grid, self.target):
            self._level_complete = True

        self._render()
        return self.frame.copy(), self._level_complete, False

    def _toggle_lights_out(self, gx: int, gy: int) -> None:
        """Toggle a cell and its 4 neighbors."""
        for dx, dy in [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = gx + dx, gy + dy
            if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                self.grid[ny, nx] = (
                    self.color_b if self.grid[ny, nx] == self.color_a else self.color_a
                )

    def _render(self) -> None:
        """Render the grid and a small target indicator."""
        # Main grid
        for gy in range(self.grid_h):
            for gx in range(self.grid_w):
                px = gx * self.cell_size
                py = gy * self.cell_size
                self.render_rect(
                    self.frame, px, py,
                    self.cell_size, self.cell_size,
                    int(self.grid[gy, gx]),
                )

        # Draw thin grid lines using wall_color for visual structure
        c = self.config
        if self.cell_size >= 4:
            for gx in range(1, self.grid_w):
                px = gx * self.cell_size
                self.frame[:, px] = c.wall_color
            for gy in range(1, self.grid_h):
                py = gy * self.cell_size
                self.frame[py, :] = c.wall_color

    def get_ground_truth(self) -> GroundTruth:
        c = self.config
        gt = GroundTruth(game_type=GameType.CLICK_PUZZLE)

        # Entities
        gt.entities[self.color_a] = EntityInfo(
            color=self.color_a, role=EntityRole.BACKGROUND
        )
        gt.entities[self.color_b] = EntityInfo(
            color=self.color_b, role=EntityRole.TARGET
        )
        if c.wall_color not in (self.color_a, self.color_b):
            gt.entities[c.wall_color] = EntityInfo(
                color=c.wall_color, role=EntityRole.WALL  # grid lines
            )

        # Action effects: only click works
        gt.action_effects[6] = ActionEffect(
            action_id=6,
            change_prob=1.0,
            affected_color=self.color_b,
        )

        # Other actions do nothing
        for a in range(1, 6):
            gt.action_effects[a] = ActionEffect(
                action_id=a, change_prob=0.0
            )

        gt.confidence = 1.0
        gt.level_complete = self._level_complete
        return gt

    @property
    def available_actions(self) -> list[int]:
        return [6]
