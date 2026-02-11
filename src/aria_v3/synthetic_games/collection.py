"""
Collection synthetic game.

Player navigates grid, collecting objects by moving onto them.
Collected objects disappear. A counter tracks progress.
Level completes when all objects are collected.

Teaches: collectible detection, progress tracking, goal identification,
         combined movement + collection mechanics.
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


class CollectionGame(SyntheticGame):
    """Navigate and collect objects. Counter tracks progress."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        self.player_x = 0
        self.player_y = 0
        self.walls: set[tuple[int, int]] = set()
        self.collectibles: list[tuple[int, int]] = []  # pixel positions
        self.collected: int = 0
        self.total_collectibles: int = config.num_collectibles
        self.grid_cells_x = 64 // config.step_size
        self.grid_cells_y = 64 // config.step_size
        self._level_complete = False
        # Counter display position (top-left corner)
        self.counter_x = 0
        self.counter_y = 0
        self.counter_size = max(4, config.step_size // 2)

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self._level_complete = False
        self.collected = 0
        c = self.config
        rng = self.rng

        all_cells = [
            (gx, gy)
            for gx in range(self.grid_cells_x)
            for gy in range(self.grid_cells_y)
        ]
        rng.shuffle(all_cells)

        # Reserve counter area (top-left 2x2 grid cells)
        counter_cells = set()
        for cx in range(min(2, self.grid_cells_x)):
            for cy in range(min(2, self.grid_cells_y)):
                counter_cells.add((cx, cy))
        all_cells = [(gx, gy) for gx, gy in all_cells if (gx, gy) not in counter_cells]

        # Place player
        self.player_x = all_cells[0][0] * c.step_size
        self.player_y = all_cells[0][1] * c.step_size

        # Place collectibles
        self.collectibles = []
        for i in range(1, 1 + c.num_collectibles):
            if i < len(all_cells):
                gx, gy = all_cells[i]
                self.collectibles.append((gx * c.step_size, gy * c.step_size))
        self.total_collectibles = len(self.collectibles)

        # Place walls
        wall_start = 1 + c.num_collectibles
        num_walls = int(self.grid_cells_x * self.grid_cells_y * c.wall_density)
        self.walls = set()
        for i in range(wall_start, wall_start + num_walls):
            if i < len(all_cells):
                self.walls.add(all_cells[i])

        self._render()
        return self.frame.copy()

    def step(self, action: int, x: int = 0, y: int = 0) -> tuple[np.ndarray, bool, bool]:
        self.step_count += 1
        c = self.config

        dx, dy = 0, 0
        if action == c.action_up:
            dy = -c.step_size
        elif action == c.action_down:
            dy = c.step_size
        elif action == c.action_left:
            dx = -c.step_size
        elif action == c.action_right:
            dx = c.step_size

        if dx != 0 or dy != 0:
            new_x = self.player_x + dx
            new_y = self.player_y + dy

            if 0 <= new_x < 64 and 0 <= new_y < 64:
                grid_x = new_x // c.step_size
                grid_y = new_y // c.step_size
                if (grid_x, grid_y) not in self.walls:
                    self.player_x = new_x
                    self.player_y = new_y

                    # Check collection
                    pos = (self.player_x, self.player_y)
                    if pos in self.collectibles:
                        self.collectibles.remove(pos)
                        self.collected += 1

                        if len(self.collectibles) == 0:
                            self._level_complete = True

        self._render()
        return self.frame.copy(), self._level_complete, False

    def _render(self) -> None:
        c = self.config
        self.frame[:] = c.background_color

        # Draw walls
        for gx, gy in self.walls:
            px, py = gx * c.step_size, gy * c.step_size
            self.render_rect(self.frame, px, py, c.step_size, c.step_size, c.wall_color)

        # Draw collectibles
        for cx, cy in self.collectibles:
            # Draw slightly smaller than cell for visual distinction
            margin = max(1, c.step_size // 4)
            self.render_rect(
                self.frame,
                cx + margin, cy + margin,
                c.step_size - 2 * margin, c.step_size - 2 * margin,
                c.collectible_color,
            )

        # Draw player
        self.render_rect(
            self.frame,
            self.player_x, self.player_y,
            c.step_size, c.step_size,
            c.player_color,
        )

        # Draw counter (small rectangle in top-left, color changes with progress)
        if self.total_collectibles > 0:
            # Counter color intensity based on progress
            progress = self.collected / self.total_collectibles
            # Use counter_color, change brightness/variant based on progress
            counter_val = c.counter_color
            if progress >= 1.0:
                counter_val = c.target_color  # complete!
            self.render_rect(
                self.frame,
                self.counter_x, self.counter_y,
                self.counter_size, self.counter_size,
                counter_val,
            )
            # Show collected count as width of a bar
            bar_width = max(1, int(self.counter_size * 2 * progress))
            self.render_rect(
                self.frame,
                self.counter_x, self.counter_y + self.counter_size,
                bar_width, 2,
                c.counter_color,
            )

    def get_ground_truth(self) -> GroundTruth:
        c = self.config
        gt = GroundTruth(game_type=GameType.COLLECTION)

        gt.entities[c.background_color] = EntityInfo(
            color=c.background_color, role=EntityRole.BACKGROUND
        )
        gt.entities[c.player_color] = EntityInfo(
            color=c.player_color,
            role=EntityRole.PLAYER,
            positions=[(self.player_x, self.player_y)],
        )
        gt.entities[c.wall_color] = EntityInfo(
            color=c.wall_color,
            role=EntityRole.WALL,
            positions=[(gx * c.step_size, gy * c.step_size) for gx, gy in self.walls],
        )
        gt.entities[c.collectible_color] = EntityInfo(
            color=c.collectible_color,
            role=EntityRole.COLLECTIBLE,
            positions=list(self.collectibles),
        )
        gt.entities[c.counter_color] = EntityInfo(
            color=c.counter_color, role=EntityRole.COUNTER
        )

        # Action effects (same as navigation)
        for action_id in [c.action_up, c.action_down, c.action_left, c.action_right]:
            dx, dy = 0, 0
            if action_id == c.action_up:
                dy = -c.step_size
            elif action_id == c.action_down:
                dy = c.step_size
            elif action_id == c.action_left:
                dx = -c.step_size
            elif action_id == c.action_right:
                dx = c.step_size

            gt.action_effects[action_id] = ActionEffect(
                action_id=action_id,
                shift_dx=float(dx),
                shift_dy=float(dy),
                change_prob=1.0 - c.wall_density * 0.5,
                affected_color=c.player_color,
                blocked_prob=c.wall_density * 0.5,
            )

        if 5 in self.available_actions:
            gt.action_effects[5] = ActionEffect(
                action_id=5, change_prob=0.0
            )

        gt.confidence = 1.0
        gt.level_complete = self._level_complete
        return gt

    @property
    def available_actions(self) -> list[int]:
        c = self.config
        actions = [c.action_up, c.action_down, c.action_left, c.action_right]
        if 5 not in actions:
            actions.append(5)
        return sorted(set(actions))
