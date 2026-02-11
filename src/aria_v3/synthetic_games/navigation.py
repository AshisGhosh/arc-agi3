"""
Grid Navigation synthetic game.

A player entity moves on a grid via directional actions.
Walls block movement. The goal is to reach a target position.

Teaches: player detection, movement mapping, wall detection, pathfinding.
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


class NavigationGame(SyntheticGame):
    """Grid navigation: move player to target, avoiding walls."""

    def __init__(self, config: GameConfig):
        super().__init__(config)
        self.player_x = 0
        self.player_y = 0
        self.target_x = 0
        self.target_y = 0
        self.walls: set[tuple[int, int]] = set()  # grid cell positions
        self.grid_cells_x = 64 // config.step_size
        self.grid_cells_y = 64 // config.step_size
        self._level_complete = False

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self._level_complete = False
        c = self.config
        rng = self.rng

        # Generate wall layout
        self.walls = set()
        num_cells = self.grid_cells_x * self.grid_cells_y
        num_walls = int(num_cells * c.wall_density)

        all_cells = [
            (gx, gy)
            for gx in range(self.grid_cells_x)
            for gy in range(self.grid_cells_y)
        ]
        rng.shuffle(all_cells)

        # Place player and target first (guaranteed not walls)
        self.player_x = all_cells[0][0] * c.step_size
        self.player_y = all_cells[0][1] * c.step_size
        self.target_x = all_cells[1][0] * c.step_size
        self.target_y = all_cells[1][1] * c.step_size

        # Place walls (skip player and target cells)
        wall_candidates = all_cells[2:]
        for i in range(min(num_walls, len(wall_candidates))):
            self.walls.add(wall_candidates[i])

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
        # Action 5 and others: no movement (intentional dead action)

        if dx != 0 or dy != 0:
            new_x = self.player_x + dx
            new_y = self.player_y + dy

            # Bounds check
            if 0 <= new_x < 64 and 0 <= new_y < 64:
                # Wall check (convert to grid cell)
                grid_x = new_x // c.step_size
                grid_y = new_y // c.step_size
                if (grid_x, grid_y) not in self.walls:
                    self.player_x = new_x
                    self.player_y = new_y

        # Check win condition
        if self.player_x == self.target_x and self.player_y == self.target_y:
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

        # Draw target
        self.render_rect(
            self.frame,
            self.target_x, self.target_y,
            c.step_size, c.step_size,
            c.target_color,
        )

        # Draw player (on top)
        self.render_rect(
            self.frame,
            self.player_x, self.player_y,
            c.step_size, c.step_size,
            c.player_color,
        )

    def get_ground_truth(self) -> GroundTruth:
        c = self.config
        gt = GroundTruth(game_type=GameType.NAVIGATION)

        # Entities
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
        gt.entities[c.target_color] = EntityInfo(
            color=c.target_color,
            role=EntityRole.TARGET,
            positions=[(self.target_x, self.target_y)],
        )

        # Action effects
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

            # Estimate blocked probability from wall density
            gt.action_effects[action_id] = ActionEffect(
                action_id=action_id,
                shift_dx=float(dx),
                shift_dy=float(dy),
                change_prob=1.0 - c.wall_density * 0.5,  # approximate
                affected_color=c.player_color,
                blocked_prob=c.wall_density * 0.5,
            )

        # Dead actions
        if 5 in self.available_actions:
            gt.action_effects[5] = ActionEffect(
                action_id=5, change_prob=0.0
            )

        gt.confidence = 1.0  # ground truth is always fully confident
        gt.level_complete = self._level_complete
        return gt

    @property
    def available_actions(self) -> list[int]:
        c = self.config
        actions = [c.action_up, c.action_down, c.action_left, c.action_right]
        # Optionally include a dead action
        if 5 not in actions:
            actions.append(5)
        return sorted(set(actions))
