"""A* navigation solver."""

import heapq
from typing import Optional

import torch

from ..training.synthetic_env import Action, CellType, EnvState
from .base import ExpertSolver, SolverResult, find_cell, manhattan_distance


def astar(
    grid: torch.Tensor,
    start: tuple[int, int],
    goal: tuple[int, int],
    passable: Optional[set[int]] = None,
) -> Optional[list[tuple[int, int]]]:
    """
    A* pathfinding on grid.

    Args:
        grid: [H, W] grid tensor
        start: Starting position (y, x)
        goal: Goal position (y, x)
        passable: Set of cell types that can be walked on.
                  Defaults to {EMPTY, GOAL, COLLECTIBLE, SWITCH, KEY}

    Returns:
        List of positions from start to goal, or None if no path
    """
    if passable is None:
        passable = {
            CellType.EMPTY,
            CellType.GOAL,
            CellType.COLLECTIBLE,
            CellType.SWITCH,
            CellType.KEY,
        }

    H, W = grid.shape

    # Priority queue: (f_score, counter, position, path)
    counter = 0
    open_set = [(manhattan_distance(start, goal), counter, start, [start])]
    visited = {start}

    while open_set:
        _, _, current, path = heapq.heappop(open_set)

        if current == goal:
            return path

        y, x = current
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            if not (0 <= ny < H and 0 <= nx < W):
                continue

            neighbor = (ny, nx)
            if neighbor in visited:
                continue

            cell = grid[ny, nx].item()
            if cell not in passable and neighbor != goal:
                continue

            visited.add(neighbor)
            new_path = path + [neighbor]
            g_score = len(new_path)
            f_score = g_score + manhattan_distance(neighbor, goal)

            counter += 1
            heapq.heappush(open_set, (f_score, counter, neighbor, new_path))

    return None


def path_to_actions(path: list[tuple[int, int]]) -> list[int]:
    """Convert position path to action sequence."""
    actions = []
    for i in range(len(path) - 1):
        y1, x1 = path[i]
        y2, x2 = path[i + 1]

        dy, dx = y2 - y1, x2 - x1

        if dy == -1:
            actions.append(Action.UP)
        elif dy == 1:
            actions.append(Action.DOWN)
        elif dx == -1:
            actions.append(Action.LEFT)
        elif dx == 1:
            actions.append(Action.RIGHT)

    return actions


class NavigationSolver(ExpertSolver):
    """Expert solver for navigation tasks (reach the goal)."""

    def __init__(self):
        self._cached_path: Optional[list[int]] = None
        self._path_index: int = 0

    def can_solve(self, state: EnvState) -> bool:
        """Can solve if there's a goal and a path to it."""
        goal_pos = find_cell(state.grid, CellType.GOAL)
        if goal_pos is None:
            return False

        path = astar(state.grid, state.agent_pos, goal_pos)
        return path is not None

    def solve(self, state: EnvState) -> SolverResult:
        """Get next action to reach goal."""
        goal_pos = find_cell(state.grid, CellType.GOAL)

        if goal_pos is None:
            return SolverResult(action=Action.NOOP, solved=False)

        # Already at goal
        if state.agent_pos == goal_pos:
            return SolverResult(action=Action.NOOP, solved=True)

        # Find path
        path = astar(state.grid, state.agent_pos, goal_pos)

        if path is None:
            return SolverResult(action=Action.NOOP, solved=False)

        actions = path_to_actions(path)

        if not actions:
            return SolverResult(action=Action.NOOP, solved=True)

        return SolverResult(
            action=actions[0],
            solved=True,
            path=actions,
        )

    def get_full_solution(self, state: EnvState) -> Optional[list[int]]:
        """Get complete action sequence to goal."""
        goal_pos = find_cell(state.grid, CellType.GOAL)
        if goal_pos is None:
            return None

        path = astar(state.grid, state.agent_pos, goal_pos)
        if path is None:
            return None

        return path_to_actions(path)
