"""Switches solver - toggle all switches on."""

from typing import Optional

from ..training.synthetic_env import Action, CellType, EnvState
from .base import ExpertSolver, SolverResult, find_all_cells
from .navigation import astar, path_to_actions


def get_adjacent_positions(pos: tuple[int, int]) -> list[tuple[int, int]]:
    """Get positions adjacent to pos."""
    y, x = pos
    return [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]


class SwitchesSolver(ExpertSolver):
    """Expert solver for switch tasks (toggle all switches on)."""

    def can_solve(self, state: EnvState) -> bool:
        """Can solve if there are switches and we can reach them."""
        switches = find_all_cells(state.grid, CellType.SWITCH)
        if not switches:
            return True  # No switches = done

        # Check if at least one is reachable (to adjacent cell)
        for switch_pos in switches:
            for adj in get_adjacent_positions(switch_pos):
                if 0 <= adj[0] < state.grid.shape[0] and 0 <= adj[1] < state.grid.shape[1]:
                    if state.grid[adj].item() in {CellType.EMPTY, CellType.AGENT}:
                        path = astar(state.grid, state.agent_pos, adj)
                        if path is not None:
                            return True
        return False

    def solve(self, state: EnvState) -> SolverResult:
        """Get next action to toggle switches."""
        switches = find_all_cells(state.grid, CellType.SWITCH)

        # Find switches that are not yet on
        remaining = [s for s in switches if s not in state.switches_on]

        if not remaining:
            return SolverResult(action=Action.NOOP, solved=True)

        # Check if adjacent to any switch we need to toggle
        for switch_pos in remaining:
            for adj in get_adjacent_positions(switch_pos):
                if state.agent_pos == adj:
                    # We're next to a switch, interact!
                    return SolverResult(action=Action.INTERACT, solved=True)

        # Navigate to nearest switch (to adjacent cell)
        best_path = None
        best_distance = float("inf")

        for switch_pos in remaining:
            for adj in get_adjacent_positions(switch_pos):
                ay, ax = adj
                if not (0 <= ay < state.grid.shape[0] and 0 <= ax < state.grid.shape[1]):
                    continue

                cell = state.grid[ay, ax].item()
                if cell not in {CellType.EMPTY, CellType.AGENT}:
                    continue

                path = astar(state.grid, state.agent_pos, adj)
                if path is not None and len(path) < best_distance:
                    best_distance = len(path)
                    best_path = path

        if best_path is None:
            return SolverResult(action=Action.NOOP, solved=False)

        actions = path_to_actions(best_path)

        if not actions:
            # Already adjacent, interact
            return SolverResult(action=Action.INTERACT, solved=True)

        return SolverResult(
            action=actions[0],
            solved=True,
            path=actions + [Action.INTERACT],
        )

    def get_full_solution(self, state: EnvState) -> Optional[list[int]]:
        """Get complete action sequence to toggle all switches."""
        switches = find_all_cells(state.grid, CellType.SWITCH)
        remaining = [s for s in switches if s not in state.switches_on]

        if not remaining:
            return []

        actions = []
        current_pos = state.agent_pos
        grid = state.grid.clone()

        while remaining:
            # Find nearest switch
            best_target_adj = None
            best_switch = None
            best_path = None
            best_distance = float("inf")

            for switch_pos in remaining:
                for adj in get_adjacent_positions(switch_pos):
                    ay, ax = adj
                    if not (0 <= ay < grid.shape[0] and 0 <= ax < grid.shape[1]):
                        continue

                    cell = grid[ay, ax].item()
                    if cell not in {CellType.EMPTY, CellType.AGENT}:
                        continue

                    path = astar(grid, current_pos, adj)
                    if path is not None and len(path) < best_distance:
                        best_distance = len(path)
                        best_path = path
                        best_target_adj = adj
                        best_switch = switch_pos

            if best_path is None:
                return None

            actions.extend(path_to_actions(best_path))
            actions.append(Action.INTERACT)
            current_pos = best_target_adj
            remaining.remove(best_switch)

        return actions
