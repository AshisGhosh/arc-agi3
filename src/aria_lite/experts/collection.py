"""Collection solver - collect all items using greedy nearest neighbor."""

from typing import Optional

from ..training.synthetic_env import Action, CellType, EnvState
from .base import ExpertSolver, SolverResult, find_all_cells, manhattan_distance
from .navigation import astar, path_to_actions


class CollectionSolver(ExpertSolver):
    """Expert solver for collection tasks (collect all collectibles)."""

    def can_solve(self, state: EnvState) -> bool:
        """Can solve if there are collectibles and paths to them."""
        collectibles = find_all_cells(state.grid, CellType.COLLECTIBLE)
        if not collectibles:
            return True  # Already done

        # Check if at least one is reachable
        for target in collectibles:
            path = astar(state.grid, state.agent_pos, target)
            if path is not None:
                return True
        return False

    def solve(self, state: EnvState) -> SolverResult:
        """Get next action to collect nearest item."""
        # Find remaining collectibles (not yet collected)
        all_collectibles = find_all_cells(state.grid, CellType.COLLECTIBLE)

        # Filter out already collected
        remaining = [c for c in all_collectibles if c not in state.collected]

        if not remaining:
            return SolverResult(action=Action.NOOP, solved=True)

        # Find nearest reachable collectible (greedy)
        best_target = None
        best_path = None
        best_distance = float("inf")

        for target in remaining:
            path = astar(state.grid, state.agent_pos, target)
            if path is not None:
                dist = len(path)
                if dist < best_distance:
                    best_distance = dist
                    best_target = target
                    best_path = path

        if best_path is None:
            return SolverResult(action=Action.NOOP, solved=False)

        actions = path_to_actions(best_path)

        if not actions:
            # At the collectible, will auto-collect on next step
            return SolverResult(action=Action.NOOP, solved=True)

        return SolverResult(
            action=actions[0],
            solved=True,
            path=actions,
        )

    def get_full_solution(self, state: EnvState) -> Optional[list[int]]:
        """Get complete action sequence using greedy TSP."""
        all_collectibles = find_all_cells(state.grid, CellType.COLLECTIBLE)
        remaining = [c for c in all_collectibles if c not in state.collected]

        if not remaining:
            return []

        actions = []
        current_pos = state.agent_pos
        current_grid = state.grid.clone()

        while remaining:
            # Find nearest
            best_target = None
            best_path = None
            best_distance = float("inf")

            for target in remaining:
                path = astar(current_grid, current_pos, target)
                if path is not None:
                    dist = len(path)
                    if dist < best_distance:
                        best_distance = dist
                        best_target = target
                        best_path = path

            if best_path is None:
                return None  # Can't reach remaining

            actions.extend(path_to_actions(best_path))
            current_pos = best_target
            remaining.remove(best_target)

        return actions
