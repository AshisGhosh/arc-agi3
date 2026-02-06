"""
A* Navigation - Pathfinding without ML.

Pure algorithmic pathfinding.
"""

import heapq
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class PathResult:
    """Result of pathfinding."""
    found: bool
    path: list[tuple[int, int]]  # List of (x, y) positions
    actions: list[int]  # List of actions to take
    distance: int


class AStarNavigator:
    """
    A* pathfinding on a grid.

    No ML - pure algorithm.
    """

    # Action mapping
    ACTIONS = {
        1: (0, -1),   # UP
        2: (0, 1),    # DOWN
        3: (-1, 0),   # LEFT
        4: (1, 0),    # RIGHT
    }

    def __init__(self, step_size: int = 1):
        self.step_size = step_size

    def find_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: set[tuple[int, int]],
        grid_size: tuple[int, int],
    ) -> PathResult:
        """
        Find path from start to goal avoiding obstacles.

        Args:
            start: (x, y) starting position
            goal: (x, y) goal position
            obstacles: Set of (x, y) positions that are blocked
            grid_size: (width, height) of the grid

        Returns:
            PathResult with path and actions
        """
        if start == goal:
            return PathResult(found=True, path=[start], actions=[], distance=0)

        width, height = grid_size

        # Priority queue: (f_score, count, x, y, path, actions)
        count = 0
        open_set = [(self._heuristic(start, goal), count, start[0], start[1], [start], [])]
        closed_set = set()

        while open_set:
            _, _, x, y, path, actions = heapq.heappop(open_set)
            current = (x, y)

            if current in closed_set:
                continue

            closed_set.add(current)

            # Check if close enough to goal
            dist_to_goal = self._heuristic(current, goal)
            if dist_to_goal <= self.step_size:
                # If we have actions, we found a path
                if actions:
                    return PathResult(
                        found=True,
                        path=path + [goal],
                        actions=actions,
                        distance=len(path),
                    )
                # If no actions but close to goal, give one step toward goal
                elif dist_to_goal > 0:
                    dx = goal[0] - current[0]
                    dy = goal[1] - current[1]
                    if abs(dx) >= abs(dy):
                        action = 4 if dx > 0 else 3  # RIGHT or LEFT
                    else:
                        action = 2 if dy > 0 else 1  # DOWN or UP
                    return PathResult(
                        found=True,
                        path=path + [goal],
                        actions=[action],
                        distance=1,
                    )
                else:
                    # Already at goal
                    return PathResult(
                        found=True,
                        path=path,
                        actions=[],
                        distance=0,
                    )

            # Explore neighbors
            for action, (dx, dy) in self.ACTIONS.items():
                nx = x + dx * self.step_size
                ny = y + dy * self.step_size

                # Bounds check
                if not (0 <= nx < width and 0 <= ny < height):
                    continue

                next_pos = (nx, ny)

                # Obstacle check
                if next_pos in obstacles:
                    continue

                if next_pos in closed_set:
                    continue

                # Add to open set
                g_score = len(path)
                f_score = g_score + self._heuristic(next_pos, goal)
                count += 1
                heapq.heappush(
                    open_set,
                    (f_score, count, nx, ny, path + [next_pos], actions + [action])
                )

        # No path found
        return PathResult(found=False, path=[], actions=[], distance=-1)

    def _heuristic(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_next_action(
        self,
        current: tuple[int, int],
        goal: tuple[int, int],
        obstacles: set[tuple[int, int]],
        grid_size: tuple[int, int],
    ) -> Optional[int]:
        """
        Get the next action to take toward goal.

        Returns action (1-4) or None if no path exists.
        """
        result = self.find_path(current, goal, obstacles, grid_size)
        if result.found and result.actions:
            return result.actions[0]
        return None


def positions_of_color(frame: np.ndarray, color: int) -> set[tuple[int, int]]:
    """Get all positions with a specific color."""
    positions = set()
    ys, xs = np.where(frame == color)
    for x, y in zip(xs, ys):
        positions.add((int(x), int(y)))
    return positions


def find_nearest(
    current: tuple[int, int],
    targets: set[tuple[int, int]],
) -> Optional[tuple[int, int]]:
    """Find the nearest target by Manhattan distance."""
    if not targets:
        return None

    best = None
    best_dist = float('inf')

    for target in targets:
        dist = abs(target[0] - current[0]) + abs(target[1] - current[1])
        if dist < best_dist:
            best_dist = dist
            best = target

    return best


def test_navigation():
    """Test A* navigation."""
    nav = AStarNavigator(step_size=1)

    # Simple test
    start = (0, 0)
    goal = (5, 5)
    obstacles = {(2, 2), (2, 3), (3, 2)}
    grid_size = (10, 10)

    result = nav.find_path(start, goal, obstacles, grid_size)
    print(f"Path found: {result.found}")
    print(f"Path length: {result.distance}")
    print(f"Actions: {result.actions[:10]}...")

    # Test next action
    action = nav.get_next_action(start, goal, obstacles, grid_size)
    print(f"Next action from {start} to {goal}: {action}")


if __name__ == "__main__":
    test_navigation()
