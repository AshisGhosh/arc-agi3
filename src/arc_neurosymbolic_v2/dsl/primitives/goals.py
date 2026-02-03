"""
Goal-Directedness Primitives (6 total)

Core knowledge prior: Understanding that agents pursue goals and act intentionally.

These primitives handle:
- Movement toward targets
- Avoidance behaviors
- Path planning
- Collection tasks
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, List, Optional, Set, Tuple

if TYPE_CHECKING:
    pass

from .geometry import Direction, direction_to, distance
from .objectness import GridObject, Position


class MoveAction(Enum):
    """Low-level movement actions."""

    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()
    INTERACT = auto()
    WAIT = auto()


@dataclass
class ActionSequence:
    """A sequence of actions to execute."""

    actions: List[MoveAction]
    target: Optional[Position] = None
    estimated_steps: int = 0

    def __len__(self) -> int:
        return len(self.actions)

    def __iter__(self):
        return iter(self.actions)

    def is_empty(self) -> bool:
        return len(self.actions) == 0


def move_toward(agent: GridObject, target: GridObject) -> MoveAction:
    """
    Generate a single action to move agent closer to target.

    Uses greedy movement toward target's center.

    Args:
        agent: The controllable agent object
        target: The target object to approach

    Returns:
        MoveAction that moves toward target
    """
    agent_pos = agent.position
    target_pos = target.position

    direction = direction_to(agent_pos, target_pos)

    # Map direction to action
    direction_to_action = {
        Direction.UP: MoveAction.UP,
        Direction.DOWN: MoveAction.DOWN,
        Direction.LEFT: MoveAction.LEFT,
        Direction.RIGHT: MoveAction.RIGHT,
        Direction.UP_LEFT: MoveAction.LEFT,  # Prefer horizontal
        Direction.UP_RIGHT: MoveAction.RIGHT,
        Direction.DOWN_LEFT: MoveAction.LEFT,
        Direction.DOWN_RIGHT: MoveAction.RIGHT,
        Direction.NONE: MoveAction.WAIT,
    }

    return direction_to_action.get(direction, MoveAction.WAIT)


def move_away(agent: GridObject, threat: GridObject) -> MoveAction:
    """
    Generate a single action to move agent away from threat.

    Args:
        agent: The controllable agent object
        threat: The object to avoid

    Returns:
        MoveAction that moves away from threat
    """
    agent_pos = agent.position
    threat_pos = threat.position

    # Get direction TO threat, then invert
    direction = direction_to(agent_pos, threat_pos)

    # Invert direction
    inverse = {
        Direction.UP: MoveAction.DOWN,
        Direction.DOWN: MoveAction.UP,
        Direction.LEFT: MoveAction.RIGHT,
        Direction.RIGHT: MoveAction.LEFT,
        Direction.UP_LEFT: MoveAction.DOWN_RIGHT
        if hasattr(MoveAction, "DOWN_RIGHT")
        else MoveAction.DOWN,
        Direction.UP_RIGHT: MoveAction.DOWN_LEFT
        if hasattr(MoveAction, "DOWN_LEFT")
        else MoveAction.DOWN,
        Direction.DOWN_LEFT: MoveAction.UP_RIGHT
        if hasattr(MoveAction, "UP_RIGHT")
        else MoveAction.UP,
        Direction.DOWN_RIGHT: MoveAction.UP_LEFT
        if hasattr(MoveAction, "UP_LEFT")
        else MoveAction.UP,
        Direction.NONE: MoveAction.WAIT,
    }

    return inverse.get(direction, MoveAction.WAIT)


def reach(
    agent: GridObject,
    target: GridObject,
    obstacles: Optional[Set[Tuple[int, int]]] = None,
    grid_size: Tuple[int, int] = (64, 64),
) -> ActionSequence:
    """
    Plan a complete path for agent to reach target.

    Uses A* pathfinding with Manhattan heuristic.

    Args:
        agent: The controllable agent object
        target: The target object to reach
        obstacles: Set of blocked positions
        grid_size: (width, height) of the grid

    Returns:
        ActionSequence to reach the target
    """
    obstacles = obstacles or set()
    width, height = grid_size

    start = agent.position.to_tuple()
    goal = target.position.to_tuple()

    # A* search
    open_set: List[Tuple[int, Tuple[int, int]]] = [(0, start)]
    came_from: dict = {}
    g_score = {start: 0}

    while open_set:
        # Get lowest f-score node
        open_set.sort(key=lambda x: x[0])
        _, current = open_set.pop(0)

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                prev = came_from[current]
                dx = current[0] - prev[0]
                dy = current[1] - prev[1]

                if dx > 0:
                    path.append(MoveAction.RIGHT)
                elif dx < 0:
                    path.append(MoveAction.LEFT)
                elif dy > 0:
                    path.append(MoveAction.DOWN)
                elif dy < 0:
                    path.append(MoveAction.UP)

                current = prev

            path.reverse()
            return ActionSequence(
                actions=path,
                target=Position(*goal),
                estimated_steps=len(path),
            )

        # Explore neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            # Bounds check
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue

            # Obstacle check
            if neighbor in obstacles:
                continue

            tentative_g = g_score[current] + 1

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # f = g + h (Manhattan heuristic)
                f_score = tentative_g + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
                open_set.append((f_score, neighbor))

    # No path found
    return ActionSequence(actions=[], target=Position(*goal), estimated_steps=0)


def collect(
    agent: GridObject,
    collectibles: List[GridObject],
    obstacles: Optional[Set[Tuple[int, int]]] = None,
) -> ActionSequence:
    """
    Plan efficient collection of multiple targets.

    Uses greedy nearest-neighbor strategy for simplicity.

    Args:
        agent: The controllable agent object
        collectibles: List of objects to collect
        obstacles: Set of blocked positions

    Returns:
        ActionSequence to collect all targets
    """
    if not collectibles:
        return ActionSequence(actions=[])

    obstacles = obstacles or set()
    all_actions: List[MoveAction] = []

    # Current position (will be updated as we plan)
    current = agent

    remaining = list(collectibles)

    while remaining:
        # Find nearest collectible
        nearest = min(remaining, key=lambda c: distance(current.position, c.position))

        # Plan path to it
        path = reach(current, nearest, obstacles)

        if path.is_empty() and current.position != nearest.position:
            # Can't reach - skip this collectible
            remaining.remove(nearest)
            continue

        # Add path actions
        all_actions.extend(path.actions)

        # Add interact action
        all_actions.append(MoveAction.INTERACT)

        # Update current position
        current = nearest
        remaining.remove(nearest)

    return ActionSequence(
        actions=all_actions,
        estimated_steps=len(all_actions),
    )


def avoid(
    agent: GridObject,
    obstacles: List[GridObject],
    while_reaching: Optional[GridObject] = None,
    grid_size: Tuple[int, int] = (64, 64),
) -> ActionSequence:
    """
    Navigate while avoiding specified objects.

    If while_reaching is specified, plan path to target avoiding obstacles.
    Otherwise, just move away from nearest threat.

    Args:
        agent: The controllable agent object
        obstacles: List of objects to avoid
        while_reaching: Optional target to reach while avoiding
        grid_size: Grid dimensions

    Returns:
        ActionSequence that avoids obstacles
    """
    # Build obstacle set
    blocked: Set[Tuple[int, int]] = set()
    for obs in obstacles:
        blocked.update(obs.pixels)

        # Also add buffer zone around each obstacle
        for px, py in obs.pixels:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    blocked.add((px + dx, py + dy))

    if while_reaching:
        # Plan path avoiding obstacles
        return reach(agent, while_reaching, blocked, grid_size)
    else:
        # Just move away from nearest threat
        if not obstacles:
            return ActionSequence(actions=[])

        nearest_threat = min(obstacles, key=lambda o: distance(agent.position, o.position))
        action = move_away(agent, nearest_threat)
        return ActionSequence(actions=[action], estimated_steps=1)


def approach_until(
    agent: GridObject,
    target: GridObject,
    condition: Callable[[GridObject, GridObject], bool],
    obstacles: Optional[Set[Tuple[int, int]]] = None,
    max_steps: int = 100,
) -> ActionSequence:
    """
    Move toward target until a condition is met.

    Args:
        agent: The controllable agent
        target: Target to approach
        condition: Function(agent, target) -> bool, stop when True
        obstacles: Blocked positions
        max_steps: Maximum steps to plan

    Returns:
        ActionSequence that approaches until condition
    """
    obstacles = obstacles or set()
    actions: List[MoveAction] = []

    # Simulate movement
    current_pos = agent.position

    for _ in range(max_steps):
        # Check if condition is met (approximately)
        simulated_agent = GridObject(
            object_id=agent.object_id,
            color=agent.color,
            pixels=frozenset([(current_pos.x, current_pos.y)]),
            bounding_box=agent.bounding_box,
            shape=agent.shape,
            object_type=agent.object_type,
        )

        if condition(simulated_agent, target):
            break

        # Get direction toward target
        direction = direction_to(current_pos, target.position)

        # Determine next position
        dx, dy = 0, 0
        if direction in (Direction.UP, Direction.UP_LEFT, Direction.UP_RIGHT):
            dy = -1
            actions.append(MoveAction.UP)
        elif direction in (Direction.DOWN, Direction.DOWN_LEFT, Direction.DOWN_RIGHT):
            dy = 1
            actions.append(MoveAction.DOWN)
        elif direction in (Direction.LEFT,):
            dx = -1
            actions.append(MoveAction.LEFT)
        elif direction in (Direction.RIGHT,):
            dx = 1
            actions.append(MoveAction.RIGHT)
        else:
            break  # Already at target

        next_pos = Position(current_pos.x + dx, current_pos.y + dy)

        # Check for obstacles
        if next_pos.to_tuple() in obstacles:
            break  # Blocked

        current_pos = next_pos

    return ActionSequence(
        actions=actions,
        target=target.position,
        estimated_steps=len(actions),
    )
