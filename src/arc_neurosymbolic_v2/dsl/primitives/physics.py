"""
Elementary Physics Primitives (8 total)

Core knowledge prior: Basic physical concepts like containment, support, and collision.

Note: These are "intuitive physics" concepts that humans understand from infancy,
not formal physics simulations.

These primitives handle:
- Containment relationships
- Support and gravity
- Occlusion and visibility
- Movement blocking
- Collision prediction
"""

from __future__ import annotations

from typing import Optional, Set

from .geometry import Direction
from .objectness import GridObject, Position


def contains(container: GridObject, obj: GridObject) -> bool:
    """
    Check if a container object contains another object.

    A container "contains" an object if:
    1. The object is within the container's bounding box
    2. The container forms a closed boundary around the object

    Args:
        container: The potential container object
        obj: The object that might be contained

    Returns:
        True if container contains obj
    """
    # Check bounding box containment first
    c_bbox = container.bounding_box
    o_bbox = obj.bounding_box

    if not (
        c_bbox.x_min <= o_bbox.x_min
        and c_bbox.y_min <= o_bbox.y_min
        and c_bbox.x_max >= o_bbox.x_max
        and c_bbox.y_max >= o_bbox.y_max
    ):
        return False

    # Check that obj's pixels are not part of container's pixels
    # (i.e., obj is in the "interior" space)
    if obj.pixels & container.pixels:
        return False

    # Check that container forms a boundary
    # Simplified: check that all edges of obj's bbox have container pixels
    has_top = any(
        (x, o_bbox.y_min - 1) in container.pixels for x in range(o_bbox.x_min, o_bbox.x_max + 1)
    )
    has_bottom = any(
        (x, o_bbox.y_max + 1) in container.pixels for x in range(o_bbox.x_min, o_bbox.x_max + 1)
    )
    has_left = any(
        (o_bbox.x_min - 1, y) in container.pixels for y in range(o_bbox.y_min, o_bbox.y_max + 1)
    )
    has_right = any(
        (o_bbox.x_max + 1, y) in container.pixels for y in range(o_bbox.y_min, o_bbox.y_max + 1)
    )

    return has_top and has_bottom and has_left and has_right


def supports(supporter: GridObject, obj: GridObject) -> bool:
    """
    Check if supporter is holding up obj (in a gravity context).

    Assumes gravity pulls downward (positive y direction).
    An object is "supported" if there's another object directly below it.

    Args:
        supporter: The potential supporting object
        obj: The object that might be supported

    Returns:
        True if supporter is directly below and touching obj
    """
    # For each pixel in obj, check if there's a supporter pixel directly below
    for ox, oy in obj.pixels:
        below_pos = (ox, oy + 1)
        if below_pos in supporter.pixels:
            return True

    return False


def occludes(
    front_obj: GridObject,
    back_obj: GridObject,
    viewpoint: Optional[Position] = None,
) -> bool:
    """
    Check if front_obj blocks the view of back_obj.

    If viewpoint is None, assumes top-down view where overlapping pixels occlude.
    If viewpoint is specified, does simple ray-casting check.

    Args:
        front_obj: Object potentially blocking view
        back_obj: Object potentially being occluded
        viewpoint: Optional viewer position

    Returns:
        True if front_obj occludes any part of back_obj
    """
    if viewpoint is None:
        # Top-down view: check pixel overlap
        return bool(front_obj.pixels & back_obj.pixels)

    # Ray-casting from viewpoint
    # Simplified: check if any ray from viewpoint to back_obj passes through front_obj
    vx, vy = viewpoint.x, viewpoint.y

    for bx, by in back_obj.pixels:
        # Check points along the ray from viewpoint to this back pixel
        dx = bx - vx
        dy = by - vy
        steps = max(abs(dx), abs(dy))

        if steps == 0:
            continue

        for i in range(1, steps):
            rx = int(vx + dx * i / steps)
            ry = int(vy + dy * i / steps)
            if (rx, ry) in front_obj.pixels:
                return True

    return False


def blocks_movement(
    obstacle: GridObject,
    from_pos: Position,
    to_pos: Position,
) -> bool:
    """
    Check if obstacle prevents movement between two positions.

    Args:
        obstacle: The potential blocking object
        from_pos: Starting position
        to_pos: Target position

    Returns:
        True if obstacle blocks the path
    """
    # Check if to_pos is within obstacle
    if to_pos.to_tuple() in obstacle.pixels:
        return True

    # Check intermediate positions (for diagonal movement)
    dx = to_pos.x - from_pos.x
    dy = to_pos.y - from_pos.y

    steps = max(abs(dx), abs(dy))
    if steps <= 1:
        return to_pos.to_tuple() in obstacle.pixels

    for i in range(1, steps + 1):
        ix = from_pos.x + int(dx * i / steps)
        iy = from_pos.y + int(dy * i / steps)
        if (ix, iy) in obstacle.pixels:
            return True

    return False


def can_pass_through(
    obj: GridObject,
    learned_passable: Optional[Set[int]] = None,
) -> bool:
    """
    Determine if an object is passable.

    This is typically learned from interaction - objects that don't
    block the agent are passable.

    Args:
        obj: Object to check
        learned_passable: Set of object colors known to be passable

    Returns:
        True if object can be passed through
    """
    learned_passable = learned_passable or set()

    # Check learned passability
    if obj.color in learned_passable:
        return True

    # Heuristics for unlearned objects:
    # - Very small objects (1-2 pixels) might be passable
    # - Background-like colors (0) are passable
    if obj.color == 0:
        return True

    if obj.size <= 2:
        return True  # Assume small objects are passable until learned otherwise

    return False


def collides(
    obj1: GridObject,
    obj2: GridObject,
    trajectory1: Optional[Direction] = None,
    trajectory2: Optional[Direction] = None,
    steps: int = 1,
) -> bool:
    """
    Predict if two objects will collide given their trajectories.

    Args:
        obj1: First object
        obj2: Second object
        trajectory1: Movement direction of obj1 (None = stationary)
        trajectory2: Movement direction of obj2 (None = stationary)
        steps: Number of steps to simulate

    Returns:
        True if objects will collide within steps
    """
    # Current positions
    pos1 = obj1.position
    pos2 = obj2.position

    dx1 = trajectory1.dx if trajectory1 else 0
    dy1 = trajectory1.dy if trajectory1 else 0
    dx2 = trajectory2.dx if trajectory2 else 0
    dy2 = trajectory2.dy if trajectory2 else 0

    for step in range(1, steps + 1):
        # Predict positions
        future1 = Position(pos1.x + dx1 * step, pos1.y + dy1 * step)
        future2 = Position(pos2.x + dx2 * step, pos2.y + dy2 * step)

        # Check collision (simplified: center proximity)
        if future1.manhattan_distance(future2) < 2:
            return True

    return False


def pushes(
    agent: GridObject,
    obj: GridObject,
    direction: Direction,
    learned_pushable: Optional[Set[int]] = None,
) -> bool:
    """
    Check if agent's action in direction would push the object.

    Pushing typically requires:
    1. Agent adjacent to object
    2. Agent moving toward object
    3. Object is pushable (learned property)

    Args:
        agent: The agent performing the action
        obj: The potential target of pushing
        direction: Direction of agent's movement
        learned_pushable: Set of object colors known to be pushable

    Returns:
        True if this would push the object
    """
    learned_pushable = learned_pushable or set()

    # Check if object is pushable
    if obj.color not in learned_pushable:
        return False

    # Check adjacency in the direction of movement
    agent_pos = agent.position
    next_pos = Position(agent_pos.x + direction.dx, agent_pos.y + direction.dy)

    # Agent must be moving toward the object
    if next_pos.to_tuple() not in obj.pixels:
        return False

    return True


def gravity_applies(
    obj: GridObject,
    learned_floating: Optional[Set[int]] = None,
) -> bool:
    """
    Determine if gravity affects an object.

    This is typically learned - some objects float, others fall.

    Args:
        obj: Object to check
        learned_floating: Set of colors that don't fall

    Returns:
        True if gravity affects this object
    """
    learned_floating = learned_floating or set()

    if obj.color in learned_floating:
        return False

    # Heuristics:
    # - Background color (0) doesn't fall
    # - Very large objects (walls, floors) don't fall
    if obj.color == 0:
        return False

    if obj.size > 20:
        return False  # Assume large objects are fixed

    return True
