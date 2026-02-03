"""
Spatial Relations Primitives (9 total)

Core knowledge prior: Understanding how objects relate to each other in space.

These primitives handle:
- Adjacency and proximity
- Directional relationships
- Containment and overlap
- Path finding
"""

from __future__ import annotations

from collections import deque
from typing import List, Optional, Set, Tuple

from .geometry import distance
from .objectness import GridObject, Position


def adjacent_to(obj1: GridObject, obj2: GridObject) -> bool:
    """
    Check if two objects are adjacent (sharing edge or corner).

    Args:
        obj1: First object
        obj2: Second object

    Returns:
        True if objects share at least one adjacent cell
    """
    # Check all 8 directions (including diagonals)
    for px, py in obj1.pixels:
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if (px + dx, py + dy) in obj2.pixels:
                    return True
    return False


def above(obj1: GridObject, obj2: GridObject) -> bool:
    """
    Check if obj1 is above obj2.

    obj1 is above obj2 if obj1's bottom edge is above or at obj2's top edge,
    and they overlap horizontally.

    Args:
        obj1: First object (potential "above")
        obj2: Second object (potential "below")

    Returns:
        True if obj1 is above obj2
    """
    # obj1's max y should be <= obj2's min y (remember y increases downward)
    if obj1.bounding_box.y_max > obj2.bounding_box.y_min:
        return False

    # Check horizontal overlap
    x_overlap = (
        obj1.bounding_box.x_max >= obj2.bounding_box.x_min
        and obj1.bounding_box.x_min <= obj2.bounding_box.x_max
    )

    return x_overlap


def below(obj1: GridObject, obj2: GridObject) -> bool:
    """
    Check if obj1 is below obj2.

    Args:
        obj1: First object (potential "below")
        obj2: Second object (potential "above")

    Returns:
        True if obj1 is below obj2
    """
    return above(obj2, obj1)


def left_of(obj1: GridObject, obj2: GridObject) -> bool:
    """
    Check if obj1 is to the left of obj2.

    obj1 is left of obj2 if obj1's right edge is left of or at obj2's left edge,
    and they overlap vertically.

    Args:
        obj1: First object (potential "left")
        obj2: Second object (potential "right")

    Returns:
        True if obj1 is left of obj2
    """
    if obj1.bounding_box.x_max > obj2.bounding_box.x_min:
        return False

    # Check vertical overlap
    y_overlap = (
        obj1.bounding_box.y_max >= obj2.bounding_box.y_min
        and obj1.bounding_box.y_min <= obj2.bounding_box.y_max
    )

    return y_overlap


def right_of(obj1: GridObject, obj2: GridObject) -> bool:
    """
    Check if obj1 is to the right of obj2.

    Args:
        obj1: First object (potential "right")
        obj2: Second object (potential "left")

    Returns:
        True if obj1 is right of obj2
    """
    return left_of(obj2, obj1)


def inside(obj1: GridObject, obj2: GridObject) -> bool:
    """
    Check if obj1 is completely inside obj2.

    Args:
        obj1: First object (potential inner)
        obj2: Second object (potential container)

    Returns:
        True if all of obj1's pixels are within obj2's bounding box
    """
    bbox = obj2.bounding_box

    for px, py in obj1.pixels:
        if not (bbox.x_min <= px <= bbox.x_max and bbox.y_min <= py <= bbox.y_max):
            return False

    return True


def overlaps(obj1: GridObject, obj2: GridObject) -> bool:
    """
    Check if two objects share any pixels.

    Args:
        obj1: First object
        obj2: Second object

    Returns:
        True if objects share at least one pixel
    """
    return bool(obj1.pixels & obj2.pixels)


def nearest(obj: GridObject, objects: List[GridObject]) -> Optional[GridObject]:
    """
    Find the nearest object to a reference.

    Distance is measured from center to center using Manhattan distance.

    Args:
        obj: Reference object
        objects: Candidate objects

    Returns:
        Nearest object, or None if list is empty
    """
    if not objects:
        return None

    obj_pos = obj.position

    nearest_obj = None
    min_dist = float("inf")

    for candidate in objects:
        if candidate.object_id == obj.object_id:
            continue

        dist = distance(obj_pos, candidate.position)
        if dist < min_dist:
            min_dist = dist
            nearest_obj = candidate

    return nearest_obj


def path_exists(
    from_pos: Position,
    to_pos: Position,
    obstacles: Set[Tuple[int, int]],
    grid_size: Tuple[int, int] = (64, 64),
) -> bool:
    """
    Check if an unobstructed path exists between two positions.

    Uses BFS to find a path avoiding obstacles.

    Args:
        from_pos: Starting position
        to_pos: Target position
        obstacles: Set of blocked pixel coordinates
        grid_size: (width, height) of the grid

    Returns:
        True if a path exists
    """
    width, height = grid_size

    # Quick check if start or end is blocked
    if from_pos.to_tuple() in obstacles or to_pos.to_tuple() in obstacles:
        return False

    # BFS
    visited: Set[Tuple[int, int]] = set()
    queue = deque([from_pos.to_tuple()])
    visited.add(from_pos.to_tuple())

    target = to_pos.to_tuple()

    while queue:
        x, y = queue.popleft()

        if (x, y) == target:
            return True

        # 4-directional movement
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy

            # Bounds check
            if not (0 <= nx < width and 0 <= ny < height):
                continue

            # Obstacle check
            if (nx, ny) in obstacles:
                continue

            # Visited check
            if (nx, ny) in visited:
                continue

            visited.add((nx, ny))
            queue.append((nx, ny))

    return False
