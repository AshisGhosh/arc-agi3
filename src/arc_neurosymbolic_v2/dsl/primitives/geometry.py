"""
Basic Geometry Primitives (14 total)

Core knowledge prior: Understanding of lines, shapes, symmetry, and spatial transformations.

These primitives handle:
- Shape detection and classification
- Spatial measurements
- Symmetry analysis
- Geometric transformations
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Set, Tuple

import numpy as np

from .objectness import BoundingBox, GridObject, Position, ShapeType


class Axis(Enum):
    """Axis for symmetry and reflection operations."""

    HORIZONTAL = auto()  # Left-right symmetry
    VERTICAL = auto()  # Top-bottom symmetry
    DIAGONAL_MAIN = auto()  # Top-left to bottom-right
    DIAGONAL_ANTI = auto()  # Top-right to bottom-left


class Direction(Enum):
    """Cardinal and ordinal directions."""

    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)
    UP_LEFT = (-1, -1)
    UP_RIGHT = (1, -1)
    DOWN_LEFT = (-1, 1)
    DOWN_RIGHT = (1, 1)
    NONE = (0, 0)

    @property
    def dx(self) -> int:
        return self.value[0]

    @property
    def dy(self) -> int:
        return self.value[1]


# =============================================================================
# SHAPE DETECTION
# =============================================================================


def is_line(obj: GridObject) -> bool:
    """
    Check if object is a line (horizontal, vertical, or diagonal).

    A line has all pixels along a single axis or diagonal.

    Args:
        obj: Object to check

    Returns:
        True if object is a line
    """
    pixels = list(obj.pixels)
    if len(pixels) < 2:
        return False

    xs = [p[0] for p in pixels]
    ys = [p[1] for p in pixels]

    # Horizontal line: all same y
    if len(set(ys)) == 1:
        return True

    # Vertical line: all same x
    if len(set(xs)) == 1:
        return True

    # Diagonal: check if all points on same diagonal
    # y = x + c or y = -x + c
    if len(pixels) >= 2:
        # Check main diagonal (y - x constant)
        diffs = [p[1] - p[0] for p in pixels]
        if len(set(diffs)) == 1:
            return True

        # Check anti-diagonal (y + x constant)
        sums = [p[1] + p[0] for p in pixels]
        if len(set(sums)) == 1:
            return True

    return False


def is_rectangle(obj: GridObject) -> bool:
    """
    Check if object forms a filled rectangle.

    Args:
        obj: Object to check

    Returns:
        True if object is a filled rectangle
    """
    bbox = obj.bounding_box
    expected_size = bbox.width * bbox.height

    # Check if all pixels fill the bounding box
    if len(obj.pixels) != expected_size:
        return False

    # Verify all bbox pixels are present
    for x in range(bbox.x_min, bbox.x_max + 1):
        for y in range(bbox.y_min, bbox.y_max + 1):
            if (x, y) not in obj.pixels:
                return False

    return True


def is_square(obj: GridObject) -> bool:
    """
    Check if object is a filled square.

    Args:
        obj: Object to check

    Returns:
        True if object is a filled square
    """
    if not is_rectangle(obj):
        return False

    bbox = obj.bounding_box
    return bbox.width == bbox.height


def is_l_shape(obj: GridObject) -> bool:
    """
    Check if object is an L-shape (or rotated L).

    An L-shape has exactly 2 lines meeting at a corner.

    Args:
        obj: Object to check

    Returns:
        True if object is an L-shape
    """
    pixels = set(obj.pixels)

    if len(pixels) < 3:
        return False

    # Find corner candidates (pixels with exactly 2 neighbors)
    for px, py in pixels:
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (px + dx, py + dy) in pixels:
                neighbors.append((px + dx, py + dy))

        if len(neighbors) == 2:
            # This could be the corner - check if it's the only one
            # and the arms extend in perpendicular directions
            n1, n2 = neighbors
            dx1, dy1 = n1[0] - px, n1[1] - py
            dx2, dy2 = n2[0] - px, n2[1] - py

            # Perpendicular check
            if dx1 * dx2 + dy1 * dy2 == 0:  # Dot product = 0
                # Check that arms are straight lines
                arm1 = _trace_line(pixels, (px, py), (dx1, dy1))
                arm2 = _trace_line(pixels, (px, py), (dx2, dy2))

                if len(arm1) + len(arm2) - 1 == len(pixels):
                    return True

    return False


def _trace_line(
    pixels: Set[Tuple[int, int]], start: Tuple[int, int], direction: Tuple[int, int]
) -> Set[Tuple[int, int]]:
    """Trace a line from start in given direction."""
    line = {start}
    x, y = start
    dx, dy = direction

    while True:
        x, y = x + dx, y + dy
        if (x, y) in pixels:
            line.add((x, y))
        else:
            break

    return line


def detect_shape(obj: GridObject) -> ShapeType:
    """
    Classify the shape of an object.

    Args:
        obj: Object to classify

    Returns:
        ShapeType classification
    """
    if is_line(obj):
        # Check if diagonal
        pixels = list(obj.pixels)
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]
        if len(set(xs)) > 1 and len(set(ys)) > 1:
            return ShapeType.DIAGONAL
        return ShapeType.LINE

    if is_square(obj):
        return ShapeType.SQUARE

    if is_rectangle(obj):
        return ShapeType.RECTANGLE

    if is_l_shape(obj):
        return ShapeType.L_SHAPE

    # TODO: Add T_SHAPE and CROSS detection
    return ShapeType.IRREGULAR


# =============================================================================
# SPATIAL MEASUREMENTS
# =============================================================================


def bounding_box(obj: GridObject) -> Tuple[int, int, int, int]:
    """
    Get the bounding box of an object.

    Args:
        obj: Object to measure

    Returns:
        Tuple of (x_min, y_min, x_max, y_max)
    """
    bbox = obj.bounding_box
    return (bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max)


def center_of_mass(obj: GridObject) -> Position:
    """
    Calculate the centroid of an object.

    Args:
        obj: Object to measure

    Returns:
        Position of center of mass
    """
    if not obj.pixels:
        return obj.position

    xs = [p[0] for p in obj.pixels]
    ys = [p[1] for p in obj.pixels]

    return Position(sum(xs) // len(xs), sum(ys) // len(ys))


def area(obj: GridObject) -> int:
    """
    Count the number of pixels in an object.

    Args:
        obj: Object to measure

    Returns:
        Area in pixels
    """
    return len(obj.pixels)


def perimeter(obj: GridObject) -> int:
    """
    Count the edge pixels of an object.

    An edge pixel has at least one non-object neighbor.

    Args:
        obj: Object to measure

    Returns:
        Perimeter length in pixels
    """
    pixels = obj.pixels
    edge_count = 0

    for px, py in pixels:
        # Check 4-neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            if (px + dx, py + dy) not in pixels:
                edge_count += 1
                break

    return edge_count


# =============================================================================
# SYMMETRY
# =============================================================================


def has_symmetry(obj: GridObject, axis: Axis) -> bool:
    """
    Check if object has symmetry along given axis.

    Args:
        obj: Object to check
        axis: Axis of symmetry

    Returns:
        True if object is symmetric along the axis
    """
    pixels = obj.pixels
    bbox = obj.bounding_box

    # Center of symmetry
    cx = (bbox.x_min + bbox.x_max) / 2
    cy = (bbox.y_min + bbox.y_max) / 2

    for px, py in pixels:
        # Calculate reflected point
        if axis == Axis.HORIZONTAL:
            rx, ry = int(2 * cx - px), py
        elif axis == Axis.VERTICAL:
            rx, ry = px, int(2 * cy - py)
        elif axis == Axis.DIAGONAL_MAIN:
            # Reflect across y = x (relative to center)
            rx = int(cx + (py - cy))
            ry = int(cy + (px - cx))
        elif axis == Axis.DIAGONAL_ANTI:
            # Reflect across y = -x (relative to center)
            rx = int(cx - (py - cy))
            ry = int(cy - (px - cx))
        else:
            return False

        # Check if reflected point exists
        if (rx, ry) not in pixels:
            return False

    return True


# =============================================================================
# TRANSFORMATIONS
# =============================================================================


def rotate(obj: GridObject, degrees: int) -> GridObject:
    """
    Rotate object by given degrees (90, 180, 270).

    Args:
        obj: Object to rotate
        degrees: Rotation angle (must be 90, 180, or 270)

    Returns:
        New GridObject with rotated pixels
    """
    assert degrees in (90, 180, 270), "Degrees must be 90, 180, or 270"

    # Use center as rotation point
    cx, cy = center_of_mass(obj).x, center_of_mass(obj).y

    new_pixels: Set[Tuple[int, int]] = set()

    for px, py in obj.pixels:
        # Translate to origin
        x, y = px - cx, py - cy

        # Rotate
        if degrees == 90:
            rx, ry = -y, x
        elif degrees == 180:
            rx, ry = -x, -y
        else:  # 270
            rx, ry = y, -x

        # Translate back
        new_pixels.add((int(rx + cx), int(ry + cy)))

    # Recompute bounding box
    xs = [p[0] for p in new_pixels]
    ys = [p[1] for p in new_pixels]
    new_bbox = BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    return GridObject(
        object_id=obj.object_id,
        color=obj.color,
        pixels=frozenset(new_pixels),
        bounding_box=new_bbox,
        shape=obj.shape,
        object_type=obj.object_type,
    )


def reflect(obj: GridObject, axis: Axis) -> GridObject:
    """
    Reflect object across given axis.

    Args:
        obj: Object to reflect
        axis: Axis of reflection

    Returns:
        New GridObject with reflected pixels
    """
    bbox = obj.bounding_box
    cx = (bbox.x_min + bbox.x_max) / 2
    cy = (bbox.y_min + bbox.y_max) / 2

    new_pixels: Set[Tuple[int, int]] = set()

    for px, py in obj.pixels:
        if axis == Axis.HORIZONTAL:
            rx, ry = int(2 * cx - px), py
        elif axis == Axis.VERTICAL:
            rx, ry = px, int(2 * cy - py)
        elif axis == Axis.DIAGONAL_MAIN:
            rx = int(cx + (py - cy))
            ry = int(cy + (px - cx))
        else:  # DIAGONAL_ANTI
            rx = int(cx - (py - cy))
            ry = int(cy - (px - cx))

        new_pixels.add((rx, ry))

    # Recompute bounding box
    xs = [p[0] for p in new_pixels]
    ys = [p[1] for p in new_pixels]
    new_bbox = BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    return GridObject(
        object_id=obj.object_id,
        color=obj.color,
        pixels=frozenset(new_pixels),
        bounding_box=new_bbox,
        shape=obj.shape,
        object_type=obj.object_type,
    )


def scale(obj: GridObject, factor: float) -> GridObject:
    """
    Scale object by given factor.

    Note: Scaling by non-integer factors may lose pixels.

    Args:
        obj: Object to scale
        factor: Scale factor (e.g., 2.0 doubles size)

    Returns:
        New GridObject with scaled pixels
    """
    cx, cy = center_of_mass(obj).x, center_of_mass(obj).y

    new_pixels: Set[Tuple[int, int]] = set()

    for px, py in obj.pixels:
        # Scale from center
        sx = int(cx + (px - cx) * factor)
        sy = int(cy + (py - cy) * factor)
        new_pixels.add((sx, sy))

        # For upscaling, fill in gaps
        if factor > 1:
            for dx in range(int(factor)):
                for dy in range(int(factor)):
                    new_pixels.add((sx + dx, sy + dy))

    # Recompute bounding box
    xs = [p[0] for p in new_pixels]
    ys = [p[1] for p in new_pixels]
    new_bbox = BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

    return GridObject(
        object_id=obj.object_id,
        color=obj.color,
        pixels=frozenset(new_pixels),
        bounding_box=new_bbox,
        shape=obj.shape,
        object_type=obj.object_type,
    )


# =============================================================================
# DISTANCE AND DIRECTION
# =============================================================================


def distance(pos1: Position, pos2: Position, metric: str = "manhattan") -> int:
    """
    Calculate distance between two positions.

    Args:
        pos1: First position
        pos2: Second position
        metric: Distance metric ("manhattan" or "chebyshev")

    Returns:
        Distance value
    """
    dx = abs(pos1.x - pos2.x)
    dy = abs(pos1.y - pos2.y)

    if metric == "manhattan":
        return dx + dy
    elif metric == "chebyshev":
        return max(dx, dy)
    elif metric == "euclidean":
        return int(np.sqrt(dx * dx + dy * dy))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def direction_to(from_pos: Position, to_pos: Position) -> Direction:
    """
    Get the primary direction from one position to another.

    Args:
        from_pos: Starting position
        to_pos: Target position

    Returns:
        Direction towards target
    """
    dx = to_pos.x - from_pos.x
    dy = to_pos.y - from_pos.y

    if dx == 0 and dy == 0:
        return Direction.NONE

    # Determine primary direction based on which component is larger
    if abs(dx) > abs(dy):
        return Direction.RIGHT if dx > 0 else Direction.LEFT
    elif abs(dy) > abs(dx):
        return Direction.DOWN if dy > 0 else Direction.UP
    else:
        # Equal - use diagonal
        if dx > 0 and dy > 0:
            return Direction.DOWN_RIGHT
        elif dx > 0 and dy < 0:
            return Direction.UP_RIGHT
        elif dx < 0 and dy > 0:
            return Direction.DOWN_LEFT
        else:
            return Direction.UP_LEFT
