"""
Numbers and Counting Primitives (8 total)

Core knowledge prior: Basic numeracy and quantitative reasoning.

These primitives handle:
- Counting objects
- Comparing quantities
- Enumerating and ordering
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from .objectness import GridObject, ObjectType, Position

T = TypeVar("T")


class Comparison(Enum):
    """Result of comparing two quantities."""

    LESS_THAN = auto()
    EQUAL = auto()
    GREATER_THAN = auto()


def count(objects: List[GridObject]) -> int:
    """
    Count the number of objects in a collection.

    Args:
        objects: List of objects to count

    Returns:
        Number of objects
    """
    return len(objects)


def count_by_color(objects: List[GridObject], color: int) -> int:
    """
    Count objects of a specific color.

    Args:
        objects: List of objects
        color: Color value to filter by (0-9)

    Returns:
        Number of objects with the specified color
    """
    return sum(1 for obj in objects if obj.color == color)


def count_by_type(objects: List[GridObject], obj_type: ObjectType) -> int:
    """
    Count objects of a specific type.

    Args:
        objects: List of objects
        obj_type: Object type to filter by

    Returns:
        Number of objects with the specified type
    """
    return sum(1 for obj in objects if obj.object_type == obj_type)


def enumerate_positions(objects: List[GridObject]) -> List[Position]:
    """
    Get an ordered list of object positions.

    Objects are ordered by their center position (top-to-bottom, left-to-right).

    Args:
        objects: List of objects

    Returns:
        List of positions, sorted
    """
    positions = [obj.position for obj in objects]
    # Sort by y first, then x (top-to-bottom, left-to-right)
    positions.sort(key=lambda p: (p.y, p.x))
    return positions


def compare_counts(count1: int, count2: int) -> Comparison:
    """
    Compare two quantities.

    Args:
        count1: First count
        count2: Second count

    Returns:
        Comparison result (LESS_THAN, EQUAL, or GREATER_THAN)
    """
    if count1 < count2:
        return Comparison.LESS_THAN
    elif count1 > count2:
        return Comparison.GREATER_THAN
    else:
        return Comparison.EQUAL


def nth_object(
    objects: List[GridObject],
    n: int,
    key: Optional[Callable[[GridObject], Any]] = None,
) -> Optional[GridObject]:
    """
    Get the nth object from a collection.

    Args:
        objects: List of objects
        n: Index (0-based)
        key: Optional sorting key function

    Returns:
        The nth object, or None if n is out of bounds
    """
    if not objects:
        return None

    sorted_objects = objects
    if key is not None:
        sorted_objects = sorted(objects, key=key)

    if 0 <= n < len(sorted_objects):
        return sorted_objects[n]
    return None


def sum_property(
    objects: List[GridObject],
    property_getter: Callable[[GridObject], int],
) -> int:
    """
    Sum a numeric property across all objects.

    Args:
        objects: List of objects
        property_getter: Function to extract numeric property

    Returns:
        Sum of the property across all objects
    """
    return sum(property_getter(obj) for obj in objects)


def max_min(
    objects: List[GridObject],
    property_getter: Callable[[GridObject], Any],
) -> Tuple[Optional[GridObject], Optional[GridObject]]:
    """
    Find objects with maximum and minimum values of a property.

    Args:
        objects: List of objects
        property_getter: Function to extract property value

    Returns:
        Tuple of (max_object, min_object), either may be None if list is empty
    """
    if not objects:
        return (None, None)

    max_obj = max(objects, key=property_getter)
    min_obj = min(objects, key=property_getter)

    return (max_obj, min_obj)
