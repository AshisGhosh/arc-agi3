"""
DSL Primitives organized by ARC Core Knowledge Priors

Total: 57 primitives

Categories:
- objectness: 12 primitives for object detection, tracking, classification
- counting: 8 primitives for numbers and quantitative reasoning
- geometry: 14 primitives for shapes, positions, transformations
- relations: 9 primitives for spatial relationships
- goals: 6 primitives for goal-directed behavior
- physics: 8 primitives for elementary physics concepts
- control: 10 primitives for control flow and temporal reasoning
"""

from .control import (
    after,
    for_each,
    if_then,
    if_then_else,
    parallel,
    repeat,
    seq,
    until,
    wait,
    while_do,
)
from .counting import (
    compare_counts,
    count,
    count_by_color,
    count_by_type,
    enumerate_positions,
    max_min,
    nth_object,
    sum_property,
)
from .geometry import (
    area,
    bounding_box,
    center_of_mass,
    detect_shape,
    direction_to,
    distance,
    has_symmetry,
    is_l_shape,
    is_line,
    is_rectangle,
    is_square,
    perimeter,
    reflect,
    rotate,
    scale,
)
from .goals import (
    approach_until,
    avoid,
    collect,
    move_away,
    move_toward,
    reach,
)
from .objectness import (
    classify_object,
    detect_objects,
    find_similar,
    get_object_at,
    group_by_color,
    group_by_shape,
    identify_agent,
    object_appeared,
    object_changed,
    object_disappeared,
    object_exists,
    track_object,
)
from .physics import (
    blocks_movement,
    can_pass_through,
    collides,
    contains,
    gravity_applies,
    occludes,
    pushes,
    supports,
)
from .relations import (
    above,
    adjacent_to,
    below,
    inside,
    left_of,
    nearest,
    overlaps,
    path_exists,
    right_of,
)

# Registry of all primitives for introspection
PRIMITIVES = {
    "objectness": [
        detect_objects,
        identify_agent,
        track_object,
        object_exists,
        object_appeared,
        object_disappeared,
        object_changed,
        group_by_color,
        group_by_shape,
        find_similar,
        classify_object,
        get_object_at,
    ],
    "counting": [
        count,
        count_by_color,
        count_by_type,
        enumerate_positions,
        compare_counts,
        nth_object,
        sum_property,
        max_min,
    ],
    "geometry": [
        is_line,
        is_rectangle,
        is_square,
        is_l_shape,
        detect_shape,
        bounding_box,
        center_of_mass,
        area,
        perimeter,
        has_symmetry,
        rotate,
        reflect,
        scale,
        distance,
        direction_to,
    ],
    "relations": [
        adjacent_to,
        above,
        below,
        left_of,
        right_of,
        inside,
        overlaps,
        nearest,
        path_exists,
    ],
    "goals": [
        move_toward,
        move_away,
        reach,
        collect,
        avoid,
        approach_until,
    ],
    "physics": [
        contains,
        supports,
        occludes,
        blocks_movement,
        can_pass_through,
        collides,
        pushes,
        gravity_applies,
    ],
    "control": [
        seq,
        parallel,
        if_then,
        if_then_else,
        while_do,
        repeat,
        until,
        for_each,
        wait,
        after,
    ],
}

PRIMITIVE_COUNT = sum(len(prims) for prims in PRIMITIVES.values())
assert PRIMITIVE_COUNT >= 57, f"Expected 57+ primitives, got {PRIMITIVE_COUNT}"

__all__ = [
    # Objectness
    "detect_objects",
    "identify_agent",
    "track_object",
    "object_exists",
    "object_appeared",
    "object_disappeared",
    "object_changed",
    "group_by_color",
    "group_by_shape",
    "find_similar",
    "classify_object",
    "get_object_at",
    # Counting
    "count",
    "count_by_color",
    "count_by_type",
    "enumerate_positions",
    "compare_counts",
    "nth_object",
    "sum_property",
    "max_min",
    # Geometry
    "is_line",
    "is_rectangle",
    "is_square",
    "is_l_shape",
    "detect_shape",
    "bounding_box",
    "center_of_mass",
    "area",
    "perimeter",
    "has_symmetry",
    "rotate",
    "reflect",
    "scale",
    "distance",
    "direction_to",
    # Relations
    "adjacent_to",
    "above",
    "below",
    "left_of",
    "right_of",
    "inside",
    "overlaps",
    "nearest",
    "path_exists",
    # Goals
    "move_toward",
    "move_away",
    "reach",
    "collect",
    "avoid",
    "approach_until",
    # Physics
    "contains",
    "supports",
    "occludes",
    "blocks_movement",
    "can_pass_through",
    "collides",
    "pushes",
    "gravity_applies",
    # Control
    "seq",
    "parallel",
    "if_then",
    "if_then_else",
    "while_do",
    "repeat",
    "until",
    "for_each",
    "wait",
    "after",
    # Registry
    "PRIMITIVES",
    "PRIMITIVE_COUNT",
]
