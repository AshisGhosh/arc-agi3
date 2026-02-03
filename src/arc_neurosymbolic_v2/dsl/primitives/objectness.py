"""
Objectness Primitives (12 total)

Core knowledge prior: The world contains discrete, persistent objects.

These primitives handle:
- Object detection and segmentation
- Object tracking across frames
- Object classification
- Change detection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ...perception.symbolic_state import SymbolicState


class ObjectType(Enum):
    """Classification of objects by their role in the environment."""

    AGENT = auto()  # The controllable entity
    OBSTACLE = auto()  # Blocks movement
    COLLECTIBLE = auto()  # Can be picked up
    TRIGGER = auto()  # Activates something when interacted
    GOAL = auto()  # Target destination
    DECORATION = auto()  # Non-interactive
    UNKNOWN = auto()  # Unclassified


class ShapeType(Enum):
    """Basic shape classifications."""

    LINE = auto()
    RECTANGLE = auto()
    SQUARE = auto()
    L_SHAPE = auto()
    T_SHAPE = auto()
    CROSS = auto()
    DIAGONAL = auto()
    IRREGULAR = auto()


@dataclass(frozen=True, slots=True)
class Position:
    """2D grid position."""

    x: int
    y: int

    def __add__(self, other: Position) -> Position:
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Position) -> Position:
        return Position(self.x - other.x, self.y - other.y)

    def manhattan_distance(self, other: Position) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min + 1

    @property
    def height(self) -> int:
        return self.y_max - self.y_min + 1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Position:
        return Position((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)


@dataclass
class GridObject:
    """
    A detected object in the grid.

    Represents a contiguous region of same-colored cells.
    """

    object_id: int
    color: int
    pixels: FrozenSet[Tuple[int, int]]
    bounding_box: BoundingBox
    shape: ShapeType = ShapeType.IRREGULAR
    object_type: ObjectType = ObjectType.UNKNOWN

    @property
    def position(self) -> Position:
        """Primary position (center of bounding box)."""
        return self.bounding_box.center

    @property
    def size(self) -> int:
        """Number of pixels in object."""
        return len(self.pixels)

    def contains_point(self, pos: Position) -> bool:
        """Check if point is within this object."""
        return pos.to_tuple() in self.pixels


@dataclass
class ChangeSet:
    """Describes how an object changed between frames."""

    object_id: int
    position_delta: Optional[Position] = None
    color_changed: Optional[Tuple[int, int]] = None  # (old, new)
    size_delta: int = 0
    pixels_added: FrozenSet[Tuple[int, int]] = field(default_factory=lambda: frozenset())
    pixels_removed: FrozenSet[Tuple[int, int]] = field(default_factory=lambda: frozenset())

    @property
    def moved(self) -> bool:
        return self.position_delta is not None and (
            self.position_delta.x != 0 or self.position_delta.y != 0
        )

    @property
    def resized(self) -> bool:
        return self.size_delta != 0

    @property
    def recolored(self) -> bool:
        return self.color_changed is not None


# =============================================================================
# PRIMITIVE FUNCTIONS
# =============================================================================


def detect_objects(grid: NDArray[np.int_]) -> List[GridObject]:
    """
    Segment grid into discrete objects using connected components.

    Each object is a contiguous region of same-colored non-background cells.
    Uses 4-connectivity (not diagonal).

    Args:
        grid: 2D numpy array with cell colors (0-9)

    Returns:
        List of detected GridObject instances
    """
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    objects: List[GridObject] = []
    object_id = 0

    # Background color (typically 0)
    background = 0

    def flood_fill(start_y: int, start_x: int, color: int) -> Set[Tuple[int, int]]:
        """BFS flood fill to find connected component."""
        pixels: Set[Tuple[int, int]] = set()
        stack = [(start_y, start_x)]

        while stack:
            y, x = stack.pop()

            if y < 0 or y >= height or x < 0 or x >= width or visited[y, x] or grid[y, x] != color:
                continue

            visited[y, x] = True
            pixels.add((x, y))

            # 4-connectivity
            stack.extend([(y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)])

        return pixels

    for y in range(height):
        for x in range(width):
            if not visited[y, x] and grid[y, x] != background:
                color = int(grid[y, x])
                pixels = flood_fill(y, x, color)

                if pixels:
                    # Compute bounding box
                    xs = [p[0] for p in pixels]
                    ys = [p[1] for p in pixels]
                    bbox = BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))

                    obj = GridObject(
                        object_id=object_id,
                        color=color,
                        pixels=frozenset(pixels),
                        bounding_box=bbox,
                    )
                    objects.append(obj)
                    object_id += 1

    return objects


def identify_agent(grid: NDArray[np.int_]) -> Optional[GridObject]:
    """
    Find the controllable entity (agent) in the grid.

    Heuristics:
    1. Unique color (appears only once)
    2. Small size (1-4 pixels typically)
    3. Often in a corner or edge initially

    Args:
        grid: 2D numpy array with cell colors

    Returns:
        The identified agent object, or None if not found
    """
    objects = detect_objects(grid)

    if not objects:
        return None

    # Count objects by color
    color_counts: Dict[int, List[GridObject]] = {}
    for obj in objects:
        if obj.color not in color_counts:
            color_counts[obj.color] = []
        color_counts[obj.color].append(obj)

    # Find unique-color objects that are small
    candidates: List[GridObject] = []
    for color, objs in color_counts.items():
        if len(objs) == 1 and objs[0].size <= 4:
            candidates.append(objs[0])

    if not candidates:
        # Fallback: smallest unique object
        for color, objs in color_counts.items():
            if len(objs) == 1:
                candidates.append(objs[0])

    if candidates:
        # Return smallest candidate
        return min(candidates, key=lambda o: o.size)

    return None


def track_object(
    obj_id: int,
    prev_objects: List[GridObject],
    curr_objects: List[GridObject],
) -> Optional[GridObject]:
    """
    Track an object across frames by matching features.

    Matching criteria (in order of priority):
    1. Same color and similar position
    2. Same color and similar size
    3. Maximum pixel overlap

    Args:
        obj_id: ID of object to track from previous frame
        prev_objects: Objects from previous frame
        curr_objects: Objects from current frame

    Returns:
        Matched object in current frame, or None if lost
    """
    # Find the object in previous frame
    prev_obj = None
    for obj in prev_objects:
        if obj.object_id == obj_id:
            prev_obj = obj
            break

    if prev_obj is None:
        return None

    # Find best match in current frame
    best_match: Optional[GridObject] = None
    best_score = float("-inf")

    for curr_obj in curr_objects:
        score = 0.0

        # Same color is essential
        if curr_obj.color != prev_obj.color:
            continue

        # Position similarity (closer is better)
        dist = prev_obj.position.manhattan_distance(curr_obj.position)
        score += 100.0 / (1.0 + dist)

        # Size similarity
        size_diff = abs(curr_obj.size - prev_obj.size)
        score += 50.0 / (1.0 + size_diff)

        # Pixel overlap
        overlap = len(prev_obj.pixels & curr_obj.pixels)
        score += overlap * 10.0

        if score > best_score:
            best_score = score
            best_match = curr_obj

    return best_match


def object_exists(obj_id: int, objects: List[GridObject]) -> bool:
    """
    Check if an object with given ID exists.

    Args:
        obj_id: Object ID to check
        objects: List of objects from current frame

    Returns:
        True if object exists, False otherwise
    """
    return any(obj.object_id == obj_id for obj in objects)


def object_appeared(
    prev_objects: List[GridObject], curr_objects: List[GridObject]
) -> List[GridObject]:
    """
    Detect objects that appeared in current frame.

    An object is considered "new" if no object in the previous frame
    has significant overlap with it.

    Args:
        prev_objects: Objects from previous frame
        curr_objects: Objects from current frame

    Returns:
        List of newly appeared objects
    """
    appeared: List[GridObject] = []
    prev_pixels = set()
    for obj in prev_objects:
        prev_pixels.update(obj.pixels)

    for curr_obj in curr_objects:
        overlap = len(curr_obj.pixels & prev_pixels)
        overlap_ratio = overlap / max(len(curr_obj.pixels), 1)

        # Consider "new" if <50% overlap with previous state
        if overlap_ratio < 0.5:
            appeared.append(curr_obj)

    return appeared


def object_disappeared(
    prev_objects: List[GridObject], curr_objects: List[GridObject]
) -> List[GridObject]:
    """
    Detect objects that disappeared from current frame.

    An object is considered "gone" if no object in the current frame
    has significant overlap with it.

    Args:
        prev_objects: Objects from previous frame
        curr_objects: Objects from current frame

    Returns:
        List of disappeared objects (from previous frame)
    """
    disappeared: List[GridObject] = []
    curr_pixels = set()
    for obj in curr_objects:
        curr_pixels.update(obj.pixels)

    for prev_obj in prev_objects:
        overlap = len(prev_obj.pixels & curr_pixels)
        overlap_ratio = overlap / max(len(prev_obj.pixels), 1)

        # Consider "gone" if <50% overlap with current state
        if overlap_ratio < 0.5:
            disappeared.append(prev_obj)

    return disappeared


def object_changed(
    obj_id: int,
    prev_objects: List[GridObject],
    curr_objects: List[GridObject],
) -> Optional[ChangeSet]:
    """
    Describe how an object changed between frames.

    Args:
        obj_id: ID of object to track
        prev_objects: Objects from previous frame
        curr_objects: Objects from current frame

    Returns:
        ChangeSet describing the changes, or None if object not found
    """
    # Find objects
    prev_obj = next((o for o in prev_objects if o.object_id == obj_id), None)
    curr_obj = track_object(obj_id, prev_objects, curr_objects)

    if prev_obj is None:
        return None

    if curr_obj is None:
        # Object disappeared
        return ChangeSet(
            object_id=obj_id,
            pixels_removed=prev_obj.pixels,
            size_delta=-prev_obj.size,
        )

    # Compute changes
    pos_delta = None
    if prev_obj.position != curr_obj.position:
        pos_delta = curr_obj.position - prev_obj.position

    color_changed = None
    if prev_obj.color != curr_obj.color:
        color_changed = (prev_obj.color, curr_obj.color)

    pixels_added = curr_obj.pixels - prev_obj.pixels
    pixels_removed = prev_obj.pixels - curr_obj.pixels

    return ChangeSet(
        object_id=obj_id,
        position_delta=pos_delta,
        color_changed=color_changed,
        size_delta=curr_obj.size - prev_obj.size,
        pixels_added=frozenset(pixels_added),
        pixels_removed=frozenset(pixels_removed),
    )


def group_by_color(objects: List[GridObject]) -> Dict[int, List[GridObject]]:
    """
    Group objects by their color.

    Args:
        objects: List of objects to group

    Returns:
        Dict mapping color to list of objects with that color
    """
    groups: Dict[int, List[GridObject]] = {}
    for obj in objects:
        if obj.color not in groups:
            groups[obj.color] = []
        groups[obj.color].append(obj)
    return groups


def group_by_shape(objects: List[GridObject]) -> Dict[ShapeType, List[GridObject]]:
    """
    Group objects by their shape classification.

    Args:
        objects: List of objects to group

    Returns:
        Dict mapping shape type to list of objects with that shape
    """
    groups: Dict[ShapeType, List[GridObject]] = {}
    for obj in objects:
        if obj.shape not in groups:
            groups[obj.shape] = []
        groups[obj.shape].append(obj)
    return groups


def find_similar(
    reference: GridObject, objects: List[GridObject], threshold: float = 0.7
) -> List[GridObject]:
    """
    Find objects similar to a reference object.

    Similarity is based on:
    - Same color (required)
    - Similar size (within 50%)
    - Similar shape

    Args:
        reference: Object to compare against
        objects: Candidate objects
        threshold: Minimum similarity score (0-1)

    Returns:
        List of similar objects (excluding reference)
    """
    similar: List[GridObject] = []

    for obj in objects:
        if obj.object_id == reference.object_id:
            continue

        score = 0.0

        # Color match (required)
        if obj.color != reference.color:
            continue
        score += 0.4

        # Size similarity
        size_ratio = min(obj.size, reference.size) / max(obj.size, reference.size)
        score += 0.3 * size_ratio

        # Shape match
        if obj.shape == reference.shape:
            score += 0.3

        if score >= threshold:
            similar.append(obj)

    return similar


def classify_object(obj: GridObject, context: Optional[SymbolicState] = None) -> ObjectType:
    """
    Classify an object by its likely role.

    Uses heuristics and context:
    - Single pixel or small + unique color = AGENT
    - Large rectangular outline = OBSTACLE
    - Small, multiple similar = COLLECTIBLE
    - Single, prominent = GOAL

    Args:
        obj: Object to classify
        context: Optional symbolic state for context

    Returns:
        Classified ObjectType
    """
    # Small unique objects are likely agents
    if obj.size <= 4:
        return ObjectType.AGENT

    # Large hollow rectangles are obstacles
    if obj.shape == ShapeType.RECTANGLE:
        bbox_area = obj.bounding_box.area
        fill_ratio = obj.size / max(bbox_area, 1)
        if fill_ratio < 0.5:  # Hollow
            return ObjectType.OBSTACLE

    # Default to unknown
    return ObjectType.UNKNOWN


def get_object_at(position: Position, objects: List[GridObject]) -> Optional[GridObject]:
    """
    Get the object at a specific position.

    Args:
        position: Grid position to check
        objects: List of objects

    Returns:
        Object at position, or None if no object there
    """
    pos_tuple = position.to_tuple()
    for obj in objects:
        if pos_tuple in obj.pixels:
            return obj
    return None
