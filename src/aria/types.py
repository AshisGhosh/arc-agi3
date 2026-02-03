"""
ARIA Type Definitions

Core data structures used across the ARIA architecture.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, List, Optional, Set, Tuple

import numpy as np
import torch

# =============================================================================
# Actions
# =============================================================================


class ActionType(Enum):
    """Action types available in ARC-AGI-3."""

    ACTION1 = 0  # Usually UP
    ACTION2 = 1  # Usually DOWN
    ACTION3 = 2  # Usually LEFT
    ACTION4 = 3  # Usually RIGHT
    ACTION5 = 4  # Context-dependent
    ACTION6 = 5  # Context-dependent (often click)
    ACTION7 = 6  # Context-dependent
    RESET = 7

    def requires_coords(self) -> bool:
        """Check if action requires x, y coordinates."""
        return self in {ActionType.ACTION6}


@dataclass
class Action:
    """Internal action representation."""

    type: ActionType
    x: int = 0
    y: int = 0
    reasoning: Optional[str] = None

    @classmethod
    def from_type(cls, action_type: ActionType, x: int = 0, y: int = 0) -> "Action":
        return cls(type=action_type, x=x, y=y)

    @classmethod
    def UP(cls) -> "Action":
        return cls(type=ActionType.ACTION1)

    @classmethod
    def DOWN(cls) -> "Action":
        return cls(type=ActionType.ACTION2)

    @classmethod
    def LEFT(cls) -> "Action":
        return cls(type=ActionType.ACTION3)

    @classmethod
    def RIGHT(cls) -> "Action":
        return cls(type=ActionType.ACTION4)

    @classmethod
    def RESET(cls) -> "Action":
        return cls(type=ActionType.RESET)

    def with_target(self, target_id: int) -> "Action":
        """Create copy with target annotation."""
        return Action(type=self.type, x=self.x, y=self.y, reasoning=f"target:{target_id}")


# =============================================================================
# Symbolic State
# =============================================================================


@dataclass
class GridObject:
    """A detected object in the grid."""

    id: int
    color: int
    pixels: Set[Tuple[int, int]]
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    centroid: Tuple[float, float]
    shape_signature: str  # 'rectangle', 'line_h', 'line_v', 'L', 'custom'
    area: int
    is_agent: bool = False

    def __hash__(self) -> int:
        return self.id

    def contains(self, x: int, y: int) -> bool:
        """Check if point is inside object."""
        return (x, y) in self.pixels

    def distance_to(self, other: "GridObject") -> float:
        """Euclidean distance between centroids."""
        dx = self.centroid[0] - other.centroid[0]
        dy = self.centroid[1] - other.centroid[1]
        return (dx * dx + dy * dy) ** 0.5


@dataclass
class SpatialRelation:
    """Relation between two objects."""

    subject_id: int
    relation: str  # 'left_of', 'right_of', 'above', 'below', 'adjacent', 'inside', 'overlaps'
    object_id: int
    distance: float

    def __hash__(self) -> int:
        return hash((self.subject_id, self.relation, self.object_id))


@dataclass
class SymbolicState:
    """Complete symbolic representation of a frame."""

    objects: List[GridObject]
    relations: List[SpatialRelation]
    agent_pos: Optional[Tuple[int, int]]
    raw_grid: np.ndarray
    frame_index: int = 0

    def to_description(self) -> str:
        """Generate natural language description for LLM."""
        lines = [f"Grid size: {self.raw_grid.shape}"]
        lines.append(f"Objects detected: {len(self.objects)}")

        if self.agent_pos:
            lines.append(f"Agent position: {self.agent_pos}")

        for obj in self.objects[:10]:  # Limit to avoid too long descriptions
            lines.append(
                f"  - Object {obj.id}: color={obj.color}, "
                f"shape={obj.shape_signature}, "
                f"pos=({obj.centroid[0]:.1f}, {obj.centroid[1]:.1f}), "
                f"area={obj.area}"
            )

        if len(self.objects) > 10:
            lines.append(f"  ... and {len(self.objects) - 10} more objects")

        return "\n".join(lines)

    def get_object_by_id(self, obj_id: int) -> Optional[GridObject]:
        """Find object by ID."""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def get_adjacent_objects(self, obj: GridObject) -> List[GridObject]:
        """Get objects adjacent to given object."""
        adjacent = []
        for rel in self.relations:
            if rel.subject_id == obj.id and rel.relation == "adjacent":
                adj_obj = self.get_object_by_id(rel.object_id)
                if adj_obj:
                    adjacent.append(adj_obj)
        return adjacent


# =============================================================================
# Predicates (for rules and goals)
# =============================================================================


class PredicateType(Enum):
    """Types of predicates for symbolic reasoning."""

    TRUE = auto()
    FALSE = auto()
    AGENT_AT = auto()
    OBJECT_AT = auto()
    OBJECT_COLOR = auto()
    ADJACENT = auto()
    PATH_EXISTS = auto()
    LEVEL_COMPLETE = auto()
    STATE_CHANGED = auto()
    CUSTOM = auto()


@dataclass
class Predicate:
    """A logical predicate for rule conditions."""

    type: PredicateType
    args: Tuple[Any, ...] = ()
    custom_fn: Optional[Callable[[SymbolicState], bool]] = None

    # Singleton predicates
    TRUE: "Predicate" = None  # type: ignore
    FALSE: "Predicate" = None  # type: ignore
    LEVEL_COMPLETE: "Predicate" = None  # type: ignore
    STATE_CHANGED: "Predicate" = None  # type: ignore

    def evaluate(self, state: SymbolicState) -> bool:
        """Evaluate predicate on state."""
        if self.type == PredicateType.TRUE:
            return True
        elif self.type == PredicateType.FALSE:
            return False
        elif self.type == PredicateType.AGENT_AT:
            x, y = self.args
            return state.agent_pos == (x, y)
        elif self.type == PredicateType.OBJECT_AT:
            obj_id, x, y = self.args
            obj = state.get_object_by_id(obj_id)
            return obj is not None and obj.contains(x, y)
        elif self.type == PredicateType.ADJACENT:
            obj1_id, obj2_id = self.args
            for rel in state.relations:
                if (
                    rel.subject_id == obj1_id
                    and rel.object_id == obj2_id
                    and rel.relation == "adjacent"
                ):
                    return True
            return False
        elif self.type == PredicateType.CUSTOM and self.custom_fn:
            return self.custom_fn(state)
        return False

    def apply(self, state: SymbolicState) -> SymbolicState:
        """Apply predicate as effect (modify state)."""
        # This is a placeholder - effects would need more complex implementation
        return state

    def __str__(self) -> str:
        if self.type in {PredicateType.TRUE, PredicateType.FALSE}:
            return self.type.name
        return f"{self.type.name}({', '.join(str(a) for a in self.args)})"


# Initialize singleton predicates
Predicate.TRUE = Predicate(PredicateType.TRUE)
Predicate.FALSE = Predicate(PredicateType.FALSE)
Predicate.LEVEL_COMPLETE = Predicate(PredicateType.LEVEL_COMPLETE)
Predicate.STATE_CHANGED = Predicate(PredicateType.STATE_CHANGED)


# =============================================================================
# Goals
# =============================================================================


@dataclass
class Goal:
    """Explicit goal representation for planning."""

    description: str
    preconditions: List[Predicate]
    target_state_repr: Optional[torch.Tensor] = None
    priority: float = 1.0

    # Special goals (initialized after class definition)
    EXPLORE: "Goal" = None  # type: ignore
    COMPLETE_LEVEL: "Goal" = None  # type: ignore

    def is_satisfied(self, state: SymbolicState) -> bool:
        """Check if goal is achieved."""
        return all(pred.evaluate(state) for pred in self.preconditions)

    def count_unsatisfied(self, state: SymbolicState) -> int:
        """Count remaining conditions (for heuristic)."""
        return sum(1 for pred in self.preconditions if not pred.evaluate(state))


# Initialize singleton goals
Goal.EXPLORE = Goal(
    description="Explore to discover mechanics",
    preconditions=[],
    target_state_repr=None,
    priority=0.5,
)

Goal.COMPLETE_LEVEL = Goal(
    description="Complete the current level",
    preconditions=[Predicate.LEVEL_COMPLETE],
    target_state_repr=None,
    priority=1.0,
)


# =============================================================================
# Rules
# =============================================================================


@dataclass
class Rule:
    """A discovered environment rule."""

    precondition: Predicate
    action: Action
    effect: Predicate
    context: Optional[str] = None
    confidence: float = 0.5

    def precondition_matches(self, state: SymbolicState) -> bool:
        """Check if rule precondition matches state."""
        return self.precondition.evaluate(state)

    def predict_effect(self, state: SymbolicState) -> SymbolicState:
        """Apply rule effect to state."""
        return self.effect.apply(state)

    def to_hash(self) -> str:
        """Generate unique hash for rule."""
        return f"{self.precondition}|{self.action.type.name}|{self.effect}"

    def to_description(self) -> str:
        """Human-readable description."""
        return f"IF {self.precondition} AND {self.action.type.name} THEN {self.effect}"

    def suggested_actions(self, state: SymbolicState) -> List[Action]:
        """If precondition matches, suggest the action."""
        if self.precondition_matches(state):
            return [self.action]
        return []


# =============================================================================
# Transitions and Episodes
# =============================================================================


@dataclass
class Transition:
    """A single environment transition."""

    observation: Any  # FrameData from arcengine
    action: Action
    reward: float = 0.0
    done: bool = False
    done_reason: str = ""

    # Optional enriched data
    features: Optional[torch.Tensor] = None
    symbolic_state: Optional[SymbolicState] = None
    belief: Optional[Any] = None  # BeliefState
    predicted_features: Optional[torch.Tensor] = None
    goal_discovery_output: Optional[Any] = None

    # Metadata
    frame_index: int = 0
    episode_index: int = 0


@dataclass
class Episode:
    """A complete episode."""

    transitions: List[Transition] = field(default_factory=list)
    game_id: str = ""
    total_reward: float = 0.0
    achieved_goal: bool = False
    num_levels_completed: int = 0

    def add(self, transition: Transition) -> None:
        """Add transition to episode."""
        transition.frame_index = len(self.transitions)
        self.transitions.append(transition)
        self.total_reward += transition.reward

        if transition.done and transition.done_reason == "win":
            self.achieved_goal = True

    def __len__(self) -> int:
        return len(self.transitions)


# =============================================================================
# Network Outputs
# =============================================================================


@dataclass
class PerceptionOutput:
    """Output from perception tower."""

    neural_features: torch.Tensor
    symbolic_state: Optional[SymbolicState]
    grid: torch.Tensor


@dataclass
class FastPolicyOutput:
    """Output from fast policy network."""

    action: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    action_logits: torch.Tensor
    confidence: Optional[torch.Tensor] = None
    action_log_prob: Optional[torch.Tensor] = None


@dataclass
class BeliefState:
    """Belief state from RSSM."""

    deterministic: torch.Tensor
    stochastic: torch.Tensor
    prior_dist: Any  # Distribution
    posterior_dist: Optional[Any]  # Distribution (None during imagination)
    obs_prediction: torch.Tensor
    hidden_hypotheses: List[Tuple[str, float]]
    uncertainty: torch.Tensor


@dataclass
class GoalDiscoveryOutput:
    """Output from goal discovery module."""

    goal_likelihood: torch.Tensor
    goal_representation: torch.Tensor
    prototype_attention: torch.Tensor
    transition_logits: torch.Tensor
    prediction_error_magnitude: torch.Tensor


@dataclass
class ArbiterDecision:
    """Decision from metacognitive arbiter."""

    use_slow_system: torch.Tensor
    fast_score: torch.Tensor
    slow_score: torch.Tensor
