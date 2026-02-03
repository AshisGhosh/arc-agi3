"""
ARC-DREAMER v2: Error-Correcting World Models with Symbolic Grounding

This module implements the ARC-DREAMER v2 architecture for ARC-AGI-3,
addressing critical weaknesses identified in v1:

1. Error-Correcting World Model (ensemble + consistency + grounding)
2. Principled Intrinsic Motivation (information-theoretic formulation)
3. Defined Hierarchical Structure (object-centric subgoals)
4. Goal Discovery Module (contrastive learning)
5. Hidden State Inference (POMDP belief tracking)
6. Symbolic Grounding (slot attention + disentanglement)
7. Extended Planning (MCTS with 50+ step horizon)

Target Score: 9/10 on ARC-AGI-3
"""

from .agent import ARCDreamerV2Agent
from .belief_tracking import (
    AnomalyDetector,
    BeliefStateTracker,
)
from .goal_discovery import (
    GoalConditionedPolicy,
    GoalDiscoveryModule,
)
from .hierarchy import (
    HierarchicalPolicy,
    ObjectCentricSubgoal,
    OptionDiscovery,
    SubgoalType,
)
from .intrinsic_motivation import (
    PrincipledIntrinsicMotivation,
    StateHasher,
)
from .planning import (
    MCTSNode,
    MCTSPlanner,
)
from .symbolic_grounding import (
    RuleExtractor,
    SlotAttention,
    SymbolicGrounding,
)
from .world_model import (
    BidirectionalConsistency,
    EnsembleWorldModel,
    GroundingController,
)

__version__ = "2.0.0"
__all__ = [
    # World Model
    "EnsembleWorldModel",
    "BidirectionalConsistency",
    "GroundingController",
    # Intrinsic Motivation
    "PrincipledIntrinsicMotivation",
    "StateHasher",
    # Hierarchy
    "ObjectCentricSubgoal",
    "SubgoalType",
    "OptionDiscovery",
    "HierarchicalPolicy",
    # Goal Discovery
    "GoalDiscoveryModule",
    "GoalConditionedPolicy",
    # Belief Tracking
    "BeliefStateTracker",
    "AnomalyDetector",
    # Symbolic Grounding
    "SymbolicGrounding",
    "SlotAttention",
    "RuleExtractor",
    # Planning
    "MCTSPlanner",
    "MCTSNode",
    # Agent
    "ARCDreamerV2Agent",
]
