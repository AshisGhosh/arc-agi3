"""
ARIA v2 Core - Evidence-based game understanding.

No pre-training required. Learn during play through observation.

Components:
- ObservationTracker: Detect changes between frames
- BeliefState: Track evidence about colors/positions
- Navigation: A* pathfinding
- Exploration: Random vs Learned policies (A/B testable)
- ActionSelector: Combine everything into decisions
"""

from .observation_tracker import (
    ObservationTracker,
    Observation,
    PixelChange,
    RegionChange,
)

from .belief_state import (
    BeliefState,
    ColorBelief,
    PositionBelief,
    TimerState,
)

from .navigation import (
    AStarNavigator,
    PathResult,
    find_nearest,
    positions_of_color,
)

from .exploration import (
    ExplorationPolicy,
    ExplorationDecision,
    ExplorationStrategy,
    RandomExplorationPolicy,
    SystematicExplorationPolicy,
    LearnedExplorationPolicy,
    ExplorationPolicyNetwork,
)

from .action_selector import (
    ActionSelector,
    ActionDecision,
    SimpleRuleGeneralizer,
)

from .agent import (
    ARIAAgent,
    AgentConfig,
    EpisodeStats,
    ABTestRunner,
)

from .exploration_training import (
    ExplorationTrainer,
    TrainingConfig,
    SyntheticGameSimulator,
)

from .llm_advisor import (
    LLMAdvisor,
    LLMAdvice,
)

__all__ = [
    # Observation
    "ObservationTracker",
    "Observation",
    "PixelChange",
    "RegionChange",
    # Belief
    "BeliefState",
    "ColorBelief",
    "PositionBelief",
    "TimerState",
    # Navigation
    "AStarNavigator",
    "PathResult",
    "find_nearest",
    "positions_of_color",
    # Exploration
    "ExplorationPolicy",
    "ExplorationDecision",
    "ExplorationStrategy",
    "RandomExplorationPolicy",
    "SystematicExplorationPolicy",
    "LearnedExplorationPolicy",
    "ExplorationPolicyNetwork",
    # Action Selection
    "ActionSelector",
    "ActionDecision",
    "SimpleRuleGeneralizer",
    # Agent
    "ARIAAgent",
    "AgentConfig",
    "EpisodeStats",
    "ABTestRunner",
    # Training
    "ExplorationTrainer",
    "TrainingConfig",
    "SyntheticGameSimulator",
    # LLM
    "LLMAdvisor",
    "LLMAdvice",
]
