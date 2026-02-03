"""
ARIA: Adaptive Reasoning with Integrated Abstractions

A hybrid architecture for ARC-AGI-3 combining:
- Fast neural habits + Slow symbolic planning
- Prediction-error driven goal discovery
- Bayesian belief state tracking
- Test-time adaptation via in-context learning
"""

from .adaptation import TestTimeAdapter
from .agent import ARIA
from .arbiter import ArbiterDecision, MetacognitiveArbiter
from .belief import BeliefState, BeliefStateTracker
from .config import ARIAConfig
from .goals import Goal, GoalDiscoveryModule
from .memory import ExperienceReplay, Transition
from .perception import GridObject, PerceptionTower, SymbolicState
from .planner import Rule, RuleLibrary, SlowPlanner
from .policy import FastPolicy, FastPolicyOutput

__version__ = "0.1.0"
__all__ = [
    "ARIA",
    "ARIAConfig",
    "PerceptionTower",
    "SymbolicState",
    "GridObject",
    "FastPolicy",
    "FastPolicyOutput",
    "SlowPlanner",
    "Rule",
    "RuleLibrary",
    "BeliefStateTracker",
    "BeliefState",
    "GoalDiscoveryModule",
    "Goal",
    "MetacognitiveArbiter",
    "ArbiterDecision",
    "ExperienceReplay",
    "Transition",
    "TestTimeAdapter",
]
