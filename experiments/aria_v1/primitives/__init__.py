"""
ARIA-Lite Primitives Module

Procedural generators for primitive action skills:
1. Navigation - Move to targets, avoid obstacles
2. Click/Selection - Predict coordinates, select cells
3. Pattern - Match templates, find differences
4. State Tracking - Remember and apply state
5. Composition - Combine primitives

Each primitive is a self-contained environment that can be used
for pretraining before meta-learning on ARC-AGI-3 games.
"""

from .base import (
    Action,
    PrimitiveEnv,
    PrimitiveFamily,
    PrimitiveResult,
)
from .click import ClickEnv, ClickPrimitiveGenerator
from .composition import CompositionEnv, CompositionGenerator
from .generator import PrimitiveGenerator
from .navigation import NavigationEnv, NavigationPrimitiveGenerator
from .pattern import PatternEnv, PatternPrimitiveGenerator
from .state_tracking import StateTrackingEnv, StateTrackingGenerator

__all__ = [
    "Action",
    "PrimitiveEnv",
    "PrimitiveFamily",
    "PrimitiveResult",
    "NavigationEnv",
    "NavigationPrimitiveGenerator",
    "ClickEnv",
    "ClickPrimitiveGenerator",
    "PatternEnv",
    "PatternPrimitiveGenerator",
    "StateTrackingEnv",
    "StateTrackingGenerator",
    "CompositionEnv",
    "CompositionGenerator",
    "PrimitiveGenerator",
]
