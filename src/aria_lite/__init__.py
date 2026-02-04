"""
ARIA-Lite: Minimal Dual-System Architecture for ARC-AGI-3

A self-contained 29M parameter architecture implementing:
- Fast policy (neural habits)
- Slow policy (deliberate planning)
- Metacognitive arbiter for system switching

Target: 7GB VRAM, 7.0/10 expected score on ARC-AGI-3
"""

from .arbiter import Arbiter, ArbiterDecision, create_arbiter
from .belief import BeliefOutput, BeliefStateTracker, create_belief_tracker
from .config import ARIALiteConfig
from .encoder import GridEncoderLite, create_encoder
from .fast_policy import FastPolicy, FastPolicyOutput, create_fast_policy
from .slow_policy import SlowPolicy, SlowPolicyOutput, create_slow_policy
from .world_model import EnsembleWorldModel, WorldModelOutput, create_world_model

__version__ = "0.1.0"
__all__ = [
    "ARIALiteConfig",
    "GridEncoderLite",
    "create_encoder",
    "EnsembleWorldModel",
    "WorldModelOutput",
    "create_world_model",
    "BeliefStateTracker",
    "BeliefOutput",
    "create_belief_tracker",
    "FastPolicy",
    "FastPolicyOutput",
    "create_fast_policy",
    "SlowPolicy",
    "SlowPolicyOutput",
    "create_slow_policy",
    "Arbiter",
    "ArbiterDecision",
    "create_arbiter",
]
