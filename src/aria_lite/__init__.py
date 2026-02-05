"""
ARIA-Lite: Core Components (Archived)

This module contains reusable components from the ARIA v1 experiments.
The full v1 implementation has been archived to experiments/aria_v1/.

For the current approach, see src/aria_v2/ (Language-Guided Meta-Learning).
"""

from .config import ARIALiteConfig
from .encoder import GridEncoderLite, create_encoder
from .encoder_simple import SimpleGridEncoder
from .fast_policy import FastPolicy, FastPolicyOutput, create_fast_policy
from .world_model import EnsembleWorldModel, WorldModelOutput, create_world_model
from .demo_collector import DemoDataset, GameDemo

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "ARIALiteConfig",
    # Encoders (may reuse in v2)
    "GridEncoderLite",
    "SimpleGridEncoder",
    "create_encoder",
    # World Model (reference)
    "EnsembleWorldModel",
    "WorldModelOutput",
    "create_world_model",
    # Policy (reference)
    "FastPolicy",
    "FastPolicyOutput",
    "create_fast_policy",
    # Demo utilities (reusable)
    "DemoDataset",
    "GameDemo",
]
