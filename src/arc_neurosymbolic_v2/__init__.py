"""
ARC-AGI-3 Neurosymbolic v2 Agent

A hybrid neurosymbolic architecture for interactive reasoning that addresses
critical weaknesses in previous approaches:

1. Expanded DSL (57 primitives) covering all ARC core knowledge priors
2. Goal inference via contrastive learning and predictive coding
3. Hidden state detection via Bayesian belief tracking
4. Causal rule induction with intervention-based testing
5. Multi-tier latency optimization (target: 2000+ FPS)
6. Neural fallback for graceful degradation

Target Score: 9/10
"""

__version__ = "2.0.0"

from .agent.neurosymbolic_v2 import NeurosymbolicV2Agent, NeurosymbolicV2Config

__all__ = [
    "NeurosymbolicV2Agent",
    "NeurosymbolicV2Config",
]
