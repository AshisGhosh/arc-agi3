"""
Domain-Specific Language (DSL) for ARC-AGI-3

57 primitives organized by ARC core knowledge priors:
- Objectness (12): detect, track, classify objects
- Numbers/Counting (8): quantitative reasoning
- Basic Geometry (14): shapes, positions, transformations
- Spatial Relations (9): inter-object relationships
- Goal-Directedness (6): intentional behavior
- Elementary Physics (8): basic physical concepts
- Conditional/Temporal (10): control flow and timing
"""

from .compiler import ProgramCompiler
from .interpreter import DSLInterpreter
from .templates import ProgramTemplate, ProgramTemplateLibrary

__all__ = [
    "DSLInterpreter",
    "ProgramCompiler",
    "ProgramTemplate",
    "ProgramTemplateLibrary",
]
