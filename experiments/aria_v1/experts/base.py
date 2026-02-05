"""Base class for expert solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from ..training.synthetic_env import Action, CellType, EnvState


@dataclass
class SolverResult:
    """Result from expert solver."""

    action: int
    solved: bool  # True if solver knows the solution
    path: Optional[list[int]] = None  # Full action sequence if available


class ExpertSolver(ABC):
    """Abstract base for expert solvers."""

    @abstractmethod
    def solve(self, state: EnvState) -> SolverResult:
        """
        Compute the next action for the given state.

        Args:
            state: Current environment state

        Returns:
            SolverResult with action and metadata
        """
        pass

    @abstractmethod
    def can_solve(self, state: EnvState) -> bool:
        """Check if this solver can handle the given state."""
        pass

    def get_full_solution(self, state: EnvState) -> Optional[list[int]]:
        """Get complete action sequence to solve. Override if available."""
        return None


def find_cell(grid: torch.Tensor, cell_type: int) -> Optional[tuple[int, int]]:
    """Find first occurrence of cell type in grid."""
    positions = (grid == cell_type).nonzero()
    if len(positions) == 0:
        return None
    return (positions[0, 0].item(), positions[0, 1].item())


def find_all_cells(grid: torch.Tensor, cell_type: int) -> list[tuple[int, int]]:
    """Find all occurrences of cell type in grid."""
    positions = (grid == cell_type).nonzero()
    return [(p[0].item(), p[1].item()) for p in positions]


def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Manhattan distance between two points."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
