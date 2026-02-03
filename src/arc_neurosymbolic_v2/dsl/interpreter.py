"""
DSL Interpreter

Executes programs written in the 57-primitive DSL.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from arcengine import GameAction

if TYPE_CHECKING:
    from ..agent.neurosymbolic_v2 import SymbolicState


class DSLInterpreter:
    """
    Execute DSL programs step by step.

    Maintains program counter and execution context.
    """

    def __init__(self):
        self.program: Optional[str] = None
        self.program_counter: int = 0
        self.context: Dict[str, Any] = {}

    def load_program(self, program: str) -> None:
        """Load a new program for execution."""
        self.program = program
        self.program_counter = 0
        self.context = {}

    def execute_step(self, program: str, state: SymbolicState) -> Optional[GameAction]:
        """
        Execute one step of the program.

        Returns the next action to take, or None if program complete.
        """
        if not program:
            return None

        # Parse and execute first action
        action = self._parse_action(program, state)
        return action

    def _parse_action(self, program: str, state: SymbolicState) -> GameAction:
        """Parse program and extract next action."""
        program_lower = program.lower()

        # Simple pattern matching for demo
        if "move_toward" in program_lower or "reach" in program_lower:
            return self._move_toward_target(state)
        elif "up" in program_lower:
            return GameAction.ACTION1
        elif "down" in program_lower:
            return GameAction.ACTION2
        elif "left" in program_lower:
            return GameAction.ACTION3
        elif "right" in program_lower:
            return GameAction.ACTION4
        elif "interact" in program_lower:
            return GameAction.ACTION5
        else:
            return GameAction.ACTION1

    def _move_toward_target(self, state: SymbolicState) -> GameAction:
        """Generate action to move toward nearest object."""
        if not state.agent or not state.objects:
            return GameAction.ACTION1

        agent_pos = state.agent.position

        # Find nearest non-agent object
        nearest = None
        min_dist = float("inf")

        for obj in state.objects:
            if obj.object_id == state.agent.object_id:
                continue

            dist = abs(obj.position.x - agent_pos.x) + abs(obj.position.y - agent_pos.y)
            if dist < min_dist:
                min_dist = dist
                nearest = obj

        if not nearest:
            return GameAction.ACTION1

        dx = nearest.position.x - agent_pos.x
        dy = nearest.position.y - agent_pos.y

        if abs(dx) > abs(dy):
            return GameAction.ACTION4 if dx > 0 else GameAction.ACTION3
        else:
            return GameAction.ACTION2 if dy > 0 else GameAction.ACTION1
