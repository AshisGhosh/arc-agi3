"""
Program Templates

Pre-defined parameterized program patterns for common ARC-AGI-3 scenarios.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from ..agent.neurosymbolic_v2 import SymbolicState


@dataclass(frozen=True)
class ProgramTemplate:
    """A parameterized program pattern that can be instantiated."""

    template_id: str
    pattern: str
    parameter_types: Tuple[str, ...]
    applicability_predicate: str
    estimated_actions: int

    def instantiate(self, **params) -> str:
        """Fill in template parameters to create concrete program."""
        program = self.pattern
        for name, value in params.items():
            program = program.replace(f"{{{name}}}", str(value))
        return program


class ProgramTemplateLibrary:
    """
    Pre-defined templates covering common ARC-AGI-3 patterns.

    Templates are organized by category:
    - Navigation: Moving to targets
    - Collection: Gathering items
    - Avoidance: Dodging obstacles
    - Interaction: Triggering objects
    - Pattern: Completing patterns
    """

    TEMPLATES = [
        # Navigation templates
        ProgramTemplate(
            template_id="nav_to_object",
            pattern="reach(agent, find_nearest({target_type}))",
            parameter_types=("ObjectType",),
            applicability_predicate="exists(target_type) and not adjacent_to(agent, target)",
            estimated_actions=10,
        ),
        ProgramTemplate(
            template_id="nav_to_position",
            pattern="reach(agent, position({x}, {y}))",
            parameter_types=("int", "int"),
            applicability_predicate="valid_position(x, y)",
            estimated_actions=10,
        ),
        # Collection templates
        ProgramTemplate(
            template_id="collect_all",
            pattern="""for_each(
                detect_objects_of_type({collectible_type}),
                lambda obj: seq(reach(agent, obj), interact(obj))
            )""",
            parameter_types=("ObjectType",),
            applicability_predicate="count(collectible_type) > 0",
            estimated_actions=20,
        ),
        ProgramTemplate(
            template_id="collect_nearest",
            pattern="seq(reach(agent, find_nearest({collectible_type})), interact(find_nearest({collectible_type})))",
            parameter_types=("ObjectType",),
            applicability_predicate="count(collectible_type) > 0",
            estimated_actions=8,
        ),
        # Avoidance templates
        ProgramTemplate(
            template_id="avoid_and_reach",
            pattern="avoid(agent, detect_objects_of_type({threat_type}), while_reaching={target})",
            parameter_types=("ObjectType", "Object"),
            applicability_predicate="exists(threat_type) and exists(target)",
            estimated_actions=15,
        ),
        # Interaction templates
        ProgramTemplate(
            template_id="trigger_then_proceed",
            pattern="seq(reach(agent, {trigger}), interact({trigger}), reach(agent, {goal}))",
            parameter_types=("Object", "Object"),
            applicability_predicate="exists(trigger) and exists(goal)",
            estimated_actions=12,
        ),
        ProgramTemplate(
            template_id="interact_adjacent",
            pattern="seq(move_toward(agent, {target}), interact({target}))",
            parameter_types=("Object",),
            applicability_predicate="distance(agent, target) <= 2",
            estimated_actions=3,
        ),
        # Pattern templates
        ProgramTemplate(
            template_id="follow_path",
            pattern="for_each({path_positions}, lambda pos: reach(agent, pos))",
            parameter_types=("List[Position]",),
            applicability_predicate="len(path_positions) > 0",
            estimated_actions=25,
        ),
        # Exploration templates
        ProgramTemplate(
            template_id="explore_grid",
            pattern="""seq(
                reach(agent, position(0, 0)),
                reach(agent, position(max_x, 0)),
                reach(agent, position(max_x, max_y)),
                reach(agent, position(0, max_y))
            )""",
            parameter_types=(),
            applicability_predicate="True",
            estimated_actions=50,
        ),
        # Conditional templates
        ProgramTemplate(
            template_id="if_blocked_go_around",
            pattern="""if_then_else(
                path_exists(agent.position, {target}.position),
                reach(agent, {target}),
                avoid(agent, obstacles, while_reaching={target})
            )""",
            parameter_types=("Object",),
            applicability_predicate="exists(target)",
            estimated_actions=15,
        ),
    ]

    def __init__(self):
        self.templates = {t.template_id: t for t in self.TEMPLATES}
        self._applicability_cache: Dict[str, List[ProgramTemplate]] = {}

    def get_template(self, template_id: str) -> Optional[ProgramTemplate]:
        """Get template by ID."""
        return self.templates.get(template_id)

    def find_applicable(self, state: SymbolicState) -> List[Tuple[ProgramTemplate, Dict]]:
        """
        Find all templates applicable to current state.

        Returns list of (template, parameter_bindings) tuples.
        """
        applicable = []

        for template in self.TEMPLATES:
            bindings = self._check_applicability(template, state)
            if bindings is not None:
                applicable.append((template, bindings))

        # Sort by estimated efficiency
        applicable.sort(key=lambda x: x[0].estimated_actions)

        return applicable

    def _check_applicability(
        self, template: ProgramTemplate, state: SymbolicState
    ) -> Optional[Dict]:
        """
        Check if template applies to state.

        Returns parameter bindings if applicable, None otherwise.
        """
        # Simplified check - full implementation would evaluate predicates
        bindings = {}

        if not state.objects:
            return None

        # Extract common bindings
        if "target_type" in template.pattern or "collectible_type" in template.pattern:
            # Find most common object type
            types = {}
            for obj in state.objects:
                t = obj.object_type.name
                types[t] = types.get(t, 0) + 1

            if types:
                most_common = max(types.items(), key=lambda x: x[1])[0]
                bindings["target_type"] = most_common
                bindings["collectible_type"] = most_common

        if "{target}" in template.pattern:
            # Bind to nearest object
            if state.agent and state.objects:
                nearest = min(
                    (o for o in state.objects if o.object_id != state.agent.object_id),
                    key=lambda o: abs(o.position.x - state.agent.position.x)
                    + abs(o.position.y - state.agent.position.y),
                    default=None,
                )
                if nearest:
                    bindings["target"] = f"object_{nearest.object_id}"

        return bindings if bindings else None
