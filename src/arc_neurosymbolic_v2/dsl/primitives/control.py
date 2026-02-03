"""
Conditional and Temporal Primitives (10 total)

These primitives handle control flow and temporal reasoning:
- Sequencing and parallelization
- Conditional execution
- Loops and iteration
- Timing and deferral
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
)

from .goals import ActionSequence, MoveAction

if TYPE_CHECKING:
    from .objectness import GridObject

T = TypeVar("T")
Action = Union[MoveAction, "ConditionalAction", "LoopAction", "DeferredAction"]


@dataclass
class ActionSet:
    """Actions that can potentially execute in parallel (for planning)."""

    actions: List[Action]

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)


@dataclass
class ConditionalAction:
    """An action that executes based on a condition."""

    condition: Callable[[], bool]
    true_branch: Action
    false_branch: Optional[Action] = None

    def evaluate(self) -> Optional[Action]:
        """Evaluate condition and return the appropriate action."""
        if self.condition():
            return self.true_branch
        return self.false_branch


@dataclass
class LoopAction:
    """A looping action that repeats while condition holds."""

    condition: Callable[[], bool]
    body: Action
    max_iterations: int = 100
    _iteration: int = 0

    def should_continue(self) -> bool:
        """Check if loop should continue."""
        return self._iteration < self.max_iterations and self.condition()

    def step(self) -> Optional[Action]:
        """Execute one iteration."""
        if self.should_continue():
            self._iteration += 1
            return self.body
        return None


@dataclass
class DeferredAction:
    """An action that waits for a trigger before executing."""

    trigger_condition: Callable[[], bool]
    action: Action
    _triggered: bool = False

    def check_trigger(self) -> Optional[Action]:
        """Check if trigger fired and return action if so."""
        if not self._triggered and self.trigger_condition():
            self._triggered = True
            return self.action
        return None


# =============================================================================
# SEQUENCING PRIMITIVES
# =============================================================================


def seq(*actions: Action) -> ActionSequence:
    """
    Execute actions in sequence.

    Args:
        *actions: Actions to execute sequentially

    Returns:
        ActionSequence containing all actions in order
    """
    all_actions: List[MoveAction] = []

    for action in actions:
        if isinstance(action, MoveAction):
            all_actions.append(action)
        elif isinstance(action, ActionSequence):
            all_actions.extend(action.actions)
        elif isinstance(action, ConditionalAction):
            result = action.evaluate()
            if result:
                expanded = seq(result)
                all_actions.extend(expanded.actions)
        elif isinstance(action, LoopAction):
            while action.should_continue():
                step_result = action.step()
                if step_result:
                    expanded = seq(step_result)
                    all_actions.extend(expanded.actions)

    return ActionSequence(actions=all_actions, estimated_steps=len(all_actions))


def parallel(*actions: Action) -> ActionSet:
    """
    Mark actions as potentially parallelizable.

    Note: In ARC-AGI-3, actual parallel execution may not be possible.
    This is mainly for planning purposes to identify independent actions.

    Args:
        *actions: Actions that could execute in parallel

    Returns:
        ActionSet containing all actions
    """
    return ActionSet(actions=list(actions))


# =============================================================================
# CONDITIONAL PRIMITIVES
# =============================================================================


def if_then(condition: Callable[[], bool], true_branch: Action) -> ConditionalAction:
    """
    Execute action only if condition is true.

    Args:
        condition: Callable that returns bool
        true_branch: Action to execute if condition is true

    Returns:
        ConditionalAction
    """
    return ConditionalAction(condition=condition, true_branch=true_branch, false_branch=None)


def if_then_else(
    condition: Callable[[], bool], true_branch: Action, false_branch: Action
) -> ConditionalAction:
    """
    Execute one action if condition is true, another if false.

    Args:
        condition: Callable that returns bool
        true_branch: Action to execute if condition is true
        false_branch: Action to execute if condition is false

    Returns:
        ConditionalAction
    """
    return ConditionalAction(
        condition=condition, true_branch=true_branch, false_branch=false_branch
    )


# =============================================================================
# LOOP PRIMITIVES
# =============================================================================


def while_do(condition: Callable[[], bool], body: Action, max_iterations: int = 100) -> LoopAction:
    """
    Repeat body while condition holds.

    Args:
        condition: Callable that returns bool (continue while True)
        body: Action to repeat
        max_iterations: Safety limit on iterations

    Returns:
        LoopAction
    """
    return LoopAction(condition=condition, body=body, max_iterations=max_iterations)


def repeat(n: int, action: Action) -> ActionSequence:
    """
    Execute action n times.

    Args:
        n: Number of times to repeat
        action: Action to repeat

    Returns:
        ActionSequence with action repeated n times
    """
    actions: List[MoveAction] = []

    for _ in range(n):
        if isinstance(action, MoveAction):
            actions.append(action)
        elif isinstance(action, ActionSequence):
            actions.extend(action.actions)
        else:
            # For complex actions, expand them
            expanded = seq(action)
            actions.extend(expanded.actions)

    return ActionSequence(actions=actions, estimated_steps=len(actions))


def until(action: Action, stop_condition: Callable[[], bool]) -> ActionSequence:
    """
    Repeat action until condition becomes true.

    This is evaluated during planning/execution, not at construction time.

    Args:
        action: Action to repeat
        stop_condition: Callable that returns True when should stop

    Returns:
        ActionSequence
    """
    # Note: This returns a "plan" that will be executed until the condition
    # In actual execution, the executor would check the condition each step
    max_iterations = 100
    actions: List[MoveAction] = []

    for _ in range(max_iterations):
        if stop_condition():
            break

        if isinstance(action, MoveAction):
            actions.append(action)
        elif isinstance(action, ActionSequence):
            actions.extend(action.actions)
            break  # Only one sequence execution per iteration
        else:
            expanded = seq(action)
            actions.extend(expanded.actions)

    return ActionSequence(actions=actions, estimated_steps=len(actions))


def for_each(
    objects: List["GridObject"], action_fn: Callable[["GridObject"], Action]
) -> ActionSequence:
    """
    Apply an action to each object in a collection.

    Args:
        objects: List of objects to iterate over
        action_fn: Function that takes an object and returns an action

    Returns:
        ActionSequence with actions for each object
    """
    all_actions: List[MoveAction] = []

    for obj in objects:
        action = action_fn(obj)

        if isinstance(action, MoveAction):
            all_actions.append(action)
        elif isinstance(action, ActionSequence):
            all_actions.extend(action.actions)
        else:
            expanded = seq(action)
            all_actions.extend(expanded.actions)

    return ActionSequence(actions=all_actions, estimated_steps=len(all_actions))


# =============================================================================
# TEMPORAL PRIMITIVES
# =============================================================================


def wait(n_frames: int) -> ActionSequence:
    """
    No-op for n frames (for timing-dependent mechanics).

    Args:
        n_frames: Number of frames to wait

    Returns:
        ActionSequence with WAIT actions
    """
    return ActionSequence(actions=[MoveAction.WAIT] * n_frames, estimated_steps=n_frames)


def after(trigger_condition: Callable[[], bool], action: Action) -> DeferredAction:
    """
    Execute action after trigger condition is met.

    Args:
        trigger_condition: Callable that returns True when trigger fires
        action: Action to execute after trigger

    Returns:
        DeferredAction
    """
    return DeferredAction(trigger_condition=trigger_condition, action=action)
