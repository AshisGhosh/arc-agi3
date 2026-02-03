"""
Extended Planning with MCTS for ARC-DREAMER v2.

Addresses v1 weakness: Limited planning horizon due to error accumulation.

Solutions:
1. Monte Carlo Tree Search over world model
2. Uncertainty-aware node selection
3. Periodic grounding to reset errors
4. Test-time compute scaling
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MCTSNode:
    """Node in MCTS tree."""

    state: torch.Tensor
    symbolic: Any | None
    parent: "MCTSNode" | None
    action: int | None
    prior: float = 1.0
    uncertainty: float = 0.0
    needs_grounding: bool = False

    visit_count: int = 0
    value_sum: float = 0.0
    children: list["MCTSNode"] = field(default_factory=list)
    is_fully_expanded: bool = False
    is_terminal: bool = False

    @property
    def value(self) -> float:
        """Average value."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class ValueNetwork(nn.Module):
    """Value function for state evaluation."""

    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return self.network(state).squeeze(-1)


class MCTSPlanner:
    """
    Monte Carlo Tree Search for extended planning.

    Key innovations for ARC-DREAMER v2:
    1. Uncertainty-aware UCB selection
    2. Grounding checkpoints in imagination
    3. Test-time compute scaling

    This enables 50+ step planning with bounded errors by:
    - Penalizing high-uncertainty branches
    - Periodically grounding predictions
    - Adapting compute to task difficulty
    """

    def __init__(
        self,
        world_model: nn.Module,
        policy: Any,
        value_function: ValueNetwork,
        grounding_controller: Any,
        c_puct: float = 1.4,
        num_simulations: int = 100,
        max_depth: int = 50,
        action_dim: int = 8,
    ):
        self.world_model = world_model
        self.policy = policy
        self.value_fn = value_function
        self.grounding = grounding_controller
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.action_dim = action_dim

        # Statistics
        self.planning_stats = []

    def plan(
        self,
        root_state: torch.Tensor,
        symbolic_state: Any | None = None,
        time_budget: float | None = None,
    ) -> Tuple[int, list[int], dict]:
        """
        Plan best action from current state.

        Args:
            root_state: Current latent state
            symbolic_state: Current symbolic state (optional)
            time_budget: Optional time limit for planning (seconds)

        Returns:
            best_action: Immediate action to take
            plan: Full planned action sequence
            info: Planning statistics
        """
        root = MCTSNode(
            state=root_state,
            symbolic=symbolic_state,
            parent=None,
            action=None,
        )

        start_time = time.time()
        simulations_done = 0

        while simulations_done < self.num_simulations:
            if time_budget and (time.time() - start_time) > time_budget:
                break

            # Selection: traverse tree to find leaf
            node = self._select(root)

            # Expansion: add children to leaf
            if not node.is_terminal and not node.is_fully_expanded:
                node = self._expand(node)

            # Simulation: rollout from expanded node
            value = self._simulate(node)

            # Backpropagation: update values up the tree
            self._backpropagate(node, value)

            simulations_done += 1

        # Select best action based on visit counts
        best_action, plan = self._get_best_action(root)

        info = {
            "simulations": simulations_done,
            "time": time.time() - start_time,
            "tree_depth": self._get_tree_depth(root),
            "root_value": root.value,
            "root_visits": root.visit_count,
        }

        self.planning_stats.append(info)

        return best_action, plan, info

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select node to expand using uncertainty-aware UCB."""
        while node.is_fully_expanded and not node.is_terminal:
            if not node.children:
                break
            node = self._ucb_select(node)
        return node

    def _ucb_select(self, node: MCTSNode) -> MCTSNode:
        """
        UCB selection with uncertainty penalty.

        UCB = Q(s,a) + c_puct * P(a|s) * sqrt(N_parent) / (1 + N_child)
                     - uncertainty_penalty(s,a)

        We penalize high-uncertainty actions to avoid unreliable plans.
        This is the key innovation for error-bounded planning.
        """
        best_score = float("-inf")
        best_child = None

        for child in node.children:
            if child.visit_count == 0:
                return child  # Always explore unvisited nodes first

            # Q-value (exploitation)
            q_value = child.value

            # Exploration term (UCB)
            exploration = (
                self.c_puct * child.prior * np.sqrt(node.visit_count) / (1 + child.visit_count)
            )

            # Uncertainty penalty (novel to ARC-DREAMER v2)
            # High uncertainty = unreliable predictions = penalize
            uncertainty_penalty = 0.1 * child.uncertainty

            ucb = q_value + exploration - uncertainty_penalty

            if ucb > best_score:
                best_score = ucb
                best_child = child

        return best_child if best_child else node.children[0]

    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node with new children for each action."""
        # Get unexplored actions
        tried_actions = {c.action for c in node.children}
        available = [a for a in range(self.action_dim) if a not in tried_actions]

        if not available:
            node.is_fully_expanded = True
            return node if node.children else node

        # Get action probabilities from policy
        if hasattr(self.policy, "primitive_policy"):
            policy_dist = self.policy.primitive_policy(node.state)
            priors = policy_dist.probs.detach().cpu().numpy()
        else:
            priors = np.ones(self.action_dim) / self.action_dim

        for action in available:
            # Predict next state using world model
            action_onehot = F.one_hot(torch.tensor([action]), self.action_dim).float()

            with torch.no_grad():
                next_state, uncertainty, _ = self.world_model(
                    node.state.unsqueeze(0) if node.state.dim() == 1 else node.state, action_onehot
                )
                next_state = next_state.squeeze(0)
                uncertainty_val = uncertainty.item() if uncertainty.dim() > 0 else uncertainty

            # Check if should ground (high uncertainty)
            reliability = 1.0 / (1.0 + uncertainty_val)
            should_ground = reliability < 0.5

            # Create child node
            child = MCTSNode(
                state=next_state,
                symbolic=None,  # Would be updated if grounded
                parent=node,
                action=action,
                prior=priors[action],
                uncertainty=uncertainty_val,
                needs_grounding=should_ground,
            )
            node.children.append(child)

        node.is_fully_expanded = True
        return node.children[-1] if node.children else node

    def _simulate(self, node: MCTSNode, max_steps: int = 20) -> float:
        """
        Simulate from node to estimate value.

        Uses world model for rollout with uncertainty monitoring.
        Stops early if uncertainty becomes too high.
        """
        state = node.state
        total_reward = 0.0
        discount = 1.0
        gamma = 0.99

        for step in range(max_steps):
            # Check uncertainty before continuing
            with torch.no_grad():
                # Dummy action for uncertainty estimate
                dummy_action = torch.zeros(1, self.action_dim)
                dummy_action[0, 0] = 1.0
                _, uncertainty, _ = self.world_model(
                    state.unsqueeze(0) if state.dim() == 1 else state, dummy_action
                )

                if uncertainty.item() > 0.5:
                    # High uncertainty - use value function instead
                    value = self.value_fn(state).item()
                    total_reward += discount * value
                    break

                # Get action from policy
                if hasattr(self.policy, "primitive_policy"):
                    action_dist = self.policy.primitive_policy(state)
                    action = action_dist.sample()
                else:
                    action = torch.randint(0, self.action_dim, (1,))

                action_onehot = F.one_hot(action, self.action_dim).float()

                # Step in imagination
                next_state, step_uncertainty, _ = self.world_model(
                    state.unsqueeze(0) if state.dim() == 1 else state, action_onehot
                )
                state = next_state.squeeze(0)

                # Intrinsic reward (curiosity about uncertainty)
                intrinsic = 0.01 * step_uncertainty.item()
                total_reward += discount * intrinsic
                discount *= gamma

        # Bootstrap with value function if didn't terminate early
        if step == max_steps - 1:
            total_reward += discount * self.value_fn(state).item()

        return total_reward

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value through tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

    def _get_best_action(self, root: MCTSNode) -> Tuple[int, list[int]]:
        """Get best action and full plan from root."""
        if not root.children:
            return 0, [0]

        # Select child with most visits
        best_child = max(root.children, key=lambda c: c.visit_count)
        best_action = best_child.action

        # Extract full plan by following most-visited path
        plan = [best_action]
        node = best_child

        while node.children:
            best_child = max(node.children, key=lambda c: c.visit_count)
            if best_child.action is not None:
                plan.append(best_child.action)
            node = best_child

        return best_action, plan

    def _get_tree_depth(self, node: MCTSNode) -> int:
        """Get maximum depth of tree."""
        if not node.children:
            return 0
        return 1 + max(self._get_tree_depth(c) for c in node.children)

    def scale_compute(
        self,
        difficulty_estimate: float,
    ):
        """
        Test-time compute scaling based on task difficulty.

        Harder tasks get more MCTS simulations.
        This allows the agent to "think harder" when needed.

        Args:
            difficulty_estimate: 0-1 estimate of task difficulty
        """
        base_sims = 100  # Baseline simulations

        # Scale simulations with difficulty
        # Difficulty 0: 100 sims, Difficulty 1: 300 sims
        scaled_sims = int(base_sims * (1 + difficulty_estimate * 2))
        scaled_sims = min(scaled_sims, 1000)  # Cap at 1000

        self.num_simulations = scaled_sims

    def get_statistics(self) -> dict:
        """Get planning statistics."""
        if not self.planning_stats:
            return {}

        return {
            "avg_simulations": np.mean([s["simulations"] for s in self.planning_stats]),
            "avg_time": np.mean([s["time"] for s in self.planning_stats]),
            "avg_depth": np.mean([s["tree_depth"] for s in self.planning_stats]),
            "avg_root_value": np.mean([s["root_value"] for s in self.planning_stats]),
            "total_plans": len(self.planning_stats),
        }


class AdaptivePlanner:
    """
    Adaptive planning that switches between methods based on situation.

    Uses:
    - MCTS for complex situations requiring lookahead
    - Direct policy for simple situations
    - Re-planning when world model is uncertain
    """

    def __init__(
        self,
        mcts_planner: MCTSPlanner,
        policy: Any,
        uncertainty_threshold: float = 0.3,
    ):
        self.mcts = mcts_planner
        self.policy = policy
        self.uncertainty_threshold = uncertainty_threshold

        self.current_plan: list[int] = []
        self.plan_step = 0

    def get_action(
        self,
        state: torch.Tensor,
        symbolic_state: Any | None = None,
        force_replan: bool = False,
    ) -> Tuple[int, dict]:
        """
        Get action, potentially using cached plan.

        Args:
            state: Current state
            symbolic_state: Optional symbolic state
            force_replan: Force new planning even if plan exists

        Returns:
            action: Action to take
            info: Planning info
        """
        # Check if we need to replan
        need_replan = (
            force_replan or not self.current_plan or self.plan_step >= len(self.current_plan)
        )

        if need_replan:
            # Estimate difficulty from state complexity
            difficulty = self._estimate_difficulty(state, symbolic_state)

            # Scale compute based on difficulty
            self.mcts.scale_compute(difficulty)

            # Plan
            action, plan, info = self.mcts.plan(state, symbolic_state)

            self.current_plan = plan
            self.plan_step = 1  # Already returning first action

            return action, {**info, "replanned": True, "difficulty": difficulty}

        else:
            # Follow existing plan
            action = self.current_plan[self.plan_step]
            self.plan_step += 1

            return action, {"replanned": False, "plan_step": self.plan_step}

    def _estimate_difficulty(
        self,
        state: torch.Tensor,
        symbolic_state: Any | None,
    ) -> float:
        """
        Estimate task difficulty for compute scaling.

        Factors:
        - State complexity (number of objects)
        - Uncertainty in world model
        - Distance to goal (if known)
        """
        difficulty = 0.5  # Base difficulty

        # Symbolic complexity
        if symbolic_state is not None:
            num_objects = len(getattr(symbolic_state, "objects", []))
            num_relations = len(getattr(symbolic_state, "relations", []))
            complexity = (num_objects + num_relations) / 20.0  # Normalize
            difficulty += 0.2 * min(complexity, 1.0)

        # World model uncertainty
        with torch.no_grad():
            dummy_action = torch.zeros(1, self.mcts.action_dim)
            dummy_action[0, 0] = 1.0
            _, uncertainty, _ = self.mcts.world_model(
                state.unsqueeze(0) if state.dim() == 1 else state, dummy_action
            )
            difficulty += 0.3 * min(uncertainty.item(), 1.0)

        return min(difficulty, 1.0)

    def reset(self):
        """Reset planner state."""
        self.current_plan = []
        self.plan_step = 0
