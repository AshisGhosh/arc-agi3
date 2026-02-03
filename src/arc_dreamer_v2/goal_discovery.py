"""
Goal Discovery Module for ARC-DREAMER v2.

Addresses v1 weakness: Assumes reward signal indicates completion.

Solutions:
1. Unsupervised goal detection from state transitions
2. Contrastive learning for goal state identification
3. Goal-conditioned policy that can pursue discovered goals
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Goal:
    """Discovered goal representation."""

    embedding: torch.Tensor
    example_state: torch.Tensor
    description: str | None = None
    times_achieved: int = 0
    achievability_score: float = 0.5


class GoalEncoder(nn.Module):
    """Encodes states into goal space."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, goal_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to goal embedding."""
        return F.normalize(self.encoder(state), dim=-1)


class SignificanceDetector(nn.Module):
    """
    Detects significant state transitions.

    Learns to identify transitions where something "important" happened,
    even without explicit reward signals.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),  # (s, s') concatenated
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, state_pair: torch.Tensor) -> torch.Tensor:
        """Predict significance of state transition."""
        return self.encoder(state_pair)


class GoalMemory:
    """Memory of discovered goals."""

    def __init__(self, max_goals: int = 100):
        self.max_goals = max_goals
        self.goals: list[Goal] = []
        self.goal_embeddings: torch.Tensor | None = None

    def add_goal(self, goal: Goal):
        """Add a discovered goal to memory."""
        if len(self.goals) >= self.max_goals:
            # Remove least-achieved goal
            min_idx = min(range(len(self.goals)), key=lambda i: self.goals[i].times_achieved)
            self.goals.pop(min_idx)

        self.goals.append(goal)
        self._update_embeddings()

    def _update_embeddings(self):
        """Update cached embeddings tensor."""
        if not self.goals:
            self.goal_embeddings = None
        else:
            self.goal_embeddings = torch.stack([g.embedding for g in self.goals])

    def retrieve_nearest(
        self,
        query: torch.Tensor,
        k: int = 5,
    ) -> list[Goal]:
        """Retrieve k nearest goals to query embedding."""
        if not self.goals or self.goal_embeddings is None:
            return []

        # Cosine similarity
        similarities = F.cosine_similarity(query.unsqueeze(0), self.goal_embeddings, dim=-1)

        # Get top-k
        k = min(k, len(self.goals))
        top_k = similarities.topk(k)

        return [self.goals[i] for i in top_k.indices.tolist()]


class GoalConditionedValue(nn.Module):
    """Value function conditioned on a goal."""

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate value of state for achieving goal."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)
        combined = torch.cat([state, goal], dim=-1)
        return self.network(combined).squeeze(-1)


class GoalDiscoveryModule:
    """
    Discovers goals from state transitions without explicit reward.

    Key insight: "Significant" state changes indicate goal states.
    We use contrastive learning to identify what makes these states special.

    Significance indicators:
    1. Object count changes (something appeared/disappeared)
    2. Agent inventory changes
    3. Environmental structure changes (doors opening)
    4. Large state representation changes

    This addresses v1's assumption that reward indicates goals by
    discovering goals from the structure of transitions.

    Reference: MEGA (Pitis et al.) + RIG (Nair et al.)
    """

    def __init__(
        self,
        state_dim: int = 256,
        goal_dim: int = 64,
        significance_threshold: float = 0.5,
    ):
        self.state_dim = state_dim
        self.goal_dim = goal_dim
        self.significance_threshold = significance_threshold

        # Goal encoder: maps states to goal space
        self.goal_encoder = GoalEncoder(state_dim, goal_dim)

        # Significance detector: identifies "important" transitions
        self.significance_detector = SignificanceDetector(state_dim)

        # Goal memory: stores discovered goal states
        self.goal_memory = GoalMemory(max_goals=100)

        # Goal-conditioned value function
        self.goal_value = GoalConditionedValue(state_dim, goal_dim)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.goal_encoder.parameters())
            + list(self.significance_detector.parameters())
            + list(self.goal_value.parameters()),
            lr=3e-4,
        )

        # Statistics
        self.discovered_goals_count = 0

    def detect_significant_transitions(
        self,
        trajectory: Any,
        symbolic_extractor: Any | None = None,
    ) -> list[int]:
        """
        Identify indices where significant state changes occur.

        Combines:
        1. Symbolic analysis (if extractor provided)
        2. Neural significance detector
        3. Large state changes

        Returns indices of "after" states in significant transitions.
        """
        significant_indices = []

        for i in range(len(trajectory.states) - 1):
            s1 = trajectory.states[i]
            s2 = trajectory.states[i + 1]

            significance_score = 0.0

            # Symbolic analysis (if available)
            if symbolic_extractor is not None:
                try:
                    sym1 = symbolic_extractor.extract(s1)
                    sym2 = symbolic_extractor.extract(s2)

                    # Object count change
                    if len(sym1.objects) != len(sym2.objects):
                        significance_score += 0.3

                    # Inventory change
                    if hasattr(sym1, "inventory") and sym1.inventory != sym2.inventory:
                        significance_score += 0.4

                    # Object state change
                    for obj_id in getattr(sym1, "object_ids", set()) & getattr(
                        sym2, "object_ids", set()
                    ):
                        obj1 = sym1.get_object(obj_id)
                        obj2 = sym2.get_object(obj_id)
                        if obj1 is not None and obj2 is not None:
                            if getattr(obj1, "state", None) != getattr(obj2, "state", None):
                                significance_score += 0.3
                                break
                except Exception:
                    pass  # Symbolic extraction failed, rely on neural

            # Neural significance detector (learned)
            with torch.no_grad():
                if s1.dim() == 1:
                    s1_batch = s1.unsqueeze(0)
                    s2_batch = s2.unsqueeze(0)
                else:
                    s1_batch = s1
                    s2_batch = s2

                combined = torch.cat([s1_batch, s2_batch], dim=-1)
                neural_significance = self.significance_detector(combined).item()
                significance_score += 0.5 * neural_significance

            # Large state change
            state_change = F.mse_loss(s1, s2).item()
            if state_change > 0.1:  # Threshold
                significance_score += 0.2

            if significance_score >= self.significance_threshold:
                significant_indices.append(i + 1)  # The "after" state

        return significant_indices

    def learn_goal_representations(
        self,
        trajectories: list,
        symbolic_extractor: Any | None = None,
        num_epochs: int = 100,
    ):
        """
        Learn goal representations using contrastive learning.

        Positive pairs: (state_before_goal, goal_state)
        Negative pairs: (random_state, goal_state)

        This learns what makes goal states special relative to non-goals.
        """
        # Collect goal states
        goal_states = []
        pre_goal_states = []

        for traj in trajectories:
            significant_idx = self.detect_significant_transitions(traj, symbolic_extractor)
            for idx in significant_idx:
                goal_states.append(traj.states[idx])
                pre_goal_states.append(traj.states[idx - 1])

        if not goal_states:
            return

        # Add discovered goals to memory
        for goal_state in goal_states:
            goal_emb = self.goal_encoder(goal_state)
            goal = Goal(
                embedding=goal_emb.detach(),
                example_state=goal_state.detach(),
            )
            self.goal_memory.add_goal(goal)
            self.discovered_goals_count += 1

        # Collect all non-goal states for negatives
        all_states = []
        for traj in trajectories:
            all_states.extend(traj.states)

        # Contrastive learning
        for epoch in range(num_epochs):
            epoch_loss = 0.0

            for goal, pre_goal in zip(goal_states, pre_goal_states):
                # Sample negative (random non-goal state)
                neg_idx = torch.randint(0, len(all_states), (1,)).item()
                negative = all_states[neg_idx]

                # Contrastive loss
                loss = self._contrastive_loss(
                    anchor=self.goal_encoder(goal),
                    positive=self.goal_encoder(pre_goal),
                    negative=self.goal_encoder(negative),
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

    def _contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        """
        InfoNCE contrastive loss.

        Encourages goal encoder to map goal states close to their
        preceding states and far from random states.
        """
        if anchor.dim() == 1:
            anchor = anchor.unsqueeze(0)
        if positive.dim() == 1:
            positive = positive.unsqueeze(0)
        if negative.dim() == 1:
            negative = negative.unsqueeze(0)

        pos_sim = F.cosine_similarity(anchor, positive, dim=-1) / temperature
        neg_sim = F.cosine_similarity(anchor, negative, dim=-1) / temperature

        logits = torch.stack([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(anchor.shape[0], dtype=torch.long, device=anchor.device)

        return F.cross_entropy(logits, labels)

    def propose_goals(
        self,
        current_state: torch.Tensor,
        symbolic_state: Any | None = None,
        num_goals: int = 5,
    ) -> list[Goal]:
        """
        Propose possible goals given current state.

        Combines:
        1. Previously discovered goals from memory
        2. Goals generated by perturbing current state
        3. Object-centric goals (if symbolic state available)
        """
        proposed = []

        # From memory: nearest achievable goals
        current_emb = self.goal_encoder(current_state)
        memory_goals = self.goal_memory.retrieve_nearest(current_emb, k=num_goals // 2 + 1)
        proposed.extend(memory_goals)

        # Perturbation-based goals
        for _ in range(num_goals // 2):
            # Add noise to current state embedding
            noise = torch.randn_like(current_emb) * 0.1
            perturbed_emb = F.normalize(current_emb + noise, dim=-1)
            perturbed_goal = Goal(
                embedding=perturbed_emb.detach(),
                example_state=current_state.detach(),
                description="exploration_goal",
            )
            proposed.append(perturbed_goal)

        # Rank by estimated achievability
        ranked = sorted(
            proposed, key=lambda g: self.goal_value(current_state, g.embedding).item(), reverse=True
        )

        return ranked[:num_goals]

    def compute_goal_reward(
        self,
        state: torch.Tensor,
        goal: Goal,
        threshold: float = 0.1,
    ) -> float:
        """
        Compute reward for progress toward goal.

        Returns sparse reward when goal is achieved,
        plus shaped reward for progress.
        """
        current_emb = self.goal_encoder(state)
        similarity = F.cosine_similarity(
            current_emb.unsqueeze(0), goal.embedding.unsqueeze(0), dim=-1
        ).item()

        # Sparse reward for achievement
        if similarity > 1.0 - threshold:
            goal.times_achieved += 1
            return 1.0

        # Shaped reward for progress (small to avoid reward hacking)
        return 0.01 * similarity


class GoalConditionedPolicy(nn.Module):
    """
    Policy that can pursue any discovered goal.

    pi(a | s, g) where g is a goal embedding.

    This enables the agent to pursue goals discovered by
    the GoalDiscoveryModule.
    """

    def __init__(
        self,
        state_dim: int,
        goal_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.distributions.Categorical:
        """Return action distribution for pursuing goal from state."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)

        combined = torch.cat([state, goal], dim=-1)
        features = self.encoder(combined)
        logits = self.policy_head(features)
        return torch.distributions.Categorical(logits=logits)

    def get_value(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """Return value estimate for (state, goal) pair."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if goal.dim() == 1:
            goal = goal.unsqueeze(0)

        combined = torch.cat([state, goal], dim=-1)
        features = self.encoder(combined)
        return self.value_head(features).squeeze(-1)

    def train_with_hindsight(
        self,
        trajectory: Any,
        achieved_goals: list[torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ):
        """
        Hindsight Experience Replay: relabel failed goals with achieved ones.

        If we tried to reach goal G but ended at state S,
        we can still learn "how to reach S" from this trajectory.

        This dramatically improves sample efficiency for
        goal-conditioned learning.
        """
        for i in range(len(trajectory.states) - 1):
            state = trajectory.states[i]
            action = trajectory.actions[i]
            next_state = trajectory.states[i + 1]

            # Train on achieved goals
            for achieved in achieved_goals:
                # Check if we got closer to this achieved goal
                dist_before = F.mse_loss(state, achieved)
                dist_after = F.mse_loss(next_state, achieved)

                if dist_after < dist_before:
                    # This was a good action for reaching 'achieved'
                    dist = self(state, achieved)
                    loss = -dist.log_prob(torch.tensor([action]))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
