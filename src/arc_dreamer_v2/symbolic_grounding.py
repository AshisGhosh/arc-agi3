"""
Symbolic Grounding for ARC-DREAMER v2.

Addresses v1 weakness: Latent space is uninterpretable.

Solutions:
1. Slot attention for object discovery
2. Auxiliary losses for interpretable dimensions
3. Rule extraction from learned dynamics
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GridObject:
    """Detected object in grid."""

    object_id: int
    color: int
    position: Tuple[int, int]  # Center position
    bounding_box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    pixels: set[Tuple[int, int]] = field(default_factory=set)
    state: str = "default"
    properties: dict = field(default_factory=dict)


@dataclass
class SymbolicState:
    """Symbolic representation of grid state."""

    objects: list[GridObject]
    agent_position: Tuple[int, int] | None
    relations: list[Tuple[int, str, int]]  # (obj1, relation, obj2)
    inventory: set[int] = field(default_factory=set)
    last_interaction: int | None = None
    attention_maps: torch.Tensor | None = None

    @property
    def object_ids(self) -> set[int]:
        return {obj.object_id for obj in self.objects}

    def get_object(self, obj_id: int) -> GridObject | None:
        for obj in self.objects:
            if obj.object_id == obj_id:
                return obj
        return None

    @property
    def object_positions(self) -> torch.Tensor:
        """Return positions of all objects as tensor."""
        if not self.objects:
            return torch.zeros(0, 4)
        positions = []
        for obj in self.objects:
            x, y = obj.position
            x1, y1, x2, y2 = obj.bounding_box
            positions.append([x / 30.0, y / 30.0, (x2 - x1) / 30.0, (y2 - y1) / 30.0])
        return torch.tensor(positions)

    @property
    def object_colors(self) -> torch.Tensor:
        """Return colors of all objects as tensor."""
        if not self.objects:
            return torch.zeros(0, dtype=torch.long)
        return torch.tensor([obj.color for obj in self.objects])

    def path_exists(
        self,
        from_pos: Tuple[int, int],
        to_pos: Tuple[int, int],
    ) -> bool:
        """Check if path exists (placeholder)."""
        # Simple Manhattan distance check
        return True  # Placeholder

    def region_explored(self, position: Tuple[int, int]) -> bool:
        """Check if region explored (placeholder)."""
        return True  # Placeholder


class SlotAttention(nn.Module):
    """
    Slot Attention mechanism for object discovery.

    Iteratively refines slot representations to attend to
    different parts of the input, naturally discovering objects.

    Reference: Locatello et al. "Object-Centric Learning with Slot Attention"
    """

    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        num_slots: int,
        num_iterations: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_dim = slot_dim

        # Learnable slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Normalization
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Attention projections
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # GRU for slot updates
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply slot attention.

        Args:
            inputs: [B, N, input_dim] flattened grid features

        Returns:
            slots: [B, num_slots, slot_dim] discovered object representations
            attention: [B, num_slots, N] attention weights (which pixels each slot attends to)
        """
        B, N, _ = inputs.shape

        # Initialize slots with learned parameters + noise
        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(B, self.num_slots, -1)
        slots = mu + sigma * torch.randn_like(sigma)

        # Prepare inputs
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # [B, N, slot_dim]
        v = self.project_v(inputs)  # [B, N, slot_dim]

        # Iterative refinement
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)  # [B, num_slots, slot_dim]

            # Attention: softmax over slots (competition)
            attn_logits = torch.einsum("bnd,bmd->bnm", k, q)  # [B, N, num_slots]
            attn_logits = attn_logits / (self.slot_dim**0.5)
            attn = F.softmax(attn_logits, dim=-1)  # Softmax over slots

            # Weighted mean of values
            attn_normalized = attn / (attn.sum(dim=1, keepdim=True) + 1e-8)
            updates = torch.einsum("bnm,bnd->bmd", attn_normalized, v)

            # GRU update
            slots = self.gru(
                updates.reshape(B * self.num_slots, -1),
                slots_prev.reshape(B * self.num_slots, -1),
            ).reshape(B, self.num_slots, -1)

            # MLP refinement
            slots = slots + self.mlp(self.norm_mlp(slots))

        # Return attention transposed for [B, num_slots, N] format
        return slots, attn.transpose(1, 2)


class RelationPredictor(nn.Module):
    """Predicts relations between objects."""

    def __init__(
        self,
        slot_dim: int,
        num_relations: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_relations = num_relations

        # Relation types: adjacent, above, below, left, right, contains, overlaps, same_color
        self.relation_net = nn.Sequential(
            nn.Linear(slot_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_relations),
            nn.Sigmoid(),
        )

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        """
        Predict relations between all pairs of slots.

        Args:
            slots: [B, num_slots, slot_dim]

        Returns:
            relations: [B, num_slots, num_slots, num_relations]
        """
        B, N, D = slots.shape

        # Create all pairs
        slots_i = slots.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, D]
        slots_j = slots.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, D]
        pairs = torch.cat([slots_i, slots_j], dim=-1)  # [B, N, N, 2D]

        # Predict relations
        relations = self.relation_net(pairs)  # [B, N, N, num_relations]

        return relations


class SymbolicGrounding(nn.Module):
    """
    Extracts interpretable symbolic state from grid frames.

    Uses slot attention for object discovery with auxiliary losses
    for interpretable slot properties.

    This addresses v1's uninterpretable latent space by providing:
    1. Object-centric representations (each slot = one object)
    2. Disentangled properties (position, color, type in separate dimensions)
    3. Explicit relations between objects

    Reference: Locatello et al. "Object-Centric Learning with Slot Attention"
    """

    def __init__(
        self,
        grid_channels: int = 10,  # 10 colors in ARC (0-9)
        slot_dim: int = 64,
        num_slots: int = 16,
        num_iterations: int = 3,
        grid_size: int = 30,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.grid_size = grid_size

        # Grid encoder (CNN)
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(grid_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 128, grid_size, grid_size) * 0.02)

        # Slot attention
        self.slot_attention = SlotAttention(
            input_dim=128,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_iterations=num_iterations,
        )

        # Property decoders (disentangled)
        self.position_decoder = nn.Linear(slot_dim, 4)  # x, y, w, h
        self.color_decoder = nn.Linear(slot_dim, 10)  # 10 colors
        self.type_decoder = nn.Linear(slot_dim, 8)  # object types
        self.state_decoder = nn.Linear(slot_dim, 4)  # object states

        # Relation predictor
        self.relation_predictor = RelationPredictor(slot_dim)

        # Slot decoder (for reconstruction loss)
        self.slot_decoder = nn.Sequential(
            nn.Linear(slot_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )

    def forward(
        self,
        grid: torch.Tensor,
    ) -> Tuple[torch.Tensor, SymbolicState]:
        """
        Extract slots and symbolic state from grid.

        Args:
            grid: [B, H, W] grid of color values 0-9

        Returns:
            slots: [B, num_slots, slot_dim] object representations
            symbolic: SymbolicState with interpretable properties
        """
        H, W = grid.shape[1], grid.shape[2]

        # One-hot encode colors
        grid_onehot = F.one_hot(grid.long(), 10).permute(0, 3, 1, 2).float()

        # Encode grid with CNN
        features = self.grid_encoder(grid_onehot)  # [B, 128, H, W]

        # Add positional encoding (resize if needed)
        if H != self.grid_size or W != self.grid_size:
            pos_embed = F.interpolate(
                self.pos_embed, size=(H, W), mode="bilinear", align_corners=False
            )
        else:
            pos_embed = self.pos_embed

        features = features + pos_embed

        # Flatten for slot attention
        features_flat = features.flatten(2).permute(0, 2, 1)  # [B, H*W, 128]

        # Slot attention
        slots, attention = self.slot_attention(features_flat)  # [B, num_slots, slot_dim]

        # Decode properties
        positions = torch.sigmoid(self.position_decoder(slots))  # [B, num_slots, 4]
        colors = self.color_decoder(slots)  # [B, num_slots, 10]
        types = self.type_decoder(slots)  # [B, num_slots, 8]
        states = self.state_decoder(slots)  # [B, num_slots, 4]

        # Predict relations
        relations = self.relation_predictor(slots)  # [B, num_slots, num_slots, num_relations]

        # Build symbolic state (for first batch element)
        symbolic = self._build_symbolic_state(
            positions[0], colors[0], types[0], states[0], relations[0], attention[0], H, W
        )

        return slots, symbolic

    def _build_symbolic_state(
        self,
        positions: torch.Tensor,
        colors: torch.Tensor,
        types: torch.Tensor,
        states: torch.Tensor,
        relations: torch.Tensor,
        attention: torch.Tensor,
        H: int,
        W: int,
    ) -> SymbolicState:
        """Build symbolic state from decoded properties."""
        objects = []

        for i in range(self.num_slots):
            # Check if slot is "active" (has significant attention)
            attn_sum = attention[i].sum().item()
            if attn_sum < 0.1:  # Skip empty slots
                continue

            # Decode position
            pos = positions[i]
            x = int(pos[0].item() * W)
            y = int(pos[1].item() * H)
            w = max(1, int(pos[2].item() * W))
            h = max(1, int(pos[3].item() * H))

            # Decode color (argmax)
            color = colors[i].argmax().item()

            # Create object
            obj = GridObject(
                object_id=i,
                color=color,
                position=(x, y),
                bounding_box=(
                    max(0, x - w // 2),
                    max(0, y - h // 2),
                    min(W, x + w // 2),
                    min(H, y + h // 2),
                ),
                state=f"type_{types[i].argmax().item()}",
            )
            objects.append(obj)

        # Extract relations
        relation_list = []
        relation_names = [
            "adjacent",
            "above",
            "below",
            "left",
            "right",
            "contains",
            "overlaps",
            "same_color",
        ]
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue
                for r, rel_name in enumerate(relation_names):
                    if relations[i, j, r].item() > 0.5:
                        relation_list.append((obj1.object_id, rel_name, obj2.object_id))

        # Find agent (color 0 or specific pattern)
        agent_pos = None
        for obj in objects:
            if obj.color == 0:  # Assume agent is color 0
                agent_pos = obj.position
                break

        return SymbolicState(
            objects=objects,
            agent_position=agent_pos,
            relations=relation_list,
            attention_maps=attention,
        )

    def auxiliary_losses(
        self,
        slots: torch.Tensor,
        symbolic: SymbolicState,
        ground_truth: SymbolicState | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Auxiliary losses for interpretable slots.

        These losses encourage:
        1. Slot specialization (each slot = one object)
        2. Disentanglement (each property dimension = one property)
        3. Consistency with ground truth if available
        """
        losses = {}

        # Slot specialization: attention should be sparse
        # Each pixel should be attended by mostly one slot
        if symbolic.attention_maps is not None:
            attention = symbolic.attention_maps  # [num_slots, H*W]
            # Entropy per pixel (lower = more specialized)
            attn_probs = F.softmax(attention, dim=0)  # Softmax over slots
            entropy_per_pixel = -(attn_probs * torch.log(attn_probs + 1e-8)).sum(dim=0)
            losses["attention_entropy"] = entropy_per_pixel.mean()

        # Disentanglement: property predictions should be independent
        # Use correlation penalty
        positions = self.position_decoder(slots)
        colors = self.color_decoder(slots)

        combined = torch.cat([positions, colors], dim=-1)  # [B, num_slots, 14]
        tc_loss = self._total_correlation(combined)
        losses["disentanglement"] = tc_loss

        # Ground truth supervision (if available)
        if ground_truth is not None:
            # Position loss
            if len(ground_truth.objects) > 0:
                gt_positions = ground_truth.object_positions
                pred_positions = positions[0, : len(ground_truth.objects)]
                if pred_positions.shape[0] == gt_positions.shape[0]:
                    losses["position"] = F.mse_loss(pred_positions, gt_positions)

                # Color classification loss
                gt_colors = ground_truth.object_colors
                pred_colors = colors[0, : len(ground_truth.objects)]
                if pred_colors.shape[0] == gt_colors.shape[0]:
                    losses["color"] = F.cross_entropy(pred_colors, gt_colors)

        return losses

    def _total_correlation(self, z: torch.Tensor) -> torch.Tensor:
        """
        Estimate total correlation for disentanglement.

        Total correlation measures how much the dimensions
        depend on each other. Lower = more disentangled.
        """
        # Flatten batch and slots
        z_flat = z.reshape(-1, z.shape[-1])

        if z_flat.shape[0] < 2:
            return torch.tensor(0.0, device=z.device)

        # Compute correlation matrix
        z_centered = z_flat - z_flat.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1)

        # Normalize to correlation
        std = z_flat.std(dim=0) + 1e-8
        corr = cov / (std.unsqueeze(0) * std.unsqueeze(1))

        # Off-diagonal elements indicate entanglement
        diag_mask = torch.eye(corr.shape[0], device=corr.device).bool()
        off_diag = corr[~diag_mask]

        return off_diag.abs().mean()


@dataclass
class Rule:
    """Extracted symbolic rule."""

    preconditions: dict
    action: int
    effects: frozenset
    confidence: float
    support: int


class RuleExtractor:
    """
    Extracts symbolic rules from learned dynamics.

    Rules are of the form:
    IF precondition(s, a) THEN effect(s')

    These can be used for:
    1. Interpretability (explain agent behavior)
    2. Transfer (apply rules to new situations)
    3. Planning (symbolic search with rules)
    """

    def __init__(
        self,
        world_model: nn.Module,
        symbolic_grounding: SymbolicGrounding,
        min_confidence: float = 0.8,
        min_support: int = 10,
    ):
        self.world_model = world_model
        self.symbolic_grounding = symbolic_grounding
        self.min_confidence = min_confidence
        self.min_support = min_support

        self.extracted_rules: list[Rule] = []

    def extract_rules(
        self,
        trajectories: list,
    ) -> list[Rule]:
        """
        Extract rules from observed trajectories.

        Algorithm:
        1. For each (s, a, s') transition, extract symbolic representation
        2. Group by action type
        3. Find consistent precondition -> effect patterns
        4. Filter by confidence and support
        """
        # Collect symbolic transitions
        symbolic_transitions = []

        for traj in trajectories:
            for i in range(len(traj.states) - 1):
                state = traj.states[i]
                next_state = traj.states[i + 1]
                action = traj.actions[i]

                # Extract symbolic states
                _, sym_s = self.symbolic_grounding(state.unsqueeze(0))
                _, sym_s_next = self.symbolic_grounding(next_state.unsqueeze(0))

                symbolic_transitions.append(
                    {
                        "state": sym_s,
                        "action": action,
                        "next_state": sym_s_next,
                        "effect": self._compute_effect(sym_s, sym_s_next),
                    }
                )

        # Group by action
        by_action: dict[int, list] = defaultdict(list)
        for trans in symbolic_transitions:
            by_action[trans["action"]].append(trans)

        # Extract rules for each action
        rules = []
        for action, transitions in by_action.items():
            action_rules = self._extract_action_rules(action, transitions)
            rules.extend(action_rules)

        self.extracted_rules = rules
        return rules

    def _compute_effect(
        self,
        state: SymbolicState,
        next_state: SymbolicState,
    ) -> dict:
        """Compute what changed between states."""
        effect = {}

        # Agent position change
        if state.agent_position != next_state.agent_position:
            effect["agent_moved"] = {
                "from": state.agent_position,
                "to": next_state.agent_position,
            }

        # Object changes
        old_ids = state.object_ids
        new_ids = next_state.object_ids

        # New objects
        for obj_id in new_ids - old_ids:
            obj = next_state.get_object(obj_id)
            if obj:
                effect[f"object_{obj_id}_appeared"] = obj.color

        # Removed objects
        for obj_id in old_ids - new_ids:
            obj = state.get_object(obj_id)
            if obj:
                effect[f"object_{obj_id}_disappeared"] = obj.color

        # Changed objects
        for obj_id in old_ids & new_ids:
            obj_before = state.get_object(obj_id)
            obj_after = next_state.get_object(obj_id)
            if obj_before and obj_after:
                if obj_before.state != obj_after.state:
                    effect[f"object_{obj_id}_state_changed"] = {
                        "from": obj_before.state,
                        "to": obj_after.state,
                    }
                if obj_before.position != obj_after.position:
                    effect[f"object_{obj_id}_moved"] = {
                        "from": obj_before.position,
                        "to": obj_after.position,
                    }

        return effect

    def _extract_action_rules(
        self,
        action: int,
        transitions: list[dict],
    ) -> list[Rule]:
        """Extract rules for a specific action."""
        # Group by effect
        by_effect: dict[frozenset, list] = defaultdict(list)
        for trans in transitions:
            effect_key = frozenset(trans["effect"].keys())
            by_effect[effect_key].append(trans)

        rules = []
        for effect_key, effect_trans in by_effect.items():
            if len(effect_trans) < self.min_support:
                continue

            # Find common preconditions
            preconditions = self._find_common_preconditions(effect_trans)

            # Compute confidence
            confidence = self._compute_rule_confidence(
                preconditions, action, effect_key, transitions
            )

            if confidence >= self.min_confidence:
                rule = Rule(
                    preconditions=preconditions,
                    action=action,
                    effects=effect_key,
                    confidence=confidence,
                    support=len(effect_trans),
                )
                rules.append(rule)

        return rules

    def _find_common_preconditions(
        self,
        transitions: list[dict],
    ) -> dict:
        """Find common state properties across transitions."""
        if not transitions:
            return {}

        # Simple: check if agent was near any object
        preconditions = {}

        # Count how often agent was adjacent to objects
        adjacency_counts: dict[int, int] = defaultdict(int)
        for trans in transitions:
            state = trans["state"]
            if state.agent_position:
                for obj in state.objects:
                    dist = abs(state.agent_position[0] - obj.position[0]) + abs(
                        state.agent_position[1] - obj.position[1]
                    )
                    if dist <= 1:
                        adjacency_counts[obj.color] += 1

        # If agent was often adjacent to same color object, that's a precondition
        for color, count in adjacency_counts.items():
            if count >= len(transitions) * 0.8:  # 80% of transitions
                preconditions["adjacent_to_color"] = color

        return preconditions

    def _compute_rule_confidence(
        self,
        preconditions: dict,
        action: int,
        effects: frozenset,
        all_transitions: list[dict],
    ) -> float:
        """Compute confidence of a rule."""
        # Count transitions matching preconditions
        matching_prec = 0
        matching_effect = 0

        for trans in all_transitions:
            if trans["action"] != action:
                continue

            # Check preconditions
            prec_match = True
            if "adjacent_to_color" in preconditions:
                state = trans["state"]
                found_adj = False
                if state.agent_position:
                    for obj in state.objects:
                        if obj.color == preconditions["adjacent_to_color"]:
                            dist = abs(state.agent_position[0] - obj.position[0]) + abs(
                                state.agent_position[1] - obj.position[1]
                            )
                            if dist <= 1:
                                found_adj = True
                                break
                prec_match = found_adj

            if prec_match:
                matching_prec += 1
                if frozenset(trans["effect"].keys()) == effects:
                    matching_effect += 1

        if matching_prec == 0:
            return 0.0

        return matching_effect / matching_prec
