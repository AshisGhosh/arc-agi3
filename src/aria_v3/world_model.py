"""
Transformer-based world model for online Dreamer agent.

Architecture:
    Encoder: 64 VQ code tokens → Transformer → hidden states
    Dynamics: FiLM(hidden, action) → predict next 64 VQ codes
    Value: mean(hidden) → MLP → scalar V(state) ∈ [0, 1]
    Policy: mean(hidden) → MLP → action logits

The encoder processes the current frame's VQ codes.
The dynamics head predicts next-frame VQ codes conditioned on an action.
Value and policy heads read from the frame representation (no action).

~2.5M parameters. ~2ms inference on GPU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMConditioner(nn.Module):
    """Feature-wise Linear Modulation: conditions hidden states on action."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.film = nn.Linear(action_dim, 2 * hidden_dim)

    def forward(self, h: torch.Tensor, action_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, 64, hidden_dim] frame hidden states
            action_embed: [B, action_dim] action embedding

        Returns:
            [B, 64, hidden_dim] conditioned hidden states
        """
        params = self.film(action_embed)  # [B, 2*hidden]
        gamma, beta = params.chunk(2, dim=-1)  # each [B, hidden]
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


class DreamerWorldModel(nn.Module):
    """Small Transformer world model with dynamics, value, and policy heads.

    Processes 64 VQ code tokens through a Transformer encoder.
    Dynamics: FiLM-conditioned prediction of next VQ codes.
    Value: scalar estimate of state quality.
    Policy: action distribution for the learned planner.
    """

    def __init__(
        self,
        num_vq_codes: int = 512,
        num_positions: int = 64,
        num_action_types: int = 8,
        num_action_locs: int = 65,  # 64 VQ cells + NULL
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
        num_simple_actions: int = 5,  # actions 1-5
        num_click_positions: int = 64,  # 8x8 VQ grid
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_vq_codes = num_vq_codes
        self.num_positions = num_positions
        self.num_simple_actions = num_simple_actions
        self.num_click_positions = num_click_positions
        self.total_policy_actions = num_simple_actions + num_click_positions  # 69

        # Embeddings
        self.vq_embed = nn.Embedding(num_vq_codes, hidden_dim)
        self.pos_embed = nn.Embedding(num_positions, hidden_dim)
        self.action_type_embed = nn.Embedding(num_action_types, hidden_dim)
        self.action_loc_embed = nn.Embedding(num_action_locs, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Dynamics head: FiLM conditioning + prediction
        self.action_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # combine type + loc
        self.film = FiLMConditioner(hidden_dim, hidden_dim)
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, num_vq_codes),
        )

        # Value head: V(state) ∈ [0, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Policy head: action logits
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.total_policy_actions),
        )

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, vq_codes: torch.Tensor) -> torch.Tensor:
        """Encode VQ codes through Transformer.

        Args:
            vq_codes: [B, 64] long tensor of VQ code indices (0-511)

        Returns:
            [B, 64, hidden_dim] hidden states
        """
        B, T = vq_codes.shape
        positions = torch.arange(T, device=vq_codes.device).unsqueeze(0).expand(B, -1)

        x = self.vq_embed(vq_codes) + self.pos_embed(positions)
        h = self.encoder(x)  # [B, 64, hidden_dim]
        return h

    def predict_dynamics(
        self,
        h: torch.Tensor,
        action_type: torch.Tensor,
        action_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Predict next frame VQ codes given current hidden states and action.

        Args:
            h: [B, 64, hidden_dim] from encode()
            action_type: [B] long tensor of action type indices (0-7)
            action_loc: [B] long tensor of action location indices (0-64)

        Returns:
            [B, 64, num_vq_codes] logits over next VQ codes per position
        """
        type_emb = self.action_type_embed(action_type)   # [B, hidden]
        loc_emb = self.action_loc_embed(action_loc)       # [B, hidden]
        action_emb = self.action_proj(torch.cat([type_emb, loc_emb], dim=-1))  # [B, hidden]

        h_cond = self.film(h, action_emb)  # [B, 64, hidden]
        return self.dynamics_head(h_cond)   # [B, 64, num_vq_codes]

    def predict_value(self, h: torch.Tensor) -> torch.Tensor:
        """Predict state value.

        Args:
            h: [B, 64, hidden_dim] from encode()

        Returns:
            [B] value predictions
        """
        pooled = h.mean(dim=1)  # [B, hidden]
        return self.value_head(pooled).squeeze(-1)  # [B]

    def predict_policy(self, h: torch.Tensor) -> torch.Tensor:
        """Predict action logits.

        Args:
            h: [B, 64, hidden_dim] from encode()

        Returns:
            [B, total_policy_actions] logits (5 simple + 64 click positions)
        """
        pooled = h.mean(dim=1)  # [B, hidden]
        return self.policy_head(pooled)  # [B, 69]

    def forward_all(
        self,
        vq_codes: torch.Tensor,
        action_type: torch.Tensor | None = None,
        action_loc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode, then compute all heads.

        Args:
            vq_codes: [B, 64] VQ code indices
            action_type: [B] action type (optional, needed for dynamics)
            action_loc: [B] action location (optional, needed for dynamics)

        Returns:
            Dict with 'hidden', 'value', 'policy_logits', and optionally 'dynamics_logits'
        """
        h = self.encode(vq_codes)
        result = {
            "hidden": h,
            "value": self.predict_value(h),
            "policy_logits": self.predict_policy(h),
        }

        if action_type is not None and action_loc is not None:
            result["dynamics_logits"] = self.predict_dynamics(h, action_type, action_loc)

        return result

    @torch.no_grad()
    def imagine(
        self,
        initial_vq: torch.Tensor,
        horizon: int = 20,
        num_trajectories: int = 16,
        available_actions: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Imagine rollouts from initial state using dynamics + policy.

        Args:
            initial_vq: [64] VQ codes of the starting state
            horizon: number of steps to imagine
            num_trajectories: number of parallel rollouts
            available_actions: [total_policy_actions] bool mask of available actions
            temperature: sampling temperature for policy and dynamics

        Returns:
            Dict with:
                'states': [K, H+1, 64] VQ codes at each step
                'actions': [K, H] action indices taken
                'action_types': [K, H] action type IDs
                'action_locs': [K, H] action location IDs
                'values': [K, H+1] value estimates
        """
        K, H = num_trajectories, horizon
        device = initial_vq.device

        # Expand initial state: [64] → [K, 64]
        z = initial_vq.unsqueeze(0).expand(K, -1).clone()

        states = [z.clone()]
        actions = []
        action_types_list = []
        action_locs_list = []
        values = []

        for t in range(H):
            h = self.encode(z)

            # Value
            v = self.predict_value(h)  # [K]
            values.append(v)

            # Policy → sample action
            logits = self.predict_policy(h)  # [K, 69]
            if available_actions is not None:
                logits[:, ~available_actions] = float("-inf")

            probs = F.softmax(logits / temperature, dim=-1).clamp(min=0.0)
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            action_idx = torch.multinomial(probs, 1).squeeze(-1)  # [K]
            actions.append(action_idx)

            # Convert action index to type + location
            a_type, a_loc = self._action_idx_to_type_loc(action_idx)
            action_types_list.append(a_type)
            action_locs_list.append(a_loc)

            # Dynamics → predict next state
            dyn_logits = self.predict_dynamics(h, a_type, a_loc)  # [K, 64, 512]
            next_probs = F.softmax(dyn_logits / temperature, dim=-1)

            # Sample next VQ codes
            B, T, C = next_probs.shape
            # Clamp to avoid NaN in multinomial
            flat_probs = next_probs.reshape(B * T, C).clamp(min=0.0)
            flat_probs = flat_probs / flat_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            z = torch.multinomial(flat_probs, 1).reshape(B, T)

            states.append(z.clone())

        # Final value
        h_final = self.encode(z)
        values.append(self.predict_value(h_final))

        return {
            "states": torch.stack(states, dim=1),      # [K, H+1, 64]
            "actions": torch.stack(actions, dim=1),      # [K, H]
            "action_types": torch.stack(action_types_list, dim=1),  # [K, H]
            "action_locs": torch.stack(action_locs_list, dim=1),    # [K, H]
            "values": torch.stack(values, dim=1),        # [K, H+1]
        }

    def _action_idx_to_type_loc(
        self, action_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert unified action index to (type, location).

        Actions 0..4 → simple actions (types 1-5), location = NULL (64)
        Actions 5..68 → click (type 6), location = action_idx - 5

        Args:
            action_idx: [B] indices in unified space

        Returns:
            (action_type, action_loc) each [B]
        """
        device = action_idx.device
        B = action_idx.shape[0]

        is_click = action_idx >= self.num_simple_actions

        # Simple actions: type = idx + 1 (game API is 1-indexed)
        action_type = torch.where(
            is_click,
            torch.tensor(6, device=device).expand(B),
            action_idx + 1,
        )

        # Location: click → VQ cell index, simple → NULL (64)
        action_loc = torch.where(
            is_click,
            action_idx - self.num_simple_actions,
            torch.tensor(64, device=device).expand(B),
        )

        return action_type, action_loc

    def action_type_loc_to_idx(
        self, action_type: int, action_loc: int
    ) -> int:
        """Convert (type, location) to unified action index.

        Inverse of _action_idx_to_type_loc.
        """
        if action_type == 6:
            return self.num_simple_actions + action_loc
        else:
            return action_type - 1  # 1-indexed → 0-indexed
