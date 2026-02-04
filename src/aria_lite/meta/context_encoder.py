"""
Context Encoder for Meta-Learning

Enables the policy to condition on:
1. Task demonstrations (few-shot learning)
2. Task descriptions (language-conditioned)
3. Task embeddings (learned task representations)

This allows generalization to new tasks from limited examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DemonstrationEncoder(nn.Module):
    """
    Encode a sequence of (observation, action) pairs from demonstrations.

    Architecture:
    1. Encode each (obs, action) pair
    2. Aggregate with attention over demonstration steps
    3. Output fixed-size task embedding
    """

    def __init__(
        self,
        obs_dim: int = 128,
        action_dim: int = 9,
        hidden_dim: int = 128,
        output_dim: int = 64,
        max_demo_length: int = 20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Action embedding
        self.action_embed = nn.Embedding(action_dim, 16)

        # Combine obs features + action
        self.step_encoder = nn.Sequential(
            nn.Linear(obs_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Position encoding for demo steps
        self.pos_embed = nn.Embedding(max_demo_length, hidden_dim)

        # Attention to aggregate demo steps
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable query for aggregation
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))

    def forward(
        self,
        obs_features: torch.Tensor,
        actions: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode demonstration sequence.

        Args:
            obs_features: [B, T, obs_dim] observation features from encoder
            actions: [B, T] action indices
            mask: [B, T] bool mask (True = valid step)

        Returns:
            task_embedding: [B, output_dim]
        """
        B, T, _ = obs_features.shape
        device = obs_features.device

        # Encode actions
        action_emb = self.action_embed(actions)  # [B, T, 16]

        # Combine obs + action
        combined = torch.cat([obs_features, action_emb], dim=-1)
        step_features = self.step_encoder(combined)  # [B, T, hidden_dim]

        # Add position encoding
        positions = torch.arange(T, device=device)
        pos_emb = self.pos_embed(positions).unsqueeze(0)  # [1, T, hidden_dim]
        step_features = step_features + pos_emb

        # Attention: query attends to all demo steps
        query = self.query.expand(B, -1, -1)  # [B, 1, hidden_dim]

        # Create attention mask from padding mask
        attn_mask = None
        if mask is not None:
            attn_mask = ~mask  # Invert: True = ignore

        attended, _ = self.attention(
            query, step_features, step_features,
            key_padding_mask=attn_mask,
        )  # [B, 1, hidden_dim]

        # Project to output
        task_embedding = self.output_proj(attended.squeeze(1))  # [B, output_dim]

        return task_embedding


class TaskConditionedPolicy(nn.Module):
    """
    Policy conditioned on task embedding from demonstrations.

    The task embedding modulates the policy via:
    1. FiLM conditioning (scale + shift)
    2. Concatenation to features
    """

    def __init__(
        self,
        obs_encoder: nn.Module,
        task_dim: int = 64,
        hidden_dim: int = 128,
        num_actions: int = 9,
        max_grid: int = 20,
    ):
        super().__init__()
        self.obs_encoder = obs_encoder
        self.task_dim = task_dim

        # FiLM: generate scale and shift from task embedding
        self.film = nn.Sequential(
            nn.Linear(task_dim, hidden_dim * 2),
        )

        # Policy heads
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.x_head = nn.Linear(hidden_dim, max_grid)
        self.y_head = nn.Linear(hidden_dim, max_grid)

    def forward(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        grid_size: int = 10,
    ) -> dict:
        """
        Forward pass with task conditioning.

        Args:
            obs: [B, H, W] observation
            task_embedding: [B, task_dim] from demonstration encoder
            grid_size: for coordinate output masking

        Returns:
            dict with action_logits, x_logits, y_logits
        """
        # Encode observation
        obs_features = self.obs_encoder(obs)  # [B, hidden_dim]

        # FiLM conditioning
        film_params = self.film(task_embedding)  # [B, hidden_dim * 2]
        scale, shift = film_params.chunk(2, dim=-1)
        conditioned = obs_features * (1 + scale) + shift  # [B, hidden_dim]

        # Policy outputs
        action_logits = self.action_head(conditioned)
        x_logits = self.x_head(conditioned)[:, :grid_size]
        y_logits = self.y_head(conditioned)[:, :grid_size]

        return {
            "action_logits": action_logits,
            "x_logits": x_logits,
            "y_logits": y_logits,
            "features": conditioned,
        }


class MetaLearningAgent(nn.Module):
    """
    Full meta-learning agent combining:
    1. Observation encoder
    2. Demonstration encoder
    3. Task-conditioned policy

    Usage:
    1. Given K demonstrations, encode to task embedding
    2. Use task embedding to condition policy
    3. Act on new observations from same task
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        task_dim: int = 64,
        num_colors: int = 16,
        num_actions: int = 9,
        max_grid: int = 20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.task_dim = task_dim

        # Observation encoder (shared)
        self.obs_encoder = ObservationEncoder(
            num_colors=num_colors,
            hidden_dim=hidden_dim,
            max_grid=max_grid,
        )

        # Demonstration encoder
        self.demo_encoder = DemonstrationEncoder(
            obs_dim=hidden_dim,
            action_dim=num_actions,
            hidden_dim=hidden_dim,
            output_dim=task_dim,
        )

        # Task-conditioned policy
        self.policy = TaskConditionedPolicy(
            obs_encoder=self.obs_encoder,
            task_dim=task_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            max_grid=max_grid,
        )

    def encode_demonstrations(
        self,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Encode demonstrations to task embedding.

        Args:
            demo_obs: [B, K, H, W] K demonstration observations
            demo_actions: [B, K] K demonstration actions
            demo_mask: [B, K] valid steps mask

        Returns:
            task_embedding: [B, task_dim]
        """
        B, K, H, W = demo_obs.shape

        # Encode each demo observation
        demo_obs_flat = demo_obs.view(B * K, H, W)
        demo_features_flat = self.obs_encoder(demo_obs_flat)  # [B*K, hidden_dim]
        demo_features = demo_features_flat.view(B, K, -1)  # [B, K, hidden_dim]

        # Encode demonstration sequence
        task_embedding = self.demo_encoder(
            demo_features, demo_actions, demo_mask
        )

        return task_embedding

    def forward(
        self,
        obs: torch.Tensor,
        task_embedding: torch.Tensor,
        grid_size: int = 10,
    ) -> dict:
        """
        Forward pass with pre-computed task embedding.
        """
        return self.policy(obs, task_embedding, grid_size)

    def act(
        self,
        obs: torch.Tensor,
        demo_obs: torch.Tensor,
        demo_actions: torch.Tensor,
        demo_mask: torch.Tensor = None,
        grid_size: int = 10,
    ) -> dict:
        """
        Full forward: encode demos then act.

        Args:
            obs: [B, H, W] current observation
            demo_obs: [B, K, H, W] K demonstration observations
            demo_actions: [B, K] K demonstration actions
        """
        task_embedding = self.encode_demonstrations(demo_obs, demo_actions, demo_mask)
        return self.forward(obs, task_embedding, grid_size)


class ObservationEncoder(nn.Module):
    """Simple observation encoder for the meta-learning agent."""

    def __init__(
        self,
        num_colors: int = 16,
        hidden_dim: int = 128,
        max_grid: int = 20,
    ):
        super().__init__()

        self.embed = nn.Embedding(num_colors, 32)
        self.pos_y = nn.Embedding(max_grid, 8)
        self.pos_x = nn.Embedding(max_grid, 8)

        self.conv = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B, H, W = obs.shape
        device = obs.device

        # Embed cells
        cell = self.embed(obs.long().clamp(0, 15))

        # Position embeddings
        y_idx = torch.arange(H, device=device)
        x_idx = torch.arange(W, device=device)
        pos_y = self.pos_y(y_idx).unsqueeze(1).expand(H, W, -1)
        pos_x = self.pos_x(x_idx).unsqueeze(0).expand(H, W, -1)
        pos = torch.cat([pos_y, pos_x], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Combine
        features = torch.cat([cell, pos], dim=-1)
        features = features.permute(0, 3, 1, 2)

        features = self.conv(features)
        features = features.reshape(B, -1)
        return self.fc(features)
