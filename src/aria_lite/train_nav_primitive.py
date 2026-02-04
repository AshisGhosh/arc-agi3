"""
Focused training script for navigation primitive.

Uses simpler action space (no coordinates) to validate learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.aria_lite.primitives import NavigationEnv, NavigationPrimitiveGenerator
from src.aria_lite.primitives.base import Action  # noqa: F401
from src.aria_lite.primitives.base import Action


class SimpleNavEncoder(nn.Module):
    """
    Simple encoder for navigation grids.

    Uses position-aware encoding: cell value + 2D position.
    """

    def __init__(self, hidden_dim: int = 128, max_values: int = 16, max_grid: int = 20):
        super().__init__()
        self.max_grid = max_grid

        # Cell value embedding
        self.cell_embed = nn.Embedding(max_values, 16)

        # Position embeddings
        self.pos_y_embed = nn.Embedding(max_grid, 8)
        self.pos_x_embed = nn.Embedding(max_grid, 8)

        # Input: 16 (cell) + 8 (y) + 8 (x) = 32 channels
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc = nn.Sequential(
            nn.Linear(128 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.shape
        x = x.long().clamp(0, 15)

        # Cell embeddings
        cell = self.cell_embed(x)  # [B, H, W, 16]

        # Position embeddings
        y_indices = torch.arange(H, device=x.device).unsqueeze(1).expand(H, W)
        x_indices = torch.arange(W, device=x.device).unsqueeze(0).expand(H, W)

        pos_y = self.pos_y_embed(y_indices)  # [H, W, 8]
        pos_x = self.pos_x_embed(x_indices)  # [H, W, 8]

        # Expand position to batch
        pos_y = pos_y.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 8]
        pos_x = pos_x.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 8]

        # Concatenate: [B, H, W, 32]
        features = torch.cat([cell, pos_y, pos_x], dim=-1)
        features = features.permute(0, 3, 1, 2)  # [B, 32, H, W]

        # Conv
        features = F.relu(self.conv1(features))
        features = F.relu(self.conv2(features))
        features = F.relu(self.conv3(features))

        # Pool
        features = self.pool(features)
        features = features.reshape(B, -1)

        return self.fc(features)


class NavActorCritic(nn.Module):
    """Actor-Critic for navigation (5 actions: NOOP, UP, DOWN, LEFT, RIGHT)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.encoder = SimpleNavEncoder(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 5)  # 5 actions only
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor):
        features = self.encoder(obs)
        action_logits = self.action_head(features)
        value = self.value_head(features)
        return action_logits, value

    def get_action(self, obs: torch.Tensor):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1), dist.entropy()


class VecNavEnv:
    """Vectorized navigation environment."""

    def __init__(self, num_envs: int, difficulty: int = 1):
        self.num_envs = num_envs
        self.generator = NavigationPrimitiveGenerator()
        self.envs = [self.generator.generate(difficulty=difficulty) for _ in range(num_envs)]
        self.grid_size = self.envs[0].grid_size

    def reset(self) -> torch.Tensor:
        return torch.stack([env.reset() for env in self.envs])

    def step(self, actions: torch.Tensor):
        obs_list, rewards, dones, infos = [], [], [], []

        for i, env in enumerate(self.envs):
            result = env.step(actions[i].item())
            obs = result.observation
            if result.done:
                infos.append({"success": result.success, **result.info})
                obs = env.reset()
            else:
                infos.append(result.info)

            obs_list.append(obs)
            rewards.append(result.reward)
            dones.append(result.done)

        return (
            torch.stack(obs_list),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(dones, dtype=torch.bool),
            infos,
        )


def train_navigation(
    num_envs: int = 8,
    rollout_steps: int = 64,
    num_updates: int = 500,
    difficulty: int = 1,
):
    """Train navigation primitive."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    vec_env = VecNavEnv(num_envs, difficulty)
    model = NavActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    obs = vec_env.reset().to(device)
    successes = []

    for update in range(num_updates):
        # Collect rollout
        obs_batch, actions_batch, log_probs_batch = [], [], []
        values_batch, rewards_batch, dones_batch, entropies_batch = [], [], [], []

        for _ in range(rollout_steps):
            action, log_prob, value, entropy = model.get_action(obs)

            obs_batch.append(obs)
            actions_batch.append(action)
            log_probs_batch.append(log_prob)
            values_batch.append(value)
            entropies_batch.append(entropy)

            # Step
            obs, rewards, dones, infos = vec_env.step(action.cpu())
            obs = obs.to(device)

            rewards_batch.append(rewards.to(device))
            dones_batch.append(dones.to(device))

            for info in infos:
                if "success" in info:
                    successes.append(1.0 if info["success"] else 0.0)

        # Compute returns with GAE
        with torch.no_grad():
            _, last_value = model.forward(obs)
            last_value = last_value.squeeze(-1)

        gamma, gae_lambda = 0.99, 0.95
        advantages, returns = [], []
        gae = torch.zeros(num_envs, device=device)
        next_value = last_value

        for t in reversed(range(rollout_steps)):
            mask = (~dones_batch[t]).float()
            delta = rewards_batch[t] + gamma * next_value * mask - values_batch[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values_batch[t])
            next_value = values_batch[t]

        # Flatten
        obs_flat = torch.cat(obs_batch, dim=0)
        actions_flat = torch.cat(actions_batch, dim=0)
        old_log_probs = torch.cat(log_probs_batch, dim=0)
        advantages_flat = torch.cat(advantages, dim=0)
        returns_flat = torch.cat(returns, dim=0)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # PPO update
        logits, values = model.forward(obs_flat)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_flat)
        entropy = dist.entropy()

        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_flat
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns_flat)

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        if update % 50 == 0:
            success_rate = sum(successes[-100:]) / max(len(successes[-100:]), 1)
            avg_reward = torch.stack(rewards_batch).mean().item()
            print(f"Update {update}: reward={avg_reward:.3f}, success={success_rate:.1%}")

    # Final eval
    eval_successes = []
    for _ in range(100):
        obs = vec_env.reset().to(device)
        for _ in range(vec_env.envs[0].max_steps):
            with torch.no_grad():
                action, _, _, _ = model.get_action(obs)
            obs, _, dones, infos = vec_env.step(action.cpu())
            obs = obs.to(device)
            for info in infos:
                if "success" in info:
                    eval_successes.append(1.0 if info["success"] else 0.0)
            if dones.all():
                break

    final_success = sum(eval_successes) / max(len(eval_successes), 1)
    print(f"\nFinal eval success: {final_success:.1%}")
    return final_success


if __name__ == "__main__":
    train_navigation(num_updates=300, difficulty=1)
