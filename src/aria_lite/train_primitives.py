"""
Training script for primitive task families.

Tests whether the architecture can learn each primitive type.
"""

import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from src.aria_lite.primitives import (
    PrimitiveFamily,
    PrimitiveGenerator,
    NavigationEnv,
    ClickEnv,
)
from src.aria_lite.primitives.base import Action


@dataclass
class TrainConfig:
    """Training configuration."""
    num_envs: int = 8
    rollout_steps: int = 64
    num_updates: int = 300
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128


class PrimitiveEncoder(nn.Module):
    """
    Simple encoder for primitive grids.

    Handles variable grid sizes by adaptive pooling.
    """

    def __init__(self, hidden_dim: int = 128, max_values: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(max_values, 32)

        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Linear(128 * 16, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W] or [B, C, H, W]
        if x.dim() == 3:
            x = x.long()
        else:
            x = x.long()

        # Handle non-square inputs
        if x.dim() == 3:
            B, H, W = x.shape
        else:
            B, _, H, W = x.shape
            x = x[:, 0]  # Take first channel

        # Clamp values to valid range
        x = x.clamp(0, 15)

        # Embed
        embedded = self.embedding(x)  # [B, H, W, 32]
        embedded = embedded.permute(0, 3, 1, 2)  # [B, 32, H, W]

        # Conv
        features = self.conv(embedded)  # [B, 128, 4, 4]
        features = features.reshape(B, -1)  # [B, 128*16]

        return self.fc(features)  # [B, hidden_dim]


class PrimitiveActorCritic(nn.Module):
    """
    Actor-Critic for primitive tasks.

    Supports both discrete actions and coordinate prediction.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_actions: int = 9,
        max_grid_size: int = 20,
    ):
        super().__init__()
        self.encoder = PrimitiveEncoder(hidden_dim)

        # Actor heads
        self.action_head = nn.Linear(hidden_dim, num_actions)
        self.coord_x_head = nn.Linear(hidden_dim, max_grid_size)
        self.coord_y_head = nn.Linear(hidden_dim, max_grid_size)

        # Critic
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, grid_size: int = 10):
        features = self.encoder(obs)

        # Action logits
        action_logits = self.action_head(features)

        # Coordinate logits (masked to valid range)
        x_logits = self.coord_x_head(features)[:, :grid_size]
        y_logits = self.coord_y_head(features)[:, :grid_size]

        # Value
        value = self.value_head(features)

        return {
            "action_logits": action_logits,
            "x_logits": x_logits,
            "y_logits": y_logits,
            "value": value,
            "features": features,
        }

    def get_action(self, obs: torch.Tensor, grid_size: int = 10):
        out = self.forward(obs, grid_size)

        # Sample action
        action_dist = Categorical(logits=out["action_logits"])
        action = action_dist.sample()

        # Sample coordinates
        x_dist = Categorical(logits=out["x_logits"])
        y_dist = Categorical(logits=out["y_logits"])
        x = x_dist.sample()
        y = y_dist.sample()

        # Log probs
        action_log_prob = action_dist.log_prob(action)
        x_log_prob = x_dist.log_prob(x)
        y_log_prob = y_dist.log_prob(y)

        return {
            "action": action,
            "x": x,
            "y": y,
            "action_log_prob": action_log_prob,
            "x_log_prob": x_log_prob,
            "y_log_prob": y_log_prob,
            "value": out["value"].squeeze(-1),
            "entropy": action_dist.entropy() + x_dist.entropy() + y_dist.entropy(),
        }


class VecPrimitiveEnv:
    """Vectorized primitive environment."""

    def __init__(
        self,
        num_envs: int,
        family: PrimitiveFamily,
        difficulty: int = 1,
    ):
        self.num_envs = num_envs
        self.family = family
        self.difficulty = difficulty
        self.generator = PrimitiveGenerator()

        self.envs = [
            self.generator.generate(family=family, difficulty=difficulty)
            for _ in range(num_envs)
        ]

        # Track grid size (assume uniform)
        self.grid_size = self.envs[0].grid_size

    def reset(self) -> torch.Tensor:
        obs = []
        for env in self.envs:
            o = env.reset()
            obs.append(o)
        return self._stack_obs(obs)

    def step(self, actions, xs, ys):
        obs, rewards, dones, infos = [], [], [], []

        for i, env in enumerate(self.envs):
            action = actions[i].item()
            x = xs[i].item() if xs is not None else None
            y = ys[i].item() if ys is not None else None

            result = env.step(action, x, y)
            obs.append(result.observation)
            rewards.append(result.reward)
            dones.append(result.done)
            infos.append(result.info)

            if result.done:
                # Reset env
                obs[-1] = env.reset()
                infos[-1]["success"] = result.success

        return (
            self._stack_obs(obs),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(dones, dtype=torch.bool),
            infos,
        )

    def _stack_obs(self, obs_list):
        # Pad to same size if needed
        max_h = max(o.shape[0] for o in obs_list)
        max_w = max(o.shape[1] for o in obs_list)

        padded = []
        for o in obs_list:
            if o.shape[0] < max_h or o.shape[1] < max_w:
                p = torch.zeros(max_h, max_w, dtype=o.dtype)
                p[:o.shape[0], :o.shape[1]] = o
                padded.append(p)
            else:
                padded.append(o)

        return torch.stack(padded)


def train_primitive(
    family: PrimitiveFamily,
    difficulty: int = 1,
    config: Optional[TrainConfig] = None,
) -> dict:
    """Train on a single primitive family."""
    config = config or TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environments
    vec_env = VecPrimitiveEnv(config.num_envs, family, difficulty)

    # Create model
    model = PrimitiveActorCritic(
        hidden_dim=config.hidden_dim,
        max_grid_size=20,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    obs = vec_env.reset().to(device)
    successes = []
    rewards_history = []

    for update in range(config.num_updates):
        # Collect rollout
        rollout_obs = []
        rollout_actions = []
        rollout_xs = []
        rollout_ys = []
        rollout_log_probs = []
        rollout_values = []
        rollout_rewards = []
        rollout_dones = []
        rollout_entropies = []

        for step in range(config.rollout_steps):
            with torch.no_grad():
                out = model.get_action(obs, grid_size=vec_env.grid_size)

            rollout_obs.append(obs)
            rollout_actions.append(out["action"])
            rollout_xs.append(out["x"])
            rollout_ys.append(out["y"])
            rollout_log_probs.append(
                out["action_log_prob"] + out["x_log_prob"] + out["y_log_prob"]
            )
            rollout_values.append(out["value"])
            rollout_entropies.append(out["entropy"])

            # Step environment
            obs, rewards, dones, infos = vec_env.step(
                out["action"].cpu(),
                out["x"].cpu(),
                out["y"].cpu(),
            )
            obs = obs.to(device)

            rollout_rewards.append(rewards.to(device))
            rollout_dones.append(dones.to(device))

            # Track successes
            for info in infos:
                if "success" in info:
                    successes.append(1.0 if info["success"] else 0.0)

        # Compute advantages (GAE)
        with torch.no_grad():
            last_value = model.forward(obs, grid_size=vec_env.grid_size)["value"].squeeze(-1)

        advantages = []
        returns = []
        gae = torch.zeros(config.num_envs, device=device)
        next_value = last_value

        for t in reversed(range(config.rollout_steps)):
            mask = (~rollout_dones[t]).float()
            delta = rollout_rewards[t] + config.gamma * next_value * mask - rollout_values[t]
            gae = delta + config.gamma * config.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + rollout_values[t])
            next_value = rollout_values[t]

        # Flatten
        obs_flat = torch.cat(rollout_obs, dim=0)
        actions_flat = torch.cat(rollout_actions, dim=0)
        xs_flat = torch.cat(rollout_xs, dim=0)
        ys_flat = torch.cat(rollout_ys, dim=0)
        old_log_probs = torch.cat(rollout_log_probs, dim=0)
        advantages_flat = torch.cat(advantages, dim=0)
        returns_flat = torch.cat(returns, dim=0)

        # Normalize advantages
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        # PPO update
        out = model.forward(obs_flat, grid_size=vec_env.grid_size)

        # Recompute log probs
        action_dist = Categorical(logits=out["action_logits"])
        x_dist = Categorical(logits=out["x_logits"])
        y_dist = Categorical(logits=out["y_logits"])

        new_log_probs = (
            action_dist.log_prob(actions_flat)
            + x_dist.log_prob(xs_flat)
            + y_dist.log_prob(ys_flat)
        )
        entropy = action_dist.entropy() + x_dist.entropy() + y_dist.entropy()

        # Ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages_flat
        surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantages_flat
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        values = out["value"].squeeze(-1)
        value_loss = F.mse_loss(values, returns_flat)

        # Total loss
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # Log
        avg_reward = torch.stack(rollout_rewards).mean().item()
        rewards_history.append(avg_reward)

        if update % 50 == 0:
            success_rate = sum(successes[-100:]) / max(len(successes[-100:]), 1)
            print(
                f"[{family.name}] Update {update}: "
                f"reward={avg_reward:.3f}, success={success_rate:.1%}, "
                f"policy_loss={policy_loss.item():.4f}"
            )

    # Final evaluation
    eval_successes = []
    for _ in range(50):
        obs = vec_env.reset().to(device)
        for _ in range(vec_env.envs[0].max_steps):
            with torch.no_grad():
                out = model.get_action(obs, grid_size=vec_env.grid_size)
            obs, _, dones, infos = vec_env.step(
                out["action"].cpu(),
                out["x"].cpu(),
                out["y"].cpu(),
            )
            obs = obs.to(device)
            for i, info in enumerate(infos):
                if "success" in info:
                    eval_successes.append(1.0 if info["success"] else 0.0)
            if dones.all():
                break

    eval_success_rate = sum(eval_successes) / max(len(eval_successes), 1)
    print(f"\n[{family.name}] Final eval success rate: {eval_success_rate:.1%}")

    return {
        "family": family.name,
        "difficulty": difficulty,
        "eval_success": eval_success_rate,
        "rewards_history": rewards_history,
    }


def main():
    print("=" * 60)
    print("PRIMITIVE TRAINING VALIDATION")
    print("=" * 60)

    results = []

    # Test each family at difficulty 1
    for family in [PrimitiveFamily.NAVIGATION, PrimitiveFamily.CLICK]:
        print(f"\n{'=' * 60}")
        print(f"Training {family.name}")
        print("=" * 60)

        result = train_primitive(family, difficulty=1)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = "PASS" if r["eval_success"] > 0.5 else "FAIL"
        print(f"{r['family']:20} | diff={r['difficulty']} | success={r['eval_success']:.1%} | {status}")


if __name__ == "__main__":
    main()
