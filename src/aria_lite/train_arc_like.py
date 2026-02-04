#!/usr/bin/env python3
"""
Train on ARC-like synthetic tasks.

These tasks bridge the gap between simple grid navigation
and actual ARC-AGI-3 challenges.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .training.arc_like_env import ARCLikeEnv, ARCAction, get_arc_task_types
from .training.ppo_trainer import PPOConfig, RolloutBuffer, compute_gae


class ARCEncoder(nn.Module):
    """Encoder for ARC-like tasks (input + output grids)."""

    def __init__(self, num_colors: int = 10, embed_dim: int = 32, hidden_dim: int = 128, output_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Process input and output grids separately, then combine
        self.input_conv = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

        # Combine and project
        self.combine = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden_dim * 2 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, obs):
        """
        Encode observation.

        Args:
            obs: [B, 2, H, W] - stacked input and output grids
        """
        B, _, H, W = obs.shape

        # Embed colors
        input_grid = obs[:, 0]  # [B, H, W]
        output_grid = obs[:, 1]  # [B, H, W]

        input_embed = self.color_embed(input_grid).permute(0, 3, 1, 2)  # [B, E, H, W]
        output_embed = self.color_embed(output_grid).permute(0, 3, 1, 2)

        # Process each grid
        input_features = self.input_conv(input_embed)  # [B, H, 4, 4]
        output_features = self.output_conv(output_embed)

        # Pool and combine
        input_pooled = F.adaptive_avg_pool2d(input_features, (4, 4))
        output_pooled = F.adaptive_avg_pool2d(output_features, (4, 4))

        combined = torch.cat([input_pooled, output_pooled], dim=1)  # [B, 2H, 4, 4]

        return self.combine(combined)


class ARCActorCritic(nn.Module):
    """Actor-critic for ARC-like tasks."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, num_actions: int = 16):
        super().__init__()
        self.encoder = encoder

        self.policy = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        self.value = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def get_action_and_value(self, obs, action=None):
        state = self.encoder(obs)
        logits = self.policy(state)
        value = self.value(state).squeeze(-1)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        state = self.encoder(obs)
        return self.value(state).squeeze(-1)


class VectorizedARCEnv:
    """Vectorized ARC environment."""

    def __init__(self, task_type: str, num_envs: int, grid_size: int = 10, device: str = "cuda"):
        self.envs = [
            ARCLikeEnv(grid_size=grid_size, task_type=task_type, max_steps=100, seed=i * 1000)
            for i in range(num_envs)
        ]
        self.num_envs = num_envs
        self.device = device
        self.obs = None

    def reset(self):
        obs_list = [env.reset() for env in self.envs]
        self.obs = torch.stack(obs_list).to(self.device)
        return self.obs

    def step(self, actions):
        next_obs_list = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            result = env.step(action.item())

            if result.done:
                # Auto-reset
                next_obs = env.reset()
            else:
                next_obs = result.observation

            next_obs_list.append(next_obs)
            rewards.append(result.reward)
            dones.append(result.done)
            infos.append(result.info)

        self.obs = torch.stack(next_obs_list).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return self.obs, rewards, dones, infos


def train_arc_task(task_type: str, num_updates: int = 300, device: str = "cuda"):
    """Train on a single ARC-like task type."""
    print(f"\n{'='*50}")
    print(f"ARC Training: {task_type}")
    print(f"{'='*50}", flush=True)

    # Create model
    encoder = ARCEncoder(output_dim=256)
    model = ARCActorCritic(encoder, hidden_dim=256, num_actions=16)
    model.to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Config
    config = PPOConfig(
        num_envs=8,
        rollout_steps=64,
        learning_rate=3e-4,
        ppo_epochs=4,
        device=device,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Create env
    env = VectorizedARCEnv(task_type, num_envs=8, grid_size=10, device=device)

    # Training loop
    episode_rewards = []
    episode_successes = []

    env.reset()
    global_step = 0

    for update in range(num_updates):
        # Collect rollouts
        buffer = RolloutBuffer()
        obs = env.obs

        for _ in range(config.rollout_steps):
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs)

            next_obs, reward, done, infos = env.step(action)

            buffer.add(obs.clone(), action.clone(), log_prob.clone(),
                      reward.clone(), done.clone(), value.clone())

            # Track completed episodes
            for i, info in enumerate(infos):
                if done[i]:
                    if "success" in info:
                        episode_successes.append(1.0 if info["success"] else 0.0)
                    episode_rewards.append(reward[i].item())

            obs = next_obs
            global_step += env.num_envs

        # Compute returns
        with torch.no_grad():
            next_value = model.get_value(obs)

        b_obs = torch.stack(buffer.observations)
        b_actions = torch.stack(buffer.actions)
        b_log_probs = torch.stack(buffer.log_probs)
        b_rewards = torch.stack(buffer.rewards)
        b_dones = torch.stack(buffer.dones)
        b_values = torch.stack(buffer.values)

        all_advantages = []
        all_returns = []

        for i in range(env.num_envs):
            adv, ret = compute_gae(
                b_rewards[:, i], b_values[:, i], b_dones[:, i],
                next_value[i], config.gamma, config.gae_lambda
            )
            all_advantages.append(adv)
            all_returns.append(ret)

        b_advantages = torch.stack(all_advantages, dim=1)
        b_returns = torch.stack(all_returns, dim=1)

        # Flatten
        T, N = b_obs.shape[:2]
        flat_obs = b_obs.reshape(T * N, *b_obs.shape[2:])
        flat_actions = b_actions.reshape(T * N)
        flat_log_probs = b_log_probs.reshape(T * N)
        flat_advantages = b_advantages.reshape(T * N)
        flat_returns = b_returns.reshape(T * N)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)

        # PPO update
        batch_size = flat_obs.shape[0]
        minibatch_size = batch_size // config.num_minibatches

        for _ in range(config.ppo_epochs):
            indices = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = indices[start:end]

                _, new_log_probs, entropy, new_values = model.get_action_and_value(
                    flat_obs[mb_idx], flat_actions[mb_idx]
                )

                ratio = torch.exp(new_log_probs - flat_log_probs[mb_idx])
                pg_loss1 = -flat_advantages[mb_idx] * ratio
                pg_loss2 = -flat_advantages[mb_idx] * torch.clamp(
                    ratio, 1 - config.clip_epsilon, 1 + config.clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                value_loss = F.mse_loss(new_values, flat_returns[mb_idx])
                entropy_loss = -entropy.mean()

                loss = pg_loss + config.value_coef * value_loss + config.entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()

        # Logging
        if (update + 1) % 20 == 0:
            mean_reward = sum(episode_rewards[-50:]) / max(len(episode_rewards[-50:]), 1)
            success_rate = sum(episode_successes[-50:]) / max(len(episode_successes[-50:]), 1)
            print(f"Update {update + 1}/{num_updates} | "
                  f"Reward: {mean_reward:.2f} | "
                  f"Success: {success_rate:.1%}",
                  flush=True)

    # Final evaluation
    print("\nEvaluating...", flush=True)
    model.eval()
    eval_successes = 0
    eval_episodes = 100

    for i in range(eval_episodes):
        env_eval = ARCLikeEnv(grid_size=10, task_type=task_type, max_steps=100, seed=99999 + i)
        obs = env_eval.reset().unsqueeze(0).to(device)
        done = False

        while not done:
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs)
            result = env_eval.step(action.item())
            obs = result.observation.unsqueeze(0).to(device)
            done = result.done

            if done and result.info.get("success", False):
                eval_successes += 1

    success_rate = eval_successes / eval_episodes
    print(f"EVAL SUCCESS RATE: {success_rate:.1%}", flush=True)

    return success_rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results = {}

    # Train on each task type
    task_types = ["copy", "fill", "reflect", "translate", "color_map"]

    for task_type in task_types:
        results[task_type] = train_arc_task(task_type, num_updates=300, device=device)

    # Summary
    print("\n" + "="*60)
    print("ARC-LIKE TRAINING SUMMARY")
    print("="*60)

    all_good = True
    for task_type, rate in results.items():
        status = "PASS" if rate > 0.3 else "FAIL"
        if rate <= 0.3:
            all_good = False
        print(f"  {task_type}: {rate:.1%} [{status}]")

    print()
    if all_good:
        print("SUCCESS: ARC-like tasks show learning!")
    else:
        print("PARTIAL: Some tasks need more work.")


if __name__ == "__main__":
    main()
