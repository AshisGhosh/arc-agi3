"""
PPO Trainer for ARIA-Lite

Proper on-policy reinforcement learning with:
- Rollout collection from environment
- GAE advantage estimation
- PPO clipped objective
- Value function training
"""

from dataclasses import dataclass, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


@dataclass
class PPOConfig:
    """PPO training configuration."""

    # Environment
    num_envs: int = 8
    rollout_steps: int = 128
    max_episode_steps: int = 50

    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    ppo_epochs: int = 4
    num_minibatches: int = 4
    learning_rate: float = 3e-4

    # Logging
    log_interval: int = 10
    eval_interval: int = 50

    device: str = "cuda"


@dataclass
class RolloutBuffer:
    """Storage for rollout data."""

    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    dones: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, obs, action, log_prob, reward, done, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)


class ActorCritic(nn.Module):
    """Simple actor-critic for PPO."""

    def __init__(self, encoder: nn.Module, hidden_dim: int = 256, num_actions: int = 8):
        super().__init__()
        self.encoder = encoder

        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

        # Value head
        self.value = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs):
        state = self.encoder(obs)
        return self.policy(state), self.value(state)

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


class VectorizedEnv:
    """Vectorized environment wrapper for parallel rollouts."""

    def __init__(self, env_fn, num_envs: int, device: str = "cuda"):
        self.envs = [env_fn(i) for i in range(num_envs)]
        self.num_envs = num_envs
        self.device = device
        self.obs = None

    def reset(self):
        obs_list = [env.reset() for env in self.envs]
        self.obs = torch.stack(obs_list).to(self.device)
        return self.obs

    def step(self, actions):
        """Take a step in all environments."""
        next_obs_list = []
        rewards = []
        dones = []

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

        self.obs = torch.stack(next_obs_list).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return self.obs, rewards, dones


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation."""
    advantages = torch.zeros_like(rewards)
    last_gae = 0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae

    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    """PPO trainer for ARIA-Lite."""

    def __init__(
        self,
        model: ActorCritic,
        config: PPOConfig,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def collect_rollouts(self, env: VectorizedEnv, buffer: RolloutBuffer):
        """Collect rollouts from environment."""
        buffer.clear()

        obs = env.obs
        episode_reward = torch.zeros(env.num_envs, device=self.device)
        episode_length = torch.zeros(env.num_envs, device=self.device)

        for _ in range(self.config.rollout_steps):
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(obs)

            next_obs, reward, done = env.step(action)

            buffer.add(
                obs.clone(),
                action.clone(),
                log_prob.clone(),
                reward.clone(),
                done.clone(),
                value.clone(),
            )

            episode_reward += reward
            episode_length += 1

            # Track completed episodes
            for i in range(env.num_envs):
                if done[i]:
                    self.episode_rewards.append(episode_reward[i].item())
                    self.episode_lengths.append(episode_length[i].item())
                    episode_reward[i] = 0
                    episode_length[i] = 0

            obs = next_obs
            self.global_step += env.num_envs

        # Compute returns and advantages
        with torch.no_grad():
            next_value = self.model.get_value(obs)

        # Stack buffer tensors
        b_obs = torch.stack(buffer.observations)  # [T, N, H, W]
        b_actions = torch.stack(buffer.actions)  # [T, N]
        b_log_probs = torch.stack(buffer.log_probs)  # [T, N]
        b_rewards = torch.stack(buffer.rewards)  # [T, N]
        b_dones = torch.stack(buffer.dones)  # [T, N]
        b_values = torch.stack(buffer.values)  # [T, N]

        # Compute GAE per environment, then flatten
        all_advantages = []
        all_returns = []

        for i in range(env.num_envs):
            adv, ret = compute_gae(
                b_rewards[:, i],
                b_values[:, i],
                b_dones[:, i],
                next_value[i],
                self.config.gamma,
                self.config.gae_lambda,
            )
            all_advantages.append(adv)
            all_returns.append(ret)

        b_advantages = torch.stack(all_advantages, dim=1)  # [T, N]
        b_returns = torch.stack(all_returns, dim=1)  # [T, N]

        # Flatten everything: [T, N, ...] -> [T*N, ...]
        T, N = b_obs.shape[:2]

        return {
            "obs": b_obs.reshape(T * N, *b_obs.shape[2:]),
            "actions": b_actions.reshape(T * N),
            "log_probs": b_log_probs.reshape(T * N),
            "advantages": b_advantages.reshape(T * N),
            "returns": b_returns.reshape(T * N),
            "values": b_values.reshape(T * N),
        }

    def update(self, rollout_data: dict):
        """Perform PPO update."""
        obs = rollout_data["obs"]
        actions = rollout_data["actions"]
        old_log_probs = rollout_data["log_probs"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs.shape[0]
        minibatch_size = batch_size // self.config.num_minibatches

        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]

                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Get new log probs and values
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    mb_obs, mb_actions
                )

                # Policy loss (PPO clip)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value_loss = F.mse_loss(new_values, mb_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    pg_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += -entropy_loss.item()
                num_updates += 1

        return {
            "pg_loss": total_pg_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
        }

    def train(self, env: VectorizedEnv, num_updates: int):
        """Main training loop."""
        buffer = RolloutBuffer()

        env.reset()

        for update in range(num_updates):
            # Collect rollouts
            rollout_data = self.collect_rollouts(env, buffer)

            # Update policy
            losses = self.update(rollout_data)

            # Logging
            if (update + 1) % self.config.log_interval == 0:
                if self.episode_rewards:
                    mean_reward = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:])
                    mean_length = sum(self.episode_lengths[-100:]) / len(self.episode_lengths[-100:])
                else:
                    mean_reward = 0
                    mean_length = 0

                print(
                    f"Update {update + 1}/{num_updates} | "
                    f"Steps: {self.global_step:,} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Length: {mean_length:.1f} | "
                    f"PG Loss: {losses['pg_loss']:.4f} | "
                    f"Value Loss: {losses['value_loss']:.4f}",
                    flush=True,
                )

        return self.episode_rewards, self.episode_lengths
