"""
PPO Training on ARC-AGI-3 Games.

Trains the ARIA agent using PPO with:
- BC-pretrained encoder as initialization
- Sparse reward from level completion
- Optional reward shaping
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .config import ARIALiteConfig
from .encoder_simple import SimpleGridEncoder
from .training.ppo_trainer import PPOConfig, RolloutBuffer, compute_gae

# ARC-AGI imports
try:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState
    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False


@dataclass
class ARCEnvResult:
    """Result from environment step."""
    observation: torch.Tensor
    reward: float
    done: bool
    info: dict


class ARCGameEnv:
    """
    Wrapper for ARC-AGI-3 game as RL environment.

    Provides:
    - Observation preprocessing (64x64 -> grid_size x grid_size)
    - Reward shaping options
    - Episode management
    """

    def __init__(
        self,
        game_id: str,
        grid_size: int = 16,
        max_steps: int = 300,
        reward_shaping: bool = True,
        device: str = "cuda",
    ):
        self.game_id = game_id
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.device = device

        self.arcade = None
        self.env = None
        self.step_count = 0
        self.prev_levels = 0
        self.prev_obs_hash = None
        self.visited_states = set()

    def _preprocess_obs(self, raw_frame) -> torch.Tensor:
        """Preprocess 64x64 observation to grid_size."""
        if raw_frame is None or raw_frame.frame is None:
            return torch.zeros(self.grid_size, self.grid_size, dtype=torch.long)

        obs = torch.from_numpy(raw_frame.frame[0]).float()
        obs = F.interpolate(
            obs.unsqueeze(0).unsqueeze(0),
            size=(self.grid_size, self.grid_size),
            mode="nearest",
        ).squeeze().long().clamp(0, 15)
        return obs

    def reset(self) -> torch.Tensor:
        """Reset environment and return initial observation."""
        if self.arcade is not None:
            self.arcade.close_scorecard()

        self.arcade = Arcade(operation_mode=OperationMode.OFFLINE)
        self.env = self.arcade.make(self.game_id)

        if self.env is None:
            raise RuntimeError(f"Could not create game {self.game_id}")

        raw_frame = self.env.reset()
        self.step_count = 0
        self.prev_levels = 0
        self.visited_states = set()
        self.prev_obs_hash = None

        obs = self._preprocess_obs(raw_frame)
        self.prev_obs_hash = hash(obs.numpy().tobytes())
        self.visited_states.add(self.prev_obs_hash)

        return obs

    def step(self, action: int) -> ARCEnvResult:
        """Take action and return result."""
        # Map action (0-5) to game action (1-6)
        game_action_id = min(action + 1, 6)
        game_action = GameAction.from_id(game_action_id)

        raw_frame = self.env.step(game_action)
        self.step_count += 1

        # Check for various terminal conditions
        if raw_frame is None:
            return ARCEnvResult(
                observation=torch.zeros(self.grid_size, self.grid_size, dtype=torch.long),
                reward=-1.0,
                done=True,
                info={"reason": "null_frame"},
            )

        # Preprocess observation
        obs = self._preprocess_obs(raw_frame)
        obs_hash = hash(obs.numpy().tobytes())

        # Base reward
        reward = 0.0

        # Reward for level completion - make this VERY strong to overcome sparse signal
        current_levels = raw_frame.levels_completed
        if current_levels > self.prev_levels:
            reward += 100.0  # Very big reward for level completion
            self.prev_levels = current_levels

        # Win bonus
        won = raw_frame.state == GameState.WIN
        if won:
            reward += 200.0

        # Reward shaping
        if self.reward_shaping:
            # Smaller penalty for revisiting states
            if obs_hash in self.visited_states:
                reward -= 0.02
            else:
                # Larger bonus for exploring new states
                reward += 0.2
                self.visited_states.add(obs_hash)

            # Tiny step penalty to encourage efficiency (but not dominate)
            reward -= 0.005

        # Check done conditions
        done = False
        info = {}

        if won:
            done = True
            info["reason"] = "win"
        elif raw_frame.state == GameState.GAME_OVER:
            done = True
            info["reason"] = "game_over"
            reward -= 1.0
        elif self.step_count >= self.max_steps:
            done = True
            info["reason"] = "max_steps"

        self.prev_obs_hash = obs_hash
        info["levels"] = current_levels
        info["steps"] = self.step_count

        return ARCEnvResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )

    def close(self):
        """Close environment."""
        if self.arcade is not None:
            self.arcade.close_scorecard()
            self.arcade = None
            self.env = None


class VectorizedARCEnv:
    """Vectorized ARC environment for parallel rollouts."""

    def __init__(
        self,
        game_id: str,
        num_envs: int,
        grid_size: int = 16,
        max_steps: int = 300,
        reward_shaping: bool = True,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.device = device
        self.grid_size = grid_size

        self.envs = [
            ARCGameEnv(
                game_id=game_id,
                grid_size=grid_size,
                max_steps=max_steps,
                reward_shaping=reward_shaping,
                device=device,
            )
            for _ in range(num_envs)
        ]

        self.obs = None

    def reset(self) -> torch.Tensor:
        """Reset all environments."""
        obs_list = []
        for env in self.envs:
            try:
                obs = env.reset()
                obs_list.append(obs)
            except Exception as e:
                print(f"Warning: env reset failed: {e}")
                obs_list.append(torch.zeros(self.grid_size, self.grid_size, dtype=torch.long))

        self.obs = torch.stack(obs_list).to(self.device)
        return self.obs

    def step(self, actions: torch.Tensor):
        """Step all environments."""
        next_obs_list = []
        rewards = []
        dones = []
        infos = []

        for i, (env, action) in enumerate(zip(self.envs, actions)):
            try:
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

            except Exception as e:
                print(f"Warning: env step failed: {e}")
                next_obs_list.append(torch.zeros(self.grid_size, self.grid_size, dtype=torch.long))
                rewards.append(-1.0)
                dones.append(True)
                infos.append({"reason": "error"})
                # Reset this env
                try:
                    env.reset()
                except:
                    pass

        self.obs = torch.stack(next_obs_list).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return self.obs, rewards, dones, infos

    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()


class ARIAActorCritic(nn.Module):
    """
    Actor-Critic using ARIA encoder.

    Can initialize from BC-pretrained encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_dim: int = 256,
        num_actions: int = 6,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_actions = num_actions

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


class ARCPPOTrainer:
    """PPO trainer for ARC-AGI-3 games."""

    def __init__(
        self,
        model: ARIAActorCritic,
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
        self.episode_levels = []

    def collect_rollouts(self, env: VectorizedARCEnv, buffer: RolloutBuffer):
        """Collect rollouts from environment."""
        buffer.clear()

        obs = env.obs
        episode_reward = torch.zeros(env.num_envs, device=self.device)
        episode_length = torch.zeros(env.num_envs, device=self.device)

        for _ in range(self.config.rollout_steps):
            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(obs)

            next_obs, reward, done, infos = env.step(action)

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
                    if "levels" in infos[i]:
                        self.episode_levels.append(infos[i]["levels"])
                    episode_reward[i] = 0
                    episode_length[i] = 0

            obs = next_obs
            self.global_step += env.num_envs

        # Compute returns and advantages
        with torch.no_grad():
            next_value = self.model.get_value(obs)

        # Stack buffer tensors
        b_obs = torch.stack(buffer.observations)
        b_actions = torch.stack(buffer.actions)
        b_log_probs = torch.stack(buffer.log_probs)
        b_rewards = torch.stack(buffer.rewards)
        b_dones = torch.stack(buffer.dones)
        b_values = torch.stack(buffer.values)

        # Compute GAE per environment
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

        b_advantages = torch.stack(all_advantages, dim=1)
        b_returns = torch.stack(all_returns, dim=1)

        # Flatten
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
        minibatch_size = max(1, batch_size // self.config.num_minibatches)

        total_pg_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0

        for _ in range(self.config.ppo_epochs):
            indices = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
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
            "pg_loss": total_pg_loss / max(1, num_updates),
            "value_loss": total_value_loss / max(1, num_updates),
            "entropy": total_entropy / max(1, num_updates),
        }

    def train(self, env: VectorizedARCEnv, num_updates: int):
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
                recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
                recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else [0]
                recent_levels = self.episode_levels[-100:] if self.episode_levels else [0]

                mean_reward = sum(recent_rewards) / len(recent_rewards)
                mean_length = sum(recent_lengths) / len(recent_lengths)
                mean_levels = sum(recent_levels) / len(recent_levels)
                max_levels = max(recent_levels) if recent_levels else 0

                print(
                    f"Update {update + 1}/{num_updates} | "
                    f"Steps: {self.global_step:,} | "
                    f"Reward: {mean_reward:.2f} | "
                    f"Levels: {mean_levels:.2f} (max: {max_levels}) | "
                    f"PG: {losses['pg_loss']:.4f} | "
                    f"Entropy: {losses['entropy']:.3f}",
                    flush=True,
                )

            # Periodic checkpoint saving (every 100 updates)
            if (update + 1) % 100 == 0 and hasattr(self, "_save_path"):
                self.save(self._save_path)

        return self.episode_rewards, self.episode_lengths, self.episode_levels

    def save(self, path: str):
        """Save model."""
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "episode_rewards": self.episode_rewards,
            "episode_levels": self.episode_levels,
        }, path)
        print(f"Saved to {path}")


def main():
    parser = argparse.ArgumentParser(description="PPO Training on ARC-AGI-3")
    parser.add_argument("--game", "-g", default="ls20", help="Game ID")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--num-updates", type=int, default=500, help="Number of PPO updates")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Steps per rollout")
    parser.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    parser.add_argument("--grid-size", type=int, default=16, help="Observation grid size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--entropy-coef", type=float, default=0.05, help="Entropy coefficient")
    parser.add_argument("--no-reward-shaping", action="store_true", help="Disable reward shaping")
    parser.add_argument(
        "--bc-checkpoint",
        default=None,
        help="BC checkpoint for encoder initialization",
    )
    parser.add_argument("--save", "-s", default="checkpoints/ppo_arc.pt", help="Save path")

    args = parser.parse_args()

    if not ARC_AGI_AVAILABLE:
        print("arc_agi not available!")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Game: {args.game}")
    print(f"Parallel envs: {args.num_envs}")

    # Create encoder
    config = ARIALiteConfig()
    encoder = SimpleGridEncoder(
        num_colors=16,
        embed_dim=32,
        hidden_dim=128,
        output_dim=config.fast_policy.state_dim,
    )

    # Load BC checkpoint if provided
    if args.bc_checkpoint and Path(args.bc_checkpoint).exists():
        bc_ckpt = torch.load(args.bc_checkpoint, map_location=device, weights_only=False)
        encoder.load_state_dict(bc_ckpt["encoder"])
        print(f"Loaded encoder from {args.bc_checkpoint}")
    else:
        print("Training encoder from scratch")

    # Create actor-critic
    model = ARIAActorCritic(
        encoder=encoder,
        hidden_dim=256,
        num_actions=6,
    )

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create PPO config
    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        max_episode_steps=args.max_steps,
        learning_rate=args.lr,
        entropy_coef=args.entropy_coef,
        log_interval=10,
        device=str(device),
    )

    # Create vectorized environment
    env = VectorizedARCEnv(
        game_id=args.game,
        num_envs=args.num_envs,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        reward_shaping=not args.no_reward_shaping,
        device=str(device),
    )

    # Create trainer
    trainer = ARCPPOTrainer(model, ppo_config)
    trainer._save_path = args.save  # For periodic checkpointing

    print(f"\nStarting PPO training for {args.num_updates} updates...")
    print(f"Reward shaping: {not args.no_reward_shaping}")
    print()

    try:
        rewards, lengths, levels = trainer.train(env, args.num_updates)

        # Save
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        trainer.save(args.save)

        # Summary
        print("\n=== Training Summary ===")
        print(f"Total steps: {trainer.global_step:,}")
        print(f"Total episodes: {len(rewards)}")
        if levels:
            print(f"Max levels reached: {max(levels)}")
            print(f"Episodes with levels > 0: {sum(1 for l in levels if l > 0)}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
