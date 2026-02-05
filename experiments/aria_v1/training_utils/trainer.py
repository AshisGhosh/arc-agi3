"""
ARIA-Lite Multi-Phase Trainer

Training loop for the dual-system architecture with:
- Phase 1: World model pretraining (random rollouts)
- Phase 2: Fast policy (Behavioral Cloning + PPO)
- Phase 3: Slow policy (MCTS supervision)
- Phase 4: Arbiter calibration
- Phase 5: Joint fine-tuning

Designed for 7GB VRAM budget on RTX 4090.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..agent import ARIALiteAgent
from ..config import ARIALiteConfig
from .replay_buffer import PrioritizedReplayBuffer, ReplayBuffer, Transition
from .synthetic_env import SyntheticEnv, SyntheticEnvGenerator, collect_episode


def _to_one_hot(actions: torch.Tensor, num_actions: int = 8) -> torch.Tensor:
    """Convert action indices to one-hot encoding."""
    return F.one_hot(actions.long(), num_classes=num_actions).float()


class TrainingPhase(Enum):
    """Training phases for ARIA-Lite."""

    WORLD_MODEL = "world_model"
    FAST_POLICY = "fast_policy"
    SLOW_POLICY = "slow_policy"
    ARBITER = "arbiter"
    JOINT = "joint"


@dataclass
class TrainerConfig:
    """Configuration for ARIA-Lite trainer."""

    # General
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # World model training
    wm_epochs: int = 100
    wm_batch_size: int = 64
    wm_lr: float = 1e-4
    wm_horizon: int = 5

    # Fast policy training (BC + PPO)
    fp_epochs: int = 50
    fp_batch_size: int = 128
    fp_lr: float = 3e-4
    fp_clip_range: float = 0.2
    fp_value_coef: float = 0.5
    fp_entropy_coef: float = 0.01

    # Slow policy training
    sp_epochs: int = 100
    sp_batch_size: int = 32
    sp_lr: float = 1e-4
    sp_mcts_simulations: int = 50

    # Arbiter calibration
    arb_epochs: int = 20
    arb_batch_size: int = 256
    arb_lr: float = 1e-3

    # Joint fine-tuning
    joint_epochs: int = 50
    joint_batch_size: int = 64
    joint_lr: float = 1e-5

    # Buffer
    buffer_capacity: int = 100_000
    min_buffer_size: int = 1000

    # Environment generation
    num_train_envs: int = 1000
    num_val_envs: int = 100
    env_min_grid: int = 8
    env_max_grid: int = 32

    # Checkpointing
    checkpoint_dir: str = "checkpoints/aria_lite"
    checkpoint_every: int = 10
    log_every: int = 10

    # Gradient clipping
    max_grad_norm: float = 1.0


@dataclass
class TrainingMetrics:
    """Metrics from a training epoch."""

    phase: str
    epoch: int
    loss: float
    aux_losses: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)


class ARIALiteTrainer:
    """
    Multi-phase trainer for ARIA-Lite.

    Implements curriculum learning across components:
    1. World model learns dynamics
    2. Fast policy learns habitual responses
    3. Slow policy learns deliberate planning
    4. Arbiter learns when to switch
    5. Joint fine-tuning for coherence
    """

    def __init__(
        self,
        agent: ARIALiteAgent,
        config: TrainerConfig,
        aria_config: Optional[ARIALiteConfig] = None,
    ):
        self.agent = agent
        self.config = config
        self.aria_config = aria_config or ARIALiteConfig()
        self.device = torch.device(config.device)

        # Move agent to device
        self.agent.to(self.device)

        # Create optimizers for each phase
        self._create_optimizers()

        # Create replay buffer
        self.buffer = PrioritizedReplayBuffer(
            capacity=config.buffer_capacity,
            observation_shape=(self.aria_config.encoder.max_grid_size, self.aria_config.encoder.max_grid_size),
        )

        # Environment generator
        self.env_generator = SyntheticEnvGenerator(
            min_grid_size=config.env_min_grid,
            max_grid_size=config.env_max_grid,
            min_mechanics=1,
            max_mechanics=3,
        )

        # Training state
        self.current_phase = TrainingPhase.WORLD_MODEL
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Metrics history
        self.metrics_history: list[TrainingMetrics] = []

    def _create_optimizers(self):
        """Create optimizers for each training phase."""
        # World model optimizer
        self.wm_optimizer = AdamW(
            self.agent.world_model.parameters(),
            lr=self.config.wm_lr,
            weight_decay=0.01,
        )

        # Fast policy optimizer
        self.fp_optimizer = AdamW(
            self.agent.fast_policy.parameters(),
            lr=self.config.fp_lr,
            weight_decay=0.01,
        )

        # Slow policy optimizer
        self.sp_optimizer = AdamW(
            self.agent.slow_policy.parameters(),
            lr=self.config.sp_lr,
            weight_decay=0.01,
        )

        # Joint optimizer (all trainable params)
        all_params = list(self.agent.parameters())
        self.joint_optimizer = AdamW(
            all_params,
            lr=self.config.joint_lr,
            weight_decay=0.01,
        )

    def collect_data(self, num_episodes: int = 100):
        """Collect experience data from synthetic environments."""
        for i in range(num_episodes):
            env = self.env_generator.generate(seed=i + self.global_step)
            observations, actions, rewards, dones = collect_episode(env)

            # Store transitions
            for t in range(len(actions)):
                transition = Transition(
                    observation=observations[t],
                    action=actions[t],
                    reward=rewards[t],
                    next_observation=observations[t + 1],
                    done=dones[t],
                )
                # Use TD error as priority (estimate with reward magnitude)
                priority = abs(rewards[t]) + 0.1
                self.buffer.push(transition, priority=priority)

    def train_world_model(self, num_epochs: Optional[int] = None) -> list[TrainingMetrics]:
        """Phase 1: Train world model on collected trajectories."""
        num_epochs = num_epochs or self.config.wm_epochs
        metrics_list = []

        # Create scheduler
        scheduler = CosineAnnealingLR(self.wm_optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            if not self.buffer.is_ready(self.config.wm_batch_size):
                self.collect_data(100)

            epoch_loss = 0.0
            num_batches = 0

            # Train on buffer samples
            for _ in range(10):
                batch = self.buffer.sample(self.config.wm_batch_size)

                # Get state embeddings
                with torch.no_grad():
                    states = self.agent.encoder(batch.observations.to(self.device))
                    next_states = self.agent.encoder(batch.next_observations.to(self.device))

                # Train world model
                self.wm_optimizer.zero_grad()

                # Convert actions to one-hot for world model
                actions_onehot = _to_one_hot(batch.actions.to(self.device))
                wm_output = self.agent.world_model(
                    state=states,
                    action=actions_onehot,
                )

                # Compute losses
                next_state_loss = F.mse_loss(wm_output.next_state, next_states)
                reward_loss = F.mse_loss(
                    wm_output.reward.squeeze(-1), batch.rewards.to(self.device)
                )
                done_loss = F.binary_cross_entropy(
                    wm_output.done.squeeze(-1), batch.dones.float().to(self.device)
                )

                loss = next_state_loss + 0.1 * reward_loss + 0.1 * done_loss
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(
                    self.agent.world_model.parameters(), self.config.max_grad_norm
                )

                self.wm_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / num_batches

            metrics = TrainingMetrics(
                phase="world_model",
                epoch=epoch,
                loss=avg_loss,
                aux_losses={
                    "next_state": next_state_loss.item(),
                    "reward": reward_loss.item(),
                    "done": done_loss.item(),
                },
            )
            metrics_list.append(metrics)
            self.metrics_history.append(metrics)

            if epoch % self.config.log_every == 0:
                print(f"[WM] Epoch {epoch}: loss={avg_loss:.4f}")

        return metrics_list

    def train_fast_policy(self, num_epochs: Optional[int] = None) -> list[TrainingMetrics]:
        """Phase 2: Train fast policy with BC + PPO."""
        num_epochs = num_epochs or self.config.fp_epochs
        metrics_list = []

        scheduler = CosineAnnealingLR(self.fp_optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            if not self.buffer.is_ready(self.config.fp_batch_size):
                self.collect_data(100)

            epoch_loss = 0.0
            num_batches = 0

            for _ in range(10):
                batch = self.buffer.sample(self.config.fp_batch_size)

                # Get state embeddings
                with torch.no_grad():
                    states = self.agent.encoder(batch.observations.to(self.device))

                # Forward through fast policy
                self.fp_optimizer.zero_grad()

                fp_output = self.agent.fast_policy(states)

                # BC loss (imitation)
                action_loss = F.cross_entropy(
                    fp_output.action_logits, batch.actions.to(self.device)
                )

                # Entropy bonus for exploration
                action_probs = F.softmax(fp_output.action_logits, dim=-1)
                entropy = -(action_probs * (action_probs + 1e-8).log()).sum(dim=-1).mean()

                # Total loss
                loss = action_loss - self.config.fp_entropy_coef * entropy
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.agent.fast_policy.parameters(), self.config.max_grad_norm
                )

                self.fp_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / num_batches

            metrics = TrainingMetrics(
                phase="fast_policy",
                epoch=epoch,
                loss=avg_loss,
                aux_losses={
                    "action": action_loss.item(),
                    "entropy": entropy.item(),
                },
            )
            metrics_list.append(metrics)
            self.metrics_history.append(metrics)

            if epoch % self.config.log_every == 0:
                print(f"[FP] Epoch {epoch}: loss={avg_loss:.4f}, entropy={entropy.item():.4f}")

        return metrics_list

    def train_slow_policy(self, num_epochs: Optional[int] = None) -> list[TrainingMetrics]:
        """Phase 3: Train slow policy with MCTS-like supervision."""
        num_epochs = num_epochs or self.config.sp_epochs
        metrics_list = []

        scheduler = CosineAnnealingLR(self.sp_optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            if not self.buffer.is_ready(self.config.sp_batch_size):
                self.collect_data(100)

            epoch_loss = 0.0
            num_batches = 0

            for _ in range(10):
                batch = self.buffer.sample(self.config.sp_batch_size)

                # Get state and belief embeddings
                with torch.no_grad():
                    states = self.agent.encoder(batch.observations.to(self.device))
                    # Use zero belief for now (will be updated during trajectory)
                    beliefs = torch.zeros(
                        len(states), self.aria_config.belief.hidden_dim, device=self.device
                    )

                # Zero goal for unsupervised training
                goals = torch.zeros(
                    len(states), self.aria_config.slow_policy.goal_dim, device=self.device
                )

                self.sp_optimizer.zero_grad()

                sp_output = self.agent.slow_policy(states, beliefs, goals)

                # Policy loss (BC)
                policy_loss = F.cross_entropy(
                    sp_output.action_logits, batch.actions.to(self.device)
                )

                # Value loss (predict discounted returns)
                # Simplified: use immediate reward as target
                value_loss = F.mse_loss(
                    sp_output.value.squeeze(), batch.rewards.to(self.device)
                )

                loss = policy_loss + self.config.fp_value_coef * value_loss
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.agent.slow_policy.parameters(), self.config.max_grad_norm
                )

                self.sp_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / num_batches

            metrics = TrainingMetrics(
                phase="slow_policy",
                epoch=epoch,
                loss=avg_loss,
                aux_losses={
                    "policy": policy_loss.item(),
                    "value": value_loss.item(),
                },
            )
            metrics_list.append(metrics)
            self.metrics_history.append(metrics)

            if epoch % self.config.log_every == 0:
                print(f"[SP] Epoch {epoch}: loss={avg_loss:.4f}")

        return metrics_list

    def calibrate_arbiter(self, num_epochs: Optional[int] = None) -> list[TrainingMetrics]:
        """Phase 4: Calibrate arbiter thresholds."""
        num_epochs = num_epochs or self.config.arb_epochs
        metrics_list = []

        # Collect statistics on fast vs slow performance
        fast_correct = 0
        slow_correct = 0
        total = 0

        for epoch in range(num_epochs):
            env = self.env_generator.generate(seed=epoch)
            obs = env.reset().to(self.device).unsqueeze(0)

            done = False
            while not done:
                with torch.no_grad():
                    state = self.agent.encoder(obs)

                    # Get fast policy output
                    fp_output = self.agent.fast_policy(state)

                    # Get slow policy output
                    beliefs = torch.zeros(
                        1, self.aria_config.belief.hidden_dim, device=self.device
                    )
                    goals = torch.zeros(1, self.aria_config.slow_policy.goal_dim, device=self.device)
                    sp_output = self.agent.slow_policy(state, beliefs, goals)

                    # Compare policies (get action from argmax of probs)
                    fast_action = fp_output.action_probs.argmax(dim=-1)
                    _slow_action = sp_output.action_probs.argmax(dim=-1)

                # Take fast action and check result
                result = env.step(fast_action.item())

                if result.reward > 0:
                    fast_correct += 1
                total += 1

                obs = result.observation.to(self.device).unsqueeze(0)
                done = result.done

            metrics = TrainingMetrics(
                phase="arbiter",
                epoch=epoch,
                loss=0.0,
                metrics={
                    "fast_accuracy": fast_correct / max(total, 1),
                    "samples": total,
                },
            )
            metrics_list.append(metrics)
            self.metrics_history.append(metrics)

            if epoch % self.config.log_every == 0:
                acc = fast_correct / max(total, 1)
                print(f"[ARB] Epoch {epoch}: fast_acc={acc:.4f}")

        return metrics_list

    def joint_finetune(self, num_epochs: Optional[int] = None) -> list[TrainingMetrics]:
        """Phase 5: Joint fine-tuning of all components."""
        num_epochs = num_epochs or self.config.joint_epochs
        metrics_list = []

        scheduler = CosineAnnealingLR(self.joint_optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            if not self.buffer.is_ready(self.config.joint_batch_size):
                self.collect_data(100)

            epoch_loss = 0.0
            num_batches = 0

            for _ in range(10):
                batch = self.buffer.sample(self.config.joint_batch_size)

                self.joint_optimizer.zero_grad()

                # Full forward pass through agent
                states = self.agent.encoder(batch.observations.to(self.device))
                next_states = self.agent.encoder(batch.next_observations.to(self.device))

                # World model loss (one-hot actions)
                actions_onehot = _to_one_hot(batch.actions.to(self.device))
                wm_output = self.agent.world_model(
                    state=states, action=actions_onehot
                )
                wm_loss = F.mse_loss(wm_output.next_state, next_states)

                # Fast policy loss
                fp_output = self.agent.fast_policy(states)
                fp_loss = F.cross_entropy(
                    fp_output.action_logits, batch.actions.to(self.device)
                )

                # Slow policy loss
                beliefs = torch.zeros(
                    len(states), self.aria_config.belief.hidden_dim, device=self.device
                )
                goals = torch.zeros(len(states), self.aria_config.slow_policy.goal_dim, device=self.device)
                sp_output = self.agent.slow_policy(states, beliefs, goals)
                sp_loss = F.cross_entropy(
                    sp_output.action_logits, batch.actions.to(self.device)
                )

                # Combined loss
                loss = wm_loss + fp_loss + sp_loss
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.config.max_grad_norm
                )

                self.joint_optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / num_batches

            metrics = TrainingMetrics(
                phase="joint",
                epoch=epoch,
                loss=avg_loss,
                aux_losses={
                    "world_model": wm_loss.item(),
                    "fast_policy": fp_loss.item(),
                    "slow_policy": sp_loss.item(),
                },
            )
            metrics_list.append(metrics)
            self.metrics_history.append(metrics)

            if epoch % self.config.log_every == 0:
                print(f"[JOINT] Epoch {epoch}: loss={avg_loss:.4f}")

        return metrics_list

    def train_full(self) -> dict[str, list[TrainingMetrics]]:
        """Run complete training pipeline through all phases."""
        print("=" * 60)
        print("ARIA-Lite Training Pipeline")
        print("=" * 60)

        results = {}

        # Collect initial data
        print("\n[DATA] Collecting initial experience...")
        self.collect_data(500)

        # Phase 1: World Model
        print("\n" + "=" * 60)
        print("Phase 1: World Model Training")
        print("=" * 60)
        self.current_phase = TrainingPhase.WORLD_MODEL
        results["world_model"] = self.train_world_model()

        # Phase 2: Fast Policy
        print("\n" + "=" * 60)
        print("Phase 2: Fast Policy Training")
        print("=" * 60)
        self.current_phase = TrainingPhase.FAST_POLICY
        results["fast_policy"] = self.train_fast_policy()

        # Phase 3: Slow Policy
        print("\n" + "=" * 60)
        print("Phase 3: Slow Policy Training")
        print("=" * 60)
        self.current_phase = TrainingPhase.SLOW_POLICY
        results["slow_policy"] = self.train_slow_policy()

        # Phase 4: Arbiter
        print("\n" + "=" * 60)
        print("Phase 4: Arbiter Calibration")
        print("=" * 60)
        self.current_phase = TrainingPhase.ARBITER
        results["arbiter"] = self.calibrate_arbiter()

        # Phase 5: Joint Fine-tuning
        print("\n" + "=" * 60)
        print("Phase 5: Joint Fine-tuning")
        print("=" * 60)
        self.current_phase = TrainingPhase.JOINT
        results["joint"] = self.joint_finetune()

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return results

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            "agent_state_dict": self.agent.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "fp_optimizer": self.fp_optimizer.state_dict(),
            "sp_optimizer": self.sp_optimizer.state_dict(),
            "joint_optimizer": self.joint_optimizer.state_dict(),
            "current_phase": self.current_phase.value,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
            "aria_config": self.aria_config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.agent.load_state_dict(checkpoint["agent_state_dict"])
        self.wm_optimizer.load_state_dict(checkpoint["wm_optimizer"])
        self.fp_optimizer.load_state_dict(checkpoint["fp_optimizer"])
        self.sp_optimizer.load_state_dict(checkpoint["sp_optimizer"])
        self.joint_optimizer.load_state_dict(checkpoint["joint_optimizer"])
        self.current_phase = TrainingPhase(checkpoint["current_phase"])
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        print(f"Checkpoint loaded from {path}")

    def evaluate(self, num_episodes: int = 100) -> dict:
        """Evaluate agent on validation environments."""
        self.agent.eval()

        total_reward = 0.0
        total_steps = 0
        successes = 0
        fast_usage = 0
        total_actions = 0

        for i in range(num_episodes):
            env = self.env_generator.generate(seed=10000 + i)
            obs = env.reset()

            done = False
            episode_reward = 0.0

            # Reset agent state for new episode
            self.agent.reset(batch_size=1)

            while not done:
                with torch.no_grad():
                    # Add batch dimension and move to device
                    obs_batch = obs.unsqueeze(0).to(self.device)
                    output = self.agent.act(obs_batch)

                # action is a tensor with batch dim, get scalar
                action_scalar = output.action.item() if output.action.dim() == 0 else output.action[0].item()
                result = env.step(action_scalar)

                episode_reward += result.reward
                total_steps += 1

                if output.system_used == "fast":
                    fast_usage += 1
                total_actions += 1

                obs = result.observation
                done = result.done

            total_reward += episode_reward
            if episode_reward > 5:  # Goal reached
                successes += 1

        self.agent.train()

        return {
            "mean_reward": total_reward / num_episodes,
            "mean_steps": total_steps / num_episodes,
            "success_rate": successes / num_episodes,
            "fast_usage_rate": fast_usage / max(total_actions, 1),
        }


def create_trainer(
    agent: Optional[ARIALiteAgent] = None,
    config: Optional[TrainerConfig] = None,
    aria_config: Optional[ARIALiteConfig] = None,
) -> ARIALiteTrainer:
    """Factory function to create trainer with defaults."""
    if aria_config is None:
        aria_config = ARIALiteConfig()

    if agent is None:
        from ..agent import create_agent

        agent = create_agent(aria_config)

    if config is None:
        config = TrainerConfig()

    return ARIALiteTrainer(agent, config, aria_config)
