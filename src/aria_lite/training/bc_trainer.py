"""Behavioral Cloning trainer for expert imitation."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from ..agent import ARIALiteAgent, create_agent
from ..config import ARIALiteConfig
from .expert_data import ExpertDataset, collect_expert_dataset
from .synthetic_env import SyntheticEnv


@dataclass
class BCConfig:
    """Configuration for BC training."""

    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Data
    max_obs_size: int = 64

    # Logging
    log_every: int = 10
    eval_every: int = 20

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class BCMetrics:
    """Metrics from BC training."""

    epoch: int
    loss: float
    accuracy: float
    eval_success_rate: Optional[float] = None


class BCTrainer:
    """
    Behavioral Cloning trainer.

    Trains policy to imitate expert demonstrations.
    """

    def __init__(
        self,
        agent: ARIALiteAgent,
        config: BCConfig,
    ):
        self.agent = agent
        self.config = config
        self.device = torch.device(config.device)

        self.agent.to(self.device)

        # Optimizer for encoder + fast policy (BC targets fast policy)
        self.optimizer = AdamW(
            list(self.agent.encoder.parameters()) +
            list(self.agent.fast_policy.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.metrics_history: list[BCMetrics] = []

    def train(
        self,
        dataset: ExpertDataset,
        mechanic: str = "navigation",
    ) -> list[BCMetrics]:
        """
        Train on expert dataset.

        Args:
            dataset: Expert trajectories
            mechanic: Which mechanic (for evaluation)

        Returns:
            List of metrics per epoch
        """
        # Prepare data with masks
        observations, masks, actions = dataset.get_tensors(self.config.max_obs_size)

        print(f"Training on {len(actions)} transitions")
        print(f"Dataset success rate: {dataset.success_rate:.1%}")

        # Create dataloader
        tensor_dataset = TensorDataset(observations, masks, actions)
        dataloader = DataLoader(
            tensor_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )

        metrics_list = []

        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            self.agent.train()

            for batch_obs, batch_masks, batch_actions in dataloader:
                batch_obs = batch_obs.to(self.device)
                batch_masks = batch_masks.to(self.device)
                batch_actions = batch_actions.to(self.device)

                self.optimizer.zero_grad()

                # Forward through encoder + fast policy (with mask)
                states = self.agent.encoder(batch_obs, mask=batch_masks)
                fp_output = self.agent.fast_policy(states)

                # Cross-entropy loss
                loss = F.cross_entropy(fp_output.action_logits, batch_actions)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
                self.optimizer.step()

                # Track metrics
                epoch_loss += loss.item() * len(batch_actions)
                predictions = fp_output.action_logits.argmax(dim=-1)
                epoch_correct += (predictions == batch_actions).sum().item()
                epoch_total += len(batch_actions)

            avg_loss = epoch_loss / epoch_total
            accuracy = epoch_correct / epoch_total

            # Evaluate periodically
            eval_success = None
            if (epoch + 1) % self.config.eval_every == 0:
                eval_success = self.evaluate(mechanic, num_episodes=50)

            metrics = BCMetrics(
                epoch=epoch,
                loss=avg_loss,
                accuracy=accuracy,
                eval_success_rate=eval_success,
            )
            metrics_list.append(metrics)
            self.metrics_history.append(metrics)

            if (epoch + 1) % self.config.log_every == 0:
                msg = f"[BC] Epoch {epoch}: loss={avg_loss:.4f}, acc={accuracy:.1%}"
                if eval_success is not None:
                    msg += f", eval={eval_success:.1%}"
                print(msg)

        return metrics_list

    def evaluate(
        self,
        mechanic: str,
        num_episodes: int = 100,
        grid_size: int = 10,
        max_steps: int = 50,
    ) -> float:
        """Evaluate trained policy on fresh environments."""
        self.agent.eval()
        successes = 0

        for i in range(num_episodes):
            env = SyntheticEnv(
                grid_size=grid_size,
                mechanics=[mechanic],
                max_steps=max_steps,
                seed=50000 + i,  # Different seeds than training
            )

            obs = env.reset()
            done = False
            episode_reward = 0.0

            while not done:
                with torch.no_grad():
                    h, w = obs.shape

                    # Pad observation
                    obs_padded = torch.zeros(
                        self.config.max_obs_size,
                        self.config.max_obs_size,
                        dtype=obs.dtype,
                    )
                    obs_padded[:h, :w] = obs

                    # Create mask (True = invalid/padding)
                    mask = torch.ones(
                        self.config.max_obs_size,
                        self.config.max_obs_size,
                        dtype=torch.bool,
                    )
                    mask[:h, :w] = False

                    obs_batch = obs_padded.unsqueeze(0).to(self.device)
                    mask_batch = mask.unsqueeze(0).to(self.device)

                    states = self.agent.encoder(obs_batch, mask=mask_batch)
                    fp_output = self.agent.fast_policy(states)

                    action = fp_output.action_logits.argmax(dim=-1).item()

                result = env.step(action)
                episode_reward += result.reward
                obs = result.observation
                done = result.done

            if episode_reward > 0:  # Positive reward = success
                successes += 1

        self.agent.train()
        return successes / num_episodes


def train_bc(
    mechanic: str = "navigation",
    num_trajectories: int = 5000,
    epochs: int = 100,
    device: str = "cuda",
) -> tuple[ARIALiteAgent, list[BCMetrics]]:
    """
    Convenience function to train BC from scratch.

    Args:
        mechanic: Which mechanic to train on
        num_trajectories: How many expert trajectories
        epochs: Training epochs
        device: Device to train on

    Returns:
        Trained agent and metrics
    """
    print(f"Collecting expert data for {mechanic}...")
    dataset = collect_expert_dataset(
        mechanic=mechanic,
        num_trajectories=num_trajectories,
        grid_size=10,
        max_steps=50,
    )

    print(f"Collected {len(dataset.trajectories)} trajectories "
          f"({dataset.num_transitions} transitions)")

    config = BCConfig(
        epochs=epochs,
        device=device,
    )

    aria_config = ARIALiteConfig()
    agent = create_agent(aria_config)

    trainer = BCTrainer(agent, config)
    metrics = trainer.train(dataset, mechanic)

    # Final evaluation
    print("\nFinal evaluation...")
    final_success = trainer.evaluate(mechanic, num_episodes=200)
    print(f"Final success rate: {final_success:.1%}")

    return agent, metrics
