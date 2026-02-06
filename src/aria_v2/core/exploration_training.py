"""
Training for Learned Exploration Policy.

Trains on abstract features (not visual patterns) from synthetic games.
Uses information gain as reward signal.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .belief_state import BeliefState
from .exploration import (
    ExplorationStrategy,
    ExplorationPolicyNetwork,
    LearnedExplorationPolicy,
)


@dataclass
class ExplorationExperience:
    """One experience for training."""
    features: list[float]  # Abstract belief features
    strategy_taken: int  # Which strategy was used
    information_gain: float  # How much we learned
    actions_to_success: Optional[int]  # If level completed, how many actions


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    feature_dim: int = 9
    hidden_dim: int = 32
    num_strategies: int = 5

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    gamma: float = 0.99  # Discount for information gain

    # Reward shaping
    info_gain_weight: float = 1.0
    efficiency_weight: float = 0.5
    level_complete_bonus: float = 10.0

    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "checkpoints/exploration_policy"


class SyntheticGameSimulator:
    """
    Simulates simple games for exploration training.

    Generates abstract game scenarios without actual visuals.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.reset()

    def reset(self) -> BeliefState:
        """Reset to new game."""
        self.belief_state = BeliefState()

        # Generate game properties
        self.num_colors = self.rng.randint(3, 8)
        self.num_blockers = self.rng.randint(1, 3)
        self.num_collectibles = self.rng.randint(0, 4)
        self.has_goal = True

        # Assign properties to colors
        self.blocker_colors = set(self.rng.choice(
            range(1, self.num_colors + 1),
            size=self.num_blockers,
            replace=False
        ))
        remaining = set(range(1, self.num_colors + 1)) - self.blocker_colors
        if self.num_collectibles > 0 and remaining:
            self.collectible_colors = set(self.rng.choice(
                list(remaining),
                size=min(self.num_collectibles, len(remaining)),
                replace=False
            ))
        else:
            self.collectible_colors = set()
        remaining -= self.collectible_colors
        self.goal_color = self.rng.choice(list(remaining)) if remaining else 1

        # Game state
        self.player_pos = (32, 32)
        self.collected = 0
        self.actions = 0
        self.done = False

        return self.belief_state

    def step(self, strategy: ExplorationStrategy) -> tuple[BeliefState, float, bool]:
        """
        Simulate one step with given strategy.

        Returns: (belief_state, information_gain, level_completed)
        """
        self.actions += 1
        info_gain = 0.0
        level_completed = False

        if strategy == ExplorationStrategy.TEST_NEW_COLOR:
            # Pick untested color and test it
            untested = set(range(1, self.num_colors + 1)) - self.belief_state.colors_tested
            if untested:
                color = self.rng.choice(list(untested))
                self.belief_state.colors_tested.add(color)

                # Learn something about the color
                if color in self.blocker_colors:
                    self.belief_state.get_color_belief(color).times_blocked_movement += 1
                    info_gain = 1.0  # Learned it's a blocker
                elif color in self.collectible_colors:
                    self.belief_state.get_color_belief(color).times_disappeared_on_touch += 1
                    self.collected += 1
                    info_gain = 1.0  # Learned it's collectible
                elif color == self.goal_color:
                    self.belief_state.get_color_belief(color).times_triggered_level_complete += 1
                    info_gain = 2.0  # Big reward for finding goal
                    level_completed = True
                    self.done = True
                else:
                    self.belief_state.get_color_belief(color).times_walked_through += 1
                    info_gain = 0.5  # Learned it's passable

        elif strategy == ExplorationStrategy.VISIT_UNVISITED:
            # Move to unvisited area
            new_pos = (
                self.player_pos[0] + self.rng.randint(-10, 11),
                self.player_pos[1] + self.rng.randint(-10, 11)
            )
            if new_pos not in self.belief_state.positions_visited:
                self.belief_state.positions_visited.add(new_pos)
                info_gain = 0.3  # Some value for exploration
            self.player_pos = new_pos

        elif strategy == ExplorationStrategy.EXPLOIT_KNOWLEDGE:
            # Try to use what we know
            if self.collected < len(self.collectible_colors):
                # Collect known collectibles
                self.collected += 1
                info_gain = 0.2

        elif strategy == ExplorationStrategy.REVISIT_UNCERTAIN:
            # Revisit something we're uncertain about
            uncertain_colors = [
                color for color, belief in self.belief_state.color_beliefs.items()
                if belief.total_observations < 3 and belief.total_observations > 0
            ]
            if uncertain_colors:
                color = self.rng.choice(uncertain_colors)
                if color in self.blocker_colors:
                    self.belief_state.get_color_belief(color).times_blocked_movement += 1
                else:
                    self.belief_state.get_color_belief(color).times_walked_through += 1
                info_gain = 0.3  # Reduces uncertainty

        else:  # RANDOM_ACTION
            # Random exploration
            info_gain = 0.1 * self.rng.random()

        # Update player identification
        if not self.belief_state.player_identified:
            self.belief_state.player_identified = self.rng.random() < 0.3
            if self.belief_state.player_identified:
                self.belief_state.player_position = self.player_pos
                info_gain += 0.5

        self.belief_state.total_actions = self.actions

        return self.belief_state, info_gain, level_completed

    def get_frame(self) -> np.ndarray:
        """Generate a dummy frame (not used for training, just for interface)."""
        frame = np.zeros((64, 64), dtype=np.int32)
        # Add some colored regions
        for i, color in enumerate(range(1, self.num_colors + 1)):
            x = 10 + (i % 4) * 15
            y = 10 + (i // 4) * 15
            frame[y:y+5, x:x+5] = color
        return frame


class ExplorationTrainer:
    """
    Trains the learned exploration policy.

    Uses REINFORCE with information gain as reward.
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.config = config or TrainingConfig()

        # Create model
        self.model = ExplorationPolicyNetwork(
            input_dim=self.config.feature_dim,
            hidden_dim=self.config.hidden_dim,
            num_strategies=self.config.num_strategies,
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Experience buffer
        self.experiences: list[ExplorationExperience] = []

        # Stats
        self.training_stats = {
            "epoch": [],
            "loss": [],
            "avg_info_gain": [],
            "level_complete_rate": [],
        }

    def extract_features(
        self,
        belief_state: BeliefState,
        frame: np.ndarray,
    ) -> list[float]:
        """Extract abstract features from belief state."""
        h, w = frame.shape
        all_colors = set(np.unique(frame)) - {0}

        features = [
            # Player knowledge
            float(belief_state.player_identified),
            float(belief_state.player_position is not None),

            # Exploration progress
            len(belief_state.colors_tested) / max(len(all_colors), 1),
            len(belief_state.positions_visited) / (h * w / 25),

            # Uncertainty
            belief_state.get_uncertainty_score(),

            # Knowledge counts
            len(belief_state.get_confident_blockers()) / max(len(all_colors), 1),
            len(belief_state.get_confident_collectibles()) / max(len(all_colors), 1),

            # Actions taken
            min(belief_state.total_actions / 100, 1.0),

            # Levels completed
            float(belief_state.levels_completed > 0),
        ]

        return features

    def collect_experience(
        self,
        num_episodes: int = 100,
        max_steps: int = 50,
    ):
        """Collect experience from synthetic games."""
        sim = SyntheticGameSimulator()
        strategies = list(ExplorationStrategy)

        for _ in range(num_episodes):
            belief_state = sim.reset()
            episode_experiences = []

            for step in range(max_steps):
                if sim.done:
                    break

                # Get features
                frame = sim.get_frame()
                features = self.extract_features(belief_state, frame)

                # Get action from policy
                with torch.no_grad():
                    features_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
                    logits = self.model(features_t)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                    strategy_idx = torch.multinomial(probs, 1).item()

                strategy = strategies[strategy_idx]

                # Take step
                belief_state, info_gain, level_completed = sim.step(strategy)

                # Store experience
                exp = ExplorationExperience(
                    features=features,
                    strategy_taken=strategy_idx,
                    information_gain=info_gain,
                    actions_to_success=sim.actions if level_completed else None,
                )
                episode_experiences.append(exp)

            # Add efficiency bonus for completing levels quickly
            if sim.done and episode_experiences:
                efficiency_bonus = max(0, 1.0 - sim.actions / max_steps)
                episode_experiences[-1].information_gain += (
                    self.config.level_complete_bonus + efficiency_bonus * 5
                )

            self.experiences.extend(episode_experiences)

    def compute_returns(
        self,
        experiences: list[ExplorationExperience],
    ) -> list[float]:
        """Compute discounted returns."""
        returns = []
        G = 0.0

        for exp in reversed(experiences):
            reward = (
                exp.information_gain * self.config.info_gain_weight
            )
            G = reward + self.config.gamma * G
            returns.insert(0, G)

        return returns

    def train_epoch(self) -> dict:
        """Train one epoch on collected experience."""
        if not self.experiences:
            return {"loss": 0.0, "avg_info_gain": 0.0}

        # Compute returns
        returns = self.compute_returns(self.experiences)

        # Normalize returns
        returns = np.array(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Train
        self.model.train()
        total_loss = 0.0
        total_info_gain = 0.0
        num_batches = 0

        # Shuffle experiences
        indices = np.random.permutation(len(self.experiences))

        for i in range(0, len(indices), self.config.batch_size):
            batch_idx = indices[i:i + self.config.batch_size]

            # Get batch
            features = torch.tensor(
                [self.experiences[j].features for j in batch_idx],
                dtype=torch.float32
            )
            actions = torch.tensor(
                [self.experiences[j].strategy_taken for j in batch_idx],
                dtype=torch.long
            )
            batch_returns = torch.tensor(
                [returns[j] for j in batch_idx],
                dtype=torch.float32
            )

            # Forward pass
            logits = self.model(features)
            log_probs = torch.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # REINFORCE loss
            loss = -(selected_log_probs * batch_returns).mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_info_gain += sum(
                self.experiences[j].information_gain for j in batch_idx
            )
            num_batches += 1

        # Clear experiences
        level_complete_count = sum(
            1 for exp in self.experiences if exp.actions_to_success is not None
        )
        level_complete_rate = level_complete_count / len(self.experiences)

        avg_loss = total_loss / max(num_batches, 1)
        avg_info_gain = total_info_gain / len(self.experiences)

        self.experiences = []

        return {
            "loss": avg_loss,
            "avg_info_gain": avg_info_gain,
            "level_complete_rate": level_complete_rate,
        }

    def train(
        self,
        num_epochs: int = None,
        episodes_per_epoch: int = 100,
    ):
        """Full training loop."""
        num_epochs = num_epochs or self.config.num_epochs

        print(f"Training exploration policy for {num_epochs} epochs...")
        print(f"Episodes per epoch: {episodes_per_epoch}")
        print()

        for epoch in range(num_epochs):
            start_time = time.time()

            # Collect experience
            self.collect_experience(num_episodes=episodes_per_epoch)

            # Train
            stats = self.train_epoch()

            # Log
            self.training_stats["epoch"].append(epoch)
            self.training_stats["loss"].append(stats["loss"])
            self.training_stats["avg_info_gain"].append(stats["avg_info_gain"])
            self.training_stats["level_complete_rate"].append(stats["level_complete_rate"])

            elapsed = time.time() - start_time

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"Loss: {stats['loss']:.4f} | "
                    f"Info Gain: {stats['avg_info_gain']:.3f} | "
                    f"Level %: {stats['level_complete_rate']:.1%} | "
                    f"Time: {elapsed:.1f}s"
                )

            # Checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)

        print("\nTraining complete!")

    def save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"exploration_policy_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_stats": self.training_stats,
            "config": self.config,
        }, checkpoint_path)

        print(f"  Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]
        print(f"Loaded checkpoint from {path}")

    def get_trained_policy(self) -> LearnedExplorationPolicy:
        """Get the trained policy for use in agent."""
        return LearnedExplorationPolicy(model=self.model)


def test_training():
    """Test the exploration training."""
    print("Testing exploration training...\n")

    # Quick training run
    config = TrainingConfig(
        num_epochs=5,
        batch_size=16,
    )
    trainer = ExplorationTrainer(config)

    # Train
    trainer.train(num_epochs=5, episodes_per_epoch=50)

    # Get trained policy
    policy = trainer.get_trained_policy()

    # Test on a synthetic game
    sim = SyntheticGameSimulator(seed=123)
    belief_state = sim.reset()
    frame = sim.get_frame()

    decision = policy.decide(belief_state, frame, [1, 2, 3, 4])
    print(f"\nTrained policy decision: {decision.strategy.value}")
    print(f"Confidence: {decision.confidence:.3f}")


if __name__ == "__main__":
    test_training()
