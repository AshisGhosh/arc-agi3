"""
World Model Training for ARIA-Lite.

Trains the ensemble world model to predict:
- Next state given (state, action)
- Reward signal
- Done probability

Uses BC-trained encoder to convert observations to states,
then learns dynamics in latent space.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ARIALiteConfig
from .encoder_simple import SimpleGridEncoder
from .world_model import create_world_model
from .demo_collector import DemoDataset


class TransitionDataset(Dataset):
    """Dataset of (state, action, next_state, reward, done) transitions."""

    def __init__(
        self,
        demo_datasets: list[DemoDataset],
        encoder: nn.Module,
        device: torch.device,
        grid_size: int = 16,
        max_steps_per_demo: int = 500,
    ):
        self.transitions = []
        self.grid_size = grid_size

        encoder.eval()

        for dataset in demo_datasets:
            for demo in dataset.demos:
                # Only use successful demos
                if not demo.won and demo.levels_completed == 0:
                    continue

                # Encode all observations
                encoded_states = []
                for i, obs in enumerate(demo.observations):
                    if i >= max_steps_per_demo:
                        break

                    # Preprocess observation
                    obs_tensor = torch.from_numpy(obs).float()
                    obs_small = F.interpolate(
                        obs_tensor.unsqueeze(0).unsqueeze(0),
                        size=(grid_size, grid_size),
                        mode="nearest",
                    ).squeeze(0).long().clamp(0, 15).to(device)  # [1, H, W]

                    # Encode
                    with torch.no_grad():
                        state = encoder(obs_small).cpu()
                    encoded_states.append(state.squeeze(0))

                # Create transitions
                for i in range(len(encoded_states) - 1):
                    if i >= len(demo.actions):
                        break

                    state = encoded_states[i]
                    action = demo.actions[i]
                    next_state = encoded_states[i + 1]

                    # Reward: sparse signal for level completion
                    # Check if levels increased at this step
                    reward = 0.0
                    if hasattr(demo, 'rewards') and i < len(demo.rewards):
                        reward = demo.rewards[i]

                    # Done: last step or game over
                    done = (i == len(encoded_states) - 2)

                    self.transitions.append({
                        'state': state,
                        'action': action,
                        'next_state': next_state,
                        'reward': reward,
                        'done': done,
                    })

        print(f"Created {len(self.transitions)} transitions")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        t = self.transitions[idx]
        return (
            t['state'],
            t['action'],
            t['next_state'],
            t['reward'],
            t['done'],
        )


class WorldModelTrainer:
    """Trainer for the ensemble world model."""

    def __init__(
        self,
        config: ARIALiteConfig,
        encoder: nn.Module,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.encoder = encoder

        # Create world model
        self.world_model = create_world_model(config).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.world_model.parameters(),
            lr=1e-3,
        )

        # Count params
        wm_params = sum(p.numel() for p in self.world_model.parameters())
        print(f"World model params: {wm_params:,}")

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.world_model.train()

        total_state_loss = 0
        total_reward_loss = 0
        total_done_loss = 0
        num_batches = 0

        for states, actions, next_states, rewards, dones in dataloader:
            states = states.to(self.device)
            next_states = next_states.to(self.device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

            # Convert actions to one-hot
            actions_onehot = F.one_hot(
                torch.tensor(actions),
                num_classes=self.config.world_model.action_dim,
            ).float().to(self.device)

            # Forward pass
            output = self.world_model(states, actions_onehot)

            # Losses
            state_loss = F.mse_loss(output.next_state, next_states)
            reward_loss = F.mse_loss(output.reward.squeeze(-1), rewards)
            done_loss = F.binary_cross_entropy(output.done.squeeze(-1), dones)

            # Combined loss
            loss = state_loss + 0.1 * reward_loss + 0.1 * done_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_state_loss += state_loss.item()
            total_reward_loss += reward_loss.item()
            total_done_loss += done_loss.item()
            num_batches += 1

        return {
            'state_loss': total_state_loss / num_batches,
            'reward_loss': total_reward_loss / num_batches,
            'done_loss': total_done_loss / num_batches,
        }

    def evaluate(self, dataloader: DataLoader) -> dict:
        """Evaluate world model predictions."""
        self.world_model.eval()

        total_state_error = 0
        total_uncertainty = 0
        num_samples = 0

        with torch.no_grad():
            for states, actions, next_states, rewards, dones in dataloader:
                states = states.to(self.device)
                next_states = next_states.to(self.device)

                # Convert actions to one-hot
                actions_onehot = F.one_hot(
                    torch.tensor(actions),
                    num_classes=self.config.world_model.action_dim,
                ).float().to(self.device)

                # Forward pass
                output = self.world_model(states, actions_onehot)

                # Compute errors
                state_error = (output.next_state - next_states).pow(2).mean(dim=-1)
                total_state_error += state_error.sum().item()
                total_uncertainty += output.uncertainty.sum().item()
                num_samples += states.shape[0]

        return {
            'mean_state_error': total_state_error / num_samples,
            'mean_uncertainty': total_uncertainty / num_samples,
        }

    def save(self, path: str, encoder_path: str = None):
        """Save world model and optionally encoder reference."""
        torch.save({
            'world_model': self.world_model.state_dict(),
            'config': self.config,
            'encoder_path': encoder_path,
        }, path)
        print(f"Saved world model to {path}")


def main():
    parser = argparse.ArgumentParser(description="World Model Training")
    parser.add_argument(
        "--demos", "-d", nargs="+",
        default=["demos/human/ls20_human_demos.json"],
        help="Demo files",
    )
    parser.add_argument(
        "--encoder-checkpoint", "-e",
        default="checkpoints/aria_bc_simple.pt",
        help="BC checkpoint with trained encoder",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--save", "-s",
        default="checkpoints/world_model.pt",
        help="Save path",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config and encoder from BC checkpoint
    bc_checkpoint = torch.load(
        args.encoder_checkpoint,
        map_location=device,
        weights_only=False,
    )
    config = bc_checkpoint.get('config', ARIALiteConfig())

    # Create encoder and load weights
    encoder = SimpleGridEncoder(
        num_colors=16,
        embed_dim=32,
        hidden_dim=128,
        output_dim=config.fast_policy.state_dim,
    ).to(device)
    encoder.load_state_dict(bc_checkpoint['encoder'])
    encoder.eval()
    print(f"Loaded encoder from {args.encoder_checkpoint}")

    # Load demos
    datasets = []
    for path in args.demos:
        if Path(path).exists():
            ds = DemoDataset.load(path)
            print(f"Loaded {len(ds.demos)} demos from {path}")
            datasets.append(ds)

    if not datasets:
        print("No demos found!")
        return

    # Create transition dataset
    train_dataset = TransitionDataset(
        datasets,
        encoder,
        device,
        grid_size=args.grid_size,
        max_steps_per_demo=args.max_steps,
    )

    if len(train_dataset) == 0:
        print("No transitions!")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create trainer
    trainer = WorldModelTrainer(config, encoder, device)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(train_loader)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            eval_metrics = trainer.evaluate(train_loader)
            print(
                f"Epoch {epoch}: "
                f"state_loss={metrics['state_loss']:.4f}, "
                f"uncertainty={eval_metrics['mean_uncertainty']:.4f}"
            )

    # Save
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.save, encoder_path=args.encoder_checkpoint)

    # Final evaluation
    print("\nFinal evaluation:")
    eval_metrics = trainer.evaluate(train_loader)
    print(f"  Mean state error: {eval_metrics['mean_state_error']:.4f}")
    print(f"  Mean uncertainty: {eval_metrics['mean_uncertainty']:.4f}")


if __name__ == "__main__":
    main()
