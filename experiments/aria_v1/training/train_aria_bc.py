"""
ARIA-Lite Behavioral Cloning Training.

Trains the encoder + fast policy on human demonstrations,
compatible with the full ARIA-Lite architecture.

This trains:
- Encoder: observation → state
- Fast Policy: state → action + confidence

The trained weights can be loaded into ARIALiteAgent for evaluation.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import ARIALiteConfig
from .encoder import create_encoder
from .encoder_simple import SimpleGridEncoder
from .fast_policy import FastPolicy
from .demo_collector import DemoDataset

# ARC-AGI imports
try:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState
    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False


class ARIADemoDataset(Dataset):
    """Dataset for ARIA-Lite training from human demos."""

    def __init__(
        self,
        demo_datasets: list[DemoDataset],
        max_steps_per_demo: int = 10000,  # Use all steps by default
        grid_size: int = 16,
    ):
        self.observations = []
        self.actions = []
        self.grid_size = grid_size

        for dataset in demo_datasets:
            for demo in dataset.demos:
                # Only use successful demos
                if not demo.won and demo.levels_completed == 0:
                    continue

                for i, (obs, action) in enumerate(
                    zip(demo.observations, demo.actions)
                ):
                    if i >= max_steps_per_demo:
                        break

                    # Downsample observation to grid_size
                    obs_tensor = torch.from_numpy(obs).float()
                    if obs_tensor.shape[0] != grid_size or obs_tensor.shape[1] != grid_size:
                        obs_tensor = F.interpolate(
                            obs_tensor.unsqueeze(0).unsqueeze(0),
                            size=(grid_size, grid_size),
                            mode="nearest",
                        ).squeeze()
                    # Values are already 0-15 color indices
                    obs_tensor = obs_tensor.long().clamp(0, 15)
                    self.observations.append(obs_tensor)
                    self.actions.append(action)

        print(f"Created dataset with {len(self.observations)} samples (grid_size={grid_size})")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class ARIABCTrainer:
    """Trainer for ARIA encoder + fast policy using BC."""

    def __init__(
        self,
        config: ARIALiteConfig,
        device: torch.device,
        use_simple_encoder: bool = True,
        grid_size: int = 16,
    ):
        self.config = config
        self.device = device
        self.grid_size = grid_size
        self.use_simple_encoder = use_simple_encoder

        # Create encoder (simple for limited data, full for more data)
        if use_simple_encoder:
            self.encoder = SimpleGridEncoder(
                num_colors=16,
                embed_dim=32,
                hidden_dim=128,
                output_dim=config.fast_policy.state_dim,
            ).to(device)
        else:
            self.encoder = create_encoder(config).to(device)

        self.fast_policy = FastPolicy(config.fast_policy).to(device)

        # Combined parameters
        self.params = list(self.encoder.parameters()) + list(self.fast_policy.parameters())
        self.optimizer = torch.optim.Adam(self.params, lr=1e-3)

        # Count params
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        fp_params = sum(p.numel() for p in self.fast_policy.parameters())
        print(f"Encoder params: {enc_params:,}")
        print(f"Fast policy params: {fp_params:,}")
        print(f"Total trainable: {enc_params + fp_params:,}")

        # Class weights (will be computed from data)
        self.class_weights = None

    def set_class_weights(self, actions: list[int], num_classes: int = None):
        """Compute class weights for balanced training."""
        from collections import Counter
        counts = Counter(actions)
        # Use the number of actions from fast policy config
        if num_classes is None:
            num_classes = self.config.fast_policy.num_actions
        total = len(actions)
        # Inverse frequency weighting
        weights = []
        for i in range(num_classes):
            count = counts.get(i, 1)  # Avoid div by zero, use 1 for missing classes
            weights.append(total / (num_classes * count))
        self.class_weights = torch.tensor(weights, device=self.device, dtype=torch.float)
        print(f"Class weights ({num_classes} classes): {[f'{w:.2f}' for w in weights]}")

    def train_epoch(self, dataloader: DataLoader) -> dict:
        """Train for one epoch."""
        self.encoder.train()
        self.fast_policy.train()

        total_loss = 0
        correct = 0
        total = 0

        for obs, actions in dataloader:
            obs = obs.to(self.device)
            # Convert actions properly (avoid tensor copy warning)
            if isinstance(actions, torch.Tensor):
                actions = actions.clone().detach().to(self.device)
            else:
                actions = torch.tensor(actions, device=self.device)

            # Encode observation
            state = self.encoder(obs)

            # Get fast policy output
            action_logits = self.fast_policy(state).action_logits

            # BC loss with optional class weighting
            loss = F.cross_entropy(action_logits, actions, weight=self.class_weights)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = action_logits.argmax(-1)
            correct += (pred == actions).sum().item()
            total += len(actions)

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total if total > 0 else 0,
        }

    def evaluate(
        self,
        game_id: str,
        max_steps: int = 300,
        num_episodes: int = 3,
        epsilon: float = 0.0,
    ) -> dict:
        """Evaluate on actual game.

        Args:
            epsilon: Exploration rate (random action probability for loop breaking)
        """
        if not ARC_AGI_AVAILABLE:
            return {"error": "arc_agi not available"}

        self.encoder.eval()
        self.fast_policy.eval()

        results = []
        import random

        for ep in range(num_episodes):
            arcade = Arcade(operation_mode=OperationMode.OFFLINE)
            env = arcade.make(game_id)

            if env is None:
                results.append({"error": f"Could not create {game_id}"})
                continue

            raw_frame = env.reset()
            if raw_frame is None:
                arcade.close_scorecard()
                continue

            levels = 0
            step = 0
            action_counts = {}
            system_usage = {"fast": 0, "slow_trigger": 0, "random": 0}

            # Track recent states for loop detection
            recent_obs_hashes = []
            loop_count = 0

            while step < max_steps:
                if raw_frame.state == GameState.WIN:
                    break
                if raw_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                    raw_frame = env.reset()
                    if raw_frame is None:
                        break
                    recent_obs_hashes = []  # Reset loop detection
                    continue

                # Get observation (downsample to match training)
                if raw_frame.frame:
                    obs = torch.from_numpy(raw_frame.frame[0]).float()
                    # Downsample to grid_size
                    obs = F.interpolate(
                        obs.unsqueeze(0).unsqueeze(0),
                        size=(self.grid_size, self.grid_size),
                        mode="nearest",
                    ).squeeze(0).long().clamp(0, 15).to(self.device)
                else:
                    obs = torch.zeros(1, self.grid_size, self.grid_size).long().to(self.device)

                # Loop detection: hash recent observations
                obs_hash = hash(obs.cpu().numpy().tobytes())

                # Check for loop (same state seen in last N steps)
                in_loop = obs_hash in recent_obs_hashes[-10:] if len(recent_obs_hashes) >= 2 else False

                recent_obs_hashes.append(obs_hash)
                if len(recent_obs_hashes) > 20:
                    recent_obs_hashes.pop(0)

                # Forward pass
                with torch.no_grad():
                    state = self.encoder(obs)
                    output = self.fast_policy(state)
                    action_id = output.action_logits.argmax(-1).item()
                    confidence = output.confidence.item()

                # Track confidence for arbiter simulation
                if confidence < 0.7:
                    system_usage["slow_trigger"] += 1
                else:
                    system_usage["fast"] += 1

                # Use random action if in loop or for exploration
                use_random = in_loop or (epsilon > 0 and random.random() < epsilon)
                if use_random:
                    action_id = random.randint(0, 5)  # Random from 6 actions
                    system_usage["random"] += 1
                    if in_loop:
                        loop_count += 1

                action_counts[action_id] = action_counts.get(action_id, 0) + 1

                # Map to game action
                game_action_id = min(action_id + 1, 6)
                action = GameAction.from_id(game_action_id)
                raw_frame = env.step(action)
                if raw_frame is None:
                    break

                levels = max(levels, raw_frame.levels_completed)
                step += 1

            arcade.close_scorecard()
            won = raw_frame.state == GameState.WIN if raw_frame else False

            results.append({
                "episode": ep,
                "steps": step,
                "levels": levels,
                "won": won,
                "actions": action_counts,
                "system_usage": system_usage,
                "loops_detected": loop_count,
            })

            print(f"  Ep {ep}: levels={levels}, won={won}, loops={loop_count}, random={system_usage['random']}/{step}")

        total_levels = sum(r.get("levels", 0) for r in results)
        total_wins = sum(1 for r in results if r.get("won", False))

        return {
            "game_id": game_id,
            "total_levels": total_levels,
            "total_wins": total_wins,
            "episodes": results,
        }

    def save(self, path: str):
        """Save encoder and fast policy weights."""
        torch.save({
            "encoder": self.encoder.state_dict(),
            "fast_policy": self.fast_policy.state_dict(),
            "config": self.config,
        }, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        """Load encoder and fast policy weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.fast_policy.load_state_dict(checkpoint["fast_policy"])
        print(f"Loaded from {path}")


def main():
    parser = argparse.ArgumentParser(description="ARIA-Lite BC Training")
    parser.add_argument(
        "--demos", "-d", nargs="+",
        default=["demos/human/ls20_human_demos.json"],
        help="Demo files",
    )
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--eval-game", "-g", type=str, default="ls20")
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--save", "-s", type=str, default="checkpoints/aria_bc.pt")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--grid-size", type=int, default=16, help="Grid size for observation")
    parser.add_argument("--use-full-encoder", action="store_true", help="Use full ARIA encoder instead of simple")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Exploration rate for evaluation")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # Create dataset
    train_dataset = ARIADemoDataset(
        datasets,
        max_steps_per_demo=args.max_steps,
        grid_size=args.grid_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create trainer with ARIA config
    config = ARIALiteConfig()
    trainer = ARIABCTrainer(
        config,
        device,
        use_simple_encoder=not args.use_full_encoder,
        grid_size=args.grid_size,
    )

    # Compute and set class weights for balanced training
    trainer.set_class_weights(train_dataset.actions)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(train_loader)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.1%}")

    # Save
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.save)

    # Evaluate
    if ARC_AGI_AVAILABLE:
        print(f"\nEvaluating on {args.eval_game} (epsilon={args.epsilon})...")
        result = trainer.evaluate(
            args.eval_game,
            max_steps=300,
            num_episodes=args.eval_episodes,
            epsilon=args.epsilon,
        )
        print(f"\nResults: {result['total_levels']} levels, {result['total_wins']} wins")


if __name__ == "__main__":
    main()
