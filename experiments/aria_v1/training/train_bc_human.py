"""
Behavioral Cloning on Human Demonstrations.

Simple observation → action policy trained on human gameplay.
No meta-learning, just direct imitation.

This tests if the architecture can learn from human demos at all.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .demo_collector import DemoDataset

# Check for ARC-AGI availability
try:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState

    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False


class HumanDemoDataset(Dataset):
    """PyTorch dataset from human demonstrations."""

    def __init__(
        self,
        demo_datasets: list[DemoDataset],
        grid_size: int = 10,
        max_steps_per_demo: int = 100,
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
                    if len(obs_tensor.shape) == 2:
                        obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
                    else:
                        obs_tensor = obs_tensor.mean(dim=-1).unsqueeze(0).unsqueeze(0)

                    obs_small = F.interpolate(
                        obs_tensor,
                        size=(grid_size, grid_size),
                        mode="nearest",
                    ).squeeze()

                    # Values are already color indices (0-15), just convert to long
                    obs_quant = obs_small.long().clamp(0, 15)

                    self.observations.append(obs_quant)
                    self.actions.append(action)

        print(f"Created dataset with {len(self.observations)} (obs, action) pairs")

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class SimplePolicy(nn.Module):
    """Simple CNN policy for behavioral cloning."""

    def __init__(
        self,
        grid_size: int = 10,
        num_colors: int = 16,
        num_actions: int = 9,
        hidden_dim: int = 128,
        use_action_history: bool = False,
        history_len: int = 4,
    ):
        super().__init__()

        self.use_action_history = use_action_history
        self.history_len = history_len
        self.num_actions = num_actions

        self.color_embed = nn.Embedding(num_colors, 32)

        # CNN encoder
        self.conv1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        # Calculate flattened size
        self.flat_size = 128 * grid_size * grid_size

        # Action history embedding
        if use_action_history:
            self.action_embed = nn.Embedding(num_actions + 1, 16)  # +1 for padding
            history_dim = 16 * history_len
        else:
            history_dim = 0

        # Policy head
        self.fc1 = nn.Linear(self.flat_size + history_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(
        self, obs: torch.Tensor, action_history: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            obs: [B, H, W] grid of color indices
            action_history: [B, history_len] previous actions (optional)

        Returns:
            action_logits: [B, num_actions]
        """
        # Embed colors
        x = self.color_embed(obs)  # [B, H, W, 32]
        x = x.permute(0, 3, 1, 2)  # [B, 32, H, W]

        # CNN
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten
        x = x.flatten(1)

        # Add action history if enabled
        if self.use_action_history and action_history is not None:
            action_emb = self.action_embed(action_history)  # [B, history_len, 16]
            action_emb = action_emb.flatten(1)  # [B, history_len * 16]
            x = torch.cat([x, action_emb], dim=-1)

        # Predict
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        return logits


def train_bc(
    model: SimplePolicy,
    train_loader: DataLoader,
    num_epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device = None,
):
    """Train policy with behavioral cloning."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0
        correct = 0
        total = 0

        for obs, actions in train_loader:
            obs = obs.to(device)
            actions = torch.tensor(actions).to(device)

            logits = model(obs)
            loss = F.cross_entropy(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(-1)
            correct += (pred == actions).sum().item()
            total += len(actions)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            acc = correct / total if total > 0 else 0
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.1%}")

    return model


def evaluate_on_game(
    model: SimplePolicy,
    game_id: str,
    max_steps: int = 200,
    num_episodes: int = 3,
    device: torch.device = None,
    grid_size: int = 10,
):
    """Evaluate trained policy on actual game."""
    if not ARC_AGI_AVAILABLE:
        print("arc_agi not available for evaluation")
        return {"error": "arc_agi not available"}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    results = []

    for episode in range(num_episodes):
        arcade = Arcade(operation_mode=OperationMode.OFFLINE)
        env = arcade.make(game_id)

        if env is None:
            results.append({"error": f"Could not create {game_id}"})
            continue

        raw_frame = env.reset()
        if raw_frame is None:
            results.append({"error": "Failed to reset"})
            arcade.close_scorecard()
            continue

        levels_completed = 0
        step = 0
        action_counts = {}

        while step < max_steps:
            if raw_frame.state == GameState.WIN:
                break
            if raw_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                raw_frame = env.reset()
                if raw_frame is None:
                    break
                continue

            # Get observation
            if raw_frame.frame:
                obs = torch.from_numpy(raw_frame.frame[0]).float()
            else:
                obs = torch.zeros(64, 64)

            # Downsample - obs is already color indices (0-15)
            obs_small = F.interpolate(
                obs.unsqueeze(0).unsqueeze(0),
                size=(grid_size, grid_size),
                mode="nearest",
            )
            obs_quant = obs_small.long().clamp(0, 15).squeeze()

            # Get action
            with torch.no_grad():
                logits = model(obs_quant.unsqueeze(0).to(device))
                action_id = logits.argmax(-1).item()

            # Track action distribution
            action_counts[action_id] = action_counts.get(action_id, 0) + 1

            # Map to GameAction (actions are 1-indexed in ARC-AGI-3)
            game_action_id = min(action_id + 1, 6)  # Clamp to valid range
            action = GameAction.from_id(game_action_id)
            if action.is_complex():
                action.set_data({"x": 32, "y": 32})

            raw_frame = env.step(action)
            if raw_frame is None:
                break

            levels_completed = max(levels_completed, raw_frame.levels_completed)
            step += 1

        arcade.close_scorecard()

        won = raw_frame.state == GameState.WIN if raw_frame else False
        results.append({
            "episode": episode,
            "steps": step,
            "levels_completed": levels_completed,
            "won": won,
            "action_distribution": action_counts,
        })

        print(
            f"  Episode {episode}: {step} steps, "
            f"levels={levels_completed}, won={won}, "
            f"actions={action_counts}"
        )

    # Aggregate results
    total_levels = sum(r.get("levels_completed", 0) for r in results)
    total_wins = sum(1 for r in results if r.get("won", False))

    return {
        "game_id": game_id,
        "num_episodes": num_episodes,
        "total_levels_completed": total_levels,
        "total_wins": total_wins,
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser(description="BC on human demos")
    parser.add_argument(
        "--demos",
        "-d",
        nargs="+",
        default=[
            "demos/human/ls20_human_demos.json",
            "demos/human/vc33_human_demos.json",
            "demos/human/ft09_human_demos.json",
        ],
        help="Demo files to train on",
    )
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "--eval-game", "-g", type=str, default="ls20", help="Game to evaluate"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=3, help="Evaluation episodes"
    )
    parser.add_argument("--save", "-s", type=str, default=None, help="Save model path")
    parser.add_argument(
        "--grid-size", type=int, default=10, help="Grid size for observation"
    )
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Max steps per demo to use"
    )

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
    train_dataset = HumanDemoDataset(
        datasets,
        grid_size=args.grid_size,
        max_steps_per_demo=args.max_steps,
    )

    if len(train_dataset) == 0:
        print("No training data!")
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    # Create model
    model = SimplePolicy(
        grid_size=args.grid_size,
        num_colors=16,
        num_actions=9,
        hidden_dim=args.hidden_dim,
    )

    print(f"\nTraining on {len(train_dataset)} samples for {args.epochs} epochs...")
    model = train_bc(model, train_loader, num_epochs=args.epochs, device=device)

    # Save model
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.save)
        print(f"Saved model to {args.save}")

    # Evaluate
    if ARC_AGI_AVAILABLE:
        print(f"\nEvaluating on {args.eval_game}...")
        result = evaluate_on_game(
            model,
            args.eval_game,
            max_steps=200,
            num_episodes=args.eval_episodes,
            device=device,
            grid_size=args.grid_size,
        )
        print(f"\nResults: {result['total_levels_completed']} levels, {result['total_wins']} wins")


if __name__ == "__main__":
    main()
