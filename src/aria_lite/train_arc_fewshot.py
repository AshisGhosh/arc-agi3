"""
Few-Shot Training on ARC-AGI-3 Game Demos.

Trains the MetaLearningAgent on collected game demonstrations
and evaluates few-shot adaptation performance.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .demo_collector import DemoDataset
from .meta import MetaLearningAgent

# Check for ARC-AGI availability
try:
    from arc_agi import Arcade, OperationMode
    from arcengine import FrameData, GameAction, GameState

    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False


def load_demos(demo_paths: list[str]) -> list[DemoDataset]:
    """Load demos from multiple files."""
    datasets = []
    for path in demo_paths:
        if Path(path).exists():
            dataset = DemoDataset.load(path)
            print(f"Loaded {len(dataset.demos)} demos from {path}")
            datasets.append(dataset)
    return datasets


def prepare_training_data(
    datasets: list[DemoDataset],
    grid_size: int = 10,
    max_demos_per_game: int = 20,
    max_steps: int = 10,
):
    """
    Prepare training data from demo datasets.

    Converts game demos to (observation, action) pairs suitable for meta-learning.
    """
    all_tasks = []

    for dataset in datasets:
        for demo in dataset.demos[:max_demos_per_game]:
            if len(demo.observations) < 2:
                continue

            # Downsample observations to grid_size
            obs_list = []
            for obs in demo.observations[:max_steps]:
                # Convert 64x64 to grid_size x grid_size
                obs_tensor = torch.from_numpy(obs).float()
                if len(obs_tensor.shape) == 2:
                    obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)
                else:
                    obs_tensor = obs_tensor.permute(2, 0, 1).unsqueeze(0)
                    obs_tensor = obs_tensor.mean(dim=1, keepdim=True)  # Grayscale

                # Downsample
                obs_small = F.interpolate(
                    obs_tensor,
                    size=(grid_size, grid_size),
                    mode="nearest",
                )
                # Quantize to color indices (0-15)
                obs_quant = (obs_small / 256 * 16).long().clamp(0, 15)
                obs_list.append(obs_quant.squeeze())

            actions = demo.actions[:max_steps]

            # Pad if needed
            while len(obs_list) < max_steps:
                obs_list.append(obs_list[-1].clone())
                actions.append(0)

            # Create task dict
            task = {
                "demo_obs": torch.stack(obs_list[:max_steps]).unsqueeze(0),  # [1, T, H, W]
                "demo_actions": torch.tensor(actions[:max_steps]).unsqueeze(0),  # [1, T]
                "game_id": demo.game_id,
            }
            all_tasks.append(task)

    print(f"Prepared {len(all_tasks)} training tasks")
    return all_tasks


def train_on_demos(
    model: MetaLearningAgent,
    tasks: list[dict],
    num_epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: torch.device = None,
):
    """Train meta-learning model on demo tasks."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        random.shuffle(tasks)
        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            if len(batch) < 2:
                continue

            # Stack batch
            # Use first K-1 steps as demos, predict action at step K
            demo_k = 3  # Number of demo steps to condition on

            demo_obs_list = []
            demo_actions_list = []
            query_obs_list = []
            query_actions_list = []

            for task in batch:
                # Demo: first K steps
                demo_obs_list.append(task["demo_obs"][:, :demo_k, :, :])
                demo_actions_list.append(task["demo_actions"][:, :demo_k])
                # Query: step K
                query_obs_list.append(task["demo_obs"][:, demo_k, :, :])
                query_actions_list.append(task["demo_actions"][:, demo_k])

            demo_obs = torch.cat(demo_obs_list, dim=0).to(device)  # [B, K, H, W]
            demo_actions = torch.cat(demo_actions_list, dim=0).to(device)  # [B, K]
            query_obs = torch.cat(query_obs_list, dim=0).to(device)  # [B, H, W]
            query_actions = torch.cat(query_actions_list, dim=0).to(device)  # [B]

            # Forward pass
            out = model.act(
                obs=query_obs,
                demo_obs=demo_obs,
                demo_actions=demo_actions,
                grid_size=query_obs.shape[1],
            )

            # Loss
            loss = F.cross_entropy(out["action_logits"], query_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out["action_logits"].argmax(-1)
            correct += (pred == query_actions).sum().item()
            total += len(batch)

        if epoch % 10 == 0 or epoch == num_epochs - 1:
            acc = correct / total if total > 0 else 0
            avg_loss = total_loss / (len(tasks) // batch_size + 1)
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.1%}")

    return model


def evaluate_fewshot(
    model: MetaLearningAgent,
    game_id: str,
    demo_dataset: DemoDataset,
    num_demos: int = 3,
    max_steps: int = 50,
    device: torch.device = None,
):
    """
    Evaluate few-shot performance on a game.

    Uses K demos from dataset to condition the model, then plays the game.
    """
    if not ARC_AGI_AVAILABLE:
        print("arc_agi not available for evaluation")
        return {"error": "arc_agi not available"}

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # Prepare demo context
    demo_obs, demo_actions = demo_dataset.to_meta_format(
        max_demos=num_demos, max_steps_per_demo=10
    )

    # Downsample demos
    grid_size = 10
    K, T, H, W = demo_obs.shape
    demo_obs_flat = demo_obs.float().reshape(K * T, 1, H, W)
    demo_obs_small = F.interpolate(
        demo_obs_flat,
        size=(grid_size, grid_size),
        mode="nearest",
    ).reshape(K, T, grid_size, grid_size)
    demo_obs_quant = (demo_obs_small / 256 * 16).long().clamp(0, 15)

    # Use first step of each demo
    demo_obs_input = demo_obs_quant[:, 0, :, :].unsqueeze(0).to(device)  # [1, K, H, W]
    demo_actions_input = demo_actions[:, 0].unsqueeze(0).to(device)  # [1, K]

    # Play game
    arcade = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arcade.make(game_id)

    if env is None:
        return {"error": f"Could not create {game_id}"}

    raw_frame = env.reset()
    if raw_frame is None:
        return {"error": "Failed to reset"}

    levels_completed = 0
    step = 0

    while step < max_steps:
        if raw_frame.state == GameState.WIN:
            break
        if raw_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            raw_frame = env.reset()
            if raw_frame is None:
                break
            continue

        # Get observation
        obs = torch.from_numpy(raw_frame.frame[0]).float() if raw_frame.frame else torch.zeros(64, 64)

        # Downsample
        obs_small = F.interpolate(
            obs.unsqueeze(0).unsqueeze(0),
            size=(grid_size, grid_size),
            mode="nearest",
        )
        obs_quant = (obs_small / 256 * 16).long().clamp(0, 15).squeeze().to(device)

        # Get action
        with torch.no_grad():
            out = model.act(
                obs=obs_quant.unsqueeze(0),
                demo_obs=demo_obs_input,
                demo_actions=demo_actions_input,
                grid_size=grid_size,
            )
            action_id = out["action_logits"].argmax(-1).item()

        # Map to GameAction
        action = GameAction.from_id(min(action_id + 1, 6))  # Actions 1-6
        if action.is_complex():
            action.set_data({"x": 32, "y": 32})

        raw_frame = env.step(action)
        if raw_frame is None:
            break

        levels_completed = max(levels_completed, raw_frame.levels_completed)
        step += 1

    arcade.close_scorecard()

    won = raw_frame.state == GameState.WIN if raw_frame else False

    return {
        "game_id": game_id,
        "num_demos": num_demos,
        "steps": step,
        "levels_completed": levels_completed,
        "won": won,
    }


def main():
    parser = argparse.ArgumentParser(description="Few-shot training on ARC-AGI-3 demos")
    parser.add_argument(
        "--demos",
        "-d",
        nargs="+",
        default=["demos/ls20_expert.json", "demos/vc33_demos.json", "demos/ft09_demos.json"],
        help="Demo files to train on",
    )
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Training epochs")
    parser.add_argument("--eval-game", "-g", type=str, default="ls20", help="Game to evaluate")
    parser.add_argument("--num-demos", "-k", type=int, default=3, help="Demos for few-shot")
    parser.add_argument("--save", "-s", type=str, default=None, help="Save model path")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load demos
    datasets = load_demos(args.demos)
    if not datasets:
        print("No demos found!")
        return

    # Prepare training data
    tasks = prepare_training_data(datasets)
    if not tasks:
        print("No training tasks!")
        return

    # Create model
    model = MetaLearningAgent(
        hidden_dim=128,
        task_dim=64,
        num_colors=16,
        num_actions=9,
        max_grid=20,
    )

    print(f"\nTraining on {len(tasks)} tasks for {args.epochs} epochs...")
    model = train_on_demos(model, tasks, num_epochs=args.epochs, device=device)

    # Save model
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"Saved model to {args.save}")

    # Evaluate
    if ARC_AGI_AVAILABLE and datasets:
        print(f"\nEvaluating on {args.eval_game} with {args.num_demos} demos...")

        # Find dataset for eval game
        eval_dataset = None
        for ds in datasets:
            if ds.game_id == args.eval_game:
                eval_dataset = ds
                break

        if eval_dataset is None:
            eval_dataset = datasets[0]

        result = evaluate_fewshot(
            model,
            args.eval_game,
            eval_dataset,
            num_demos=args.num_demos,
            device=device,
        )
        print(f"\nFew-shot results: {result}")


if __name__ == "__main__":
    main()
