"""
Visual Grounding Training Script.

Trains the entity detector and classifier on synthetic game data.

Usage:
    uv run python -m src.aria_v2.pretraining.visual_grounding_trainer --num_samples 10000
"""

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..visual_grounding import VisualGroundingModule
from .synthetic_games import (
    GameState,
    SyntheticGameConfig,
    SyntheticGameGenerator,
)


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_samples: int = 10000
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 20
    grid_size: int = 32
    val_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 100


class SyntheticGameDataset(Dataset):
    """Dataset of synthetic game states for visual grounding training."""

    def __init__(self, states: list[GameState]):
        self.states = states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx: int) -> dict:
        state = self.states[idx]

        return {
            "grid": torch.from_numpy(state.grid).long(),
            "entity_mask": torch.from_numpy(state.entity_mask).float(),
            "entity_labels": torch.from_numpy(state.entity_labels).long(),
        }


def create_datasets(
    num_samples: int,
    grid_size: int,
    val_split: float,
    seed: int = 42,
) -> tuple[SyntheticGameDataset, SyntheticGameDataset]:
    """Create training and validation datasets."""
    config = SyntheticGameConfig(
        grid_size=grid_size,
        num_items=5,
        num_obstacles=10,
        num_triggers=2,
        has_goal=True,
        randomize_colors=True,
    )

    generator = SyntheticGameGenerator(config)
    states = generator.generate_dataset(num_samples, seed=seed)

    # Split into train/val
    split_idx = int(len(states) * (1 - val_split))
    train_states = states[:split_idx]
    val_states = states[split_idx:]

    return (
        SyntheticGameDataset(train_states),
        SyntheticGameDataset(val_states),
    )


def compute_metrics(
    pred_mask: torch.Tensor,
    pred_labels: torch.Tensor,
    target_mask: torch.Tensor,
    target_labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    """Compute evaluation metrics."""
    # Detection metrics
    pred_binary = (pred_mask > threshold).float()
    tp = ((pred_binary == 1) & (target_mask == 1)).sum().item()
    fp = ((pred_binary == 1) & (target_mask == 0)).sum().item()
    fn = ((pred_binary == 0) & (target_mask == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    # Classification metrics (only on entity pixels)
    entity_pixels = target_mask == 1
    if entity_pixels.sum() > 0:
        pred_classes = pred_labels.argmax(dim=-1)
        correct = (pred_classes == target_labels) & entity_pixels
        class_accuracy = correct.sum().item() / entity_pixels.sum().item()
    else:
        class_accuracy = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "class_accuracy": class_accuracy,
    }


def train_epoch(
    model: VisualGroundingModule,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_interval: int = 100,
) -> dict:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_mask_loss = 0.0
    total_class_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        grid = batch["grid"].to(device)
        target_mask = batch["entity_mask"].to(device)
        target_labels = batch["entity_labels"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_mask, features, class_logits = model(grid)

        # Mask loss (binary cross-entropy)
        mask_loss = F.binary_cross_entropy(pred_mask, target_mask)

        # Classification loss (cross-entropy, weighted by mask)
        B, H, W, C = class_logits.shape
        class_logits_flat = class_logits.view(B * H * W, C)
        target_labels_flat = target_labels.view(B * H * W)

        # Weight entity pixels more heavily
        weight = target_mask.view(B * H * W) * 9 + 1  # 10x weight for entities
        class_loss = F.cross_entropy(class_logits_flat, target_labels_flat, reduction="none")
        class_loss = (class_loss * weight).mean()

        # Total loss
        loss = mask_loss + class_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mask_loss += mask_loss.item()
        total_class_loss += class_loss.item()
        num_batches += 1

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Batch {batch_idx + 1}/{len(dataloader)}: "
                  f"loss={loss.item():.4f}, mask={mask_loss.item():.4f}, "
                  f"class={class_loss.item():.4f}")

    return {
        "loss": total_loss / num_batches,
        "mask_loss": total_mask_loss / num_batches,
        "class_loss": total_class_loss / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: VisualGroundingModule,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on validation set."""
    model.eval()

    all_metrics = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "class_accuracy": 0.0,
        "loss": 0.0,
    }
    num_batches = 0

    for batch in dataloader:
        grid = batch["grid"].to(device)
        target_mask = batch["entity_mask"].to(device)
        target_labels = batch["entity_labels"].to(device)

        pred_mask, features, class_logits = model(grid)

        # Compute loss
        mask_loss = F.binary_cross_entropy(pred_mask, target_mask)
        B, H, W, C = class_logits.shape
        class_logits_flat = class_logits.view(B * H * W, C)
        target_labels_flat = target_labels.view(B * H * W)
        class_loss = F.cross_entropy(class_logits_flat, target_labels_flat)
        loss = mask_loss + class_loss

        # Compute metrics
        metrics = compute_metrics(pred_mask, class_logits, target_mask, target_labels)
        metrics["loss"] = loss.item()

        for k, v in metrics.items():
            all_metrics[k] += v
        num_batches += 1

    # Average
    for k in all_metrics:
        all_metrics[k] /= num_batches

    return all_metrics


def train(config: TrainingConfig) -> VisualGroundingModule:
    """Full training loop."""
    print(f"Training visual grounding with {config.num_samples} samples")
    print(f"Device: {config.device}")

    device = torch.device(config.device)

    # Create datasets
    print("Generating synthetic data...")
    train_dataset, val_dataset = create_datasets(
        num_samples=config.num_samples,
        grid_size=config.grid_size,
        val_split=config.val_split,
    )
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Create model
    model = VisualGroundingModule(
        num_colors=16,
        embed_dim=32,
        hidden_dim=64,
        feature_dim=32,
        num_classes=6,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Training loop
    best_f1 = 0.0
    best_state = None

    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{config.num_epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config.log_interval
        )
        print(f"Train: loss={train_metrics['loss']:.4f}")

        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val: loss={val_metrics['loss']:.4f}, "
              f"precision={val_metrics['precision']:.3f}, "
              f"recall={val_metrics['recall']:.3f}, "
              f"F1={val_metrics['f1']:.3f}, "
              f"class_acc={val_metrics['class_accuracy']:.3f}")

        # Save best
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = model.state_dict().copy()
            print(f"  New best F1: {best_f1:.3f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    print(f"\nTraining complete. Best F1: {best_f1:.3f}")
    return model


def main():
    parser = argparse.ArgumentParser(description="Train visual grounding module")
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of training samples")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grid_size", type=int, default=32)
    parser.add_argument("--save_path", type=str, default="checkpoints/visual_grounding.pt")

    args = parser.parse_args()

    config = TrainingConfig(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        grid_size=args.grid_size,
    )

    model = train(config)

    # Save model
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "num_colors": 16,
            "embed_dim": 32,
            "hidden_dim": 64,
            "feature_dim": 32,
            "num_classes": 6,
        },
    }, args.save_path)
    print(f"Saved model to {args.save_path}")


if __name__ == "__main__":
    main()
