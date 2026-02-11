"""
Pretraining script for the understanding model.

Trains on synthetic game data to learn:
1. Action effects (shift vectors, change/blocked probabilities)
2. Entity roles (player, wall, collectible, background, counter)
3. Game type classification
4. Calibrated confidence

Loss components:
- Action shifts: MSE loss
- Change/blocked probs: BCE loss
- Entity roles: BCE loss (multi-label per color)
- Game type: CE loss
- Confidence: MSE loss (calibrated against prediction accuracy)
- Frame prediction: CE loss (self-supervised, for TTT warm-start)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from .model import UnderstandingModel
from .dataset import SyntheticDataset, collate_fn


def compute_loss(
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    frame_pred_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined training loss.

    Returns:
        total_loss: scalar tensor
        loss_dict: per-component losses for logging
    """
    from .decoder import SHIFT_BINS, shift_to_class

    device = predictions["shift"].device
    losses = {}

    # 1. Shift classification CE (7 bins per axis)
    target_shift = batch["action_shifts"].to(device)  # [B, 8, 2]
    B, A = target_shift.shape[:2]

    # Convert continuous shifts to class indices
    target_dx_cls = torch.zeros(B, A, dtype=torch.long, device=device)
    target_dy_cls = torch.zeros(B, A, dtype=torch.long, device=device)
    for b in range(B):
        for a in range(A):
            target_dx_cls[b, a] = shift_to_class(target_shift[b, a, 0].item())
            target_dy_cls[b, a] = shift_to_class(target_shift[b, a, 1].item())

    dx_logits = predictions["shift_dx_logits"]  # [B, 8, 7]
    dy_logits = predictions["shift_dy_logits"]  # [B, 8, 7]
    losses["shift_dx"] = F.cross_entropy(dx_logits.reshape(-1, len(SHIFT_BINS)), target_dx_cls.reshape(-1))
    losses["shift_dy"] = F.cross_entropy(dy_logits.reshape(-1, len(SHIFT_BINS)), target_dy_cls.reshape(-1))

    # Also track shift MAE for logging
    pred_shift = predictions["shift"]  # [B, 8, 2]
    losses["shift_mae"] = (pred_shift - target_shift).abs().mean()

    # 2. Change probability BCE
    pred_change = predictions["change_prob"]  # [B, 8]
    target_change = batch["action_change_prob"].to(device)  # [B, 8]
    losses["change_prob"] = F.binary_cross_entropy(pred_change, target_change)

    # 3. Blocked probability BCE
    pred_blocked = predictions["blocked_prob"]  # [B, 8]
    target_blocked = batch["action_blocked_prob"].to(device)  # [B, 8]
    losses["blocked_prob"] = F.binary_cross_entropy(pred_blocked, target_blocked)

    # 4. Entity role BCE (multi-label)
    pred_roles = predictions["entity_roles"]  # [B, 16, 5]
    target_roles = batch["entity_roles"].to(device)  # [B, 16, 5]
    losses["entity_roles"] = F.binary_cross_entropy_with_logits(pred_roles, target_roles)

    # 5. Game type CE
    pred_type = predictions["game_type"]  # [B, 8]
    target_type = batch["game_type"].to(device)  # [B]
    losses["game_type"] = F.cross_entropy(pred_type, target_type)

    # 6. Confidence MSE
    pred_conf = predictions["confidence"]  # [B]
    target_conf = batch["confidence"].to(device)  # [B]
    losses["confidence"] = F.mse_loss(pred_conf, target_conf)

    # Combine losses with weights
    total = (
        2.0 * losses["shift_dx"]
        + 2.0 * losses["shift_dy"]
        + 1.0 * losses["change_prob"]
        + 1.0 * losses["blocked_prob"]
        + 1.5 * losses["entity_roles"]
        + 1.0 * losses["game_type"]
        + 0.5 * losses["confidence"]
    )

    loss_dict = {k: v.item() for k, v in losses.items()}
    return total, loss_dict


def compute_frame_pred_loss(
    model: UnderstandingModel,
    predictions: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    max_steps: int = 30,
    max_samples: int = 128,
) -> torch.Tensor:
    """Compute frame prediction loss separately (memory-intensive)."""
    device = predictions["spatial_features"].device
    spatial = predictions["spatial_features"]  # [B, L, 256, 4, 4]
    B, L = spatial.shape[:2]
    mask = batch["mask"].to(device)

    L_use = min(L, max_steps)
    spatial_flat = spatial[:, :L_use].reshape(-1, 256, 4, 4)
    mask_flat = mask[:, :L_use].reshape(-1)
    next_frames = batch["next_frames"].to(device)[:, :L_use].reshape(-1, 64, 64)

    valid_idx = mask_flat.nonzero(as_tuple=True)[0]
    if len(valid_idx) == 0:
        return torch.tensor(0.0, device=device)

    if len(valid_idx) > max_samples:
        perm = torch.randperm(len(valid_idx), device=device)[:max_samples]
        valid_idx = valid_idx[perm]

    logits = model.frame_predictor(spatial_flat[valid_idx])  # [N, 16, 64, 64]
    targets = next_frames[valid_idx]  # [N, 64, 64]

    return F.cross_entropy(logits, targets.long())


def evaluate(
    model: UnderstandingModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_losses: dict[str, float] = {}
    count = 0

    # Accuracy metrics
    correct_game_type = 0
    total_game_type = 0
    shift_mae_sum = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            next_frames = batch["next_frames"].to(device)
            mask = batch["mask"].to(device)

            predictions = model(frames, actions, next_frames, mask=mask)
            loss, loss_dict = compute_loss(predictions, batch)

            for k, v in loss_dict.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

            # Game type accuracy
            pred_type = predictions["game_type"].argmax(dim=-1)  # [B]
            target_type = batch["game_type"].to(device)
            correct_game_type += (pred_type == target_type).sum().item()
            total_game_type += target_type.shape[0]

            # Shift MAE
            pred_shift = predictions["shift"]
            target_shift = batch["action_shifts"].to(device)
            shift_mae_sum += (pred_shift - target_shift).abs().mean().item()

            # Entity role accuracy (per color, check if top predicted role matches GT)
            pred_roles_sigmoid = torch.sigmoid(predictions["entity_roles"])  # [B, 16, 5]
            target_roles = batch["entity_roles"].to(device)

            count += 1

    metrics = {k: v / count for k, v in total_losses.items()}
    metrics["game_type_acc"] = correct_game_type / max(total_game_type, 1)
    metrics["shift_mae"] = shift_mae_sum / max(count, 1)
    return metrics


def train(
    data_dir: str = "data/synthetic",
    output_dir: str = "checkpoints/understanding",
    window_size: int = 100,
    batch_size: int = 16,
    lr: float = 3e-4,
    epochs: int = 30,
    frame_pred_weight: float = 0.1,
    val_split: float = 0.1,
    log_interval: int = 20,
    save_interval: int = 5,
) -> None:
    """Train the understanding model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    print(f"Loading data from {data_dir}...")
    dataset = SyntheticDataset(data_dir, window_size=window_size, stage="all")
    print(f"Total samples: {len(dataset)}")

    # Train/val split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # Model
    model = UnderstandingModel().to(device)
    params = model.count_params()
    print(f"Model parameters: {params}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses: dict[str, float] = {}
        epoch_frame_loss = 0.0
        step_count = 0

        for batch_idx, batch in enumerate(train_loader):
            frames = batch["frames"].to(device)
            actions = batch["actions"].to(device)
            next_frames_t = batch["next_frames"].to(device)
            mask = batch["mask"].to(device)

            # Forward
            predictions = model(frames, actions, next_frames_t, mask=mask)

            # Understanding loss
            loss, loss_dict = compute_loss(predictions, batch)

            # Frame prediction loss (self-supervised)
            frame_loss = compute_frame_pred_loss(model, predictions, batch)
            total_loss = loss + frame_pred_weight * frame_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_frame_loss += frame_loss.item()
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v
            step_count += 1

            if (batch_idx + 1) % log_interval == 0:
                avg_loss = epoch_loss / step_count
                elapsed = time.time() - start_time
                print(
                    f"  [{epoch+1}/{epochs}] step {batch_idx+1}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} frame={epoch_frame_loss/step_count:.4f} "
                    f"({elapsed:.0f}s)"
                )

        scheduler.step()

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(step_count, 1)
        avg_losses = {k: v / max(step_count, 1) for k, v in epoch_losses.items()}
        avg_frame = epoch_frame_loss / max(step_count, 1)

        print(
            f"\nEpoch {epoch+1}/{epochs}: loss={avg_epoch_loss:.4f} "
            f"frame={avg_frame:.4f} lr={scheduler.get_last_lr()[0]:.6f}"
        )
        for k, v in sorted(avg_losses.items()):
            print(f"  {k}: {v:.4f}")

        # Validation
        val_metrics = evaluate(model, val_loader, device)
        val_total = sum(v for k, v in val_metrics.items() if k not in ("game_type_acc", "shift_mae"))
        print(
            f"  Val: total={val_total:.4f} game_type_acc={val_metrics['game_type_acc']:.3f} "
            f"shift_mae={val_metrics['shift_mae']:.3f}"
        )

        # Save best
        if val_total < best_val_loss:
            best_val_loss = val_total
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_total,
                    "val_metrics": val_metrics,
                    "params": params,
                },
                output_path / "best.pt",
            )
            print(f"  Saved best model (val_loss={val_total:.4f})")

        # Periodic save
        if (epoch + 1) % save_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                output_path / f"epoch_{epoch+1}.pt",
            )

        print()

    elapsed = time.time() - start_time
    print(f"Training complete in {elapsed:.0f}s. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train understanding model")
    parser.add_argument("--data", "-d", default="data/synthetic")
    parser.add_argument("--output", "-o", default="checkpoints/understanding")
    parser.add_argument("--window", "-w", type=int, default=30)
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--frame-weight", type=float, default=0.1)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        output_dir=args.output,
        window_size=args.window,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        frame_pred_weight=args.frame_weight,
    )
