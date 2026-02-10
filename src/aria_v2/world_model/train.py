#!/usr/bin/env python3
"""
Train the SmolLM2 world model on tokenized game trajectories.

Usage:
    uv run python -m src.aria_v2.world_model.train
    uv run python -m src.aria_v2.world_model.train --epochs 50 --batch-size 8

Requires a trained VQ-VAE checkpoint (run train_vqvae.py first).
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from .config import WorldModelConfig, TrainingConfig
from .game_transformer import create_game_transformer
from ..tokenizer.frame_tokenizer import FrameVQVAE, VQVAEConfig
from ..tokenizer.trajectory_dataset import (
    tokenize_all_demos, load_cached_trajectories,
    TrajectoryWindowDataset,
    VQ_OFFSET, ACT_TYPE_OFFSET, ACT_LOC_OFFSET, ACT_LOC_NULL,
    FRAME_TOKEN, ACT_TOKEN, LEVEL_COMPLETE, GAME_START,
)


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor):
    """Compute per-token-type accuracy metrics."""
    preds = logits.argmax(dim=-1)  # [B, T]
    correct = (preds == targets)

    # Token type masks
    is_vq = (targets >= VQ_OFFSET) & (targets < VQ_OFFSET + 512)
    is_action_type = (targets >= ACT_TYPE_OFFSET) & (targets < ACT_TYPE_OFFSET + 8)
    is_action_loc = (targets >= ACT_LOC_OFFSET) & (targets <= ACT_LOC_NULL)
    is_spatial_loc = (targets >= ACT_LOC_OFFSET) & (targets < ACT_LOC_NULL)  # cells 0-63 only
    is_level = (targets == LEVEL_COMPLETE)

    metrics = {}

    # VQ frame token accuracy
    if is_vq.any():
        metrics["frame_acc"] = correct[is_vq].float().mean().item()
    else:
        metrics["frame_acc"] = 0.0

    # Action type accuracy
    if is_action_type.any():
        metrics["action_type_acc"] = correct[is_action_type].float().mean().item()
    else:
        metrics["action_type_acc"] = 0.0

    # Spatial location accuracy (cells 0-63 only, excludes NULL)
    if is_spatial_loc.any():
        metrics["spatial_loc_acc"] = correct[is_spatial_loc].float().mean().item()
    else:
        metrics["spatial_loc_acc"] = 0.0

    # All location accuracy (including NULL — for backward compat)
    if is_action_loc.any():
        metrics["action_loc_acc"] = correct[is_action_loc].float().mean().item()
    else:
        metrics["action_loc_acc"] = 0.0

    # Level completion prediction
    if is_level.any():
        metrics["level_acc"] = correct[is_level].float().mean().item()
    else:
        metrics["level_acc"] = 0.0

    # Overall weighted accuracy (on non-zero weight tokens)
    weighted_mask = weights > 0
    if weighted_mask.any():
        metrics["weighted_acc"] = correct[weighted_mask].float().mean().item()
    else:
        metrics["weighted_acc"] = 0.0

    return metrics


def weighted_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy with per-token weights.

    Args:
        logits: [B, T, V] model output logits
        targets: [B, T] target token IDs
        weights: [B, T] per-token loss weights (0 = ignore)
    """
    B, T, V = logits.shape

    # Flatten for cross_entropy
    logits_flat = logits.reshape(-1, V)  # [B*T, V]
    targets_flat = targets.reshape(-1)  # [B*T]

    # Per-token CE loss
    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # [B*T]
    loss_per_token = loss_per_token.reshape(B, T)

    # Apply weights
    weighted_loss = loss_per_token * weights
    # Average over non-zero weight tokens
    denom = weights.sum()
    if denom > 0:
        return weighted_loss.sum() / denom
    return weighted_loss.sum()


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> float:
    """Compute perplexity on weighted tokens."""
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    targets_flat = targets.reshape(-1)

    loss_per_token = F.cross_entropy(logits_flat, targets_flat, reduction="none").reshape(B, T)

    mask = weights > 0
    if mask.any():
        avg_loss = loss_per_token[mask].mean().item()
        return min(torch.exp(torch.tensor(avg_loss)).item(), 1e6)
    return float("inf")


def train_world_model(
    training_config: TrainingConfig | None = None,
    model_config: WorldModelConfig | None = None,
):
    """Full training pipeline for the world model."""
    training_config = training_config or TrainingConfig()
    model_config = model_config or WorldModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("World Model Training (SmolLM2 + LoRA)")
    print("  Unified action tokenization: (type, location) pairs")
    print("=" * 60)

    # --- Step 1: Load VQ-VAE ---
    print("\n--- Loading VQ-VAE ---")
    vqvae_ckpt = torch.load(training_config.vqvae_checkpoint, weights_only=False, map_location=device)
    vqvae_config = vqvae_ckpt["config"]
    vqvae = FrameVQVAE(vqvae_config).to(device)
    vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
    vqvae.eval()
    print(f"VQ-VAE loaded (val_acc={vqvae_ckpt.get('val_acc', '?'):.4f})")

    # --- Step 2: Build/load trajectory dataset ---
    print("\n--- Building Trajectory Dataset ---")
    cache_path = Path(training_config.cache_dir) / "trajectories_v2.pt"

    if cache_path.exists():
        print(f"Loading cached trajectories from {cache_path}")
        trajectories = load_cached_trajectories(cache_path)
    else:
        print("Tokenizing demos with unified action format...")
        trajectories = tokenize_all_demos(
            vqvae,
            training_config.demo_dir,
            cache_path=cache_path,
            device=device,
        )

    if not trajectories:
        print("ERROR: No trajectories found!")
        return

    # Create windowed dataset
    dataset = TrajectoryWindowDataset(
        trajectories,
        window_size=training_config.max_seq_len,
        stride=training_config.window_stride,
        spatial_loc_weight=training_config.spatial_loc_weight,
        null_loc_weight=training_config.null_loc_weight,
    )

    # Split train/val (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                     generator=torch.Generator().manual_seed(42))

    # Game-balanced sampling: each game gets equal probability per batch
    if training_config.game_balanced:
        full_weights = dataset.get_game_balanced_weights()
        train_weights = full_weights[train_ds.indices]
        train_sampler = WeightedRandomSampler(
            train_weights, num_samples=len(train_weights), replacement=True
        )
        train_dl = DataLoader(train_ds, batch_size=training_config.batch_size,
                              sampler=train_sampler, num_workers=2,
                              pin_memory=True, drop_last=True)
        print("Game-balanced sampling: ON")
    else:
        train_dl = DataLoader(train_ds, batch_size=training_config.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    val_dl = DataLoader(val_ds, batch_size=training_config.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {n_train} windows, Val: {n_val} windows")
    print(f"Spatial loc weight: {training_config.spatial_loc_weight}, "
          f"NULL loc weight: {training_config.null_loc_weight}")
    print(f"Steps/epoch: {len(train_dl)}, Effective batch: "
          f"{training_config.batch_size * training_config.grad_accumulation_steps}")

    # --- Step 3: Create model ---
    print("\n--- Creating Game Transformer ---")
    vqvae_codebook = vqvae.vq.embedding.weight.data.clone()
    model = create_game_transformer(model_config, vqvae_codebook=vqvae_codebook)
    model = model.to(device)

    if training_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # --- Step 4: Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    total_steps = len(train_dl) * training_config.num_epochs // training_config.grad_accumulation_steps
    warmup_steps = int(total_steps * training_config.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Free VQ-VAE memory
    del vqvae
    torch.cuda.empty_cache()

    # --- Step 5: Training loop ---
    print(f"\n--- Training for {training_config.num_epochs} epochs ---")
    output_dir = Path(training_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use bfloat16 autocast (no scaler needed, 4090 has native bf16 support)
    use_amp = training_config.fp16 and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()

    metric_keys = ["frame_acc", "action_type_acc", "spatial_loc_acc", "action_loc_acc", "level_acc", "weighted_acc"]

    for epoch in range(1, training_config.num_epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        epoch_metrics = {k: 0 for k in metric_keys}
        n_batches = 0

        optimizer.zero_grad()

        for batch_idx, (tokens, weights) in enumerate(train_dl):
            tokens = tokens.to(device)
            weights = weights.to(device)

            # Input = tokens[:-1], target = tokens[1:]
            input_ids = tokens[:, :-1]
            target_ids = tokens[:, 1:]
            target_weights = weights[:, 1:]  # Shift weights to align with targets

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits  # [B, T-1, V]

            # Compute loss in float32 (outside autocast to avoid overflow)
            loss = weighted_cross_entropy(logits.float(), target_ids, target_weights)
            loss = loss / training_config.grad_accumulation_steps

            # NaN check (early detection)
            if torch.isnan(loss):
                print(f"WARNING: NaN loss at epoch {epoch}, batch {batch_idx}")
                print(f"  logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
                print(f"  target range: [{target_ids.min().item()}, {target_ids.max().item()}]")
                print(f"  weights sum: {target_weights.sum().item():.2f}")
                optimizer.zero_grad()
                continue

            loss.backward()

            if (batch_idx + 1) % training_config.grad_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # Track metrics
            with torch.no_grad():
                batch_metrics = compute_metrics(logits.float(), target_ids, target_weights)
                for k in epoch_metrics:
                    epoch_metrics[k] += batch_metrics.get(k, 0)
            epoch_loss += loss.item() * training_config.grad_accumulation_steps
            n_batches += 1

        # Average
        epoch_loss /= max(n_batches, 1)
        for k in epoch_metrics:
            epoch_metrics[k] /= max(n_batches, 1)

        # --- Eval ---
        val_loss = 0.0
        val_metrics = {k: 0 for k in metric_keys}
        val_batches = 0

        if epoch % training_config.eval_every_epochs == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                for tokens, weights in val_dl:
                    tokens = tokens.to(device)
                    weights = weights.to(device)

                    input_ids = tokens[:, :-1]
                    target_ids = tokens[:, 1:]
                    target_weights = weights[:, 1:]

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        outputs = model(input_ids=input_ids)
                        logits = outputs.logits

                    # Compute loss in float32 (outside autocast)
                    loss = weighted_cross_entropy(logits.float(), target_ids, target_weights)

                    val_loss += loss.item()
                    batch_metrics = compute_metrics(logits.float(), target_ids, target_weights)
                    for k in val_metrics:
                        val_metrics[k] += batch_metrics.get(k, 0)
                    val_batches += 1

            val_loss /= max(val_batches, 1)
            for k in val_metrics:
                val_metrics[k] /= max(val_batches, 1)
            val_ppl = min(torch.exp(torch.tensor(val_loss)).item(), 1e6)

            # Print
            elapsed = time.time() - start_time
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch:3d}/{training_config.num_epochs} | "
                f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} ppl={val_ppl:.1f} | "
                f"frame={val_metrics['frame_acc']:.3f} "
                f"act_type={val_metrics['action_type_acc']:.3f} "
                f"spat_loc={val_metrics['spatial_loc_acc']:.3f} "
                f"level={val_metrics['level_acc']:.3f} | "
                f"lr={lr:.2e} step={global_step} | {elapsed:.0f}s"
            )

            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, model_config, training_config, epoch,
                                val_loss, val_metrics, output_dir / "best.pt")

        elif epoch % 5 == 0:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{training_config.num_epochs} | "
                f"train_loss={epoch_loss:.4f} | "
                f"frame={epoch_metrics['frame_acc']:.3f} "
                f"act_type={epoch_metrics['action_type_acc']:.3f} "
                f"spat_loc={epoch_metrics['spatial_loc_acc']:.3f} | "
                f"step={global_step} | {elapsed:.0f}s"
            )

        # Periodic save
        if epoch % training_config.save_every_epochs == 0:
            save_checkpoint(model, model_config, training_config, epoch,
                            val_loss, val_metrics, output_dir / f"epoch_{epoch}.pt")

    # Final save
    save_checkpoint(model, model_config, training_config, training_config.num_epochs,
                    val_loss, val_metrics, output_dir / "final.pt")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Checkpoints saved to: {output_dir}")


def save_checkpoint(model, model_config, training_config, epoch, val_loss, val_metrics, path):
    """Save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "training_config": training_config,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
    }, path)


def main():
    parser = argparse.ArgumentParser(description="Train world model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--vqvae-checkpoint", default="checkpoints/vqvae/best.pt")
    parser.add_argument("--output-dir", default="checkpoints/world_model")
    args = parser.parse_args()

    training_config = TrainingConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        vqvae_checkpoint=args.vqvae_checkpoint,
        output_dir=args.output_dir,
    )

    train_world_model(training_config=training_config)


if __name__ == "__main__":
    main()
