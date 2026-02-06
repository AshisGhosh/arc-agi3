#!/usr/bin/env python3
"""
Train VQ-VAE frame tokenizer on ARC-AGI-3 human demonstration frames.

Usage:
    uv run python -m src.aria_v2.tokenizer.train_vqvae
    uv run python -m src.aria_v2.tokenizer.train_vqvae --epochs 200 --batch-size 128

Loads all JSONL demo files, extracts frames, trains VQ-VAE to reconstruct them.
Target: >95% pixel accuracy, >50% codebook utilization.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .frame_tokenizer import FrameVQVAE, VQVAEConfig


DEMO_DIR = Path("videos/ARC-AGI-3 Human Performance")
CHECKPOINT_DIR = Path("checkpoints/vqvae")


class FrameDataset(Dataset):
    """Dataset of 64x64 game frames from JSONL demos."""

    def __init__(self, frames: list[np.ndarray], augment: bool = True):
        self.frames = frames
        self.augment = augment

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                frame = np.flip(frame, axis=1).copy()
            # Random vertical flip
            if torch.rand(1).item() > 0.5:
                frame = np.flip(frame, axis=0).copy()
        return torch.tensor(frame, dtype=torch.long)


def load_all_frames(demo_dir: Path) -> list[np.ndarray]:
    """Load all frames from all JSONL demo files."""
    frames = []
    n_files = 0

    for jsonl_path in sorted(demo_dir.rglob("*.jsonl")):
        n_files += 1
        with open(jsonl_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                data = entry.get("data", {})
                frame_data = data.get("frame")
                if not frame_data or len(frame_data) == 0:
                    continue
                frame = np.array(frame_data[0], dtype=np.int32)
                if frame.shape == (64, 64):
                    frames.append(frame)

    print(f"Loaded {len(frames)} frames from {n_files} JSONL files")
    return frames


def train_vqvae(
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    save_every: int = 25,
):
    """Train VQ-VAE frame tokenizer."""
    print("=" * 60)
    print("VQ-VAE Frame Tokenizer Training")
    print("=" * 60)

    # Load data
    frames = load_all_frames(DEMO_DIR)
    if not frames:
        print("ERROR: No frames found! Check demo directory.")
        return

    # Split train/val (90/10)
    n_val = max(1, len(frames) // 10)
    np.random.seed(42)
    indices = np.random.permutation(len(frames))
    val_frames = [frames[i] for i in indices[:n_val]]
    train_frames = [frames[i] for i in indices[n_val:]]

    print(f"Train: {len(train_frames)}, Val: {len(val_frames)}")

    train_ds = FrameDataset(train_frames, augment=True)
    val_ds = FrameDataset(val_frames, augment=False)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True)

    # Create model
    config = VQVAEConfig()
    model = FrameVQVAE(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer with cosine schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_recon = 0.0
        train_vq = 0.0
        train_correct = 0
        train_total = 0
        n_batches = 0

        for batch in train_dl:
            batch = batch.to(device)
            logits, indices, recon_loss, vq_loss = model(batch)

            loss = recon_loss + vq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                train_recon += recon_loss.item()
                train_vq += vq_loss.item()
                pred = logits.argmax(dim=1)
                train_correct += (pred == batch).sum().item()
                train_total += batch.numel()
            n_batches += 1
            del logits, indices, recon_loss, vq_loss, loss, pred

        scheduler.step()

        train_acc = train_correct / train_total if train_total > 0 else 0
        train_recon /= max(n_batches, 1)
        train_vq /= max(n_batches, 1)

        # --- Val ---
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = batch.to(device)
                logits, indices, _, _ = model(batch)
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch).sum().item()
                val_total += batch.numel()

        val_acc = val_correct / val_total if val_total > 0 else 0
        utilization = model.get_codebook_utilization()

        elapsed = time.time() - start_time
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"recon={train_recon:.4f} vq={train_vq:.4f} | "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} | "
                f"codebook={utilization:.2%} | "
                f"lr={scheduler.get_last_lr()[0]:.2e} | "
                f"{elapsed:.0f}s"
            )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_acc": val_acc,
                "utilization": utilization,
            }, CHECKPOINT_DIR / "best.pt")

        # Periodic save
        if epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_acc": val_acc,
                "utilization": utilization,
            }, CHECKPOINT_DIR / f"epoch_{epoch}.pt")

    # Final save
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "config": config,
        "val_acc": val_acc,
        "utilization": utilization,
    }, CHECKPOINT_DIR / "final.pt")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Final codebook utilization: {utilization:.2%}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")

    # Verification: encode/decode a batch and print comparison
    print()
    print("=" * 60)
    print("Verification: Encode → Decode")
    print("=" * 60)
    model.eval()
    sample = torch.tensor(np.stack(val_frames[:4]), dtype=torch.long).to(device)
    decoded = model.decode(model.encode(sample))

    for i in range(min(4, len(val_frames))):
        orig = sample[i].cpu().numpy()
        recon = decoded[i].cpu().numpy()
        match = (orig == recon).mean()
        print(f"  Frame {i}: {match:.2%} pixel match")
        if match < 0.90:
            # Show a small patch for debugging
            print(f"    Original[0:4, 0:8]:      {orig[0:4, 0:8].tolist()}")
            print(f"    Reconstructed[0:4, 0:8]:  {recon[0:4, 0:8].tolist()}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE frame tokenizer")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_vqvae(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
