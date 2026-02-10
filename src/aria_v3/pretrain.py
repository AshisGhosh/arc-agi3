#!/usr/bin/env python3
"""
Pretrain the Dreamer world model on public game demos.

Loads human demonstrations (ls20, vc33, ft09), VQ-encodes frames,
and trains the dynamics model to predict next-frame VQ codes given
(current VQ codes, action).

Usage:
    uv run python -m src.aria_v3.pretrain
    uv run python -m src.aria_v3.pretrain --epochs 100 --lr 3e-4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..aria_v2.tokenizer.frame_tokenizer import FrameVQVAE
from ..aria_v2.tokenizer.trajectory_dataset import pixel_to_vq_cell
from .world_model import DreamerWorldModel


def load_transitions(
    demo_dir: str | Path,
    vqvae: FrameVQVAE,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Load all demo JSONL files and extract (vq_t, action, vq_{t+1}) transitions.

    Returns:
        Dict with tensors:
            'vq_current': [N, 64] current frame VQ codes
            'vq_next': [N, 64] next frame VQ codes
            'action_type': [N] action type (0-7)
            'action_loc': [N] action location (0-64)
            'game_over': [N] bool, True if this transition led to game-over/reset
            'level_complete': [N] bool, True if this transition completed a level
    """
    demo_dir = Path(demo_dir)
    jsonl_files = sorted(demo_dir.rglob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {demo_dir}")

    all_frames = []
    all_actions = []
    all_locs = []
    all_scores = []

    for jsonl_path in jsonl_files:
        frames, actions, locs, scores = _parse_jsonl(jsonl_path)
        if len(frames) >= 2:
            all_frames.extend(frames)
            all_actions.extend(actions)
            all_locs.extend(locs)
            all_scores.extend(scores)

    print(f"Loaded {len(all_frames)} frames from {len(jsonl_files)} demos")

    # VQ-encode all frames
    vqvae.eval()
    all_vq = []
    batch_size = 128

    with torch.no_grad():
        for i in range(0, len(all_frames), batch_size):
            batch = torch.tensor(
                np.stack(all_frames[i:i + batch_size]), dtype=torch.long
            ).to(device)
            indices = vqvae.encode(batch)  # [B, 8, 8]
            all_vq.append(indices.cpu().reshape(-1, 64))  # [B, 64]

    all_vq = torch.cat(all_vq, dim=0)  # [N, 64]
    print(f"VQ-encoded: {all_vq.shape}")

    # Build transition pairs: (frame_t, action_t, frame_{t+1})
    N = len(all_frames) - 1
    vq_current = all_vq[:N]       # [N, 64]
    vq_next = all_vq[1:N + 1]    # [N, 64]
    action_type = torch.tensor(all_actions[:N], dtype=torch.long)
    action_loc = torch.tensor(all_locs[:N], dtype=torch.long)

    # Detect game-over (score dropped or reset to 0)
    scores_curr = torch.tensor(all_scores[:N])
    scores_next = torch.tensor(all_scores[1:N + 1])
    game_over = scores_next < scores_curr

    # Detect level completion
    level_complete = scores_next > scores_curr

    print(f"Transitions: {N}, game_overs: {game_over.sum()}, completions: {level_complete.sum()}")

    return {
        "vq_current": vq_current,
        "vq_next": vq_next,
        "action_type": action_type,
        "action_loc": action_loc,
        "game_over": game_over,
        "level_complete": level_complete,
    }


def _parse_jsonl(path: Path) -> tuple[list, list, list, list]:
    """Parse a single JSONL demo file."""
    frames = []
    actions = []
    locs = []
    scores = []

    with open(path) as f:
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
            if frame.shape != (64, 64):
                continue

            action_input = data.get("action_input")
            action_type = action_input["id"] if action_input and "id" in action_input else 0
            score = data.get("score", 0)

            action_data = action_input.get("data", {}) if action_input else {}
            x = action_data.get("x")
            y = action_data.get("y")

            if x is not None and y is not None:
                loc = pixel_to_vq_cell(int(x), int(y))
            else:
                loc = 64  # NULL

            frames.append(frame)
            actions.append(min(action_type, 7))
            locs.append(loc)
            scores.append(score)

    return frames, actions, locs, scores


def pretrain(
    model: DreamerWorldModel,
    transitions: dict[str, torch.Tensor],
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    save_path: str = "checkpoints/dreamer/pretrained.pt",
) -> dict:
    """Pretrain the dynamics model on demo transitions.

    Trains:
    1. Dynamics head: cross-entropy on next VQ codes
    2. Value head: binary cross-entropy (game_over→0, level_complete→1, other→0.5)

    Returns:
        Training stats dict
    """
    model = model.to(device)
    model.train()

    # Build dataset
    vq_current = transitions["vq_current"].to(device)
    vq_next = transitions["vq_next"].to(device)
    action_type = transitions["action_type"].to(device)
    action_loc = transitions["action_loc"].to(device)
    game_over = transitions["game_over"].float().to(device)
    level_complete = transitions["level_complete"].float().to(device)

    # Value targets: game_over→0, level_complete→1, other→0.5
    value_targets = torch.where(
        game_over.bool(), torch.zeros_like(game_over),
        torch.where(level_complete.bool(), torch.ones_like(level_complete),
                     0.5 * torch.ones_like(game_over))
    )

    N = vq_current.shape[0]
    dataset = TensorDataset(vq_current, vq_next, action_type, action_loc, value_targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    history = []

    print(f"\nPretraining on {N} transitions for {epochs} epochs")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    start_time = time.time()

    for epoch in range(epochs):
        total_dyn_loss = 0.0
        total_val_loss = 0.0
        total_dyn_acc = 0.0
        num_batches = 0

        for vq_c, vq_n, a_type, a_loc, v_target in loader:
            # Forward
            h = model.encode(vq_c)
            dyn_logits = model.predict_dynamics(h, a_type, a_loc)  # [B, 64, 512]
            value_pred = model.predict_value(h)  # [B]

            # Dynamics loss: cross-entropy per position
            dyn_loss = F.cross_entropy(
                dyn_logits.reshape(-1, model.num_vq_codes),
                vq_n.reshape(-1),
            )

            # Value loss: MSE
            val_loss = F.mse_loss(value_pred, v_target)

            # Combined loss
            loss = dyn_loss + 0.1 * val_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Accuracy
            with torch.no_grad():
                predicted = dyn_logits.argmax(dim=-1)  # [B, 64]
                acc = (predicted == vq_n).float().mean().item()
                total_dyn_acc += acc

            total_dyn_loss += dyn_loss.item()
            total_val_loss += val_loss.item()
            num_batches += 1

        scheduler.step()

        avg_dyn = total_dyn_loss / max(num_batches, 1)
        avg_val = total_val_loss / max(num_batches, 1)
        avg_acc = total_dyn_acc / max(num_batches, 1)
        history.append({"epoch": epoch, "dyn_loss": avg_dyn, "val_loss": avg_val, "acc": avg_acc})

        if avg_dyn < best_loss:
            best_loss = avg_dyn
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "dyn_loss": avg_dyn,
                "acc": avg_acc,
            }, save_path)

        if epoch % 10 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"dyn_loss={avg_dyn:.4f} acc={avg_acc:.4f} | "
                f"val_loss={avg_val:.4f} | "
                f"best={best_loss:.4f} | {elapsed:.1f}s"
            )

    duration = time.time() - start_time
    print(f"\nPretraining complete: {duration:.1f}s, best dyn_loss={best_loss:.4f}")
    print(f"Saved best model to {save_path}")

    return {
        "best_loss": best_loss,
        "best_acc": max(h["acc"] for h in history),
        "epochs": epochs,
        "duration": duration,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Pretrain Dreamer world model")
    parser.add_argument("--demo-dir", default="videos/ARC-AGI-3 Human Performance")
    parser.add_argument("--vqvae", default="checkpoints/vqvae/best.pt")
    parser.add_argument("--save", default="checkpoints/dreamer/pretrained.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load VQ-VAE
    print("Loading VQ-VAE...")
    vqvae_ckpt = torch.load(args.vqvae, weights_only=False, map_location=args.device)
    vqvae = FrameVQVAE(vqvae_ckpt["config"]).to(args.device)
    vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
    vqvae.eval()

    # Load transitions
    transitions = load_transitions(args.demo_dir, vqvae, device=args.device)

    # Create model
    model = DreamerWorldModel(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    )
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Train
    pretrain(
        model, transitions,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
