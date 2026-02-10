#!/usr/bin/env python3
"""
Train policy heads on frozen backbone with masked action tokens.

Optimization: Since the backbone is frozen, we pre-compute all hidden states
ONCE (~30 seconds), then train the lightweight policy heads (~500K params)
purely on cached features. This reduces training from ~12 hours to ~2 minutes.

Phase 1: Pre-compute backbone hidden states for all unique windows
Phase 2: Train policy heads on cached (h_frame, h_vq_cells) features

Usage:
    uv run python -m src.aria_v2.world_model.train_policy
    uv run python -m src.aria_v2.world_model.train_policy --epochs 50
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .config import WorldModelConfig, PolicyConfig
from .game_transformer import create_game_transformer
from .policy_head import PolicyHeads
from ..tokenizer.frame_tokenizer import FrameVQVAE
from ..tokenizer.trajectory_dataset import (
    tokenize_all_demos, load_cached_trajectories,
    TokenizedTrajectory,
    VQ_OFFSET, ACT_TYPE_OFFSET, ACT_LOC_OFFSET, ACT_LOC_NULL,
    FRAME_TOKEN, ACT_TOKEN, LEVEL_COMPLETE, GAME_START,
    MASK_TOKEN,
)


def find_action_positions(token_types: list[str]) -> list[dict]:
    """Find all action (type, loc) pairs and their context in a window.

    Returns list of dicts with:
        - type_pos: position of action type token
        - last_vq_pos: position of last VQ token before this action
        - vq_positions: list of 64 positions of VQ tokens in current frame
    """
    actions = []
    current_frame_vq_positions = []

    for i, tt in enumerate(token_types):
        if tt == "frame":
            current_frame_vq_positions = []
        elif tt == "vq":
            current_frame_vq_positions.append(i)
        elif tt == "action_type":
            if len(current_frame_vq_positions) == 64:
                actions.append({
                    "type_pos": i,
                    "last_vq_pos": current_frame_vq_positions[-1],
                    "vq_positions": list(current_frame_vq_positions),
                })

    return actions


def build_windows_with_actions(
    trajectories: list[TokenizedTrajectory],
    window_size: int = 2048,
    stride: int = 690,
) -> list[dict]:
    """Build unique masked windows and track all action targets in each.

    Each window is processed by the backbone exactly once. Multiple actions
    per window share the same backbone forward pass.
    """
    windows = []
    total_actions = 0

    for traj in trajectories:
        tokens = traj.tokens
        types = traj.token_types

        for start in range(0, max(1, len(tokens) - window_size + 1), stride):
            end = min(start + window_size, len(tokens))
            window_tokens = tokens[start:end]
            window_types = types[start:end]

            # Pad if needed
            if len(window_tokens) < window_size:
                pad_len = window_size - len(window_tokens)
                window_tokens = window_tokens + [0] * pad_len
                window_types = window_types + ["pad"] * pad_len

            # Find action positions
            actions = find_action_positions(window_types)
            if not actions:
                continue

            # Create masked version (replace action tokens with MASK)
            masked_tokens = list(window_tokens)
            for i, tt in enumerate(window_types):
                if tt in ("act", "action_type", "action_loc"):
                    masked_tokens[i] = MASK_TOKEN

            # Extract targets for each action
            action_list = []
            for action in actions:
                type_pos = action["type_pos"]
                loc_pos = type_pos + 1

                if loc_pos >= len(window_tokens):
                    continue

                type_target = window_tokens[type_pos] - ACT_TYPE_OFFSET
                loc_target = window_tokens[loc_pos] - ACT_LOC_OFFSET

                if not (0 <= type_target < 8) or not (0 <= loc_target < 65):
                    continue

                action_list.append({
                    "type_target": type_target,
                    "loc_target": loc_target,
                    "last_vq_pos": action["last_vq_pos"],
                    "vq_positions": action["vq_positions"],
                })

            if action_list:
                windows.append({
                    "masked_tokens": torch.tensor(masked_tokens, dtype=torch.long),
                    "actions": action_list,
                })
                total_actions += len(action_list)

    print(f"Built {len(windows)} unique windows with {total_actions} total actions")
    return windows


def precompute_features(
    windows: list[dict],
    backbone: torch.nn.Module,
    mask_embedding: torch.Tensor,
    device: str,
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Run backbone once per unique window, extract per-action features.

    Returns dict with h_frames, h_vq_cells, type_targets, loc_targets.
    Features stored in bfloat16 to save memory.
    """
    all_h_frames = []
    all_h_vq_cells = []
    all_type_targets = []
    all_loc_targets = []

    n_windows = len(windows)
    amp_dtype = torch.bfloat16

    print(f"Pre-computing hidden states for {n_windows} windows (batch_size={batch_size})...")
    t0 = time.time()

    for batch_start in range(0, n_windows, batch_size):
        batch_end = min(batch_start + batch_size, n_windows)
        batch = windows[batch_start:batch_end]

        tokens = torch.stack([w["masked_tokens"] for w in batch]).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=amp_dtype):
            mask_pos = (tokens == MASK_TOKEN)
            safe = tokens.clamp(max=MASK_TOKEN - 1)
            embeds = backbone.get_input_embeddings()(safe)
            embeds[mask_pos] = mask_embedding.to(embeds.dtype)

            outputs = backbone(inputs_embeds=embeds, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # [B, T, 960]

        # Extract features for all actions in each window
        for b_idx, w in enumerate(batch):
            for action in w["actions"]:
                h_frame = hidden[b_idx, action["last_vq_pos"]].cpu().to(torch.bfloat16)
                h_vq = hidden[b_idx, action["vq_positions"]].cpu().to(torch.bfloat16)

                all_h_frames.append(h_frame)
                all_h_vq_cells.append(h_vq)
                all_type_targets.append(action["type_target"])
                all_loc_targets.append(action["loc_target"])

        if (batch_start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {batch_end}/{n_windows} windows ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    n_actions = len(all_type_targets)
    print(f"Pre-computed {n_actions} action features in {elapsed:.1f}s")

    return {
        "h_frames": torch.stack(all_h_frames),         # [N, 960] bf16
        "h_vq_cells": torch.stack(all_h_vq_cells),     # [N, 64, 960] bf16
        "type_targets": torch.tensor(all_type_targets, dtype=torch.long),
        "loc_targets": torch.tensor(all_loc_targets, dtype=torch.long),
    }


class CachedPolicyDataset(Dataset):
    """Simple dataset over pre-computed backbone features."""

    def __init__(self, h_frames, h_vq_cells, type_targets, loc_targets):
        self.h_frames = h_frames
        self.h_vq_cells = h_vq_cells
        self.type_targets = type_targets
        self.loc_targets = loc_targets

    def __len__(self):
        return len(self.type_targets)

    def __getitem__(self, idx):
        return (
            self.h_frames[idx],
            self.h_vq_cells[idx],
            self.type_targets[idx],
            self.loc_targets[idx],
        )


def train_policy(
    policy_config: PolicyConfig | None = None,
    model_config: WorldModelConfig | None = None,
):
    """Train policy heads on frozen backbone with pre-computed features."""
    policy_config = policy_config or PolicyConfig()
    model_config = model_config or WorldModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Policy Head Training (pre-computed features)")
    print("=" * 60)

    # --- Step 1: Load VQ-VAE (for tokenization if needed) ---
    print("\n--- Loading VQ-VAE ---")
    vqvae_ckpt = torch.load(policy_config.vqvae_checkpoint, weights_only=False, map_location=device)
    vqvae = FrameVQVAE(vqvae_ckpt["config"]).to(device)
    vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
    vqvae.eval()

    # --- Step 2: Load trajectories ---
    print("\n--- Loading Trajectories ---")
    cache_path = Path(policy_config.cache_dir) / "trajectories_v2.pt"

    if cache_path.exists():
        trajectories = load_cached_trajectories(cache_path)
    else:
        trajectories = tokenize_all_demos(
            vqvae, policy_config.demo_dir, cache_path=cache_path, device=device
        )

    del vqvae
    torch.cuda.empty_cache()

    if not trajectories:
        print("ERROR: No trajectories!")
        return

    # --- Step 3: Build windows with actions ---
    print("\n--- Building Windows ---")
    windows = build_windows_with_actions(trajectories, window_size=2048, stride=690)
    del trajectories

    # --- Step 4: Load backbone (frozen) ---
    print("\n--- Loading Backbone (frozen) ---")
    wm_ckpt = torch.load(policy_config.world_model_checkpoint, weights_only=False, map_location=device)
    backbone_config = wm_ckpt.get("model_config", model_config)
    backbone = create_game_transformer(backbone_config)
    backbone.load_state_dict(wm_ckpt["model_state_dict"])
    backbone = backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False

    # --- Step 5: Create policy heads ---
    print("\n--- Creating Policy Heads ---")
    policy = PolicyHeads(policy_config).to(device)
    print(f"Policy parameters: {policy.count_parameters():,}")

    # --- Step 6: Pre-compute all hidden states ---
    print("\n--- Pre-computing Hidden States ---")
    features = precompute_features(
        windows, backbone, policy.mask_embedding, device, batch_size=8,
    )

    # Free backbone memory
    del backbone, wm_ckpt, windows
    torch.cuda.empty_cache()

    mem_mb = (
        features["h_frames"].nbytes +
        features["h_vq_cells"].nbytes
    ) / 1024 / 1024
    print(f"Feature cache: {mem_mb:.0f} MB "
          f"({len(features['type_targets'])} samples)")

    # --- Step 7: Create train/val split ---
    dataset = CachedPolicyDataset(
        features["h_frames"],
        features["h_vq_cells"],
        features["type_targets"],
        features["loc_targets"],
    )

    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    # Large batch sizes since we're just training small MLPs
    train_batch = min(256, n_train)
    val_batch = min(256, n_val)

    train_dl = DataLoader(train_ds, batch_size=train_batch, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=val_batch, shuffle=False,
                        num_workers=0, pin_memory=True)

    print(f"Train: {n_train}, Val: {n_val}, Batch: {train_batch}")

    # --- Step 8: Optimizer ---
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=policy_config.learning_rate,
        weight_decay=policy_config.weight_decay,
    )

    # --- Step 9: Training loop (fast — just MLP forward/backward) ---
    print(f"\n--- Training for {policy_config.num_epochs} epochs ---")
    output_dir = Path(policy_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_type_acc = 0.0
    best_val_loc_acc = 0.0
    start_time = time.time()

    for epoch in range(1, policy_config.num_epochs + 1):
        # --- Train ---
        policy.train()
        epoch_loss = 0.0
        epoch_type_correct = 0
        epoch_loc_correct = 0
        epoch_loc_total = 0
        n_samples = 0

        for h_frames, h_vq_cells, type_targets, loc_targets in train_dl:
            h_frames = h_frames.to(device).float()
            h_vq_cells = h_vq_cells.to(device).float()
            type_targets = type_targets.to(device)
            loc_targets = loc_targets.to(device)
            B = h_frames.shape[0]

            type_logits, loc_logits = policy(h_frames, h_vq_cells)

            type_loss = F.cross_entropy(type_logits, type_targets)

            has_location = (loc_targets < 64)
            if has_location.any():
                loc_loss = F.cross_entropy(loc_logits[has_location], loc_targets[has_location])
            else:
                loc_loss = torch.tensor(0.0, device=device)

            loss = type_loss + loc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), policy_config.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item() * B
            epoch_type_correct += (type_logits.argmax(-1) == type_targets).sum().item()
            if has_location.any():
                epoch_loc_correct += (loc_logits[has_location].argmax(-1) == loc_targets[has_location]).sum().item()
                epoch_loc_total += has_location.sum().item()
            n_samples += B

        avg_loss = epoch_loss / max(n_samples, 1)
        type_acc = epoch_type_correct / max(n_samples, 1)
        loc_acc = epoch_loc_correct / max(epoch_loc_total, 1)

        # --- Eval ---
        if epoch % 5 == 0 or epoch == 1:
            policy.eval()
            val_loss = 0.0
            val_type_correct = 0
            val_loc_correct = 0
            val_loc_total = 0
            val_samples = 0

            with torch.no_grad():
                for h_frames, h_vq_cells, type_targets, loc_targets in val_dl:
                    h_frames = h_frames.to(device).float()
                    h_vq_cells = h_vq_cells.to(device).float()
                    type_targets = type_targets.to(device)
                    loc_targets = loc_targets.to(device)
                    B = h_frames.shape[0]

                    type_logits, loc_logits = policy(h_frames, h_vq_cells)
                    type_loss = F.cross_entropy(type_logits, type_targets)

                    has_location = (loc_targets < 64)
                    if has_location.any():
                        loc_loss = F.cross_entropy(loc_logits[has_location], loc_targets[has_location])
                    else:
                        loc_loss = torch.tensor(0.0, device=device)

                    val_loss += (type_loss + loc_loss).item() * B
                    val_type_correct += (type_logits.argmax(-1) == type_targets).sum().item()
                    if has_location.any():
                        val_loc_correct += (loc_logits[has_location].argmax(-1) == loc_targets[has_location]).sum().item()
                        val_loc_total += has_location.sum().item()
                    val_samples += B

            v_loss = val_loss / max(val_samples, 1)
            v_type_acc = val_type_correct / max(val_samples, 1)
            v_loc_acc = val_loc_correct / max(val_loc_total, 1)

            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{policy_config.num_epochs} | "
                f"train_loss={avg_loss:.4f} val_loss={v_loss:.4f} | "
                f"type_acc={v_type_acc:.3f} loc_acc={v_loc_acc:.3f} | "
                f"{elapsed:.0f}s"
            )

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_val_type_acc = v_type_acc
                best_val_loc_acc = v_loc_acc
                torch.save({
                    "epoch": epoch,
                    "policy_state_dict": policy.state_dict(),
                    "policy_config": policy_config,
                    "val_loss": v_loss,
                    "val_type_acc": v_type_acc,
                    "val_loc_acc": v_loc_acc,
                }, output_dir / "best.pt")
        else:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d}/{policy_config.num_epochs} | "
                f"train_loss={avg_loss:.4f} | "
                f"type_acc={type_acc:.3f} loc_acc={loc_acc:.3f} | "
                f"{elapsed:.0f}s"
            )

    # Final save
    torch.save({
        "epoch": policy_config.num_epochs,
        "policy_state_dict": policy.state_dict(),
        "policy_config": policy_config,
        "val_loss": best_val_loss,
        "val_type_acc": best_val_type_acc,
        "val_loc_acc": best_val_loc_acc,
    }, output_dir / "final.pt")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Policy Training Complete")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best type_acc: {best_val_type_acc:.3f}")
    print(f"Best loc_acc:  {best_val_loc_acc:.3f}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


def main():
    parser = argparse.ArgumentParser(description="Train policy heads")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--world-model", default="checkpoints/world_model/best.pt")
    parser.add_argument("--vqvae", default="checkpoints/vqvae/best.pt")
    parser.add_argument("--output-dir", default="checkpoints/policy")
    args = parser.parse_args()

    policy_config = PolicyConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        world_model_checkpoint=args.world_model,
        vqvae_checkpoint=args.vqvae,
        output_dir=args.output_dir,
    )

    train_policy(policy_config=policy_config)


if __name__ == "__main__":
    main()
