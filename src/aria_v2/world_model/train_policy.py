#!/usr/bin/env python3
"""
Train policy heads on frozen backbone with masked action tokens.

The backbone (SmolLM2 + LoRA world model) is frozen. Only the policy heads
(ActionTypeHead + ActionLocationHead, ~500K params) are trained.

Action tokens in the context are replaced with MASK tokens so the policy
can only learn from visual consequences, not from action token copying.

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
    TrajectoryWindowDataset, TokenizedTrajectory,
    VQ_OFFSET, ACT_TYPE_OFFSET, ACT_LOC_OFFSET, ACT_LOC_NULL,
    FRAME_TOKEN, ACT_TOKEN, LEVEL_COMPLETE, GAME_START,
    MASK_TOKEN,
)


def find_action_positions(token_types: list[str]) -> list[dict]:
    """Find all action (type, loc) pairs and their context in a window.

    Returns list of dicts with:
        - type_target: action type index (0-7)
        - loc_target: action location index (0-64, 64=NULL)
        - last_vq_pos: position of last VQ token before this action
        - vq_positions: list of 64 positions of VQ tokens in current frame
    """
    actions = []
    current_frame_vq_positions = []
    last_frame_start = -1

    for i, tt in enumerate(token_types):
        if tt == "frame":
            current_frame_vq_positions = []
            last_frame_start = i
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


class PolicyDataset(Dataset):
    """Dataset that extracts (masked_context, action_targets) from windows.

    For each training window, finds all action positions and creates
    samples where:
    - Input: window with action tokens (ACT, ACT_TYPE, ACT_LOC) replaced by MASK
    - Target: (action_type, action_loc) at each action position
    """

    def __init__(
        self,
        trajectories: list[TokenizedTrajectory],
        window_size: int = 2048,
        stride: int = 690,
    ):
        self.window_size = window_size
        self.samples: list[dict] = []

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

                # Find action positions in this window
                actions = find_action_positions(window_types)
                if not actions:
                    continue

                # Create masked version (replace action tokens with MASK)
                masked_tokens = list(window_tokens)
                for i, tt in enumerate(window_types):
                    if tt in ("act", "action_type", "action_loc"):
                        masked_tokens[i] = MASK_TOKEN

                for action in actions:
                    type_pos = action["type_pos"]
                    loc_pos = type_pos + 1  # action_loc always follows action_type

                    if loc_pos >= len(window_tokens):
                        continue

                    type_target = window_tokens[type_pos] - ACT_TYPE_OFFSET
                    loc_target = window_tokens[loc_pos] - ACT_LOC_OFFSET

                    if not (0 <= type_target < 8) or not (0 <= loc_target < 65):
                        continue

                    self.samples.append({
                        "masked_tokens": torch.tensor(masked_tokens, dtype=torch.long),
                        "type_target": type_target,
                        "loc_target": loc_target,
                        "last_vq_pos": action["last_vq_pos"],
                        "vq_positions": action["vq_positions"],
                    })

        print(f"PolicyDataset: {len(self.samples)} action samples from {len(trajectories)} trajectories")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["masked_tokens"],
            s["type_target"],
            s["loc_target"],
            s["last_vq_pos"],
            torch.tensor(s["vq_positions"], dtype=torch.long),
        )


def train_policy(
    policy_config: PolicyConfig | None = None,
    model_config: WorldModelConfig | None = None,
):
    """Train policy heads on frozen backbone."""
    policy_config = policy_config or PolicyConfig()
    model_config = model_config or WorldModelConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Policy Head Training (frozen backbone)")
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

    # --- Step 3: Create policy dataset ---
    print("\n--- Creating Policy Dataset ---")
    dataset = PolicyDataset(trajectories, window_size=2048, stride=690)

    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_dl = DataLoader(train_ds, batch_size=policy_config.batch_size,
                          shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=policy_config.batch_size,
                        shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {n_train}, Val: {n_val}")

    # --- Step 4: Load backbone (frozen) ---
    print("\n--- Loading Backbone (frozen) ---")
    wm_ckpt = torch.load(policy_config.world_model_checkpoint, weights_only=False, map_location=device)
    backbone_config = wm_ckpt.get("model_config", model_config)
    backbone = create_game_transformer(backbone_config)
    backbone.load_state_dict(wm_ckpt["model_state_dict"])
    backbone = backbone.to(device)
    backbone.eval()

    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False

    # Extend backbone embeddings to include MASK token
    # We inject the mask embedding directly into the input
    print("Backbone frozen, all parameters non-trainable")

    # --- Step 5: Create policy heads ---
    print("\n--- Creating Policy Heads ---")
    policy = PolicyHeads(policy_config).to(device)
    print(f"Policy parameters: {policy.count_parameters():,}")

    # --- Step 6: Optimizer ---
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=policy_config.learning_rate,
        weight_decay=policy_config.weight_decay,
    )

    # --- Step 7: Training loop ---
    print(f"\n--- Training for {policy_config.num_epochs} epochs ---")
    output_dir = Path(policy_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    start_time = time.time()

    use_amp = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    for epoch in range(1, policy_config.num_epochs + 1):
        # --- Train ---
        policy.train()
        epoch_loss = 0.0
        epoch_type_correct = 0
        epoch_loc_correct = 0
        epoch_loc_total = 0
        n_samples = 0

        for masked_tokens, type_targets, loc_targets, last_vq_pos, vq_positions in train_dl:
            masked_tokens = masked_tokens.to(device)
            type_targets = type_targets.to(device)
            loc_targets = loc_targets.to(device)

            B = masked_tokens.shape[0]

            # Replace MASK token IDs with mask embedding in the input
            # First, get backbone embeddings
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    # Get input embeddings manually
                    input_embeds = backbone.get_input_embeddings()(masked_tokens)

                    # Replace MASK positions with learned mask embedding
                    mask_positions = (masked_tokens == MASK_TOKEN)
                    input_embeds[mask_positions] = policy.mask_embedding.to(input_embeds.dtype)

                    # Forward through backbone
                    outputs = backbone(inputs_embeds=input_embeds, output_hidden_states=True)
                    hidden_states = outputs.hidden_states[-1]  # [B, T, 960]

            # Extract h_frame and h_vq_cells for each sample
            h_frames = []
            h_vq_cells_list = []

            for b in range(B):
                lvp = last_vq_pos[b].item()
                vp = vq_positions[b].tolist()

                h_frames.append(hidden_states[b, lvp])
                h_vq = hidden_states[b, vp]  # [64, 960]
                h_vq_cells_list.append(h_vq)

            h_frame = torch.stack(h_frames)  # [B, 960]
            h_vq_cells = torch.stack(h_vq_cells_list)  # [B, 64, 960]

            # Policy forward
            type_logits, loc_logits = policy(h_frame.float(), h_vq_cells.float())

            # Loss
            type_loss = F.cross_entropy(type_logits, type_targets)

            # Location loss only for spatial actions (loc != NULL)
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

            # Metrics
            epoch_loss += loss.item() * B
            epoch_type_correct += (type_logits.argmax(-1) == type_targets).sum().item()
            if has_location.any():
                epoch_loc_correct += (loc_logits[has_location].argmax(-1) == loc_targets[has_location]).sum().item()
                epoch_loc_total += has_location.sum().item()
            n_samples += B

        # Average metrics
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
                for masked_tokens, type_targets, loc_targets, last_vq_pos, vq_positions in val_dl:
                    masked_tokens = masked_tokens.to(device)
                    type_targets = type_targets.to(device)
                    loc_targets = loc_targets.to(device)
                    B = masked_tokens.shape[0]

                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        input_embeds = backbone.get_input_embeddings()(masked_tokens)
                        mask_positions = (masked_tokens == MASK_TOKEN)
                        input_embeds[mask_positions] = policy.mask_embedding.to(input_embeds.dtype)
                        outputs = backbone(inputs_embeds=input_embeds, output_hidden_states=True)
                        hidden_states = outputs.hidden_states[-1]

                    h_frames = []
                    h_vq_cells_list = []
                    for b in range(B):
                        h_frames.append(hidden_states[b, last_vq_pos[b].item()])
                        h_vq_cells_list.append(hidden_states[b, vq_positions[b].tolist()])

                    h_frame = torch.stack(h_frames)
                    h_vq_cells = torch.stack(h_vq_cells_list)

                    type_logits, loc_logits = policy(h_frame.float(), h_vq_cells.float())
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
                torch.save({
                    "epoch": epoch,
                    "policy_state_dict": policy.state_dict(),
                    "policy_config": policy_config,
                    "val_loss": v_loss,
                    "val_type_acc": v_type_acc,
                    "val_loc_acc": v_loc_acc,
                }, output_dir / "best.pt")

    # Final save
    torch.save({
        "epoch": policy_config.num_epochs,
        "policy_state_dict": policy.state_dict(),
        "policy_config": policy_config,
        "val_loss": v_loss,
        "val_type_acc": v_type_acc,
        "val_loc_acc": v_loc_acc,
    }, output_dir / "final.pt")

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Policy Training Complete")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


def main():
    parser = argparse.ArgumentParser(description="Train policy heads")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
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
