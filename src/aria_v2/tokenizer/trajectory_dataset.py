"""
Trajectory Dataset: JSONL demos → tokenized sequences for SmolLM2.

Loads human recordings, tokenizes frames through a frozen VQ-VAE,
and creates sliding-window training sequences with special tokens.

Vocabulary (589 new tokens added to SmolLM2's 49,152):
    <VQ_000>..<VQ_511>      - 512 frame codes        (IDs 49152-49663)
    <ACT_TYPE_0>..<ACT_TYPE_7> - 8 action types       (IDs 49664-49671)
    <ACT_LOC_0>..<ACT_LOC_64>  - 65 action locations  (IDs 49672-49736)
        0-63 = VQ cell index (8x8 spatial grid)
        64 = NULL (non-spatial actions like navigation)
    <FRAME>                 - frame start marker      (ID 49737)
    <ACT>                   - action marker           (ID 49738)
    <LEVEL_COMPLETE>        - level boundary          (ID 49739)
    <GAME_START>            - trajectory start         (ID 49740)

Sequence format (69 tokens per step):
    <GAME_START> <FRAME> v1..v64 <ACT> <ACT_TYPE_i> <ACT_LOC_j> <FRAME> v1..v64 ...
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .frame_tokenizer import FrameVQVAE

# Token ID offsets (added to SmolLM2's base vocab of 49152)
BASE_VOCAB_SIZE = 49152
VQ_OFFSET = BASE_VOCAB_SIZE            # 49152-49663 (512 VQ codes)
ACT_TYPE_OFFSET = VQ_OFFSET + 512      # 49664-49671 (8 action types)
ACT_LOC_OFFSET = ACT_TYPE_OFFSET + 8   # 49672-49736 (65 locations: 64 cells + NULL)
ACT_LOC_NULL = ACT_LOC_OFFSET + 64     # 49736 (non-spatial action)
FRAME_TOKEN = ACT_LOC_OFFSET + 65      # 49737
ACT_TOKEN = FRAME_TOKEN + 1            # 49738
LEVEL_COMPLETE = ACT_TOKEN + 1         # 49739
GAME_START = LEVEL_COMPLETE + 1        # 49740
TOTAL_NEW_TOKENS = 589                 # 512 + 8 + 65 + 4 special
TOTAL_VOCAB_SIZE = BASE_VOCAB_SIZE + TOTAL_NEW_TOKENS  # 49741

# For policy mask token (added during policy training, not world model)
MASK_TOKEN = GAME_START + 1            # 49741
TOTAL_VOCAB_WITH_MASK = TOTAL_VOCAB_SIZE + 1  # 49742

# Backward-compat aliases (old code used these names)
ACT_OFFSET = ACT_TYPE_OFFSET  # Old code used ACT_OFFSET for action types


def pixel_to_vq_cell(x: int, y: int) -> int:
    """Convert pixel coordinates to VQ cell index (8x8 grid).

    VQ-VAE encodes 64x64 frames into 8x8 grids. Each cell covers 8x8 pixels.
    Cell index = row * 8 + col, where row = y // 8, col = x // 8.
    """
    row = min(y // 8, 7)  # Clamp to 0-7
    col = min(x // 8, 7)
    return row * 8 + col


def vq_cell_to_pixel(cell: int) -> tuple[int, int]:
    """Convert VQ cell index to center pixel coordinates.

    Returns (x, y) at the center of the 8x8 cell.
    """
    row = cell // 8
    col = cell % 8
    x = col * 8 + 4  # Center of cell
    y = row * 8 + 4
    return x, y


@dataclass
class TokenizedTrajectory:
    """A full tokenized trajectory."""
    game_id: str
    tokens: list[int]  # Full token sequence
    token_types: list[str]  # "vq", "action_type", "action_loc", "frame", "act", "level", "start"


def get_special_tokens() -> dict[str, int]:
    """Return mapping of special token names to IDs."""
    tokens = {}
    for i in range(512):
        tokens[f"<VQ_{i:03d}>"] = VQ_OFFSET + i
    for i in range(8):
        tokens[f"<ACT_TYPE_{i}>"] = ACT_TYPE_OFFSET + i
    for i in range(64):
        tokens[f"<ACT_LOC_{i}>"] = ACT_LOC_OFFSET + i
    tokens["<ACT_LOC_NULL>"] = ACT_LOC_NULL
    tokens["<FRAME>"] = FRAME_TOKEN
    tokens["<ACT>"] = ACT_TOKEN
    tokens["<LEVEL_COMPLETE>"] = LEVEL_COMPLETE
    tokens["<GAME_START>"] = GAME_START
    tokens["<MASK>"] = MASK_TOKEN
    return tokens


def tokenize_trajectory(
    vqvae: FrameVQVAE,
    jsonl_path: str | Path,
    device: str = "cuda",
    batch_size: int = 128,
) -> TokenizedTrajectory | None:
    """
    Tokenize a single JSONL demo file into a token sequence.

    Reads frames and actions from JSONL, encodes frames through VQ-VAE,
    and builds a token sequence with unified (type, location) action pairs
    and LEVEL_COMPLETE markers at score increases.
    """
    path = Path(jsonl_path)
    game_id = path.stem.split("-")[0] if "-" in path.stem else "unknown"

    # Parse JSONL
    raw_frames = []
    actions = []       # action type ID (0-7)
    action_locs = []   # VQ cell index (0-63) or 64 for NULL
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

            # Extract click coordinates if present
            action_data = action_input.get("data", {}) if action_input else {}
            x = action_data.get("x")
            y = action_data.get("y")

            if x is not None and y is not None:
                # Spatial action (click) — map to VQ cell
                loc = pixel_to_vq_cell(int(x), int(y))
            else:
                # Non-spatial action (navigation, reset, etc.)
                loc = 64  # NULL

            raw_frames.append(frame)
            actions.append(min(action_type, 7))  # Clamp to valid range
            action_locs.append(loc)
            scores.append(score)

    if len(raw_frames) < 2:
        return None

    # Encode all frames through VQ-VAE in batches
    vqvae.eval()
    all_indices = []
    for i in range(0, len(raw_frames), batch_size):
        batch = torch.tensor(
            np.stack(raw_frames[i:i + batch_size]), dtype=torch.long
        ).to(device)
        indices = vqvae.encode(batch)  # [B, 8, 8]
        all_indices.append(indices.cpu())
    all_indices = torch.cat(all_indices, dim=0)  # [N, 8, 8]

    # Build token sequence
    tokens = [GAME_START]
    token_types = ["start"]

    for t in range(len(raw_frames)):
        # Frame tokens
        tokens.append(FRAME_TOKEN)
        token_types.append("frame")

        frame_codes = all_indices[t].flatten().tolist()  # 64 codes
        for code in frame_codes:
            tokens.append(VQ_OFFSET + code)
            token_types.append("vq")

        # Action tokens: [ACT] <ACT_TYPE_i> <ACT_LOC_j>
        tokens.append(ACT_TOKEN)
        token_types.append("act")

        tokens.append(ACT_TYPE_OFFSET + actions[t])
        token_types.append("action_type")

        tokens.append(ACT_LOC_OFFSET + action_locs[t])
        token_types.append("action_loc")

        # Level completion: score increased
        if t + 1 < len(scores) and scores[t + 1] > scores[t]:
            tokens.append(LEVEL_COMPLETE)
            token_types.append("level")

    return TokenizedTrajectory(
        game_id=game_id,
        tokens=tokens,
        token_types=token_types,
    )


def tokenize_all_demos(
    vqvae: FrameVQVAE,
    demo_dir: str | Path,
    cache_path: str | Path | None = None,
    device: str = "cuda",
) -> list[TokenizedTrajectory]:
    """Tokenize all JSONL demos in directory. Optionally cache to disk."""
    demo_dir = Path(demo_dir)
    trajectories = []

    for jsonl_path in sorted(demo_dir.rglob("*.jsonl")):
        traj = tokenize_trajectory(vqvae, jsonl_path, device=device)
        if traj:
            trajectories.append(traj)
            # Count action types and locations
            n_spatial = sum(1 for tt in traj.token_types if tt == "action_loc")
            print(f"  {jsonl_path.name}: {len(traj.tokens)} tokens, "
                  f"game={traj.game_id}")

    total_tokens = sum(len(t.tokens) for t in trajectories)
    print(f"Tokenized {len(trajectories)} trajectories, {total_tokens} total tokens")

    # Cache to disk
    if cache_path:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "trajectories": [
                {"game_id": t.game_id, "tokens": t.tokens, "token_types": t.token_types}
                for t in trajectories
            ],
            "version": 2,  # New format with unified action tokens
        }, cache_path)
        print(f"Cached to {cache_path}")

    return trajectories


def load_cached_trajectories(cache_path: str | Path) -> list[TokenizedTrajectory]:
    """Load previously cached tokenized trajectories."""
    data = torch.load(cache_path, weights_only=False)
    version = data.get("version", 1)
    if version < 2:
        raise ValueError(
            f"Cache version {version} is outdated (need v2 with unified action tokens). "
            f"Delete {cache_path} and re-tokenize."
        )
    return [
        TokenizedTrajectory(**t) for t in data["trajectories"]
    ]


class TrajectoryWindowDataset(Dataset):
    """
    Sliding-window dataset over tokenized trajectories.

    Creates fixed-length windows for training. Windows containing
    LEVEL_COMPLETE tokens are oversampled 3x.
    """

    def __init__(
        self,
        trajectories: list[TokenizedTrajectory],
        window_size: int = 2048,
        stride: int = 690,  # ~10 frames * 69 tokens/frame
        level_complete_oversample: int = 3,
    ):
        self.window_size = window_size
        self.windows: list[torch.Tensor] = []
        self.token_type_windows: list[list[str]] = []

        for traj in trajectories:
            tokens = traj.tokens
            types = traj.token_types

            # Create windows
            for start in range(0, max(1, len(tokens) - window_size + 1), stride):
                end = min(start + window_size, len(tokens))
                window = tokens[start:end]
                type_window = types[start:end]

                # Pad if needed
                if len(window) < window_size:
                    pad_len = window_size - len(window)
                    window = window + [0] * pad_len  # Pad with 0 (will be masked)
                    type_window = type_window + ["pad"] * pad_len

                self.windows.append(torch.tensor(window, dtype=torch.long))
                self.token_type_windows.append(type_window)

                # Oversample windows with LEVEL_COMPLETE
                has_level = LEVEL_COMPLETE in tokens[start:end]
                if has_level:
                    for _ in range(level_complete_oversample - 1):
                        self.windows.append(torch.tensor(window, dtype=torch.long))
                        self.token_type_windows.append(type_window)

        print(f"Created {len(self.windows)} training windows "
              f"(window_size={window_size}, stride={stride})")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        tokens = self.windows[idx]
        types = self.token_type_windows[idx]

        # Build loss weight mask based on token types
        weights = torch.zeros(len(types), dtype=torch.float32)
        for i, t in enumerate(types):
            if t == "vq":
                weights[i] = 1.0
            elif t == "action_type":
                weights[i] = 3.0
            elif t == "action_loc":
                weights[i] = 3.0
            elif t == "level":
                weights[i] = 5.0
            # frame, act, start, pad → weight 0.0

        return tokens, weights
