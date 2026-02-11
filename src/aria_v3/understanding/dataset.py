"""
Dataset class for synthetic game data.

Loads .npz sequences and produces (frames, actions, next_frames, labels) batches
for pretraining the understanding model.

Each sample is a fixed-length window of transitions from a sequence,
paired with the ground truth labels for that sequence.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Maps from string game type to integer index
GAME_TYPE_TO_IDX = {
    "navigation": 0,
    "click_puzzle": 1,
    "collection": 2,
    "mixed": 3,
    "push": 4,
    "conditional": 5,
    "physics": 6,
    "unknown": 7,
}

# Maps from string entity role to integer index
ENTITY_ROLE_TO_IDX = {
    "player": 0,
    "wall": 1,
    "collectible": 2,
    "background": 3,
    "counter": 4,
}

NUM_COLORS = 16
NUM_ACTIONS = 8
NUM_ROLES = 5


class SyntheticDataset(Dataset):
    """Dataset of synthetic game transition sequences with ground truth labels."""

    def __init__(
        self,
        data_dir: str | Path,
        window_size: int = 100,
        stage: str = "all",
        transform: bool = False,
    ):
        """Initialize dataset.

        Args:
            data_dir: Path to synthetic data directory (e.g., data/synthetic/)
            window_size: Number of transitions per sample
            stage: Which ground truth stage to use: "10", "50", "100", "200", or "all"
            transform: Whether to apply additional random augmentations on-the-fly
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stage = stage
        self.transform = transform

        # Find all .npz files
        self.files: list[Path] = sorted(self.data_dir.rglob("*.npz"))
        if not self.files:
            raise ValueError(f"No .npz files found in {data_dir}")

        # Build index: (file_idx, start_step, gt_stage_key)
        self.samples: list[tuple[int, int, str]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Build sample index from all files."""
        for file_idx, path in enumerate(self.files):
            try:
                data = np.load(path, allow_pickle=True)
                num_transitions = len(data["actions"])

                # Determine which ground truth stages to use
                gt_data = json.loads(str(data["ground_truth"]))
                available_stages = sorted(gt_data.keys(), key=lambda x: int(x))

                if self.stage == "all":
                    stages_to_use = available_stages
                else:
                    stages_to_use = [self.stage] if self.stage in available_stages else available_stages[-1:]

                for stage_key in stages_to_use:
                    stage_step = int(stage_key)
                    # The window ends at stage_step (or at num_transitions)
                    end = min(stage_step, num_transitions)
                    start = max(0, end - self.window_size)
                    if end - start >= 5:  # need at least 5 transitions
                        self.samples.append((file_idx, start, stage_key))

                data.close()
            except Exception:
                continue  # skip corrupted files

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample.

        Returns dict with:
            frames: [L, 64, 64] long tensor
            actions: [L] long tensor
            next_frames: [L, 64, 64] long tensor
            mask: [window_size] bool tensor (True for valid positions)
            --- Labels ---
            action_shifts: [NUM_ACTIONS, 2] float (dx, dy per action)
            action_change_prob: [NUM_ACTIONS] float
            action_blocked_prob: [NUM_ACTIONS] float
            action_affected_color: [NUM_ACTIONS] long
            entity_roles: [NUM_COLORS, NUM_ROLES] float (one-hot per color)
            game_type: long scalar (0-7)
            confidence: float scalar (0-1)
        """
        file_idx, start, stage_key = self.samples[idx]
        data = np.load(self.files[file_idx], allow_pickle=True)

        # Extract transition window
        end = start + self.window_size
        frames = data["frames"][start:end]       # [L', 64, 64]
        actions = data["actions"][start:end]      # [L']
        next_frames = data["next_frames"][start:end]  # [L', 64, 64]
        actual_len = len(frames)

        # Pad to window_size if needed
        frames_padded = np.zeros((self.window_size, 64, 64), dtype=np.int64)
        actions_padded = np.zeros(self.window_size, dtype=np.int64)
        next_frames_padded = np.zeros((self.window_size, 64, 64), dtype=np.int64)
        mask = np.zeros(self.window_size, dtype=bool)

        frames_padded[:actual_len] = frames
        actions_padded[:actual_len] = actions
        next_frames_padded[:actual_len] = next_frames
        mask[:actual_len] = True

        # Parse ground truth labels
        gt_data = json.loads(str(data["ground_truth"]))
        gt = gt_data[stage_key]

        # Action effects
        action_shifts = np.zeros((NUM_ACTIONS, 2), dtype=np.float32)
        action_change_prob = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_blocked_prob = np.zeros(NUM_ACTIONS, dtype=np.float32)
        action_affected_color = np.zeros(NUM_ACTIONS, dtype=np.int64)

        for aid_str, eff in gt.get("action_effects", {}).items():
            aid = int(aid_str)
            if 0 <= aid < NUM_ACTIONS:
                action_shifts[aid, 0] = eff.get("shift_dx", 0.0)
                action_shifts[aid, 1] = eff.get("shift_dy", 0.0)
                action_change_prob[aid] = eff.get("change_prob", 0.0)
                action_blocked_prob[aid] = eff.get("blocked_prob", 0.0)
                action_affected_color[aid] = eff.get("affected_color", 0)

        # Entity roles (one-hot per color)
        entity_roles = np.zeros((NUM_COLORS, NUM_ROLES), dtype=np.float32)
        for color_str, info in gt.get("entities", {}).items():
            color = int(color_str)
            if 0 <= color < NUM_COLORS:
                role_str = info.get("role", "background")
                role_idx = ENTITY_ROLE_TO_IDX.get(role_str, 3)  # default background
                entity_roles[color, role_idx] = 1.0

        # Fill unassigned colors as background
        for c in range(NUM_COLORS):
            if entity_roles[c].sum() == 0:
                entity_roles[c, ENTITY_ROLE_TO_IDX["background"]] = 1.0

        # Game type
        game_type_str = gt.get("game_type", str(data["game_type"]))
        game_type = GAME_TYPE_TO_IDX.get(game_type_str, 7)

        # Confidence
        confidence = float(gt.get("confidence", 0.5))

        data.close()

        return {
            "frames": torch.from_numpy(frames_padded),
            "actions": torch.from_numpy(actions_padded),
            "next_frames": torch.from_numpy(next_frames_padded),
            "mask": torch.from_numpy(mask),
            # Labels
            "action_shifts": torch.from_numpy(action_shifts),
            "action_change_prob": torch.from_numpy(action_change_prob),
            "action_blocked_prob": torch.from_numpy(action_blocked_prob),
            "action_affected_color": torch.from_numpy(action_affected_color),
            "entity_roles": torch.from_numpy(entity_roles),
            "game_type": torch.tensor(game_type, dtype=torch.long),
            "confidence": torch.tensor(confidence, dtype=torch.float32),
        }


def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Custom collate that stacks all tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}
