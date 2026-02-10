"""
Online CNN that learns P(frame_change | state, action) during gameplay.

This is the core learning component. The model:
1. Takes a 16-channel one-hot encoded 64x64 frame
2. Predicts a scalar P(change) for each candidate action
3. Trains online from (frame, action, did_change) tuples
4. Resets per level to prevent catastrophic forgetting

Architecture follows StochasticGoose (1st place preview) with lighter action head:
- 4-layer conv backbone (16→32→64→64 channels)
- Simple action head for non-spatial actions (1-5)
- Region head for click actions (masked pooling per region)
"""

import hashlib
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .frame_processor import Region


@dataclass
class Experience:
    """A single (state, action, reward) tuple."""
    frame: np.ndarray       # [64, 64] int array
    action_idx: int         # index into the unified action space
    reward: float           # 1.0 if frame changed, 0.0 if not


class ChangePredictor(nn.Module):
    """CNN that predicts P(frame_change | state, action).

    The output is a vector of sigmoid probabilities, one per action.
    For simple actions (1-5): 5 outputs from a pooled action head.
    For click actions: 1 output per region via masked average pooling.

    Total actions = num_simple + num_regions (varies per frame).
    """

    def __init__(self, num_simple_actions: int = 5):
        super().__init__()
        self.num_simple_actions = num_simple_actions

        # Shared backbone: 4 conv layers, no pooling
        self.backbone = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )

        # Action head: for simple (non-spatial) actions
        self.action_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),  # [B, 64, 4, 4]
            nn.Flatten(),             # [B, 1024]
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_simple_actions),  # [B, 5]
        )

        # Region head: project features to scalar per region
        self.region_proj = nn.Linear(64, 1)  # applied to pooled region features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature map from frame.

        Args:
            x: [B, 16, 64, 64] one-hot encoded frame

        Returns:
            [B, 64, 64, 64] feature map
        """
        return self.backbone(x)

    def forward_actions(self, features: torch.Tensor) -> torch.Tensor:
        """Predict change probability for simple actions.

        Args:
            features: [B, 64, 64, 64] from backbone

        Returns:
            [B, num_simple_actions] logits
        """
        return self.action_head(features)

    def forward_regions(
        self,
        features: torch.Tensor,
        region_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Predict change probability for click regions.

        Args:
            features: [1, 64, 64, 64] from backbone (single frame)
            region_masks: List of [64, 64] boolean tensors, one per region

        Returns:
            [num_regions] logits
        """
        if not region_masks:
            return torch.zeros(0, device=features.device)

        feat = features[0]  # [64, 64, 64]
        logits = []

        for mask in region_masks:
            # Masked average pooling: mean of feature vectors at region pixels
            mask_expanded = mask.unsqueeze(0).float()  # [1, 64, 64]
            masked_feat = feat * mask_expanded  # [64, 64, 64]
            num_pixels = mask_expanded.sum().clamp(min=1.0)
            pooled = masked_feat.sum(dim=(1, 2)) / num_pixels  # [64]
            logit = self.region_proj(pooled)  # [1]
            logits.append(logit.squeeze(-1))

        return torch.stack(logits)  # [num_regions]


class OnlineLearner:
    """Manages online training of the ChangePredictor.

    Collects experiences, deduplicates them, and trains the model
    every N steps. Resets on level change.
    """

    def __init__(
        self,
        device: str = "cuda",
        buffer_size: int = 50000,
        batch_size: int = 64,
        lr: float = 1e-4,
        train_every: int = 10,
        num_simple_actions: int = 5,
    ):
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.train_every = train_every

        self.model = ChangePredictor(num_simple_actions).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.buffer: deque[Experience] = deque(maxlen=buffer_size)
        self.experience_hashes: set[str] = set()
        self.step_count = 0
        self.train_count = 0

    def frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Convert frame to one-hot tensor.

        Args:
            frame: [64, 64] int array with values 0-15

        Returns:
            [1, 16, 64, 64] float tensor
        """
        tensor = torch.zeros(16, 64, 64, dtype=torch.float32)
        frame_torch = torch.from_numpy(frame.astype(np.int64)).unsqueeze(0)  # [1, 64, 64]
        tensor.scatter_(0, frame_torch, 1.0)
        return tensor.unsqueeze(0).to(self.device)  # [1, 16, 64, 64]

    def predict(
        self,
        frame: np.ndarray,
        regions: list[Region],
        available_simple: list[int],
    ) -> np.ndarray:
        """Predict P(frame_change) for all available actions.

        Args:
            frame: [64, 64] int array
            regions: List of Region objects (for click actions)
            available_simple: List of available simple action IDs (1-5)

        Returns:
            Array of probabilities, shape [num_simple + num_regions]
            First entries correspond to simple actions, rest to regions.
        """
        self.model.eval()
        with torch.no_grad():
            x = self.frame_to_tensor(frame)
            features = self.model.forward_features(x)

            # Simple action predictions
            action_logits = self.model.forward_actions(features)  # [1, 5]
            action_probs = torch.sigmoid(action_logits[0]).cpu().numpy()

            # Filter to available simple actions and map to indices
            simple_probs = []
            for aid in available_simple:
                idx = aid - 1  # action IDs are 1-indexed, head outputs are 0-indexed
                if 0 <= idx < len(action_probs):
                    simple_probs.append(float(action_probs[idx]))
                else:
                    simple_probs.append(0.5)  # unknown action, neutral prior

            # Region predictions (click actions)
            if regions:
                region_masks = [
                    torch.from_numpy(r.mask).to(self.device) for r in regions
                ]
                region_logits = self.model.forward_regions(features, region_masks)
                region_probs = torch.sigmoid(region_logits).cpu().numpy()
            else:
                region_probs = np.array([])

            return np.concatenate([simple_probs, region_probs])

    def add_experience(
        self, frame: np.ndarray, action_idx: int, frame_changed: bool
    ) -> None:
        """Add a (state, action, reward) experience to the buffer.

        Deduplicates by (frame_hash, action_idx).
        """
        # Hash for dedup
        hash_input = frame.tobytes() + str(action_idx).encode()
        exp_hash = hashlib.md5(hash_input).hexdigest()

        if exp_hash in self.experience_hashes:
            return

        self.experience_hashes.add(exp_hash)
        self.buffer.append(Experience(
            frame=frame.copy(),
            action_idx=action_idx,
            reward=1.0 if frame_changed else 0.0,
        ))
        self.step_count += 1

    def maybe_train(self) -> float | None:
        """Train one batch if enough steps have passed.

        Returns:
            Loss value if training happened, None otherwise.
        """
        if self.step_count % self.train_every != 0:
            return None
        if len(self.buffer) < self.batch_size:
            return None

        return self._train_step()

    def _train_step(self) -> float:
        """Run one training step on a random batch."""
        self.model.train()

        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)

        # Build batch tensors
        frames = []
        action_indices = []
        rewards = []

        for idx in indices:
            exp = self.buffer[idx]
            frames.append(self.frame_to_tensor(exp.frame))
            action_indices.append(exp.action_idx)
            rewards.append(exp.reward)

        frames_batch = torch.cat(frames, dim=0)  # [B, 16, 64, 64]
        rewards_batch = torch.tensor(rewards, device=self.device)
        action_indices_batch = torch.tensor(action_indices, dtype=torch.long, device=self.device)

        # Forward
        features = self.model.forward_features(frames_batch)
        action_logits = self.model.forward_actions(features)  # [B, 5]

        # For simple actions: gather the logit for the taken action
        # Action indices in buffer use the unified space:
        #   0..num_simple-1 = simple actions, num_simple+ = regions
        # For region actions, we skip training on this batch (would need masks)
        num_simple = self.model.num_simple_actions
        simple_mask = action_indices_batch < num_simple
        if simple_mask.any():
            simple_logits = action_logits[simple_mask]
            simple_actions = action_indices_batch[simple_mask]
            simple_rewards = rewards_batch[simple_mask]

            selected = simple_logits.gather(
                1, simple_actions.unsqueeze(1)
            ).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(selected, simple_rewards)
        else:
            loss = torch.tensor(0.0, device=self.device)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_count += 1
        return loss.item()

    def sample_action(
        self,
        frame: np.ndarray,
        regions: list[Region],
        available_simple: list[int],
        has_click: bool,
    ) -> int:
        """Sample an action proportional to predicted change probability.

        Args:
            frame: [64, 64] int array
            regions: List of regions for click actions
            available_simple: Available simple action IDs (1-indexed)
            has_click: Whether click (action 6) is available

        Returns:
            Action index in unified space (0..num_simple-1 for simple,
            num_simple+ for regions)
        """
        probs = self.predict(frame, regions if has_click else [], available_simple)

        if len(probs) == 0:
            return 0

        # Normalize: scale region probs down to compete fairly with simple actions
        num_simple = len(available_simple)
        num_regions = len(probs) - num_simple

        if num_regions > 0 and num_simple > 0:
            # Scale region probs down so total region weight ~ total simple weight
            region_scale = max(num_simple / max(num_regions, 1), 0.01)
            probs[num_simple:] *= region_scale

        # Ensure minimum exploration probability
        probs = np.clip(probs, 0.01, None)

        # Normalize to probability distribution
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs = np.ones_like(probs) / len(probs)

        return int(np.random.choice(len(probs), p=probs))

    def reset(self) -> None:
        """Reset model and buffer for a new level."""
        # Re-initialize model weights
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Reset optimizer state
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.optimizer.defaults["lr"])

        # Clear buffer
        self.buffer.clear()
        self.experience_hashes.clear()
        self.step_count = 0
        self.train_count = 0

    def get_stats(self) -> dict:
        return {
            "buffer_size": len(self.buffer),
            "unique_experiences": len(self.experience_hashes),
            "train_steps": self.train_count,
            "total_steps": self.step_count,
        }
