"""
Memory task with explicit value extraction.

Key insight: The model needs to learn to:
1. Extract the center value in show phase
2. Map it to the correct click position in hide phase

We use an architecture that explicitly extracts center features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aria_lite.primitives import StateTrackingEnv
from src.aria_lite.primitives.base import Action


class MemoryPolicy(nn.Module):
    """
    Policy that explicitly handles memory task structure.

    Architecture:
    1. Extract center region features (where value is shown)
    2. Extract bottom row features (where options are shown)
    3. Learn to match center value to correct option
    """

    def __init__(self, hidden_dim: int = 64, num_values: int = 8):
        super().__init__()
        self.num_values = num_values

        # Embedding
        self.embed = nn.Embedding(num_values + 1, 16)  # +1 for 0 (empty)

        # Center feature extractor (for show phase)
        self.center_proj = nn.Sequential(
            nn.Linear(16 * 9, hidden_dim),  # 3x3 center region
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Full grid encoder
        self.grid_encoder = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(64 * 4, hidden_dim),
        )

        # Memory cell (stores center value encoding)
        self.memory = None

        # Output heads
        self.action_head = nn.Linear(hidden_dim * 2, 9)
        self.x_head = nn.Linear(hidden_dim * 2, 16)

    def forward(self, obs: torch.Tensor, store_memory: bool = False):
        """
        Args:
            obs: [B, H, W] observation
            store_memory: if True, store center features in memory
        """
        B, H, W = obs.shape
        device = obs.device

        # Embed
        embedded = self.embed(obs.long().clamp(0, self.num_values))  # [B, H, W, 16]

        # Extract center region (where value is shown)
        center_y, center_x = H // 2, W // 2
        center_region = embedded[:, center_y-1:center_y+2, center_x-1:center_x+2, :]  # [B, 3, 3, 16]
        center_flat = center_region.reshape(B, -1)  # [B, 144]
        center_features = self.center_proj(center_flat)  # [B, hidden_dim]

        # Encode full grid
        grid_embed = embedded.permute(0, 3, 1, 2)  # [B, 16, H, W]
        grid_features = self.grid_encoder(grid_embed)  # [B, hidden_dim]

        # Memory handling
        if store_memory:
            self.memory = center_features.detach()

        # Combine with memory
        if self.memory is not None and self.memory.shape[0] == B:
            combined = torch.cat([grid_features, self.memory], dim=-1)
        else:
            combined = torch.cat([grid_features, center_features], dim=-1)

        # Output
        action_logits = self.action_head(combined)
        x_logits = self.x_head(combined)

        return {
            "action_logits": action_logits,
            "x_logits": x_logits,
            "center_features": center_features,
        }

    def reset_memory(self):
        self.memory = None


def collect_memory_data(num_episodes: int = 1000):
    """Collect memory task data with explicit phase labels."""
    data = []

    for _ in range(num_episodes):
        env = StateTrackingEnv(grid_size=8, num_values=4, max_steps=15, variant="memory")
        obs_show = env.reset()

        # Extract target value from center (stored as value+1)
        center = env.grid_size // 2
        target_value = obs_show[center, center].item() - 1

        # Step 1: Show phase -> advance
        result = env.step(Action.INTERACT)
        obs_hide = result.observation

        # Step 2: Hide phase -> click correct value
        target_x = target_value * 2

        data.append({
            "obs_show": obs_show.clone(),
            "obs_hide": obs_hide.clone(),
            "target_value": target_value,
            "target_x": target_x,
        })

    return data


def train_memory():
    """Train memory task with explicit architecture."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    print("Collecting data...")
    data = collect_memory_data(1000)
    print(f"Collected {len(data)} episodes")

    # Stack data
    obs_show = torch.stack([d["obs_show"] for d in data])
    obs_hide = torch.stack([d["obs_hide"] for d in data])
    target_values = torch.tensor([d["target_value"] for d in data], dtype=torch.long)
    target_xs = torch.tensor([d["target_x"] for d in data], dtype=torch.long)

    print(f"Target value distribution: {torch.bincount(target_values)}")

    model = MemoryPolicy(hidden_dim=64, num_values=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    num_epochs = 100

    for epoch in range(num_epochs):
        perm = torch.randperm(len(data))
        obs_show = obs_show[perm]
        obs_hide = obs_hide[perm]
        target_values = target_values[perm]
        target_xs = target_xs[perm]

        total_loss = 0
        correct = 0

        for i in range(0, len(data), batch_size):
            show_batch = obs_show[i:i+batch_size].to(device)
            hide_batch = obs_hide[i:i+batch_size].to(device)
            target_x_batch = target_xs[i:i+batch_size].to(device)

            # Process show phase (store memory)
            model.reset_memory()
            _ = model(show_batch, store_memory=True)

            # Process hide phase (use memory)
            out = model(hide_batch, store_memory=False)

            # Loss on x prediction
            loss = F.cross_entropy(out["x_logits"][:, :8], target_x_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred_x = out["x_logits"][:, :8].argmax(-1)
            correct += (pred_x == target_x_batch).sum().item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/len(data):.1%}")

    # Evaluate
    print("\nEvaluating...")
    successes = []
    for _ in range(200):
        env = StateTrackingEnv(grid_size=8, num_values=4, max_steps=15, variant="memory")
        obs = env.reset()

        # Show phase
        model.reset_memory()
        with torch.no_grad():
            _ = model(obs.unsqueeze(0).to(device), store_memory=True)

        result = env.step(Action.INTERACT)
        obs = result.observation

        # Hide phase
        with torch.no_grad():
            out = model(obs.unsqueeze(0).to(device), store_memory=False)
            pred_x = out["x_logits"][0, :8].argmax().item()

        result = env.step(Action.CLICK, pred_x, 0)
        successes.append(1.0 if result.success else 0.0)

    success_rate = sum(successes) / len(successes)
    print(f"\nMemory eval: {success_rate:.1%}")
    return success_rate


if __name__ == "__main__":
    print("=" * 60)
    print("MEMORY TASK - EXPLICIT ARCHITECTURE")
    print("=" * 60)
    success = train_memory()
    print(f"\nResult: {success:.1%} {'PASS' if success > 0.8 else 'FAIL'}")
