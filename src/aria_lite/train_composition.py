"""
Training for composition primitives.

Composition variants:
- nav_then_click: Navigate to zone, then click target
- pattern_then_act: Find pattern, then click it
- conditional: Action depends on indicator value
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aria_lite.primitives import CompositionEnv
from src.aria_lite.primitives.base import Action


class CompositionEncoder(nn.Module):
    """Encoder for composition tasks."""

    def __init__(self, hidden_dim: int = 128, num_values: int = 16, max_grid: int = 20):
        super().__init__()
        self.embed = nn.Embedding(num_values, 32)
        self.pos_y = nn.Embedding(max_grid, 8)
        self.pos_x = nn.Embedding(max_grid, 8)

        self.conv = nn.Sequential(
            nn.Conv2d(48, 64, 3, padding=1),  # 32 + 8 + 8 = 48
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, obs: torch.Tensor):
        B, H, W = obs.shape
        device = obs.device

        # Embed cells
        cell = self.embed(obs.long().clamp(0, 15))  # [B, H, W, 32]

        # Position embeddings
        y_idx = torch.arange(H, device=device)
        x_idx = torch.arange(W, device=device)
        pos_y = self.pos_y(y_idx).unsqueeze(1).expand(H, W, -1)
        pos_x = self.pos_x(x_idx).unsqueeze(0).expand(H, W, -1)
        pos = torch.cat([pos_y, pos_x], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

        # Combine
        features = torch.cat([cell, pos], dim=-1)  # [B, H, W, 48]
        features = features.permute(0, 3, 1, 2)  # [B, 48, H, W]

        features = self.conv(features)
        features = features.reshape(B, -1)
        return self.fc(features)


class CompositionPolicy(nn.Module):
    """Policy for composition tasks."""

    def __init__(self, hidden_dim: int = 128, max_grid: int = 20):
        super().__init__()
        self.encoder = CompositionEncoder(hidden_dim, max_grid=max_grid)

        self.action_head = nn.Linear(hidden_dim, 9)
        self.x_head = nn.Linear(hidden_dim, max_grid)
        self.y_head = nn.Linear(hidden_dim, max_grid)

    def forward(self, obs: torch.Tensor, grid_size: int = 10):
        features = self.encoder(obs)
        action_logits = self.action_head(features)
        x_logits = self.x_head(features)[:, :grid_size]
        y_logits = self.y_head(features)[:, :grid_size]
        return action_logits, x_logits, y_logits


# ============================================================================
# Expert Solvers
# ============================================================================


def nav_then_click_expert(env: CompositionEnv):
    """Expert for nav_then_click."""
    if env.phase == "navigate" and env.agent_pos:
        # Navigate towards goal zone (right side)
        ay, ax = env.agent_pos
        target_x = env.grid_size - 4  # Roughly where zone starts

        if ax < target_x:
            return Action.RIGHT, 0, 0
        elif ay > env.grid_size // 2:
            return Action.UP, 0, 0
        elif ay < env.grid_size // 2:
            return Action.DOWN, 0, 0
        else:
            return Action.RIGHT, 0, 0

    elif env.phase == "click" and env.click_target:
        y, x = env.click_target
        return Action.CLICK, x, y

    return Action.NOOP, 0, 0


def pattern_then_act_expert(env: CompositionEnv):
    """Expert for pattern_then_act."""
    if env.click_target:
        y, x = env.click_target
        return Action.CLICK, x, y
    return Action.NOOP, 0, 0


def conditional_expert(env: CompositionEnv):
    """Expert for conditional."""
    if env.click_target:
        y, x = env.click_target
        return Action.CLICK, x, y
    return Action.NOOP, 0, 0


def get_expert_action(env: CompositionEnv):
    """Get expert action for any composition variant."""
    if env.variant == "nav_then_click":
        return nav_then_click_expert(env)
    elif env.variant == "pattern_then_act":
        return pattern_then_act_expert(env)
    else:  # conditional
        return conditional_expert(env)


# ============================================================================
# Training
# ============================================================================


def collect_composition_episodes(variant: str, num_episodes: int = 500):
    """Collect expert episodes for a composition variant."""
    episodes = []

    for _ in range(num_episodes * 2):  # Over-collect since some fail
        if variant == "nav_then_click":
            env = CompositionEnv(grid_size=10, num_obstacles=0, variant=variant, max_steps=30)
        elif variant == "pattern_then_act":
            env = CompositionEnv(grid_size=10, variant=variant, max_steps=10)
        else:  # conditional
            env = CompositionEnv(grid_size=10, variant=variant, max_steps=10)

        obs = env.reset()
        episode = {"observations": [obs.clone()], "actions": [], "coords": []}

        for _ in range(env.max_steps):
            action, x, y = get_expert_action(env)
            result = env.step(action, x, y)

            episode["actions"].append(action)
            episode["coords"].append((x, y))
            episode["observations"].append(result.observation.clone())

            if result.done:
                if result.success:
                    episodes.append(episode)
                break

        if len(episodes) >= num_episodes:
            break

    return episodes[:num_episodes]


def train_variant(variant: str, num_episodes: int = 500, num_epochs: int = 50):
    """Train a composition variant."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Collecting {variant} episodes...")
    episodes = collect_composition_episodes(variant, num_episodes)
    print(f"Collected {len(episodes)} successful episodes")

    if len(episodes) < 10:
        print("Not enough episodes")
        return 0.0

    # Flatten data
    all_obs = []
    all_actions = []
    all_x = []
    all_y = []

    for ep in episodes:
        all_obs.extend(ep["observations"][:-1])
        all_actions.extend(ep["actions"])
        for x, y in ep["coords"]:
            all_x.append(x)
            all_y.append(y)

    obs_tensor = torch.stack(all_obs)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)
    x_tensor = torch.tensor(all_x, dtype=torch.long)
    y_tensor = torch.tensor(all_y, dtype=torch.long)

    print(f"Dataset: {len(obs_tensor)} samples")

    model = CompositionPolicy(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    grid_size = obs_tensor.shape[1]

    for epoch in range(num_epochs):
        perm = torch.randperm(len(obs_tensor))
        obs_tensor = obs_tensor[perm]
        actions_tensor = actions_tensor[perm]
        x_tensor = x_tensor[perm]
        y_tensor = y_tensor[perm]

        total_loss = 0
        correct = 0

        for i in range(0, len(obs_tensor), batch_size):
            obs_batch = obs_tensor[i:i+batch_size].to(device)
            actions_batch = actions_tensor[i:i+batch_size].to(device)
            x_batch = x_tensor[i:i+batch_size].to(device).clamp(0, grid_size-1)
            y_batch = y_tensor[i:i+batch_size].to(device).clamp(0, grid_size-1)

            action_logits, x_logits, y_logits = model(obs_batch, grid_size)

            loss = (
                F.cross_entropy(action_logits, actions_batch)
                + F.cross_entropy(x_logits, x_batch)
                + F.cross_entropy(y_logits, y_batch)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (action_logits.argmax(-1) == actions_batch).sum().item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/len(obs_tensor):.1%}")

    # Evaluate
    successes = []
    for _ in range(100):
        if variant == "nav_then_click":
            env = CompositionEnv(grid_size=10, num_obstacles=0, variant=variant, max_steps=30)
        elif variant == "pattern_then_act":
            env = CompositionEnv(grid_size=10, variant=variant, max_steps=10)
        else:
            env = CompositionEnv(grid_size=10, variant=variant, max_steps=10)

        obs = env.reset()

        for _ in range(env.max_steps):
            with torch.no_grad():
                action_logits, x_logits, y_logits = model(obs.unsqueeze(0).to(device), env.grid_size)
                action = action_logits.argmax(-1).item()
                x = x_logits.argmax(-1).item()
                y = y_logits.argmax(-1).item()

            result = env.step(action, x, y)
            obs = result.observation

            if result.done:
                successes.append(1.0 if result.success else 0.0)
                break

    success_rate = sum(successes) / len(successes)
    print(f"\n{variant} eval: {success_rate:.1%}")
    return success_rate


if __name__ == "__main__":
    print("=" * 60)
    print("COMPOSITION PRIMITIVES")
    print("=" * 60)

    results = {}

    for variant in ["conditional", "pattern_then_act", "nav_then_click"]:
        print(f"\n--- {variant} ---")
        results[variant] = train_variant(variant)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for variant, success in results.items():
        status = "PASS" if success > 0.5 else "FAIL"
        print(f"{variant}: {success:.1%} {status}")
