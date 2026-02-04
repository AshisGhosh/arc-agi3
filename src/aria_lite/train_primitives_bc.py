"""
BC training for primitives using expert demonstrations.

Expert solvers provide optimal actions, BC learns to imitate.
"""

import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aria_lite.primitives import (
    NavigationEnv,
    ClickEnv,
    PrimitiveFamily,
)
from src.aria_lite.primitives.base import Action, PrimitiveEnv


# ============================================================================
# Expert Solvers
# ============================================================================


def nav_expert_action(env: NavigationEnv) -> int:
    """BFS-based expert for navigation."""
    if env.agent_pos is None or not env.goals:
        return Action.NOOP

    # Find nearest unvisited goal
    targets = [g for g in env.goals if g not in env.visited_goals]
    if not targets:
        return Action.NOOP

    target = min(targets, key=lambda g: abs(g[0] - env.agent_pos[0]) + abs(g[1] - env.agent_pos[1]))

    ay, ax = env.agent_pos
    gy, gx = target

    # Simple greedy: move towards goal
    if ay < gy:
        return Action.DOWN
    elif ay > gy:
        return Action.UP
    elif ax < gx:
        return Action.RIGHT
    elif ax > gx:
        return Action.LEFT

    return Action.NOOP


def click_expert_action(env: ClickEnv) -> tuple[int, int, int]:
    """Expert for click primitive."""
    if env.variant == "sequence":
        # Click targets in order
        if env.current_target_idx < len(env.targets):
            y, x = env.targets[env.current_target_idx]
            return Action.CLICK, x, y
    else:
        # Click any unclicked target
        for target in env.targets:
            if target not in env.clicked:
                y, x = target
                return Action.CLICK, x, y

    return Action.NOOP, 0, 0


# ============================================================================
# Simple Position-Aware Encoder
# ============================================================================


class PositionAwareEncoder(nn.Module):
    """Encoder that's position-aware for sparse grids."""

    def __init__(self, hidden_dim: int = 128, num_values: int = 16, max_grid: int = 20):
        super().__init__()

        # Separate embeddings for different cell types
        self.cell_embed = nn.Embedding(num_values, 16)
        self.pos_y_embed = nn.Embedding(max_grid, 8)
        self.pos_x_embed = nn.Embedding(max_grid, 8)

        # Process: cell + pos = 32 channels
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.shape
        device = x.device
        x = x.long().clamp(0, 15)

        # Cell embedding
        cell = self.cell_embed(x)  # [B, H, W, 16]

        # Position embedding (shared across batch)
        y_idx = torch.arange(H, device=device)
        x_idx = torch.arange(W, device=device)
        pos_y = self.pos_y_embed(y_idx).unsqueeze(1).expand(H, W, -1)  # [H, W, 8]
        pos_x = self.pos_x_embed(x_idx).unsqueeze(0).expand(H, W, -1)  # [H, W, 8]

        pos = torch.cat([pos_y, pos_x], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 16]

        # Combine
        features = torch.cat([cell, pos], dim=-1)  # [B, H, W, 32]
        features = features.permute(0, 3, 1, 2)  # [B, 32, H, W]

        # CNN
        features = self.conv(features)
        features = features.reshape(B, -1)

        return self.fc(features)


# ============================================================================
# Policy Networks
# ============================================================================


class NavPolicy(nn.Module):
    """Policy for navigation (5 discrete actions)."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.encoder = PositionAwareEncoder(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 5)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(obs)
        return self.action_head(features)


class ClickPolicy(nn.Module):
    """Policy for click (action + coordinates)."""

    def __init__(self, hidden_dim: int = 128, max_grid: int = 20):
        super().__init__()
        self.encoder = PositionAwareEncoder(hidden_dim)
        self.action_head = nn.Linear(hidden_dim, 9)  # All action types
        self.x_head = nn.Linear(hidden_dim, max_grid)
        self.y_head = nn.Linear(hidden_dim, max_grid)

    def forward(self, obs: torch.Tensor, grid_size: int = 10):
        features = self.encoder(obs)
        action_logits = self.action_head(features)
        x_logits = self.x_head(features)[:, :grid_size]
        y_logits = self.y_head(features)[:, :grid_size]
        return action_logits, x_logits, y_logits


# ============================================================================
# Data Collection
# ============================================================================


def collect_nav_episodes(num_episodes: int = 500, difficulty: int = 1):
    """Collect expert navigation episodes."""
    from src.aria_lite.primitives import NavigationPrimitiveGenerator

    gen = NavigationPrimitiveGenerator()
    episodes = []

    for _ in range(num_episodes):
        env = gen.generate(difficulty=difficulty)
        obs = env.reset()

        episode_obs = [obs.clone()]
        episode_actions = []

        for step in range(env.max_steps):
            action = nav_expert_action(env)
            result = env.step(action)

            episode_actions.append(action)
            episode_obs.append(result.observation.clone())

            if result.done:
                break

        if result.success:
            episodes.append({
                "observations": episode_obs[:-1],  # Don't include terminal
                "actions": episode_actions,
            })

    return episodes


def collect_click_episodes(num_episodes: int = 500, difficulty: int = 1):
    """Collect expert click episodes."""
    from src.aria_lite.primitives import ClickPrimitiveGenerator

    gen = ClickPrimitiveGenerator()
    episodes = []

    for _ in range(num_episodes):
        env = gen.generate(difficulty=difficulty)
        obs = env.reset()

        episode_obs = [obs.clone()]
        episode_actions = []
        episode_coords = []

        for step in range(env.max_steps):
            action, x, y = click_expert_action(env)
            result = env.step(action, x, y)

            episode_actions.append(action)
            episode_coords.append((x, y))
            episode_obs.append(result.observation.clone())

            if result.done:
                break

        if result.success:
            episodes.append({
                "observations": episode_obs[:-1],
                "actions": episode_actions,
                "coordinates": episode_coords,
                "grid_size": env.grid_size,
            })

    return episodes


# ============================================================================
# Training
# ============================================================================


def train_nav_bc(num_episodes: int = 500, num_epochs: int = 50):
    """Train navigation via BC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Collecting {num_episodes} expert navigation episodes...")
    episodes = collect_nav_episodes(num_episodes)
    print(f"Collected {len(episodes)} successful episodes")

    # Flatten to dataset
    all_obs = []
    all_actions = []
    for ep in episodes:
        all_obs.extend(ep["observations"])
        all_actions.extend(ep["actions"])

    obs_tensor = torch.stack(all_obs)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)

    print(f"Dataset size: {len(obs_tensor)} samples")

    # Model
    model = NavPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(len(obs_tensor))
        obs_tensor = obs_tensor[perm]
        actions_tensor = actions_tensor[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(obs_tensor), batch_size):
            obs_batch = obs_tensor[i:i+batch_size].to(device)
            actions_batch = actions_tensor[i:i+batch_size].to(device)

            logits = model(obs_batch)
            loss = F.cross_entropy(logits, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=-1) == actions_batch).sum().item()
            total += len(actions_batch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/total:.1%}")

    # Evaluate
    from src.aria_lite.primitives import NavigationPrimitiveGenerator
    gen = NavigationPrimitiveGenerator()

    successes = []
    for _ in range(100):
        env = gen.generate(difficulty=1)
        obs = env.reset()

        for _ in range(env.max_steps):
            with torch.no_grad():
                logits = model(obs.unsqueeze(0).to(device))
                action = logits.argmax(dim=-1).item()

            result = env.step(action)
            obs = result.observation

            if result.done:
                successes.append(1.0 if result.success else 0.0)
                break

    print(f"\nNavigation BC eval success: {sum(successes)/len(successes):.1%}")
    return sum(successes) / len(successes)


def train_click_bc(num_episodes: int = 500, num_epochs: int = 50):
    """Train click via BC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Collecting {num_episodes} expert click episodes...")
    episodes = collect_click_episodes(num_episodes)
    print(f"Collected {len(episodes)} successful episodes")

    # Flatten to dataset
    all_obs = []
    all_actions = []
    all_x = []
    all_y = []
    grid_sizes = []

    for ep in episodes:
        all_obs.extend(ep["observations"])
        all_actions.extend(ep["actions"])
        for x, y in ep["coordinates"]:
            all_x.append(x)
            all_y.append(y)
        grid_sizes.extend([ep["grid_size"]] * len(ep["observations"]))

    obs_tensor = torch.stack(all_obs)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)
    x_tensor = torch.tensor(all_x, dtype=torch.long)
    y_tensor = torch.tensor(all_y, dtype=torch.long)

    print(f"Dataset size: {len(obs_tensor)} samples")

    # Model
    model = ClickPolicy().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(len(obs_tensor))
        obs_tensor = obs_tensor[perm]
        actions_tensor = actions_tensor[perm]
        x_tensor = x_tensor[perm]
        y_tensor = y_tensor[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(obs_tensor), batch_size):
            obs_batch = obs_tensor[i:i+batch_size].to(device)
            actions_batch = actions_tensor[i:i+batch_size].to(device)
            x_batch = x_tensor[i:i+batch_size].to(device)
            y_batch = y_tensor[i:i+batch_size].to(device)

            action_logits, x_logits, y_logits = model(obs_batch)

            loss = (
                F.cross_entropy(action_logits, actions_batch)
                + F.cross_entropy(x_logits, x_batch)
                + F.cross_entropy(y_logits, y_batch)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy: all three must be correct
            pred_action = action_logits.argmax(dim=-1)
            pred_x = x_logits.argmax(dim=-1)
            pred_y = y_logits.argmax(dim=-1)

            correct_mask = (
                (pred_action == actions_batch)
                & (pred_x == x_batch)
                & (pred_y == y_batch)
            )
            correct += correct_mask.sum().item()
            total += len(actions_batch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/total:.1%}")

    # Evaluate
    from src.aria_lite.primitives import ClickPrimitiveGenerator
    gen = ClickPrimitiveGenerator()

    successes = []
    for _ in range(100):
        env = gen.generate(difficulty=1)
        obs = env.reset()

        for _ in range(env.max_steps):
            with torch.no_grad():
                action_logits, x_logits, y_logits = model(obs.unsqueeze(0).to(device), env.grid_size)
                action = action_logits.argmax(dim=-1).item()
                x = x_logits.argmax(dim=-1).item()
                y = y_logits.argmax(dim=-1).item()

            result = env.step(action, x, y)
            obs = result.observation

            if result.done:
                successes.append(1.0 if result.success else 0.0)
                break

    print(f"\nClick BC eval success: {sum(successes)/len(successes):.1%}")
    return sum(successes) / len(successes)


# ============================================================================
# Pattern Experts and Training
# ============================================================================


def pattern_expert_action(env) -> tuple[int, int, int]:
    """Expert for pattern primitive."""
    if env.variant == "match" and env.target_pos:
        y, x = env.target_pos
        return Action.CLICK, x, y
    elif env.variant == "difference" and env.target_pos:
        y, x = env.target_pos
        return Action.CLICK, x, y
    elif env.variant == "complete" and hasattr(env, "expected_color"):
        return Action.CLICK, env.expected_color, 0
    elif env.variant == "cycle" and env.target_pos:
        y, x = env.target_pos
        return Action.CLICK, x, y

    return Action.NOOP, 0, 0


def collect_pattern_episodes(num_episodes: int = 500, difficulty: int = 1):
    """Collect expert pattern episodes - only match and difference variants."""
    import random
    from src.aria_lite.primitives import PatternEnv

    episodes = []
    attempts = 0

    while len(episodes) < num_episodes and attempts < num_episodes * 5:
        attempts += 1
        # Only use variants that work with single click
        variant = random.choice(["match", "difference"])

        env = PatternEnv(
            grid_size=10 if variant == "match" else 8,
            pattern_size=3,
            num_colors=4,
            max_steps=10,
            variant=variant,
        )
        obs = env.reset()

        # Expert knows the target position
        y, x = env.target_pos
        action = Action.CLICK

        result = env.step(action, x, y)

        if result.success:
            episodes.append({
                "observations": [obs.clone()],
                "actions": [action],
                "coordinates": [(x, y)],
                "obs_shape": obs.shape,
            })

    return episodes


def train_pattern_bc(num_episodes: int = 500, num_epochs: int = 50):
    """Train pattern via BC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Collecting {num_episodes} expert pattern episodes...")
    episodes = collect_pattern_episodes(num_episodes, difficulty=1)
    print(f"Collected {len(episodes)} successful episodes")

    if len(episodes) < 10:
        print("Not enough successful episodes")
        return 0.0

    # Flatten - handle variable obs shapes by padding
    max_h = max(ep["obs_shape"][0] for ep in episodes)
    max_w = max(ep["obs_shape"][1] for ep in episodes)

    all_obs = []
    all_actions = []
    all_x = []
    all_y = []

    for ep in episodes:
        for obs in ep["observations"]:
            # Pad to max size
            padded = torch.zeros(max_h, max_w, dtype=obs.dtype)
            padded[:obs.shape[0], :obs.shape[1]] = obs
            all_obs.append(padded)
        all_actions.extend(ep["actions"])
        for x, y in ep["coordinates"]:
            all_x.append(x)
            all_y.append(y)

    obs_tensor = torch.stack(all_obs)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)
    x_tensor = torch.tensor(all_x, dtype=torch.long)
    y_tensor = torch.tensor(all_y, dtype=torch.long)

    print(f"Dataset size: {len(obs_tensor)} samples")

    # Model (reuse click policy architecture)
    model = ClickPolicy(max_grid=max(max_h, max_w)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64

    for epoch in range(num_epochs):
        perm = torch.randperm(len(obs_tensor))
        obs_tensor = obs_tensor[perm]
        actions_tensor = actions_tensor[perm]
        x_tensor = x_tensor[perm]
        y_tensor = y_tensor[perm]

        total_loss = 0

        for i in range(0, len(obs_tensor), batch_size):
            obs_batch = obs_tensor[i:i+batch_size].to(device)
            actions_batch = actions_tensor[i:i+batch_size].to(device)
            x_batch = x_tensor[i:i+batch_size].to(device).clamp(0, max_w-1)
            y_batch = y_tensor[i:i+batch_size].to(device).clamp(0, max_h-1)

            action_logits, x_logits, y_logits = model(obs_batch, max_w)

            loss = (
                F.cross_entropy(action_logits, actions_batch)
                + F.cross_entropy(x_logits, x_batch)
                + F.cross_entropy(y_logits, y_batch)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}")

    # Evaluate using same variants
    import random
    from src.aria_lite.primitives import PatternEnv

    successes = []
    for _ in range(100):
        variant = random.choice(["match", "difference"])
        env = PatternEnv(
            grid_size=10 if variant == "match" else 8,
            pattern_size=3,
            num_colors=4,
            max_steps=10,
            variant=variant,
        )
        obs = env.reset()

        # Pad observation
        padded = torch.zeros(max_h, max_w, dtype=obs.dtype)
        padded[:obs.shape[0], :obs.shape[1]] = obs

        with torch.no_grad():
            action_logits, x_logits, y_logits = model(padded.unsqueeze(0).to(device), max_w)
            action = action_logits.argmax(dim=-1).item()
            x = x_logits.argmax(dim=-1).item()
            y = y_logits.argmax(dim=-1).item()

        result = env.step(action, x, y)
        successes.append(1.0 if result.success else 0.0)

    print(f"\nPattern BC eval success: {sum(successes)/len(successes):.1%}")
    return sum(successes) / len(successes)


if __name__ == "__main__":
    print("=" * 60)
    print("NAVIGATION BC")
    print("=" * 60)
    nav_success = train_nav_bc()

    print("\n" + "=" * 60)
    print("CLICK BC")
    print("=" * 60)
    click_success = train_click_bc()

    print("\n" + "=" * 60)
    print("PATTERN BC")
    print("=" * 60)
    pattern_success = train_pattern_bc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Navigation BC: {nav_success:.1%} {'PASS' if nav_success > 0.8 else 'FAIL'}")
    print(f"Click BC: {click_success:.1%} {'PASS' if click_success > 0.8 else 'FAIL'}")
    print(f"Pattern BC: {pattern_success:.1%} {'PASS' if pattern_success > 0.5 else 'FAIL'}")
