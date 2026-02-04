"""
Training script for pattern matching with cross-attention.

Tests whether cross-attention can learn template matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aria_lite.primitives import PatternEnv
from src.aria_lite.primitives.base import Action
from src.aria_lite.primitives.pattern_encoder import (
    PatternMatchingPolicy,
    DifferenceMatchingEncoder,
)


def collect_match_data(num_samples: int = 1000):
    """Collect pattern matching data (match variant)."""
    data = []

    for _ in range(num_samples):
        env = PatternEnv(
            grid_size=10,
            pattern_size=3,
            num_colors=4,
            max_steps=1,
            variant="match",
        )
        obs = env.reset()

        # obs shape: [13, 10] = grid (10x10) + template row (3x10)
        # Template is at bottom: obs[10:13, 0:3]
        grid = obs[:10, :]  # [10, 10]
        template = obs[10:13, :3]  # [3, 3]

        y, x = env.target_pos  # Where template is in grid

        data.append({
            "grid": grid.clone(),
            "template": template.clone(),
            "x": x,
            "y": y,
        })

    return data


def collect_difference_data(num_samples: int = 1000):
    """Collect difference data (difference variant)."""
    data = []

    for _ in range(num_samples):
        env = PatternEnv(
            grid_size=8,
            pattern_size=3,
            num_colors=4,
            max_steps=1,
            variant="difference",
        )
        obs = env.reset()

        # obs shape: [16, 8] = two 8x8 grids stacked
        y, x = env.target_pos

        data.append({
            "obs": obs.clone(),
            "x": x,
            "y": y,
        })

    return data


def train_match_pattern():
    """Train cross-attention model on match task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    print("Collecting match data...")
    data = collect_match_data(1000)
    print(f"Collected {len(data)} samples")

    # Stack data
    grids = torch.stack([d["grid"] for d in data])
    templates = torch.stack([d["template"] for d in data])
    xs = torch.tensor([d["x"] for d in data], dtype=torch.long)
    ys = torch.tensor([d["y"] for d in data], dtype=torch.long)

    print(f"Grid shape: {grids.shape}, Template shape: {templates.shape}")
    print(f"X range: {xs.min()}-{xs.max()}, Y range: {ys.min()}-{ys.max()}")

    # Model
    model = PatternMatchingPolicy(num_colors=16, d_model=64, max_grid=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    num_epochs = 100

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(len(grids))
        grids = grids[perm]
        templates = templates[perm]
        xs = xs[perm]
        ys = ys[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(grids), batch_size):
            grid_batch = grids[i:i+batch_size].to(device)
            template_batch = templates[i:i+batch_size].to(device)
            x_batch = xs[i:i+batch_size].to(device)
            y_batch = ys[i:i+batch_size].to(device)

            out = model(grid_batch, template_batch, grid_size=8)  # Max target is 7

            loss = (
                F.cross_entropy(out["x_logits"], x_batch)
                + F.cross_entropy(out["y_logits"], y_batch)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            pred_x = out["x_logits"].argmax(dim=-1)
            pred_y = out["y_logits"].argmax(dim=-1)
            correct += ((pred_x == x_batch) & (pred_y == y_batch)).sum().item()
            total += len(x_batch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/total:.1%}")

    # Evaluate on fresh data
    print("\nEvaluating...")
    eval_data = collect_match_data(200)

    successes = []
    for d in eval_data:
        grid = d["grid"].unsqueeze(0).to(device)
        template = d["template"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(grid, template, grid_size=8)
            pred_x = out["x_logits"].argmax(dim=-1).item()
            pred_y = out["y_logits"].argmax(dim=-1).item()

        # Check against environment
        env = PatternEnv(grid_size=10, pattern_size=3, num_colors=4, variant="match")
        env.reset()
        # Manually set state to match our data
        env.target_pos = (d["y"], d["x"])

        result = env.step(Action.CLICK, pred_x, pred_y)
        successes.append(1.0 if result.success else 0.0)

    eval_success = sum(successes) / len(successes)
    print(f"Match pattern eval success: {eval_success:.1%}")
    return eval_success


def train_difference_pattern():
    """Train model on difference task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    print("Collecting difference data...")
    data = collect_difference_data(1000)
    print(f"Collected {len(data)} samples")

    # Stack data
    obs = torch.stack([d["obs"] for d in data])
    xs = torch.tensor([d["x"] for d in data], dtype=torch.long)
    ys = torch.tensor([d["y"] for d in data], dtype=torch.long)

    print(f"Obs shape: {obs.shape}")
    print(f"X range: {xs.min()}-{xs.max()}, Y range: {ys.min()}-{ys.max()}")

    # Model
    model = DifferenceMatchingEncoder(num_colors=16, d_model=64, max_grid=20).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    num_epochs = 100

    for epoch in range(num_epochs):
        # Shuffle
        perm = torch.randperm(len(obs))
        obs = obs[perm]
        xs = xs[perm]
        ys = ys[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(obs), batch_size):
            obs_batch = obs[i:i+batch_size].to(device)
            x_batch = xs[i:i+batch_size].to(device)
            y_batch = ys[i:i+batch_size].to(device)

            out = model(obs_batch, grid_size=8)

            loss = (
                F.cross_entropy(out["x_logits"], x_batch)
                + F.cross_entropy(out["y_logits"], y_batch)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred_x = out["x_logits"].argmax(dim=-1)
            pred_y = out["y_logits"].argmax(dim=-1)
            correct += ((pred_x == x_batch) & (pred_y == y_batch)).sum().item()
            total += len(x_batch)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/total:.1%}")

    # Evaluate
    print("\nEvaluating...")
    successes = []
    for _ in range(200):
        env = PatternEnv(grid_size=8, pattern_size=3, num_colors=4, variant="difference")
        obs = env.reset()

        with torch.no_grad():
            out = model(obs.unsqueeze(0).to(device), grid_size=8)
            pred_x = out["x_logits"].argmax(dim=-1).item()
            pred_y = out["y_logits"].argmax(dim=-1).item()

        result = env.step(Action.CLICK, pred_x, pred_y)
        successes.append(1.0 if result.success else 0.0)

    eval_success = sum(successes) / len(successes)
    print(f"Difference pattern eval success: {eval_success:.1%}")
    return eval_success


if __name__ == "__main__":
    print("=" * 60)
    print("PATTERN MATCHING WITH CROSS-ATTENTION")
    print("=" * 60)

    print("\n--- Match Task ---")
    match_success = train_match_pattern()

    print("\n--- Difference Task ---")
    diff_success = train_difference_pattern()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Match: {match_success:.1%} {'PASS' if match_success > 0.5 else 'FAIL'}")
    print(f"Difference: {diff_success:.1%} {'PASS' if diff_success > 0.5 else 'FAIL'}")
