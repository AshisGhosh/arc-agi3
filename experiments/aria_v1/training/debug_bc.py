#!/usr/bin/env python3
"""Debug BC training issues."""

import torch
import torch.nn.functional as F
from collections import Counter

from .training.expert_data import collect_expert_dataset
from .training.synthetic_env import SyntheticEnv, Action
from .config import ARIALiteConfig
from .agent import create_agent


def analyze_expert_data():
    """Analyze the expert data distribution."""
    print("Collecting expert data...")
    dataset = collect_expert_dataset(
        mechanic="navigation",
        num_trajectories=1000,
        grid_size=10,
    )

    print(f"\nTrajectories: {len(dataset.trajectories)}")
    print(f"Transitions: {dataset.num_transitions}")
    print(f"Success rate: {dataset.success_rate:.1%}")

    # Action distribution
    action_counts = Counter()
    for traj in dataset.trajectories:
        for action in traj.actions:
            action_counts[action] += 1

    print("\nAction distribution:")
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "PICKUP", "DROP", "NOOP"]
    total = sum(action_counts.values())
    for i in range(8):
        count = action_counts.get(i, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {action_names[i]}: {count} ({pct:.1f}%)")

    # Trajectory lengths
    lengths = [len(t.actions) for t in dataset.trajectories]
    print(f"\nTrajectory lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")

    return dataset


def test_model_forward():
    """Test model forward pass."""
    print("\nTesting model forward pass...")

    config = ARIALiteConfig()
    agent = create_agent(config)
    agent.eval()

    # Create test input
    obs = torch.zeros(1, 64, 64, dtype=torch.long)
    obs[0, 5, 5] = 1  # Some non-zero cell

    print(f"Input shape: {obs.shape}")
    print(f"Input range: [{obs.min()}, {obs.max()}]")

    with torch.no_grad():
        states = agent.encoder(obs)
        print(f"Encoded state shape: {states.shape}")
        print(f"Encoded state range: [{states.min():.3f}, {states.max():.3f}]")

        fp_out = agent.fast_policy(states)
        print(f"Action logits shape: {fp_out.action_logits.shape}")
        print(f"Action logits: {fp_out.action_logits[0].numpy()}")
        probs = F.softmax(fp_out.action_logits[0], dim=-1)
        print(f"Action probs: {probs.numpy()}")


def test_training_step():
    """Test a single training step."""
    print("\nTesting training step (WITH masks)...")

    dataset = collect_expert_dataset(
        mechanic="navigation",
        num_trajectories=100,
        grid_size=10,
    )

    observations, masks, actions = dataset.get_tensors(max_obs_size=64)
    print(f"Observations shape: {observations.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Actions shape: {actions.shape}")
    print(f"Action range: [{actions.min()}, {actions.max()}]")
    print(f"Unique actions: {actions.unique().tolist()}")

    # Check observation encoding
    print(f"Observation range: [{observations.min()}, {observations.max()}]")
    print(f"Non-zero cells per obs: {(observations > 0).sum(dim=(1,2)).float().mean():.1f}")
    print(f"Valid cells per obs: {(~masks).sum(dim=(1,2)).float().mean():.1f}")

    config = ARIALiteConfig()
    agent = create_agent(config)

    # Single forward pass WITH masks
    batch_obs = observations[:32]
    batch_masks = masks[:32]
    batch_actions = actions[:32]

    states = agent.encoder(batch_obs, mask=batch_masks)
    fp_out = agent.fast_policy(states)

    loss = F.cross_entropy(fp_out.action_logits, batch_actions)
    print(f"Initial loss: {loss.item():.4f}")
    print(f"Random baseline: {-torch.log(torch.tensor(1/8)):.4f}")  # log(8) = 2.08

    # Check predictions
    preds = fp_out.action_logits.argmax(dim=-1)
    acc = (preds == batch_actions).float().mean()
    print(f"Initial accuracy: {acc.item():.1%}")

    # Train for a few steps
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    for i in range(100):
        optimizer.zero_grad()
        states = agent.encoder(batch_obs, mask=batch_masks)
        fp_out = agent.fast_policy(states)
        loss = F.cross_entropy(fp_out.action_logits, batch_actions)
        loss.backward()
        optimizer.step()

    states = agent.encoder(batch_obs, mask=batch_masks)
    fp_out = agent.fast_policy(states)
    loss = F.cross_entropy(fp_out.action_logits, batch_actions)
    preds = fp_out.action_logits.argmax(dim=-1)
    acc = (preds == batch_actions).float().mean()
    print(f"After 100 steps - loss: {loss.item():.4f}, acc: {acc.item():.1%}")


def check_encoder_output():
    """Check if encoder produces varied outputs."""
    print("\nChecking encoder outputs (WITHOUT masks)...")

    dataset = collect_expert_dataset(
        mechanic="navigation",
        num_trajectories=100,
        grid_size=10,
    )

    observations, masks, _ = dataset.get_tensors(max_obs_size=64)

    config = ARIALiteConfig()
    agent = create_agent(config)
    agent.eval()

    # Test WITHOUT masks
    with torch.no_grad():
        states_no_mask = agent.encoder(observations[:10])

    print(f"State shape: {states_no_mask.shape}")
    print(f"State mean: {states_no_mask.mean():.4f}")
    print(f"State std: {states_no_mask.std():.4f}")

    diffs = []
    for i in range(9):
        diff = (states_no_mask[i] - states_no_mask[i+1]).abs().mean().item()
        diffs.append(diff)
    print(f"Pairwise differences (no mask): {sum(diffs)/len(diffs):.4f}")

    if sum(diffs)/len(diffs) < 0.01:
        print("WARNING: Encoder outputs are nearly identical for different inputs!")

    # Test WITH masks
    print("\nChecking encoder outputs (WITH masks)...")
    with torch.no_grad():
        states_with_mask = agent.encoder(observations[:10], mask=masks[:10])

    print(f"State mean: {states_with_mask.mean():.4f}")
    print(f"State std: {states_with_mask.std():.4f}")

    diffs = []
    for i in range(9):
        diff = (states_with_mask[i] - states_with_mask[i+1]).abs().mean().item()
        diffs.append(diff)
    print(f"Pairwise differences (with mask): {sum(diffs)/len(diffs):.4f}")

    if sum(diffs)/len(diffs) > 0.1:
        print("SUCCESS: Masked encoder produces varied outputs!")


if __name__ == "__main__":
    analyze_expert_data()
    test_model_forward()
    check_encoder_output()
    test_training_step()
