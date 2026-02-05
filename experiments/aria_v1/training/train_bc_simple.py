#!/usr/bin/env python3
"""BC training with simple encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .encoder_simple import SimpleGridEncoder
from .fast_policy import FastPolicy
from .config import FastPolicyConfig
from .training.expert_data import collect_expert_dataset


class SimpleAgent(nn.Module):
    """Minimal agent with simple encoder + fast policy."""

    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.encoder = SimpleGridEncoder(output_dim=output_dim)
        self.policy = FastPolicy(FastPolicyConfig(state_dim=output_dim))

    def forward(self, grid, mask=None):
        state = self.encoder(grid, mask)
        return self.policy(state)


def train_mechanic(mechanic: str, device: str):
    """Train and evaluate on a single mechanic."""
    print(f"\n{'='*50}")
    print(f"Training: {mechanic}")
    print(f"{'='*50}", flush=True)

    # Collect data
    print("Collecting expert data...", flush=True)
    dataset = collect_expert_dataset(
        mechanic=mechanic,
        num_trajectories=2000,
        grid_size=10,
        max_steps=50,
    )

    observations, masks, actions = dataset.get_tensors(max_obs_size=10)
    print(f"Data: {len(actions)} transitions", flush=True)

    # Create agent
    agent = SimpleAgent(output_dim=256).to(device)

    # Training
    train_data = TensorDataset(observations, actions)
    loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    agent.train()

    for epoch in range(100):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            output = agent(batch_obs)
            loss = F.cross_entropy(output.action_logits, batch_actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_actions)
            preds = output.action_logits.argmax(dim=-1)
            epoch_correct += (preds == batch_actions).sum().item()
            epoch_total += len(batch_actions)

        if (epoch + 1) % 20 == 0:
            acc = epoch_correct / epoch_total
            avg_loss = epoch_loss / epoch_total
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.1%}", flush=True)

    # Evaluation
    from .training.synthetic_env import SyntheticEnv

    agent.eval()
    successes = 0
    num_episodes = 200

    for i in range(num_episodes):
        env = SyntheticEnv(grid_size=10, mechanics=[mechanic], seed=50000+i)
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                obs_batch = obs.unsqueeze(0).to(device)
                output = agent(obs_batch)
                action = output.action_logits.argmax(dim=-1).item()

            result = env.step(action)
            episode_reward += result.reward
            obs = result.observation
            done = result.done

        if episode_reward > 0:
            successes += 1

    success_rate = successes / num_episodes
    print(f"SUCCESS RATE: {success_rate:.1%}", flush=True)
    return success_rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results = {}
    for mechanic in ["navigation", "collection", "switches"]:
        results[mechanic] = train_mechanic(mechanic, device)

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    all_good = True
    for mechanic, rate in results.items():
        status = "PASS" if rate > 0.8 else "FAIL"
        if rate <= 0.8:
            all_good = False
        print(f"  {mechanic}: {rate:.1%} [{status}]")

    if all_good:
        print("\nVALIDATED: Architecture can learn from expert demonstrations!")
    else:
        print("\nISSUES: Some mechanics failed")


def main_old():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Collect data
    print("\nCollecting expert data...", flush=True)
    dataset = collect_expert_dataset(
        mechanic="navigation",
        num_trajectories=2000,
        grid_size=10,
        max_steps=50,
    )

    observations, masks, actions = dataset.get_tensors(max_obs_size=10)  # No padding!
    print(f"Data: {len(actions)} transitions", flush=True)
    print(f"Observation shape: {observations.shape}", flush=True)

    # Create agent
    agent = SimpleAgent(output_dim=256).to(device)
    print(f"Parameters: {sum(p.numel() for p in agent.parameters()):,}", flush=True)

    # Test encoder differentiation
    print("\nTesting encoder...", flush=True)
    agent.eval()
    with torch.no_grad():
        sample = observations[:10].to(device)
        states = agent.encoder(sample)
        diffs = []
        for i in range(9):
            diff = (states[i] - states[i+1]).abs().mean().item()
            diffs.append(diff)
        avg_diff = sum(diffs) / len(diffs)
        print(f"Pairwise state differences: {avg_diff:.4f}")
        if avg_diff > 0.1:
            print("GOOD: Encoder differentiates states!")
        else:
            print("BAD: States too similar")

    # Training
    print("\nTraining...", flush=True)
    train_data = TensorDataset(observations, actions)
    loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    agent.train()

    for epoch in range(100):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_obs, batch_actions in loader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            output = agent(batch_obs)
            loss = F.cross_entropy(output.action_logits, batch_actions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_actions)
            preds = output.action_logits.argmax(dim=-1)
            epoch_correct += (preds == batch_actions).sum().item()
            epoch_total += len(batch_actions)

        acc = epoch_correct / epoch_total
        avg_loss = epoch_loss / epoch_total

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.1%}", flush=True)

    # Evaluation
    print("\nEvaluating...", flush=True)
    from .training.synthetic_env import SyntheticEnv

    agent.eval()
    successes = 0
    num_episodes = 200

    for i in range(num_episodes):
        env = SyntheticEnv(grid_size=10, mechanics=["navigation"], seed=50000+i)
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                obs_batch = obs.unsqueeze(0).to(device)
                output = agent(obs_batch)
                action = output.action_logits.argmax(dim=-1).item()

            result = env.step(action)
            episode_reward += result.reward
            obs = result.observation
            done = result.done

        if episode_reward > 0:
            successes += 1

    success_rate = successes / num_episodes
    print(f"\nFINAL SUCCESS RATE: {success_rate:.1%}", flush=True)

    if success_rate > 0.5:
        print("SUCCESS: Simple encoder + BC works!")
    else:
        print("FAILURE: Still not learning")


if __name__ == "__main__":
    main()
