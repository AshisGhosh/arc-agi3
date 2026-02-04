"""
Meta-learning training script.

Tests whether the agent can:
1. Learn from few demonstrations
2. Generalize to new tasks from the same family
3. Transfer across task families
"""

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aria_lite.meta import MetaLearningAgent
from src.aria_lite.primitives import (
    NavigationEnv,
    ClickEnv,
    PrimitiveFamily,
    PrimitiveGenerator,
)
from src.aria_lite.primitives.base import Action


# ============================================================================
# Expert Solvers (simplified)
# ============================================================================


def nav_expert(env: NavigationEnv) -> int:
    """Simple greedy navigation expert."""
    if not env.goals or env.agent_pos is None:
        return Action.NOOP

    target = env.goals[0]
    ay, ax = env.agent_pos
    gy, gx = target

    if ay < gy:
        return Action.DOWN
    elif ay > gy:
        return Action.UP
    elif ax < gx:
        return Action.RIGHT
    elif ax > gx:
        return Action.LEFT
    return Action.NOOP


def click_expert(env: ClickEnv) -> tuple[int, int, int]:
    """Click expert."""
    for target in env.targets:
        if target not in env.clicked:
            y, x = target
            return Action.CLICK, x, y
    return Action.NOOP, 0, 0


# ============================================================================
# Data Collection
# ============================================================================


def collect_task_episodes(
    family: PrimitiveFamily,
    num_tasks: int = 100,
    demos_per_task: int = 3,
    queries_per_task: int = 5,
    difficulty: int = 1,
):
    """
    Collect meta-learning data: demonstrations + query episodes per task.

    Returns list of tasks, each with:
    - demo_obs: [K, T, H, W] demonstration observations
    - demo_actions: [K, T] demonstration actions
    - query_obs: [Q, H, W] query observations
    - query_actions: [Q] target actions for queries
    """
    gen = PrimitiveGenerator()
    tasks = []

    for _ in range(num_tasks):
        # Create task (same seed for demos and queries)
        task_seed = random.randint(0, 2**31)

        demo_obs_list = []
        demo_actions_list = []

        # Collect demonstrations
        for _ in range(demos_per_task):
            if family == PrimitiveFamily.NAVIGATION:
                env = gen.generate(family=family, difficulty=difficulty)
            else:
                env = gen.generate(family=family, difficulty=difficulty)

            obs = env.reset()
            episode_obs = [obs.clone()]
            episode_actions = []

            for _ in range(min(10, env.max_steps)):  # Limit demo length
                if family == PrimitiveFamily.NAVIGATION:
                    action = nav_expert(env)
                    result = env.step(action)
                else:
                    action, x, y = click_expert(env)
                    result = env.step(action, x, y)

                episode_actions.append(action)
                episode_obs.append(result.observation.clone())

                if result.done:
                    break

            # Pad to fixed length
            max_len = 10
            while len(episode_obs) < max_len + 1:
                episode_obs.append(episode_obs[-1].clone())
                episode_actions.append(Action.NOOP)

            demo_obs_list.append(torch.stack(episode_obs[:max_len]))
            demo_actions_list.append(torch.tensor(episode_actions[:max_len]))

        # Collect query examples (single step predictions)
        query_obs_list = []
        query_actions_list = []

        for _ in range(queries_per_task):
            if family == PrimitiveFamily.NAVIGATION:
                env = gen.generate(family=family, difficulty=difficulty)
            else:
                env = gen.generate(family=family, difficulty=difficulty)

            obs = env.reset()

            if family == PrimitiveFamily.NAVIGATION:
                action = nav_expert(env)
            else:
                action, _, _ = click_expert(env)

            query_obs_list.append(obs.clone())
            query_actions_list.append(action)

        tasks.append({
            "demo_obs": torch.stack(demo_obs_list),  # [K, T, H, W]
            "demo_actions": torch.stack(demo_actions_list),  # [K, T]
            "query_obs": torch.stack(query_obs_list),  # [Q, H, W]
            "query_actions": torch.tensor(query_actions_list),  # [Q]
            "family": family,
        })

    return tasks


# ============================================================================
# Training
# ============================================================================


def train_meta(
    family: PrimitiveFamily = PrimitiveFamily.NAVIGATION,
    num_tasks: int = 200,
    num_epochs: int = 50,
    demos_per_task: int = 3,
):
    """Train meta-learning agent on a primitive family."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training meta-learning on {family.name}, device={device}")

    # Collect data
    print("Collecting training tasks...")
    train_tasks = collect_task_episodes(family, num_tasks, demos_per_task)
    print(f"Collected {len(train_tasks)} tasks")

    # Model
    model = MetaLearningAgent(
        hidden_dim=128,
        task_dim=64,
        num_colors=16,
        num_actions=9,
        max_grid=20,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 16

    for epoch in range(num_epochs):
        random.shuffle(train_tasks)

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(train_tasks), batch_size):
            batch_tasks = train_tasks[i:i+batch_size]
            B = len(batch_tasks)

            # Stack batch - use first demo step for simplicity
            demo_obs = torch.stack([t["demo_obs"][:, 0, :, :] for t in batch_tasks]).to(device)  # [B, K, H, W]
            demo_actions = torch.stack([t["demo_actions"][:, 0] for t in batch_tasks]).to(device)  # [B, K]
            query_obs = torch.stack([t["query_obs"][0] for t in batch_tasks]).to(device)  # [B, H, W]
            query_actions = torch.stack([t["query_actions"][0:1] for t in batch_tasks]).squeeze(-1).to(device)  # [B]

            # Forward pass
            out = model.act(
                obs=query_obs,
                demo_obs=demo_obs,
                demo_actions=demo_actions,
                grid_size=query_obs.shape[1],
            )

            # Loss on action prediction
            loss = F.cross_entropy(out["action_logits"], query_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = out["action_logits"].argmax(-1)
            correct += (pred == query_actions).sum().item()
            total += B

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/total:.1%}")

    # Evaluate on new tasks
    print("\nEvaluating on new tasks...")
    eval_tasks = collect_task_episodes(family, 50, demos_per_task)

    correct = 0
    total = 0

    for task in eval_tasks:
        demo_obs = task["demo_obs"][:, 0, :, :].unsqueeze(0).to(device)  # [1, K, H, W]
        demo_actions = task["demo_actions"][:, 0].unsqueeze(0).to(device)  # [1, K]

        for q_idx in range(len(task["query_obs"])):
            query_obs = task["query_obs"][q_idx].unsqueeze(0).to(device)  # [1, H, W]
            target = task["query_actions"][q_idx].item()

            with torch.no_grad():
                out = model.act(
                    obs=query_obs,
                    demo_obs=demo_obs,
                    demo_actions=demo_actions,
                    grid_size=query_obs.shape[1],
                )
                pred = out["action_logits"].argmax(-1).item()

            correct += int(pred == target)
            total += 1

    eval_acc = correct / total
    print(f"\nMeta-learning eval accuracy: {eval_acc:.1%}")
    return eval_acc


def test_transfer():
    """Test if meta-learning transfers across task families."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train on navigation
    print("=" * 60)
    print("Training on NAVIGATION")
    print("=" * 60)
    nav_acc = train_meta(PrimitiveFamily.NAVIGATION, num_tasks=200, num_epochs=30)

    # Train on click
    print("\n" + "=" * 60)
    print("Training on CLICK")
    print("=" * 60)
    click_acc = train_meta(PrimitiveFamily.CLICK, num_tasks=200, num_epochs=30)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Navigation meta-learning: {nav_acc:.1%} {'PASS' if nav_acc > 0.5 else 'FAIL'}")
    print(f"Click meta-learning: {click_acc:.1%} {'PASS' if click_acc > 0.5 else 'FAIL'}")

    return nav_acc, click_acc


if __name__ == "__main__":
    test_transfer()
