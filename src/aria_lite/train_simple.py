#!/usr/bin/env python3
"""
ARIA-Lite Simple Training Script

Train on single-mechanic environments to establish baseline performance.

Usage:
    PYTHONPATH=src python -m aria_lite.train_simple
"""

import torch

from .agent import create_agent
from .config import ARIALiteConfig
from .training.synthetic_env import SyntheticEnv, collect_episode
from .training.trainer import ARIALiteTrainer, TrainerConfig


def evaluate_on_mechanic(agent, mechanic: str, num_episodes: int = 50, device="cuda"):
    """Evaluate agent on a specific mechanic."""
    agent.eval()

    total_reward = 0.0
    successes = 0
    total_steps = 0

    for i in range(num_episodes):
        env = SyntheticEnv(
            grid_size=10,  # Small grid
            mechanics=[mechanic],
            max_steps=50,
            seed=i,
        )
        obs = env.reset()
        agent.reset(batch_size=1)

        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                obs_batch = obs.unsqueeze(0).to(device)
                output = agent.act(obs_batch)

            action = output.action.item() if output.action.dim() == 0 else output.action[0].item()
            result = env.step(action)

            episode_reward += result.reward
            total_steps += 1
            obs = result.observation
            done = result.done

        total_reward += episode_reward
        if episode_reward > 5:  # Goal reached
            successes += 1

    agent.train()
    return {
        "mechanic": mechanic,
        "mean_reward": total_reward / num_episodes,
        "success_rate": successes / num_episodes,
        "mean_steps": total_steps / num_episodes,
    }


def train_on_mechanic(mechanic: str, device="cuda", epochs=20):
    """Train focused on a single mechanic."""
    print(f"\n{'='*60}")
    print(f"Training on: {mechanic}")
    print(f"{'='*60}")

    # Create focused config
    config = TrainerConfig(
        device=device,

        # Reduced epochs for focused training
        wm_epochs=epochs,
        fp_epochs=epochs,
        sp_epochs=epochs,
        arb_epochs=5,
        joint_epochs=epochs,

        # Standard batches
        wm_batch_size=32,
        fp_batch_size=64,
        sp_batch_size=32,
        joint_batch_size=32,

        # Smaller buffer
        buffer_capacity=10000,
        min_buffer_size=500,

        # Smaller, single-mechanic environments
        env_min_grid=8,
        env_max_grid=12,

        log_every=5,
    )

    aria_config = ARIALiteConfig()
    agent = create_agent(aria_config)
    trainer = ARIALiteTrainer(agent, config, aria_config)

    # Override environment generator to use single mechanic
    from .training.synthetic_env import SyntheticEnvGenerator
    trainer.env_generator = SyntheticEnvGenerator(
        min_grid_size=8,
        max_grid_size=12,
        min_mechanics=1,
        max_mechanics=1,  # Single mechanic only
    )
    # Monkey-patch to always use specified mechanic
    original_generate = trainer.env_generator.generate
    def focused_generate(seed=None):
        env = SyntheticEnv(
            grid_size=10,
            mechanics=[mechanic],
            max_steps=50,
            seed=seed,
        )
        return env
    trainer.env_generator.generate = focused_generate

    # Collect initial data
    print("\nCollecting experience...")
    trainer.collect_data(200)

    # Train phases
    print("\nTraining world model...")
    trainer.train_world_model(num_epochs=epochs)

    print("\nTraining fast policy...")
    trainer.train_fast_policy(num_epochs=epochs)

    print("\nTraining slow policy...")
    trainer.train_slow_policy(num_epochs=epochs)

    print("\nJoint fine-tuning...")
    trainer.joint_finetune(num_epochs=epochs)

    # Evaluate
    print("\nEvaluating...")
    results = evaluate_on_mechanic(agent, mechanic, num_episodes=100, device=device)

    print(f"\nResults for {mechanic}:")
    print(f"  Success rate: {results['success_rate']:.1%}")
    print(f"  Mean reward: {results['mean_reward']:.2f}")
    print(f"  Mean steps: {results['mean_steps']:.1f}")

    return agent, results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Train on each mechanic separately
    mechanics = ["navigation", "collection", "switches"]
    all_results = {}

    for mechanic in mechanics:
        agent, results = train_on_mechanic(mechanic, device=device, epochs=30)
        all_results[mechanic] = results

        # Save checkpoint for this mechanic
        torch.save(agent.state_dict(), f"checkpoints/aria_lite/{mechanic}_agent.pt")

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    for mechanic, results in all_results.items():
        print(f"\n{mechanic}:")
        print(f"  Success rate: {results['success_rate']:.1%}")
        print(f"  Mean reward: {results['mean_reward']:.2f}")


if __name__ == "__main__":
    main()
