#!/usr/bin/env python3
"""
Focused training on navigation mechanic.

Usage:
    PYTHONPATH=src python -m aria_lite.train_navigation
"""

import torch

from .agent import create_agent
from .config import ARIALiteConfig
from .training.synthetic_env import SyntheticEnv
from .training.trainer import ARIALiteTrainer, TrainerConfig


def evaluate(agent, num_episodes: int = 100, device="cuda"):
    """Evaluate agent on navigation."""
    agent.eval()

    total_reward = 0.0
    successes = 0

    for i in range(num_episodes):
        env = SyntheticEnv(
            grid_size=10,
            mechanics=["navigation"],
            max_steps=50,
            seed=1000 + i,
        )
        obs = env.reset()
        agent.reset(batch_size=1)

        done = False
        episode_reward = 0.0

        while not done:
            with torch.no_grad():
                obs_batch = obs.unsqueeze(0).to(device)
                output = agent.act(obs_batch)

            action = output.action[0].item()
            result = env.step(action)

            episode_reward += result.reward
            obs = result.observation
            done = result.done

        total_reward += episode_reward
        if episode_reward > 5:
            successes += 1

    agent.train()
    return successes / num_episodes, total_reward / num_episodes


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("="*60)
    print("Focused Navigation Training")
    print("="*60)

    config = TrainerConfig(
        device=device,
        wm_epochs=50,
        fp_epochs=100,  # More epochs for policy
        sp_epochs=100,
        arb_epochs=10,
        joint_epochs=50,
        wm_batch_size=64,
        fp_batch_size=128,
        sp_batch_size=64,
        joint_batch_size=64,
        buffer_capacity=50000,
        env_min_grid=8,
        env_max_grid=12,
        log_every=10,
    )

    aria_config = ARIALiteConfig()
    agent = create_agent(aria_config)
    trainer = ARIALiteTrainer(agent, config, aria_config)

    # Override to navigation only
    def nav_env(seed=None):
        return SyntheticEnv(
            grid_size=10,
            mechanics=["navigation"],
            max_steps=50,
            seed=seed,
        )
    trainer.env_generator.generate = nav_env

    # Initial evaluation
    print("\nInitial evaluation...")
    success_rate, mean_reward = evaluate(agent, 50, device)
    print(f"Before training: {success_rate:.1%} success, {mean_reward:.2f} reward")

    # Collect data
    print("\nCollecting experience (500 episodes)...")
    trainer.collect_data(500)

    # Train world model
    print("\nPhase 1: World Model...")
    trainer.train_world_model()
    success_rate, mean_reward = evaluate(agent, 50, device)
    print(f"After WM: {success_rate:.1%} success, {mean_reward:.2f} reward")

    # Train fast policy
    print("\nPhase 2: Fast Policy...")
    trainer.train_fast_policy()
    success_rate, mean_reward = evaluate(agent, 50, device)
    print(f"After FP: {success_rate:.1%} success, {mean_reward:.2f} reward")

    # Train slow policy
    print("\nPhase 3: Slow Policy...")
    trainer.train_slow_policy()
    success_rate, mean_reward = evaluate(agent, 50, device)
    print(f"After SP: {success_rate:.1%} success, {mean_reward:.2f} reward")

    # Joint fine-tuning
    print("\nPhase 5: Joint Fine-tuning...")
    trainer.joint_finetune()
    success_rate, mean_reward = evaluate(agent, 100, device)
    print(f"After Joint: {success_rate:.1%} success, {mean_reward:.2f} reward")

    # Final evaluation
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    success_rate, mean_reward = evaluate(agent, 200, device)
    print(f"Success rate: {success_rate:.1%}")
    print(f"Mean reward: {mean_reward:.2f}")

    # Save
    torch.save(agent.state_dict(), "checkpoints/aria_lite/navigation_focused.pt")
    print("\nCheckpoint saved.")


if __name__ == "__main__":
    main()
