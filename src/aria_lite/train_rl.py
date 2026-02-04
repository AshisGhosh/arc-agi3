#!/usr/bin/env python3
"""
RL Training with PPO for ARIA-Lite

Uses the validated simple encoder with proper PPO training.
"""

import torch

from .encoder_simple import SimpleGridEncoder
from .training.ppo_trainer import ActorCritic, PPOConfig, PPOTrainer, VectorizedEnv
from .training.synthetic_env import SyntheticEnv


def make_env(mechanic: str, grid_size: int = 10, max_steps: int = 50):
    """Create environment factory."""
    def _make(seed: int):
        return SyntheticEnv(
            grid_size=grid_size,
            mechanics=[mechanic],
            max_steps=max_steps,
            seed=seed * 1000,  # Different seeds per env
        )
    return _make


def evaluate(model, mechanic: str, num_episodes: int = 100, device: str = "cuda"):
    """Evaluate trained model."""
    model.eval()
    successes = 0
    total_reward = 0

    for i in range(num_episodes):
        env = SyntheticEnv(grid_size=10, mechanics=[mechanic], seed=99999 + i)
        obs = env.reset().unsqueeze(0).to(device)
        done = False
        episode_reward = 0

        while not done:
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs)

            result = env.step(action.item())
            episode_reward += result.reward
            obs = result.observation.unsqueeze(0).to(device)
            done = result.done

        total_reward += episode_reward
        if episode_reward > 0:
            successes += 1

    model.train()
    return successes / num_episodes, total_reward / num_episodes


def train_mechanic(mechanic: str, num_updates: int = 500, device: str = "cuda"):
    """Train on a single mechanic."""
    print(f"\n{'='*60}")
    print(f"RL Training: {mechanic}")
    print(f"{'='*60}", flush=True)

    # Create model
    encoder = SimpleGridEncoder(output_dim=256)
    model = ActorCritic(encoder, hidden_dim=256, num_actions=8)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    # Create config
    config = PPOConfig(
        num_envs=8,
        rollout_steps=128,
        max_episode_steps=50,
        learning_rate=3e-4,
        ppo_epochs=4,
        num_minibatches=4,
        log_interval=20,
        device=device,
    )

    # Create trainer
    trainer = PPOTrainer(model, config)

    # Create vectorized env
    env = VectorizedEnv(
        make_env(mechanic, grid_size=10, max_steps=50),
        num_envs=config.num_envs,
        device=device,
    )

    # Train
    print("\nTraining...", flush=True)
    rewards, lengths = trainer.train(env, num_updates=num_updates)

    # Evaluate
    print("\nEvaluating...", flush=True)
    success_rate, mean_reward = evaluate(model, mechanic, num_episodes=200, device=device)
    print(f"SUCCESS RATE: {success_rate:.1%}", flush=True)
    print(f"MEAN REWARD: {mean_reward:.2f}", flush=True)

    return success_rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Quick test first
    print("\n" + "="*60)
    print("QUICK VALIDATION (100 updates)")
    print("="*60, flush=True)

    success = train_mechanic("navigation", num_updates=100, device=device)

    if success < 0.3:
        print("\nWARNING: Quick test shows low success. Checking learning curve...")

    # Full training on all mechanics
    print("\n" + "="*60)
    print("FULL TRAINING (500 updates per mechanic)")
    print("="*60, flush=True)

    results = {}
    for mechanic in ["navigation", "collection", "switches"]:
        results[mechanic] = train_mechanic(mechanic, num_updates=500, device=device)

    # Summary
    print("\n" + "="*60)
    print("RL TRAINING SUMMARY")
    print("="*60)

    all_good = True
    for mechanic, rate in results.items():
        status = "PASS" if rate > 0.5 else "FAIL"
        if rate <= 0.5:
            all_good = False
        print(f"  {mechanic}: {rate:.1%} [{status}]")

    print()
    if all_good:
        print("SUCCESS: RL training validated!")
        print("Architecture can learn from rewards (not just imitation).")
    else:
        print("PARTIAL: Some mechanics need more training or tuning.")


if __name__ == "__main__":
    main()
