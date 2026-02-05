#!/usr/bin/env python3
"""
Train ARIA-Lite with Behavioral Cloning on expert data.

This validates that the architecture can represent solutions.

Usage:
    PYTHONPATH=src python -m aria_lite.train_bc
"""

import torch

from .config import ARIALiteConfig
from .agent import create_agent
from .training.bc_trainer import BCTrainer, BCConfig
from .training.expert_data import collect_expert_dataset, collect_mixed_dataset


def train_single_mechanic(
    mechanic: str,
    num_trajectories: int = 5000,
    epochs: int = 100,
    device: str = "cuda",
):
    """Train and evaluate on a single mechanic."""
    print(f"\n{'='*60}")
    print(f"BC Training: {mechanic}")
    print(f"{'='*60}")

    # Collect expert data
    print("\nCollecting expert data...")
    dataset = collect_expert_dataset(
        mechanic=mechanic,
        num_trajectories=num_trajectories,
        grid_size=10,
        max_steps=50,
    )

    print(f"Collected: {len(dataset.trajectories)} trajectories, "
          f"{dataset.num_transitions} transitions")
    print(f"Expert success rate: {dataset.success_rate:.1%}")

    if dataset.success_rate < 0.9:
        print("WARNING: Expert success rate is low. Solver may have issues.")

    # Create agent and trainer
    config = BCConfig(
        epochs=epochs,
        batch_size=128,
        learning_rate=3e-4,
        device=device,
        log_every=10,
        eval_every=20,
    )

    aria_config = ARIALiteConfig()
    agent = create_agent(aria_config)
    trainer = BCTrainer(agent, config)

    # Train
    print("\nTraining...")
    metrics = trainer.train(dataset, mechanic)

    # Final evaluation
    print("\nFinal evaluation (200 episodes)...")
    final_success = trainer.evaluate(mechanic, num_episodes=200)
    print(f"\nFINAL SUCCESS RATE: {final_success:.1%}")

    return agent, final_success


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}

    # Train on each mechanic
    for mechanic in ["navigation", "collection", "switches"]:
        agent, success_rate = train_single_mechanic(
            mechanic=mechanic,
            num_trajectories=5000,
            epochs=100,
            device=device,
        )

        results[mechanic] = success_rate

        # Save checkpoint
        torch.save(
            agent.state_dict(),
            f"checkpoints/aria_lite/bc_{mechanic}.pt"
        )

    # Summary
    print("\n" + "="*60)
    print("BC TRAINING SUMMARY")
    print("="*60)

    all_good = True
    for mechanic, success in results.items():
        status = "PASS" if success > 0.8 else "FAIL"
        if success <= 0.8:
            all_good = False
        print(f"  {mechanic}: {success:.1%} [{status}]")

    print()
    if all_good:
        print("SUCCESS: Architecture validated. Policies can represent solutions.")
    else:
        print("FAILURE: Architecture issues detected. Investigate before proceeding.")


if __name__ == "__main__":
    main()
