#!/usr/bin/env python
"""
Run A/B test comparing exploration policies.

Usage:
    python -m src.aria_v2.core.run_ab_test

This script:
1. Trains a learned exploration policy on synthetic games
2. Runs both random and learned policies on test games
3. Reports comparison results
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agent import ARIAAgent, AgentConfig, EpisodeStats
from .exploration_training import ExplorationTrainer, TrainingConfig, SyntheticGameSimulator


def run_policy_test(
    policy_type: str,
    num_episodes: int,
    max_steps: int,
    seed: int,
    trained_trainer: ExplorationTrainer = None,
) -> dict:
    """
    Run test episodes with a specific exploration policy.

    Tests the policy directly (not through the full agent) for clean comparison.
    """
    from .exploration import (
        ExplorationPolicy,
        RandomExplorationPolicy,
        SystematicExplorationPolicy,
        LearnedExplorationPolicy,
        ExplorationStrategy,
    )
    from .belief_state import BeliefState

    results = {
        "policy_type": policy_type,
        "num_episodes": num_episodes,
        "episodes": [],
        "total_levels": 0,
        "total_actions": 0,
        "player_found_count": 0,
        "avg_info_gain": 0.0,
    }

    # Create policy
    if policy_type == "random":
        policy = RandomExplorationPolicy(seed=seed)
    elif policy_type == "systematic":
        policy = SystematicExplorationPolicy()
    elif policy_type == "learned":
        if trained_trainer is not None:
            policy = trained_trainer.get_trained_policy()
        else:
            policy = LearnedExplorationPolicy()
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    for ep in range(num_episodes):
        # Create simulator
        sim = SyntheticGameSimulator(seed=seed + ep + 1000)
        belief_state = sim.reset()

        # Run episode
        total_info_gain = 0.0
        player_found = False

        for step in range(max_steps):
            if sim.done:
                break

            frame = sim.get_frame()

            # Get policy decision
            decision = policy.decide(belief_state, frame, [1, 2, 3, 4])
            strategy = decision.strategy

            # Step simulation with the chosen strategy
            belief_state, info_gain, level_completed = sim.step(strategy)
            total_info_gain += info_gain

            if belief_state.player_identified and not player_found:
                player_found = True

        # Record episode results
        episode_result = {
            "actions": sim.actions,
            "levels_completed": sim.done,
            "player_found": player_found,
            "info_gain": total_info_gain,
        }
        results["episodes"].append(episode_result)

        results["total_actions"] += sim.actions
        results["total_levels"] += 1 if sim.done else 0
        results["player_found_count"] += 1 if player_found else 0
        results["avg_info_gain"] += total_info_gain

    results["avg_info_gain"] /= num_episodes
    results["level_complete_rate"] = results["total_levels"] / num_episodes
    results["player_found_rate"] = results["player_found_count"] / num_episodes
    results["avg_actions"] = results["total_actions"] / num_episodes

    return results


def run_ab_test(
    training_epochs: int = 50,
    training_episodes_per_epoch: int = 100,
    test_episodes: int = 200,
    max_steps: int = 100,
    seed: int = 42,
) -> dict:
    """
    Run full A/B test.

    Returns comparison results.
    """
    print("=" * 60)
    print("ARIA v2 A/B Test: Random vs Learned Exploration")
    print("=" * 60)
    print()

    # Phase 1: Train learned policy
    print("Phase 1: Training Learned Exploration Policy")
    print("-" * 40)

    config = TrainingConfig(
        num_epochs=training_epochs,
        batch_size=32,
    )
    trainer = ExplorationTrainer(config)
    trainer.train(num_epochs=training_epochs, episodes_per_epoch=training_episodes_per_epoch)

    print()

    # Phase 2: Test random policy
    print("Phase 2: Testing Random Policy")
    print("-" * 40)
    start_time = time.time()
    random_results = run_policy_test(
        policy_type="random",
        num_episodes=test_episodes,
        max_steps=max_steps,
        seed=seed,
    )
    random_time = time.time() - start_time
    print(f"  Completed {test_episodes} episodes in {random_time:.1f}s")
    print(f"  Level complete rate: {random_results['level_complete_rate']:.1%}")
    print(f"  Avg info gain: {random_results['avg_info_gain']:.3f}")
    print()

    # Phase 3: Test systematic policy
    print("Phase 3: Testing Systematic Policy")
    print("-" * 40)
    start_time = time.time()
    systematic_results = run_policy_test(
        policy_type="systematic",
        num_episodes=test_episodes,
        max_steps=max_steps,
        seed=seed,
    )
    systematic_time = time.time() - start_time
    print(f"  Completed {test_episodes} episodes in {systematic_time:.1f}s")
    print(f"  Level complete rate: {systematic_results['level_complete_rate']:.1%}")
    print(f"  Avg info gain: {systematic_results['avg_info_gain']:.3f}")
    print()

    # Phase 4: Test learned policy
    print("Phase 4: Testing Learned Policy")
    print("-" * 40)
    start_time = time.time()
    learned_results = run_policy_test(
        policy_type="learned",
        num_episodes=test_episodes,
        max_steps=max_steps,
        seed=seed,
        trained_trainer=trainer,
    )
    learned_time = time.time() - start_time
    print(f"  Completed {test_episodes} episodes in {learned_time:.1f}s")
    print(f"  Level complete rate: {learned_results['level_complete_rate']:.1%}")
    print(f"  Avg info gain: {learned_results['avg_info_gain']:.3f}")
    print()

    # Comparison
    print("=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print()
    print(f"{'Metric':<25} {'Random':>12} {'Systematic':>12} {'Learned':>12}")
    print("-" * 60)
    print(f"{'Level Complete Rate':<25} {random_results['level_complete_rate']:>11.1%} {systematic_results['level_complete_rate']:>11.1%} {learned_results['level_complete_rate']:>11.1%}")
    print(f"{'Avg Info Gain':<25} {random_results['avg_info_gain']:>12.3f} {systematic_results['avg_info_gain']:>12.3f} {learned_results['avg_info_gain']:>12.3f}")
    print(f"{'Player Found Rate':<25} {random_results['player_found_rate']:>11.1%} {systematic_results['player_found_rate']:>11.1%} {learned_results['player_found_rate']:>11.1%}")
    print(f"{'Avg Actions':<25} {random_results['avg_actions']:>12.1f} {systematic_results['avg_actions']:>12.1f} {learned_results['avg_actions']:>12.1f}")
    print()

    # Calculate improvements
    random_baseline = random_results['avg_info_gain']
    if random_baseline > 0:
        systematic_improvement = (systematic_results['avg_info_gain'] - random_baseline) / random_baseline
        learned_improvement = (learned_results['avg_info_gain'] - random_baseline) / random_baseline

        print("Improvements vs Random:")
        print(f"  Systematic: {systematic_improvement:+.1%}")
        print(f"  Learned:    {learned_improvement:+.1%}")
        print()

        # Winner
        if learned_improvement > systematic_improvement:
            print("WINNER: Learned exploration policy")
            print(f"  Improvement over systematic: {(learned_improvement - systematic_improvement):.1%}")
        else:
            print("WINNER: Systematic exploration policy")
            print(f"  Learned was {(systematic_improvement - learned_improvement):.1%} worse")

    print()

    return {
        "random": random_results,
        "systematic": systematic_results,
        "learned": learned_results,
        "training_stats": trainer.training_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Run A/B test for exploration policies")
    parser.add_argument("--training-epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--training-episodes", type=int, default=100, help="Episodes per training epoch")
    parser.add_argument("--test-episodes", type=int, default=200, help="Number of test episodes")
    parser.add_argument("--max-steps", type=int, default=100, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    results = run_ab_test(
        training_epochs=args.training_epochs,
        training_episodes_per_epoch=args.training_episodes,
        test_episodes=args.test_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        def clean_results(obj):
            if isinstance(obj, dict):
                return {k: clean_results(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_results(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        with open(output_path, "w") as f:
            json.dump(clean_results(results), f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
