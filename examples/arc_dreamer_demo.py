#!/usr/bin/env python3
"""
ARC-DREAMER v2 Demo

This script demonstrates the ARC-DREAMER v2 agent on ARC-AGI-3 environments.

Usage:
    uv run python examples/arc_dreamer_demo.py
    uv run python examples/arc_dreamer_demo.py --game vc33
    uv run python examples/arc_dreamer_demo.py --no-planning
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_api_key() -> bool:
    """Check if API key is configured."""
    if not os.getenv("ARC_API_KEY"):
        print("=" * 60)
        print("ARC-AGI-3 Setup Required")
        print("=" * 60)
        print()
        print("API key not found. Please configure your environment:")
        print()
        print("1. Get your API key from: https://three.arcprize.org/")
        print("2. Add it to your .env file:")
        print("   ARC_API_KEY=your-api-key-here")
        print()
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="ARC-DREAMER v2 Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python arc_dreamer_demo.py                    Run on ls20 with full planning
    python arc_dreamer_demo.py --game vc33        Run on vc33
    python arc_dreamer_demo.py --no-planning      Run without MCTS planning
    python arc_dreamer_demo.py --max-steps 200    Run for 200 steps
        """,
    )
    parser.add_argument("--game", "-g", default="ls20", help="Game ID to play (default: ls20)")
    parser.add_argument(
        "--max-steps", "-m", type=int, default=100, help="Maximum steps per episode (default: 100)"
    )
    parser.add_argument(
        "--no-planning",
        action="store_true",
        help="Disable MCTS planning (use hierarchical policy only)",
    )
    parser.add_argument("--no-render", action="store_true", help="Disable terminal rendering")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Check API key
    if not check_api_key():
        sys.exit(1)

    print("=" * 60)
    print("ARC-DREAMER v2 Agent Demo")
    print("=" * 60)
    print()
    print("Components:")
    print("  1. Error-Correcting World Model (ensemble + grounding)")
    print("  2. Principled Intrinsic Motivation (information-theoretic)")
    print("  3. Defined Hierarchical Policy (options + subgoals)")
    print("  4. Goal Discovery (contrastive learning)")
    print("  5. Belief State Tracking (POMDP)")
    print("  6. Symbolic Grounding (slot attention)")
    print("  7. Extended Planning (MCTS 50+ steps)")
    print()
    print(f"Game: {args.game}")
    print(f"Max steps: {args.max_steps}")
    print(f"Planning: {'Enabled' if not args.no_planning else 'Disabled'}")
    print("-" * 60)

    # Import ARC-AGI components
    try:
        import arc_agi
        from arcengine import GameAction, GameState
    except ImportError as e:
        print(f"Error: Could not import arc-agi: {e}")
        print("Run 'uv sync' to install dependencies")
        sys.exit(1)

    # Import agent from src
    from arc_dreamer_v2 import AgentConfig, ARCDreamerV2Agent

    # Create agent
    print("\nInitializing ARC-DREAMER v2 agent...")
    config = AgentConfig(
        mcts_simulations=50 if not args.no_planning else 0,
    )
    agent = ARCDreamerV2Agent(config)
    print(f"  Device: {agent.device}")
    print(f"  State dim: {config.state_dim}")
    print(f"  World model ensemble size: {config.num_ensemble}")
    print(f"  MCTS simulations: {config.mcts_simulations}")

    # Create environment
    print(f"\nCreating environment for '{args.game}'...")
    arc = arc_agi.Arcade()
    render_mode = None if args.no_render else "terminal"
    env = arc.make(args.game, render_mode=render_mode)

    # Get initial observation
    observation = env.observation_space
    print(f"Initial state: {observation.state}")
    print(f"Levels to complete: {observation.win_levels}")

    # Convert frame to tensor
    def frame_to_tensor(frame) -> torch.Tensor:
        """Convert frame data to tensor."""
        if hasattr(frame, "frame") and frame.frame is not None:
            # Frame is a 2D grid of colors
            return torch.tensor(frame.frame, dtype=torch.long)
        else:
            # Placeholder if no frame data
            return torch.zeros(30, 30, dtype=torch.long)

    # Main loop
    print("\nStarting game loop...")
    print("-" * 60)

    step_count = 0
    episode_reward = 0.0
    start_time = time.time()

    use_planning = not args.no_planning

    while step_count < args.max_steps:
        observation = env.observation_space

        # Check game state
        if observation.state == GameState.WIN:
            print(f"\nWon after {step_count} steps!")
            break
        elif observation.state == GameState.NOT_PLAYED:
            action = GameAction.RESET
            if args.verbose:
                print(f"Step {step_count}: RESET (starting game)")
        elif observation.state == GameState.GAME_OVER:
            action = GameAction.RESET
            if args.verbose:
                print(f"Step {step_count}: RESET (game over)")
        else:
            # Get frame as tensor
            frame = frame_to_tensor(observation)

            # Agent selects action
            action_idx, info = agent.act(frame, use_planning=use_planning)

            # Map to GameAction
            if action_idx == 0:
                action = GameAction.RESET
            else:
                action = getattr(GameAction, f"ACTION{min(action_idx, 7)}")

            if args.verbose:
                print(f"Step {step_count}: {action.name}")
                if "strategy" in info:
                    print(f"  Strategy: {info.get('strategy')}")
                if "subgoal" in info:
                    print(f"  Subgoal: {info.get('subgoal')}")
                if "replanned" in info:
                    print(f"  Replanned: {info.get('replanned')}")

        # Execute action
        prev_observation = observation
        env.step(action)
        observation = env.observation_space

        # Determine reward (heuristic based on state change)
        reward = 0.0
        if observation.state == GameState.WIN:
            reward = 10.0
        elif observation.levels_completed > prev_observation.levels_completed:
            reward = 1.0

        episode_reward += reward

        # Agent observes transition
        if prev_observation.state == GameState.PLAYING:
            prev_frame = frame_to_tensor(prev_observation)
            next_frame = frame_to_tensor(observation)
            done = observation.state in [GameState.WIN, GameState.GAME_OVER]

            agent.observe(
                prev_frame,
                action_idx if "action_idx" in dir() else 0,
                reward,
                next_frame,
                done,
            )

        step_count += 1

        # Periodic progress report
        if step_count % 20 == 0:
            elapsed = time.time() - start_time
            fps = step_count / elapsed if elapsed > 0 else 0
            print(
                f"\nProgress: {step_count}/{args.max_steps} steps, "
                f"{observation.levels_completed}/{observation.win_levels} levels, "
                f"{fps:.1f} steps/sec"
            )

    # Final statistics
    elapsed = time.time() - start_time
    fps = step_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total steps: {step_count}")
    print(f"Episode reward: {episode_reward:.2f}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Steps/second: {fps:.2f}")

    # Agent statistics
    stats = agent.get_statistics()
    print("\nAgent Statistics:")
    print(f"  Total episodes: {stats['episode_count']}")
    print(f"  Total steps: {stats['total_steps']}")

    if stats.get("intrinsic_motivation"):
        im = stats["intrinsic_motivation"]
        print(f"  Unique states visited: {im.get('unique_states', 0)}")
        print(f"  Avg alpha (dynamics): {im.get('avg_alpha', 0):.3f}")
        print(f"  Avg beta (entropy): {im.get('avg_beta', 0):.3f}")
        print(f"  Avg gamma (coverage): {im.get('avg_gamma', 0):.3f}")

    if stats.get("grounding"):
        gr = stats["grounding"]
        print(f"  Total groundings: {gr.get('total_groundings', 0)}")
        print(f"  Avg reliability: {gr.get('avg_reliability', 0):.3f}")

    if stats.get("anomaly_detection"):
        ad = stats["anomaly_detection"]
        print(f"  Anomalies detected: {ad.get('total_anomalies', 0)}")

    if stats.get("planning"):
        pl = stats["planning"]
        print(f"  Total plans: {pl.get('total_plans', 0)}")
        print(f"  Avg simulations: {pl.get('avg_simulations', 0):.1f}")
        print(f"  Avg tree depth: {pl.get('avg_depth', 0):.1f}")

    # Scorecard
    print("\n" + "-" * 60)
    print("Scorecard:")
    print("-" * 60)
    scorecard = arc.get_scorecard()
    print(scorecard)


if __name__ == "__main__":
    main()
