#!/usr/bin/env python3
"""
Random Agent for ARC-AGI-3

This agent selects random actions to explore the environment.
Useful for understanding how the game mechanics work.

Usage:
    uv run python examples/random_agent.py
    uv run python examples/random_agent.py --game vc33
    uv run python examples/random_agent.py --max-actions 100
"""

import argparse
import os
import random
import sys
import time

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("ARC_API_KEY"):
    print("Error: ARC_API_KEY not found in environment")
    print("Please set your API key in .env file or export it:")
    print("  export ARC_API_KEY='your-api-key-here'")
    sys.exit(1)

import arc_agi  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402


class RandomAgent:
    """An agent that selects random actions."""

    def __init__(self, seed: int | None = None):
        """Initialize the random agent.

        Args:
            seed: Random seed for reproducibility. If None, uses current time.
        """
        if seed is None:
            seed = int(time.time() * 1000000) % (2**32)
        random.seed(seed)
        print(f"Random seed: {seed}")

    def is_done(self, state: GameState, levels_completed: int, win_levels: int) -> bool:
        """Check if we should stop playing.

        Args:
            state: Current game state
            levels_completed: Number of levels completed
            win_levels: Number of levels needed to win

        Returns:
            True if we should stop
        """
        return state == GameState.WIN

    def choose_action(self, state: GameState, available_actions: list | None = None) -> GameAction:
        """Choose a random action.

        Args:
            state: Current game state
            available_actions: List of valid actions (if provided)

        Returns:
            The chosen GameAction
        """
        # Handle game states that require RESET
        if state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            return GameAction.RESET

        # Get list of actions to choose from
        if available_actions:
            actions = available_actions
        else:
            # All actions except RESET
            actions = [a for a in GameAction if a is not GameAction.RESET]

        # Choose random action
        action = random.choice(actions)

        # If action requires coordinates, set random values
        if action.is_complex():
            action.set_data(
                {
                    "x": random.randint(0, 63),
                    "y": random.randint(0, 63),
                }
            )
            action.reasoning = {
                "action": action.name,
                "reason": "Random exploration",
            }
        else:
            action.reasoning = f"Random choice: {action.name}"

        return action


def main():
    """Run the random agent."""
    parser = argparse.ArgumentParser(description="Random agent for ARC-AGI-3")
    parser.add_argument("--game", "-g", default="ls20", help="Game ID to play (default: ls20)")
    parser.add_argument(
        "--max-actions",
        "-m",
        type=int,
        default=80,
        help="Maximum actions before stopping (default: 80)",
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable terminal rendering (faster)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ARC-AGI-3 Random Agent")
    print("=" * 60)
    print(f"Game: {args.game}")
    print(f"Max actions: {args.max_actions}")

    # Create agent
    agent = RandomAgent(seed=args.seed)

    # Create arcade and environment
    arc = arc_agi.Arcade()
    render_mode = None if args.no_render else "terminal"
    env = arc.make(args.game, render_mode=render_mode)

    # Main game loop
    action_count = 0
    start_time = time.time()

    print("\nStarting game loop...")
    print("-" * 40)

    while action_count < args.max_actions:
        observation = env.observation_space

        # Check if done
        if agent.is_done(observation.state, observation.levels_completed, observation.win_levels):
            print(f"\n🎉 Won after {action_count} actions!")
            break

        # Choose action
        action = agent.choose_action(observation.state, observation.available_actions)

        # Log the action
        if action_count % 10 == 0:  # Log every 10th action
            print(
                f"Action {action_count + 1}: {action.name} "
                f"(levels: {observation.levels_completed}/{observation.win_levels})"
            )

        # Execute action
        env.step(action)
        action_count += 1

    # Calculate statistics
    elapsed = time.time() - start_time
    fps = action_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Total actions: {action_count}")
    print(f"Time elapsed: {elapsed:.2f}s")
    print(f"Actions/second: {fps:.2f}")

    # Get scorecard
    print("\nScorecard:")
    print("-" * 40)
    scorecard = arc.get_scorecard()
    print(scorecard)


if __name__ == "__main__":
    main()
