#!/usr/bin/env python3
"""
Basic ARC-AGI-3 Agent Example

This script demonstrates the fundamental structure of an ARC-AGI-3 agent.
It creates a simple game loop that interacts with an environment.

Usage:
    uv run python examples/basic_agent.py
"""

import os
import sys

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


def main():
    """Run a basic agent that takes a few actions in the ls20 game."""
    print("=" * 60)
    print("ARC-AGI-3 Basic Agent Example")
    print("=" * 60)

    # Create an arcade instance
    arc = arc_agi.Arcade()

    # Create an environment for the ls20 game
    # Use render_mode="terminal" to see the game state
    print("\nCreating environment for 'ls20' game...")
    env = arc.make("ls20", render_mode="terminal")

    # Get initial observation
    observation = env.observation_space
    print(f"\nInitial state: {observation.state}")
    print(f"Levels to complete: {observation.win_levels}")

    # Take some actions
    max_actions = 20
    action_count = 0

    print(f"\nTaking up to {max_actions} actions...")
    print("-" * 40)

    while action_count < max_actions:
        observation = env.observation_space

        # Check game state
        if observation.state == GameState.WIN:
            print(f"\n🎉 Won after {action_count} actions!")
            break
        elif observation.state == GameState.NOT_PLAYED:
            # Game hasn't started, send RESET
            action = GameAction.RESET
            print(f"Action {action_count + 1}: RESET (starting game)")
        elif observation.state == GameState.GAME_OVER:
            # Game over, can reset to try again
            action = GameAction.RESET
            print(f"Action {action_count + 1}: RESET (restarting)")
        else:
            # Game is playing, take an action
            # For this basic example, we just cycle through actions
            action_num = (action_count % 6) + 1  # ACTION1 through ACTION6
            action = getattr(GameAction, f"ACTION{action_num}")
            print(f"Action {action_count + 1}: ACTION{action_num}")

        # Execute the action
        env.step(action)
        action_count += 1

    # Get final scorecard
    print("\n" + "=" * 60)
    print("Final Scorecard")
    print("=" * 60)
    scorecard = arc.get_scorecard()
    print(scorecard)

    print("\nBasic agent complete!")


if __name__ == "__main__":
    main()
