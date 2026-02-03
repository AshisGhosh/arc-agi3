#!/usr/bin/env python3
"""
ARC-AGI-3 Agent Runner

This is the main entry point for running ARC-AGI-3 agents.
For more sophisticated agent management, see the agents-reference/ directory.

Usage:
    uv run python main.py                    # Run basic agent on ls20
    uv run python main.py --game vc33        # Run on different game
    uv run python main.py --help             # Show all options
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_api_key():
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
        print("Or export it directly:")
        print("   export ARC_API_KEY='your-api-key-here'")
        print()
        return False
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ARC-AGI-3 Agent Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    uv run python main.py                    Run basic agent on ls20
    uv run python main.py --game vc33        Run on vc33 game
    uv run python main.py --list             List available games

For more agents, see:
    examples/basic_agent.py    - Simple sequential agent
    examples/random_agent.py   - Random action exploration
    agents-reference/          - Official agent framework
        """
    )
    parser.add_argument(
        "--game", "-g",
        default="ls20",
        help="Game ID to play (default: ls20)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available games and exit"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check environment setup and exit"
    )

    args = parser.parse_args()

    # Check API key
    if not check_api_key():
        if not args.check:
            sys.exit(1)

    if args.check:
        print("✓ API key configured")
        print("✓ Environment ready")
        print()
        print("Run 'uv run python main.py --game ls20' to start playing!")
        return

    # Import here to avoid import errors if arc-agi not installed
    try:
        import arc_agi
        from arcengine import GameAction, GameState
    except ImportError as e:
        print(f"Error: Could not import arc-agi: {e}")
        print("Run 'uv sync' to install dependencies")
        sys.exit(1)

    # Create arcade
    arc = arc_agi.Arcade()

    if args.list:
        print("Fetching available games...")
        # Note: The API will return available games
        # For now, list known games
        print("\nKnown games:")
        print("  ls20 - Locksmith (conditional interactions)")
        print("  vc33 - Budget puzzle logic")
        print("  ft09 - Pattern matching")
        print("\nFor full list, visit: https://three.arcprize.org/")
        return

    # Run simple agent
    print(f"Starting game: {args.game}")
    print("Use Ctrl+C to stop")
    print("-" * 40)

    env = arc.make(args.game, render_mode="terminal")

    action_count = 0
    max_actions = 50

    while action_count < max_actions:
        observation = env.observation_space

        if observation.state == GameState.WIN:
            print(f"\n🎉 Won after {action_count} actions!")
            break
        elif observation.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            # Simple cycling through actions
            action_num = (action_count % 6) + 1
            action = getattr(GameAction, f"ACTION{action_num}")

        env.step(action)
        action_count += 1

    print("\n" + "-" * 40)
    print("Scorecard:")
    print(arc.get_scorecard())


if __name__ == "__main__":
    main()
