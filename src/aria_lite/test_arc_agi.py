"""
Test ARIA-Lite on ARC-AGI-3 games.

Validates the agent's ability to play real games:
- ls20: Light Switch game
- vc33: Vector Crossing game
- ft09: Figure Transformation game

Usage:
    python -m src.aria_lite.test_arc_agi --game ls20
    python -m src.aria_lite.test_arc_agi --all
"""

import argparse
import logging
import sys

# Check if required packages are available
try:
    from arc_agi import Arcade, OperationMode
    from arcengine import FrameData, GameState

    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False

from .arc_agent import ARCAgentConfig, ARCAGIAgent, create_arc_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def run_game(
    game_id: str,
    agent: ARCAGIAgent,
    max_actions: int = 80,
    render: bool = False,
) -> dict:
    """
    Run a single game with the agent.

    Args:
        game_id: ARC-AGI-3 game ID (e.g., 'ls20')
        agent: Configured ARCAGIAgent
        max_actions: Maximum actions before stopping
        render: Whether to render frames

    Returns:
        Results dict with metrics
    """
    if not ARC_AGI_AVAILABLE:
        logger.error("arc_agi package not available")
        return {"error": "arc_agi not installed"}

    # Create arcade environment
    arcade = Arcade(operation_mode=OperationMode.OFFLINE)

    # Create environment wrapper
    render_mode = "terminal" if render else None
    env = arcade.make(game_id, render_mode=render_mode)

    if env is None:
        logger.error(f"Failed to create environment for {game_id}")
        return {"error": f"Game {game_id} not found"}

    logger.info(f"Starting game: {game_id}")

    # Reset environment
    raw_frame = env.reset()
    if raw_frame is None:
        return {"error": "Failed to reset environment"}

    # Convert raw frame to FrameData
    frame = FrameData(
        game_id=raw_frame.game_id,
        frame=[arr.tolist() for arr in raw_frame.frame],
        state=raw_frame.state,
        levels_completed=raw_frame.levels_completed,
        win_levels=raw_frame.win_levels,
        guid=raw_frame.guid,
        full_reset=raw_frame.full_reset,
        available_actions=raw_frame.available_actions,
    )

    frames = [frame]
    action_count = 0

    # Game loop
    while action_count < max_actions:
        # Check if done
        if agent.is_done(frames, frame):
            logger.info(f"Agent signaled done at step {action_count}")
            break

        # Choose action
        action = agent.choose_action(frames, frame)

        # Execute action
        if action.is_complex():
            # Complex actions need data (x, y coordinates)
            action.set_data({
                "x": 32,  # Center of 64x64 frame
                "y": 32,
            })

        raw_frame = env.step(action)
        if raw_frame is None:
            logger.warning(f"Received None frame at step {action_count}")
            break

        # Convert to FrameData
        frame = FrameData(
            game_id=raw_frame.game_id,
            frame=[arr.tolist() for arr in raw_frame.frame],
            state=raw_frame.state,
            levels_completed=raw_frame.levels_completed,
            win_levels=raw_frame.win_levels,
            guid=raw_frame.guid,
            full_reset=raw_frame.full_reset,
            available_actions=raw_frame.available_actions,
        )
        frames.append(frame)

        action_count += 1

        # Log progress
        if action_count % 10 == 0:
            logger.info(
                f"Step {action_count}: state={frame.state.name}, "
                f"levels={frame.levels_completed}"
            )

        # Check for win
        if frame.state == GameState.WIN:
            logger.info(f"Game WON at step {action_count}!")
            break

    # Compute results
    final_state = frames[-1].state if frames else GameState.NOT_PLAYED
    levels_completed = frames[-1].levels_completed if frames else 0

    results = {
        "game_id": game_id,
        "actions": action_count,
        "final_state": final_state.name,
        "levels_completed": levels_completed,
        "won": final_state == GameState.WIN,
        "total_frames": len(frames),
    }

    logger.info(f"Game finished: {results}")

    # Close scorecard
    arcade.close_scorecard()

    return results


def evaluate_agent(
    agent: ARCAGIAgent,
    games: list[str],
    max_actions: int = 80,
    render: bool = False,
) -> dict:
    """
    Evaluate agent on multiple games.

    Args:
        agent: Configured ARCAGIAgent
        games: List of game IDs
        max_actions: Maximum actions per game
        render: Whether to render

    Returns:
        Evaluation metrics
    """
    results = []

    for game_id in games:
        logger.info(f"\n{'='*60}\nEvaluating on {game_id}\n{'='*60}")

        agent.reset()
        game_result = run_game(game_id, agent, max_actions, render)
        results.append(game_result)

    # Aggregate metrics
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {"error": "No valid results", "results": results}

    wins = sum(1 for r in valid_results if r["won"])
    total_levels = sum(r["levels_completed"] for r in valid_results)
    avg_actions = sum(r["actions"] for r in valid_results) / len(valid_results)

    summary = {
        "games_played": len(valid_results),
        "wins": wins,
        "win_rate": wins / len(valid_results) if valid_results else 0,
        "total_levels_completed": total_levels,
        "avg_actions_per_game": avg_actions,
        "results": results,
    }

    logger.info(f"\n{'='*60}\nEVALUATION SUMMARY\n{'='*60}")
    logger.info(f"Games: {summary['games_played']}")
    logger.info(f"Wins: {summary['wins']} ({summary['win_rate']:.1%})")
    logger.info(f"Total levels: {summary['total_levels_completed']}")
    logger.info(f"Avg actions: {summary['avg_actions_per_game']:.1f}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Test ARIA-Lite on ARC-AGI-3 games"
    )
    parser.add_argument(
        "--game", "-g",
        type=str,
        default=None,
        help="Specific game ID to test (e.g., 'ls20')"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Test on all available games (ls20, vc33, ft09)"
    )
    parser.add_argument(
        "--max-actions", "-m",
        type=int,
        default=80,
        help="Maximum actions per game"
    )
    parser.add_argument(
        "--render", "-r",
        action="store_true",
        help="Render game frames"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model checkpoint"
    )

    args = parser.parse_args()

    if not ARC_AGI_AVAILABLE:
        logger.error("Required packages not available!")
        logger.error("Install with: pip install arc-agi arcengine")
        sys.exit(1)

    # Determine games to test
    if args.all:
        games = ["ls20", "vc33", "ft09"]
    elif args.game:
        games = [args.game]
    else:
        games = ["ls20"]  # Default to light switch game

    # Create agent
    config = ARCAgentConfig()
    agent = create_arc_agent(config, meta_model_path=args.model_path)

    logger.info(f"Testing on games: {games}")
    logger.info(f"Max actions: {args.max_actions}")
    logger.info(f"Agent type: {'meta-learning' if agent.meta_agent else 'ARIA-Lite'}")

    # Run evaluation
    summary = evaluate_agent(
        agent,
        games,
        max_actions=args.max_actions,
        render=args.render,
    )

    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    for result in summary.get("results", []):
        status = "✓ WIN" if result.get("won") else "✗ LOSS"
        print(f"  {result.get('game_id', 'unknown')}: {status} "
              f"(levels={result.get('levels_completed', 0)}, "
              f"actions={result.get('actions', 0)})")

    return summary


if __name__ == "__main__":
    main()
