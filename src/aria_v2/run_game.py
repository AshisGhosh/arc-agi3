#!/usr/bin/env python3
"""
Run ARIA v2 agent on ARC-AGI-3 games.

Usage:
    uv run python -m src.aria_v2.run_game --game ls20
    uv run python -m src.aria_v2.run_game --game ls20 --policy learned --train-first
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GameResult:
    """Result from playing one game."""
    game_id: str
    policy_type: str
    actions_taken: int
    levels_completed: int
    player_found_at: int | None
    beliefs_summary: str
    duration_seconds: float


def run_aria_agent(
    game_id: str = "ls20",
    policy_type: str = "systematic",
    max_actions: int = 80,
    train_first: bool = False,
    training_epochs: int = 20,
    use_llm: bool = False,
    llm_provider: str = "auto",
    verbose: bool = True,
) -> GameResult:
    """
    Run ARIA v2 agent on an ARC-AGI-3 game.

    Args:
        game_id: Game to play (ls20, vc33, ft09, etc.)
        policy_type: "random", "systematic", or "learned"
        max_actions: Maximum actions before stopping
        train_first: If True and policy_type="learned", train the policy first
        training_epochs: Epochs for training if train_first=True
        verbose: Print progress

    Returns:
        GameResult with statistics
    """
    # Check API key
    if not os.getenv("ARC_API_KEY"):
        print("Error: ARC_API_KEY not set")
        print("Set it in .env or export ARC_API_KEY='your-key'")
        sys.exit(1)

    # Import arc_agi
    try:
        import arc_agi
        from arcengine import GameAction, GameState
    except ImportError as e:
        print(f"Error importing arc_agi: {e}")
        print("Run 'uv sync' to install dependencies")
        sys.exit(1)

    # Import ARIA v2
    from src.aria_v2.core import (
        ARIAAgent,
        AgentConfig,
        ExplorationTrainer,
        TrainingConfig,
    )

    # Train learned policy if requested
    trained_policy = None
    if policy_type == "learned" and train_first:
        if verbose:
            print("=" * 60)
            print("Training Learned Exploration Policy")
            print("=" * 60)

        config = TrainingConfig(num_epochs=training_epochs)
        trainer = ExplorationTrainer(config)
        trainer.train(num_epochs=training_epochs, episodes_per_epoch=100)
        trained_policy = trainer.get_trained_policy()

        if verbose:
            print()

    # Create ARIA agent
    agent_config = AgentConfig(
        exploration_type=policy_type,
        max_actions=max_actions,
        use_llm=use_llm,
        llm_provider=llm_provider,
    )
    agent = ARIAAgent(agent_config)

    # Swap in trained policy if we have one
    if trained_policy:
        agent.exploration_policy = trained_policy
        agent.action_selector.exploration_policy = trained_policy

    # Create game environment
    if verbose:
        print("=" * 60)
        print(f"Playing {game_id} with {policy_type} exploration")
        print("=" * 60)

    arc = arc_agi.Arcade()
    # Don't use terminal render mode - too much output
    env = arc.make(game_id, render_mode=None)

    # Map ARIA actions to GameActions
    # ARIA: 1=up, 2=down, 3=left, 4=right (movement)
    # GameAction: ACTION1-ACTION4 are typically movement
    # Get available actions from the game
    obs = env.observation_space
    available = obs.available_actions
    if verbose:
        print(f"Available actions: {available}")

    # Build action map based on available actions
    # Default: ARIA action 1-4 map to GameAction ACTION1-ACTION4
    action_map = {}
    for i in range(1, 7):
        if i in available:
            action_map[i] = getattr(GameAction, f"ACTION{i}")

    start_time = time.time()
    action_count = 0
    levels_completed = 0

    while action_count < max_actions:
        # Get observation
        observation = env.observation_space

        # Check game state
        if observation.state == GameState.WIN:
            if verbose:
                print(f"\n🎉 Won after {action_count} actions!")
            levels_completed = observation.levels_completed
            break
        elif observation.state == GameState.NOT_PLAYED:
            # Start game
            env.step(GameAction.RESET)
            action_count += 1
            continue
        elif observation.state == GameState.GAME_OVER:
            # Restart
            env.step(GameAction.RESET)
            action_count += 1
            continue

        # Get frame from observation
        # observation.frame is a list of 2D arrays (channels)
        frame = np.array(observation.frame[0])  # Use first channel

        # Track level completion
        level_completed = observation.levels_completed > levels_completed
        if level_completed:
            levels_completed = observation.levels_completed
            if verbose:
                print(f"  Level {levels_completed} completed!")

        # Get action from ARIA agent
        aria_action = agent.act(frame, level_completed=level_completed)

        # Map to GameAction
        game_action = action_map.get(aria_action, GameAction.ACTION1)

        # Get reasoning for verbose output
        decision = agent.get_last_decision()
        stats = agent.get_stats()
        if verbose and action_count % 10 == 0:
            player_status = "FOUND" if stats.player_found_at_action else "searching..."
            pos = agent.belief_state.player_position
            timer = agent.belief_state.timer
            timer_status = f"timer={timer.current_size}/{timer.initial_size}"
            if timer.is_critical:
                timer_status += " CRITICAL!"
            elif timer.is_urgent:
                timer_status += " urgent"
            print(f"Action {action_count}: aria_action={aria_action} | {decision.reasoning} | pos={pos} | {timer_status}")

        # Execute action
        env.step(game_action)
        action_count += 1

    duration = time.time() - start_time

    # Get final stats
    stats = agent.get_stats()
    beliefs = agent.get_belief_summary()

    if verbose:
        print("\n" + "=" * 60)
        print("Results")
        print("=" * 60)
        print(f"Actions taken: {action_count}")
        print(f"Levels completed: {levels_completed}")
        print(f"Player found at action: {stats.player_found_at_action}")
        print(f"Duration: {duration:.1f}s")
        print(f"\nDecision breakdown:")
        print(f"  Evidence-based: {stats.evidence_decisions}")
        print(f"  Rule-based: {stats.fast_decisions}")
        print(f"  Exploration: {stats.exploration_decisions}")
        print("\nBelief State:")
        print(beliefs)

        # Get scorecard
        print("\n" + "=" * 60)
        print("Scorecard")
        print("=" * 60)
        print(arc.get_scorecard())

    return GameResult(
        game_id=game_id,
        policy_type=policy_type,
        actions_taken=action_count,
        levels_completed=levels_completed,
        player_found_at=stats.player_found_at_action,
        beliefs_summary=beliefs,
        duration_seconds=duration,
    )


def run_comparison(
    game_id: str = "ls20",
    max_actions: int = 80,
    training_epochs: int = 20,
) -> dict:
    """
    Run all three policies and compare results.
    """
    print("=" * 60)
    print(f"A/B Test on Real Game: {game_id}")
    print("=" * 60)
    print()

    results = {}

    for policy in ["random", "systematic", "learned"]:
        print(f"\n{'='*60}")
        print(f"Testing: {policy}")
        print(f"{'='*60}")

        result = run_aria_agent(
            game_id=game_id,
            policy_type=policy,
            max_actions=max_actions,
            train_first=(policy == "learned"),
            training_epochs=training_epochs,
            verbose=True,
        )
        results[policy] = result

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Policy':<15} {'Actions':<10} {'Levels':<10} {'Player Found':<15}")
    print("-" * 50)
    for policy, result in results.items():
        player_found = result.player_found_at if result.player_found_at else "N/A"
        print(f"{policy:<15} {result.actions_taken:<10} {result.levels_completed:<10} {player_found:<15}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run ARIA v2 agent on ARC-AGI-3 games"
    )
    parser.add_argument(
        "--game", "-g",
        default="ls20",
        help="Game ID to play (default: ls20)"
    )
    parser.add_argument(
        "--policy", "-p",
        choices=["random", "systematic", "learned"],
        default="systematic",
        help="Exploration policy (default: systematic)"
    )
    parser.add_argument(
        "--max-actions", "-m",
        type=int,
        default=80,
        help="Maximum actions (default: 80)"
    )
    parser.add_argument(
        "--train-first",
        action="store_true",
        help="Train learned policy before playing"
    )
    parser.add_argument(
        "--training-epochs",
        type=int,
        default=20,
        help="Training epochs if --train-first (default: 20)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all three policies and compare"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM advisor for strategic guidance"
    )
    parser.add_argument(
        "--llm-provider",
        choices=["auto", "anthropic", "openai", "ollama", "heuristic"],
        default="auto",
        help="LLM provider to use (default: auto)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Reduce output verbosity"
    )

    args = parser.parse_args()

    if args.compare:
        run_comparison(
            game_id=args.game,
            max_actions=args.max_actions,
            training_epochs=args.training_epochs,
        )
    else:
        run_aria_agent(
            game_id=args.game,
            policy_type=args.policy,
            max_actions=args.max_actions,
            train_first=args.train_first,
            training_epochs=args.training_epochs,
            use_llm=args.llm,
            llm_provider=args.llm_provider,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
