#!/usr/bin/env python3
"""
Online learning agent for ARC-AGI-3.

The agent learns P(frame_change | state, action) online during gameplay.
It combines graph-based exploration (to avoid repeating dead actions)
with a CNN that learns which actions are productive.

Three action selection modes:
1. Graph-guided: try untested actions at current state
2. Navigation: follow graph to states with untested actions
3. CNN-guided: sample actions proportional to predicted change probability

Usage:
    uv run python -m src.aria_v3.agent --game ls20
    uv run python -m src.aria_v3.agent --game vc33 --max-actions 2000
"""

import argparse
import os
import sys
import time

import numpy as np

from .frame_processor import FrameProcessor
from .state_graph import StateGraph
from .change_predictor import OnlineLearner


class OnlineAgent:
    """Online learning agent that explores unknown games.

    The agent maintains:
    - A frame processor for region segmentation and hashing
    - A state graph for deduplication and dead-end pruning
    - A CNN that learns P(frame_change | state, action) online

    Action space mapping:
    - Simple actions: indices 0..N-1 map to game actions from available_simple
    - Click regions: indices N..N+R-1 map to clicking region centroids
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.frame_processor = FrameProcessor()
        self.state_graph = StateGraph()
        self.learner = OnlineLearner(device=device)

        # Game state
        self.available_simple: list[int] = []  # simple action IDs (1-5)
        self.has_click = False                  # whether action 6 is available
        self.prev_frame: np.ndarray | None = None
        self.prev_hash: str | None = None
        self.prev_action_idx: int | None = None
        self.prev_regions: list = []

        # Stats
        self.action_count = 0
        self.graph_guided_count = 0
        self.navigate_count = 0
        self.cnn_guided_count = 0
        self.frame_changes = 0

    def setup(self, available_actions: list[int]) -> None:
        """Configure agent for a specific game's action space.

        Args:
            available_actions: List of available action IDs (1-7) from the game
        """
        self.available_simple = [a for a in available_actions if 1 <= a <= 5]
        self.has_click = 6 in available_actions

    def act(self, frame: np.ndarray) -> tuple[int, int | None, int | None]:
        """Choose an action given the current frame.

        Args:
            frame: [64, 64] numpy array with values 0-15

        Returns:
            (action_type, x, y) where action_type is 1-indexed game API ID,
            and x,y are pixel coords for click actions (None for simple).
        """
        self.action_count += 1

        # Process frame
        regions = self.frame_processor.segment(frame)
        frame_hash = self.frame_processor.hash_frame(frame)

        # Build unified action space for this state
        num_simple = len(self.available_simple)
        num_regions = len(regions) if self.has_click else 0
        total_actions = num_simple + num_regions

        # Register state in graph
        self.state_graph.register_state(frame_hash, total_actions)

        # Update graph with result of previous action
        if self.prev_hash is not None and self.prev_action_idx is not None:
            frame_changed = not np.array_equal(self.prev_frame, frame)
            if frame_changed:
                self.frame_changes += 1
            self.state_graph.update(
                self.prev_hash, self.prev_action_idx, frame_hash, frame_changed
            )
            self.learner.add_experience(
                self.prev_frame, self.prev_action_idx, frame_changed
            )
            self.learner.maybe_train()

        # Select action
        action_idx = self._select_action(frame, frame_hash, regions, total_actions)

        # Convert unified index to game action
        action_type, x, y = self._index_to_game_action(action_idx, regions)

        # Save state for next step
        self.prev_frame = frame.copy()
        self.prev_hash = frame_hash
        self.prev_action_idx = action_idx
        self.prev_regions = regions

        return action_type, x, y

    def _select_action(
        self,
        frame: np.ndarray,
        frame_hash: str,
        regions: list,
        total_actions: int,
    ) -> int:
        """Select an action using the three-mode priority system.

        1. Graph: untested action at current state
        2. Navigate: follow graph to state with untested actions
        3. CNN: sample proportional to predicted change probability
        """
        # Mode 1: Try untested actions
        untested = self.state_graph.get_untested_action(frame_hash)
        if untested is not None:
            self.graph_guided_count += 1
            return untested

        # Mode 2: Navigate to frontier
        nav_action = self.state_graph.get_path_to_frontier(frame_hash)
        if nav_action is not None:
            self.navigate_count += 1
            return nav_action

        # Mode 3: CNN-guided sampling
        self.cnn_guided_count += 1
        return self.learner.sample_action(
            frame, regions, self.available_simple, self.has_click
        )

    def _index_to_game_action(
        self, action_idx: int, regions: list
    ) -> tuple[int, int | None, int | None]:
        """Convert unified action index to game API call.

        Args:
            action_idx: Index in unified space
            regions: Current frame regions

        Returns:
            (action_type, x, y) for the game API
        """
        num_simple = len(self.available_simple)

        if action_idx < num_simple:
            # Simple action
            action_type = self.available_simple[action_idx]
            return action_type, None, None
        else:
            # Click action on a region
            region_idx = action_idx - num_simple
            if region_idx < len(regions):
                region = regions[region_idx]
                x, y = self.frame_processor.get_click_point(region)
                return 6, x, y
            else:
                # Fallback: click center of frame
                return 6, 32, 32

    def on_level_complete(self) -> None:
        """Reset learning state for a new level."""
        self.learner.reset()
        self.state_graph.reset()
        self.prev_frame = None
        self.prev_hash = None
        self.prev_action_idx = None

    def get_stats(self) -> dict:
        graph_stats = self.state_graph.get_stats()
        learner_stats = self.learner.get_stats()
        return {
            "actions": self.action_count,
            "frame_changes": self.frame_changes,
            "graph_guided": self.graph_guided_count,
            "navigate": self.navigate_count,
            "cnn_guided": self.cnn_guided_count,
            **graph_stats,
            **learner_stats,
        }


def run_agent(
    game_id: str = "ls20",
    max_actions: int = 5000,
    verbose: bool = True,
    device: str = "cuda",
):
    """Run the online learning agent on an ARC-AGI-3 game."""
    from dotenv import load_dotenv
    load_dotenv()

    if not os.getenv("ARC_API_KEY"):
        print("Error: ARC_API_KEY not set")
        sys.exit(1)

    try:
        import arc_agi
        from arcengine import GameAction, GameState
    except ImportError as e:
        print(f"Error: {e}\nRun 'uv sync' to install dependencies")
        sys.exit(1)

    # Create agent
    agent = OnlineAgent(device=device)

    # Create game
    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing {game_id} with Online Learning Agent (v3)")
        print(f"{'='*60}")

    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    available = env.observation_space.available_actions
    agent.setup(list(available))

    # Build action map for game API
    action_map = {}
    for i in range(1, 8):
        if i in available:
            action_map[i] = getattr(GameAction, f"ACTION{i}", None)

    if verbose:
        print(f"Available actions: {list(available)}")
        print(f"Simple actions: {agent.available_simple}")
        print(f"Has click: {agent.has_click}")

    start_time = time.time()
    action_count = 0
    levels_completed = 0

    while action_count < max_actions:
        elapsed = time.time() - start_time
        if elapsed > 180:  # 3-minute time budget
            if verbose:
                print(f"\nTime budget exceeded ({elapsed:.0f}s)")
            break

        observation = env.observation_space

        if observation.state == GameState.WIN:
            if verbose:
                print(f"\nWon after {action_count} actions!")
            levels_completed = observation.levels_completed
            break
        elif observation.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            env.step(GameAction.RESET)
            action_count += 1
            continue

        # Check for level completion
        if observation.levels_completed > levels_completed:
            levels_completed = observation.levels_completed
            agent.on_level_complete()
            if verbose:
                print(f"  Level {levels_completed} completed! (action {action_count})")

        frame = np.array(observation.frame[0])
        action_type, x, y = agent.act(frame)

        # Execute action
        game_action = action_map.get(action_type)
        if game_action is None:
            game_action = GameAction.RESET
            if verbose:
                print(f"  Warning: unknown action type {action_type}, resetting")

        if x is not None and y is not None:
            env.step(game_action, data={"x": x, "y": y})
        else:
            env.step(game_action)

        if verbose and action_count % 100 == 0:
            stats = agent.get_stats()
            loc_str = f"({x},{y})" if x is not None else "null"
            ms_per_action = (elapsed * 1000 / max(action_count, 1))
            print(
                f"Step {action_count}: type={action_type} loc={loc_str} | "
                f"changes={stats['frame_changes']} | "
                f"graph={stats['graph_guided']} nav={stats['navigate']} "
                f"cnn={stats['cnn_guided']} | "
                f"nodes={stats['nodes']} dead={stats['dead']} | "
                f"{ms_per_action:.1f}ms/act"
            )

        action_count += 1

    duration = time.time() - start_time

    if verbose:
        stats = agent.get_stats()
        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"Actions: {action_count}")
        print(f"Levels completed: {levels_completed}")
        print(f"Time: {duration:.1f}s ({duration/max(action_count,1)*1000:.1f}ms/action)")
        print(f"Frame changes: {stats['frame_changes']}")
        print(f"Action modes: graph={stats['graph_guided']} "
              f"nav={stats['navigate']} cnn={stats['cnn_guided']}")
        print(f"State graph: {stats['nodes']} nodes, "
              f"{stats['dead']} dead edges, "
              f"{stats['untested']} untested")
        print(f"CNN: {stats['buffer_size']} experiences, "
              f"{stats['train_steps']} train steps")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(description="Run online learning agent (v3)")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=5000)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    run_agent(
        game_id=args.game,
        max_actions=args.max_actions,
        verbose=not args.quiet,
        device=args.device,
    )


if __name__ == "__main__":
    main()
