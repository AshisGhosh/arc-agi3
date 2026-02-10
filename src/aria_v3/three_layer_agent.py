#!/usr/bin/env python3
"""
Three-Layer Agent for ARC-AGI-3.

Layer 1 (Reactive): State graph + plan execution. <2ms/action.
Layer 2 (Learning): Frame diffs + statistics. ~20ms every 20 actions.
Layer 3 (Reasoning): LLM strategy oracle. ~100-200ms every 200 actions.

Usage:
    uv run python -m src.aria_v3.three_layer_agent --game ls20
    uv run python -m src.aria_v3.three_layer_agent --game vc33 --max-actions 5000
    uv run python -m src.aria_v3.three_layer_agent --game vc33 --no-llm  # skip Layer 3
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np

from .frame_processor import FrameProcessor, Region
from .state_graph import StateGraph
from .learning_engine import LearningEngine
from .reasoning_oracle import ReasoningOracle, Strategy


class ThreeLayerAgent:
    """Three-layer agent: Reactive + Learning + Reasoning."""

    def __init__(
        self,
        device: str = "cuda",
        use_llm: bool = True,
        llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
        first_oracle_step: int = 50,
        oracle_interval: int = 200,
        report_interval: int = 20,
    ):
        # Layer 1 components
        self.frame_processor = FrameProcessor()
        self.state_graph = StateGraph()

        # Layer 2
        self.learning = LearningEngine()

        # Layer 3
        self.use_llm = use_llm
        if use_llm:
            self.oracle = ReasoningOracle(
                model_name=llm_model,
                device=device,
                max_new_tokens=300,
                load_in_4bit=True,
            )
        else:
            self.oracle = None

        # Schedule
        self.first_oracle_step = first_oracle_step
        self.oracle_interval = oracle_interval
        self.report_interval = report_interval

        # Game state
        self.available_simple: list[int] = []
        self.has_click: bool = False
        self.prev_frame: np.ndarray | None = None
        self.prev_hash: str | None = None
        self.prev_action_idx: int | None = None

        # Strategy from Layer 3
        self.strategy: Strategy | None = None
        self.movement_map: dict[int, tuple[int, int]] | None = None
        self.plan_queue: deque[tuple[int, int | None, int | None]] = deque()
        self.target_positions: list[tuple[int, int]] = []
        self.player_color: int | None = None

        # Stats
        self.step_count = 0
        self.levels_completed = 0
        self.plan_executions = 0
        self.graph_actions = 0
        self.navigate_actions = 0
        self.random_actions = 0
        self.oracle_calls = 0
        self.last_report_text: str = ""

    def setup(self, available_actions: list[int]) -> None:
        """Configure for a game's action space."""
        self.available_simple = [a for a in available_actions if 1 <= a <= 5]
        self.has_click = 6 in available_actions
        self.learning.setup(list(available_actions))

    def act(self, frame: np.ndarray) -> tuple[int, int | None, int | None]:
        """Choose an action. Returns (action_type, x, y)."""
        self.step_count += 1

        # Process frame
        regions = self.frame_processor.segment(frame)
        frame_hash = self.frame_processor.hash_frame(frame)

        # Build unified action space
        num_simple = len(self.available_simple)
        num_regions = len(regions) if self.has_click else 0
        total_actions = num_simple + num_regions

        # Register state
        self.state_graph.register_state(frame_hash, total_actions)

        # Update graph and Layer 2 with previous transition
        if self.prev_hash is not None and self.prev_action_idx is not None:
            frame_changed = not np.array_equal(self.prev_frame, frame)
            next_hash = frame_hash

            # Detect game over (returned to level start)
            game_over = (
                next_hash == self.state_graph.level_start_hash
                and self.prev_hash != self.state_graph.level_start_hash
                and not frame_changed
            )

            self.state_graph.update(
                self.prev_hash, self.prev_action_idx, next_hash, frame_changed
            )

            # Layer 2: record transition
            prev_action_type = self._idx_to_action_type(self.prev_action_idx)
            self.learning.record(
                self.prev_frame,
                prev_action_type,
                frame,
                self.prev_hash,
                next_hash,
                game_over=game_over,
            )

        # Layer 2: generate report periodically
        if self.step_count % self.report_interval == 0:
            report = self.learning.generate_report()
            self.last_report_text = self.learning.report_to_text(report)

        # Layer 3: call oracle periodically
        if self.use_llm and self.last_report_text:
            should_call = (
                self.step_count == self.first_oracle_step
                or (
                    self.step_count > self.first_oracle_step
                    and self.step_count % self.oracle_interval == 0
                )
            )
            if should_call:
                self._call_oracle()

        # Layer 1: select action
        action_type, x, y = self._select_action(frame, frame_hash, regions, total_actions)

        # Save state
        self.prev_frame = frame.copy()
        self.prev_hash = frame_hash
        self.prev_action_idx = self._action_to_idx(action_type, x, y, regions)

        return action_type, x, y

    def _call_oracle(self) -> None:
        """Call Layer 3 to get a strategy update."""
        self.oracle_calls += 1
        try:
            self.strategy = self.oracle.analyze(self.last_report_text)
        except Exception as e:
            print(f"  Oracle error: {e}")
            return

        # Apply strategy — only accept movement_map if game has simple actions
        if self.strategy.movement_map and self.available_simple:
            # Validate: only accept actions that are actually available
            valid_map = {}
            for action_id, displacement in self.strategy.movement_map.items():
                if action_id in self.available_simple:
                    valid_map[action_id] = displacement
            if valid_map:
                self.movement_map = valid_map

        if self.strategy.player_color is not None:
            self.player_color = self.strategy.player_color

        # Build click target queue if strategy says to click
        if self.strategy.click_targets:
            self.plan_queue.clear()
            for x, y in self.strategy.click_targets:
                self.plan_queue.append((6, x, y))

        # Build pathfinding targets
        if self.strategy.targets:
            self._build_target_positions()

    def _build_target_positions(self) -> None:
        """Find positions of target-colored objects in the current frame."""
        if self.prev_frame is None or not self.strategy or not self.strategy.targets:
            return

        self.target_positions = []
        regions = self.frame_processor.segment(self.prev_frame)
        for target in self.strategy.targets:
            target_color = target.get("color")
            if target_color is None:
                continue
            for region in regions:
                if region.color == target_color:
                    x, y = self.frame_processor.get_click_point(region)
                    self.target_positions.append((x, y))

    def _select_action(
        self,
        frame: np.ndarray,
        frame_hash: str,
        regions: list[Region],
        total_actions: int,
    ) -> tuple[int, int | None, int | None]:
        """Layer 1: Select action using priority modes.

        1. Execute plan (from Layer 3)
        2. Pathfind to targets (using movement_map from Layer 3)
        3. Graph exploration (untested actions, frontier)
        4. Random fallback
        """
        # Mode 1: Execute plan queue
        if self.plan_queue:
            action_type, x, y = self.plan_queue.popleft()
            self.plan_executions += 1
            return action_type, x, y

        # Mode 2: Pathfind to nearest target using movement_map
        if self.movement_map and self.target_positions and self.player_color is not None:
            action = self._pathfind_step(frame, regions)
            if action is not None:
                self.plan_executions += 1
                return action

        # Mode 3: Graph exploration — untested actions
        untested = self.state_graph.get_untested_action(frame_hash)
        if untested is not None:
            self.graph_actions += 1
            return self._idx_to_game_action(untested, regions)

        # Mode 4: Navigate to frontier
        nav = self.state_graph.get_path_to_frontier(frame_hash)
        if nav is not None:
            self.navigate_actions += 1
            return self._idx_to_game_action(nav, regions)

        # Mode 5: Random
        self.random_actions += 1
        return self._random_action(regions)

    def _pathfind_step(
        self,
        frame: np.ndarray,
        regions: list[Region],
    ) -> tuple[int, int | None, int | None] | None:
        """Use movement_map to step toward nearest target.

        Returns (action_type, x, y) or None if can't pathfind.
        """
        if not self.movement_map or not self.target_positions:
            return None

        # Find player position
        player_pos = self._find_player(frame, regions)
        if player_pos is None:
            return None

        px, py = player_pos

        # Find nearest target
        nearest = None
        nearest_dist = float("inf")
        for tx, ty in self.target_positions:
            dist = abs(tx - px) + abs(ty - py)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest = (tx, ty)

        if nearest is None or nearest_dist < 2:
            # Already at target or no target
            return None

        tx, ty = nearest

        # Pick best action from movement_map
        best_action = None
        best_reduction = -float("inf")

        for action_id, (dx, dy) in self.movement_map.items():
            new_x = px + dx
            new_y = py + dy
            new_dist = abs(tx - new_x) + abs(ty - new_y)
            reduction = nearest_dist - new_dist
            if reduction > best_reduction:
                best_reduction = reduction
                best_action = action_id

        if best_action is not None and best_reduction > 0:
            return best_action, None, None

        return None

    def _find_player(
        self, frame: np.ndarray, regions: list[Region]
    ) -> tuple[int, int] | None:
        """Find the player's position using player_color from strategy."""
        if self.player_color is None:
            return None

        for region in regions:
            if region.color == self.player_color:
                x, y = self.frame_processor.get_click_point(region)
                return (x, y)
        return None

    def _idx_to_game_action(
        self, action_idx: int, regions: list[Region]
    ) -> tuple[int, int | None, int | None]:
        """Convert unified index to (action_type, x, y)."""
        num_simple = len(self.available_simple)
        if action_idx < num_simple:
            return self.available_simple[action_idx], None, None
        else:
            region_idx = action_idx - num_simple
            if region_idx < len(regions):
                x, y = self.frame_processor.get_click_point(regions[region_idx])
                return 6, x, y
            else:
                return 6, 32, 32

    def _idx_to_action_type(self, action_idx: int) -> int:
        """Convert unified index to action type ID (for Layer 2)."""
        num_simple = len(self.available_simple)
        if action_idx < num_simple:
            return self.available_simple[action_idx]
        return 6

    def _action_to_idx(
        self,
        action_type: int,
        x: int | None,
        y: int | None,
        regions: list[Region],
    ) -> int:
        """Convert game action back to unified index."""
        if action_type != 6:
            try:
                return self.available_simple.index(action_type)
            except ValueError:
                return 0
        # Click action — find closest region
        if x is not None and y is not None:
            num_simple = len(self.available_simple)
            best_idx = num_simple  # default to first region
            best_dist = float("inf")
            for i, region in enumerate(regions):
                rx, ry = self.frame_processor.get_click_point(region)
                dist = abs(rx - x) + abs(ry - y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = num_simple + i
            return best_idx
        return len(self.available_simple)

    def _random_action(
        self, regions: list[Region]
    ) -> tuple[int, int | None, int | None]:
        """Pick a random action."""
        if self.has_click and regions and np.random.random() < 0.3:
            region = regions[np.random.randint(len(regions))]
            x, y = self.frame_processor.get_click_point(region)
            return 6, x, y
        if self.available_simple:
            action = self.available_simple[np.random.randint(len(self.available_simple))]
            return action, None, None
        return 1, None, None

    def on_level_complete(self) -> None:
        """Handle level completion."""
        self.levels_completed += 1
        # Layer 2: reset per-level stats (keep notable events)
        self.learning.on_level_complete()
        # Layer 1: reset graph (new layout) but keep movement_map and strategy
        self.state_graph.reset()
        self.prev_frame = None
        self.prev_hash = None
        self.prev_action_idx = None
        # Clear plan queue — old targets are invalid
        self.plan_queue.clear()
        self.target_positions = []

    def get_stats(self) -> dict:
        """Get combined stats from all layers."""
        graph_stats = self.state_graph.get_stats()
        oracle_stats = self.oracle.get_stats() if self.oracle else {"calls": 0}
        return {
            "step": self.step_count,
            "levels": self.levels_completed,
            "plan_exec": self.plan_executions,
            "graph": self.graph_actions,
            "navigate": self.navigate_actions,
            "random": self.random_actions,
            "oracle_calls": self.oracle_calls,
            "strategy_conf": self.strategy.confidence if self.strategy else 0.0,
            "game_type": self.strategy.game_type if self.strategy else "unknown",
            "plan": self.strategy.plan if self.strategy else "none",
            **{f"graph_{k}": v for k, v in graph_stats.items()},
            **{f"oracle_{k}": v for k, v in oracle_stats.items()},
        }


def run_agent(
    game_id: str = "ls20",
    max_actions: int = 5000,
    verbose: bool = True,
    device: str = "cuda",
    use_llm: bool = True,
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """Run the three-layer agent on an ARC-AGI-3 game."""
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

    agent = ThreeLayerAgent(
        device=device,
        use_llm=use_llm,
        llm_model=llm_model,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Three-Layer Agent v3.1 — {game_id}")
        print(f"{'='*60}")
        print(f"LLM: {llm_model if use_llm else 'DISABLED'}")

    # Preload LLM during setup (overlaps with game init)
    if use_llm:
        print("Preloading LLM...")
        agent.oracle.load()

    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    available = env.observation_space.available_actions
    agent.setup(list(available))

    action_map = {}
    for i in range(1, 8):
        if i in available:
            action_map[i] = getattr(GameAction, f"ACTION{i}", None)

    if verbose:
        print(f"Available actions: {list(available)}")
        print(f"Simple: {agent.available_simple}, Click: {agent.has_click}")
        print()

    start_time = time.time()
    action_count = 0
    levels_completed = 0
    prev_oracle_calls = 0

    while action_count < max_actions:
        elapsed = time.time() - start_time
        if elapsed > 180:
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
                print(f"  *** Level {levels_completed} completed! (action {action_count}) ***")

        frame = np.array(observation.frame[0])

        step_start = time.time()
        action_type, x, y = agent.act(frame)
        step_ms = (time.time() - step_start) * 1000

        # Execute
        game_action = action_map.get(action_type)
        if game_action is None:
            game_action = GameAction.RESET

        if x is not None and y is not None:
            env.step(game_action, data={"x": x, "y": y})
        else:
            env.step(game_action)

        if verbose and action_count % 100 == 0:
            stats = agent.get_stats()
            loc_str = f"({x},{y})" if x is not None else "null"
            avg_ms = elapsed * 1000 / max(action_count, 1)
            print(
                f"Step {action_count}: a={action_type} {loc_str} "
                f"({step_ms:.0f}ms) | "
                f"plan={stats['plan_exec']} graph={stats['graph']} "
                f"nav={stats['navigate']} rnd={stats['random']} | "
                f"nodes={stats['graph_nodes']} | "
                f"type={stats['game_type']} conf={stats['strategy_conf']:.1f} | "
                f"avg={avg_ms:.1f}ms/act"
            )

        # Print when oracle was just called
        if verbose and agent.oracle_calls > prev_oracle_calls and agent.strategy:
            s = agent.strategy
            print(
                f"  >>> Oracle #{agent.oracle_calls}: "
                f"type={s.game_type} plan={s.plan} "
                f"player={s.player_color} conf={s.confidence:.1f} "
                f"({s.generation_time_ms:.0f}ms)"
            )
            if s.movement_map:
                print(f"      movement_map={s.movement_map}")
            if s.targets:
                print(f"      targets={s.targets}")
            prev_oracle_calls = agent.oracle_calls

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
        print(f"Action modes: plan={stats['plan_exec']} graph={stats['graph']} "
              f"nav={stats['navigate']} rnd={stats['random']}")
        print(f"Graph: {stats['graph_nodes']} nodes, {stats['graph_dead']} dead")
        print(f"Oracle: {stats['oracle_calls']} calls")
        if agent.strategy:
            s = agent.strategy
            print(f"Strategy: type={s.game_type} plan={s.plan} "
                  f"player={s.player_color} conf={s.confidence:.1f}")
            if s.movement_map:
                print(f"  movement_map={s.movement_map}")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(description="Three-layer agent (v3.1)")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=5000)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-llm", action="store_true", help="Disable Layer 3 (LLM)")
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    run_agent(
        game_id=args.game,
        max_actions=args.max_actions,
        verbose=not args.quiet,
        device=args.device,
        use_llm=not args.no_llm,
        llm_model=args.llm_model,
    )


if __name__ == "__main__":
    main()
