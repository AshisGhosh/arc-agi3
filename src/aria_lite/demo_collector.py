"""
Demo Collection for ARC-AGI-3 Games.

Collects (observation, action) pairs from game episodes for meta-learning.
Supports both expert policies and random exploration.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

try:
    from arc_agi import Arcade, OperationMode
    from arcengine import FrameData, GameAction, GameState

    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False


@dataclass
class GameDemo:
    """A single demonstration episode."""

    game_id: str
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    levels_completed: int = 0
    won: bool = False
    total_steps: int = 0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "game_id": self.game_id,
            "observations": [obs.tolist() for obs in self.observations],
            "actions": self.actions,
            "rewards": self.rewards,
            "levels_completed": self.levels_completed,
            "won": self.won,
            "total_steps": self.total_steps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GameDemo":
        """Load from dict."""
        return cls(
            game_id=data["game_id"],
            observations=[np.array(obs) for obs in data["observations"]],
            actions=data["actions"],
            rewards=data["rewards"],
            levels_completed=data["levels_completed"],
            won=data["won"],
            total_steps=data["total_steps"],
        )


@dataclass
class DemoDataset:
    """Collection of demonstrations for a game."""

    game_id: str
    demos: list[GameDemo] = field(default_factory=list)

    def add(self, demo: GameDemo) -> None:
        """Add a demonstration."""
        self.demos.append(demo)

    def save(self, path: str) -> None:
        """Save demos to JSON file."""
        data = {
            "game_id": self.game_id,
            "num_demos": len(self.demos),
            "demos": [d.to_dict() for d in self.demos],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "DemoDataset":
        """Load demos from JSON file."""
        with open(path) as f:
            data = json.load(f)
        dataset = cls(game_id=data["game_id"])
        dataset.demos = [GameDemo.from_dict(d) for d in data["demos"]]
        return dataset

    def get_successful_demos(self) -> list[GameDemo]:
        """Get demos where agent made progress."""
        return [d for d in self.demos if d.levels_completed > 0 or d.won]

    def to_meta_format(
        self,
        max_demos: int = 5,
        max_steps_per_demo: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert to format for MetaLearningAgent.

        Returns:
            demo_obs: [K, T, H, W] observation tensors
            demo_actions: [K, T] action indices
        """
        # Get best demos (most progress)
        sorted_demos = sorted(
            self.demos,
            key=lambda d: (d.won, d.levels_completed, -d.total_steps),
            reverse=True,
        )

        selected = sorted_demos[:max_demos]

        all_obs = []
        all_actions = []

        for demo in selected:
            # Truncate to max steps
            obs = demo.observations[:max_steps_per_demo]
            actions = demo.actions[:max_steps_per_demo]

            # Pad if needed
            while len(obs) < max_steps_per_demo:
                obs.append(obs[-1] if obs else np.zeros((64, 64), dtype=np.uint8))
                actions.append(0)  # NOOP

            all_obs.append(np.stack(obs[:max_steps_per_demo]))
            all_actions.append(actions[:max_steps_per_demo])

        # Pad demos if needed
        while len(all_obs) < max_demos:
            all_obs.append(np.zeros((max_steps_per_demo, 64, 64), dtype=np.uint8))
            all_actions.append([0] * max_steps_per_demo)

        demo_obs = torch.from_numpy(np.stack(all_obs[:max_demos]))
        demo_actions = torch.tensor(all_actions[:max_demos])

        return demo_obs, demo_actions


class DemoCollector:
    """Collects demonstrations from ARC-AGI-3 games."""

    def __init__(
        self,
        game_id: str,
        policy: Optional[Callable[[np.ndarray, list[int]], int]] = None,
        max_steps: int = 100,
        render: bool = False,
    ):
        """
        Initialize collector.

        Args:
            game_id: ARC-AGI-3 game ID (e.g., 'ls20')
            policy: Function (observation, available_actions) -> action_id
                   If None, uses random policy
            max_steps: Maximum steps per episode
            render: Whether to render during collection
        """
        self.game_id = game_id
        self.policy = policy or self._random_policy
        self.max_steps = max_steps
        self.render = render

    def _random_policy(
        self, observation: np.ndarray, available_actions: list[int]
    ) -> int:
        """Random action selection."""
        if not available_actions:
            return 0
        return random.choice(available_actions)

    def collect_episode(self) -> Optional[GameDemo]:
        """Collect a single demonstration episode."""
        if not ARC_AGI_AVAILABLE:
            print("arc_agi package not available")
            return None

        arcade = Arcade(operation_mode=OperationMode.OFFLINE)
        render_mode = "terminal" if self.render else None
        env = arcade.make(self.game_id, render_mode=render_mode)

        if env is None:
            print(f"Failed to create environment for {self.game_id}")
            return None

        demo = GameDemo(game_id=self.game_id)

        # Reset
        raw_frame = env.reset()
        if raw_frame is None:
            return None

        # Game loop
        step = 0
        prev_levels = 0

        while step < self.max_steps:
            # Get observation
            obs = np.array(raw_frame.frame[0]) if raw_frame.frame else np.zeros((64, 64))
            demo.observations.append(obs)

            # Check game state
            if raw_frame.state == GameState.WIN:
                demo.won = True
                break
            if raw_frame.state == GameState.GAME_OVER:
                # Reset and continue
                raw_frame = env.reset()
                if raw_frame is None:
                    break
                continue
            if raw_frame.state == GameState.NOT_PLAYED:
                # Need to start game
                action = GameAction.RESET
            else:
                # Get action from policy
                available = raw_frame.available_actions or [1, 2, 3, 4, 5, 6]
                action_id = self.policy(obs, available)
                action = GameAction.from_id(action_id)

                # Handle complex actions (need x,y)
                if action.is_complex():
                    # Use center of observation or random position
                    action.set_data({"x": 32, "y": 32})

            demo.actions.append(action.id if hasattr(action, "id") else 0)

            # Step environment
            raw_frame = env.step(action)
            if raw_frame is None:
                break

            # Track reward (level completion)
            curr_levels = raw_frame.levels_completed
            reward = 1.0 if curr_levels > prev_levels else 0.0
            demo.rewards.append(reward)
            prev_levels = curr_levels

            step += 1

        demo.levels_completed = prev_levels
        demo.total_steps = step

        arcade.close_scorecard()

        return demo

    def collect_dataset(
        self,
        num_episodes: int = 100,
        save_path: Optional[str] = None,
    ) -> DemoDataset:
        """
        Collect multiple demonstration episodes.

        Args:
            num_episodes: Number of episodes to collect
            save_path: Optional path to save dataset

        Returns:
            DemoDataset with collected demos
        """
        dataset = DemoDataset(game_id=self.game_id)

        for i in range(num_episodes):
            demo = self.collect_episode()
            if demo is not None:
                dataset.add(demo)

                if (i + 1) % 10 == 0:
                    successful = len(dataset.get_successful_demos())
                    print(
                        f"Collected {i + 1}/{num_episodes} episodes, "
                        f"{successful} successful"
                    )

        if save_path:
            dataset.save(save_path)
            print(f"Saved {len(dataset.demos)} demos to {save_path}")

        return dataset


# ============================================================================
# Game-specific Expert Policies
# ============================================================================


def ls20_expert(observation: np.ndarray, available_actions: list[int]) -> int:
    """
    Simple heuristic expert for ls20 (navigation game).

    Strategy: Navigate towards bright pixels (likely targets).
    """
    # Find brightest region (likely goal)
    h, w = observation.shape[:2] if len(observation.shape) >= 2 else (64, 64)

    # Compute center of mass of bright pixels
    if len(observation.shape) == 2:
        bright_mask = observation > 100
    else:
        bright_mask = observation.max(axis=-1) > 100

    if bright_mask.sum() == 0:
        # No bright pixels, move randomly
        nav_actions = [a for a in available_actions if a in [1, 2, 3, 4]]
        return random.choice(nav_actions) if nav_actions else 1

    y_coords, x_coords = np.where(bright_mask)
    target_y = y_coords.mean()
    target_x = x_coords.mean()

    # Current position (assume center or sprite position)
    # For simplicity, assume agent is at center
    curr_y, curr_x = h // 2, w // 2

    # Decide direction
    dy = target_y - curr_y
    dx = target_x - curr_x

    if abs(dy) > abs(dx):
        # Move vertically
        if dy > 0:
            return 2  # DOWN
        else:
            return 1  # UP
    else:
        # Move horizontally
        if dx > 0:
            return 4  # RIGHT
        else:
            return 3  # LEFT


def random_click_expert(observation: np.ndarray, available_actions: list[int]) -> int:
    """Random click expert for click-based games (vc33, ft09)."""
    # Prefer click action (6) for click-based games
    if 6 in available_actions:
        return 6
    return random.choice(available_actions) if available_actions else 1


def get_expert_for_game(game_id: str) -> Callable:
    """Get appropriate expert policy for a game."""
    experts = {
        "ls20": ls20_expert,
        "vc33": random_click_expert,
        "ft09": random_click_expert,
    }
    return experts.get(game_id, lambda obs, acts: random.choice(acts) if acts else 1)


# ============================================================================
# Main
# ============================================================================


def main():
    """Collect demos for all available games."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect ARC-AGI-3 game demos")
    parser.add_argument("--game", "-g", type=str, default="ls20", help="Game ID")
    parser.add_argument(
        "--episodes", "-n", type=int, default=50, help="Number of episodes"
    )
    parser.add_argument("--render", "-r", action="store_true", help="Render games")
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output path for demos"
    )
    parser.add_argument(
        "--expert", "-e", action="store_true", help="Use expert policy"
    )

    args = parser.parse_args()

    if not ARC_AGI_AVAILABLE:
        print("Error: arc_agi package not available")
        return

    # Get policy
    if args.expert:
        policy = get_expert_for_game(args.game)
        print(f"Using expert policy for {args.game}")
    else:
        policy = None
        print(f"Using random policy for {args.game}")

    # Create collector
    collector = DemoCollector(
        game_id=args.game,
        policy=policy,
        render=args.render,
    )

    # Collect demos
    output_path = args.output or f"demos/{args.game}_demos.json"
    dataset = collector.collect_dataset(
        num_episodes=args.episodes,
        save_path=output_path,
    )

    # Summary
    successful = len(dataset.get_successful_demos())
    print(f"\nCollection complete:")
    print(f"  Total episodes: {len(dataset.demos)}")
    print(f"  Successful (made progress): {successful}")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
