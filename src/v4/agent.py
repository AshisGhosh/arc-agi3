#!/usr/bin/env python3
"""
v4 Agent: Online P(frame_change) CNN.

StochasticGoose-inspired approach: one CNN learns online which actions change
the frame. CNN-guided stochastic sampling for action selection.
No pretraining, no entity detection, no game type classification.
Model resets per level.

Key design: CNN predictions guide ALL action selection from the start.
No state graph for action selection (it prevented CNN from learning).
Hash-based dedup in experience buffer only.

Usage:
    uv run python -m src.v4.agent --game ls20
    uv run python -m src.v4.agent --game vc33 --max-actions 20000
    uv run python -m src.v4.agent --game ft09 --max-actions 10000
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.aria_v3.frame_processor import FrameProcessor

from .model import FrameChangeCNN, frame_to_onehot


class ExperienceBuffer:
    """Hash-deduped experience buffer for online learning.

    Stores (frame, action_type, click_y, click_x, frame_changed).
    Deduplicates by (frame_hash, action_key) to maximize sample diversity.
    action_key encodes the action type and click position.
    """

    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.frames: list[np.ndarray] = []
        self.action_types: list[int] = []  # 0-4 for simple, 5 for click
        self.click_ys: list[int] = []  # -1 if not click
        self.click_xs: list[int] = []  # -1 if not click
        self.changed: list[float] = []  # 1.0 or 0.0
        self.seen: set[str] = set()  # hash keys for dedup

    def add(
        self,
        frame: np.ndarray,
        frame_hash: str,
        action_type: int,
        click_y: int,
        click_x: int,
        frame_changed: bool,
    ) -> None:
        if action_type < 5:
            key = f"{frame_hash}:a{action_type}"
        else:
            key = f"{frame_hash}:c{click_y},{click_x}"

        if key in self.seen:
            return
        if len(self.frames) >= self.max_size:
            return

        self.seen.add(key)
        self.frames.append(frame)
        self.action_types.append(action_type)
        self.click_ys.append(click_y)
        self.click_xs.append(click_x)
        self.changed.append(1.0 if frame_changed else 0.0)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        n = len(self.frames)
        if n < batch_size:
            return None
        idx = np.random.choice(n, size=batch_size, replace=False)
        return (
            np.stack([self.frames[i] for i in idx]),
            np.array([self.action_types[i] for i in idx]),
            np.array([self.click_ys[i] for i in idx]),
            np.array([self.click_xs[i] for i in idx]),
            np.array([self.changed[i] for i in idx]),
        )

    def clear(self) -> None:
        self.frames.clear()
        self.action_types.clear()
        self.click_ys.clear()
        self.click_xs.clear()
        self.changed.clear()
        self.seen.clear()

    def __len__(self) -> int:
        return len(self.frames)


class V4Agent:
    """Online P(frame_change) CNN agent.

    Action selection: CNN predicts P(frame_change) for each candidate action.
    Stochastic sampling weighted by predictions. No graph-based exploration.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.frame_processor = FrameProcessor()
        self.buffer = ExperienceBuffer(max_size=200_000)

        self.model: FrameChangeCNN | None = None
        self.optimizer: torch.optim.Adam | None = None
        self._init_model()

        # Game config
        self.available_simple: list[int] = []
        self.has_click = False
        self.num_simple = 0

        # Per-step state
        self.prev_frame: np.ndarray | None = None
        self.prev_hash: str | None = None
        self.prev_action_type: int = -1  # 0-4 for simple, 5 for click
        self.prev_click_y: int = -1
        self.prev_click_x: int = -1

        # Stats
        self.step_count = 0
        self.level_step_count = 0
        self.frame_changes = 0
        self.train_steps = 0
        self.levels_completed = 0
        self.cnn_actions = 0
        self.random_actions = 0

        # Hyperparameters
        self.train_every = 5
        self.batch_size = 64
        self.entropy_coeff_action = 1e-4
        self.entropy_coeff_coord = 1e-5

    def _init_model(self) -> None:
        self.model = FrameChangeCNN().to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def setup(self, available_actions: list[int]) -> None:
        self.available_simple = sorted(a for a in available_actions if 1 <= a <= 5)
        self.has_click = 6 in available_actions
        self.num_simple = len(self.available_simple)

    def _hash_frame(self, frame: np.ndarray) -> str:
        flat = frame.ravel()
        packed = (flat[0::2].astype(np.uint8) << 4) | (flat[1::2].astype(np.uint8) & 0x0F)
        return hashlib.blake2b(packed.tobytes(), digest_size=16).hexdigest()

    def act(self, frame: np.ndarray) -> tuple[int, int | None, int | None]:
        """Choose an action. Returns (action_type, x, y)."""
        self.step_count += 1
        self.level_step_count += 1

        frame_hash = self._hash_frame(frame)

        # Update from previous step
        if self.prev_frame is not None:
            frame_changed = not np.array_equal(self.prev_frame, frame)
            if frame_changed:
                self.frame_changes += 1
            self.buffer.add(
                self.prev_frame, self.prev_hash,
                self.prev_action_type, self.prev_click_y, self.prev_click_x,
                frame_changed,
            )

        # Maybe train
        if self.level_step_count % self.train_every == 0 and len(self.buffer) >= self.batch_size:
            self._train_step()

        # Segment frame once (only if we have click actions)
        regions = None
        if self.has_click:
            regions = self.frame_processor.segment(frame)

        # Select action
        action_type, x, y = self._select_action(frame, regions)

        # Save state
        self.prev_frame = frame.copy()
        self.prev_hash = frame_hash
        if x is not None:
            # Click action
            self.prev_action_type = 5
            self.prev_click_y = y
            self.prev_click_x = x
        else:
            # Simple action — map game action ID to index 0-4
            self.prev_action_type = self.available_simple.index(action_type)
            self.prev_click_y = -1
            self.prev_click_x = -1

        return action_type, x, y

    def _train_step(self) -> None:
        """One gradient step on sampled batch."""
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        frames_np, atypes_np, cy_np, cx_np, changed_np = batch

        frames_t = torch.from_numpy(frames_np).to(self.device)
        frames_oh = frame_to_onehot(frames_t)
        changed_t = torch.from_numpy(changed_np).float().to(self.device)
        atypes_t = torch.from_numpy(atypes_np).long().to(self.device)

        action_logits, coord_logits = self.model(frames_oh)

        loss = torch.tensor(0.0, device=self.device)
        count = 0

        # Simple action losses: train action_logits at the taken action index
        simple_mask = atypes_t < 5
        if simple_mask.any():
            s_actions = atypes_t[simple_mask]
            s_logits = action_logits[simple_mask]
            s_changed = changed_t[simple_mask]
            taken = s_logits.gather(1, s_actions.unsqueeze(1)).squeeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(taken, s_changed)
            count += 1

        # Click action losses: train coord_logits at the actual click position
        click_mask = atypes_t == 5
        if click_mask.any():
            c_logits = coord_logits[click_mask]  # [B_click, 64, 64]
            c_changed = changed_t[click_mask]
            c_y = torch.from_numpy(cy_np[click_mask.cpu().numpy()]).long().to(self.device)
            c_x = torch.from_numpy(cx_np[click_mask.cpu().numpy()]).long().to(self.device)
            # Gather logit at (y, x) for each sample
            pixel_logits = c_logits[torch.arange(c_logits.shape[0], device=self.device), c_y, c_x]
            loss = loss + F.binary_cross_entropy_with_logits(pixel_logits, c_changed)
            count += 1

        if count == 0:
            return
        loss = loss / count

        # Entropy regularization
        action_probs = torch.sigmoid(action_logits)
        action_entropy = -(
            action_probs * torch.log(action_probs.clamp(min=1e-8))
            + (1 - action_probs) * torch.log((1 - action_probs).clamp(min=1e-8))
        ).mean()

        coord_probs = torch.sigmoid(coord_logits)
        coord_entropy = -(
            coord_probs * torch.log(coord_probs.clamp(min=1e-8))
            + (1 - coord_probs) * torch.log((1 - coord_probs).clamp(min=1e-8))
        ).mean()

        loss = loss - self.entropy_coeff_action * action_entropy - self.entropy_coeff_coord * coord_entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1

    @torch.no_grad()
    def _select_action(
        self, frame: np.ndarray, regions: list | None
    ) -> tuple[int, int | None, int | None]:
        """Select action using CNN predictions.

        For early steps (before enough training data), use uniform random.
        After that, stochastic sampling weighted by P(frame_change).
        """
        # Early exploration: uniform random
        if len(self.buffer) < self.batch_size:
            self.random_actions += 1
            return self._random_action(regions)

        self.cnn_actions += 1

        # Get CNN predictions
        frame_t = torch.from_numpy(frame).unsqueeze(0).to(self.device)
        frame_oh = frame_to_onehot(frame_t)
        self.model.eval()
        action_logits, coord_logits = self.model(frame_oh)
        self.model.train()

        action_probs = torch.sigmoid(action_logits[0]).cpu().numpy()  # [5]
        coord_probs = torch.sigmoid(coord_logits[0]).cpu().numpy()  # [64, 64]

        # Build unified probability distribution over all candidate actions
        # Simple actions (each is one candidate)
        candidates = []  # (probability, action_type_game, x, y)
        for i, game_action in enumerate(self.available_simple):
            p = float(action_probs[i]) if i < 5 else 0.5
            candidates.append((max(p, 0.01), game_action, None, None))

        # Click actions: use region centroids for efficiency
        if self.has_click and regions:
            for region in regions:
                cy = max(0, min(63, int(round(region.centroid_y))))
                cx = max(0, min(63, int(round(region.centroid_x))))
                p = float(coord_probs[cy, cx])
                # Scale click probs relative to simple action probs
                # (otherwise 50+ regions dominate the distribution)
                candidates.append((max(p, 0.01) / max(len(regions), 1), 6, cx, cy))

            # Also add a random click with small probability
            ry, rx = np.random.randint(0, 64, size=2)
            candidates.append((0.02, 6, int(rx), int(ry)))

        # Sample
        probs = np.array([c[0] for c in candidates], dtype=np.float64)
        probs = probs / probs.sum()
        idx = np.random.choice(len(candidates), p=probs)
        _, action_type, x, y = candidates[idx]
        return action_type, x, y

    def _random_action(self, regions: list | None) -> tuple[int, int | None, int | None]:
        """Uniform random action for early exploration."""
        total = self.num_simple + (1 if self.has_click else 0)
        choice = np.random.randint(0, total)
        if choice < self.num_simple:
            return self.available_simple[choice], None, None
        else:
            if regions:
                region = regions[np.random.randint(0, len(regions))]
                x, y = self.frame_processor.get_click_point(region)
                return 6, x, y
            else:
                return 6, np.random.randint(0, 64), np.random.randint(0, 64)

    def on_level_complete(self) -> None:
        """Reset for a new level."""
        self._init_model()
        self.buffer.clear()
        self.prev_frame = None
        self.prev_hash = None
        self.prev_action_type = -1
        self.level_step_count = 0
        self.levels_completed += 1

    def get_stats(self) -> dict:
        return {
            "step": self.step_count,
            "frame_changes": self.frame_changes,
            "levels": self.levels_completed,
            "buffer": len(self.buffer),
            "train_steps": self.train_steps,
            "cnn_actions": self.cnn_actions,
            "random_actions": self.random_actions,
        }


def run_agent(
    game_id: str = "ls20",
    max_actions: int = 20_000,
    verbose: bool = True,
    device: str = "cuda",
):
    """Run v4 agent on an ARC-AGI-3 game."""
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

    agent = V4Agent(device=device)

    if verbose:
        print(f"\n{'='*60}")
        print(f"v4 Agent — Online P(frame_change) CNN")
        print(f"Game: {game_id} | Max actions: {max_actions}")
        print(f"Model params: {agent.model.count_params():,}")
        print(f"{'='*60}")

    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    available = env.observation_space.available_actions
    agent.setup(list(available))

    action_map = {}
    for i in range(1, 8):
        if i in available:
            action_map[i] = getattr(GameAction, f"ACTION{i}", None)

    if verbose:
        print(f"Actions: {list(available)} | Simple: {agent.available_simple} | Click: {agent.has_click}")

    start_time = time.time()
    action_count = 0
    levels_completed = 0

    while action_count < max_actions:
        elapsed = time.time() - start_time
        if elapsed > 180:
            if verbose:
                print(f"\nTime budget exceeded ({elapsed:.0f}s)")
            break

        observation = env.observation_space

        if observation.state == GameState.WIN:
            if verbose:
                print(f"\nGame won after {action_count} actions!")
            levels_completed = observation.levels_completed
            break
        elif observation.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            env.step(GameAction.RESET)
            action_count += 1
            continue

        if observation.levels_completed > levels_completed:
            levels_completed = observation.levels_completed
            agent.on_level_complete()
            if verbose:
                elapsed_now = time.time() - start_time
                print(f"  *** Level {levels_completed} complete! (action {action_count}, {elapsed_now:.1f}s)")

        frame = np.array(observation.frame[0])
        action_type, x, y = agent.act(frame)

        game_action = action_map.get(action_type)
        if game_action is None:
            game_action = GameAction.RESET

        if x is not None and y is not None:
            env.step(game_action, data={"x": x, "y": y})
        else:
            env.step(game_action)

        if verbose and action_count % 1000 == 0:
            stats = agent.get_stats()
            ms = elapsed * 1000 / max(action_count, 1)
            chg_rate = stats['frame_changes'] / max(action_count, 1) * 100
            print(
                f"  [{action_count:>6d}] "
                f"changes={stats['frame_changes']:>5d} ({chg_rate:.0f}%) "
                f"buf={stats['buffer']:>6d} "
                f"train={stats['train_steps']:>4d} "
                f"cnn={stats['cnn_actions']:>5d} "
                f"rnd={stats['random_actions']:>3d} | "
                f"{ms:.1f}ms/act"
            )

        action_count += 1

    duration = time.time() - start_time
    stats = agent.get_stats()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Results — {game_id}")
        print(f"{'='*60}")
        print(f"Actions: {action_count}")
        print(f"Levels completed: {levels_completed}")
        print(f"Time: {duration:.1f}s ({duration/max(action_count,1)*1000:.1f}ms/action)")
        print(f"Frame changes: {stats['frame_changes']} ({stats['frame_changes']/max(action_count,1)*100:.0f}%)")
        print(f"Buffer: {stats['buffer']} unique experiences")
        print(f"Training: {stats['train_steps']} steps")
        print(f"CNN-guided: {stats['cnn_actions']}, Random: {stats['random_actions']}")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(description="v4 Online P(frame_change) CNN Agent")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=20_000)
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
