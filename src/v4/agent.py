#!/usr/bin/env python3
"""
v4 Agent: Hybrid graph BFS + CNN P(state_novelty).

Strategy:
  - State graph for systematic simple action exploration (BFS to frontier)
  - CNN for click target prediction (top-K heatmap pixels)
  - Graph handles navigation efficiently, CNN handles visual prediction
  - Committed path navigation (follow full BFS path, not just first step)

Speed optimizations:
  - Pre-allocated contiguous frame buffer (fast sampling)
  - Pre-allocated one-hot tensors (avoid allocation in hot path)
  - Fast hash: Python SipHash on frame.tobytes() (no packing/blake2b)
  - No CCL segmentation — CNN heatmap replaces region centroids
  - Hash comparison instead of np.array_equal for frame change detection
  - GPU-only training indexing (no CPU round-trips)
  - GroupNorm model (no train/eval mismatch)
  - Train every 20 steps, batch_size=32

Usage:
    uv run python -m src.v4.agent --game ls20
    uv run python -m src.v4.agent --game vc33 --max-actions 200000
    uv run python -m src.v4.agent --game ft09 --max-actions 200000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .model import FrameChangeCNN


# ---------------------------------------------------------------------------
# Experience Buffer (contiguous pre-allocated arrays)
# ---------------------------------------------------------------------------

class ExperienceBuffer:
    """Hash-deduped experience buffer with contiguous memory layout.

    Each unique (frame_hash, action) pair is stored exactly once.
    Larger capacity (100K) ensures frontier data isn't lost when exploring.
    """

    def __init__(self, max_size: int = 50_000):
        self.max_size = max_size
        self._frames = np.zeros((max_size, 64, 64), dtype=np.uint8)
        self._atypes = np.zeros(max_size, dtype=np.int8)
        self._cys = np.zeros(max_size, dtype=np.int16)
        self._cxs = np.zeros(max_size, dtype=np.int16)
        self._targets = np.zeros(max_size, dtype=np.float32)
        self._count = 0
        self.seen: set[int] = set()

    def add(
        self, frame: np.ndarray, frame_hash: int,
        action_type: int, click_y: int, click_x: int,
        target: float,
    ) -> None:
        if action_type < 5:
            key = hash((frame_hash, action_type))
        else:
            key = hash((frame_hash, click_y, click_x))
        if key in self.seen or self._count >= self.max_size:
            return
        self.seen.add(key)
        i = self._count
        self._frames[i] = frame
        self._atypes[i] = action_type
        self._cys[i] = click_y
        self._cxs[i] = click_x
        self._targets[i] = target
        self._count += 1

    def sample(self, batch_size: int):
        n = self._count
        if n < batch_size:
            return None
        idx = np.random.choice(n, size=batch_size, replace=False)
        return (
            self._frames[idx],
            self._atypes[idx].astype(np.int64),
            self._cys[idx].astype(np.int64),
            self._cxs[idx].astype(np.int64),
            self._targets[idx].copy(),
        )

    def clear(self):
        self._count = 0
        self.seen.clear()

    def __len__(self):
        return self._count


# ---------------------------------------------------------------------------
# Lightweight State Graph (inlined for speed, no external dependency)
# ---------------------------------------------------------------------------

_UNTESTED = 0
_SUCCESS = 1
_DEAD = -1


class _GraphNode:
    __slots__ = ('edges', 'targets')

    def __init__(self, n_actions: int):
        self.edges = [_UNTESTED] * n_actions  # status per simple action
        self.targets = [0] * n_actions  # target hash per simple action


class StateGraph:
    """Lightweight state graph for simple action exploration."""

    def __init__(self):
        self.nodes: dict[int, _GraphNode] = {}
        self.start_hash: int | None = None

    def register(self, frame_hash: int, n_simple: int) -> None:
        if frame_hash not in self.nodes:
            self.nodes[frame_hash] = _GraphNode(n_simple)
        if self.start_hash is None:
            self.start_hash = frame_hash

    def update(self, from_hash: int, action_idx: int,
               to_hash: int, frame_changed: bool) -> None:
        node = self.nodes.get(from_hash)
        if node is None or action_idx >= len(node.edges):
            return

        if not frame_changed:
            node.edges[action_idx] = _DEAD
            node.targets[action_idx] = from_hash
        elif to_hash == self.start_hash and from_hash != self.start_hash:
            # Reset to start = game-over
            node.edges[action_idx] = _DEAD
            node.targets[action_idx] = to_hash
        else:
            node.edges[action_idx] = _SUCCESS
            node.targets[action_idx] = to_hash

    def get_untested(self, frame_hash: int) -> int | None:
        node = self.nodes.get(frame_hash)
        if node is None:
            return None
        for i, status in enumerate(node.edges):
            if status == _UNTESTED:
                return i
        return None

    def bfs_to_frontier(self, from_hash: int) -> list[int] | None:
        """BFS for shortest path to a state with untested actions.

        Returns full path of action indices, or None.
        """
        if from_hash not in self.nodes:
            return None

        # Check current node first
        untested = self.get_untested(from_hash)
        if untested is not None:
            return [untested]

        visited = {from_hash}
        queue: deque[tuple[int, list[int]]] = deque()

        node = self.nodes[from_hash]
        for i, (status, target) in enumerate(zip(node.edges, node.targets)):
            if status == _SUCCESS and target not in visited:
                visited.add(target)
                queue.append((target, [i]))

        while queue:
            current_hash, path = queue.popleft()
            current_node = self.nodes.get(current_hash)
            if current_node is None:
                continue

            # Check for untested actions
            for i, status in enumerate(current_node.edges):
                if status == _UNTESTED:
                    return path + [i]

            # Limit BFS depth
            if len(path) >= 50:
                continue

            for i, (status, target) in enumerate(
                    zip(current_node.edges, current_node.targets)):
                if status == _SUCCESS and target not in visited:
                    visited.add(target)
                    queue.append((target, path + [i]))

        return None

    def reset(self) -> None:
        self.nodes.clear()
        self.start_hash = None


# ---------------------------------------------------------------------------
# V4 Agent
# ---------------------------------------------------------------------------

class V4Agent:
    """Hybrid graph BFS + CNN P(novelty) agent."""

    def __init__(self, device: str = "cuda", model_size: str = "small",
                 persist_model: bool = False, train_every: int = 20):
        self.device = device
        self.model_size = model_size
        self.persist_model = persist_model
        self.buffer = ExperienceBuffer(max_size=50_000)
        self.graph = StateGraph()

        self.model: FrameChangeCNN | None = None
        self.optimizer: torch.optim.Adam | None = None
        self._init_model()

        # Game config
        self.available_simple: list[int] = []
        self.has_click = False
        self.num_simple = 0

        # Per-step state
        self.prev_frame: np.ndarray | None = None
        self.prev_hash: int | None = None
        self.prev_simple_idx: int = -1
        self.prev_click_y: int = -1
        self.prev_click_x: int = -1

        # Committed path queue (from graph BFS)
        self._path_queue: deque[int] = deque()

        # State tracking (visit counts for continuous novelty targets)
        self.seen_states: dict[int, int] = {}

        # Action pruning: auto-detect useless action types
        self._action_attempts: dict[int, int] = {}  # action_type → attempt count
        self._action_changes: dict[int, int] = {}   # action_type → frame change count
        self._pruned_actions: set[int] = set()       # action types that are useless
        self._prune_threshold = 20  # prune after this many zero-change attempts

        # Stats
        self.step_count = 0
        self.level_step_count = 0
        self.frame_changes = 0
        self.novel_states = 0
        self.train_steps = 0
        self.levels_completed = 0
        self.graph_actions = 0
        self.cnn_actions = 0
        self.random_actions = 0
        self.restarts = 0

        # Stuck detection → model restart
        self.steps_since_novel = 0
        self.restart_threshold = 5000
        self.max_restarts_per_level = 3
        self._level_restarts = 0

        # Hyperparameters
        self.train_every = train_every
        self.batch_size = 32
        self.action_entropy_coeff = 1e-4
        self.coord_entropy_coeff = 1e-5

        # Pre-allocated one-hot buffers (avoid allocation in hot path)
        self._onehot_inf = torch.zeros(1, 16, 64, 64, dtype=torch.float32,
                                       device=device)
        self._onehot_train = torch.zeros(self.batch_size, 16, 64, 64,
                                         dtype=torch.float32, device=device)

    def _init_model(self) -> None:
        self.model = FrameChangeCNN(size=self.model_size).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def setup(self, available_actions: list[int]) -> None:
        self.available_simple = sorted(a for a in available_actions if 1 <= a <= 5)
        self.has_click = 6 in available_actions
        self.num_simple = len(self.available_simple)

    def _hash_frame(self, frame: np.ndarray) -> int:
        return hash(frame.tobytes())

    # -------------------------------------------------------------------
    # Main act method
    # -------------------------------------------------------------------

    def act(self, frame: np.ndarray) -> tuple[int, int | None, int | None]:
        self.step_count += 1
        self.level_step_count += 1

        frame_hash = self._hash_frame(frame)

        # --- Update from previous action ---
        if self.prev_hash is not None:
            frame_changed = frame_hash != self.prev_hash
            is_novel = frame_changed and frame_hash not in self.seen_states

            if frame_changed:
                self.frame_changes += 1
            if is_novel:
                self.novel_states += 1
                self.steps_since_novel = 0
            else:
                self.steps_since_novel += 1

            # Track action type success for auto-pruning
            prev_type = self.available_simple[self.prev_simple_idx] \
                if self.prev_simple_idx >= 0 else 6
            self._action_attempts[prev_type] = \
                self._action_attempts.get(prev_type, 0) + 1
            if frame_changed:
                self._action_changes[prev_type] = \
                    self._action_changes.get(prev_type, 0) + 1
            else:
                # Check if this action type should be pruned
                # Never prune click (type 6) — random clicks easily miss targets
                if prev_type != 6:
                    attempts = self._action_attempts.get(prev_type, 0)
                    successes = self._action_changes.get(prev_type, 0)
                    if (attempts >= self._prune_threshold and successes == 0
                            and prev_type not in self._pruned_actions):
                        self._pruned_actions.add(prev_type)

            # Visit count for continuous target (count BEFORE incrementing)
            visit_count = self.seen_states.get(frame_hash, 0)

            # Update state graph for simple actions
            if self.prev_simple_idx >= 0:
                self.graph.update(self.prev_hash, self.prev_simple_idx,
                                  frame_hash, frame_changed)

            # If frame changed unexpectedly while following committed path,
            # invalidate the path (we're at a different state than expected)
            if frame_changed and self._path_queue:
                self._path_queue.clear()

            # CNN target: continuous novelty signal
            # First visit: 1.0, second: 0.5, third: 0.33, etc.
            # No frame change: 0.0
            target = 1.0 / (1.0 + visit_count) if frame_changed else 0.0

            buf_atype = self.prev_simple_idx if self.prev_simple_idx >= 0 else 5
            self.buffer.add(self.prev_frame, self.prev_hash,
                            buf_atype, self.prev_click_y, self.prev_click_x,
                            target)

        # Register state in graph
        self.seen_states[frame_hash] = self.seen_states.get(frame_hash, 0) + 1
        self.graph.register(frame_hash, self.num_simple)

        # --- Model restart when stuck ---
        if (self.steps_since_novel >= self.restart_threshold
                and self._level_restarts < self.max_restarts_per_level
                and len(self.buffer) >= self.batch_size):
            self._init_model()
            self.buffer.clear()
            self._path_queue.clear()
            # Keep graph and tried_clicks — they're ground truth
            self._level_restarts += 1
            self.restarts += 1
            self.steps_since_novel = 0

        # --- Train CNN when it's being used (skip when graph BFS handles everything) ---
        if (self.level_step_count % self.train_every == 0
                and len(self.buffer) >= self.batch_size
                and self.cnn_actions > 0):
            self._train_step()

        # --- Action selection ---
        action_type, x, y, simple_idx = self._select_action(
            frame, frame_hash)

        # --- Save state ---
        self.prev_frame = frame
        self.prev_hash = frame_hash
        self.prev_simple_idx = simple_idx
        self.prev_click_y = y if y is not None else -1
        self.prev_click_x = x if x is not None else -1

        return action_type, x, y

    def _select_action(self, frame: np.ndarray, frame_hash: int,
                        ) -> tuple[int, int | None, int | None, int]:
        """Action selection: graph BFS for nav, CNN for click/mixed.

        Action pruning filters useless types from CNN candidates.
        """
        # Effective action availability (after pruning useless types)
        effective_simple = [a for a in self.available_simple
                           if a not in self._pruned_actions]
        effective_click = self.has_click and 6 not in self._pruned_actions

        # --- Nav-only (or effectively nav-only): graph BFS ---
        if not effective_click and effective_simple:
            # Follow committed path
            if self._path_queue:
                idx = self._path_queue.popleft()
                self.graph_actions += 1
                return self.available_simple[idx], None, None, idx
            # Untested simple action at current state
            untested = self.graph.get_untested(frame_hash)
            if untested is not None:
                self.graph_actions += 1
                return self.available_simple[untested], None, None, untested
            # BFS to nearest frontier
            path = self.graph.bfs_to_frontier(frame_hash)
            if path and len(path) > 0:
                idx = path[0]
                for step_idx in path[1:]:
                    self._path_queue.append(step_idx)
                self.graph_actions += 1
                return self.available_simple[idx], None, None, idx
            return self._cnn_select(frame, frame_hash)

        # --- Click-only or Mixed: CNN drives everything ---
        return self._cnn_select(frame, frame_hash)

    @torch.no_grad()
    def _cnn_select(self, frame: np.ndarray, frame_hash: int,
                     ) -> tuple[int, int | None, int | None, int]:
        """CNN-guided stochastic action selection."""
        have_model = len(self.buffer) >= self.batch_size

        if not have_model:
            self.random_actions += 1
            return self._random_action()

        self.cnn_actions += 1

        # Get CNN predictions (pre-allocated one-hot for speed)
        frame_t = torch.from_numpy(frame).long().to(self.device)
        self._onehot_inf.zero_()
        self._onehot_inf.scatter_(1, frame_t.unsqueeze(0).unsqueeze(0), 1.0)
        action_logits, coord_logits = self.model(
            self._onehot_inf, need_coord=(self.has_click
                                          and 6 not in self._pruned_actions))
        action_probs = torch.sigmoid(action_logits[0]).cpu().numpy()
        coord_probs = (torch.sigmoid(coord_logits[0]).cpu().numpy()
                       if coord_logits is not None else None)

        candidates = []

        # Simple actions (skip pruned and graph-dead)
        for i, game_action in enumerate(self.available_simple):
            if game_action in self._pruned_actions:
                continue
            node = self.graph.nodes.get(frame_hash)
            if node and i < len(node.edges) and node.edges[i] == _DEAD:
                continue
            score = float(action_probs[i]) if i < 5 else 0.5
            candidates.append((game_action, None, None, i, max(score, 0.01)))

        # Click actions: top-K pixels from CNN heatmap (replaces CCL regions)
        if (self.has_click and 6 not in self._pruned_actions
                and coord_probs is not None):
            flat = coord_probs.ravel()
            k = min(16, flat.shape[0])
            top_idx = np.argpartition(flat, -k)[-k:]
            for idx in top_idx:
                cy, cx = divmod(int(idx), 64)
                candidates.append((6, int(cx), int(cy), -1, max(float(flat[idx]), 0.01)))
            # Random click for exploration
            ry, rx = np.random.randint(0, 64, size=2)
            candidates.append((6, int(rx), int(ry), -1, 0.02))

        if not candidates:
            self.random_actions += 1
            return self._random_action()

        # Stochastic sampling
        scores = np.array([c[4] for c in candidates], dtype=np.float64)
        scores /= scores.sum()
        idx = np.random.choice(len(candidates), p=scores)
        c = candidates[idx]
        return c[0], c[1], c[2], c[3]

    def _random_action(self) -> tuple[int, int | None, int | None, int]:
        effective_simple = [a for a in self.available_simple
                           if a not in self._pruned_actions]
        effective_click = self.has_click and 6 not in self._pruned_actions
        total = len(effective_simple) + (1 if effective_click else 0)
        if total == 0:
            return 0, None, None, -1
        choice = np.random.randint(0, total)
        if choice < len(effective_simple):
            game_action = effective_simple[choice]
            idx = self.available_simple.index(game_action)
            return game_action, None, None, idx
        else:
            y, x = np.random.randint(0, 64, size=2)
            return 6, int(x), int(y), -1

    # -------------------------------------------------------------------
    # CNN Training
    # -------------------------------------------------------------------

    def _train_step(self) -> None:
        batch = self.buffer.sample(self.batch_size)
        if batch is None:
            return

        frames_np, atypes_np, cy_np, cx_np, targets_np = batch

        frames_t = torch.from_numpy(frames_np).long().to(self.device)
        self._onehot_train.zero_()
        self._onehot_train.scatter_(1, frames_t.unsqueeze(1), 1.0)
        targets_t = torch.from_numpy(targets_np).float().to(self.device)
        atypes_t = torch.from_numpy(atypes_np).long().to(self.device)
        cy_t = torch.from_numpy(cy_np).long().to(self.device)
        cx_t = torch.from_numpy(cx_np).long().to(self.device)

        has_clicks = (atypes_t == 5).any().item()
        action_logits, coord_logits = self.model(self._onehot_train, need_coord=has_clicks)

        loss = torch.tensor(0.0, device=self.device)
        count = 0

        simple_mask = atypes_t < 5
        if simple_mask.any():
            s_actions = atypes_t[simple_mask]
            s_logits = action_logits[simple_mask]
            s_targets = targets_t[simple_mask]
            taken = s_logits.gather(1, s_actions.unsqueeze(1)).squeeze(1)
            loss = loss + F.binary_cross_entropy_with_logits(taken, s_targets)
            count += 1

        click_mask = atypes_t == 5
        if click_mask.any() and coord_logits is not None:
            c_logits = coord_logits[click_mask]
            c_targets = targets_t[click_mask]
            c_y = cy_t[click_mask]
            c_x = cx_t[click_mask]
            pixel_logits = c_logits[
                torch.arange(c_logits.shape[0], device=self.device), c_y, c_x]
            loss = loss + F.binary_cross_entropy_with_logits(
                pixel_logits, c_targets)
            count += 1

        if count == 0:
            return
        loss = loss / count

        # Entropy regularization (separate coefficients for actions vs coords)
        if simple_mask.any():
            p_a = torch.sigmoid(taken)
            ent_a = -(p_a * torch.log(p_a.clamp(min=1e-8))
                      + (1 - p_a) * torch.log((1 - p_a).clamp(min=1e-8))).mean()
            loss = loss - self.action_entropy_coeff * ent_a
        if click_mask.any() and coord_logits is not None:
            p_c = torch.sigmoid(pixel_logits)
            ent_c = -(p_c * torch.log(p_c.clamp(min=1e-8))
                      + (1 - p_c) * torch.log((1 - p_c).clamp(min=1e-8))).mean()
            loss = loss - self.coord_entropy_coeff * ent_c

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.train_steps += 1

    # -------------------------------------------------------------------
    # Level management
    # -------------------------------------------------------------------

    def on_level_complete(self) -> None:
        if not self.persist_model:
            self._init_model()
            self.buffer.clear()
        self.graph.reset()
        self.seen_states.clear()
        self._path_queue.clear()
        self.prev_frame = None
        self.prev_hash = None
        self.prev_simple_idx = -1
        self.prev_click_y = -1
        self.prev_click_x = -1
        self.level_step_count = 0
        self.steps_since_novel = 0
        self._level_restarts = 0
        self.levels_completed += 1
        # Keep pruning across levels (same game mechanics)

    def get_stats(self) -> dict:
        return {
            "step": self.step_count,
            "frame_changes": self.frame_changes,
            "novel_states": self.novel_states,
            "seen_states": len(self.seen_states),
            "graph_nodes": len(self.graph.nodes),
            "levels": self.levels_completed,
            "buffer": len(self.buffer),
            "train_steps": self.train_steps,
            "graph_actions": self.graph_actions,
            "cnn_actions": self.cnn_actions,
            "random": self.random_actions,
            "restarts": self.restarts,
            "pruned": list(self._pruned_actions),
        }


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------

def run_agent(
    game_id: str = "ls20",
    max_actions: int = 200_000,
    verbose: bool = True,
    device: str = "cuda",
    model_size: str = "small",
    persist_model: bool = False,
    train_every: int = 20,
):
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

    agent = V4Agent(device=device, model_size=model_size,
                    persist_model=persist_model, train_every=train_every)

    if verbose:
        print(f"\n{'='*60}")
        print(f"v4 Agent — Hybrid Graph BFS + CNN P(novelty)")
        print(f"Game: {game_id} | Max actions: {max_actions}")
        print(f"Model: {model_size} ({agent.model.count_params():,} params)"
              f" | persist={persist_model} | train_every={train_every}")
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
        print(f"Actions: {list(available)} | Simple: {agent.available_simple}"
              f" | Click: {agent.has_click}")

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
                print(f"  *** Level {levels_completed} complete!"
                      f" (action {action_count}, {elapsed_now:.1f}s)")

        frame = np.array(observation.frame[0])
        action_type, x, y = agent.act(frame)

        if action_type == 0:
            env.step(GameAction.RESET)
        else:
            game_action = action_map.get(action_type)
            if game_action is None:
                game_action = GameAction.RESET

            if x is not None and y is not None:
                env.step(game_action, data={"x": x, "y": y})
            else:
                env.step(game_action)

        if verbose and action_count % 2000 == 0:
            stats = agent.get_stats()
            ms = elapsed * 1000 / max(action_count, 1)
            nov_rate = stats['novel_states'] / max(action_count, 1) * 100
            pruned_str = f" prn={stats['pruned']}" if stats['pruned'] else ""
            print(
                f"  [{action_count:>6d}] "
                f"seen={stats['seen_states']:>5d} "
                f"graph={stats['graph_nodes']:>5d} "
                f"gact={stats['graph_actions']:>5d} "
                f"cnn={stats['cnn_actions']:>5d} "
                f"buf={stats['buffer']:>5d} "
                f"nov={nov_rate:.0f}%{pruned_str} | "
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
        print(f"Time: {duration:.1f}s"
              f" ({duration / max(action_count, 1) * 1000:.1f}ms/action)")
        print(f"Frame changes: {stats['frame_changes']}"
              f" ({stats['frame_changes'] / max(action_count, 1) * 100:.0f}%)")
        print(f"Novel states: {stats['novel_states']}")
        print(f"Seen states: {stats['seen_states']}")
        print(f"Graph nodes: {stats['graph_nodes']}")
        print(f"Graph actions: {stats['graph_actions']}"
              f" | CNN actions: {stats['cnn_actions']}"
              f" | Random: {stats['random']}")
        print(f"Buffer: {stats['buffer']} unique experiences")
        print(f"Training: {stats['train_steps']} steps")
        print(f"Restarts: {stats['restarts']}")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(
        description="v4 Hybrid Graph BFS + CNN P(novelty) Agent")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=200_000)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-size", "-s", default="small",
                        choices=["small", "medium", "large", "goose"])
    parser.add_argument("--persist-model", action="store_true",
                        help="Keep CNN across levels (don't reinitialize)")
    parser.add_argument("--train-every", type=int, default=20,
                        help="Train CNN every N steps (default: 20)")
    args = parser.parse_args()

    run_agent(
        game_id=args.game,
        max_actions=args.max_actions,
        verbose=not args.quiet,
        device=args.device,
        model_size=args.model_size,
        persist_model=args.persist_model,
        train_every=args.train_every,
    )


if __name__ == "__main__":
    main()
