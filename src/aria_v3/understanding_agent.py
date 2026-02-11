#!/usr/bin/env python3
"""
Understanding Agent (v3.2) for ARC-AGI-3.

Uses a pretrained understanding model (CNN encoder + temporal transformer + decoder heads)
to discover game rules online, with TTT (test-time training) for per-game adaptation.

Action selection: every candidate action is scored using ALL available signals —
graph novelty, understanding predictions, entity roles, pathfinding toward targets.
No hard fallback boundaries; understanding always guides exploration from step 1.

Usage:
    uv run python -m src.aria_v3.understanding_agent --game ls20
    uv run python -m src.aria_v3.understanding_agent --game vc33 --max-actions 5000
"""

import argparse
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch

from .frame_processor import FrameProcessor, Region
from .state_graph import StateGraph
from .understanding.dataset import ENTITY_ROLE_TO_IDX, GAME_TYPE_TO_IDX
from .understanding.model import UnderstandingModel
from .understanding.ttt import TTTEngine

# Reverse mapping: index → game type string
IDX_TO_GAME_TYPE = {v: k for k, v in GAME_TYPE_TO_IDX.items()}
IDX_TO_ENTITY_ROLE = {v: k for k, v in ENTITY_ROLE_TO_IDX.items()}

# Understanding update schedule: step numbers where we run the temporal transformer
UNDERSTANDING_SCHEDULE = [10, 30, 100, 200, 300, 400, 500, 700, 1000,
                          1500, 2000, 3000, 4000, 5000]


@dataclass
class GameUnderstanding:
    """Structured understanding of the current game, derived from model predictions."""

    game_type: str = "unknown"
    confidence: float = 0.0

    # Per-action movement map: action_id → (dx, dy) in pixels
    movement_map: dict[int, tuple[float, float]] = field(default_factory=dict)

    # Per-action probabilities
    change_prob: dict[int, float] = field(default_factory=dict)
    blocked_prob: dict[int, float] = field(default_factory=dict)

    # Entity role assignments
    player_colors: list[int] = field(default_factory=list)
    wall_colors: list[int] = field(default_factory=list)
    collectible_colors: list[int] = field(default_factory=list)
    background_colors: list[int] = field(default_factory=list)


class UnderstandingAgent:
    """v3.2 agent: pretrained understanding model + TTT + state graph execution."""

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str = "checkpoints/understanding/best.pt",
    ):
        self.device = device
        self.frame_processor = FrameProcessor()
        self.state_graph = StateGraph()

        # Load pretrained understanding model
        self.model = UnderstandingModel()
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])

        # TTT engine (wraps model, applies LoRA to encoder)
        # Must be created BEFORE .to(device) so LoRA params are included in the move
        self.ttt = TTTEngine(self.model)

        self.model.to(device)
        self.model.eval()

        # Transition buffer for understanding model input
        self.frame_buffer: list[torch.Tensor] = []
        self.action_buffer: list[int] = []
        self.next_frame_buffer: list[torch.Tensor] = []

        # Game state
        self.available_simple: list[int] = []
        self.has_click: bool = False
        self.prev_frame: np.ndarray | None = None
        self.prev_hash: str | None = None
        self.prev_action_idx: int | None = None
        self.prev_action_type: int = 0

        # Understanding state
        self.understanding = GameUnderstanding()
        self.next_understanding_idx = 0  # index into UNDERSTANDING_SCHEDULE
        self.level_step_count = 0  # resets each level, used for understanding schedule

        # Track clicked regions to avoid re-clicking same spot
        self.clicked_region_keys: set[tuple[int, float, float]] = set()

        # Empirical per-action change tracking
        # Keyed by action type (1-7), tracks actual frame change rates
        self.empirical_changes: dict[int, int] = {}
        self.empirical_attempts: dict[int, int] = {}

        # Frontier plan queue: committed multi-step path to frontier
        self.frontier_plan: deque[int] = deque()  # action indices

        # Stats
        self.step_count = 0
        self.levels_completed = 0
        self.understanding_runs = 0
        self.ttt_updates = 0
        self.frontier_navigations = 0

    def setup(self, available_actions: list[int]) -> None:
        """Configure for a game's action space."""
        self.available_simple = [a for a in available_actions if 1 <= a <= 5]
        self.has_click = 6 in available_actions

    def act(self, frame: np.ndarray) -> tuple[int, int | None, int | None]:
        """Main agent step. Returns (action_type, x, y).

        action_type: 1-7 game API ID
        x, y: pixel coordinates for click (action 6), None for simple actions
        """
        self.step_count += 1
        self.level_step_count += 1

        # Phase 1: Process frame
        regions = self.frame_processor.segment(frame)
        frame_hash = self.frame_processor.hash_frame(frame)

        num_simple = len(self.available_simple)
        num_regions = len(regions) if self.has_click else 0
        total_actions = num_simple + num_regions

        self.state_graph.register_state(frame_hash, total_actions)

        # Phase 2: Update from previous step
        if self.prev_hash is not None and self.prev_action_idx is not None:
            frame_changed = not np.array_equal(self.prev_frame, frame)
            self.state_graph.update(
                self.prev_hash, self.prev_action_idx, frame_hash, frame_changed
            )

            # Track empirical change rate per action type
            at = self.prev_action_type
            self.empirical_attempts[at] = self.empirical_attempts.get(at, 0) + 1
            if frame_changed:
                self.empirical_changes[at] = self.empirical_changes.get(at, 0) + 1

            # Feed transition to TTT
            frame_t = torch.from_numpy(self.prev_frame).long()
            frame_tp1 = torch.from_numpy(frame).long()
            ttt_loss = self.ttt.observe(frame_t, self.prev_action_type, frame_tp1)
            if ttt_loss is not None:
                self.ttt_updates += 1

            # Buffer for understanding model
            self.frame_buffer.append(frame_t.cpu())
            self.action_buffer.append(self.prev_action_type)
            self.next_frame_buffer.append(frame_tp1.cpu())
            # Trim to 200
            if len(self.frame_buffer) > 200:
                self.frame_buffer.pop(0)
                self.action_buffer.pop(0)
                self.next_frame_buffer.pop(0)

        # Phase 3: Maybe update understanding
        self._maybe_run_understanding()

        # Fallback: empirical entity/movement detection when model fails
        if self.level_step_count % 50 == 0 and self.level_step_count >= 50:
            if not self.understanding.player_colors:
                self._empirical_entity_detection()
            if self.available_simple and len(self.understanding.movement_map) < len(self.available_simple):
                self._empirical_movement_map()

        # Phase 4: Select action
        action_type, x, y = self._select_action(frame, frame_hash, regions, total_actions)

        # Save state for next step
        self.prev_frame = frame.copy()
        self.prev_hash = frame_hash
        self.prev_action_idx = self._action_to_idx(action_type, x, y, regions)
        self.prev_action_type = action_type if action_type != 6 else 6

        return action_type, x, y

    def on_level_complete(self) -> None:
        """Reset for a new level. Keep game-level understanding, dampen confidence."""
        self.levels_completed += 1

        # Reset TTT (LoRA weights back to zero, buffer cleared)
        self.ttt.reset()

        # Reset state graph (new layout)
        self.state_graph.reset()

        # Clear transition buffers
        self.frame_buffer.clear()
        self.action_buffer.clear()
        self.next_frame_buffer.clear()

        # Clear click tracking
        self.clicked_region_keys.clear()

        # Reset empirical tracking (new level = new dynamics possible)
        self.empirical_changes.clear()
        self.empirical_attempts.clear()

        # Clear frontier plan
        self.frontier_plan.clear()

        # Reset step tracking
        self.prev_frame = None
        self.prev_hash = None
        self.prev_action_idx = None
        self.level_step_count = 0

        # Dampen confidence but keep understanding (game type persists across levels)
        self.understanding.confidence = min(self.understanding.confidence, 0.4)

        # Reset understanding schedule
        self.next_understanding_idx = 0

    # ──────────────────────────────────────────────────────────
    # Understanding model
    # ──────────────────────────────────────────────────────────

    def _maybe_run_understanding(self) -> None:
        """Run understanding model on scheduled steps (level-relative)."""
        if self.next_understanding_idx >= len(UNDERSTANDING_SCHEDULE):
            # Past the schedule — run every 500 steps
            next_step = UNDERSTANDING_SCHEDULE[-1] + 500 * (
                self.next_understanding_idx - len(UNDERSTANDING_SCHEDULE) + 1
            )
        else:
            next_step = UNDERSTANDING_SCHEDULE[self.next_understanding_idx]

        if self.level_step_count >= next_step and len(self.frame_buffer) >= 5:
            self._run_understanding()
            self.next_understanding_idx += 1

    @torch.no_grad()
    def _run_understanding(self) -> None:
        """Run the full understanding pipeline on buffered transitions."""
        self.understanding_runs += 1
        self.model.eval()

        # Take last 100 transitions (or fewer if buffer smaller)
        n = min(len(self.frame_buffer), 100)
        frames = torch.stack(self.frame_buffer[-n:]).unsqueeze(0).to(self.device)
        actions = torch.tensor(
            self.action_buffer[-n:], dtype=torch.long
        ).unsqueeze(0).to(self.device)
        next_frames = torch.stack(self.next_frame_buffer[-n:]).unsqueeze(0).to(self.device)

        mask = torch.ones(1, n, dtype=torch.bool, device=self.device)

        predictions = self.model(frames, actions, next_frames, mask=mask)
        self._derive_understanding(predictions)

    def _derive_understanding(self, predictions: dict[str, torch.Tensor]) -> None:
        """Convert raw model output to structured GameUnderstanding."""
        u = self.understanding

        # Game type: argmax of logits
        game_type_logits = predictions["game_type"][0]  # [8]
        game_type_idx = game_type_logits.argmax().item()
        u.game_type = IDX_TO_GAME_TYPE.get(game_type_idx, "unknown")

        # Confidence: already sigmoid'd
        u.confidence = predictions["confidence"][0].item()

        # Movement map: per-action shift from classification
        shift = predictions["shift"][0]  # [8, 2]
        change_prob = predictions["change_prob"][0]  # [8]
        blocked_prob = predictions["blocked_prob"][0]  # [8]

        u.movement_map = {}
        u.change_prob = {}
        u.blocked_prob = {}

        for a in range(8):
            cp = change_prob[a].item()
            bp = blocked_prob[a].item()
            u.change_prob[a] = cp
            u.blocked_prob[a] = bp

            dx = shift[a, 0].item()
            dy = shift[a, 1].item()

            # Only record movement if the action actually changes frames
            if cp > 0.3 and (abs(dx) > 1.0 or abs(dy) > 1.0):
                u.movement_map[a] = (dx, dy)

        # Entity roles: apply sigmoid to raw logits, threshold
        entity_logits = predictions["entity_roles"][0]  # [16, 5]
        entity_probs = torch.sigmoid(entity_logits)  # [16, 5]

        u.player_colors = []
        u.wall_colors = []
        u.collectible_colors = []
        u.background_colors = []

        for c in range(16):
            probs = entity_probs[c]  # [5]
            best_role = probs.argmax().item()
            best_prob = probs[best_role].item()

            if best_prob < 0.4:
                continue

            role_name = IDX_TO_ENTITY_ROLE.get(best_role, "background")
            if role_name == "player":
                u.player_colors.append(c)
            elif role_name == "wall":
                u.wall_colors.append(c)
            elif role_name == "collectible":
                u.collectible_colors.append(c)
            elif role_name == "background":
                u.background_colors.append(c)

    def _empirical_movement_map(self) -> None:
        """Build movement map from direct frame-diff observation.

        For each action type, find the consistent displacement of changed pixels.
        Only for simple actions (1-5) that show consistent patterns.
        """
        if len(self.frame_buffer) < 20:
            return

        n = min(len(self.frame_buffer), 100)
        # Per-action displacement tracking
        action_displacements: dict[int, list[tuple[float, float]]] = {}

        for i in range(-n, 0):
            f = self.frame_buffer[i].numpy()
            nf = self.next_frame_buffer[i].numpy()
            a = self.action_buffer[i]

            if a < 1 or a > 5:
                continue
            if np.array_equal(f, nf):
                continue

            # Find changed pixels
            changed = (f != nf)
            if changed.sum() < 2 or changed.sum() > 500:
                continue  # too few or too many changes

            # Find positions where changes occurred
            ys, xs = np.where(changed)
            if len(ys) == 0:
                continue

            # Colors that disappeared from these positions
            old_colors = f[ys, xs]
            new_colors = nf[ys, xs]

            # Find the most common "disappeared" color (player moving away)
            disappeared_colors = {}
            for j in range(len(ys)):
                if old_colors[j] != new_colors[j]:
                    c = old_colors[j]
                    if c not in disappeared_colors:
                        disappeared_colors[c] = []
                    disappeared_colors[c].append((xs[j], ys[j]))

            appeared_colors = {}
            for j in range(len(ys)):
                if old_colors[j] != new_colors[j]:
                    c = new_colors[j]
                    if c not in appeared_colors:
                        appeared_colors[c] = []
                    appeared_colors[c].append((xs[j], ys[j]))

            # Find color that both appeared and disappeared (likely player)
            # Prefer smallest-area color (player is typically small)
            best_color = None
            best_area = float("inf")
            for c in disappeared_colors:
                if c in appeared_colors and len(disappeared_colors[c]) > 1:
                    area = len(disappeared_colors[c])
                    if area < best_area:
                        best_area = area
                        best_color = c

            if best_color is not None:
                c = best_color
                old_pts = np.array(disappeared_colors[c], dtype=np.float64)
                new_pts = np.array(appeared_colors[c], dtype=np.float64)
                old_center = old_pts.mean(axis=0)
                new_center = new_pts.mean(axis=0)
                dx = new_center[0] - old_center[0]
                dy = new_center[1] - old_center[1]
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    if a not in action_displacements:
                        action_displacements[a] = []
                    action_displacements[a].append((dx, dy))

        # Build movement map from consistent displacements
        u = self.understanding
        for a, disps in action_displacements.items():
            if len(disps) < 3:
                continue
            dxs = [d[0] for d in disps]
            dys = [d[1] for d in disps]
            # Use median for robustness
            med_dx = float(np.median(dxs))
            med_dy = float(np.median(dys))
            # Check consistency (std should be small relative to magnitude)
            std_dx = float(np.std(dxs))
            std_dy = float(np.std(dys))
            mag = abs(med_dx) + abs(med_dy)
            if mag > 1.0 and (std_dx + std_dy) < mag * 0.5:
                u.movement_map[a] = (med_dx, med_dy)

    def _empirical_entity_detection(self) -> None:
        """Detect entity roles from direct frame statistics when model fails.

        Computes per-color statistics from the transition buffer and classifies
        using simple thresholds. Only updates when model reports no entities.
        """
        u = self.understanding
        # Only run as fallback when model detects nothing
        if u.player_colors or u.wall_colors or u.collectible_colors:
            return

        n = min(len(self.frame_buffer), 50)
        if n < 10:
            return

        npix = 64 * 64
        color_area = np.zeros(16, dtype=np.float64)
        color_volatility = np.zeros(16, dtype=np.float64)
        color_static_sum = np.zeros(16, dtype=np.float64)
        color_present_count = np.zeros(16, dtype=np.float64)

        for i in range(-n, 0):
            f = self.frame_buffer[i].numpy()   # [64, 64]
            nf = self.next_frame_buffer[i].numpy()

            for c in range(16):
                mask_f = (f == c)
                mask_nf = (nf == c)
                area = mask_f.sum()
                color_area[c] += area / npix

                appeared = (~mask_f & mask_nf).sum()
                disappeared = (mask_f & ~mask_nf).sum()
                color_volatility[c] += (appeared + disappeared) / npix

                if area > 0:
                    same = (mask_f & mask_nf).sum()
                    color_static_sum[c] += same / area
                    color_present_count[c] += 1

        # Normalize
        color_area /= n
        color_volatility /= n
        safe_count = np.where(color_present_count > 0, color_present_count, 1.0)
        color_static = np.where(
            color_present_count > 0,
            color_static_sum / safe_count,
            1.0,
        )

        # Classify using thresholds
        for c in range(16):
            if color_area[c] < 0.0005:
                continue  # not present

            vol_ratio = color_volatility[c] / max(color_area[c], 1e-8)

            if color_area[c] > 0.3:
                u.background_colors.append(c)
            elif color_static[c] > 0.98 and color_area[c] > 0.01:
                u.wall_colors.append(c)
            elif vol_ratio > 0.3 and color_area[c] < 0.05:
                u.player_colors.append(c)
            elif vol_ratio > 0.05 and color_area[c] < 0.03:
                # Small, somewhat volatile → could be collectible
                if color_static[c] < 0.95:
                    u.collectible_colors.append(c)

    def _get_effective_change_prob(self, action_type: int) -> float:
        """Blend model's change_prob with empirical observations.

        - With < 5 observations: use model as prior, weight empirical lightly
        - With >= 5 observations: weight empirical heavily (model can be wrong)
        - With >= 20 observations: almost entirely empirical
        """
        model_cp = self.understanding.change_prob.get(action_type, 0.5)
        attempts = self.empirical_attempts.get(action_type, 0)
        if attempts == 0:
            return model_cp

        empirical_cp = self.empirical_changes.get(action_type, 0) / attempts
        # Bayesian-style blending: as observations grow, trust empirical more
        empirical_weight = min(attempts / 10.0, 1.0)  # ramps 0→1 over 10 obs
        return model_cp * (1 - empirical_weight) + empirical_cp * empirical_weight

    # ──────────────────────────────────────────────────────────
    # Action selection — unified scoring
    # ──────────────────────────────────────────────────────────

    def _select_action(
        self,
        frame: np.ndarray,
        frame_hash: str,
        regions: list[Region],
        total_actions: int,
    ) -> tuple[int, int | None, int | None]:
        """Score every candidate action, pick the best.

        Every action is scored by combining:
        - Graph novelty: untested actions get a large bonus
        - Dead penalty: known-dead actions are excluded
        - Understanding change_prob: prefer actions likely to change the frame
        - Understanding blocked_prob: penalize actions likely to be blocked
        - Entity roles: penalize clicks on walls/background, bonus on collectibles
        - Pathfinding: bonus for actions that move toward a target
        - Click novelty: bonus for clicking un-tried regions
        """
        u = self.understanding
        node = self.state_graph.nodes.get(frame_hash)
        player_pos = self._find_player(frame, regions)
        target_pos = self._find_nearest_target(regions, player_pos)

        # Priority 0: Execute committed frontier plan
        # But abort if current state has untested actions (better to test locally)
        if self.frontier_plan:
            untested = self.state_graph.get_untested_action(frame_hash)
            if untested is not None:
                self.frontier_plan.clear()  # abort plan, test local action
            else:
                action_idx = self.frontier_plan.popleft()
                action_type, x, y = self._idx_to_game_action(action_idx, regions)
                return action_type, x, y

        candidates: list[tuple[float, int, int | None, int | None]] = []

        # Score simple actions
        for idx, action_id in enumerate(self.available_simple):
            score = self._score_simple_action(
                action_id, idx, node, u, player_pos, target_pos
            )
            if score is not None:
                candidates.append((score, action_id, None, None))

        # Score click actions
        if self.has_click:
            for i, r in enumerate(regions[:30]):  # cap at 30 regions
                graph_idx = len(self.available_simple) + i
                score = self._score_click_action(
                    r, graph_idx, node, u
                )
                if score is not None:
                    x, y = self.frame_processor.get_click_point(r)
                    candidates.append((score, 6, x, y))

        # Frontier navigation: if no untested actions here, commit to full path
        has_untested = any(s > 4.0 for s, _, _, _ in candidates)  # untested = +5.0
        if not has_untested:
            full_path = self.state_graph.get_full_path_to_frontier(frame_hash)
            if full_path:
                self.frontier_navigations += 1
                # Commit to the full path
                self.frontier_plan = deque(full_path)
                action_idx = self.frontier_plan.popleft()
                action_type, x, y = self._idx_to_game_action(action_idx, regions)
                return action_type, x, y

        if not candidates:
            # All actions dead or excluded — re-score without death exclusion
            # (dead actions are still better than doing nothing)
            for idx, action_id in enumerate(self.available_simple):
                candidates.append((0.0, action_id, None, None))
            if self.has_click:
                for r in regions[:10]:
                    x, y = self.frame_processor.get_click_point(r)
                    candidates.append((0.0, 6, x, y))
        if not candidates:
            # Truly nothing (no actions at all) — click center or action 1
            if self.has_click:
                return 6, 32, 32
            return self.available_simple[0] if self.available_simple else 1, None, None

        # Pick: best score with small random tie-breaking
        candidates.sort(key=lambda c: -c[0])
        # Among top candidates within 10% of best, pick randomly for diversity
        best_score = candidates[0][0]
        threshold = best_score - abs(best_score) * 0.1 - 0.01
        top = [c for c in candidates if c[0] >= threshold]
        chosen = top[np.random.randint(len(top))]
        return chosen[1], chosen[2], chosen[3]

    def _score_simple_action(
        self,
        action_id: int,
        graph_idx: int,
        node,
        u: GameUnderstanding,
        player_pos: tuple[float, float] | None,
        target_pos: tuple[float, float] | None,
    ) -> float | None:
        """Score a simple action (1-5). Returns None if dead (excluded)."""
        score = 0.0

        # Graph state
        if node is not None and graph_idx < len(node.edges):
            edge = node.edges[graph_idx]
            if edge.result.value == -1:  # DEAD
                return None  # exclude entirely
            if edge.result.value == 0:  # UNTESTED
                score += 5.0  # strong novelty bonus

        # Understanding: blended change_prob and blocked_prob
        cp = self._get_effective_change_prob(action_id)
        bp = u.blocked_prob.get(action_id, 0.5)
        score += 3.0 * cp          # reward likely frame changes
        score -= 2.0 * bp          # penalize likely blocks

        # Pathfinding: does this action move us toward a target?
        if player_pos is not None and target_pos is not None:
            move = u.movement_map.get(action_id)
            if move is not None:
                dx, dy = move
                px, py = player_pos
                tx, ty = target_pos
                cur_dist = abs(tx - px) + abs(ty - py)
                new_dist = abs(tx - (px + dx)) + abs(ty - (py + dy))
                reduction = cur_dist - new_dist
                # Scale by confidence — trust the map more as confidence grows
                score += reduction * 0.1 * max(u.confidence, 0.1)

        return score

    def _score_click_action(
        self,
        region: Region,
        graph_idx: int,
        node,
        u: GameUnderstanding,
    ) -> float | None:
        """Score a click on a region. Returns None if dead (excluded)."""
        score = 0.0

        # Graph state
        if node is not None and graph_idx < len(node.edges):
            edge = node.edges[graph_idx]
            if edge.result.value == -1:  # DEAD
                return None
            if edge.result.value == 0:  # UNTESTED
                score += 5.0

        # Entity roles: what are we clicking on?
        color = region.color
        if color in u.wall_colors:
            score -= 3.0  # strongly avoid walls
        elif color in u.background_colors:
            score -= 2.0  # avoid background
        elif color in u.collectible_colors:
            score += 4.0  # target collectibles
        elif color in u.player_colors:
            score -= 1.0  # usually not useful to click player
        else:
            score += 1.0  # unknown — mildly interesting

        # Click novelty: bonus for un-tried regions
        key = (region.color, round(region.centroid_x, 1), round(region.centroid_y, 1))
        if key not in self.clicked_region_keys:
            score += 2.0
            self.clicked_region_keys.add(key)

        # Understanding: click change probability (action 6)
        cp = u.change_prob.get(6, 0.5)
        score += 2.0 * cp

        # Prefer smaller non-background regions (more likely to be interactive)
        if region.area < 100:
            score += 0.5
        elif region.area > 1000:
            score -= 0.5

        return score

    def _find_player(
        self, frame: np.ndarray, regions: list[Region]
    ) -> tuple[float, float] | None:
        """Find player position from understood player colors."""
        u = self.understanding
        if not u.player_colors:
            return None

        # Find smallest region with player color (player is usually small)
        best = None
        best_area = float("inf")
        for r in regions:
            if r.color in u.player_colors and r.area < best_area:
                best_area = r.area
                best = r

        if best is not None:
            return (best.centroid_x, best.centroid_y)
        return None

    def _find_nearest_target(
        self,
        regions: list[Region],
        player_pos: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        """Find nearest collectible or interesting region position."""
        u = self.understanding
        if player_pos is None:
            return None

        px, py = player_pos
        target_colors = set(u.collectible_colors)
        skip = set(u.wall_colors) | set(u.background_colors) | set(u.player_colors)

        best_pos = None
        best_dist = float("inf")

        for r in regions:
            # Prefer known collectibles, but also consider unknown small regions
            if target_colors:
                if r.color not in target_colors:
                    continue
            else:
                if r.color in skip or r.area > 200:
                    continue

            dist = abs(r.centroid_x - px) + abs(r.centroid_y - py)
            if dist < best_dist and dist > 2:
                best_dist = dist
                best_pos = (r.centroid_x, r.centroid_y)

        return best_pos

    # ──────────────────────────────────────────────────────────
    # Action index conversion (for state graph)
    # ──────────────────────────────────────────────────────────

    def _idx_to_game_action(
        self, action_idx: int, regions: list[Region]
    ) -> tuple[int, int | None, int | None]:
        """Convert unified graph index to (action_type, x, y)."""
        num_simple = len(self.available_simple)
        if action_idx < num_simple:
            return self.available_simple[action_idx], None, None
        region_idx = action_idx - num_simple
        if region_idx < len(regions):
            x, y = self.frame_processor.get_click_point(regions[region_idx])
            return 6, x, y
        return 6, 32, 32

    def _action_to_idx(
        self,
        action_type: int,
        x: int | None,
        y: int | None,
        regions: list[Region],
    ) -> int:
        """Convert game action back to unified graph index."""
        if action_type != 6:
            try:
                return self.available_simple.index(action_type)
            except ValueError:
                return 0
        # Click — find closest region
        if x is not None and y is not None:
            num_simple = len(self.available_simple)
            best_idx = num_simple
            best_dist = float("inf")
            for i, r in enumerate(regions):
                rx, ry = self.frame_processor.get_click_point(r)
                dist = abs(rx - x) + abs(ry - y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = num_simple + i
            return best_idx
        return len(self.available_simple)

    def get_stats(self) -> dict:
        """Get combined stats."""
        graph_stats = self.state_graph.get_stats()
        u = self.understanding
        # Empirical change rates per action
        empirical = {}
        for at in sorted(set(self.empirical_attempts.keys())):
            attempts = self.empirical_attempts[at]
            changes = self.empirical_changes.get(at, 0)
            empirical[at] = f"{changes}/{attempts}={changes/attempts:.0%}" if attempts > 0 else "0/0"
        return {
            "step": self.step_count,
            "levels": self.levels_completed,
            "game_type": u.game_type,
            "confidence": u.confidence,
            "player": u.player_colors,
            "walls": u.wall_colors,
            "collectibles": u.collectible_colors,
            "movement": {a: (round(dx, 1), round(dy, 1))
                         for a, (dx, dy) in u.movement_map.items()},
            "empirical_change": empirical,
            "understanding_runs": self.understanding_runs,
            "ttt_updates": self.ttt_updates,
            "frontier_navs": self.frontier_navigations,
            **{f"graph_{k}": v for k, v in graph_stats.items()},
        }


# ──────────────────────────────────────────────────────────
# Game runner
# ──────────────────────────────────────────────────────────

def run_agent(
    game_id: str = "ls20",
    max_actions: int = 5000,
    verbose: bool = True,
    device: str = "cuda",
    checkpoint: str = "checkpoints/understanding/best.pt",
):
    """Run the understanding agent on an ARC-AGI-3 game."""
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
    if verbose:
        print(f"\n{'='*60}")
        print(f"Understanding Agent v3.2 — {game_id}")
        print(f"{'='*60}")
        print(f"Checkpoint: {checkpoint}")
        print(f"Device: {device}")

    agent = UnderstandingAgent(device=device, checkpoint_path=checkpoint)

    if verbose:
        params = agent.model.count_params()
        print(f"Model params: {params['total']:,} total")
        print(f"TTT params: {agent.ttt.trainable_param_count():,} (LoRA)")

    # Create game
    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    available = env.observation_space.available_actions
    agent.setup(list(available))

    # Build action map
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
    prev_understanding_runs = 0

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

        # Track level completions
        if observation.levels_completed > levels_completed:
            levels_completed = observation.levels_completed
            agent.on_level_complete()
            if verbose:
                print(f"  *** Level {levels_completed} completed! (action {action_count}) ***")

        frame = np.array(observation.frame[0])

        action_type, x, y = agent.act(frame)

        # Execute action
        game_action = action_map.get(action_type)
        if game_action is None:
            game_action = GameAction.RESET

        if x is not None and y is not None:
            env.step(game_action, data={"x": x, "y": y})
        else:
            env.step(game_action)

        # Print understanding updates
        if verbose and agent.understanding_runs > prev_understanding_runs:
            u = agent.understanding
            print(
                f"  [Understanding #{agent.understanding_runs} @ step {action_count}] "
                f"type={u.game_type} conf={u.confidence:.2f} "
                f"player={u.player_colors} walls={u.wall_colors} "
                f"collect={u.collectible_colors}"
            )
            if u.movement_map:
                moves = {a: (round(dx, 1), round(dy, 1))
                         for a, (dx, dy) in u.movement_map.items()}
                print(f"    movement={moves}")
            prev_understanding_runs = agent.understanding_runs

        # Periodic status
        if verbose and action_count % 500 == 0 and action_count > 0:
            stats = agent.get_stats()
            avg_ms = elapsed * 1000 / action_count
            print(
                f"Step {action_count}: "
                f"nodes={stats['graph_nodes']} "
                f"tested={stats['graph_tested']} "
                f"dead={stats['graph_dead']} | "
                f"ttt={stats['ttt_updates']} "
                f"understand={stats['understanding_runs']} "
                f"frontier={stats['frontier_navs']} | "
                f"{avg_ms:.1f}ms/act"
            )
            if stats['empirical_change']:
                print(f"  empirical_change={stats['empirical_change']}")

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
        print(f"Understanding: type={stats['game_type']} "
              f"conf={stats['confidence']:.2f}")
        print(f"  Player: {stats['player']}")
        print(f"  Walls: {stats['walls']}")
        print(f"  Collectibles: {stats['collectibles']}")
        print(f"  Movement: {stats['movement']}")
        print(f"Graph: {stats['graph_nodes']} nodes, "
              f"{stats['graph_tested']} tested, {stats['graph_dead']} dead")
        print(f"Model: {stats['understanding_runs']} understanding runs, "
              f"{stats['ttt_updates']} TTT updates")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(description="Understanding agent (v3.2)")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=5000)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", default="checkpoints/understanding/best.pt")
    args = parser.parse_args()

    run_agent(
        game_id=args.game,
        max_actions=args.max_actions,
        verbose=not args.quiet,
        device=args.device,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
