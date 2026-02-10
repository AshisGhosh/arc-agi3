#!/usr/bin/env python3
"""
Planning agent with KV-cache lookahead.

Instead of behavioral cloning (policy predicts action from masked context),
this agent uses the world model as a simulator: for each candidate action,
it imagines the consequences by extending the KV cache and scores them by
P(LEVEL_COMPLETE) + frame_change magnitude.

Architecture:
    1. Encode frame via VQ-VAE
    2. Forward full context with use_cache=True → base KV cache
    3. Read ACT_TYPE prior from lm_head → top-K types
    4. For each candidate type, extend cache, read ACT_LOC prior → top-K locs
    5. For each (type, loc) candidate, score by P(LEVEL_COMPLETE) + frame_change
    6. Pick highest-scoring action

No policy heads needed — the lm_head does everything.

Usage:
    uv run python -m src.aria_v2.world_model.planning_agent --game ls20
"""

import argparse
import os
import sys
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from ..tokenizer.frame_tokenizer import FrameVQVAE
from ..tokenizer.trajectory_dataset import (
    ACT_LOC_OFFSET,
    ACT_TOKEN,
    ACT_TYPE_OFFSET,
    FRAME_TOKEN,
    GAME_START,
    LEVEL_COMPLETE,
    VQ_OFFSET,
    vq_cell_to_pixel,
)
from .config import PlanningConfig, WorldModelConfig
from .game_transformer import create_game_transformer


class PlanningAgent:
    """
    Agent that uses world model lookahead to select actions.

    For each candidate action, feeds [ACT] <TYPE> <LOC> into the world model
    using cached KV states, then scores by:
        score = alpha * P(LEVEL_COMPLETE) + beta * frame_change + gamma * type_prior
    """

    def __init__(self, config: PlanningConfig | None = None, device: str = "cuda"):
        self.config = config or PlanningConfig()
        self.device = device

        # Load VQ-VAE
        print("Loading VQ-VAE...")
        vqvae_ckpt = torch.load(
            self.config.vqvae_checkpoint, weights_only=False, map_location=device
        )
        self.vqvae = FrameVQVAE(vqvae_ckpt["config"]).to(device)
        self.vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
        self.vqvae.eval()

        # Load world model backbone
        print("Loading world model backbone...")
        wm_ckpt = torch.load(
            self.config.world_model_checkpoint, weights_only=False, map_location=device
        )
        model_config = wm_ckpt.get("model_config", WorldModelConfig())
        self.backbone = create_game_transformer(model_config)
        self.backbone.load_state_dict(wm_ckpt["model_state_dict"])
        self.backbone = self.backbone.to(device)
        self.backbone.eval()

        # Context buffer (token IDs)
        self.context: list[int] = [GAME_START]
        self.max_context_tokens = self.config.max_context_frames * 69

        # Current frame VQ codes (for frame_change scoring)
        self.current_frame_codes: list[int] | None = None

        # Recent actions for anti-repetition
        self.recent_actions: deque[tuple[int, int]] = deque(
            maxlen=self.config.repeat_window
        )

        # Stats
        self.step_count = 0
        self.last_scores: list[tuple[int, int, float]] = []  # (type, loc, score)

    @torch.no_grad()
    def act(
        self,
        frame: np.ndarray,
        available_actions: list[int] | None = None,
        level_completed: bool = False,
    ) -> tuple[int, int | None, int | None]:
        """
        Choose an action via KV-cache lookahead planning.

        Args:
            frame: [64, 64] numpy array with values 0-15
            available_actions: list of available action type IDs (1-indexed)
            level_completed: True if a level was just completed

        Returns:
            (action_type, x, y) where action_type is 1-indexed, x/y are pixel coords or None
        """
        self.step_count += 1

        # 1. Encode frame to VQ tokens
        frame_tensor = torch.tensor(frame, dtype=torch.long).unsqueeze(0).to(self.device)
        vq_indices = self.vqvae.encode(frame_tensor)  # [1, 8, 8]
        frame_codes = vq_indices[0].flatten().tolist()  # 64 codes
        self.current_frame_codes = frame_codes

        # 2. Insert LEVEL_COMPLETE if applicable
        if level_completed:
            self.context.append(LEVEL_COMPLETE)

        # 3. Add frame to context: [FRAME] vq_0..vq_63
        self.context.append(FRAME_TOKEN)
        for code in frame_codes:
            self.context.append(VQ_OFFSET + code)

        # 4. Trim context if too long
        if len(self.context) > self.max_context_tokens:
            excess = len(self.context) - self.max_context_tokens
            self.context = [GAME_START] + self.context[excess + 1:]

        # 5. Forward full context → base KV cache
        ctx_tensor = torch.tensor(
            self.context, dtype=torch.long
        ).unsqueeze(0).to(self.device)

        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            base_cache = DynamicCache()
            self.backbone(
                input_ids=ctx_tensor,
                past_key_values=base_cache,
                use_cache=True,
            )
        # base_cache is updated in-place. After last VQ token, model predicts [ACT].

        # 6. Feed [ACT] token to get type priors
        # Sequence: ...vq_63 → [ACT] → ACT_TYPE_i → ACT_LOC_j → [FRAME]/[LEVEL_COMPLETE]
        act_cache = deepcopy(base_cache)
        act_input = torch.tensor([[ACT_TOKEN]], dtype=torch.long).to(self.device)
        with torch.amp.autocast("cuda", dtype=amp_dtype):
            act_outputs = self.backbone(
                input_ids=act_input,
                past_key_values=act_cache,
                use_cache=True,
            )
        act_logits = act_outputs.logits[0, -1].float()  # [vocab] — after [ACT], predicts TYPE

        # 7. Get top-K action type candidates from lm_head
        available_type_mask = self._build_type_mask(available_actions)
        type_logits = act_logits[ACT_TYPE_OFFSET:ACT_TYPE_OFFSET + 8].clone()
        type_logits[~available_type_mask] = float("-inf")
        type_priors = F.softmax(type_logits / self.config.temperature, dim=-1)
        k_types = min(self.config.top_k_types, available_type_mask.sum().item())
        if k_types == 0:
            k_types = 1  # fallback
        top_type_values, top_type_indices = torch.topk(type_priors, k=int(k_types))

        # 8. For each candidate type, branch and score locations
        candidates: list[tuple[int, int, float]] = []

        for t_idx in range(len(top_type_indices)):
            action_type = top_type_indices[t_idx].item()
            type_prior = top_type_values[t_idx].item()

            # Feed <ACT_TYPE_i> using branched act_cache
            type_cache = deepcopy(act_cache)
            type_token = torch.tensor(
                [[ACT_TYPE_OFFSET + action_type]],
                dtype=torch.long,
            ).to(self.device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                type_outputs = self.backbone(
                    input_ids=type_token,
                    past_key_values=type_cache,
                    use_cache=True,
                )
            type_logits_after = type_outputs.logits[0, -1].float()  # predicts LOC

            # Get top-K location candidates from lm_head
            loc_logits = type_logits_after[ACT_LOC_OFFSET:ACT_LOC_OFFSET + 65].clone()
            loc_priors = F.softmax(loc_logits / self.config.temperature, dim=-1)
            k_locs = min(self.config.top_k_locs, 65)
            top_loc_values, top_loc_indices = torch.topk(loc_priors, k=k_locs)

            for l_idx in range(len(top_loc_indices)):
                action_loc = top_loc_indices[l_idx].item()

                # Feed <ACT_LOC_j> using branched type cache
                loc_cache = deepcopy(type_cache)
                loc_token = torch.tensor(
                    [[ACT_LOC_OFFSET + action_loc]],
                    dtype=torch.long,
                ).to(self.device)

                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    loc_outputs = self.backbone(
                        input_ids=loc_token,
                        past_key_values=loc_cache,
                        use_cache=True,
                    )
                loc_logits_after = loc_outputs.logits[0, -1].float()  # predicts FRAME/LEVEL

                # Score this candidate
                score = self._score_candidate(
                    loc_logits_after, action_type, action_loc, type_prior
                )
                candidates.append((action_type, action_loc, score))

        # 9. Pick best candidate
        candidates.sort(key=lambda c: c[2], reverse=True)
        self.last_scores = candidates[:10]

        best_type, best_loc, best_score = candidates[0]

        # 10. Append chosen action to real context
        self.context.append(ACT_TOKEN)
        self.context.append(ACT_TYPE_OFFSET + best_type)
        self.context.append(ACT_LOC_OFFSET + best_loc)

        # Track for anti-repetition
        self.recent_actions.append((best_type, best_loc))

        # 11. Convert to game API format
        if best_loc < 64:
            x, y = vq_cell_to_pixel(best_loc)
        else:
            x, y = None, None

        # Action type already matches game API (1-indexed in training data)
        return best_type, x, y

    def _build_type_mask(self, available_actions: list[int] | None) -> torch.Tensor:
        """Build a boolean mask for available action types.

        Action type indices match the game API (1-indexed in JSONL data),
        stored directly as ACT_TYPE_OFFSET + action_id in the token vocabulary.
        Index 0 exists but is rarely used (reset/no-op in some games).
        """
        mask = torch.ones(8, dtype=torch.bool, device=self.device)
        if available_actions is not None:
            mask[:] = False
            for a in available_actions:
                if 0 <= a < 8:
                    mask[a] = True
        return mask

    def _score_candidate(
        self,
        logits_after_loc: torch.Tensor,
        action_type: int,
        action_loc: int,
        type_prior: float,
    ) -> float:
        """
        Score a candidate action based on world model predictions.

        Components:
            1. P(LEVEL_COMPLETE): probability the model predicts level completion
            2. Frame change: how many VQ codes change in the predicted next frame
            3. Type prior: how likely the lm_head considered this action type
            4. Anti-repetition: penalty for recently used actions
        """
        cfg = self.config

        # 1. P(LEVEL_COMPLETE) — read from logits at the position after ACT_LOC
        # The model should predict either FRAME or LEVEL_COMPLETE next
        next_token_probs = F.softmax(logits_after_loc, dim=-1)
        p_level_complete = next_token_probs[LEVEL_COMPLETE].item()

        # 2. Frame change — predict next frame's VQ codes and compare
        # The logits here predict what comes next. If FRAME is next, the subsequent
        # 64 tokens would be VQ codes. We can read the VQ logits directly.
        vq_logits = logits_after_loc[VQ_OFFSET:VQ_OFFSET + 512]
        # Simple heuristic: count how many of the top VQ predictions differ
        # from the current frame. We only have logits for the first VQ position,
        # so we compare against the first VQ code of the current frame.
        frame_change = 0.0
        if self.current_frame_codes is not None:
            # The position after ACT_LOC should predict FRAME token, then VQ codes.
            # Since we're at the FRAME-prediction position, the VQ logits represent
            # the model's belief about the next frame's content.
            # Use VQ logit distribution entropy as a proxy for expected change.
            vq_probs = F.softmax(vq_logits / self.config.temperature, dim=-1)
            current_code = self.current_frame_codes[0]
            # P(different from current) = 1 - P(same as current)
            frame_change = 1.0 - vq_probs[current_code].item()

        # 3. Type prior from the base context
        type_score = type_prior

        # 4. Anti-repetition penalty
        repeat_penalty = 0.0
        for recent_type, recent_loc in self.recent_actions:
            if recent_type == action_type and recent_loc == action_loc:
                repeat_penalty += cfg.repeat_penalty

        score = (
            cfg.level_complete_weight * p_level_complete
            + cfg.frame_change_weight * frame_change
            + cfg.type_prior_weight * type_score
            - repeat_penalty
        )
        return score

    def get_stats(self) -> dict:
        """Return agent statistics."""
        return {
            "steps": self.step_count,
            "context_tokens": len(self.context),
            "last_scores": self.last_scores[:5],
        }


def run_agent(
    game_id: str = "ls20",
    max_actions: int = 500,
    verbose: bool = True,
    config: PlanningConfig | None = None,
):
    """Run the planning agent on an ARC-AGI-3 game."""
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

    agent = PlanningAgent(config=config)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing {game_id} with Planning Agent (KV-cache lookahead)")
        print(f"{'='*60}")

    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    available = env.observation_space.available_actions
    action_map = {}
    for i in range(1, 8):
        if i in available:
            action_map[i] = getattr(GameAction, f"ACTION{i}", None)

    start_time = time.time()
    action_count = 0
    levels_completed = 0
    step_times: list[float] = []

    while action_count < max_actions:
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

        frame = np.array(observation.frame[0])
        level_completed = observation.levels_completed > levels_completed
        if level_completed:
            levels_completed = observation.levels_completed
            if verbose:
                print(f"  Level {levels_completed} completed!")

        step_start = time.time()
        action_type, x, y = agent.act(
            frame,
            available_actions=list(available),
            level_completed=level_completed,
        )
        step_time = time.time() - step_start
        step_times.append(step_time)

        # Execute action
        game_action = action_map.get(action_type)
        if game_action is None:
            game_action = GameAction.ACTION1

        if x is not None and y is not None:
            env.step(game_action, data={"x": x, "y": y})
        else:
            env.step(game_action)

        if verbose and action_count % 10 == 0:
            stats = agent.get_stats()
            loc_str = f"({x},{y})" if x is not None else "null"
            top_scores = stats["last_scores"]
            score_str = ""
            if top_scores:
                best = top_scores[0]
                score_str = f" score={best[2]:.3f}"
            print(
                f"Step {action_count}: type={action_type} loc={loc_str}{score_str} | "
                f"{step_time*1000:.0f}ms | ctx={stats['context_tokens']} tokens"
            )

        action_count += 1

    duration = time.time() - start_time
    avg_step = (sum(step_times) / len(step_times) * 1000) if step_times else 0

    if verbose:
        stats = agent.get_stats()
        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"Actions: {action_count}")
        print(f"Levels: {levels_completed}")
        print(f"Time: {duration:.1f}s ({avg_step:.0f}ms/action avg)")
        print(f"Context: {stats['context_tokens']} tokens")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(description="Run planning agent (KV-cache lookahead)")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=500)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--world-model", default="checkpoints/world_model/best.pt")
    parser.add_argument("--vqvae", default="checkpoints/vqvae/best.pt")
    parser.add_argument("--top-k-types", type=int, default=3)
    parser.add_argument("--top-k-locs", type=int, default=3)
    parser.add_argument("--level-weight", type=float, default=10.0)
    parser.add_argument("--frame-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.5)
    args = parser.parse_args()

    config = PlanningConfig(
        world_model_checkpoint=args.world_model,
        vqvae_checkpoint=args.vqvae,
        top_k_types=args.top_k_types,
        top_k_locs=args.top_k_locs,
        level_complete_weight=args.level_weight,
        frame_change_weight=args.frame_weight,
        type_prior_weight=args.prior_weight,
    )
    run_agent(
        game_id=args.game,
        max_actions=args.max_actions,
        verbose=not args.quiet,
        config=config,
    )


if __name__ == "__main__":
    main()
