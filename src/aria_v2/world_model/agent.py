#!/usr/bin/env python3
"""
Game-agnostic agent with separated world model and policy.

Architecture:
    World Model (SmolLM2 + LoRA): maintains full context with action tokens,
        provides dynamics prediction and surprise measurement.
    Policy (ActionTypeHead + ActionLocationHead): operates on MASKED context
        where action tokens are replaced with MASK. Cannot copy actions.

Every action = (type, location):
    - type: which action (0-7)
    - location: where (0-63 = VQ cell, 64 = NULL for non-spatial)

Usage:
    uv run python -m src.aria_v2.world_model.agent --game ls20
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from .config import AgentConfig, WorldModelConfig, PolicyConfig
from .game_transformer import create_game_transformer
from .policy_head import PolicyHeads
from ..tokenizer.frame_tokenizer import FrameVQVAE
from ..tokenizer.trajectory_dataset import (
    VQ_OFFSET, ACT_TYPE_OFFSET, ACT_LOC_OFFSET, ACT_LOC_NULL,
    FRAME_TOKEN, ACT_TOKEN, LEVEL_COMPLETE, GAME_START,
    MASK_TOKEN, vq_cell_to_pixel,
)


class WorldModelAgent:
    """
    Agent with separated dynamics (world model) and policy (masked heads).

    Inference loop:
        1. Encode frame via VQ-VAE
        2. Update world model context (includes real action tokens)
        3. Create masked context (action tokens → MASK)
        4. Run backbone on masked context
        5. Policy heads predict (action_type, action_location)
        6. Convert to game API call
    """

    def __init__(self, config: AgentConfig | None = None, device: str = "cuda"):
        self.config = config or AgentConfig()
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

        # Load policy heads
        print("Loading policy heads...")
        policy_ckpt = torch.load(
            self.config.policy_checkpoint, weights_only=False, map_location=device
        )
        policy_config = policy_ckpt.get("policy_config", PolicyConfig())
        self.policy = PolicyHeads(policy_config).to(device)
        self.policy.load_state_dict(policy_ckpt["policy_state_dict"])
        self.policy.eval()

        # Context buffer (token IDs — full context with action tokens)
        self.context: list[int] = [GAME_START]
        self.context_types: list[str] = ["start"]
        self.max_context_tokens = self.config.max_context_frames * 69

        # Surprise tracking
        self.surprise_ema = 1.0
        self.surprise_history: deque[float] = deque(maxlen=100)

        # Stats
        self.step_count = 0
        self.goal_directed_count = 0
        self.exploration_count = 0
        self.policy_count = 0
        self.last_decision_reason = ""

    def act(
        self,
        frame: np.ndarray,
        available_actions: list[int] | None = None,
        level_completed: bool = False,
    ) -> tuple[int, int | None, int | None]:
        """
        Choose an action given the current frame.

        Args:
            frame: [64, 64] numpy array with values 0-15
            available_actions: list of available action type IDs (1-indexed from game API)
            level_completed: True if a level was just completed

        Returns:
            (action_type, x, y) where x,y are pixel coords or None for non-spatial
        """
        self.step_count += 1

        # Encode frame to VQ tokens
        frame_tensor = torch.tensor(frame, dtype=torch.long).unsqueeze(0).to(self.device)
        vq_indices = self.vqvae.encode(frame_tensor)  # [1, 8, 8]
        frame_codes = vq_indices[0].flatten().tolist()  # 64 codes

        # Insert LEVEL_COMPLETE if applicable
        if level_completed:
            self.context.append(LEVEL_COMPLETE)
            self.context_types.append("level")

        # Add frame to context
        self.context.append(FRAME_TOKEN)
        self.context_types.append("frame")
        for code in frame_codes:
            self.context.append(VQ_OFFSET + code)
            self.context_types.append("vq")

        # Trim context if too long
        if len(self.context) > self.max_context_tokens:
            excess = len(self.context) - self.max_context_tokens
            self.context = [GAME_START] + self.context[excess + 1:]
            self.context_types = ["start"] + self.context_types[excess + 1:]

        # Create masked context for policy
        masked_context = self._mask_actions(self.context, self.context_types)

        # Run backbone on masked context, get hidden states
        action_type, action_loc = self._policy_forward(masked_context)

        # Mask to available actions if provided
        if available_actions is not None:
            # available_actions are 1-indexed from game API
            available_set = set(a - 1 for a in available_actions if a > 0)  # Convert to 0-indexed
            if action_type not in available_set and available_set:
                action_type = min(available_set)  # Fallback

        # Convert location to pixel coordinates
        if action_loc < 64:
            x, y = vq_cell_to_pixel(action_loc)
        else:
            x, y = None, None

        # Update world model context with chosen action
        self.context.append(ACT_TOKEN)
        self.context_types.append("act")
        self.context.append(ACT_TYPE_OFFSET + action_type)
        self.context_types.append("action_type")
        self.context.append(ACT_LOC_OFFSET + action_loc)
        self.context_types.append("action_loc")

        self.policy_count += 1
        self.last_decision_reason = f"policy (type={action_type}, loc={action_loc})"

        # Return 1-indexed action type for game API
        return action_type + 1, x, y

    def _mask_actions(self, tokens: list[int], types: list[str]) -> list[int]:
        """Replace action tokens with MASK in context."""
        masked = list(tokens)
        for i, tt in enumerate(types):
            if tt in ("act", "action_type", "action_loc"):
                masked[i] = MASK_TOKEN
        return masked

    @torch.no_grad()
    def _policy_forward(self, masked_context: list[int]) -> tuple[int, int]:
        """Run policy heads on masked context.

        Returns (action_type, action_loc) as integer indices.
        """
        ctx_tensor = torch.tensor(masked_context, dtype=torch.long).unsqueeze(0).to(self.device)

        use_amp = torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_amp else torch.float32

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            # Get input embeddings and inject mask embedding
            input_embeds = self.backbone.get_input_embeddings()(ctx_tensor)
            mask_positions = (ctx_tensor == MASK_TOKEN)
            input_embeds[mask_positions] = self.policy.mask_embedding.to(input_embeds.dtype)

            # Forward through backbone
            outputs = self.backbone(inputs_embeds=input_embeds, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # [1, T, 960]

        # Find last VQ token position (= last frame boundary for policy)
        vq_positions = []
        last_vq_pos = -1
        for i, tt in enumerate(self.context_types):
            if tt == "vq":
                vq_positions.append(i)
                last_vq_pos = i

        if last_vq_pos < 0 or len(vq_positions) < 64:
            # Not enough context, return default
            return 0, 64  # NULL location

        # Get the 64 VQ positions from the LAST frame
        last_frame_vq = vq_positions[-64:]

        h_frame = hidden_states[0, last_vq_pos].float()  # [960]
        h_vq_cells = hidden_states[0, last_frame_vq].float()  # [64, 960]

        # Policy forward
        type_logits, loc_logits = self.policy(
            h_frame.unsqueeze(0), h_vq_cells.unsqueeze(0)
        )

        # Sample actions
        type_probs = F.softmax(type_logits[0] / self.config.temperature, dim=-1)
        action_type = torch.multinomial(type_probs, 1).item()

        loc_probs = F.softmax(loc_logits[0] / self.config.temperature, dim=-1)
        action_loc = torch.multinomial(loc_probs, 1).item()

        return action_type, action_loc

    def get_stats(self) -> dict:
        """Return agent statistics."""
        total = max(self.policy_count, 1)
        return {
            "steps": self.step_count,
            "policy": self.policy_count,
            "last_reason": self.last_decision_reason,
            "context_tokens": len(self.context),
        }


def run_agent(
    game_id: str = "ls20",
    max_actions: int = 500,
    verbose: bool = True,
    agent_config: AgentConfig | None = None,
):
    """Run the world model agent on an ARC-AGI-3 game."""
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
    agent = WorldModelAgent(config=agent_config)

    # Create game
    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing {game_id} with World Model Agent (v2: separated policy)")
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

        action_type, x, y = agent.act(
            frame,
            available_actions=list(available),
            level_completed=level_completed,
        )

        # Execute action
        game_action = action_map.get(action_type)
        if game_action is None:
            game_action = GameAction.ACTION1

        if x is not None and y is not None:
            env.step(game_action, data={"x": x, "y": y})
        else:
            env.step(game_action)

        if verbose and action_count % 20 == 0:
            stats = agent.get_stats()
            loc_str = f"({x},{y})" if x is not None else "null"
            print(
                f"Step {action_count}: type={action_type} loc={loc_str} | "
                f"{stats['last_reason']} | ctx={stats['context_tokens']} tokens"
            )

        action_count += 1

    duration = time.time() - start_time

    if verbose:
        stats = agent.get_stats()
        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"Actions: {action_count}")
        print(f"Levels: {levels_completed}")
        print(f"Time: {duration:.1f}s ({duration/action_count*1000:.0f}ms/action)")
        print(f"Context: {stats['context_tokens']} tokens")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(description="Run world model agent")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=500)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--world-model", default="checkpoints/world_model/best.pt")
    parser.add_argument("--policy", default="checkpoints/policy/best.pt")
    parser.add_argument("--vqvae", default="checkpoints/vqvae/best.pt")
    args = parser.parse_args()

    config = AgentConfig(
        world_model_checkpoint=args.world_model,
        policy_checkpoint=args.policy,
        vqvae_checkpoint=args.vqvae,
    )
    run_agent(
        game_id=args.game,
        max_actions=args.max_actions,
        verbose=not args.quiet,
        agent_config=config,
    )


if __name__ == "__main__":
    main()
