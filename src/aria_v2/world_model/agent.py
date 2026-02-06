#!/usr/bin/env python3
"""
World Model Agent: uses trained SmolLM2 for game understanding and action selection.

Three capabilities from one model:
1. World model: predict next frame, measure surprise
2. Goal inference: probe LEVEL_COMPLETE probability per action
3. Action selection: goal-directed / exploration / policy

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

from .config import AgentConfig, WorldModelConfig
from .game_transformer import create_game_transformer
from ..tokenizer.frame_tokenizer import FrameVQVAE, VQVAEConfig
from ..tokenizer.trajectory_dataset import (
    VQ_OFFSET, ACT_OFFSET, FRAME_TOKEN, ACT_TOKEN,
    LEVEL_COMPLETE, GAME_START,
)


class WorldModelAgent:
    """
    Agent that uses a trained world model for action selection.

    Decision loop (no thresholds, uses relative comparisons):
        surprise = model_nll(actual_frame) vs surprise_ema
        goal_scores = [P(LEVEL_COMPLETE | action=a) for a in actions]
        policy_probs = model_action_distribution(context)

        if max(goal_scores) > 2 * mean(goal_scores):  # one action clearly better
            action = argmax(goal_scores)
        elif surprise > 2 * surprise_ema:               # something unexpected
            action = argmax(prediction_entropy)
        else:
            action = sample(policy_probs)
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

        # Load world model
        print("Loading world model...")
        wm_ckpt = torch.load(
            self.config.world_model_checkpoint, weights_only=False, map_location=device
        )
        model_config = wm_ckpt.get("model_config", WorldModelConfig())
        self.model = create_game_transformer(model_config)
        self.model.load_state_dict(wm_ckpt["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

        # Context buffer (token IDs)
        self.context: list[int] = [GAME_START]
        self.max_context_tokens = self.config.max_context_frames * 67  # ~67 tokens per frame

        # Surprise tracking
        self.surprise_ema = 1.0
        self.surprise_history: deque[float] = deque(maxlen=100)

        # Stats
        self.step_count = 0
        self.goal_directed_count = 0
        self.exploration_count = 0
        self.policy_count = 0
        self.last_decision_reason = ""

    def act(self, frame: np.ndarray, level_completed: bool = False) -> int:
        """
        Choose an action given the current frame.

        Args:
            frame: [64, 64] numpy array with values 0-15
            level_completed: True if a level was just completed

        Returns:
            Action ID (0-6)
        """
        self.step_count += 1

        # Encode frame to VQ tokens
        frame_tensor = torch.tensor(frame, dtype=torch.long).unsqueeze(0).to(self.device)
        vq_indices = self.vqvae.encode(frame_tensor)  # [1, 8, 8]
        frame_codes = vq_indices[0].flatten().tolist()  # 64 codes

        # Insert LEVEL_COMPLETE if applicable
        if level_completed:
            self.context.append(LEVEL_COMPLETE)

        # Add frame to context
        self.context.append(FRAME_TOKEN)
        for code in frame_codes:
            self.context.append(VQ_OFFSET + code)

        # Trim context if too long
        if len(self.context) > self.max_context_tokens:
            # Keep GAME_START + trim from front
            excess = len(self.context) - self.max_context_tokens
            self.context = [GAME_START] + self.context[excess + 1:]

        # Compute surprise for this frame
        surprise = self._compute_surprise()

        # Compute goal scores for each candidate action
        goal_scores = self._compute_goal_scores()

        # Get policy distribution
        policy_probs = self._get_policy_distribution()

        # Decision
        action = self._select_action(surprise, goal_scores, policy_probs)

        # Add action to context
        self.context.append(ACT_TOKEN)
        self.context.append(ACT_OFFSET + action)

        return action

    @torch.no_grad()
    def _compute_surprise(self) -> float:
        """Compute how surprised the model is by the current frame."""
        if len(self.context) < 70:  # Need at least one prior frame
            return 0.0

        # Get model prediction for current frame tokens
        # We look at the last 65 tokens (FRAME + 64 VQ codes)
        ctx = torch.tensor(self.context[:-65], dtype=torch.long).unsqueeze(0).to(self.device)
        target = torch.tensor(self.context[-65:], dtype=torch.long).unsqueeze(0).to(self.device)

        # Only predict if context is long enough
        if ctx.shape[1] < 2:
            return 0.0

        # Combine and get logits
        full_input = torch.cat([ctx, target[:, :-1]], dim=1)
        outputs = self.model(input_ids=full_input)
        logits = outputs.logits

        # NLL of the actual frame tokens
        frame_logits = logits[:, -65:, :]  # Predictions for the frame tokens
        frame_targets = target
        loss = F.cross_entropy(
            frame_logits.reshape(-1, frame_logits.shape[-1]),
            frame_targets.reshape(-1),
        )
        surprise = loss.item()

        # Update EMA
        self.surprise_ema = (
            self.config.surprise_ema_decay * self.surprise_ema
            + (1 - self.config.surprise_ema_decay) * surprise
        )
        self.surprise_history.append(surprise)

        return surprise

    @torch.no_grad()
    def _compute_goal_scores(self) -> list[float]:
        """
        For each candidate action, compute P(LEVEL_COMPLETE) in next few tokens.
        """
        scores = []
        base_ctx = self.context.copy()

        for action_id in range(self.config.num_candidate_actions):
            # Extend context with this action
            extended = base_ctx + [ACT_TOKEN, ACT_OFFSET + action_id]
            ctx_tensor = torch.tensor(extended, dtype=torch.long).unsqueeze(0).to(self.device)

            # Get prediction for next token
            outputs = self.model(input_ids=ctx_tensor)
            next_logits = outputs.logits[:, -1, :]  # [1, V]

            # P(LEVEL_COMPLETE) as next token
            probs = F.softmax(next_logits, dim=-1)
            p_level = probs[0, LEVEL_COMPLETE].item()

            # Also check P(LEVEL_COMPLETE) after predicted next frame
            # This is more expensive but more accurate - skip for now
            scores.append(p_level)

        return scores

    @torch.no_grad()
    def _get_policy_distribution(self) -> list[float]:
        """Get model's action probability distribution given current context."""
        # Context should end with frame tokens (before action)
        ctx_tensor = torch.tensor(
            self.context + [ACT_TOKEN], dtype=torch.long
        ).unsqueeze(0).to(self.device)

        outputs = self.model(input_ids=ctx_tensor)
        logits = outputs.logits[:, -1, :]  # [1, V]

        # Extract action token probabilities
        action_logits = logits[0, ACT_OFFSET:ACT_OFFSET + self.config.num_candidate_actions]
        probs = F.softmax(action_logits / self.config.temperature, dim=-1)
        return probs.cpu().tolist()

    def _select_action(
        self, surprise: float, goal_scores: list[float], policy_probs: list[float]
    ) -> int:
        """
        Select action based on surprise, goal scores, and policy.

        No magic thresholds - uses relative comparisons.
        """
        max_goal = max(goal_scores)
        mean_goal = sum(goal_scores) / len(goal_scores) if goal_scores else 0

        # 1. Goal-directed: one action clearly leads toward LEVEL_COMPLETE
        if mean_goal > 0 and max_goal > self.config.goal_threshold_factor * mean_goal:
            self.goal_directed_count += 1
            action = int(np.argmax(goal_scores))
            self.last_decision_reason = f"goal-directed (score={max_goal:.4f}, mean={mean_goal:.4f})"
            return action

        # 2. Exploration: something unexpected happened
        if surprise > self.config.surprise_threshold_factor * self.surprise_ema and self.step_count > 5:
            self.exploration_count += 1
            # Choose action that maximizes prediction entropy (most uncertain outcome)
            action = self._most_uncertain_action()
            self.last_decision_reason = f"explore (surprise={surprise:.2f}, ema={self.surprise_ema:.2f})"
            return action

        # 3. Policy: follow learned distribution
        self.policy_count += 1
        probs = np.array(policy_probs)
        probs = probs / probs.sum()  # Renormalize
        action = np.random.choice(len(probs), p=probs)
        self.last_decision_reason = f"policy (top={probs.max():.2f})"
        return int(action)

    @torch.no_grad()
    def _most_uncertain_action(self) -> int:
        """Find action that leads to most uncertain next state (max entropy)."""
        entropies = []
        base_ctx = self.context.copy()

        for action_id in range(self.config.num_candidate_actions):
            extended = base_ctx + [ACT_TOKEN, ACT_OFFSET + action_id, FRAME_TOKEN]
            ctx_tensor = torch.tensor(extended, dtype=torch.long).unsqueeze(0).to(self.device)

            outputs = self.model(input_ids=ctx_tensor)
            logits = outputs.logits[:, -1, :]

            # Entropy of next VQ token prediction
            probs = F.softmax(logits[0, VQ_OFFSET:VQ_OFFSET + 512], dim=-1)
            entropy = -(probs * (probs + 1e-10).log()).sum().item()
            entropies.append(entropy)

        return int(np.argmax(entropies))

    def get_stats(self) -> dict:
        """Return agent statistics."""
        total = max(self.goal_directed_count + self.exploration_count + self.policy_count, 1)
        return {
            "steps": self.step_count,
            "goal_directed": self.goal_directed_count,
            "exploration": self.exploration_count,
            "policy": self.policy_count,
            "goal_directed_pct": self.goal_directed_count / total * 100,
            "exploration_pct": self.exploration_count / total * 100,
            "policy_pct": self.policy_count / total * 100,
            "surprise_ema": self.surprise_ema,
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
        print(f"Playing {game_id} with World Model Agent")
        print(f"{'='*60}")

    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    available = env.observation_space.available_actions
    action_map = {}
    for i in range(1, 7):
        if i in available:
            action_map[i] = getattr(GameAction, f"ACTION{i}")

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

        agent_action = agent.act(frame, level_completed=level_completed)
        game_action = action_map.get(agent_action, GameAction.ACTION1)

        if verbose and action_count % 20 == 0:
            stats = agent.get_stats()
            print(
                f"Step {action_count}: action={agent_action} | "
                f"{stats['last_reason']} | "
                f"ctx={stats['context_tokens']} tokens"
            )

        env.step(game_action)
        action_count += 1

    duration = time.time() - start_time

    if verbose:
        stats = agent.get_stats()
        print(f"\n{'='*60}")
        print("Results")
        print(f"{'='*60}")
        print(f"Actions: {action_count}")
        print(f"Levels: {levels_completed}")
        print(f"Time: {duration:.1f}s")
        print(f"Decision breakdown:")
        print(f"  Goal-directed: {stats['goal_directed']} ({stats['goal_directed_pct']:.1f}%)")
        print(f"  Exploration: {stats['exploration']} ({stats['exploration_pct']:.1f}%)")
        print(f"  Policy: {stats['policy']} ({stats['policy_pct']:.1f}%)")
        print(f"Surprise EMA: {stats['surprise_ema']:.3f}")

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
    parser.add_argument("--vqvae", default="checkpoints/vqvae/best.pt")
    args = parser.parse_args()

    config = AgentConfig(
        world_model_checkpoint=args.world_model,
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
