"""
ARIA-Lite Integrated Evaluation

Tests the full dual-system architecture:
1. BC-trained Fast Policy (System 1)
2. World Model for uncertainty
3. Simple deliberation when uncertain (System 2)

This bridges our BC training with the ARIA architecture.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ARIALiteConfig
from .encoder_simple import SimpleGridEncoder
from .fast_policy import FastPolicy
from .world_model import create_world_model

# ARC-AGI imports
try:
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction, GameState
    ARC_AGI_AVAILABLE = True
except ImportError:
    ARC_AGI_AVAILABLE = False


class IntegratedARIAAgent:
    """
    Integrated ARIA agent using:
    - BC-trained encoder + fast policy (from train_aria_bc.py)
    - Trained world model for uncertainty estimation
    - Deliberation when uncertain (sampling with temperature)
    """

    def __init__(
        self,
        bc_checkpoint: str,
        config: ARIALiteConfig,
        device: torch.device,
        grid_size: int = 16,
        world_model_checkpoint: str = None,
    ):
        self.config = config
        self.device = device
        self.grid_size = grid_size

        # Load BC-trained encoder and fast policy
        checkpoint = torch.load(bc_checkpoint, map_location=device, weights_only=False)

        # Create encoder (simple version)
        self.encoder = SimpleGridEncoder(
            num_colors=16,
            embed_dim=32,
            hidden_dim=128,
            output_dim=config.fast_policy.state_dim,
        ).to(device)

        self.fast_policy = FastPolicy(config.fast_policy).to(device)

        # Load weights
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.fast_policy.load_state_dict(checkpoint["fast_policy"])

        # World model for uncertainty
        self.world_model = create_world_model(config).to(device)

        # Load world model if checkpoint provided
        if world_model_checkpoint and Path(world_model_checkpoint).exists():
            wm_ckpt = torch.load(world_model_checkpoint, map_location=device, weights_only=False)
            self.world_model.load_state_dict(wm_ckpt["world_model"])
            print(f"Loaded world model from {world_model_checkpoint}")
            self.world_model_trained = True
        else:
            print("World model not loaded (untrained)")
            self.world_model_trained = False

        # State tracking
        self.state_history = []
        self.action_history = []
        self.last_state = None
        self.last_action = None

        print(f"Loaded BC checkpoint from {bc_checkpoint}")
        print(f"Encoder: {sum(p.numel() for p in self.encoder.parameters()):,} params")
        print(f"Fast Policy: {sum(p.numel() for p in self.fast_policy.parameters()):,} params")
        print(f"World Model: {sum(p.numel() for p in self.world_model.parameters()):,} params")

    def reset(self):
        """Reset state for new episode."""
        self.state_history = []
        self.action_history = []
        self.last_state = None
        self.last_action = None

    def preprocess_obs(self, raw_obs) -> torch.Tensor:
        """Preprocess 64x64 observation to model input."""
        obs = torch.from_numpy(raw_obs).float()
        obs = F.interpolate(
            obs.unsqueeze(0).unsqueeze(0),
            size=(self.grid_size, self.grid_size),
            mode="nearest",
        ).squeeze(0).long().clamp(0, 15).to(self.device)
        return obs

    def act(
        self,
        observation: torch.Tensor,
        epsilon: float = 0.0,
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.2,
    ) -> tuple[int, dict]:
        """
        Select action using dual-system logic with world model uncertainty.

        Returns:
            action_id: Selected action (0-5)
            info: Dict with system used, confidence, uncertainty, etc.
        """
        self.encoder.eval()
        self.fast_policy.eval()
        self.world_model.eval()

        with torch.no_grad():
            # Encode observation
            state = self.encoder(observation)

            # Get fast policy output
            output = self.fast_policy(state)
            action_logits = output.action_logits
            confidence = output.confidence.item()
            action_probs = output.action_probs[0].cpu().numpy()

            # Get world model uncertainty if we have previous state
            uncertainty = 0.0
            if self.world_model_trained and self.last_state is not None and self.last_action is not None:
                # Convert last action to one-hot
                action_onehot = F.one_hot(
                    torch.tensor([self.last_action], device=self.device),
                    num_classes=self.config.world_model.action_dim,
                ).float()

                # Predict and get uncertainty
                wm_output = self.world_model(self.last_state, action_onehot)
                uncertainty = wm_output.uncertainty.item()

                # Also check prediction error
                pred_error = (wm_output.next_state - state).pow(2).mean().item()

        # Determine which system to use
        use_slow = confidence < confidence_threshold

        # Use world model uncertainty
        if self.world_model_trained and uncertainty > uncertainty_threshold:
            use_slow = True

        # Check for loops (same state seen recently)
        state_hash = hash(state.cpu().numpy().tobytes())
        in_loop = state_hash in self.state_history[-10:] if len(self.state_history) >= 2 else False

        if in_loop:
            use_slow = True

        # System 2: Deliberation with world model planning + curiosity
        if use_slow:
            if self.world_model_trained:
                # Use world model to evaluate actions with curiosity bonus
                best_action = None
                best_score = float('-inf')

                for a in range(6):
                    action_onehot = F.one_hot(
                        torch.tensor([a], device=self.device),
                        num_classes=self.config.world_model.action_dim,
                    ).float()

                    wm_out = self.world_model(state, action_onehot)

                    # Score components:
                    # 1. Lower uncertainty = more confident prediction (good)
                    action_unc = wm_out.uncertainty.item()

                    # 2. Check if predicted state was seen before
                    next_state_hash = hash(wm_out.next_state.detach().cpu().numpy().tobytes())

                    # 3. Curiosity bonus: prefer novel states!
                    if next_state_hash in self.state_history:
                        # Seen before - big penalty
                        novelty_bonus = -2.0
                    else:
                        # Novel state - bonus!
                        novelty_bonus = 1.0

                    # 4. Extra penalty for recent states (avoid immediate loops)
                    if next_state_hash in self.state_history[-5:]:
                        novelty_bonus -= 2.0

                    # 5. Policy prior (slight preference for likely actions)
                    policy_score = action_probs[a] if a < len(action_probs) else 0.0

                    # Combined score: novelty-driven with policy prior
                    score = novelty_bonus + 0.3 * policy_score - 0.2 * action_unc

                    if score > best_score:
                        best_score = score
                        best_action = a

                action_id = best_action if best_action is not None else action_logits.argmax(-1).item()
            else:
                # Fallback: sample from distribution with temperature
                temperature = 1.5 if in_loop else 1.2
                logits_scaled = action_logits / temperature
                probs = F.softmax(logits_scaled, dim=-1)[0].cpu().numpy()
                action_id = random.choices(range(len(probs)), weights=probs)[0]

            system = "slow"
        else:
            # System 1: Fast policy argmax
            action_id = action_logits.argmax(-1).item()
            system = "fast"

        # Random exploration
        if random.random() < epsilon:
            action_id = random.randint(0, 5)
            system = "random"

        # Track state
        self.state_history.append(state_hash)
        if len(self.state_history) > 50:
            self.state_history.pop(0)
        self.action_history.append(action_id)

        # Save for next step
        self.last_state = state
        self.last_action = action_id

        return action_id, {
            "system": system,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "in_loop": in_loop,
            "action_probs": action_probs,
        }


def evaluate_integrated(
    agent: IntegratedARIAAgent,
    game_id: str,
    max_steps: int = 300,
    num_episodes: int = 5,
    epsilon: float = 0.0,
):
    """Evaluate integrated ARIA agent on game."""
    if not ARC_AGI_AVAILABLE:
        return {"error": "arc_agi not available"}

    results = []

    for ep in range(num_episodes):
        arcade = Arcade(operation_mode=OperationMode.OFFLINE)
        env = arcade.make(game_id)

        if env is None:
            results.append({"error": f"Could not create {game_id}"})
            continue

        raw_frame = env.reset()
        if raw_frame is None:
            arcade.close_scorecard()
            continue

        agent.reset()
        levels = 0
        step = 0
        system_counts = {"fast": 0, "slow": 0, "random": 0}
        loop_count = 0

        while step < max_steps:
            if raw_frame.state == GameState.WIN:
                break
            if raw_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                raw_frame = env.reset()
                if raw_frame is None:
                    break
                agent.reset()
                continue

            # Get observation
            if raw_frame.frame:
                obs = agent.preprocess_obs(raw_frame.frame[0])
            else:
                obs = torch.zeros(1, agent.grid_size, agent.grid_size).long().to(agent.device)

            # Get action
            action_id, info = agent.act(
                obs,
                epsilon=epsilon,
                uncertainty_threshold=0.2,
            )

            system_counts[info["system"]] += 1
            if info["in_loop"]:
                loop_count += 1

            # Execute
            game_action_id = min(action_id + 1, 6)
            action = GameAction.from_id(game_action_id)
            raw_frame = env.step(action)
            if raw_frame is None:
                break

            levels = max(levels, raw_frame.levels_completed)
            step += 1

        arcade.close_scorecard()
        won = raw_frame.state == GameState.WIN if raw_frame else False

        results.append({
            "episode": ep,
            "steps": step,
            "levels": levels,
            "won": won,
            "system_counts": system_counts,
            "loops": loop_count,
        })

        fast_pct = system_counts["fast"] / step * 100 if step > 0 else 0
        slow_pct = system_counts["slow"] / step * 100 if step > 0 else 0
        print(f"  Ep {ep}: levels={levels}, won={won}, fast={fast_pct:.0f}%, slow={slow_pct:.0f}%, loops={loop_count}")

    total_levels = sum(r.get("levels", 0) for r in results)
    total_wins = sum(1 for r in results if r.get("won", False))

    return {
        "game_id": game_id,
        "total_levels": total_levels,
        "total_wins": total_wins,
        "episodes": results,
    }


def main():
    parser = argparse.ArgumentParser(description="ARIA Integrated Evaluation")
    parser.add_argument("--checkpoint", "-c", required=True, help="BC checkpoint path")
    parser.add_argument("--world-model", "-w", default=None, help="World model checkpoint")
    parser.add_argument("--game", "-g", default="ls20", help="Game to evaluate")
    parser.add_argument("--episodes", "-e", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.7)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.2)
    parser.add_argument("--grid-size", type=int, default=16)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = ARIALiteConfig()

    agent = IntegratedARIAAgent(
        bc_checkpoint=args.checkpoint,
        config=config,
        device=device,
        grid_size=args.grid_size,
        world_model_checkpoint=args.world_model,
    )

    print(f"\nEvaluating on {args.game}...")
    result = evaluate_integrated(
        agent,
        args.game,
        max_steps=args.max_steps,
        num_episodes=args.episodes,
        epsilon=args.epsilon,
    )

    print(f"\nResults: {result['total_levels']} levels, {result['total_wins']} wins")


if __name__ == "__main__":
    main()
