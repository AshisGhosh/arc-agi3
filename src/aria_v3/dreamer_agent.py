#!/usr/bin/env python3
"""
Online Dreamer agent for ARC-AGI-3.

Uses a pretrained Transformer world model to:
1. Encode frames via frozen VQ-VAE
2. Imagine rollouts using the dynamics model
3. Train value/policy by scoring imagined trajectories
4. Fine-tune dynamics online from real interactions
5. Select actions via the learned policy

The agent starts with a pretrained dynamics model (general visual knowledge)
and adapts online to each new game. Model resets per level.

Usage:
    uv run python -m src.aria_v3.dreamer_agent --game ls20
    uv run python -m src.aria_v3.dreamer_agent --game vc33 --max-actions 5000
"""

import argparse
import hashlib
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from ..aria_v2.tokenizer.frame_tokenizer import FrameVQVAE
from ..aria_v2.tokenizer.trajectory_dataset import vq_cell_to_pixel
from .world_model import DreamerWorldModel


@dataclass
class Transition:
    """A real (state, action, next_state) transition."""
    vq_current: torch.Tensor   # [64] VQ codes
    action_type: int
    action_loc: int
    vq_next: torch.Tensor      # [64] VQ codes
    game_over: bool
    level_complete: bool


class DreamerAgent:
    """Online Dreamer agent with imagination-based planning.

    Action space mapping:
        Indices 0..4 → simple actions (game API types 1-5)
        Indices 5..68 → click at VQ cell (game API type 6, location = idx - 5)
    """

    def __init__(
        self,
        vqvae_path: str = "checkpoints/vqvae/best.pt",
        model_path: str = "checkpoints/dreamer/pretrained.pt",
        device: str = "cuda",
        # Imagination
        imagine_horizon: int = 15,
        imagine_trajectories: int = 32,
        imagine_every: int = 10,
        imagine_temperature: float = 1.0,
        # Online training
        replay_buffer_size: int = 10000,
        online_batch_size: int = 32,
        online_lr: float = 1e-4,
        train_every: int = 5,
        # Policy
        policy_lr: float = 3e-4,
        exploration_noise: float = 0.1,
        gamma: float = 0.99,
        lambda_gae: float = 0.95,
    ):
        self.device = device
        self.imagine_horizon = imagine_horizon
        self.imagine_trajectories = imagine_trajectories
        self.imagine_every = imagine_every
        self.imagine_temperature = imagine_temperature
        self.online_batch_size = online_batch_size
        self.train_every = train_every
        self.exploration_noise = exploration_noise
        self.gamma = gamma
        self.lambda_gae = lambda_gae

        # Load VQ-VAE (frozen)
        vqvae_ckpt = torch.load(vqvae_path, weights_only=False, map_location=device)
        self.vqvae = FrameVQVAE(vqvae_ckpt["config"]).to(device)
        self.vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
        self.vqvae.eval()

        # Load world model
        self.model = DreamerWorldModel().to(device)
        if os.path.exists(model_path):
            ckpt = torch.load(model_path, weights_only=False, map_location=device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded pretrained model from {model_path}")
        else:
            print(f"No pretrained model at {model_path}, starting from scratch")

        # Save initial weights for level reset
        self._initial_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        # Optimizers
        self.dynamics_optimizer = torch.optim.Adam(
            list(self.model.encoder.parameters()) +
            list(self.model.action_proj.parameters()) +
            list(self.model.film.parameters()) +
            list(self.model.dynamics_head.parameters()) +
            list(self.model.vq_embed.parameters()) +
            list(self.model.pos_embed.parameters()) +
            list(self.model.action_type_embed.parameters()) +
            list(self.model.action_loc_embed.parameters()),
            lr=online_lr,
        )
        self.policy_optimizer = torch.optim.Adam(
            list(self.model.policy_head.parameters()),
            lr=policy_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            list(self.model.value_head.parameters()),
            lr=online_lr,
        )

        # Replay buffer
        self.replay_buffer: deque[Transition] = deque(maxlen=replay_buffer_size)
        self.experience_hashes: set[str] = set()

        # Game state
        self.available_actions: list[int] = []
        self.available_mask: torch.Tensor | None = None
        self.has_click = False
        self.prev_vq: torch.Tensor | None = None
        self.prev_action_idx: int | None = None
        self.prev_frame: np.ndarray | None = None

        # Stats
        self.step_count = 0
        self.imagine_count = 0
        self.dynamics_train_count = 0
        self.policy_train_count = 0
        self.frame_changes = 0

    def setup(self, available_actions: list[int]) -> None:
        """Configure for a specific game's action space."""
        self.available_actions = available_actions
        simple = [a for a in available_actions if 1 <= a <= 5]
        self.has_click = 6 in available_actions

        # Build availability mask for policy output (69 dims)
        mask = torch.zeros(self.model.total_policy_actions, dtype=torch.bool, device=self.device)
        for a in simple:
            mask[a - 1] = True  # simple action index
        if self.has_click:
            mask[5:69] = True  # all 64 click positions
        self.available_mask = mask

    @torch.no_grad()
    def encode_frame(self, frame: np.ndarray) -> torch.Tensor:
        """VQ-encode a frame.

        Args:
            frame: [64, 64] numpy array

        Returns:
            [64] long tensor of VQ code indices
        """
        frame_t = torch.tensor(frame, dtype=torch.long).unsqueeze(0).to(self.device)
        indices = self.vqvae.encode(frame_t)  # [1, 8, 8]
        return indices[0].flatten()  # [64]

    def act(self, frame: np.ndarray) -> tuple[int, int | None, int | None]:
        """Choose an action given the current frame.

        Returns:
            (action_type, x, y) for the game API
        """
        self.step_count += 1
        vq = self.encode_frame(frame)

        # Record transition from previous step
        if self.prev_vq is not None and self.prev_action_idx is not None:
            frame_changed = not np.array_equal(self.prev_frame, frame)
            if frame_changed:
                self.frame_changes += 1
            self._add_transition(self.prev_vq, self.prev_action_idx, vq, frame_changed)

        # Maybe train dynamics on real data
        if self.step_count % self.train_every == 0 and len(self.replay_buffer) >= self.online_batch_size:
            self._train_dynamics()

        # Maybe imagine and train policy
        if self.step_count % self.imagine_every == 0 and self.step_count > 20:
            self._imagine_and_train(vq)

        # Select action
        action_idx = self._select_action(vq)

        # Convert to game API
        action_type, x, y = self._to_game_action(action_idx)

        # Save for next step
        self.prev_vq = vq.clone()
        self.prev_action_idx = action_idx
        self.prev_frame = frame.copy()

        return action_type, x, y

    def _select_action(self, vq: torch.Tensor) -> int:
        """Select action using learned policy + exploration noise."""
        valid = self.available_mask.nonzero(as_tuple=True)[0]

        self.model.eval()
        with torch.no_grad():
            # Epsilon-greedy exploration
            if torch.rand(1).item() < self.exploration_noise:
                return valid[torch.randint(len(valid), (1,))].item()

            h = self.model.encode(vq.unsqueeze(0))
            logits = self.model.predict_policy(h)[0]  # [69]

            if self.available_mask is not None:
                logits[~self.available_mask] = float("-inf")

            # Guard against NaN/inf from online training instability
            if logits[self.available_mask].isnan().any() or logits[self.available_mask].isinf().all():
                return valid[torch.randint(len(valid), (1,))].item()

            probs = F.softmax(logits, dim=-1)
            probs = probs.clamp(min=0.0)
            if probs.sum() < 1e-8:
                return valid[torch.randint(len(valid), (1,))].item()

            return torch.multinomial(probs, 1).item()

    def _add_transition(
        self, vq_current: torch.Tensor, action_idx: int, vq_next: torch.Tensor, changed: bool
    ) -> None:
        """Add transition to replay buffer with dedup."""
        # Dedup hash
        hash_input = vq_current.cpu().numpy().tobytes() + str(action_idx).encode()
        h = hashlib.md5(hash_input).hexdigest()
        if h in self.experience_hashes:
            return
        self.experience_hashes.add(h)

        action_type, action_loc = self.model._action_idx_to_type_loc(
            torch.tensor([action_idx], device=self.device)
        )

        self.replay_buffer.append(Transition(
            vq_current=vq_current.cpu(),
            action_type=action_type[0].item(),
            action_loc=action_loc[0].item(),
            vq_next=vq_next.cpu(),
            game_over=not changed,  # no change = likely stuck/dead
            level_complete=False,
        ))

    def _train_dynamics(self) -> None:
        """Train dynamics model on a batch from the replay buffer."""
        self.model.train()
        batch_indices = np.random.choice(
            len(self.replay_buffer), self.online_batch_size, replace=False
        )

        vq_c = torch.stack([self.replay_buffer[i].vq_current for i in batch_indices]).to(self.device)
        vq_n = torch.stack([self.replay_buffer[i].vq_next for i in batch_indices]).to(self.device)
        a_type = torch.tensor([self.replay_buffer[i].action_type for i in batch_indices],
                              device=self.device)
        a_loc = torch.tensor([self.replay_buffer[i].action_loc for i in batch_indices],
                             device=self.device)

        h = self.model.encode(vq_c)
        dyn_logits = self.model.predict_dynamics(h, a_type, a_loc)

        loss = F.cross_entropy(
            dyn_logits.reshape(-1, self.model.num_vq_codes),
            vq_n.reshape(-1),
        )

        if loss.isnan() or loss.isinf():
            return

        self.dynamics_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.dynamics_optimizer.step()
        self.dynamics_train_count += 1

    def _imagine_and_train(self, current_vq: torch.Tensor) -> None:
        """Imagine rollouts from current state and train policy + value."""
        self.model.eval()

        # Imagine — skip if model is unstable
        try:
            rollouts = self.model.imagine(
                current_vq,
                horizon=self.imagine_horizon,
                num_trajectories=self.imagine_trajectories,
                available_actions=self.available_mask,
                temperature=self.imagine_temperature,
            )
        except RuntimeError:
            return
        self.imagine_count += 1

        values = rollouts["values"]    # [K, H+1]
        states = rollouts["states"]    # [K, H+1, 64]
        actions = rollouts["actions"]  # [K, H]

        # Skip if imagination produced NaN
        if values.isnan().any() or states.isnan().any():
            return

        K, H = actions.shape

        # Compute lambda-returns for value targets
        with torch.no_grad():
            returns = torch.zeros(K, H, device=self.device)
            last_return = values[:, -1]  # V(s_H)

            for t in reversed(range(H)):
                returns[:, t] = values[:, t + 1] + self.gamma * self.lambda_gae * (
                    last_return - values[:, t + 1]
                )
                last_return = returns[:, t]

        # Train value: predict returns
        self.model.train()
        for t in range(0, H, 3):  # Train on every 3rd step for speed
            h = self.model.encode(states[:, t])
            v_pred = self.model.predict_value(h)
            v_loss = F.mse_loss(v_pred, returns[:, t].detach())
            if v_loss.isnan():
                continue

            self.value_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.value_head.parameters(), 1.0)
            self.value_optimizer.step()

        # Train policy: REINFORCE with baseline
        self.model.train()
        for t in range(0, H, 3):  # Train on every 3rd step for speed
            h = self.model.encode(states[:, t].detach())
            logits = self.model.predict_policy(h)

            if self.available_mask is not None:
                logits[:, ~self.available_mask] = float("-inf")

            log_probs = F.log_softmax(logits, dim=-1)

            # Guard: skip if log_probs has NaN
            if log_probs.isnan().any():
                continue

            selected_log_prob = log_probs.gather(1, actions[:, t].unsqueeze(1)).squeeze(1)

            advantage = (returns[:, t] - values[:, t]).detach()
            policy_loss = -(selected_log_prob * advantage).mean()

            # Entropy bonus for exploration
            probs = F.softmax(logits, dim=-1).clamp(min=1e-8)
            entropy = -(probs * probs.log()).sum(dim=-1).mean()
            policy_loss = policy_loss - 0.01 * entropy

            if policy_loss.isnan():
                continue

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.policy_head.parameters(), 1.0)
            self.policy_optimizer.step()

        self.policy_train_count += 1

    def _to_game_action(self, action_idx: int) -> tuple[int, int | None, int | None]:
        """Convert unified action index to game API call."""
        a_type, a_loc = self.model._action_idx_to_type_loc(
            torch.tensor([action_idx], device=self.device)
        )
        action_type = a_type[0].item()
        action_loc = a_loc[0].item()

        if action_type == 6 and action_loc < 64:
            x, y = vq_cell_to_pixel(action_loc)
            return action_type, x, y
        else:
            return action_type, None, None

    def on_level_complete(self) -> None:
        """Mark last transition and reset for new level."""
        if self.replay_buffer:
            self.replay_buffer[-1] = Transition(
                vq_current=self.replay_buffer[-1].vq_current,
                action_type=self.replay_buffer[-1].action_type,
                action_loc=self.replay_buffer[-1].action_loc,
                vq_next=self.replay_buffer[-1].vq_next,
                game_over=False,
                level_complete=True,
            )

        # Reset to pretrained weights
        self.model.load_state_dict(self._initial_state)

        # Reset optimizers
        for opt in [self.dynamics_optimizer, self.policy_optimizer, self.value_optimizer]:
            for group in opt.param_groups:
                for p in group["params"]:
                    opt.state[p] = {}

        # Clear buffer
        self.replay_buffer.clear()
        self.experience_hashes.clear()
        self.prev_vq = None
        self.prev_action_idx = None
        self.prev_frame = None

    def get_stats(self) -> dict:
        return {
            "steps": self.step_count,
            "frame_changes": self.frame_changes,
            "buffer": len(self.replay_buffer),
            "dynamics_trains": self.dynamics_train_count,
            "policy_trains": self.policy_train_count,
            "imaginations": self.imagine_count,
        }


def run_agent(
    game_id: str = "ls20",
    max_actions: int = 5000,
    verbose: bool = True,
    device: str = "cuda",
    model_path: str = "checkpoints/dreamer/pretrained.pt",
    vqvae_path: str = "checkpoints/vqvae/best.pt",
):
    """Run the Dreamer agent on an ARC-AGI-3 game."""
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

    agent = DreamerAgent(
        vqvae_path=vqvae_path,
        model_path=model_path,
        device=device,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"Playing {game_id} with Dreamer Agent (v3)")
        params = sum(p.numel() for p in agent.model.parameters())
        print(f"Model: {params:,} params")
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
        print(f"Available actions: {list(available)}")

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
                print(f"\nWon after {action_count} actions!")
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
                print(f"  Level {levels_completed} completed! (action {action_count})")

        frame = np.array(observation.frame[0])
        action_type, x, y = agent.act(frame)

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
            ms = elapsed * 1000 / max(action_count, 1)
            print(
                f"Step {action_count}: type={action_type} loc={loc_str} | "
                f"changes={stats['frame_changes']} buf={stats['buffer']} | "
                f"dyn_t={stats['dynamics_trains']} pol_t={stats['policy_trains']} "
                f"img={stats['imaginations']} | {ms:.1f}ms/act"
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
        print(f"Time: {duration:.1f}s ({duration/max(action_count,1)*1000:.1f}ms/action)")
        print(f"Frame changes: {stats['frame_changes']}")
        print(f"Dynamics trains: {stats['dynamics_trains']}")
        print(f"Policy trains: {stats['policy_trains']}")
        print(f"Imaginations: {stats['imaginations']}")
        print(f"Buffer: {stats['buffer']}")

        print(f"\n{'='*60}")
        print("Scorecard")
        print(f"{'='*60}")
        print(arc.get_scorecard())


def main():
    parser = argparse.ArgumentParser(description="Run Dreamer agent (v3)")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--max-actions", "-m", type=int, default=5000)
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="checkpoints/dreamer/pretrained.pt")
    parser.add_argument("--vqvae", default="checkpoints/vqvae/best.pt")
    args = parser.parse_args()

    run_agent(
        game_id=args.game,
        max_actions=args.max_actions,
        verbose=not args.quiet,
        device=args.device,
        model_path=args.model,
        vqvae_path=args.vqvae,
    )


if __name__ == "__main__":
    main()
