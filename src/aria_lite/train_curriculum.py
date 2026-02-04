#!/usr/bin/env python3
"""
Curriculum Learning for Spatial ARC Tasks

Start with small grids (2x2) and progressively increase to larger grids (5x5).
This makes learning easier by reducing the number of cells to get right.

Curriculum: 2x2 → 3x3 → 4x4 → 5x5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .training.arc_like_simple import SimpleARCEnv
from .training.arc_position_encoder import PositionAwareARCEncoder
from .training.ppo_trainer import PPOConfig, RolloutBuffer, compute_gae


class CurriculumActorCritic(nn.Module):
    """Actor-critic that can handle variable grid sizes."""

    def __init__(self, num_colors: int = 4, max_grid_size: int = 5, hidden_dim: int = 128):
        super().__init__()
        self.num_colors = num_colors
        self.max_grid_size = max_grid_size
        self.hidden_dim = hidden_dim

        # Color embedding (shared across grid sizes)
        self.color_embed = nn.Embedding(num_colors, 32)

        # Position embeddings for max grid size
        self.row_embed = nn.Embedding(max_grid_size, 32)
        self.col_embed = nn.Embedding(max_grid_size, 32)

        # Process input grid
        self.input_proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
        )

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(64, num_heads=4, batch_first=True)

        # Process output context
        self.output_proj = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
        )

        # Combine and output
        self.policy = nn.Sequential(
            nn.Linear(64 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors),
        )

        self.value = nn.Sequential(
            nn.Linear(64 * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, obs):
        """Encode observation of any grid size."""
        B, C, H, W = obs.shape
        device = obs.device

        input_grid = obs[:, 0]   # [B, H, W]
        output_grid = obs[:, 1]  # [B, H, W]
        mask = obs[:, 2]         # [B, H, W]

        # Get position embeddings
        rows = torch.arange(H, device=device)
        cols = torch.arange(W, device=device)
        row_emb = self.row_embed(rows)  # [H, 32]
        col_emb = self.col_embed(cols)  # [W, 32]

        # Create position grid
        pos_emb = torch.zeros(H, W, 64, device=device)
        for i in range(H):
            for j in range(W):
                pos_emb[i, j] = torch.cat([row_emb[i], col_emb[j]])
        pos_emb = pos_emb.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, W, 64]

        # Embed input colors and combine with positions
        input_colors = self.color_embed(input_grid)  # [B, H, W, 32]
        input_features = torch.cat([input_colors, pos_emb[:, :, :, :32]], dim=-1)  # [B, H, W, 64]
        input_features = self.input_proj(input_features)  # [B, H, W, 64]

        # Flatten for attention
        input_flat = input_features.view(B, H * W, 64)  # [B, H*W, 64]

        # Find current position
        current_pos_emb = torch.zeros(B, 64, device=device)
        for b in range(B):
            mask_b = mask[b]
            positions = mask_b.nonzero()
            if len(positions) > 0:
                i, j = positions[0].tolist()
                current_pos_emb[b] = torch.cat([row_emb[i], col_emb[j]])

        # Cross-attention
        query = current_pos_emb.unsqueeze(1)  # [B, 1, 64]
        attended, _ = self.cross_attn(query, input_flat, input_flat)
        attended = attended.squeeze(1)  # [B, 64]

        # Output context
        output_colors = self.color_embed(output_grid)
        output_features = torch.cat([output_colors, pos_emb[:, :, :, :32]], dim=-1)
        output_features = self.output_proj(output_features)
        output_context = output_features.mean(dim=(1, 2))  # [B, 64]

        # Combine
        combined = torch.cat([attended, output_context, current_pos_emb], dim=-1)
        return combined

    def get_action_and_value(self, obs, action=None):
        state = self.encode(obs)
        logits = self.policy(state)
        value = self.value(state).squeeze(-1)

        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        return self.value(self.encode(obs)).squeeze(-1)


class CurriculumVecEnv:
    """Vectorized env with adjustable grid size."""

    def __init__(self, task_type, num_envs, grid_size, num_colors=4, device="cuda"):
        self.grid_size = grid_size
        self.envs = [
            SimpleARCEnv(grid_size, task_type, num_colors, seed=i*1000)
            for i in range(num_envs)
        ]
        self.num_envs = num_envs
        self.device = device

    def reset(self):
        obs = [e.reset() for e in self.envs]
        return torch.stack(obs).to(self.device)

    def step(self, actions):
        obs_list, rewards, dones, infos = [], [], [], []
        for env, a in zip(self.envs, actions):
            obs, r, d, info = env.step(a.item())
            if d:
                obs = env.reset()
            obs_list.append(obs)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        return (
            torch.stack(obs_list).to(self.device),
            torch.tensor(rewards, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device),
            infos,
        )

    def set_grid_size(self, new_size):
        """Change grid size for all envs."""
        self.grid_size = new_size
        for i, env in enumerate(self.envs):
            env.grid_size = new_size
            env.num_cells = new_size * new_size


def train_stage(model, optimizer, task_type, grid_size, num_updates, device):
    """Train one stage of curriculum."""
    print(f"\n  Grid size: {grid_size}x{grid_size}", flush=True)

    config = PPOConfig(num_envs=16, rollout_steps=32, device=device)
    env = CurriculumVecEnv(task_type, num_envs=16, grid_size=grid_size, device=device)

    episode_accuracies = []
    obs = env.reset()

    for update in range(num_updates):
        buffer = RolloutBuffer()

        for _ in range(config.rollout_steps):
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs)

            next_obs, reward, done, infos = env.step(action)
            buffer.add(obs.clone(), action.clone(), log_prob.clone(),
                      reward.clone(), done.clone(), value.clone())

            for info in infos:
                if "accuracy" in info:
                    episode_accuracies.append(info["accuracy"])

            obs = next_obs

        # PPO update
        with torch.no_grad():
            next_value = model.get_value(obs)

        b_obs = torch.stack(buffer.observations)
        b_actions = torch.stack(buffer.actions)
        b_log_probs = torch.stack(buffer.log_probs)
        b_rewards = torch.stack(buffer.rewards)
        b_dones = torch.stack(buffer.dones)
        b_values = torch.stack(buffer.values)

        all_adv, all_ret = [], []
        for i in range(env.num_envs):
            adv, ret = compute_gae(b_rewards[:, i], b_values[:, i], b_dones[:, i],
                                  next_value[i], config.gamma, config.gae_lambda)
            all_adv.append(adv)
            all_ret.append(ret)

        b_adv = torch.stack(all_adv, dim=1)
        b_ret = torch.stack(all_ret, dim=1)

        T, N = b_obs.shape[:2]
        flat = lambda x: x.reshape(T * N, *x.shape[2:]) if x.dim() > 2 else x.reshape(T * N)

        f_obs, f_act, f_lp = flat(b_obs), flat(b_actions), flat(b_log_probs)
        f_adv, f_ret = flat(b_adv), flat(b_ret)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        batch_size = f_obs.shape[0]
        mb_size = batch_size // 4

        for _ in range(4):
            idx = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                mb = idx[start:start + mb_size]
                _, new_lp, ent, new_v = model.get_action_and_value(f_obs[mb], f_act[mb])

                ratio = torch.exp(new_lp - f_lp[mb])
                pg1 = -f_adv[mb] * ratio
                pg2 = -f_adv[mb] * torch.clamp(ratio, 0.8, 1.2)
                loss = torch.max(pg1, pg2).mean() + 0.5 * F.mse_loss(new_v, f_ret[mb]) - 0.01 * ent.mean()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if (update + 1) % 50 == 0:
            mean_acc = sum(episode_accuracies[-100:]) / max(len(episode_accuracies[-100:]), 1)
            print(f"    Update {update + 1}/{num_updates} | Accuracy: {mean_acc:.1%}", flush=True)

    # Evaluate
    model.eval()
    successes = 0
    for i in range(50):
        test_env = SimpleARCEnv(grid_size, task_type, num_colors=4, seed=99999 + i)
        obs = test_env.reset().unsqueeze(0).to(device)
        done = False
        while not done:
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs)
            obs, _, done, info = test_env.step(action.item())
            obs = obs.unsqueeze(0).to(device)
            if done and info.get("success", False):
                successes += 1
    model.train()

    success_rate = successes / 50
    print(f"  Eval: {success_rate:.1%}", flush=True)
    return success_rate


def train_curriculum(task_type: str = "copy", device: str = "cuda"):
    """Train with curriculum: 2x2 → 3x3 → 4x4 → 5x5"""
    print(f"\n{'='*60}")
    print(f"CURRICULUM LEARNING: {task_type}")
    print(f"{'='*60}", flush=True)

    model = CurriculumActorCritic(num_colors=4, max_grid_size=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    results = {}

    # Curriculum stages
    stages = [
        (2, 100),   # 2x2: quick warmup
        (3, 150),   # 3x3: moderate
        (4, 200),   # 4x4: longer
        (5, 250),   # 5x5: final
    ]

    for grid_size, num_updates in stages:
        results[grid_size] = train_stage(model, optimizer, task_type, grid_size, num_updates, device)

    print(f"\n{'='*60}")
    print("CURRICULUM RESULTS")
    print(f"{'='*60}")
    for size, rate in results.items():
        status = "PASS" if rate > 0.8 else "PARTIAL" if rate > 0.5 else "FAIL"
        print(f"  {size}x{size}: {rate:.1%} [{status}]")

    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # Train copy with curriculum
    copy_results = train_curriculum("copy", device)

    # Train reflect with curriculum
    reflect_results = train_curriculum("reflect_h", device)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print("\nCopy:")
    for size, rate in copy_results.items():
        print(f"  {size}x{size}: {rate:.1%}")

    print("\nReflect_h:")
    for size, rate in reflect_results.items():
        print(f"  {size}x{size}: {rate:.1%}")


if __name__ == "__main__":
    main()
