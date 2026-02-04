#!/usr/bin/env python3
"""
Curriculum Learning v2 - Use the working PositionAwareARCEncoder

Train on 2x2 → 3x3 → 4x4 → 5x5 but use the encoder that already works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .training.arc_like_simple import SimpleARCEnv
from .training.ppo_trainer import PPOConfig, RolloutBuffer, compute_gae


class FlexibleARCEncoder(nn.Module):
    """Position-aware encoder that handles any grid size up to max_size."""

    def __init__(self, num_colors=4, max_grid_size=5, embed_dim=64, hidden_dim=128, output_dim=128):
        super().__init__()
        self.max_grid_size = max_grid_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Position embeddings (learnable, for max grid size)
        self.row_embed = nn.Embedding(max_grid_size, embed_dim // 2)
        self.col_embed = nn.Embedding(max_grid_size, embed_dim // 2)

        # Input processing
        self.input_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Cross-attention components
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Output processing
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Final combination
        self.combine = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def get_pos_embed(self, row, col, device):
        """Get position embedding for a single cell."""
        r = torch.tensor([row], device=device)
        c = torch.tensor([col], device=device)
        return torch.cat([self.row_embed(r), self.col_embed(c)], dim=-1).squeeze(0)

    def forward(self, obs):
        B, C, H, W = obs.shape
        device = obs.device

        input_grid = obs[:, 0]
        output_grid = obs[:, 1]
        mask = obs[:, 2]

        # Build position embeddings for this grid size
        pos_grid = torch.zeros(H, W, self.embed_dim, device=device)
        for i in range(H):
            for j in range(W):
                pos_grid[i, j] = self.get_pos_embed(i, j, device)
        pos_grid = pos_grid.unsqueeze(0).expand(B, -1, -1, -1)

        # Input features: color + position
        input_colors = self.color_embed(input_grid)
        input_features = torch.cat([input_colors, pos_grid], dim=-1)
        input_features = self.input_proj(input_features)
        input_flat = input_features.view(B, H * W, -1)

        # Find current cell position
        current_pos_emb = torch.zeros(B, self.embed_dim, device=device)
        for b in range(B):
            positions = mask[b].nonzero()
            if len(positions) > 0:
                i, j = positions[0].tolist()
                current_pos_emb[b] = self.get_pos_embed(i, j, device)

        # Cross-attention: current position attends to input
        query = self.q_proj(current_pos_emb).unsqueeze(1)
        key = self.k_proj(input_flat)
        value = self.v_proj(input_flat)

        attn_scores = torch.bmm(query, key.transpose(1, 2)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attended = torch.bmm(attn_weights, value).squeeze(1)
        attended = self.out_proj(attended)

        # Output context
        output_colors = self.color_embed(output_grid)
        output_features = torch.cat([output_colors, pos_grid], dim=-1)
        output_features = self.output_proj(output_features)
        output_context = output_features.mean(dim=(1, 2))

        # Combine
        combined = torch.cat([attended, output_context, current_pos_emb], dim=-1)
        return self.combine(combined)


class FlexibleActorCritic(nn.Module):
    def __init__(self, num_colors=4, max_grid_size=5):
        super().__init__()
        self.encoder = FlexibleARCEncoder(num_colors, max_grid_size)
        self.policy = nn.Linear(self.encoder.output_dim, num_colors)
        self.value = nn.Linear(self.encoder.output_dim, 1)

    def get_action_and_value(self, obs, action=None):
        state = self.encoder(obs)
        logits = self.policy(state)
        value = self.value(state).squeeze(-1)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def get_value(self, obs):
        return self.value(self.encoder(obs)).squeeze(-1)


class VecEnv:
    def __init__(self, task_type, num_envs, grid_size, num_colors=4, device="cuda"):
        # Training envs: truly random grids (no fixed seeds = no memorization)
        self.envs = [SimpleARCEnv(grid_size, task_type, num_colors, deterministic=False)
                     for _ in range(num_envs)]
        self.num_envs = num_envs
        self.device = device
        self.grid_size = grid_size

    def reset(self):
        return torch.stack([e.reset() for e in self.envs]).to(self.device)

    def step(self, actions):
        results = [e.step(a.item()) for e, a in zip(self.envs, actions)]
        obs = [e.reset() if r[2] else r[0] for e, r in zip(self.envs, results)]
        return (torch.stack(obs).to(self.device),
                torch.tensor([r[1] for r in results], device=self.device),
                torch.tensor([r[2] for r in results], dtype=torch.float32, device=self.device),
                [r[3] for r in results])


def train_stage(model, optimizer, task_type, grid_size, num_updates, device):
    """Train one curriculum stage."""
    print(f"\n  Stage: {grid_size}x{grid_size} ({num_updates} updates)", flush=True)

    env = VecEnv(task_type, 16, grid_size, device=device)
    obs = env.reset()
    episode_accs = []

    for update in range(num_updates):
        buffer = RolloutBuffer()

        for _ in range(32):
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs)
            next_obs, reward, done, infos = env.step(action)
            buffer.add(obs.clone(), action.clone(), log_prob.clone(),
                      reward.clone(), done.clone(), value.clone())
            for info in infos:
                if "accuracy" in info:
                    episode_accs.append(info["accuracy"])
            obs = next_obs

        # PPO update
        with torch.no_grad():
            next_value = model.get_value(obs)

        b_obs = torch.stack(buffer.observations)
        b_act = torch.stack(buffer.actions)
        b_lp = torch.stack(buffer.log_probs)
        b_rew = torch.stack(buffer.rewards)
        b_done = torch.stack(buffer.dones)
        b_val = torch.stack(buffer.values)

        advs, rets = [], []
        for i in range(16):
            a, r = compute_gae(b_rew[:, i], b_val[:, i], b_done[:, i], next_value[i], 0.99, 0.95)
            advs.append(a)
            rets.append(r)

        T, N = b_obs.shape[:2]
        f_obs = b_obs.reshape(T*N, *b_obs.shape[2:])
        f_act = b_act.reshape(T*N)
        f_lp = b_lp.reshape(T*N)
        f_adv = torch.stack(advs, dim=1).reshape(T*N)
        f_ret = torch.stack(rets, dim=1).reshape(T*N)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        bs = f_obs.shape[0]
        for _ in range(4):
            idx = torch.randperm(bs, device=device)
            for s in range(0, bs, bs//4):
                mb = idx[s:s+bs//4]
                _, nlp, ent, nv = model.get_action_and_value(f_obs[mb], f_act[mb])
                ratio = torch.exp(nlp - f_lp[mb])
                loss = (torch.max(-f_adv[mb]*ratio, -f_adv[mb]*torch.clamp(ratio, 0.8, 1.2)).mean()
                       + 0.5 * F.mse_loss(nv, f_ret[mb]) - 0.01 * ent.mean())
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if (update + 1) % 50 == 0:
            acc = sum(episode_accs[-100:]) / max(len(episode_accs[-100:]), 1)
            print(f"    Update {update+1}: {acc:.1%}", flush=True)

    # Eval (deterministic for reproducibility)
    model.eval()
    success = 0
    for i in range(50):
        te = SimpleARCEnv(grid_size, task_type, 4, seed=99999+i, deterministic=True)
        ob = te.reset().unsqueeze(0).to(device)
        done = False
        while not done:
            with torch.no_grad():
                a, _, _, _ = model.get_action_and_value(ob)
            ob, _, done, info = te.step(a.item())
            ob = ob.unsqueeze(0).to(device)
            if done and info.get("success"):
                success += 1
    model.train()
    rate = success / 50
    print(f"  Eval: {rate:.1%}", flush=True)
    return rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    for task in ["copy", "reflect_h"]:
        print(f"\n{'='*60}")
        print(f"CURRICULUM: {task}")
        print(f"{'='*60}")

        model = FlexibleActorCritic(num_colors=4, max_grid_size=5).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

        results = {}
        # More updates for harder stages
        for size, updates in [(2, 150), (3, 200), (4, 250), (5, 300)]:
            results[size] = train_stage(model, optimizer, task, size, updates, device)

        print(f"\n{task} results:")
        for sz, r in results.items():
            print(f"  {sz}x{sz}: {r:.1%}")


if __name__ == "__main__":
    main()
