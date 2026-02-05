#!/usr/bin/env python3
"""Train on simplified ARC tasks with dense rewards."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .training.arc_like_simple import SimpleARCEnv
from .training.ppo_trainer import PPOConfig, RolloutBuffer, compute_gae


class SimpleARCEncoder(nn.Module):
    """Encoder for simplified ARC (3 channels: input, output, mask)."""

    def __init__(self, num_colors: int = 4, grid_size: int = 5, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.output_dim = output_dim

        # Embed each channel
        self.embed = nn.Embedding(num_colors + 1, 16)  # +1 for mask

        # Conv layers
        self.conv = nn.Sequential(
            nn.Conv2d(16 * 3, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_dim * grid_size * grid_size, output_dim),
        )

    def forward(self, obs):
        """obs: [B, 3, H, W]"""
        B, C, H, W = obs.shape

        # Embed each channel and concat
        embedded = []
        for c in range(C):
            emb = self.embed(obs[:, c])  # [B, H, W, 16]
            emb = emb.permute(0, 3, 1, 2)  # [B, 16, H, W]
            embedded.append(emb)

        x = torch.cat(embedded, dim=1)  # [B, 48, H, W]
        return self.conv(x)


class SimpleActorCritic(nn.Module):
    """Actor-critic for simplified ARC."""

    def __init__(self, encoder, num_actions: int = 4):
        super().__init__()
        self.encoder = encoder

        self.policy = nn.Linear(encoder.output_dim, num_actions)
        self.value = nn.Linear(encoder.output_dim, 1)

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


class VecSimpleARC:
    """Vectorized simple ARC env."""

    def __init__(self, task_type, num_envs, grid_size=5, num_colors=4, device="cuda"):
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

        for i, (env, a) in enumerate(zip(self.envs, actions)):
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


def train_simple_arc(task_type: str, num_updates: int = 200, device: str = "cuda"):
    """Train on simplified ARC task."""
    print(f"\n{'='*50}")
    print(f"Simple ARC: {task_type}")
    print(f"{'='*50}", flush=True)

    grid_size = 5
    num_colors = 4

    encoder = SimpleARCEncoder(num_colors, grid_size, hidden_dim=64, output_dim=128)
    model = SimpleActorCritic(encoder, num_actions=num_colors).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    config = PPOConfig(num_envs=16, rollout_steps=32, learning_rate=3e-4, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    env = VecSimpleARC(task_type, num_envs=16, grid_size=grid_size, num_colors=num_colors, device=device)

    episode_rewards = []
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

            for i, info in enumerate(infos):
                if "accuracy" in info:
                    episode_accuracies.append(info["accuracy"])
                    episode_rewards.append(reward[i].item())

            obs = next_obs

        # Compute returns
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

        f_obs = flat(b_obs)
        f_act = flat(b_actions)
        f_lp = flat(b_log_probs)
        f_adv = flat(b_adv)
        f_ret = flat(b_ret)

        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        batch_size = f_obs.shape[0]
        mb_size = batch_size // 4

        for _ in range(4):
            idx = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mb_size):
                end = start + mb_size
                mb = idx[start:end]

                _, new_lp, ent, new_v = model.get_action_and_value(f_obs[mb], f_act[mb])

                ratio = torch.exp(new_lp - f_lp[mb])
                pg1 = -f_adv[mb] * ratio
                pg2 = -f_adv[mb] * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss = F.mse_loss(new_v, f_ret[mb])
                ent_loss = -ent.mean()

                loss = pg_loss + 0.5 * v_loss + 0.01 * ent_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        if (update + 1) % 20 == 0:
            mean_acc = sum(episode_accuracies[-100:]) / max(len(episode_accuracies[-100:]), 1)
            print(f"Update {update + 1}/{num_updates} | Accuracy: {mean_acc:.1%}", flush=True)

    # Eval
    model.eval()
    successes = 0

    for i in range(100):
        test_env = SimpleARCEnv(grid_size, task_type, num_colors, seed=99999 + i)
        obs = test_env.reset().unsqueeze(0).to(device)
        done = False

        while not done:
            with torch.no_grad():
                action, _, _, _ = model.get_action_and_value(obs)
            obs, _, done, info = test_env.step(action.item())
            obs = obs.unsqueeze(0).to(device)

            if done and info.get("success", False):
                successes += 1

    success_rate = successes / 100
    print(f"EVAL SUCCESS: {success_rate:.1%}", flush=True)
    return success_rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    results = {}
    for task in ["copy", "fill_single", "color_swap", "reflect_h"]:
        results[task] = train_simple_arc(task, num_updates=200, device=device)

    print("\n" + "="*50)
    print("SIMPLE ARC SUMMARY")
    print("="*50)

    for task, rate in results.items():
        status = "PASS" if rate > 0.5 else "FAIL"
        print(f"  {task}: {rate:.1%} [{status}]")


if __name__ == "__main__":
    main()
