#!/usr/bin/env python3
"""
Simple test: just train copy on 5x5 directly, no curriculum.
Use the PositionAwareARCEncoder that worked before (87% on copy).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .training.arc_like_simple import SimpleARCEnv
from .training.arc_position_encoder import PositionAwareARCEncoder
from .training.ppo_trainer import RolloutBuffer, compute_gae


class SimpleActorCritic(nn.Module):
    def __init__(self, num_colors=4, grid_size=5):
        super().__init__()
        self.encoder = PositionAwareARCEncoder(
            num_colors=num_colors,
            grid_size=grid_size,
            embed_dim=64,
            hidden_dim=128,
            output_dim=128,
            num_heads=4,
        )
        self.policy = nn.Linear(128, num_colors)
        self.value = nn.Linear(128, 1)

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
    def __init__(self, task_type, num_envs, grid_size=5, num_colors=4, device="cuda"):
        # Training envs: truly random (no fixed seeds)
        self.envs = [SimpleARCEnv(grid_size, task_type, num_colors, deterministic=False)
                     for _ in range(num_envs)]
        self.num_envs = num_envs
        self.device = device

    def reset(self):
        return torch.stack([e.reset() for e in self.envs]).to(self.device)

    def step(self, actions):
        results = [e.step(a.item()) for e, a in zip(self.envs, actions)]
        obs = [e.reset() if r[2] else r[0] for e, r in zip(self.envs, results)]
        return (torch.stack(obs).to(self.device),
                torch.tensor([r[1] for r in results], device=self.device),
                torch.tensor([r[2] for r in results], dtype=torch.float32, device=self.device),
                [r[3] for r in results])


def train(task_type="copy", grid_size=5, num_updates=300, device="cuda"):
    print(f"\nTask: {task_type}, Grid: {grid_size}x{grid_size}")

    model = SimpleActorCritic(num_colors=4, grid_size=grid_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

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

        if (update + 1) % 30 == 0:
            acc = sum(episode_accs[-100:]) / max(len(episode_accs[-100:]), 1)
            print(f"  Update {update+1}: {acc:.1%}", flush=True)

    # Eval (deterministic for reproducibility)
    model.eval()
    success = 0
    for i in range(100):
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
    rate = success / 100
    print(f"  EVAL: {rate:.1%}", flush=True)
    return rate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("\n*** Baseline test: Direct training without curriculum ***\n")

    # Test on different grid sizes directly
    results = {}
    for size in [3, 4, 5]:
        for task in ["copy", "reflect_h"]:
            key = f"{task}_{size}x{size}"
            results[key] = train(task, size, 300, device)

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for k, v in results.items():
        print(f"  {k}: {v:.1%}")


if __name__ == "__main__":
    main()
