"""
Training and validation for state tracking primitives.

State tracking variants:
- memory: Remember shown value, recall later
- counter: Track budget while navigating
- multi_property: Remember color AND position
- sequence: Remember and reproduce sequence
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aria_lite.primitives import StateTrackingEnv
from src.aria_lite.primitives.base import Action


class StateTrackingEncoder(nn.Module):
    """
    Encoder with memory for state tracking tasks.

    Uses GRU to maintain state across steps.
    """

    def __init__(self, hidden_dim: int = 128, num_values: int = 16):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Cell embedding
        self.embed = nn.Embedding(num_values, 32)

        # CNN for spatial features
        self.conv = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        # Project CNN output
        self.fc = nn.Linear(128 * 16, hidden_dim)

        # GRU for memory
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None):
        """
        Encode observation with memory.

        Args:
            obs: [B, H, W] grid observation
            hidden: [1, B, hidden_dim] GRU hidden state

        Returns:
            features: [B, hidden_dim]
            new_hidden: [1, B, hidden_dim]
        """
        B = obs.shape[0]

        # Embed and conv
        x = self.embed(obs.long().clamp(0, 15))
        x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.reshape(B, -1)
        x = self.fc(x)

        # GRU update
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]
        if hidden is None:
            hidden = torch.zeros(1, B, self.hidden_dim, device=obs.device)

        output, new_hidden = self.gru(x, hidden)
        features = output.squeeze(1)  # [B, hidden_dim]

        return features, new_hidden


class StateTrackingPolicy(nn.Module):
    """Policy for state tracking with memory."""

    def __init__(self, hidden_dim: int = 128, max_grid: int = 20):
        super().__init__()
        self.encoder = StateTrackingEncoder(hidden_dim)

        # Action heads
        self.action_head = nn.Linear(hidden_dim, 9)
        self.x_head = nn.Linear(hidden_dim, max_grid)
        self.y_head = nn.Linear(hidden_dim, max_grid)

    def forward(self, obs: torch.Tensor, hidden: torch.Tensor = None, grid_size: int = 10):
        features, new_hidden = self.encoder(obs, hidden)

        action_logits = self.action_head(features)
        x_logits = self.x_head(features)[:, :grid_size]
        y_logits = self.y_head(features)[:, :grid_size]

        return {
            "action_logits": action_logits,
            "x_logits": x_logits,
            "y_logits": y_logits,
            "hidden": new_hidden,
        }


# ============================================================================
# Expert Solvers
# ============================================================================


def memory_expert(env: StateTrackingEnv, step: int, remembered_value: int = None):
    """Expert for memory variant."""
    if env.phase == "show":
        # Remember the value and advance
        return Action.INTERACT, 0, 0, env.target_value
    elif env.phase == "hide":
        # Click on remembered value (values shown at bottom row)
        x = remembered_value * 2 if remembered_value is not None else 0
        return Action.CLICK, x, 0, remembered_value
    return Action.NOOP, 0, 0, remembered_value


def counter_expert(env: StateTrackingEnv):
    """Expert for counter variant - navigate to target within budget."""
    if env.budget <= 0:
        return Action.NOOP, 0, 0

    ay, ax = env.agent_pos
    ty, tx = env.target_pos

    if ay < ty:
        return Action.DOWN, 0, 0
    elif ay > ty:
        return Action.UP, 0, 0
    elif ax < tx:
        return Action.RIGHT, 0, 0
    elif ax > tx:
        return Action.LEFT, 0, 0

    return Action.NOOP, 0, 0


def multi_property_expert(env: StateTrackingEnv, step: int, remembered: dict = None):
    """Expert for multi_property variant."""
    if remembered is None:
        remembered = {}

    if env.phase == "show":
        # Remember properties and advance
        remembered = env.properties.copy()
        return Action.INTERACT, 0, 0, remembered
    elif env.phase == "question":
        # Answer color question
        x = remembered.get("color", 1) - 1
        return Action.CLICK, x, 0, remembered
    elif env.phase == "position":
        # Answer position question
        return Action.CLICK, remembered.get("col", 0), remembered.get("row", 0), remembered

    return Action.NOOP, 0, 0, remembered


def sequence_expert(env: StateTrackingEnv, step: int, sequence: list = None):
    """Expert for sequence variant."""
    if sequence is None:
        sequence = []

    if env.phase == "show":
        # Remember current element and advance
        if hasattr(env, "current_show_idx") and env.current_show_idx < len(env.target_sequence):
            sequence = env.target_sequence.copy()
        return Action.INTERACT, 0, 0, sequence
    elif env.phase == "respond":
        # Reproduce sequence
        idx = len(env.response_sequence)
        if idx < len(sequence):
            x = sequence[idx] * 2
            return Action.CLICK, x, 0, sequence

    return Action.NOOP, 0, 0, sequence


# ============================================================================
# Training Functions
# ============================================================================


def collect_memory_episodes(num_episodes: int = 500):
    """Collect expert memory episodes."""
    episodes = []

    for _ in range(num_episodes):
        env = StateTrackingEnv(
            grid_size=8,
            num_values=4,
            max_steps=15,
            variant="memory",
        )
        obs = env.reset()

        observations = [obs.clone()]
        actions = []
        coords = []
        remembered = None

        for step in range(env.max_steps):
            action, x, y, remembered = memory_expert(env, step, remembered)
            result = env.step(action, x, y)

            actions.append(action)
            coords.append((x, y))
            observations.append(result.observation.clone())

            if result.done:
                break

        if result.success:
            episodes.append({
                "observations": observations[:-1],
                "actions": actions,
                "coords": coords,
            })

    return episodes


def collect_counter_episodes(num_episodes: int = 500):
    """Collect expert counter episodes."""
    episodes = []

    for _ in range(num_episodes):
        env = StateTrackingEnv(
            grid_size=10,
            budget=8,
            max_steps=20,
            variant="counter",
        )
        obs = env.reset()

        observations = [obs.clone()]
        actions = []

        for step in range(env.max_steps):
            action, _, _ = counter_expert(env)
            result = env.step(action)

            actions.append(action)
            observations.append(result.observation.clone())

            if result.done:
                break

        if result.success:
            episodes.append({
                "observations": observations[:-1],
                "actions": actions,
            })

    return episodes


def train_memory_bc(num_episodes: int = 500, num_epochs: int = 100):
    """Train memory variant with BC using sequence training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Collecting memory episodes...")
    episodes = collect_memory_episodes(num_episodes)
    print(f"Collected {len(episodes)} successful episodes")

    if len(episodes) < 10:
        print("Not enough episodes")
        return 0.0

    # Keep episodes as sequences for proper GRU training
    print(f"Episode lengths: {[len(ep['observations']) for ep in episodes[:5]]}")

    model = StateTrackingPolicy(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 32  # Batch of episodes

    for epoch in range(num_epochs):
        # Shuffle episodes
        import random
        random.shuffle(episodes)

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(episodes), batch_size):
            batch_eps = episodes[i:i+batch_size]

            # Process each episode with GRU state
            batch_loss = 0
            for ep in batch_eps:
                hidden = None
                ep_loss = 0

                for t, (obs, action, (x, y)) in enumerate(zip(
                    ep["observations"], ep["actions"], ep["coords"]
                )):
                    obs_t = obs.unsqueeze(0).to(device)
                    action_t = torch.tensor([action], device=device)
                    x_t = torch.tensor([min(x, 7)], device=device)
                    y_t = torch.tensor([min(y, 7)], device=device)

                    out = model(obs_t, hidden, grid_size=8)
                    hidden = out["hidden"].detach()  # Detach to avoid BPTT through time

                    step_loss = (
                        F.cross_entropy(out["action_logits"], action_t)
                        + F.cross_entropy(out["x_logits"], x_t)
                        + F.cross_entropy(out["y_logits"], y_t)
                    )
                    ep_loss += step_loss

                    pred_action = out["action_logits"].argmax(-1)
                    correct += (pred_action == action_t).sum().item()
                    total += 1

                batch_loss += ep_loss / len(ep["observations"])

            batch_loss = batch_loss / len(batch_eps)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.item()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/total:.1%}")

    # Evaluate
    successes = []
    for _ in range(100):
        env = StateTrackingEnv(grid_size=8, num_values=4, max_steps=15, variant="memory")
        obs = env.reset()
        hidden = None

        for _ in range(env.max_steps):
            with torch.no_grad():
                out = model(obs.unsqueeze(0).to(device), hidden, grid_size=8)
                action = out["action_logits"].argmax(-1).item()
                x = out["x_logits"].argmax(-1).item()
                y = out["y_logits"].argmax(-1).item()
                hidden = out["hidden"]

            result = env.step(action, x, y)
            obs = result.observation

            if result.done:
                successes.append(1.0 if result.success else 0.0)
                break

    success_rate = sum(successes) / len(successes)
    print(f"\nMemory BC eval: {success_rate:.1%}")
    return success_rate


def train_counter_bc(num_episodes: int = 500, num_epochs: int = 50):
    """Train counter variant with BC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Collecting counter episodes...")
    episodes = collect_counter_episodes(num_episodes)
    print(f"Collected {len(episodes)} successful episodes")

    if len(episodes) < 10:
        print("Not enough episodes")
        return 0.0

    # Flatten
    all_obs = []
    all_actions = []

    for ep in episodes:
        all_obs.extend(ep["observations"])
        all_actions.extend(ep["actions"])

    obs_tensor = torch.stack(all_obs)
    actions_tensor = torch.tensor(all_actions, dtype=torch.long)

    print(f"Dataset: {len(obs_tensor)} samples")

    # Simple navigation model (reuse from nav training)
    from src.aria_lite.train_primitives_bc import NavPolicy

    model = NavPolicy(hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64

    for epoch in range(num_epochs):
        perm = torch.randperm(len(obs_tensor))
        obs_tensor = obs_tensor[perm]
        actions_tensor = actions_tensor[perm]

        total_loss = 0
        correct = 0

        for i in range(0, len(obs_tensor), batch_size):
            obs_batch = obs_tensor[i:i+batch_size].to(device)
            actions_batch = actions_tensor[i:i+batch_size].to(device)

            logits = model(obs_batch)
            loss = F.cross_entropy(logits, actions_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(-1) == actions_batch).sum().item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/len(obs_tensor):.1%}")

    # Evaluate
    successes = []
    for _ in range(100):
        env = StateTrackingEnv(grid_size=10, budget=8, max_steps=20, variant="counter")
        obs = env.reset()

        for _ in range(env.max_steps):
            with torch.no_grad():
                logits = model(obs.unsqueeze(0).to(device))
                action = logits.argmax(-1).item()

            result = env.step(action)
            obs = result.observation

            if result.done:
                successes.append(1.0 if result.success else 0.0)
                break

    success_rate = sum(successes) / len(successes)
    print(f"\nCounter BC eval: {success_rate:.1%}")
    return success_rate


if __name__ == "__main__":
    print("=" * 60)
    print("STATE TRACKING PRIMITIVES")
    print("=" * 60)

    print("\n--- Memory Variant ---")
    memory_success = train_memory_bc()

    print("\n--- Counter Variant ---")
    counter_success = train_counter_bc()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Memory: {memory_success:.1%} {'PASS' if memory_success > 0.5 else 'FAIL'}")
    print(f"Counter: {counter_success:.1%} {'PASS' if counter_success > 0.5 else 'FAIL'}")
