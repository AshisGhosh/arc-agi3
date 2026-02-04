"""
ARIA-Lite Replay Buffer

Experience replay buffer for storing and sampling transitions.
Supports both uniform and prioritized sampling strategies.

Features:
- Store (observation, action, reward, next_observation, done) transitions
- Uniform random sampling
- Prioritized experience replay (PER) with importance sampling
- Episode-based storage for trajectory sampling
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class Transition:
    """Single transition in the replay buffer."""

    observation: torch.Tensor  # [H, W] grid
    action: int
    reward: float
    next_observation: torch.Tensor  # [H, W] grid
    done: bool
    info: Optional[dict] = None


@dataclass
class TransitionBatch:
    """Batch of transitions for training."""

    observations: torch.Tensor  # [B, H, W]
    actions: torch.Tensor  # [B]
    rewards: torch.Tensor  # [B]
    next_observations: torch.Tensor  # [B, H, W]
    dones: torch.Tensor  # [B]
    indices: Optional[torch.Tensor] = None  # For priority updates
    weights: Optional[torch.Tensor] = None  # Importance sampling weights


class ReplayBuffer:
    """
    Uniform replay buffer for experience storage.

    Stores transitions in a circular buffer and samples uniformly.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        observation_shape: tuple[int, int] = (64, 64),
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape

        # Pre-allocate storage
        self.observations = torch.zeros(capacity, *observation_shape, dtype=torch.long)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_observations = torch.zeros(capacity, *observation_shape, dtype=torch.long)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        self.position = 0
        self.size = 0

    def push(self, transition: Transition):
        """Add a transition to the buffer."""
        obs = transition.observation
        next_obs = transition.next_observation

        # Pad or crop to fit observation_shape
        obs = self._resize_observation(obs)
        next_obs = self._resize_observation(next_obs)

        self.observations[self.position] = obs
        self.actions[self.position] = transition.action
        self.rewards[self.position] = transition.reward
        self.next_observations[self.position] = next_obs
        self.dones[self.position] = transition.done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
    ):
        """Add a batch of transitions."""
        batch_size = observations.shape[0]

        for i in range(batch_size):
            self.push(
                Transition(
                    observation=observations[i],
                    action=actions[i].item(),
                    reward=rewards[i].item(),
                    next_observation=next_observations[i],
                    done=dones[i].item(),
                )
            )

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample a random batch of transitions."""
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")

        indices = torch.randint(0, self.size, (batch_size,))

        return TransitionBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
            indices=indices,
            weights=torch.ones(batch_size),
        )

    def _resize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Resize observation to fit buffer shape."""
        H, W = self.observation_shape
        oh, ow = obs.shape[-2:]

        if oh == H and ow == W:
            return obs

        # Create padded/cropped version
        result = torch.zeros(H, W, dtype=obs.dtype)

        # Copy what fits
        copy_h = min(oh, H)
        copy_w = min(ow, W)
        result[:copy_h, :copy_w] = obs[:copy_h, :copy_w]

        return result

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= min_size

    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0


class SumTree:
    """
    Sum tree for efficient prioritized sampling.

    Allows O(log n) updates and O(log n) sampling proportional to priorities.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index for a given value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    def update(self, idx: int, priority: float):
        """Update priority at index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority: float) -> int:
        """Add new priority, return data index."""
        tree_idx = self.data_pointer + self.capacity - 1

        self.update(tree_idx, priority)

        data_idx = self.data_pointer
        self.data_pointer = (self.data_pointer + 1) % self.capacity

        return data_idx

    def get(self, value: float) -> tuple[int, float, int]:
        """
        Get leaf for value.

        Returns: (tree_idx, priority, data_idx)
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx].item(), data_idx

    @property
    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0].item()


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.

    Samples transitions proportional to their TD error priority.
    Uses importance sampling weights for unbiased updates.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        observation_shape: tuple[int, int] = (64, 64),
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        self.capacity = capacity
        self.observation_shape = observation_shape
        self.alpha = alpha  # Priority exponent
        self.beta = beta  # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon  # Small constant for non-zero priorities

        # Priority tree
        self.tree = SumTree(capacity)

        # Data storage
        self.observations = torch.zeros(capacity, *observation_shape, dtype=torch.long)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.next_observations = torch.zeros(capacity, *observation_shape, dtype=torch.long)
        self.dones = torch.zeros(capacity, dtype=torch.bool)

        self.max_priority = 1.0
        self.size = 0

    def push(self, transition: Transition, priority: Optional[float] = None):
        """Add transition with priority."""
        if priority is None:
            priority = self.max_priority

        priority = (priority + self.epsilon) ** self.alpha

        idx = self.tree.add(priority)

        obs = self._resize_observation(transition.observation)
        next_obs = self._resize_observation(transition.next_observation)

        self.observations[idx] = obs
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.next_observations[idx] = next_obs
        self.dones[idx] = transition.done

        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Sample batch proportional to priorities."""
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")

        indices = []
        tree_indices = []
        priorities = []

        # Divide total priority into segments
        segment = self.tree.total / batch_size

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = torch.rand(1).item() * (high - low) + low

            tree_idx, priority, data_idx = self.tree.get(value)

            indices.append(data_idx)
            tree_indices.append(tree_idx)
            priorities.append(priority)

        indices = torch.tensor(indices, dtype=torch.long)
        tree_indices = torch.tensor(tree_indices, dtype=torch.long)
        priorities = torch.tensor(priorities, dtype=torch.float32)

        # Compute importance sampling weights
        min_priority = priorities.min() / self.tree.total
        max_weight = (min_priority * self.size) ** (-self.beta)

        weights = (priorities / self.tree.total * self.size) ** (-self.beta)
        weights = weights / max_weight  # Normalize

        return TransitionBatch(
            observations=self.observations[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_observations=self.next_observations[indices],
            dones=self.dones[indices],
            indices=tree_indices,
            weights=weights,
        )

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices.tolist(), priorities.tolist()):
            priority = (abs(priority) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def _resize_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Resize observation to fit buffer shape."""
        H, W = self.observation_shape
        oh, ow = obs.shape[-2:]

        if oh == H and ow == W:
            return obs

        result = torch.zeros(H, W, dtype=obs.dtype)
        copy_h = min(oh, H)
        copy_w = min(ow, W)
        result[:copy_h, :copy_w] = obs[:copy_h, :copy_w]

        return result

    def __len__(self) -> int:
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= min_size


class EpisodeBuffer:
    """
    Episode-based buffer for trajectory storage.

    Useful for algorithms that need full episode context.
    """

    def __init__(self, capacity: int = 10_000):
        self.capacity = capacity
        self.episodes: list[list[Transition]] = []
        self.current_episode: list[Transition] = []

    def push(self, transition: Transition):
        """Add transition to current episode."""
        self.current_episode.append(transition)

        if transition.done:
            self.end_episode()

    def end_episode(self):
        """End current episode and store it."""
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []

            # Remove old episodes if over capacity
            while len(self.episodes) > self.capacity:
                self.episodes.pop(0)

    def sample_episodes(self, num_episodes: int) -> list[list[Transition]]:
        """Sample random episodes."""
        if len(self.episodes) < num_episodes:
            return self.episodes.copy()

        indices = torch.randint(0, len(self.episodes), (num_episodes,))
        return [self.episodes[i] for i in indices.tolist()]

    def sample_transitions(self, batch_size: int) -> list[Transition]:
        """Sample random transitions from all episodes."""
        all_transitions = [t for ep in self.episodes for t in ep]

        if len(all_transitions) < batch_size:
            return all_transitions

        indices = torch.randint(0, len(all_transitions), (batch_size,))
        return [all_transitions[i] for i in indices.tolist()]

    def __len__(self) -> int:
        return len(self.episodes)

    @property
    def total_transitions(self) -> int:
        return sum(len(ep) for ep in self.episodes)
