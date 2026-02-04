"""
Phase 10 Validation: Replay Buffer Tests

Success Criteria:
- [x] ReplayBuffer instantiates and stores transitions
- [x] Uniform sampling works correctly
- [x] PrioritizedReplayBuffer with importance sampling
- [x] Priority updates work
- [x] EpisodeBuffer for trajectory storage
- [x] Observation resizing handles different grid sizes
"""

import torch


def test_replay_buffer_instantiation():
    """Test that replay buffer instantiates."""
    from aria_lite.training.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=1000)
    assert buffer is not None
    assert len(buffer) == 0


def test_replay_buffer_push():
    """Test adding transitions."""
    from aria_lite.training.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=100, observation_shape=(10, 10))

    for i in range(50):
        transition = Transition(
            observation=torch.randint(0, 16, (10, 10)),
            action=i % 8,
            reward=1.0 if i % 10 == 0 else 0.0,
            next_observation=torch.randint(0, 16, (10, 10)),
            done=i % 20 == 19,
        )
        buffer.push(transition)

    assert len(buffer) == 50


def test_replay_buffer_sample():
    """Test uniform sampling."""
    from aria_lite.training.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=100, observation_shape=(10, 10))

    # Fill buffer
    for i in range(100):
        transition = Transition(
            observation=torch.randint(0, 16, (10, 10)),
            action=i % 8,
            reward=float(i),
            next_observation=torch.randint(0, 16, (10, 10)),
            done=False,
        )
        buffer.push(transition)

    # Sample batch
    batch = buffer.sample(32)

    assert batch.observations.shape == (32, 10, 10)
    assert batch.actions.shape == (32,)
    assert batch.rewards.shape == (32,)
    assert batch.next_observations.shape == (32, 10, 10)
    assert batch.dones.shape == (32,)


def test_replay_buffer_circular():
    """Test circular buffer behavior."""
    from aria_lite.training.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=10, observation_shape=(5, 5))

    # Fill beyond capacity
    for i in range(25):
        transition = Transition(
            observation=torch.full((5, 5), i, dtype=torch.long),
            action=0,
            reward=float(i),
            next_observation=torch.zeros(5, 5, dtype=torch.long),
            done=False,
        )
        buffer.push(transition)

    assert len(buffer) == 10

    # Should contain only the last 10 transitions
    batch = buffer.sample(10)
    assert (batch.rewards >= 15).all()


def test_replay_buffer_resize():
    """Test observation resizing."""
    from aria_lite.training.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=10, observation_shape=(20, 20))

    # Push smaller observation
    small_obs = torch.randint(0, 16, (10, 10))
    transition = Transition(
        observation=small_obs,
        action=0,
        reward=1.0,
        next_observation=small_obs,
        done=False,
    )
    buffer.push(transition)

    # Push larger observation
    large_obs = torch.randint(0, 16, (30, 30))
    transition = Transition(
        observation=large_obs,
        action=0,
        reward=1.0,
        next_observation=large_obs,
        done=False,
    )
    buffer.push(transition)

    assert len(buffer) == 2

    batch = buffer.sample(2)
    assert batch.observations.shape == (2, 20, 20)


def test_replay_buffer_is_ready():
    """Test is_ready check."""
    from aria_lite.training.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=100, observation_shape=(5, 5))

    assert not buffer.is_ready(10)

    for i in range(10):
        buffer.push(
            Transition(
                observation=torch.zeros(5, 5, dtype=torch.long),
                action=0,
                reward=0.0,
                next_observation=torch.zeros(5, 5, dtype=torch.long),
                done=False,
            )
        )

    assert buffer.is_ready(10)
    assert not buffer.is_ready(20)


def test_prioritized_replay_buffer():
    """Test prioritized replay buffer."""
    from aria_lite.training.replay_buffer import PrioritizedReplayBuffer, Transition

    buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=(10, 10))

    # Add transitions with different priorities
    for i in range(50):
        transition = Transition(
            observation=torch.randint(0, 16, (10, 10)),
            action=i % 8,
            reward=float(i),
            next_observation=torch.randint(0, 16, (10, 10)),
            done=False,
        )
        buffer.push(transition, priority=float(i + 1))

    assert len(buffer) == 50

    # Sample should work
    batch = buffer.sample(16)

    assert batch.observations.shape == (16, 10, 10)
    assert batch.weights is not None
    assert batch.weights.shape == (16,)
    assert batch.indices is not None


def test_priority_update():
    """Test priority updates."""
    from aria_lite.training.replay_buffer import PrioritizedReplayBuffer, Transition

    buffer = PrioritizedReplayBuffer(capacity=100, observation_shape=(5, 5))

    # Add transitions
    for i in range(20):
        buffer.push(
            Transition(
                observation=torch.zeros(5, 5, dtype=torch.long),
                action=0,
                reward=0.0,
                next_observation=torch.zeros(5, 5, dtype=torch.long),
                done=False,
            ),
            priority=1.0,
        )

    # Sample and update priorities
    batch = buffer.sample(10)

    new_priorities = torch.rand(10) * 10
    buffer.update_priorities(batch.indices, new_priorities)

    # Should not crash
    batch2 = buffer.sample(10)
    assert batch2 is not None


def test_importance_weights():
    """Test importance sampling weights."""
    from aria_lite.training.replay_buffer import PrioritizedReplayBuffer, Transition

    buffer = PrioritizedReplayBuffer(
        capacity=100,
        observation_shape=(5, 5),
        alpha=0.6,
        beta=0.4,
    )

    # Add transitions with varying priorities
    for i in range(50):
        buffer.push(
            Transition(
                observation=torch.zeros(5, 5, dtype=torch.long),
                action=0,
                reward=0.0,
                next_observation=torch.zeros(5, 5, dtype=torch.long),
                done=False,
            ),
            priority=float(i + 1),
        )

    batch = buffer.sample(20)

    # Weights should be in (0, 1] range
    assert (batch.weights > 0).all()
    assert (batch.weights <= 1).all()


def test_episode_buffer():
    """Test episode-based buffer."""
    from aria_lite.training.replay_buffer import EpisodeBuffer, Transition

    buffer = EpisodeBuffer(capacity=10)

    # Add 3 episodes
    for ep in range(3):
        for step in range(5):
            done = step == 4
            buffer.push(
                Transition(
                    observation=torch.full((5, 5), ep, dtype=torch.long),
                    action=step,
                    reward=1.0 if done else 0.0,
                    next_observation=torch.zeros(5, 5, dtype=torch.long),
                    done=done,
                )
            )

    assert len(buffer) == 3
    assert buffer.total_transitions == 15


def test_episode_buffer_sample_episodes():
    """Test sampling full episodes."""
    from aria_lite.training.replay_buffer import EpisodeBuffer, Transition

    buffer = EpisodeBuffer(capacity=100)

    # Add episodes of varying lengths
    for ep in range(10):
        length = 5 + ep
        for step in range(length):
            buffer.push(
                Transition(
                    observation=torch.zeros(5, 5, dtype=torch.long),
                    action=0,
                    reward=0.0,
                    next_observation=torch.zeros(5, 5, dtype=torch.long),
                    done=step == length - 1,
                )
            )

    episodes = buffer.sample_episodes(3)
    assert len(episodes) == 3


def test_episode_buffer_sample_transitions():
    """Test sampling individual transitions from episodes."""
    from aria_lite.training.replay_buffer import EpisodeBuffer, Transition

    buffer = EpisodeBuffer(capacity=100)

    # Add episodes
    for ep in range(5):
        for step in range(10):
            buffer.push(
                Transition(
                    observation=torch.zeros(5, 5, dtype=torch.long),
                    action=step,
                    reward=0.0,
                    next_observation=torch.zeros(5, 5, dtype=torch.long),
                    done=step == 9,
                )
            )

    transitions = buffer.sample_transitions(20)
    assert len(transitions) == 20


def test_push_batch():
    """Test batch push."""
    from aria_lite.training.replay_buffer import ReplayBuffer

    buffer = ReplayBuffer(capacity=100, observation_shape=(10, 10))

    observations = torch.randint(0, 16, (32, 10, 10))
    actions = torch.randint(0, 8, (32,))
    rewards = torch.randn(32)
    next_observations = torch.randint(0, 16, (32, 10, 10))
    dones = torch.zeros(32, dtype=torch.bool)

    buffer.push_batch(observations, actions, rewards, next_observations, dones)

    assert len(buffer) == 32


def test_clear():
    """Test buffer clearing."""
    from aria_lite.training.replay_buffer import ReplayBuffer, Transition

    buffer = ReplayBuffer(capacity=100, observation_shape=(5, 5))

    for i in range(50):
        buffer.push(
            Transition(
                observation=torch.zeros(5, 5, dtype=torch.long),
                action=0,
                reward=0.0,
                next_observation=torch.zeros(5, 5, dtype=torch.long),
                done=False,
            )
        )

    assert len(buffer) == 50

    buffer.clear()
    assert len(buffer) == 0


if __name__ == "__main__":
    print("Phase 10 Validation: Replay Buffer Tests")
    print("=" * 40)

    test_replay_buffer_instantiation()
    print("✓ Replay buffer instantiation")

    test_replay_buffer_push()
    print("✓ Replay buffer push")

    test_replay_buffer_sample()
    print("✓ Replay buffer sample")

    test_replay_buffer_circular()
    print("✓ Replay buffer circular")

    test_replay_buffer_resize()
    print("✓ Replay buffer resize")

    test_replay_buffer_is_ready()
    print("✓ Replay buffer is_ready")

    test_prioritized_replay_buffer()
    print("✓ Prioritized replay buffer")

    test_priority_update()
    print("✓ Priority update")

    test_importance_weights()
    print("✓ Importance weights")

    test_episode_buffer()
    print("✓ Episode buffer")

    test_episode_buffer_sample_episodes()
    print("✓ Episode buffer sample episodes")

    test_episode_buffer_sample_transitions()
    print("✓ Episode buffer sample transitions")

    test_push_batch()
    print("✓ Push batch")

    test_clear()
    print("✓ Clear")

    print("\n" + "=" * 40)
    print("Phase 10 Validation: ALL TESTS PASSED")
    print("=" * 40)
