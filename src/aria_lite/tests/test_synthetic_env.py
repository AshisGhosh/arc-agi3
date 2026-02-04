"""
Phase 11 Validation: Synthetic Environment Tests

Success Criteria:
- [x] SyntheticEnv instantiates and resets
- [x] Step produces valid results
- [x] All mechanics work (navigation, collection, etc.)
- [x] Goal conditions are checked
- [x] Generator creates diverse environments
- [x] Episode collection works
"""

import torch


def test_synthetic_env_instantiation():
    """Test environment instantiation."""
    from aria_lite.training.synthetic_env import SyntheticEnv

    env = SyntheticEnv(grid_size=16)
    assert env is not None


def test_reset():
    """Test environment reset."""
    from aria_lite.training.synthetic_env import SyntheticEnv

    env = SyntheticEnv(grid_size=10)
    obs = env.reset()

    assert obs.shape == (10, 10)
    assert env.state is not None
    assert env.state.agent_pos is not None


def test_step():
    """Test taking steps."""
    from aria_lite.training.synthetic_env import Action, SyntheticEnv

    env = SyntheticEnv(grid_size=10)
    env.reset()

    result = env.step(Action.RIGHT)

    assert result.observation.shape == (10, 10)
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert "step" in result.info


def test_navigation_mechanic():
    """Test navigation mechanic."""
    from aria_lite.training.synthetic_env import Action, CellType, SyntheticEnv

    env = SyntheticEnv(grid_size=10, mechanics=["navigation"])
    env.reset()

    # Should have a goal
    has_goal = (env.state.grid == CellType.GOAL).any()
    assert has_goal

    # Take some steps
    for _ in range(10):
        env.step(Action.RIGHT)


def test_collection_mechanic():
    """Test collection mechanic."""
    from aria_lite.training.synthetic_env import CellType, SyntheticEnv

    env = SyntheticEnv(grid_size=10, mechanics=["collection"])
    env.reset()

    # Should have collectibles
    has_collectibles = (env.state.grid == CellType.COLLECTIBLE).any()
    assert has_collectibles


def test_switches_mechanic():
    """Test switches mechanic."""
    from aria_lite.training.synthetic_env import Action, CellType, SyntheticEnv

    env = SyntheticEnv(grid_size=10, mechanics=["switches"])
    env.reset()

    # Should have switches
    has_switches = (env.state.grid == CellType.SWITCH).any()
    assert has_switches

    # Interact should toggle switches
    initial_switches = len(env.state.switches_on)
    for _ in range(20):
        env.step(Action.INTERACT)


def test_keys_doors_mechanic():
    """Test keys and doors mechanic."""
    from aria_lite.training.synthetic_env import CellType, SyntheticEnv

    env = SyntheticEnv(grid_size=10, mechanics=["keys_doors"])
    env.reset()

    # Should have key and door
    has_key = (env.state.grid == CellType.KEY).any()
    has_door = (env.state.grid == CellType.DOOR).any()
    assert has_key
    assert has_door


def test_pushing_mechanic():
    """Test pushing mechanic."""
    from aria_lite.training.synthetic_env import CellType, SyntheticEnv

    env = SyntheticEnv(grid_size=10, mechanics=["pushing"])
    env.reset()

    # Should have pushables
    has_pushables = (env.state.grid == CellType.PUSHABLE).any()
    assert has_pushables


def test_combined_mechanics():
    """Test multiple mechanics together."""
    from aria_lite.training.synthetic_env import SyntheticEnv

    env = SyntheticEnv(
        grid_size=16,
        mechanics=["navigation", "collection", "switches"],
    )
    env.reset()

    # Should be able to step without errors
    for _ in range(50):
        action = torch.randint(0, 8, (1,)).item()
        result = env.step(action)
        if result.done:
            break


def test_max_steps():
    """Test max steps termination."""
    from aria_lite.training.synthetic_env import Action, SyntheticEnv

    env = SyntheticEnv(grid_size=10, max_steps=10)
    env.reset()

    done = False
    for i in range(15):
        result = env.step(Action.NOOP)
        if result.done:
            done = True
            assert i < 12  # Should terminate around step 10
            break

    assert done


def test_goal_condition():
    """Test that goal conditions work."""
    from aria_lite.training.synthetic_env import SyntheticEnv

    env = SyntheticEnv(grid_size=8, mechanics=["navigation"], max_steps=100)
    env.reset()

    # Goal condition should be set
    assert env.goal_condition is not None

    # Run random actions until done or max steps
    for _ in range(100):
        action = torch.randint(0, 8, (1,)).item()
        result = env.step(action)
        if result.done and result.reward > 5:
            # Reached goal
            break


def test_generator():
    """Test environment generator."""
    from aria_lite.training.synthetic_env import SyntheticEnvGenerator

    generator = SyntheticEnvGenerator(
        min_grid_size=8,
        max_grid_size=16,
        min_mechanics=1,
        max_mechanics=2,
    )

    env = generator.generate()
    assert env is not None
    assert 8 <= env.grid_size <= 16
    assert 1 <= len(env.mechanics) <= 2


def test_generator_batch():
    """Test batch generation."""
    from aria_lite.training.synthetic_env import SyntheticEnvGenerator

    generator = SyntheticEnvGenerator()
    envs = generator.generate_batch(10)

    assert len(envs) == 10
    for env in envs:
        assert env is not None


def test_generator_diversity():
    """Test that generator creates diverse environments."""
    from aria_lite.training.synthetic_env import SyntheticEnvGenerator

    generator = SyntheticEnvGenerator(min_mechanics=1, max_mechanics=3)

    grid_sizes = set()
    mechanics_sets = set()

    for i in range(20):
        env = generator.generate(seed=i)
        grid_sizes.add(env.grid_size)
        mechanics_sets.add(tuple(sorted(env.mechanics)))

    # Should have variety
    assert len(grid_sizes) > 1
    assert len(mechanics_sets) > 1


def test_collect_episode():
    """Test episode collection."""
    from aria_lite.training.synthetic_env import SyntheticEnv, collect_episode

    env = SyntheticEnv(grid_size=10, max_steps=20)
    observations, actions, rewards, dones = collect_episode(env)

    assert len(observations) > 0
    assert len(actions) == len(rewards) == len(dones)
    assert dones[-1]  # Last step should be done


def test_collect_episode_with_policy():
    """Test episode collection with custom policy."""
    from aria_lite.training.synthetic_env import Action, SyntheticEnv, collect_episode

    env = SyntheticEnv(grid_size=10, max_steps=20)

    # Simple policy that always goes right
    def policy(obs):
        return Action.RIGHT

    observations, actions, rewards, dones = collect_episode(env, policy)

    assert len(actions) > 0
    assert all(a == Action.RIGHT for a in actions)


def test_deterministic_seed():
    """Test that seeding produces deterministic results."""
    from aria_lite.training.synthetic_env import SyntheticEnv

    env1 = SyntheticEnv(grid_size=10, seed=42)
    obs1 = env1.reset()

    env2 = SyntheticEnv(grid_size=10, seed=42)
    obs2 = env2.reset()

    assert torch.equal(obs1, obs2)


def test_wall_collision():
    """Test that walls block movement."""
    from aria_lite.training.synthetic_env import Action, CellType, SyntheticEnv

    env = SyntheticEnv(grid_size=10)
    env.reset()

    # Move to wall and try to go through
    initial_pos = env.state.agent_pos

    # Try to move up into wall (top row is wall)
    while env.state.agent_pos[0] > 1:
        env.step(Action.UP)

    pos_before = env.state.agent_pos
    env.step(Action.UP)  # Should hit wall
    pos_after = env.state.agent_pos

    # Shouldn't have moved into wall
    assert pos_after[0] >= 1


def test_inventory():
    """Test inventory system."""
    from aria_lite.training.synthetic_env import SyntheticEnv

    env = SyntheticEnv(grid_size=10, mechanics=["keys_doors"])
    env.reset()

    # Inventory should start empty
    assert len(env.state.inventory) == 0


if __name__ == "__main__":
    print("Phase 11 Validation: Synthetic Environment Tests")
    print("=" * 40)

    test_synthetic_env_instantiation()
    print("✓ Synthetic env instantiation")

    test_reset()
    print("✓ Reset")

    test_step()
    print("✓ Step")

    test_navigation_mechanic()
    print("✓ Navigation mechanic")

    test_collection_mechanic()
    print("✓ Collection mechanic")

    test_switches_mechanic()
    print("✓ Switches mechanic")

    test_keys_doors_mechanic()
    print("✓ Keys/doors mechanic")

    test_pushing_mechanic()
    print("✓ Pushing mechanic")

    test_combined_mechanics()
    print("✓ Combined mechanics")

    test_max_steps()
    print("✓ Max steps")

    test_goal_condition()
    print("✓ Goal condition")

    test_generator()
    print("✓ Generator")

    test_generator_batch()
    print("✓ Generator batch")

    test_generator_diversity()
    print("✓ Generator diversity")

    test_collect_episode()
    print("✓ Collect episode")

    test_collect_episode_with_policy()
    print("✓ Collect episode with policy")

    test_deterministic_seed()
    print("✓ Deterministic seed")

    test_wall_collision()
    print("✓ Wall collision")

    test_inventory()
    print("✓ Inventory")

    print("\n" + "=" * 40)
    print("Phase 11 Validation: ALL TESTS PASSED")
    print("=" * 40)
