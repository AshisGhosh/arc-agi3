"""Tests for expert solvers."""

import torch


def test_navigation_solver():
    """Test A* navigation solver."""
    from aria_lite.experts import NavigationSolver
    from aria_lite.training.synthetic_env import SyntheticEnv

    solver = NavigationSolver()

    # Test on several environments
    successes = 0
    for seed in range(20):
        env = SyntheticEnv(grid_size=10, mechanics=["navigation"], seed=seed)
        env.reset()

        if not solver.can_solve(env.state):
            continue

        # Run solver
        steps = 0
        while steps < 100:
            result = solver.solve(env.state)
            if not result.solved:
                break

            step_result = env.step(result.action)
            steps += 1

            if step_result.done and step_result.reward > 0:
                successes += 1
                break

    # Should solve most navigation tasks
    assert successes >= 15, f"Only solved {successes}/20 navigation tasks"


def test_collection_solver():
    """Test collection solver."""
    from aria_lite.experts import CollectionSolver
    from aria_lite.training.synthetic_env import SyntheticEnv

    solver = CollectionSolver()

    successes = 0
    for seed in range(20):
        env = SyntheticEnv(grid_size=10, mechanics=["collection"], seed=seed)
        env.reset()

        if not solver.can_solve(env.state):
            continue

        steps = 0
        while steps < 100:
            result = solver.solve(env.state)
            if not result.solved:
                break

            step_result = env.step(result.action)
            steps += 1

            if step_result.done and step_result.reward > 0:
                successes += 1
                break

    assert successes >= 15, f"Only solved {successes}/20 collection tasks"


def test_switches_solver():
    """Test switches solver."""
    from aria_lite.experts import SwitchesSolver
    from aria_lite.training.synthetic_env import SyntheticEnv

    solver = SwitchesSolver()

    successes = 0
    for seed in range(20):
        env = SyntheticEnv(grid_size=10, mechanics=["switches"], seed=seed)
        env.reset()

        if not solver.can_solve(env.state):
            continue

        steps = 0
        while steps < 100:
            result = solver.solve(env.state)
            if not result.solved:
                break

            step_result = env.step(result.action)
            steps += 1

            if step_result.done and step_result.reward > 0:
                successes += 1
                break

    assert successes >= 15, f"Only solved {successes}/20 switch tasks"


def test_astar_basic():
    """Test A* pathfinding."""
    from aria_lite.experts.navigation import astar
    from aria_lite.training.synthetic_env import CellType

    # Simple grid
    grid = torch.zeros(5, 5, dtype=torch.long)
    grid[0, :] = CellType.WALL
    grid[-1, :] = CellType.WALL
    grid[:, 0] = CellType.WALL
    grid[:, -1] = CellType.WALL

    # Find path from (1,1) to (3,3)
    path = astar(grid, (1, 1), (3, 3))

    assert path is not None
    assert path[0] == (1, 1)
    assert path[-1] == (3, 3)
    assert len(path) == 5  # Manhattan distance + 1


def test_path_to_actions():
    """Test path conversion to actions."""
    from aria_lite.experts.navigation import path_to_actions
    from aria_lite.training.synthetic_env import Action

    path = [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
    actions = path_to_actions(path)

    assert actions == [Action.RIGHT, Action.RIGHT, Action.DOWN, Action.DOWN]


def test_expert_data_collection():
    """Test collecting expert trajectories."""
    from aria_lite.training.expert_data import collect_expert_dataset

    dataset = collect_expert_dataset(
        mechanic="navigation",
        num_trajectories=50,
        grid_size=10,
    )

    assert len(dataset.trajectories) >= 40  # Should get most
    assert dataset.success_rate > 0.8
    assert dataset.num_transitions > 0


if __name__ == "__main__":
    print("Testing expert solvers...")

    test_astar_basic()
    print("  A* basic: PASS")

    test_path_to_actions()
    print("  Path to actions: PASS")

    test_navigation_solver()
    print("  Navigation solver: PASS")

    test_collection_solver()
    print("  Collection solver: PASS")

    test_switches_solver()
    print("  Switches solver: PASS")

    test_expert_data_collection()
    print("  Expert data collection: PASS")

    print("\nAll expert tests passed!")
