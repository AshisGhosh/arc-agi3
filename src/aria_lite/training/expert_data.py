"""Expert data collection for behavioral cloning."""

from dataclasses import dataclass
from typing import Optional

import torch

from ..experts import CollectionSolver, ExpertSolver, NavigationSolver, SwitchesSolver
from .synthetic_env import SyntheticEnv


@dataclass
class ExpertTrajectory:
    """A single expert trajectory."""

    observations: list[torch.Tensor]  # [T+1] observations
    actions: list[int]  # [T] actions taken
    rewards: list[float]  # [T] rewards received
    mechanic: str  # Which mechanic this is for
    success: bool  # Did it reach the goal?


@dataclass
class ExpertDataset:
    """Collection of expert trajectories."""

    trajectories: list[ExpertTrajectory]

    @property
    def num_transitions(self) -> int:
        return sum(len(t.actions) for t in self.trajectories)

    @property
    def success_rate(self) -> float:
        if not self.trajectories:
            return 0.0
        return sum(t.success for t in self.trajectories) / len(self.trajectories)

    def get_tensors(
        self, max_obs_size: int = 64
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get stacked tensors for training.

        Returns:
            observations: [N, max_obs_size, max_obs_size]
            masks: [N, max_obs_size, max_obs_size] - True where padding (invalid)
            actions: [N]
        """
        all_obs = []
        all_masks = []
        all_actions = []

        for traj in self.trajectories:
            for i, action in enumerate(traj.actions):
                obs = traj.observations[i]
                h, w = obs.shape

                # Pad observation
                padded = torch.zeros(max_obs_size, max_obs_size, dtype=obs.dtype)
                padded[:h, :w] = obs
                all_obs.append(padded)

                # Create mask (True = invalid/padding)
                mask = torch.ones(max_obs_size, max_obs_size, dtype=torch.bool)
                mask[:h, :w] = False
                all_masks.append(mask)

                all_actions.append(action)

        return (
            torch.stack(all_obs),
            torch.stack(all_masks),
            torch.tensor(all_actions, dtype=torch.long),
        )


def get_solver_for_mechanic(mechanic: str) -> Optional[ExpertSolver]:
    """Get the appropriate solver for a mechanic."""
    solvers = {
        "navigation": NavigationSolver(),
        "collection": CollectionSolver(),
        "switches": SwitchesSolver(),
    }
    return solvers.get(mechanic)


def collect_expert_trajectory(
    env: SyntheticEnv,
    solver: ExpertSolver,
    max_steps: int = 100,
) -> Optional[ExpertTrajectory]:
    """
    Collect a single expert trajectory.

    Returns None if solver can't solve or fails.
    """
    obs = env.reset()
    observations = [obs.clone()]
    actions = []
    rewards = []

    for _ in range(max_steps):
        if not solver.can_solve(env.state):
            return None

        result = solver.solve(env.state)

        if not result.solved:
            return None

        step_result = env.step(result.action)

        actions.append(result.action)
        rewards.append(step_result.reward)
        observations.append(step_result.observation.clone())

        if step_result.done:
            break

    # Check if we actually succeeded (positive reward indicates goal)
    success = sum(rewards) > 0

    return ExpertTrajectory(
        observations=observations,
        actions=actions,
        rewards=rewards,
        mechanic=env.mechanics[0] if env.mechanics else "unknown",
        success=success,
    )


def collect_expert_dataset(
    mechanic: str,
    num_trajectories: int = 1000,
    grid_size: int = 10,
    max_steps: int = 50,
    seed_offset: int = 0,
) -> ExpertDataset:
    """
    Collect expert dataset for a specific mechanic.

    Args:
        mechanic: Which mechanic to collect for
        num_trajectories: Number of trajectories to collect
        grid_size: Size of the grid
        max_steps: Maximum steps per episode
        seed_offset: Offset for random seeds

    Returns:
        ExpertDataset with collected trajectories
    """
    solver = get_solver_for_mechanic(mechanic)
    if solver is None:
        raise ValueError(f"No solver for mechanic: {mechanic}")

    trajectories = []
    attempts = 0
    max_attempts = num_trajectories * 3  # Allow some failures

    while len(trajectories) < num_trajectories and attempts < max_attempts:
        env = SyntheticEnv(
            grid_size=grid_size,
            mechanics=[mechanic],
            max_steps=max_steps,
            seed=seed_offset + attempts,
        )

        traj = collect_expert_trajectory(env, solver, max_steps)
        if traj is not None and traj.success:
            trajectories.append(traj)

        attempts += 1

    return ExpertDataset(trajectories=trajectories)


def collect_mixed_dataset(
    mechanics: list[str],
    trajectories_per_mechanic: int = 1000,
    **kwargs,
) -> ExpertDataset:
    """Collect dataset with multiple mechanics."""
    all_trajectories = []

    for i, mechanic in enumerate(mechanics):
        dataset = collect_expert_dataset(
            mechanic=mechanic,
            num_trajectories=trajectories_per_mechanic,
            seed_offset=i * 100000,
            **kwargs,
        )
        all_trajectories.extend(dataset.trajectories)
        print(f"  {mechanic}: {len(dataset.trajectories)} trajectories, "
              f"{dataset.success_rate:.1%} success")

    return ExpertDataset(trajectories=all_trajectories)
