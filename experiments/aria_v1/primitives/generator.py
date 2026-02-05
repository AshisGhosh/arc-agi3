"""
Unified primitive generator for training.

Generates diverse primitive tasks across all families.
"""

import random
from typing import Optional

from .base import PrimitiveEnv, PrimitiveFamily
from .click import ClickPrimitiveGenerator
from .composition import CompositionGenerator
from .navigation import NavigationPrimitiveGenerator
from .pattern import PatternPrimitiveGenerator
from .state_tracking import StateTrackingGenerator


class PrimitiveGenerator:
    """
    Unified generator for all primitive task families.

    Used to create diverse training curriculum.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

        # Initialize family generators
        self.generators = {
            PrimitiveFamily.NAVIGATION: NavigationPrimitiveGenerator(seed),
            PrimitiveFamily.CLICK: ClickPrimitiveGenerator(seed),
            PrimitiveFamily.PATTERN: PatternPrimitiveGenerator(seed),
            PrimitiveFamily.STATE_TRACKING: StateTrackingGenerator(seed),
            PrimitiveFamily.COMPOSITION: CompositionGenerator(seed),
        }

        # Default family weights (can be adjusted for curriculum)
        self.family_weights = {
            PrimitiveFamily.NAVIGATION: 1.0,
            PrimitiveFamily.CLICK: 1.0,
            PrimitiveFamily.PATTERN: 1.0,
            PrimitiveFamily.STATE_TRACKING: 1.0,
            PrimitiveFamily.COMPOSITION: 0.5,  # Harder, sample less frequently initially
        }

    def generate(
        self,
        family: Optional[PrimitiveFamily] = None,
        difficulty: int = 1,
        deterministic: bool = False,
    ) -> PrimitiveEnv:
        """
        Generate a primitive task.

        Args:
            family: Specific family, or None for weighted random
            difficulty: 1-5, higher = harder
            deterministic: Whether to use fixed seeds

        Returns:
            PrimitiveEnv instance
        """
        if family is None:
            family = self._sample_family()

        generator = self.generators[family]
        return generator.generate(difficulty=difficulty, deterministic=deterministic)

    def generate_batch(
        self,
        n: int,
        difficulty_range: tuple[int, int] = (1, 5),
        families: Optional[list[PrimitiveFamily]] = None,
        deterministic: bool = False,
    ) -> list[PrimitiveEnv]:
        """
        Generate batch of diverse primitive tasks.

        Args:
            n: Number of tasks to generate
            difficulty_range: (min, max) difficulty
            families: List of families to sample from, or None for all
            deterministic: Whether to use fixed seeds

        Returns:
            List of PrimitiveEnv instances
        """
        envs = []
        for _ in range(n):
            if families:
                family = self.rng.choice(families)
            else:
                family = self._sample_family()

            difficulty = self.rng.randint(difficulty_range[0], difficulty_range[1])
            env = self.generate(family=family, difficulty=difficulty, deterministic=deterministic)
            envs.append(env)

        return envs

    def _sample_family(self) -> PrimitiveFamily:
        """Sample family based on weights."""
        families = list(self.family_weights.keys())
        weights = [self.family_weights[f] for f in families]
        total = sum(weights)
        weights = [w / total for w in weights]

        r = self.rng.random()
        cumsum = 0
        for family, weight in zip(families, weights):
            cumsum += weight
            if r <= cumsum:
                return family

        return families[-1]

    def set_curriculum_stage(self, stage: int):
        """
        Adjust weights for curriculum learning stages.

        Stage 1: Focus on simple primitives (nav, click)
        Stage 2: Add pattern and state tracking
        Stage 3: Introduce compositions
        Stage 4: Full mix with harder compositions
        """
        if stage == 1:
            self.family_weights = {
                PrimitiveFamily.NAVIGATION: 2.0,
                PrimitiveFamily.CLICK: 2.0,
                PrimitiveFamily.PATTERN: 0.5,
                PrimitiveFamily.STATE_TRACKING: 0.5,
                PrimitiveFamily.COMPOSITION: 0.0,
            }
        elif stage == 2:
            self.family_weights = {
                PrimitiveFamily.NAVIGATION: 1.5,
                PrimitiveFamily.CLICK: 1.5,
                PrimitiveFamily.PATTERN: 1.5,
                PrimitiveFamily.STATE_TRACKING: 1.5,
                PrimitiveFamily.COMPOSITION: 0.0,
            }
        elif stage == 3:
            self.family_weights = {
                PrimitiveFamily.NAVIGATION: 1.0,
                PrimitiveFamily.CLICK: 1.0,
                PrimitiveFamily.PATTERN: 1.0,
                PrimitiveFamily.STATE_TRACKING: 1.0,
                PrimitiveFamily.COMPOSITION: 1.0,
            }
        else:  # stage >= 4
            self.family_weights = {
                PrimitiveFamily.NAVIGATION: 0.5,
                PrimitiveFamily.CLICK: 0.5,
                PrimitiveFamily.PATTERN: 1.0,
                PrimitiveFamily.STATE_TRACKING: 1.0,
                PrimitiveFamily.COMPOSITION: 2.0,
            }

    def get_family_name(self, family: PrimitiveFamily) -> str:
        """Get human-readable family name."""
        return {
            PrimitiveFamily.NAVIGATION: "Navigation",
            PrimitiveFamily.CLICK: "Click/Selection",
            PrimitiveFamily.PATTERN: "Pattern Matching",
            PrimitiveFamily.STATE_TRACKING: "State Tracking",
            PrimitiveFamily.COMPOSITION: "Composition",
        }[family]


def collect_primitive_episode(
    env: PrimitiveEnv,
    policy=None,  # If None, use random actions
    max_steps: Optional[int] = None,
) -> dict:
    """
    Collect an episode from a primitive environment.

    Returns:
        dict with observations, actions, rewards, etc.
    """
    obs = env.reset()
    max_steps = max_steps or env.max_steps

    observations = [obs.clone()]
    actions = []
    coordinates = []
    rewards = []
    dones = []

    for _ in range(max_steps):
        if policy is None:
            # Random policy
            action = random.randint(0, env.action_space_size - 1)
            x, y = None, None
            if env.requires_coordinates and action == 8:  # CLICK
                x = random.randint(0, env.grid_size - 1)
                y = random.randint(0, env.grid_size - 1)
        else:
            # Use provided policy
            action_output = policy(obs.unsqueeze(0))
            action = action_output["action"].item()
            x = action_output.get("x")
            y = action_output.get("y")
            if x is not None:
                x = x.item()
            if y is not None:
                y = y.item()

        result = env.step(action, x, y)

        actions.append(action)
        coordinates.append((x, y))
        rewards.append(result.reward)
        dones.append(result.done)
        observations.append(result.observation.clone())

        if result.done:
            break

    return {
        "observations": observations,
        "actions": actions,
        "coordinates": coordinates,
        "rewards": rewards,
        "dones": dones,
        "success": result.success,
        "family": env.family,
        "description": env.get_task_description(),
    }
