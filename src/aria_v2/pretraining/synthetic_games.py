"""
Synthetic Game Generator for Visual Grounding Pretraining.

Generates procedural grid games with labeled entities for training
the entity detector and classifier.

Entity Types:
- player: Single controllable entity
- goal: Level completion trigger
- item: Collectibles (disappear on contact)
- obstacle: Movement blockers
- trigger: Buttons/switches that affect other entities
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class EntityType(IntEnum):
    """Entity type labels for classification."""
    BACKGROUND = 0
    PLAYER = 1
    GOAL = 2
    ITEM = 3
    OBSTACLE = 4
    TRIGGER = 5


# Color palette (ARC-AGI compatible, 0-15)
COLORS = {
    "black": 0,      # Background
    "blue": 1,
    "red": 2,
    "green": 3,
    "yellow": 4,
    "grey": 5,
    "pink": 6,
    "orange": 7,
    "cyan": 8,
    "brown": 9,
    "white": 10,
}

# Default colors for entity types
ENTITY_COLORS = {
    EntityType.BACKGROUND: COLORS["black"],
    EntityType.PLAYER: COLORS["blue"],
    EntityType.GOAL: COLORS["green"],
    EntityType.ITEM: COLORS["yellow"],
    EntityType.OBSTACLE: COLORS["grey"],
    EntityType.TRIGGER: COLORS["orange"],
}


@dataclass
class Entity:
    """An entity in the game world."""
    entity_type: EntityType
    x: int
    y: int
    width: int = 1
    height: int = 1
    color: int = 0
    active: bool = True

    def occupies(self, x: int, y: int) -> bool:
        """Check if entity occupies given position."""
        return (self.x <= x < self.x + self.width and
                self.y <= y < self.y + self.height)

    @property
    def positions(self) -> list[tuple[int, int]]:
        """All positions occupied by this entity."""
        return [(self.x + dx, self.y + dy)
                for dx in range(self.width)
                for dy in range(self.height)]

    @property
    def center(self) -> tuple[int, int]:
        """Center position of entity."""
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class SyntheticGameConfig:
    """Configuration for synthetic game generation."""
    grid_size: int = 32
    num_items: int = 5
    num_obstacles: int = 10
    num_triggers: int = 0
    has_goal: bool = True
    mechanics: list[str] = field(default_factory=lambda: ["navigation", "collection"])

    # Entity size ranges
    min_entity_size: int = 1
    max_entity_size: int = 3

    # Visual variation
    randomize_colors: bool = True
    add_noise: bool = False
    noise_probability: float = 0.01

    # Spacing constraints
    min_entity_spacing: int = 2


@dataclass
class GameState:
    """A single game state with entity labels."""
    grid: np.ndarray                    # [H, W] color indices
    entity_mask: np.ndarray             # [H, W] binary mask (1 = entity)
    entity_labels: np.ndarray           # [H, W] EntityType labels
    entities: list[Entity]              # List of entities
    player_position: tuple[int, int]    # Player (x, y)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "grid": self.grid.tolist(),
            "entity_mask": self.entity_mask.tolist(),
            "entity_labels": self.entity_labels.tolist(),
            "entities": [
                {
                    "type": e.entity_type.name,
                    "x": e.x,
                    "y": e.y,
                    "width": e.width,
                    "height": e.height,
                    "color": e.color,
                }
                for e in self.entities
            ],
            "player_position": list(self.player_position),
        }


class SyntheticGameGenerator:
    """Generate synthetic game states with labeled entities."""

    def __init__(self, config: SyntheticGameConfig):
        self.config = config
        self.rng = random.Random()

    def seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def generate(self) -> GameState:
        """Generate a single game state."""
        cfg = self.config
        size = cfg.grid_size

        # Initialize empty grid
        grid = np.zeros((size, size), dtype=np.int32)
        entity_mask = np.zeros((size, size), dtype=np.int32)
        entity_labels = np.zeros((size, size), dtype=np.int32)
        entities: list[Entity] = []
        occupied: set[tuple[int, int]] = set()

        def get_color(entity_type: EntityType) -> int:
            """Get color for entity, with optional randomization."""
            base_color = ENTITY_COLORS[entity_type]
            if cfg.randomize_colors:
                # Pick from valid colors (non-background)
                valid_colors = [c for c in COLORS.values() if c != 0]
                return self.rng.choice(valid_colors)
            return base_color

        def place_entity(
            entity_type: EntityType,
            width: int = 1,
            height: int = 1,
            color: Optional[int] = None,
        ) -> Optional[Entity]:
            """Try to place an entity, return None if failed."""
            if color is None:
                color = get_color(entity_type)

            # Try multiple times to find valid position
            for _ in range(100):
                x = self.rng.randint(0, size - width)
                y = self.rng.randint(0, size - height)

                # Check spacing from other entities
                valid = True
                for dx in range(-cfg.min_entity_spacing, width + cfg.min_entity_spacing):
                    for dy in range(-cfg.min_entity_spacing, height + cfg.min_entity_spacing):
                        if (x + dx, y + dy) in occupied:
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    entity = Entity(
                        entity_type=entity_type,
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                        color=color,
                    )

                    # Mark positions as occupied
                    for pos in entity.positions:
                        occupied.add(pos)
                        grid[pos[1], pos[0]] = color
                        entity_mask[pos[1], pos[0]] = 1
                        entity_labels[pos[1], pos[0]] = entity_type

                    return entity

            return None

        def random_size() -> tuple[int, int]:
            """Get random entity size."""
            w = self.rng.randint(cfg.min_entity_size, cfg.max_entity_size)
            h = self.rng.randint(cfg.min_entity_size, cfg.max_entity_size)
            return w, h

        # Place player (always exactly 1)
        player = place_entity(EntityType.PLAYER, width=1, height=1)
        if player is None:
            raise RuntimeError("Could not place player")
        entities.append(player)

        # Place goal
        if cfg.has_goal:
            w, h = random_size()
            goal = place_entity(EntityType.GOAL, width=w, height=h)
            if goal:
                entities.append(goal)

        # Place items
        for _ in range(cfg.num_items):
            w, h = random_size()
            item = place_entity(EntityType.ITEM, width=w, height=h)
            if item:
                entities.append(item)

        # Place obstacles
        for _ in range(cfg.num_obstacles):
            w, h = random_size()
            obstacle = place_entity(EntityType.OBSTACLE, width=w, height=h)
            if obstacle:
                entities.append(obstacle)

        # Place triggers
        for _ in range(cfg.num_triggers):
            trigger = place_entity(EntityType.TRIGGER, width=1, height=1)
            if trigger:
                entities.append(trigger)

        # Add noise if configured
        if cfg.add_noise:
            noise_mask = np.random.random((size, size)) < cfg.noise_probability
            noise_colors = np.random.randint(1, 11, (size, size))
            # Only add noise to background cells
            background_mask = entity_mask == 0
            grid = np.where(noise_mask & background_mask, noise_colors, grid)

        return GameState(
            grid=grid,
            entity_mask=entity_mask,
            entity_labels=entity_labels,
            entities=entities,
            player_position=(player.x, player.y),
        )

    def generate_dataset(
        self,
        num_samples: int,
        seed: int = 42,
    ) -> list[GameState]:
        """Generate a dataset of game states."""
        self.seed(seed)
        return [self.generate() for _ in range(num_samples)]

    def generate_transition(
        self,
        state: GameState,
        action: int,
    ) -> tuple[GameState, int, str]:
        """
        Generate a transition from action.

        Args:
            state: Current game state
            action: Action (0=noop, 1=up, 2=down, 3=left, 4=right)

        Returns:
            new_state: Resulting state
            reward: 0 for normal, 1 for item collection
            event: Description of what happened
        """
        # Copy state
        grid = state.grid.copy()
        entity_mask = state.entity_mask.copy()
        entity_labels = state.entity_labels.copy()
        entities = [Entity(**{
            "entity_type": e.entity_type,
            "x": e.x,
            "y": e.y,
            "width": e.width,
            "height": e.height,
            "color": e.color,
            "active": e.active,
        }) for e in state.entities]

        # Find player
        player = next(e for e in entities if e.entity_type == EntityType.PLAYER)
        old_x, old_y = player.x, player.y

        # Calculate new position
        dx, dy = 0, 0
        if action == 1:    # up
            dy = -1
        elif action == 2:  # down
            dy = 1
        elif action == 3:  # left
            dx = -1
        elif action == 4:  # right
            dx = 1

        new_x = max(0, min(self.config.grid_size - 1, old_x + dx))
        new_y = max(0, min(self.config.grid_size - 1, old_y + dy))

        # Check collision with obstacles
        blocked = False
        for e in entities:
            if e.entity_type == EntityType.OBSTACLE and e.occupies(new_x, new_y):
                blocked = True
                break

        event = "noop"
        reward = 0

        if blocked:
            new_x, new_y = old_x, old_y
            event = "collision"
        elif (new_x, new_y) != (old_x, old_y):
            # Move player
            # Clear old position
            grid[old_y, old_x] = 0
            entity_mask[old_y, old_x] = 0
            entity_labels[old_y, old_x] = 0

            # Check for item collection
            for e in entities:
                if e.entity_type == EntityType.ITEM and e.active and e.occupies(new_x, new_y):
                    # Collect item
                    e.active = False
                    reward = 1
                    event = f"collected_item_at_{new_x}_{new_y}"
                    # Clear item from grid
                    for pos in e.positions:
                        grid[pos[1], pos[0]] = 0
                        entity_mask[pos[1], pos[0]] = 0
                        entity_labels[pos[1], pos[0]] = 0
                    break

            # Check for goal reached
            for e in entities:
                if e.entity_type == EntityType.GOAL and e.occupies(new_x, new_y):
                    event = "goal_reached"
                    reward = 10
                    break

            # Check for trigger activation
            for e in entities:
                if e.entity_type == EntityType.TRIGGER and e.occupies(new_x, new_y):
                    event = f"trigger_activated_at_{new_x}_{new_y}"
                    break

            # Update player position
            player.x = new_x
            player.y = new_y
            grid[new_y, new_x] = player.color
            entity_mask[new_y, new_x] = 1
            entity_labels[new_y, new_x] = EntityType.PLAYER

            if event == "noop":
                event = f"moved_{['noop', 'up', 'down', 'left', 'right'][action]}"

        new_state = GameState(
            grid=grid,
            entity_mask=entity_mask,
            entity_labels=entity_labels,
            entities=[e for e in entities if e.active],
            player_position=(player.x, player.y),
        )

        return new_state, reward, event


def generate_training_data(
    num_samples: int = 10000,
    grid_size: int = 32,
    seed: int = 42,
    include_transitions: bool = True,
) -> dict:
    """
    Generate training data for visual grounding.

    Returns dict with:
        - states: List of GameState dicts
        - transitions: List of (state, action, next_state, event) if include_transitions
    """
    config = SyntheticGameConfig(
        grid_size=grid_size,
        num_items=5,
        num_obstacles=10,
        num_triggers=2,
        has_goal=True,
        randomize_colors=True,
    )

    generator = SyntheticGameGenerator(config)
    generator.seed(seed)

    states = []
    transitions = []

    for i in range(num_samples):
        state = generator.generate()
        states.append(state.to_dict())

        if include_transitions:
            # Generate a few transitions from this state
            for action in [1, 2, 3, 4]:  # up, down, left, right
                next_state, reward, event = generator.generate_transition(state, action)
                transitions.append({
                    "state_idx": i,
                    "action": action,
                    "next_state": next_state.to_dict(),
                    "reward": reward,
                    "event": event,
                })

    return {
        "config": {
            "grid_size": grid_size,
            "num_samples": num_samples,
            "seed": seed,
        },
        "states": states,
        "transitions": transitions if include_transitions else None,
    }


if __name__ == "__main__":
    import json
    import sys

    # Generate sample data
    print("Generating synthetic game data...")

    config = SyntheticGameConfig(
        grid_size=32,
        num_items=5,
        num_obstacles=8,
        num_triggers=2,
        has_goal=True,
    )

    generator = SyntheticGameGenerator(config)
    generator.seed(42)

    # Generate and display a sample
    state = generator.generate()
    print(f"\nGenerated state with {len(state.entities)} entities:")
    for e in state.entities:
        print(f"  {e.entity_type.name}: ({e.x}, {e.y}) size={e.width}x{e.height} color={e.color}")

    print(f"\nPlayer position: {state.player_position}")
    print(f"Grid shape: {state.grid.shape}")
    print(f"Entity mask sum: {state.entity_mask.sum()} pixels")

    # Test transitions
    print("\nTesting transitions:")
    for action, name in [(1, "up"), (2, "down"), (3, "left"), (4, "right")]:
        next_state, reward, event = generator.generate_transition(state, action)
        print(f"  Action {name}: {event}, reward={reward}, new_pos={next_state.player_position}")

    # Generate small dataset
    if len(sys.argv) > 1 and sys.argv[1] == "--generate":
        num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        output_path = sys.argv[3] if len(sys.argv) > 3 else "data/synthetic_games.json"

        print(f"\nGenerating {num_samples} samples...")
        data = generate_training_data(num_samples=num_samples)

        # Save
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f)
        print(f"Saved to {output_path}")
