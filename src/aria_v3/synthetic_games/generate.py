"""
Data generation pipeline for synthetic games.

Generates (transition_sequence, ground_truth_labels) pairs with augmentations
for pretraining the understanding model.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base import GameConfig, GroundTruth, SyntheticGame
from .navigation import NavigationGame
from .click_puzzle import ClickPuzzleGame
from .collection import CollectionGame


@dataclass
class Transition:
    """A single (frame, action, next_frame) transition."""
    frame: np.ndarray       # [64, 64] uint8
    action: int             # action ID (1-7)
    click_x: int            # click x (for action 6)
    click_y: int            # click y (for action 6)
    next_frame: np.ndarray  # [64, 64] uint8
    level_complete: bool
    game_over: bool


@dataclass
class Sequence:
    """A sequence of transitions with ground truth at multiple stages."""
    transitions: list[Transition]
    ground_truth_stages: dict[int, GroundTruth]  # step → ground truth
    game_type: str
    available_actions: list[int]


@dataclass
class AugmentedConfig:
    """Augmentation parameters."""
    color_perm: np.ndarray | None = None  # [16] → [16] color mapping
    action_perm: dict[int, int] | None = None  # action_id → action_id
    flip_h: bool = False
    flip_v: bool = False


def random_game_config(
    rng: np.random.RandomState,
    archetype: str,
) -> GameConfig:
    """Generate a random game configuration."""
    config = GameConfig(seed=rng.randint(2**31))

    # Randomize colors
    colors = list(range(16))
    rng.shuffle(colors)
    config.background_color = colors[0]
    config.player_color = colors[1]
    config.wall_color = colors[2]
    config.collectible_color = colors[3]
    config.counter_color = colors[4]
    config.target_color = colors[5]

    # Randomize step size
    config.step_size = rng.choice([4, 8, 16])

    # Randomize action mapping
    dir_actions = [1, 2, 3, 4]
    rng.shuffle(dir_actions)
    config.action_up = dir_actions[0]
    config.action_down = dir_actions[1]
    config.action_left = dir_actions[2]
    config.action_right = dir_actions[3]

    # Archetype-specific params
    if archetype == "navigation":
        config.wall_density = rng.uniform(0.05, 0.35)
    elif archetype == "click_puzzle":
        config.step_size = rng.choice([8, 16, 32])
    elif archetype == "collection":
        config.wall_density = rng.uniform(0.05, 0.25)
        config.num_collectibles = rng.randint(2, 8)

    return config


def create_game(archetype: str, config: GameConfig) -> SyntheticGame:
    """Create a game instance from archetype name."""
    if archetype == "navigation":
        return NavigationGame(config)
    elif archetype == "click_puzzle":
        puzzle_type = np.random.choice(["toggle", "cycle", "lights_out"])
        return ClickPuzzleGame(config, puzzle_type=puzzle_type)
    elif archetype == "collection":
        return CollectionGame(config)
    else:
        raise ValueError(f"Unknown archetype: {archetype}")


def play_random(
    game: SyntheticGame,
    num_steps: int,
    rng: np.random.RandomState,
) -> list[Transition]:
    """Play randomly, collecting transitions."""
    transitions = []
    available = game.available_actions

    for _ in range(num_steps):
        frame = game.frame.copy()
        action = rng.choice(available)

        x, y = 0, 0
        if action == 6:
            x = rng.randint(64)
            y = rng.randint(64)

        next_frame, level_complete, game_over = game.step(action, x, y)

        transitions.append(Transition(
            frame=frame,
            action=action,
            click_x=x, click_y=y,
            next_frame=next_frame.copy(),
            level_complete=level_complete,
            game_over=game_over,
        ))

        if level_complete or game_over:
            break

    return transitions


def play_smart(
    game: SyntheticGame,
    num_steps: int,
    rng: np.random.RandomState,
) -> list[Transition]:
    """Play with smart exploration: prioritize untested actions, avoid repeats."""
    transitions = []
    available = game.available_actions
    action_counts: dict[int, int] = {a: 0 for a in available}
    recent_actions: list[int] = []

    for _ in range(num_steps):
        frame = game.frame.copy()

        # Prioritize least-tried actions
        min_count = min(action_counts.values())
        candidates = [a for a in available if action_counts[a] == min_count]

        # Avoid repeating last 3 actions if possible
        non_repeat = [a for a in candidates if a not in recent_actions[-3:]]
        if non_repeat:
            candidates = non_repeat

        action = rng.choice(candidates)
        action_counts[action] = action_counts.get(action, 0) + 1
        recent_actions.append(action)

        x, y = 0, 0
        if action == 6:
            x = rng.randint(64)
            y = rng.randint(64)

        next_frame, level_complete, game_over = game.step(action, x, y)

        transitions.append(Transition(
            frame=frame,
            action=action,
            click_x=x, click_y=y,
            next_frame=next_frame.copy(),
            level_complete=level_complete,
            game_over=game_over,
        ))

        if level_complete or game_over:
            break

    return transitions


def augment_color(
    transitions: list[Transition],
    rng: np.random.RandomState,
) -> list[Transition]:
    """Apply random color permutation to all frames in a sequence."""
    perm = np.arange(16, dtype=np.uint8)
    rng.shuffle(perm)

    result = []
    for t in transitions:
        result.append(Transition(
            frame=perm[t.frame],
            action=t.action,
            click_x=t.click_x,
            click_y=t.click_y,
            next_frame=perm[t.next_frame],
            level_complete=t.level_complete,
            game_over=t.game_over,
        ))
    return result


def augment_spatial(
    transitions: list[Transition],
    flip_h: bool = False,
    flip_v: bool = False,
    action_remap: dict[int, int] | None = None,
) -> list[Transition]:
    """Apply spatial flip with matching action remapping."""
    result = []
    for t in transitions:
        frame = t.frame.copy()
        next_frame = t.next_frame.copy()

        if flip_h:
            frame = np.flip(frame, axis=1).copy()
            next_frame = np.flip(next_frame, axis=1).copy()
        if flip_v:
            frame = np.flip(frame, axis=0).copy()
            next_frame = np.flip(next_frame, axis=0).copy()

        action = t.action
        if action_remap and action in action_remap:
            action = action_remap[action]

        click_x = t.click_x
        click_y = t.click_y
        if flip_h:
            click_x = 63 - click_x
        if flip_v:
            click_y = 63 - click_y

        result.append(Transition(
            frame=frame,
            action=action,
            click_x=click_x,
            click_y=click_y,
            next_frame=next_frame,
            level_complete=t.level_complete,
            game_over=t.game_over,
        ))
    return result


def generate_sequence(
    archetype: str,
    config: GameConfig,
    num_steps: int = 200,
    strategy: str = "random",
    rng: np.random.RandomState | None = None,
    ground_truth_steps: list[int] | None = None,
) -> Sequence:
    """Generate a single training sequence.

    Args:
        archetype: Game archetype name
        config: Game configuration
        num_steps: Number of transitions to generate
        strategy: "random" or "smart"
        rng: Random state
        ground_truth_steps: Steps at which to capture ground truth labels

    Returns:
        Sequence with transitions and staged ground truth
    """
    if rng is None:
        rng = np.random.RandomState()
    if ground_truth_steps is None:
        ground_truth_steps = [10, 50, 100, 200]

    game = create_game(archetype, config)
    game.reset()

    if strategy == "random":
        transitions = play_random(game, num_steps, rng)
    elif strategy == "smart":
        transitions = play_smart(game, num_steps, rng)
    else:
        transitions = play_random(game, num_steps, rng)

    # Capture ground truth at specified steps
    gt_stages = {}
    for step in ground_truth_steps:
        if step <= len(transitions):
            gt = game.get_ground_truth()
            gt_stages[step] = gt

    return Sequence(
        transitions=transitions,
        ground_truth_stages=gt_stages,
        game_type=archetype,
        available_actions=game.available_actions,
    )


def generate_dataset(
    output_dir: str | Path,
    archetypes: list[str] | None = None,
    configs_per_archetype: int = 100,
    strategies: list[str] | None = None,
    augments_per_sequence: int = 2,
    num_steps: int = 200,
    seed: int = 42,
) -> dict:
    """Generate a full dataset of training sequences.

    Args:
        output_dir: Where to save the dataset
        archetypes: List of game archetypes to generate
        configs_per_archetype: Number of unique configurations per archetype
        strategies: List of exploration strategies
        augments_per_sequence: Number of augmented copies per base sequence
        num_steps: Steps per sequence
        seed: Random seed

    Returns:
        Statistics dict
    """
    if archetypes is None:
        archetypes = ["navigation", "click_puzzle", "collection"]
    if strategies is None:
        strategies = ["random", "smart"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    stats = {
        "total_sequences": 0,
        "total_transitions": 0,
        "per_archetype": {},
    }

    start = time.time()

    for archetype in archetypes:
        archetype_dir = output_path / archetype
        archetype_dir.mkdir(exist_ok=True)
        arch_count = 0
        arch_transitions = 0

        for config_idx in range(configs_per_archetype):
            config = random_game_config(rng, archetype)

            for strategy in strategies:
                # Generate base sequence
                seq = generate_sequence(
                    archetype=archetype,
                    config=config,
                    num_steps=num_steps,
                    strategy=strategy,
                    rng=np.random.RandomState(rng.randint(2**31)),
                )

                # Save base sequence
                seq_id = f"{archetype}_{config_idx:04d}_{strategy}"
                _save_sequence(seq, archetype_dir / f"{seq_id}.npz")
                arch_count += 1
                arch_transitions += len(seq.transitions)

                # Generate augmented copies
                for aug_idx in range(augments_per_sequence):
                    aug_transitions = augment_color(
                        seq.transitions,
                        np.random.RandomState(rng.randint(2**31)),
                    )

                    # Random spatial flip
                    flip_h = rng.random() < 0.5
                    flip_v = rng.random() < 0.5
                    action_remap = {}
                    if flip_h:
                        c = config
                        action_remap[c.action_left] = c.action_right
                        action_remap[c.action_right] = c.action_left
                    if flip_v:
                        c = config
                        action_remap[c.action_up] = c.action_down
                        action_remap[c.action_down] = c.action_up

                    if flip_h or flip_v:
                        aug_transitions = augment_spatial(
                            aug_transitions, flip_h, flip_v, action_remap
                        )

                    aug_seq = Sequence(
                        transitions=aug_transitions,
                        ground_truth_stages=seq.ground_truth_stages,
                        game_type=seq.game_type,
                        available_actions=seq.available_actions,
                    )
                    aug_id = f"{seq_id}_aug{aug_idx}"
                    _save_sequence(aug_seq, archetype_dir / f"{aug_id}.npz")
                    arch_count += 1
                    arch_transitions += len(aug_seq.transitions)

            if (config_idx + 1) % 20 == 0:
                elapsed = time.time() - start
                print(
                    f"  {archetype}: {config_idx+1}/{configs_per_archetype} configs, "
                    f"{arch_count} sequences, {arch_transitions} transitions ({elapsed:.1f}s)"
                )

        stats["per_archetype"][archetype] = {
            "sequences": arch_count,
            "transitions": arch_transitions,
        }
        stats["total_sequences"] += arch_count
        stats["total_transitions"] += arch_transitions

    elapsed = time.time() - start
    stats["generation_time_s"] = elapsed

    # Save stats
    with open(output_path / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def _save_sequence(seq: Sequence, path: Path) -> None:
    """Save a sequence as a compressed numpy archive."""
    frames = np.stack([t.frame for t in seq.transitions])
    next_frames = np.stack([t.next_frame for t in seq.transitions])
    actions = np.array([t.action for t in seq.transitions], dtype=np.int8)
    click_x = np.array([t.click_x for t in seq.transitions], dtype=np.int16)
    click_y = np.array([t.click_y for t in seq.transitions], dtype=np.int16)
    level_complete = np.array([t.level_complete for t in seq.transitions], dtype=bool)
    game_over = np.array([t.game_over for t in seq.transitions], dtype=bool)

    # Ground truth: serialize as JSON string stored in the archive
    gt_data = {}
    for step, gt in seq.ground_truth_stages.items():
        gt_data[str(step)] = {
            "game_type": gt.game_type.value if hasattr(gt.game_type, 'value') else str(gt.game_type),
            "confidence": gt.confidence,
            "level_complete": gt.level_complete,
            "entities": {
                str(color): {
                    "role": info.role.value if hasattr(info.role, 'value') else str(info.role),
                }
                for color, info in gt.entities.items()
            },
            "action_effects": {
                str(aid): {
                    "shift_dx": eff.shift_dx,
                    "shift_dy": eff.shift_dy,
                    "change_prob": eff.change_prob,
                    "affected_color": eff.affected_color,
                    "blocked_prob": eff.blocked_prob,
                }
                for aid, eff in gt.action_effects.items()
            },
        }

    np.savez_compressed(
        path,
        frames=frames,
        next_frames=next_frames,
        actions=actions,
        click_x=click_x,
        click_y=click_y,
        level_complete=level_complete,
        game_over=game_over,
        game_type=seq.game_type,
        available_actions=np.array(seq.available_actions, dtype=np.int8),
        ground_truth=json.dumps(gt_data),
    )


def load_sequence(path: str | Path) -> dict:
    """Load a sequence from a compressed numpy archive."""
    data = np.load(path, allow_pickle=True)
    result = {
        "frames": data["frames"],
        "next_frames": data["next_frames"],
        "actions": data["actions"],
        "click_x": data["click_x"],
        "click_y": data["click_y"],
        "level_complete": data["level_complete"],
        "game_over": data["game_over"],
        "game_type": str(data["game_type"]),
        "available_actions": data["available_actions"],
        "ground_truth": json.loads(str(data["ground_truth"])),
    }
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic game data")
    parser.add_argument("--output", "-o", default="data/synthetic")
    parser.add_argument("--configs", "-n", type=int, default=100)
    parser.add_argument("--steps", "-s", type=int, default=200)
    parser.add_argument("--augments", "-a", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Generating synthetic game data...")
    stats = generate_dataset(
        output_dir=args.output,
        configs_per_archetype=args.configs,
        num_steps=args.steps,
        augments_per_sequence=args.augments,
        seed=args.seed,
    )

    print(f"\nDone! Generated {stats['total_sequences']} sequences "
          f"({stats['total_transitions']} transitions) "
          f"in {stats['generation_time_s']:.1f}s")
    for arch, arch_stats in stats["per_archetype"].items():
        print(f"  {arch}: {arch_stats['sequences']} sequences, "
              f"{arch_stats['transitions']} transitions")
