"""
JSONL Demo Loader for ARC-AGI-3 Human Performance Data.

Loads (observation, action) pairs from JSONL recording files.
These files contain pre-recorded human gameplay with complete frame data.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .demo_collector import DemoDataset, GameDemo


def load_jsonl_demo(jsonl_path: str) -> Optional[GameDemo]:
    """
    Load a single demonstration from a JSONL file.

    Args:
        jsonl_path: Path to .jsonl file

    Returns:
        GameDemo with observations and actions
    """
    path = Path(jsonl_path)

    # Extract game_id from filename (e.g., ls20-cb3b57cc.xxx.jsonl -> ls20)
    name = path.stem
    if "-" in name:
        game_id = name.split("-")[0]
    else:
        game_id = "unknown"

    observations = []
    actions = []
    rewards = []

    prev_score = 0
    final_state = None
    final_score = 0

    with open(path) as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            data = entry.get("data", {})
            if not data:
                continue

            # Get frame (64x64 grid)
            frame = data.get("frame")
            if frame and len(frame) > 0:
                # frame is a list of layers, use first layer
                obs = np.array(frame[0], dtype=np.uint8)
                observations.append(obs)

            # Get action
            action_input = data.get("action_input")
            if action_input and "id" in action_input:
                actions.append(action_input["id"])
            else:
                actions.append(0)  # NOOP

            # Track score for rewards
            score = data.get("score", 0)
            reward = 1.0 if score > prev_score else 0.0
            rewards.append(reward)
            prev_score = score

            # Track final state
            state = data.get("state")
            if state:
                final_state = state
            final_score = max(final_score, score)

    if len(observations) < 2:
        return None

    # Ensure actions match observations
    while len(actions) < len(observations):
        actions.append(0)
    actions = actions[: len(observations)]
    rewards = rewards[: len(observations)]

    demo = GameDemo(
        game_id=game_id,
        observations=observations,
        actions=actions,
        rewards=rewards,
        levels_completed=final_score,
        won=(final_state == "WIN"),
        total_steps=len(observations),
    )

    return demo


def load_jsonl_folder(folder_path: str) -> dict[str, DemoDataset]:
    """
    Load all JSONL demos from a folder structure.

    Args:
        folder_path: Path to folder containing game subfolders

    Returns:
        Dict mapping game_id to DemoDataset
    """
    folder = Path(folder_path)
    datasets: dict[str, DemoDataset] = {}

    # Find all JSONL files
    jsonl_files = list(folder.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files")

    for jsonl_path in sorted(jsonl_files):
        demo = load_jsonl_demo(str(jsonl_path))
        if demo is None:
            continue

        game_id = demo.game_id
        if game_id not in datasets:
            datasets[game_id] = DemoDataset(game_id=game_id)

        datasets[game_id].add(demo)
        print(
            f"  Loaded {jsonl_path.name}: {demo.total_steps} steps, "
            f"score={demo.levels_completed}, won={demo.won}"
        )

    return datasets


def convert_folder_to_demos(
    input_folder: str,
    output_folder: str = "demos/human",
) -> None:
    """
    Convert JSONL folder to DemoDataset JSON files.

    Args:
        input_folder: Folder with JSONL files
        output_folder: Output folder for demo JSON files
    """
    datasets = load_jsonl_folder(input_folder)

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    for game_id, dataset in datasets.items():
        output_file = output_path / f"{game_id}_human_demos.json"
        dataset.save(str(output_file))

        successful = len(dataset.get_successful_demos())
        print(
            f"Saved {len(dataset.demos)} demos ({successful} successful) "
            f"to {output_file}"
        )

    # Summary
    total_demos = sum(len(ds.demos) for ds in datasets.values())
    total_successful = sum(
        len(ds.get_successful_demos()) for ds in datasets.values()
    )
    print(f"\nTotal: {total_demos} demos, {total_successful} successful")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load JSONL demos")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="videos/ARC-AGI-3 Human Performance",
        help="Input folder with JSONL files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="demos/human",
        help="Output folder for demos",
    )

    args = parser.parse_args()
    convert_folder_to_demos(args.input, args.output)


if __name__ == "__main__":
    main()
