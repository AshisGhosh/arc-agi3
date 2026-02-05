"""
Evaluate visual grounding on real ARC-AGI-3 data.

Key finding: Pattern-based detection doesn't transfer from synthetic data,
but movement-based detection works well.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..visual_grounding import MovementCorrelator


@dataclass
class EvaluationResult:
    """Results from evaluating on a game recording."""
    game_id: str
    total_frames: int
    movement_frames: int
    player_found: int
    player_accuracy: float
    avg_pixels_changed: float
    sample_positions: list[tuple[int, int]]


def load_recording(path: str) -> list[dict]:
    """Load a JSONL recording file."""
    with open(path) as f:
        return [json.loads(line) for line in f]


def evaluate_movement_detection(
    recording_path: str,
    max_frames: int = 100,
) -> EvaluationResult:
    """
    Evaluate movement-based player detection on a recording.

    Returns accuracy of finding the player via movement correlation.
    """
    frames = load_recording(recording_path)
    game_id = frames[0]['data'].get('game_id', 'unknown')

    correlator = MovementCorrelator()
    player_positions = []
    movement_frames = 0
    pixels_changed_list = []

    for i in range(1, min(max_frames, len(frames))):
        prev_data = frames[i-1]['data']
        curr_data = frames[i]['data']

        prev_grid = np.array(prev_data['frame'])[0]
        curr_grid = np.array(curr_data['frame'])[0]

        # Get action
        action_input = curr_data.get('action_input', {})
        action_id = action_input.get('id', 0) if isinstance(action_input, dict) else 0

        # Count changes
        diff = (prev_grid != curr_grid)
        num_changes = diff.sum()
        if num_changes > 0:
            pixels_changed_list.append(num_changes)

        # Check movement actions
        if action_id in [1, 2, 3, 4]:
            movement_frames += 1
            player_pos = correlator.identify_player(prev_grid, action_id, curr_grid)
            if player_pos:
                player_positions.append(player_pos)

    accuracy = len(player_positions) / movement_frames if movement_frames > 0 else 0.0
    avg_changes = np.mean(pixels_changed_list) if pixels_changed_list else 0.0

    return EvaluationResult(
        game_id=game_id,
        total_frames=min(max_frames, len(frames)),
        movement_frames=movement_frames,
        player_found=len(player_positions),
        player_accuracy=accuracy,
        avg_pixels_changed=avg_changes,
        sample_positions=[(int(p[0]), int(p[1])) for p in player_positions[:10]],
    )


def evaluate_all_recordings(
    recordings_dir: str,
    game_id: str = "ls20",
) -> list[EvaluationResult]:
    """Evaluate on all recordings for a game."""
    game_dir = Path(recordings_dir) / game_id
    if not game_dir.exists():
        # Try different path structures
        for parent in Path(recordings_dir).iterdir():
            if parent.is_dir() and game_id in str(parent):
                game_dir = parent
                break

    results = []
    for recording_file in game_dir.glob("*.jsonl"):
        result = evaluate_movement_detection(str(recording_file))
        results.append(result)

    return results


def print_evaluation_summary(results: list[EvaluationResult]):
    """Print summary of evaluation results."""
    if not results:
        print("No results to summarize")
        return

    total_movement = sum(r.movement_frames for r in results)
    total_found = sum(r.player_found for r in results)
    overall_accuracy = total_found / total_movement if total_movement > 0 else 0.0

    print(f"\n{'='*60}")
    print("Movement-Based Player Detection Evaluation")
    print(f"{'='*60}")
    print(f"Game: {results[0].game_id}")
    print(f"Recordings evaluated: {len(results)}")
    print(f"Total movement frames: {total_movement}")
    print(f"Player found: {total_found}")
    print(f"Overall accuracy: {overall_accuracy:.1%}")
    print("\nPer-recording results:")

    for r in results:
        print(f"  {r.game_id}: {r.player_found}/{r.movement_frames} "
              f"({r.player_accuracy:.1%}), avg pixels changed: {r.avg_pixels_changed:.0f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--recordings_dir", type=str,
                        default="videos/ARC-AGI-3 Human Performance")
    parser.add_argument("--game", type=str, default="ls20")
    args = parser.parse_args()

    print(f"Evaluating on {args.game} recordings...")

    # Find game directory
    base_dir = Path(args.recordings_dir)
    game_dir = base_dir / args.game

    if not game_dir.exists():
        print(f"Game directory not found: {game_dir}")
        exit(1)

    results = []
    for recording_file in game_dir.glob("*.jsonl"):
        print(f"  Processing {recording_file.name}...")
        result = evaluate_movement_detection(str(recording_file))
        results.append(result)

    print_evaluation_summary(results)
