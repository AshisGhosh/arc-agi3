#!/usr/bin/env python3
"""
Pre-compute structured observations for all demo frames.

For each frame, computes:
- Entity positions (per-color center of mass, bbox, pixel count)
- Frame diff (what changed, how many pixels)
- Per-color movement (direction and magnitude)
- Return-to-go (steps until next level complete)
- Score changes

Output: one JSON file per demo, each containing a list of observation dicts.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np


def compute_entities(frame: np.ndarray, bg_threshold: float = 0.12) -> dict:
    """Identify entities (non-background colors) and their positions."""
    total_pixels = frame.size
    entities = {}
    backgrounds = []

    for color in range(16):
        mask = frame == color
        count = mask.sum()
        if count == 0:
            continue

        pct = count / total_pixels
        if pct > bg_threshold:
            backgrounds.append(int(color))
            continue

        ys, xs = np.where(mask)
        entities[int(color)] = {
            "pixels": int(count),
            "center": (round(float(ys.mean()), 1), round(float(xs.mean()), 1)),
            "bbox": (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
        }

    return {"entities": entities, "backgrounds": backgrounds}


def compute_diff(frame_curr: np.ndarray, frame_next: np.ndarray) -> dict:
    """Compute frame diff and per-color movements."""
    diff_mask = frame_curr != frame_next
    n_changed = int(diff_mask.sum())

    if n_changed == 0:
        return {"pixels_changed": 0, "movements": {}, "bbox": None}

    rows, cols = np.where(diff_mask)
    bbox = (int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max()))

    # Per-color center-of-mass shift
    movements = {}
    for color in range(16):
        mask_curr = frame_curr == color
        mask_next = frame_next == color
        cnt_curr = mask_curr.sum()
        cnt_next = mask_next.sum()

        # Skip backgrounds (>500px) and absent colors
        if cnt_curr > 500 or cnt_next > 500 or cnt_curr == 0 or cnt_next == 0:
            continue

        y_curr, x_curr = np.where(mask_curr)
        y_next, x_next = np.where(mask_next)
        dy = float(y_next.mean() - y_curr.mean())
        dx = float(x_next.mean() - x_curr.mean())

        if abs(dy) > 0.1 or abs(dx) > 0.1:
            movements[int(color)] = {
                "dx": round(dx, 1),
                "dy": round(dy, 1),
            }

    return {
        "pixels_changed": n_changed,
        "pct_changed": round(n_changed / frame_curr.size * 100, 1),
        "movements": movements,
        "bbox": bbox,
    }


def compute_click_info(action_input: dict, frame_curr: np.ndarray, frame_next: np.ndarray) -> dict | None:
    """For click actions (action 6), describe what's at the click location."""
    if not action_input or action_input.get("id") != 6:
        return None

    data = action_input.get("data", {})
    x = data.get("x")
    y = data.get("y")
    if x is None or y is None:
        return None

    # Clamp to frame bounds
    x = max(0, min(63, int(x)))
    y = max(0, min(63, int(y)))

    color_before = int(frame_curr[y, x])
    color_after = int(frame_next[y, x]) if frame_next is not None else None

    return {
        "x": x,
        "y": y,
        "color_before": color_before,
        "color_after": color_after,
    }


def process_demo(jsonl_path: str) -> list[dict]:
    """Process a single demo JSONL file into observations."""
    frames = []
    action_inputs = []
    scores = []

    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            data = entry["data"]
            frame_data = data.get("frame")
            if not frame_data or len(frame_data) == 0:
                continue
            frames.append(np.array(frame_data[0], dtype=np.int32))
            action_inputs.append(data.get("action_input", {"id": 0}))
            scores.append(data.get("score", 0))

    if len(frames) < 2:
        return []

    # Compute return-to-go for each step
    rtgs = []
    for t in range(len(frames)):
        rtg = None
        for future_t in range(t + 1, len(scores)):
            if scores[future_t] > scores[t]:
                rtg = future_t - t
                break
        rtgs.append(rtg)

    # Build observations
    observations = []
    for t in range(len(frames)):
        action_id = action_inputs[t].get("id", 0) if action_inputs[t] else 0

        obs = {
            "step": t,
            "action_id": action_id,
            "action_input": action_inputs[t],
            "score": scores[t],
            "score_delta": scores[t] - scores[t - 1] if t > 0 else 0,
            "rtg": rtgs[t],
            "entities": compute_entities(frames[t]),
        }

        # Frame diff (compare to next frame)
        if t < len(frames) - 1:
            obs["diff"] = compute_diff(frames[t], frames[t + 1])
            obs["click_info"] = compute_click_info(
                action_inputs[t], frames[t], frames[t + 1]
            )
        else:
            obs["diff"] = {"pixels_changed": 0, "movements": {}, "bbox": None}
            obs["click_info"] = None

        observations.append(obs)

    return observations


def main():
    demo_dir = Path("videos/ARC-AGI-3 Human Performance")
    output_dir = Path("data/observations")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0

    for game_dir in sorted(demo_dir.iterdir()):
        if not game_dir.is_dir():
            continue

        game = game_dir.name
        game_output_dir = output_dir / game
        game_output_dir.mkdir(parents=True, exist_ok=True)

        for jsonl_path in sorted(game_dir.glob("*.jsonl")):
            demo_name = jsonl_path.stem
            print(f"Processing {game}/{demo_name}...")

            observations = process_demo(str(jsonl_path))
            total_frames += len(observations)

            output_path = game_output_dir / f"{demo_name}.json"
            with open(output_path, "w") as f:
                json.dump(observations, f, indent=None, separators=(",", ":"))

            print(f"  → {len(observations)} frames")

    print(f"\nTotal: {total_frames} frames processed")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
