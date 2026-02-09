#!/usr/bin/env python3
"""
Generate natural language reasoning annotations for all demo frames.

Reads pre-computed observations and generates concise reasoning text
that describes game state, action effects, and progress toward goals.

Uses natural language to leverage SmolLM2's pretrained representations.
"""

import json
from pathlib import Path


# Direction names for ls20 actions
ACTION_DIRS = {
    1: "up",
    2: "down",
    3: "left",
    4: "right",
}


def describe_position(center: list[float], frame_size: int = 64) -> str:
    """Describe a position in natural language."""
    y, x = center
    # Quadrant description
    vert = "top" if y < frame_size / 3 else ("bottom" if y > 2 * frame_size / 3 else "middle")
    horiz = "left" if x < frame_size / 3 else ("right" if x > 2 * frame_size / 3 else "center")
    if vert == "middle" and horiz == "center":
        return "center"
    if vert == "middle":
        return horiz
    if horiz == "center":
        return vert
    return f"{vert}-{horiz}"


def compute_distance(c1: list[float], c2: list[float]) -> float:
    """Euclidean distance between two centers."""
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def generate_reasoning_ls20(obs: dict) -> str:
    """Generate reasoning for ls20 (block-sliding navigation puzzle)."""
    entities = obs["entities"]["entities"]
    diff = obs["diff"]
    movements = diff.get("movements", {})
    action = obs["action_id"]
    rtg = obs["rtg"]
    score_delta = obs["score_delta"]
    score = obs["score"]

    parts = []

    # Level completion
    if score_delta > 0:
        parts.append(f"level {score} complete!")
        if diff["pixels_changed"] > 500:
            parts.append("new puzzle layout.")
        if rtg:
            parts.append(f"{rtg} steps to next level.")
        return " ".join(parts)

    # Large transition (level reset without score change)
    if diff["pixels_changed"] > 500 and score_delta == 0:
        parts.append("major board change.")
        if "12" in entities:
            pos = describe_position(entities["12"]["center"])
            parts.append(f"player now at {pos}.")
        if rtg:
            parts.append(f"{rtg} steps to goal.")
        return " ".join(parts)

    # Player position
    if "12" in entities:
        p = entities["12"]
        py, px = p["center"]
        parts.append(f"player at ({py:.0f},{px:.0f}).")

    # Action description
    if action in ACTION_DIRS:
        direction = ACTION_DIRS[action]
        if "12" in movements:
            m = movements["12"]
            parts.append(f"moved {direction}.")
        else:
            parts.append(f"tried {direction}, blocked.")

    # Block interaction
    if "9" in movements and "12" in movements:
        m9 = movements["9"]
        parts.append("block pushed.")
    elif "9" in entities:
        e9 = entities["9"]
        if "12" in entities:
            dist = compute_distance(entities["12"]["center"], e9["center"])
            parts.append(f"block distance {dist:.0f}.")

    # Progress
    if rtg is not None:
        if rtg <= 3:
            parts.append(f"almost done, {rtg} steps left.")
        else:
            parts.append(f"{rtg} steps to goal.")

    # No effect
    if diff["pixels_changed"] == 0:
        parts = [p for p in parts if "moved" not in p]
        parts.append("no effect.")

    return " ".join(parts) if parts else "observing game state."


def generate_reasoning_click(obs: dict, game: str) -> str:
    """Generate reasoning for click-based games (vc33, ft09)."""
    entities = obs["entities"]["entities"]
    diff = obs["diff"]
    movements = diff.get("movements", {})
    click = obs.get("click_info")
    action = obs["action_id"]
    rtg = obs["rtg"]
    score_delta = obs["score_delta"]
    score = obs["score"]

    parts = []

    # Level completion
    if score_delta > 0:
        parts.append(f"level {score} complete!")
        if diff["pixels_changed"] > 500:
            parts.append("board reset.")
        if rtg:
            parts.append(f"{rtg} steps to next level.")
        return " ".join(parts)

    # Large transition without score
    if diff["pixels_changed"] > 500 and score_delta == 0:
        parts.append(f"major visual change, {diff['pixels_changed']} pixels.")
        if rtg:
            parts.append(f"{rtg} steps to goal.")
        return " ".join(parts)

    # Click action
    if action == 6 and click:
        x, y = click["x"], click["y"]
        color_before = click["color_before"]
        color_after = click["color_after"]
        pos_desc = describe_position([y, x])

        if diff["pixels_changed"] == 0:
            parts.append(f"clicked ({x},{y}) at {pos_desc}. no effect.")
        elif color_before != color_after:
            parts.append(f"clicked ({x},{y}). color {color_before} changed to {color_after}.")
        else:
            parts.append(f"clicked ({x},{y}) at {pos_desc}. {diff['pixels_changed']} pixels changed.")
    elif action == 0:
        if diff["pixels_changed"] == 0:
            parts.append("no action taken.")
        else:
            parts.append("reset action.")

    # Entity movements
    moving = [c for c, m in movements.items() if c not in ("11",)]
    if moving:
        for c in moving[:2]:  # Top 2 moving entities
            m = movements[c]
            dx, dy = m["dx"], m["dy"]
            if abs(dx) > abs(dy):
                direction = "right" if dx > 0 else "left"
            else:
                direction = "down" if dy > 0 else "up"
            if c in entities:
                parts.append(f"color {c} moved {direction}.")

    # Progress
    if rtg is not None:
        if rtg <= 3:
            parts.append(f"almost done, {rtg} steps left.")
        elif rtg <= 10:
            parts.append(f"{rtg} steps to goal.")
        else:
            parts.append(f"{rtg} steps remaining.")

    return " ".join(parts) if parts else "observing game state."


def generate_reasoning(obs: dict, game: str) -> str:
    """Generate reasoning for a single frame observation."""
    if game == "ls20":
        return generate_reasoning_ls20(obs)
    else:
        return generate_reasoning_click(obs, game)


def process_all():
    """Process all observation files and generate reasoning."""
    obs_dir = Path("data/observations")
    output_dir = Path("data/reasoning_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for game_dir in sorted(obs_dir.iterdir()):
        if not game_dir.is_dir():
            continue
        game = game_dir.name

        for obs_file in sorted(game_dir.glob("*.json")):
            with open(obs_file) as f:
                observations = json.load(f)

            reasoning_data = []
            for obs in observations:
                reasoning = generate_reasoning(obs, game)
                reasoning_data.append({
                    "step": obs["step"],
                    "reasoning": reasoning,
                })

            # Save per-demo (matching observation file structure)
            demo_output = output_dir / game / obs_file.name
            demo_output.parent.mkdir(parents=True, exist_ok=True)
            with open(demo_output, "w") as f:
                json.dump(reasoning_data, f, indent=None, separators=(",", ":"))

            total += len(reasoning_data)
            print(f"  {game}/{obs_file.stem}: {len(reasoning_data)} frames")

    print(f"\nTotal: {total} frames with reasoning")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    process_all()
