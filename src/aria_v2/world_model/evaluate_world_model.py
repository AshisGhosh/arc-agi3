#!/usr/bin/env python3
"""
Evaluate world model accuracy against local game engine.

Two modes:
1. Agent-driven: Model chooses actions, compare predicted vs actual next frames
2. Demo replay: Feed human demo actions, compare predictions vs actual game

Updated for unified action tokenization: (ACT_TYPE, ACT_LOC) pairs.

Usage:
    uv run python -m src.aria_v2.world_model.evaluate_world_model --game ls20
    uv run python -m src.aria_v2.world_model.evaluate_world_model --game ls20 --mode demo
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .config import AgentConfig, WorldModelConfig
from .game_transformer import create_game_transformer
from ..tokenizer.frame_tokenizer import FrameVQVAE
from ..tokenizer.trajectory_dataset import (
    VQ_OFFSET, ACT_TYPE_OFFSET, ACT_LOC_OFFSET, ACT_LOC_NULL,
    FRAME_TOKEN, ACT_TOKEN, LEVEL_COMPLETE, GAME_START,
    pixel_to_vq_cell,
)


class WorldModelEvaluator:
    """Evaluate world model predictions against real game engine."""

    def __init__(self, config: AgentConfig | None = None, device: str = "cuda"):
        self.config = config or AgentConfig()
        self.device = device

        # Load VQ-VAE
        print("Loading VQ-VAE...")
        vqvae_ckpt = torch.load(
            self.config.vqvae_checkpoint, weights_only=False, map_location=device
        )
        self.vqvae = FrameVQVAE(vqvae_ckpt["config"]).to(device)
        self.vqvae.load_state_dict(vqvae_ckpt["model_state_dict"])
        self.vqvae.eval()

        # Load world model
        print("Loading world model...")
        wm_ckpt = torch.load(
            self.config.world_model_checkpoint, weights_only=False, map_location=device
        )
        model_config = wm_ckpt.get("model_config", WorldModelConfig())
        self.model = create_game_transformer(model_config)
        self.model.load_state_dict(wm_ckpt["model_state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

        print("Models loaded.")

    def encode_frame(self, frame: np.ndarray) -> list[int]:
        """Encode a 64x64 frame to 64 VQ code indices."""
        frame_tensor = torch.tensor(frame, dtype=torch.long).unsqueeze(0).to(self.device)
        vq_indices = self.vqvae.encode(frame_tensor)  # [1, 8, 8]
        return vq_indices[0].flatten().tolist()

    def decode_vq(self, vq_codes: list[int]) -> np.ndarray:
        """Decode 64 VQ codes back to a 64x64 frame."""
        codes_tensor = torch.tensor(vq_codes, dtype=torch.long).reshape(1, 8, 8).to(self.device)
        return self.vqvae.decode(codes_tensor)[0].cpu().numpy()

    @torch.no_grad()
    def predict_next_frame(self, context: list[int]) -> tuple[list[int], list[float]]:
        """
        Given context ending with [ACT] ACT_TYPE ACT_LOC, predict next 64 VQ codes.

        Returns:
            (predicted_codes, per_token_probs) - both length 64
        """
        # Append FRAME token
        ctx = context + [FRAME_TOKEN]
        predicted_codes = []
        per_token_probs = []

        for i in range(64):
            ctx_tensor = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(self.device)
            outputs = self.model(input_ids=ctx_tensor)
            logits = outputs.logits[:, -1, :]  # [1, V]

            # Get VQ code probabilities
            vq_logits = logits[0, VQ_OFFSET:VQ_OFFSET + 512]
            probs = F.softmax(vq_logits, dim=-1)
            pred_code = probs.argmax().item()
            pred_prob = probs[pred_code].item()

            predicted_codes.append(pred_code)
            per_token_probs.append(pred_prob)
            ctx.append(VQ_OFFSET + pred_code)

        return predicted_codes, per_token_probs

    @torch.no_grad()
    def predict_next_frame_teacher(self, context: list[int], actual_codes: list[int]) -> tuple[list[int], list[float]]:
        """Teacher-forced prediction of VQ codes."""
        full_input = context + [FRAME_TOKEN] + [VQ_OFFSET + c for c in actual_codes]
        ctx_tensor = torch.tensor(full_input, dtype=torch.long).unsqueeze(0).to(self.device)

        outputs = self.model(input_ids=ctx_tensor)
        logits = outputs.logits

        ctx_len = len(context) + 1  # +1 for FRAME_TOKEN
        predicted_codes = []
        per_token_correct = []

        for i in range(64):
            pos = ctx_len + i - 1
            vq_logits = logits[0, pos, VQ_OFFSET:VQ_OFFSET + 512]
            pred_code = vq_logits.argmax().item()
            predicted_codes.append(pred_code)
            per_token_correct.append(1.0 if pred_code == actual_codes[i] else 0.0)

        return predicted_codes, per_token_correct

    @torch.no_grad()
    def predict_action(self, context: list[int]) -> tuple[int, list[float], int, list[float]]:
        """
        Predict action type and location given context ending with VQ codes.

        Returns:
            (best_type, type_probs, best_loc, loc_probs)
        """
        ctx = context + [ACT_TOKEN]
        ctx_tensor = torch.tensor(ctx, dtype=torch.long).unsqueeze(0).to(self.device)

        outputs = self.model(input_ids=ctx_tensor)
        logits = outputs.logits[:, -1, :]

        # Action type prediction
        type_logits = logits[0, ACT_TYPE_OFFSET:ACT_TYPE_OFFSET + 8]
        type_probs = F.softmax(type_logits, dim=-1)
        best_type = type_probs.argmax().item()

        # Now predict location given the type
        ctx_with_type = ctx + [ACT_TYPE_OFFSET + best_type]
        ctx_tensor2 = torch.tensor(ctx_with_type, dtype=torch.long).unsqueeze(0).to(self.device)
        outputs2 = self.model(input_ids=ctx_tensor2)
        loc_logits = outputs2.logits[:, -1, ACT_LOC_OFFSET:ACT_LOC_OFFSET + 65]
        loc_probs = F.softmax(loc_logits[0], dim=-1)
        best_loc = loc_probs.argmax().item()

        return best_type, type_probs.cpu().tolist(), best_loc, loc_probs.cpu().tolist()


def evaluate_demo_replay(evaluator: WorldModelEvaluator, game_id: str, max_steps: int = 200):
    """
    Replay human demo actions in the real game.
    At each step, use the human's action and compare model predictions vs actual.
    """
    import arc_agi
    from arcengine import GameAction, GameState
    import glob

    # Load a demo trajectory
    demo_dir = os.path.join("videos/ARC-AGI-3 Human Performance", game_id)
    pattern = os.path.join(demo_dir, f"{game_id}*.jsonl")
    demos = sorted(glob.glob(pattern))
    if not demos:
        print(f"No demos found for {game_id} in {demo_dir}")
        return []

    demo_path = demos[0]
    print(f"\n{'='*60}")
    print(f"DEMO REPLAY EVALUATION: {game_id}")
    print(f"Demo: {os.path.basename(demo_path)}")
    print(f"{'='*60}")

    # Load demo actions with coordinates
    demo_entries = []
    with open(demo_path) as f:
        for line in f:
            entry = json.loads(line)
            data = entry.get("data", entry)
            action_input = data.get("action_input", {})
            action_type = action_input.get("id", 0)
            action_data = action_input.get("data", {})
            x = action_data.get("x")
            y = action_data.get("y")
            demo_entries.append({
                "type": action_type,
                "x": x,
                "y": y,
                "loc": pixel_to_vq_cell(int(x), int(y)) if x is not None and y is not None else 64,
            })

    print(f"Demo has {len(demo_entries)} entries")

    # Start local game
    arc = arc_agi.Arcade()
    env = arc.make(game_id, render_mode=None)

    available = env.observation_space.available_actions
    action_map = {}
    for i in range(1, 8):
        if i in available:
            action_map[i] = getattr(GameAction, f"ACTION{i}", None)

    obs = env.observation_space
    if obs.state == GameState.NOT_PLAYED:
        env.step(GameAction.RESET)

    obs = env.observation_space
    frame = np.array(obs.frame[0])
    frame_codes = evaluator.encode_frame(frame)
    context = [GAME_START, FRAME_TOKEN] + [VQ_OFFSET + c for c in frame_codes]

    results = []
    levels_completed = 0
    prev_score = 0

    for step, demo_entry in enumerate(demo_entries[:max_steps]):
        obs = env.observation_space
        if obs.state in (GameState.WIN, GameState.GAME_OVER):
            print(f"  Game ended at step {step}: {obs.state}")
            break

        demo_type = demo_entry["type"]
        demo_loc = demo_entry["loc"]

        # Model predicts action type and location
        pred_type, type_probs, pred_loc, loc_probs = evaluator.predict_action(context)

        # Build context with demo's action for frame prediction
        ctx_with_action = context + [
            ACT_TOKEN,
            ACT_TYPE_OFFSET + min(demo_type, 7),
            ACT_LOC_OFFSET + demo_loc,
        ]

        # Model predicts next frame (autoregressive)
        pred_codes_auto, pred_probs = evaluator.predict_next_frame(ctx_with_action)

        # Execute demo action in real game
        game_action = action_map.get(demo_type, GameAction.ACTION1)
        if demo_entry["x"] is not None:
            env.step(game_action, data={"x": demo_entry["x"], "y": demo_entry["y"]})
        else:
            env.step(game_action)

        obs = env.observation_space
        if obs.state in (GameState.WIN, GameState.GAME_OVER):
            print(f"  Game ended at step {step+1}: {obs.state}")
            break

        actual_frame = np.array(obs.frame[0])
        actual_codes = evaluator.encode_frame(actual_frame)

        # Teacher-forced prediction
        pred_codes_tf, tf_correct = evaluator.predict_next_frame_teacher(ctx_with_action, actual_codes)

        # Metrics
        auto_vq_acc = sum(1 for p, a in zip(pred_codes_auto, actual_codes) if p == a) / 64
        tf_vq_acc = sum(tf_correct) / 64
        pred_frame = evaluator.decode_vq(pred_codes_auto)
        pixel_acc = (pred_frame == actual_frame).mean()
        type_match = pred_type == min(demo_type, 7)
        loc_match = pred_loc == demo_loc

        score = getattr(obs, 'score', 0) or 0
        level_complete = score > prev_score
        if level_complete:
            levels_completed += 1
            prev_score = score

        result = {
            "step": step,
            "demo_type": demo_type,
            "demo_loc": demo_loc,
            "pred_type": pred_type,
            "pred_loc": pred_loc,
            "type_match": type_match,
            "loc_match": loc_match,
            "auto_vq_acc": auto_vq_acc,
            "tf_vq_acc": tf_vq_acc,
            "pixel_acc": pixel_acc,
            "level_complete": level_complete,
        }
        results.append(result)

        # Update context with actual frame
        if level_complete:
            context.append(LEVEL_COMPLETE)
        context += [
            ACT_TOKEN,
            ACT_TYPE_OFFSET + min(demo_type, 7),
            ACT_LOC_OFFSET + demo_loc,
            FRAME_TOKEN,
        ]
        for c in actual_codes:
            context.append(VQ_OFFSET + c)

        # Trim context
        max_ctx = 2048
        if len(context) > max_ctx:
            excess = len(context) - max_ctx
            context = [GAME_START] + context[excess + 1:]

        if step % 10 == 0:
            loc_str = f"loc={demo_loc}" if demo_loc < 64 else "loc=NULL"
            print(
                f"  Step {step:3d}: type={demo_type} {loc_str} | "
                f"pred_type={pred_type}({type_match}) pred_loc={pred_loc}({loc_match}) | "
                f"auto_vq={auto_vq_acc:.1%} tf_vq={tf_vq_acc:.1%} px={pixel_acc:.1%}"
            )

    # Summary
    print(f"\n{'='*60}")
    print("DEMO REPLAY SUMMARY")
    print(f"{'='*60}")
    if results:
        avg_auto = sum(r["auto_vq_acc"] for r in results) / len(results)
        avg_tf = sum(r["tf_vq_acc"] for r in results) / len(results)
        avg_px = sum(r["pixel_acc"] for r in results) / len(results)
        type_match_rate = sum(r["type_match"] for r in results) / len(results)
        spatial_results = [r for r in results if r["demo_loc"] < 64]
        loc_match_rate = (
            sum(r["loc_match"] for r in spatial_results) / len(spatial_results)
            if spatial_results else 0
        )
        print(f"Steps: {len(results)}")
        print(f"Levels completed: {levels_completed}")
        print(f"Action type match: {type_match_rate:.1%}")
        print(f"Action location match: {loc_match_rate:.1%} ({len(spatial_results)} spatial actions)")
        print(f"Avg autoregressive VQ accuracy: {avg_auto:.1%}")
        print(f"Avg teacher-forced VQ accuracy: {avg_tf:.1%}")
        print(f"Avg pixel accuracy: {avg_px:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate world model against local game")
    parser.add_argument("--game", "-g", default="ls20")
    parser.add_argument("--mode", "-m", choices=["demo", "all-games"], default="demo")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--world-model", default="checkpoints/world_model/best.pt")
    parser.add_argument("--vqvae", default="checkpoints/vqvae/best.pt")
    args = parser.parse_args()

    config = AgentConfig(
        world_model_checkpoint=args.world_model,
        vqvae_checkpoint=args.vqvae,
    )
    evaluator = WorldModelEvaluator(config=config)

    if args.mode == "all-games":
        for game in ["ls20", "vc33", "ft09"]:
            evaluate_demo_replay(evaluator, game, args.max_steps)
    else:
        evaluate_demo_replay(evaluator, args.game, args.max_steps)


if __name__ == "__main__":
    main()
