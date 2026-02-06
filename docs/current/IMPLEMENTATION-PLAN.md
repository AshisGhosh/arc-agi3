# Implementation Plan

## Overview

This documents what has been built and what remains. The learned world model pipeline is complete. The immediate next step is agent evaluation on real games.

---

## Completed Stages

### Stage 1: VQ-VAE Frame Tokenizer (Done)
**Files:** `src/aria_v2/tokenizer/frame_tokenizer.py`, `train_vqvae.py`

- Encodes 64x64 16-color frames → 64 discrete tokens (8x8, 512-code codebook)
- EMA codebook updates, dead code reset
- Trained on 13K frames in 1.9 minutes
- **Result:** 99.85% pixel accuracy, 44.73% codebook utilization

### Stage 2: Trajectory Dataset (Done)
**File:** `src/aria_v2/tokenizer/trajectory_dataset.py`

- Loads 28 JSONL human demos from 3 games (ls20, vc33, ft09)
- Tokenizes frames through frozen VQ-VAE, caches to disk
- Sliding window (2048 tokens, stride 670) with LEVEL_COMPLETE oversampling
- **Result:** 876K tokens, ~1,987 training windows

### Stage 3: SmolLM2 + LoRA (Done)
**File:** `src/aria_v2/world_model/game_transformer.py`

- Extends SmolLM2-360M vocabulary by 523 game tokens
- LoRA rank=16 on Q, K, V, O (3.9M LoRA + 47M embed/lm_head trainable)
- Loads in bfloat16 (fp16 causes NaN)
- **Result:** Model creation verified, forward + backward pass clean

### Stage 4: Training Pipeline (Done)
**File:** `src/aria_v2/world_model/train.py`

- Next-token prediction with per-token-type weights
- bfloat16 autocast + float32 loss computation
- 30 epochs, 77 minutes
- **Result:** frame=88.4%, action=67.9%, ppl=1.8, level=99.5%, val_loss=0.5721

### Stage 5: Inference Agent (Done)
**File:** `src/aria_v2/world_model/agent.py`

- Surprise measurement (NLL vs running EMA)
- Goal inference (P(LEVEL_COMPLETE) per action)
- Policy: goal-directed / exploration / learned policy selection
- **Status:** Code complete, not yet evaluated on real games

---

## Next Steps

### Step 1: Agent Evaluation on ls20
**Priority: Immediate**

```bash
uv run python -m src.aria_v2.world_model.agent --game ls20
```

Measure:
- Level completion rate (baseline: 0 from heuristic, target: >0)
- Human comparison: 11/12 demos completed all levels
- Action quality: does agent move purposefully or loop?
- Surprise trace: does surprise spike at meaningful events?
- Goal inference: does P(LEVEL_COMPLETE) correlate with progress?

### Step 2: Multi-Game Testing
**Priority: After ls20 evaluation**

- Test on vc33 and ft09
- Check if model generalizes across games (trained on all three)
- Compare per-game performance

### Step 3: Iteration (if needed)
**Priority: Based on evaluation results**

Possible improvements:
- More training data (collect additional demos)
- Longer context window (currently 30 frames / 2048 tokens)
- Reward shaping during agent inference
- Temperature tuning for action sampling
- Beam search for goal-directed planning

### Step 4: Competition Submission
**Priority: After satisfactory evaluation**

- Package agent for ARC-AGI-3 submission
- Ensure arcengine API compatibility
- Handle new/unseen games gracefully

---

## File Structure

```
src/aria_v2/
├── __init__.py
├── config.py                           # Old v2 config
├── run_game.py                         # Game runner (arcengine)
├── visual_grounding.py                 # Entity detection (Phase 1)
│
├── tokenizer/                          # VQ-VAE pipeline
│   ├── __init__.py
│   ├── frame_tokenizer.py              # VQ-VAE model
│   ├── train_vqvae.py                  # VQ-VAE training
│   └── trajectory_dataset.py           # JSONL → tokens
│
├── world_model/                        # SmolLM2 pipeline
│   ├── __init__.py
│   ├── config.py                       # All configs
│   ├── game_transformer.py             # Model creation
│   ├── train.py                        # Training loop
│   └── agent.py                        # Inference agent
│
├── core/                               # Heuristic approach (earlier work)
│   ├── abstract_learner.py
│   ├── goal_induction.py
│   ├── demonstration_learner.py
│   ├── agent.py
│   └── ...
│
├── pretraining/                        # Synthetic games
│   ├── synthetic_games.py
│   └── visual_grounding_trainer*.py
│
└── evaluation/
    └── real_data_eval.py
```

### Checkpoints
```
checkpoints/
├── vqvae/
│   └── best.pt                         # VQ-VAE (99.85% acc)
└── world_model/
    ├── best.pt                         # SmolLM2 (val_loss=0.5721, 704MB)
    ├── final.pt                        # Last epoch
    └── cache/
        └── trajectories.pt             # Cached tokenized demos
```

---

## Data

### Human Demonstrations
Location: `videos/ARC-AGI-3 Human Performance/`

| Game | Demos | Frames | Notes |
|------|-------|--------|-------|
| ls20 | 12 | ~8,700 | Block-sliding puzzle, 11/12 won |
| vc33 | ~8 | ~2,200 | |
| ft09 | ~8 | ~2,100 | |
| **Total** | **28** | **~13,000** | |

### Key Facts About ls20
- Color 12 = player (controllable)
- Color 9 = target regions
- Color 11 = move counter
- Colors 3, 4 = background/walls
- Requires 29+ actions per level, 7 levels total
- Human average: 400-850 steps to complete all levels

---

## Commands

```bash
# Train VQ-VAE (if needed)
uv run python -m src.aria_v2.tokenizer.train_vqvae

# Train world model (if needed)
uv run python -m src.aria_v2.world_model.train --epochs 30 --batch-size 4

# Run agent
uv run python -m src.aria_v2.world_model.agent --game ls20

# Check VRAM
nvidia-smi
```
