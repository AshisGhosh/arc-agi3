# Implementation Plan (v2: Game-Agnostic Redesign)

## Overview

The v1 world model had three structural flaws: (1) click coordinates discarded, (2) action prediction mode-collapsed, (3) policy entangled with dynamics. v2 fixes all three while keeping the VQ-VAE (99.85% accuracy) unchanged.

**What changed:** Action tokenization (unified type+location), world model training (new loss weights), policy architecture (separated, masked).
**What's the same:** VQ-VAE, SmolLM2-360M + LoRA backbone, bfloat16, training infrastructure.

---

## Completed Stages

### Stage 1: VQ-VAE Frame Tokenizer (Done, unchanged)
**Files:** `src/aria_v2/tokenizer/frame_tokenizer.py`, `train_vqvae.py`

- Encodes 64x64 16-color frames → 64 discrete tokens (8x8, 512-code codebook)
- **Result:** 99.85% pixel accuracy, 44.73% codebook utilization

### Stage 2: Trajectory Dataset v2 (Done, updated)
**File:** `src/aria_v2/tokenizer/trajectory_dataset.py`

- **NEW:** Extracts click coordinates from JSONL (`action_input.data.x/y`)
- **NEW:** Maps pixel coords to VQ cell: `cell = (y//8)*8 + (x//8)`
- **NEW:** Emits `[ACT] <ACT_TYPE_i> <ACT_LOC_j>` per step (69 tokens/step, up from 67)
- **NEW:** 589 token vocabulary (was 523): +8 action types, +65 locations, -7 old actions
- Cache version 2 — old cache must be deleted before retraining
- **Per game:**
  - ls20: `<ACT_TYPE_1> <ACT_LOC_NULL>` (navigation, no coordinates)
  - vc33: `<ACT_TYPE_6> <ACT_LOC_46>` (click at VQ cell 46)
  - ft09: `<ACT_TYPE_6> <ACT_LOC_37>` (click at VQ cell 37)

### Stage 3: SmolLM2 + LoRA v2 (Done, updated)
**File:** `src/aria_v2/world_model/game_transformer.py`

- Extended vocabulary: 49,741 tokens (was 49,675)
- **NEW:** ACT_LOC embeddings initialized from VQ codebook (spatial grounding)
- Same LoRA config, same precision

### Stage 4: Training Pipeline v2 (Done, updated)
**File:** `src/aria_v2/world_model/train.py`

- **NEW:** Separate metrics for action_type_acc and action_loc_acc
- **NEW:** Loss weights: VQ=1.0, action_type=3.0, action_loc=3.0, level=5.0
- **NEW:** Cache file: `trajectories_v2.pt` (won't conflict with v1 cache)
- Same training loop, hyperparams, and hardware config

### Stage 5: Policy Heads (Done, new)
**File:** `src/aria_v2/world_model/policy_head.py`

- **ActionTypeHead:** Linear(960,256) → GELU → Dropout(0.1) → Linear(256,8) → ~248K params
- **ActionLocationHead:** Spatial attention (query from frame, keys from 64 VQ cells + NULL) → ~250K params
- **MASK embedding:** Learned 960-dim vector replacing action tokens
- Total: ~500K trainable parameters

### Stage 6: Policy Training (Done, new)
**File:** `src/aria_v2/world_model/train_policy.py`

- Freezes entire backbone (world model weights locked)
- Replaces action tokens (ACT, ACT_TYPE, ACT_LOC) with MASK in context
- Trains type_CE + has_location * loc_CE
- AdamW, lr=1e-3, 50 epochs

### Stage 7: Inference Agent v2 (Done, rewritten)
**File:** `src/aria_v2/world_model/agent.py`

- Two contexts: full (for world model) and masked (for policy)
- Policy forward: backbone on masked context → PolicyHeads → (type, loc)
- Converts VQ cell → pixel coordinates for click actions
- Passes spatial data to game API: `env.step(action, data={"x": x, "y": y})`

### Stage 8: Evaluation v2 (Done, updated)
**File:** `src/aria_v2/world_model/evaluate_world_model.py`

- Tracks action_type_match and action_loc_match separately
- Handles spatial actions in demo replay
- All-games mode: evaluate ls20, vc33, ft09 in one run

---

## Next Steps (Execution)

### Phase 1: Retokenize Data (~10 min)
Delete old cache, run training which will auto-retokenize:
```bash
rm checkpoints/world_model/cache/trajectories.pt
```

### Phase 2: Retrain World Model (~80 min)
```bash
uv run python -m src.aria_v2.world_model.train --epochs 30
```

**Decision gate:** frame_acc >80%, action_type_acc >50%, action_loc_acc >40% for click games.

### Phase 3: Validate World Model (~30 min)
```bash
uv run python -m src.aria_v2.world_model.evaluate_world_model --mode all-games
```

Check: Does click-game frame prediction improve now that model sees coordinates?

### Phase 4: Train Policy Heads (~20 min)
```bash
uv run python -m src.aria_v2.world_model.train_policy --epochs 50
```

**Decision gate:** type_acc >50%, loc_acc >40% for click games.

### Phase 5: Agent Evaluation (~1 hr)
```bash
uv run python -m src.aria_v2.world_model.agent --game ls20
uv run python -m src.aria_v2.world_model.agent --game vc33
uv run python -m src.aria_v2.world_model.agent --game ft09
```

Any level completion = success (baseline: 0).

### Phase 6: Competition Submission
Package agent for ARC-AGI-3, handle unseen games.

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
│   ├── frame_tokenizer.py              # VQ-VAE model (unchanged)
│   ├── train_vqvae.py                  # VQ-VAE training (unchanged)
│   └── trajectory_dataset.py           # JSONL → tokens (v2: type+loc)
│
├── world_model/                        # SmolLM2 pipeline
│   ├── __init__.py
│   ├── config.py                       # All configs (v2: +PolicyConfig)
│   ├── game_transformer.py             # Model creation (v2: 589 tokens)
│   ├── train.py                        # World model training (v2: new metrics)
│   ├── policy_head.py                  # NEW: ActionTypeHead + ActionLocationHead
│   ├── train_policy.py                 # NEW: Frozen backbone policy training
│   ├── agent.py                        # Inference agent (v2: masked policy)
│   └── evaluate_world_model.py         # Evaluation (v2: type+loc accuracy)
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
│   └── best.pt                         # VQ-VAE (99.85% acc) — current
├── world_model/
│   ├── best.pt                         # SmolLM2 — NEEDS RETRAINING
│   ├── final.pt
│   └── cache/
│       ├── trajectories.pt             # v1 cache — DELETE before retraining
│       └── trajectories_v2.pt          # v2 cache (auto-generated)
└── policy/
    └── best.pt                         # Policy heads — NOT YET TRAINED
```

---

## Token Vocabulary (v2)

| Token Range | Type | Count | IDs |
|---|---|---|---|
| VQ codes | Frame codes | 512 | 49152–49663 |
| ACT_TYPE | Action type | 8 | 49664–49671 |
| ACT_LOC | Action location | 65 | 49672–49736 (0-63=cells, 64=NULL) |
| FRAME | Frame marker | 1 | 49737 |
| ACT | Action marker | 1 | 49738 |
| LEVEL_COMPLETE | Level boundary | 1 | 49739 |
| GAME_START | Trajectory start | 1 | 49740 |
| MASK | Policy mask (not in world model) | 1 | 49741 |
| **Total new tokens** | | **589** (+1 MASK) | 49152–49741 |

### Sequence Format (69 tokens per step)
```
[GAME_START]
[FRAME] vq_0..vq_63 [ACT] <ACT_TYPE_i> <ACT_LOC_j>
[FRAME] vq_0..vq_63 [ACT] <ACT_TYPE_i> <ACT_LOC_j>
...
[LEVEL_COMPLETE]
```

2048 context ÷ 69 tokens/step ≈ 29 frames per window.

---

## Data

### Human Demonstrations
Location: `videos/ARC-AGI-3 Human Performance/`

| Game | Demos | Frames | Action Type | Click Coords |
|------|-------|--------|-------------|-------------|
| ls20 | 12 | ~8,700 | 1-4 (directional) | None (ACT_LOC_NULL) |
| vc33 | ~8 | ~2,200 | 6 (click) | 51/64 cells covered |
| ft09 | ~8 | ~2,100 | 6 (click) | 61/64 cells covered |
| **Total** | **28** | **~13,000** | | |

---

## Commands

```bash
# Delete old cache (required before first v2 training)
rm checkpoints/world_model/cache/trajectories.pt

# Train world model (v2)
uv run python -m src.aria_v2.world_model.train --epochs 30

# Evaluate world model
uv run python -m src.aria_v2.world_model.evaluate_world_model --mode all-games

# Train policy heads
uv run python -m src.aria_v2.world_model.train_policy --epochs 50

# Run agent
uv run python -m src.aria_v2.world_model.agent --game ls20

# Train VQ-VAE (only if needed, unchanged)
uv run python -m src.aria_v2.tokenizer.train_vqvae

# Check VRAM
nvidia-smi
```

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 8x8 spatial resolution too coarse for clicks | Low (80-95% cell coverage) | Add sub-cell refinement later |
| Policy overfits on 13K frames | Medium | 500K params, dropout 0.1, early stopping |
| World model quality degrades with new tokens | Low (+2 tokens/step) | Validate frame acc before policy |
| Two forward passes too slow | Low (~60ms total) | KV cache if needed |
| Unknown games have action types >7 | Low | ACT_TYPE slots expandable |
