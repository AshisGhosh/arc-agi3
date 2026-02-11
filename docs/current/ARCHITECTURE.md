# Learned Understanding Architecture (v3.2)

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ARC-AGI-3 COMPETITION RUNTIME                        │
│                                                                         │
│   Unknown Game                    Understanding Agent (v3.2)            │
│  ┌──────────┐                ┌──────────────────────────────────────┐   │
│  │  Game    │   frame[64x64] │                                      │   │
│  │  Engine  │ ──────────────▶│  1. Frame Processor (hash, segment)  │   │
│  │ (arcengine)│              │  2. State Graph (explore/navigate)   │   │
│  │          │  action (1-7)  │  3. Understanding Model (pretrained) │   │
│  │          │ ◀──────────────│  4. TTT Engine (online adaptation)   │   │
│  │          │                │  5. Action Selector (6-mode priority)│   │
│  └──────────┘                └──────────────────────────────────────┘   │
│                                                                         │
│  Budget: 3 min/game, ~1ms/action, 20K+ actions possible                │
└─────────────────────────────────────────────────────────────────────────┘
```

The agent discovers game rules through online observation using a pretrained understanding
model. A CNN encoder processes individual transitions, a temporal transformer aggregates
patterns across observations, and structured decoder heads produce game-specific predictions.
Test-time training (TTT) adapts the encoder to each new game during play.

| Spec | Value |
|------|-------|
| **Approach** | CNN encoder + temporal transformer + decoder heads |
| **Total params** | ~4.3M (578K encoder + 3.2M temporal + 85K decoder + 392K frame predictor) |
| **TTT params** | 10.7K (LoRA rank 4 on conv4/conv5) |
| **Pretrained on** | 3600 synthetic sequences (578K transitions) across 3 game types |
| **Precision** | float32 |
| **VRAM** | ~10MB inference |
| **Inference speed** | ~1.8ms/action amortized |

---

## Design Principles

1. **Learned, not heuristic.** The model learns WHAT to look for from diverse training data.
   TTT teaches it the SPECIFIC rules of each game.
2. **Direct feature paths.** Action effects use per-action embedding grouping (not transformer
   query specialization). Entity roles use computed frame statistics (not learned features).
3. **Classification over regression.** Shift prediction uses 7-bin classification to avoid
   regression-to-mean on randomized action mappings.
4. **Fast enough for competition.** ~1.8ms/action amortized, well under 3-min/game budget.

---

## Full Model Architecture Diagram

```
┌────────────────────── Understanding Model (~4.3M params) ──────────────┐
│                                                                         │
│  INPUT: Sequence of (frame_t, action, frame_t+1) transitions           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  CNN Transition Encoder (~578K params)         0.3ms/transition │    │
│  │                                                                 │    │
│  │  frame_t ──┐                                                    │    │
│  │            ├── one_hot ──┐                                      │    │
│  │  frame_t+1─┘            ├── [40, 64, 64] ── 5× Conv+BN+ReLU   │    │
│  │  action ── embed(8) ────┘                    stride 1,2,2,2,2  │    │
│  │                                                    │            │    │
│  │                              ┌─────────────────────┤            │    │
│  │                              ▼                     ▼            │    │
│  │                     [256, 4, 4] spatial    [256] global embed   │    │
│  │                     (for TTT/frame pred)   (for understanding)  │    │
│  │                                                                 │    │
│  │  LoRA (TTT): rank 4 on conv4 + conv5 (~10.7K trainable)       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│              │ spatial                    │ embeddings                   │
│              ▼                            ▼                              │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐    │
│  │  Frame Predictor    │    │  Temporal Transformer (~3.2M params) │    │
│  │  (~392K params)     │    │                           15ms/batch │    │
│  │                     │    │                                      │    │
│  │  [256,4,4]          │    │  embeddings: [B, L, 256]            │    │
│  │    ↓ 4× Upsample   │    │  + positional encoding              │    │
│  │  [16, 64, 64]       │    │       ▼                              │    │
│  │  next frame logits  │    │  16 learnable query tokens           │    │
│  │                     │    │       ▼                              │    │
│  │  (TTT target only)  │    │  4× TransformerDecoderLayer          │    │
│  └─────────────────────┘    │  (cross-attend queries to sequence)  │    │
│                              │       ▼                              │    │
│                              │  [B, 16, 256] understanding state   │    │
│                              └──────────────┬─────────────────────┘    │
│                                             │                           │
│  ┌──────────────────────────────────────────┼──────────────────────┐    │
│  │  Understanding Decoder (~85K params)     │           <0.1ms    │    │
│  │                                          │                      │    │
│  │  ┌──────────────────┐   ┌────────────────▼───────────────┐      │    │
│  │  │ Action-Effect    │   │ Game-Type Head                 │      │    │
│  │  │ Head             │   │ pool queries → MLP → 8 classes │      │    │
│  │  │                  │   └────────────────────────────────┘      │    │
│  │  │ Group embeddings │   ┌────────────────────────────────┐      │    │
│  │  │ by action type,  │   │ Confidence Head                │      │    │
│  │  │ classify shift   │   │ pool queries → MLP → sigmoid   │      │    │
│  │  │ into 7 bins/axis │   └────────────────────────────────┘      │    │
│  │  └──────────────────┘                                           │    │
│  │  ┌──────────────────┐                                           │    │
│  │  │ Entity-Role Head │   INPUT: raw frames (not embeddings)      │    │
│  │  │ Compute 5 stats  │   • area, appear, disappear,             │    │
│  │  │ per color from   │     volatility, static_ratio              │    │
│  │  │ frame diffs,     │   → MLP → 5 roles per color              │    │
│  │  │ then classify    │                                           │    │
│  │  └──────────────────┘                                           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  OUTPUT:                                                                │
│  ├── shift[8, 2]           per-action dx,dy (from classification)      │
│  ├── change_prob[8]        per-action frame change probability         │
│  ├── blocked_prob[8]       per-action blocked probability              │
│  ├── entity_roles[16, 5]   per-color role logits (sigmoid needed)      │
│  ├── game_type[8]          8-way classification logits                 │
│  └── confidence            scalar 0-1 (sigmoid applied)                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: CNN Transition Encoder

**File:** `src/aria_v3/understanding/encoder.py` (~578K params)

Encodes (frame_t, action, frame_t+1) into a 256-dim embedding.

```
Input: one_hot(frame)[16] + one_hot(next_frame)[16] + action_embed(8)[8] = [B, 40, 64, 64]
  → Conv2d(40→64, 3×3, stride 1)  + BN + ReLU    → [B, 64, 64, 64]
  → Conv2d(64→64, 3×3, stride 2)  + BN + ReLU    → [B, 64, 32, 32]
  → Conv2d(64→128, 3×3, stride 2) + BN + ReLU    → [B, 128, 16, 16]
  → Conv2d(128→128, 3×3, stride 2) + BN + ReLU   → [B, 128, 8, 8]
  → Conv2d(128→256, 3×3, stride 2) + BN + ReLU   → [B, 256, 4, 4]
  → Global avg pool                                → [B, 256] (embedding)
  → Also keep [B, 256, 4, 4] spatial features for TTT frame prediction
```

**FramePredictor** (for TTT, ~392K params):
```
Input: [B, 256, 4, 4] spatial features
  → Conv2d(256→128) + ReLU + Upsample(2×)  → [B, 128, 8, 8]
  → Conv2d(128→64)  + ReLU + Upsample(2×)  → [B, 64, 16, 16]
  → Conv2d(64→32)   + ReLU + Upsample(2×)  → [B, 32, 32, 32]
  → Conv2d(32→16)   + ReLU + Upsample(2×)  → [B, 16, 64, 64]
Output: per-pixel 16-class logits
```

---

## Component 2: Temporal Transformer

**File:** `src/aria_v3/understanding/temporal.py` (~3.2M params)

DETR-style cross-attention: 16 learnable query tokens attend to transition sequence.

```
transition_embeddings: [B, L, 256] + positional encoding (learned, up to 200)
query_tokens: [B, 16, 256] (learned parameters)
  → 4× TransformerDecoderLayer(d_model=256, nhead=4, ffn=512, dropout=0.1)
  → LayerNorm
Output: [B, 16, 256] understanding state
```

---

## Component 3: Understanding Decoder

**File:** `src/aria_v3/understanding/decoder.py` (~85K params)

### Action-Effect Head (direct action grouping)
Groups CNN embeddings by action type, averages per group, classifies shift into 7 bins per axis.

```
For each action a in [0..7]:
  embeddings_a = mean(embedding[i] where action[i] == a)  → [B, 256]
Shift MLP: Linear(256→128→14) → [B, 8, 7+7] shift logits (dx 7-way + dy 7-way)
Effects MLP: Linear(256→64→3) → change_prob, blocked_prob, affected_color

Shift bins: [-16, -8, -4, 0, +4, +8, +16] (covers all synthetic step sizes)
```

### Entity-Role Head (frame statistics)
Computes 5 statistics per color from frame diffs, then classifies.

```
For each color c in [0..15]:
  stat_0: area fraction (avg pixels of this color / total)
  stat_1: appear rate (pixels that become this color)
  stat_2: disappear rate (pixels that stop being this color)
  stat_3: volatility (appear + disappear)
  stat_4: static ratio (unchanged pixels / total of this color)

MLP: Linear(5→32→32→5) → per-color role logits
Roles: player, wall, collectible, background, counter
```

### Game-Type Head (from transformer)
```
Pool all 16 query tokens → Linear(256→128→8) → 8-way classification
Types: navigation, click_puzzle, collection, mixed, push, conditional, physics, unknown
```

### Confidence Head (from transformer)
```
Pool all 16 query tokens → Linear(256→64→1) → sigmoid → scalar [0, 1]
```

---

## Component 4: TTT (Test-Time Training)

**File:** `src/aria_v3/understanding/ttt.py`

LoRA rank 4 applied to encoder conv4 and conv5 (~10.7K trainable params).
SGD with momentum 0.9, lr=0.01. Update every 10 transitions.

```
┌──────────────── Test-Time Training (per game) ─────────────────────────┐
│                                                                         │
│  Level Start                                                            │
│      │                                                                  │
│      ▼                                                                  │
│  ttt.reset()   ← LoRA weights → 0, buffer → empty                     │
│      │                                                                  │
│      ▼                                                                  │
│  ┌─── OBSERVE LOOP (every step) ──────────────────────────────────┐    │
│  │                                                                 │    │
│  │  ttt.observe(frame_t, action, frame_t+1)                       │    │
│  │      │                                                         │    │
│  │      ├── Add to rolling buffer (max 200)                       │    │
│  │      │                                                         │    │
│  │      └── Every 10 steps:                                       │    │
│  │          ┌──────────────────────────────────────────────┐      │    │
│  │          │  Sample 32 transitions from buffer           │      │    │
│  │          │      │                                       │      │    │
│  │          │      ▼                                       │      │    │
│  │          │  CNN Encoder (with LoRA on conv4/conv5)      │      │    │
│  │          │      │                                       │      │    │
│  │          │      ▼ [32, 256, 4, 4] spatial features      │      │    │
│  │          │      │                                       │      │    │
│  │          │      ▼                                       │      │    │
│  │          │  Frame Predictor → [32, 16, 64, 64] logits  │      │    │
│  │          │      │                                       │      │    │
│  │          │      ▼                                       │      │    │
│  │          │  CE Loss vs actual next_frame                │      │    │
│  │          │      │                                       │      │    │
│  │          │      ▼                                       │      │    │
│  │          │  SGD step (lr=0.01, momentum=0.9)            │      │    │
│  │          │  Only updates LoRA params (~10.7K)           │      │    │
│  │          └──────────────────────────────────────────────┘      │    │
│  │                                                                 │    │
│  │  Result: Encoder adapts to THIS game's visual patterns          │    │
│  │  • Frame prediction improves → better embeddings                │    │
│  │  • Better embeddings → better understanding model output        │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  Level Complete → back to ttt.reset()                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Training Pipeline

```
┌─────────────────────── OFFLINE (before competition) ───────────────────┐
│                                                                         │
│  Synthetic Game Framework                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│  │ Navigation  │  │Click Puzzle │  │ Collection  │                    │
│  │ (grid+walls)│  │(toggle/cycle)│  │(nav+collect)│                    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                    │
│         │                │                │                             │
│         └────────────────┼────────────────┘                             │
│                          ▼                                              │
│              ┌───────────────────────┐                                  │
│              │  Data Generation      │                                  │
│              │  200 configs × 3 types│                                  │
│              │  × 2 strategies       │                                  │
│              │  × 3 augmentations    │                                  │
│              │  = 3600 sequences     │                                  │
│              │  = 578K transitions   │                                  │
│              └───────────┬───────────┘                                  │
│                          │                                              │
│                  Augmentations:                                         │
│                  • Color permutation (16! bijections)                   │
│                  • Action remapping (permute IDs 1-5)                   │
│                  • Spatial flips (H/V with action swaps)                │
│                          │                                              │
│                          ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    PRETRAINING (30 epochs, 19 min)               │   │
│  │                                                                  │   │
│  │  Input: (frame_t, action, frame_t+1) sequences                  │   │
│  │                                                                  │   │
│  │  Losses:                                                         │   │
│  │  ├── Shift classification CE (7 bins × 2 axes)    weight: 2.0   │   │
│  │  ├── Change/blocked probability BCE               weight: 1.0   │   │
│  │  ├── Entity role BCE (multi-label per color)      weight: 1.5   │   │
│  │  ├── Game type CE (8-way)                         weight: 1.0   │   │
│  │  ├── Confidence MSE (calibrated)                  weight: 0.5   │   │
│  │  └── Frame prediction CE (self-supervised)        weight: 0.1   │   │
│  │                                                                  │   │
│  │  Optimizer: AdamW lr=3e-4, cosine annealing                     │   │
│  └──────────────────────────┬───────────────────────────────────────┘   │
│                             │                                           │
│                             ▼                                           │
│              ┌──────────────────────────┐                               │
│              │  checkpoints/            │                               │
│              │  understanding/best.pt   │                               │
│              │  ~4.3M params, ~17MB     │                               │
│              │                          │                               │
│              │  Results:                │                               │
│              │  • Game type: 100%       │                               │
│              │  • Shift MAE: 0.128 px   │                               │
│              │  • Frame pred: 0.87 CE   │                               │
│              └──────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Training Results (30 epochs on 3600 synthetic sequences)

| Metric | Value |
|--------|-------|
| Game type accuracy (val) | **100%** |
| Shift MAE (val) | **0.128** pixels |
| Shift classification CE (dx/dy) | 0.013 / 0.015 |
| Frame prediction loss | 0.87 |
| Entity role BCE | 0.19 |
| Training time | ~19 minutes (RTX 4090) |

---

## Synthetic Game Framework

**Directory:** `src/aria_v3/synthetic_games/`

### Archetypes
| Type | Description | Actions | Key Patterns |
|------|-------------|---------|--------------|
| Navigation | Grid movement with walls | 1-5 | Shift vectors, player/wall detection |
| Click Puzzle | Toggle/cycle/lights-out | 6 | State changes, target patterns |
| Collection | Navigate + collect objects | 1-5 | Collectible detection, progress counter |

### Augmentations
- **Color permutation**: random bijection on 16 colors
- **Action remapping**: permute directional action IDs
- **Spatial transforms**: horizontal/vertical flips with matching action swaps

### Dataset
- 200 configs × 3 archetypes × 2 strategies × 3 copies (base + 2 augmented) = 3600 sequences
- ~160 transitions per sequence average, 578K total transitions
- Ground truth: action effects, entity roles, game type at steps 10/50/100/200

---

*Architecture version: 5.0 (v3.2 learned understanding)*
*Approach: CNN Encoder + Temporal Transformer + Decoder Heads + TTT*
