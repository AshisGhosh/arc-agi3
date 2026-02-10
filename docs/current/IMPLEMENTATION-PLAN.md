# Implementation Plan (v3.2: Learned Game Understanding)

## Overview

Replace hand-coded game analysis with a learned understanding model that discovers game
primitives, action effects, and game rules from observation history. Pre-train on diverse
synthetic games. Adapt to each real game via TTT (test-time training) during play.

**Key principle:** The model learns WHAT to look for from diverse training data. TTT teaches
it the SPECIFIC rules of each game. Layer 1 (state graph) handles fast execution.

---

## Competition Constraints

| Constraint | Value | Implication |
|-----------|-------|-------------|
| Games | 150+ unknown | Must generalize beyond 3 public games |
| Demos/instructions | None | Must discover rules from scratch |
| Time per game | ~3 minutes | ~20K actions at 1-2ms/act |
| Compute | RTX 5090, 32GB | ~3M param model fits easily |
| Scoring | RHAE = min(human/agent, 1) | Efficiency after understanding |
| Actions | 0-7 (0=reset, 1-5=simple, 6=click x,y) | Small action space |
| Frames | 64x64, 16 colors | Pixel-art, deterministic |

---

## Architecture

```
Per-transition CNN Encoder (0.3ms/step, ~580K params)
  Input: [40, 64, 64] = one_hot(frame_t)[16] + one_hot(frame_t+1)[16] + action_embed[8]
  Output: 256-dim transition embedding (+ 4x4 spatial grid)
  TTT: LoRA rank 4 on last 2 conv layers, self-supervised next-frame prediction
  ↓ stored in rolling buffer (last 200)

Temporal Transformer (15ms every 100 actions, ~2M params)
  Input: [100, 256] transition embeddings
  16 learnable query tokens (DETR-style)
  4 layers, d_model=256, 4 heads
  Output: 16 x 256-dim "understanding state"
  ↓

Understanding Decoder Heads (0.1ms, ~520K params)
  Action-Effect Head: per action → (shift_dx, shift_dy, change_prob, affected_color, blocked_prob)
  Entity-Role Head: per color → (player, wall, collectible, background, counter)
  Game-Type Head: → 8-way classification
  Confidence Head: → scalar 0-1
  ↓ structured understanding

Layer 1: Reactive Agent (<2ms, every action)
  State graph + plan executor
  Uses understanding to: pathfind with movement_map, target collectible colors, avoid walls
  Fallback: graph exploration when confidence < threshold
```

### VRAM Budget (RTX 5090, 32GB)

| Component | VRAM |
|-----------|------|
| CNN encoder | 1.2MB |
| Temporal transformer | 4MB |
| Understanding decoder | 1MB |
| Transition buffer (200 x 256) | 0.2MB |
| TTT LoRA + optimizer state | 2MB |
| State graph (CPU RAM) | 0.5GB |
| **Total** | **~10MB** |
| **Remaining** | **~32GB** |

### Latency Budget

| Operation | Frequency | Latency | Amortized |
|-----------|-----------|---------|-----------|
| CNN encoder | Every action | 0.3ms | 0.3ms |
| State graph | Every action | 0.01ms | 0.01ms |
| TTT update | Every 10 actions | 1ms | 0.1ms |
| Temporal transformer | Every 100 actions | 15ms | 0.15ms |
| Decoder heads | Every 100 actions | 0.1ms | 0.001ms |
| Layer 1 action select | Every action | 1ms | 1ms |
| **Total amortized** | | | **~1.6ms/act** |

---

## What the Model Learns

### 1. Primitives (from observation history)
The Entity-Role Head learns to identify object roles from temporal patterns:
- **Player**: the color that consistently moves when directional actions are taken
- **Wall**: the color that blocks player movement (action has no effect near this color)
- **Collectible**: the color that disappears when player overlaps it
- **Background**: the color covering the largest static area
- **Counter**: a small region that changes color when other events happen

These are NOT hard-coded rules. The model learns them from diverse synthetic games where
these patterns appear in different visual forms.

### 2. Action Effects (generalist concepts)
The Action-Effect Head learns per-action behaviors:
- **Navigation actions (1-5)**: shift vector (dx, dy), blocked probability
- **Click action (6)**: what changes at the click location
- **No effect**: action does nothing (dead action for this game)

### 3. Exploration → Hypothesis → Confidence (trained process)
The Confidence Head is calibrated against actual prediction accuracy:
- After 10 actions: confidence ~0.1 (almost no signal yet)
- After 50 actions: confidence ~0.4 (emerging patterns)
- After 200 actions: confidence ~0.85 (confirmed understanding)

The model learns to be UNCERTAIN early and CONFIDENT late. The execution layer
uses confidence to decide: explore (low) vs exploit (high).

---

## Synthetic Game Framework

### Game Archetypes (12 total, 6 minimum viable)

**Tier 1 (Core — build first):**

1. **Grid Navigation** — Player moves on grid, walls block. Variable step sizes.
   - Teaches: player detection, movement mapping, wall detection
2. **Click Puzzle** — Click regions to toggle states. Goal: target pattern.
   - Teaches: click-target identification, state-toggle mechanics
3. **Collection** — Navigate to objects that disappear on contact. Counter increments.
   - Teaches: collectible detection, progress tracking
4. **Mixed Nav+Click** — Navigate to position, click to interact.
   - Teaches: multi-action-type games

**Tier 2 (Important — build second):**

5. **Push/Sokoban** — Push objects to target positions.
   - Teaches: multi-entity interaction, pushing mechanics
6. **Conditional Effects** — Same action has different effects based on game state.
   - Teaches: context-dependent rules

**Tier 3 (If time permits):**

7. Physics/Gravity — Objects fall, stack
8. Memory/Simon — Reproduce patterns
9. Sorting — Arrange in order
10. Enemy/Avoidance — Moving obstacles
11. Painting — Fill areas
12. Multi-level — Rules change between levels

### Synthetic Game Interface

```python
class SyntheticGame:
    def __init__(self, config: dict): ...
    def reset(self) -> np.ndarray: ...  # Returns 64x64 frame
    def step(self, action: int, x: int = 0, y: int = 0) -> tuple[np.ndarray, bool, bool]: ...
    def get_ground_truth(self) -> dict: ...  # Labels for training
    @property
    def available_actions(self) -> list[int]: ...
```

### Data Generation

**Per archetype:** 500 base configurations (unique layouts).
**Per configuration:** 3 exploration strategies × 3 augmentations = 9 sequences.
**Per sequence:** 200 transitions with staged ground-truth labels.

**Augmentations (critical for generalization):**
- Color permutation: random bijection on 16 colors
- Action remapping: permute action IDs 1-5
- Spatial transforms: flips + 90° rotations with matching action remaps

**Total dataset:** 6 archetypes × 500 configs × 9 variants = 27,000 sequences = 5.4M transitions.

### Ground Truth Labels (staged)

After 10 transitions:
```python
{"stage": "initial", "confidence": 0.1,
 "actions_that_change_frame": [1, 2, 3, 4],
 "entity_roles": {}, "action_effects": {}}  # mostly unknown
```

After 50 transitions:
```python
{"stage": "hypothesis", "confidence": 0.4,
 "action_effects": {1: {"shift": [0, -8], "conf": 0.85}, ...},
 "entity_roles": {12: {"role": "player", "conf": 0.7}}}
```

After 200 transitions:
```python
{"stage": "confirmed", "confidence": 0.85,
 "action_effects": {1: {"shift": [0, -8], "conf": 0.95, "blocked_by": [5]}, ...},
 "entity_roles": {12: {"role": "player", "conf": 0.95},
                  9: {"role": "collectible", "conf": 0.75}},
 "game_type": "collection"}
```

---

## TTT (Test-Time Training)

### Self-Supervised Target
Predict next frame given (frame_t, action_t). Available at every step, no labels needed.

### Configuration
- LoRA rank 4 on last 2 CNN conv layers (~10K trainable params)
- Pixel prediction head: Conv2d(256→128→16), ~300K params
- SGD with momentum 0.9, lr=0.01 (fast adaptation)
- Update every 10 transitions (~1ms amortized)
- Reset LoRA to zero at each level start

### Why TTT Works Here
- Pretrained backbone provides general game primitives (warm start)
- TTT learns THIS game's specific rules (action effects, entity behaviors)
- Only updates ~10K LoRA params (not the full 580K CNN)
- Self-supervised: no need for labels during play

---

## Implementation Order

### Stage 1: Synthetic Game Framework (3-4 days)
- Base game engine with common interface
- Grid Navigation game (variable grid sizes, wall layouts, step sizes)
- Click Puzzle game (toggle, cycle, swap variants)
- Collection game (navigate + collect + counter)
- Data generation pipeline with augmentations
- **Test:** Generate 1000 sequences, verify diversity and label quality

### Stage 2: Understanding Model (3-4 days)
- Transition encoder CNN (580K params)
- Temporal transformer with query tokens (2M params)
- Understanding decoder heads (520K params)
- Loss functions: MSE for shifts, BCE for roles, CE for game type, MSE for confidence
- **Test:** Train on navigation games, verify shift vector learning

### Stage 3: Pretraining (2-3 days)
- Full dataset generation (27K sequences across 6 archetypes)
- Training pipeline with stage-weighted loss
- Evaluation on held-out game configurations
- **Test:** Model correctly identifies game type + action effects on unseen configs

### Stage 4: TTT Integration (2-3 days)
- LoRA adapter setup on CNN encoder
- Self-supervised next-frame prediction
- Online update loop during gameplay
- **Test:** TTT improves prediction accuracy on real game after 50 actions

### Stage 5: Agent Integration + Evaluation (2-3 days)
- Wire understanding model into Layer 1 execution
- Replace hand-coded Layer 2 with learned model
- Run on ls20, vc33, ft09
- Compare to basic v3 and three-layer v3.1
- **Target:** >3 levels across 3 games

---

## Success Criteria

| Metric | Target | Basic v3 | Three-Layer v3.1 |
|--------|--------|----------|-------------------|
| Levels completed (3 games) | >3 total | 1 | 1 |
| Games with >0 levels | >=2 of 3 | 1 of 3 | 1 of 3 |
| Action speed (average) | <2ms/act | 3-16ms | 4.2ms |
| Understanding accuracy (synthetic holdout) | >70% | N/A | N/A |
| VRAM usage | <1GB | <1GB | ~6GB |
| Time to correct game-type identification | <200 actions | N/A | ~50 |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Synthetic games don't match real game distribution | High | Aggressive augmentation + TTT bridges gap |
| Model overfits to synthetic archetypes | Medium | Color/action/spatial permutations force generalization |
| TTT destabilizes pretrained weights | Low | Low LoRA rank (4) + SGD + per-level reset |
| Understanding doesn't improve level completion | Medium | Fallback to pure graph exploration always available |
| Not enough time to build everything | Medium | Staged implementation: each stage independently useful |
| 3M params insufficient for game diversity | Low | Decoder heads constrain output; transformer capacity sufficient |

---

## File Structure

```
src/aria_v3/
├── synthetic_games/           # Synthetic game engines
│   ├── __init__.py
│   ├── base.py                # SyntheticGame base class
│   ├── navigation.py          # Grid navigation game
│   ├── click_puzzle.py        # Click/toggle game
│   ├── collection.py          # Navigate + collect
│   ├── mixed.py               # Nav + click combined
│   ├── push.py                # Sokoban-like
│   ├── conditional.py         # Context-dependent effects
│   └── generate.py            # Data generation pipeline
│
├── understanding/             # Learned understanding model
│   ├── __init__.py
│   ├── encoder.py             # CNN transition encoder
│   ├── temporal.py            # Temporal transformer
│   ├── decoder.py             # Understanding decoder heads
│   ├── model.py               # Full model (encoder + temporal + decoder)
│   ├── ttt.py                 # Test-time training loop
│   ├── train.py               # Pretraining script
│   └── dataset.py             # Dataset class for synthetic data
│
├── frame_processor.py         # CCL, hashing, regions (existing)
├── state_graph.py             # Graph exploration (existing)
├── three_layer_agent.py       # v3.1 agent (baseline comparison)
├── learning_engine.py         # Hand-coded Layer 2 (v3.1, reference)
├── reasoning_oracle.py        # LLM Layer 3 (v3.1, reference)
├── agent.py                   # v3 basic agent (baseline comparison)
└── ...                        # Other existing files
```
