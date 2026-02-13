# v4 Architecture: Online P(state_novelty) CNN

**Status:** Active development
**Approach:** StochasticGoose-inspired online CNN predicting P(novel_state | state, action)
**Results:** 5 levels across 3 games (vc33=2, ls20=1, ft09=2)
**Speed:** 0.3-1.4ms/act (120K actions/game in 3 min budget)

---

## Design Principles

1. **One question:** P(novel_state | state, action) — nothing else
2. **Online-only learning:** Model starts from scratch each level, no pretraining
3. **CNN visual generalization:** Convolutional structure transfers knowledge spatially
4. **No state graph for action selection:** CNN-guided stochastic sampling only
5. **Reset per level:** Fresh model for each game stage
6. **Random restart:** If stuck, reset CNN weights (keep novelty tracking)
7. **Speed first:** 0.3-1.4ms/act allows 120K actions/game in competition time budget

---

## Architecture

```
┌──────────────── v4 Agent (5.3M params) ───────────────────────────────┐
│                                                                         │
│  Game Engine → frame [64,64] (16 colors)                               │
│                                                                         │
│  ┌─ CNN Model (5.3M params, 0.32ms forward) ─────────────────────┐   │
│  │                                                                  │   │
│  │  Input: one_hot(frame) → [16, 64, 64]                          │   │
│  │                                                                  │   │
│  │  Backbone (shared):                                              │   │
│  │    Conv2d(16→32, k=3, pad=1) + BN + ReLU                       │   │
│  │    Conv2d(32→64, k=3, pad=1) + BN + ReLU + MaxPool(2)          │   │
│  │    Conv2d(64→128, k=3, pad=1) + BN + ReLU + MaxPool(2)         │   │
│  │    Conv2d(128→256, k=3, pad=1) + BN + ReLU + MaxPool(2)        │   │
│  │    → [256, 8, 8] feature map                                    │   │
│  │                                                                  │   │
│  │  Action Head (for actions 1-5):                                  │   │
│  │    Flatten → [16384]  (preserves spatial info)                  │   │
│  │    Linear(16384→256) + ReLU + Dropout(0.2)                      │   │
│  │    Linear(256→5) → sigmoid → P(novel_state) per action          │   │
│  │                                                                  │   │
│  │  Coordinate Head (for action 6 = click):                        │   │
│  │    ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + ReLU         │   │
│  │    ConvTranspose2d(128→64, k=4, s=2, p=1) + BN + ReLU          │   │
│  │    ConvTranspose2d(64→32, k=4, s=2, p=1) + BN + ReLU           │   │
│  │    Conv2d(32→1, k=1) → [1, 64, 64]                             │   │
│  │    → sigmoid → P(novel_state) per pixel                         │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Experience Buffer (max 200K unique) ─────────────────────────┐    │
│  │  Stores: (frame, frame_hash, action_idx, target)               │    │
│  │  Hash dedup: same (frame_hash, action) not stored twice        │    │
│  │  Target = 1.0 if (frame_changed AND state_novel) else 0.0     │    │
│  │  Cleared on level complete and on random restart               │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─ Novelty Tracking ────────────────────────────────────────────┐    │
│  │  seen_states: set of frame hashes (never cleared within game)  │    │
│  │  Persists across restarts — prevents re-exploring known space  │    │
│  │  Target=0 for: no frame change, OR frame changed to known state│    │
│  │  Target=1 for: frame changed to never-before-seen state        │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─ Random Restart ──────────────────────────────────────────────┐    │
│  │  If no novel state in 5000 steps: reset CNN + clear buffer     │    │
│  │  Keep seen_states (don't re-explore known territory)           │    │
│  │  Max 3 restarts per level                                      │    │
│  └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─ Action Selection (CNN-guided stochastic sampling) ───────────┐    │
│  │                                                                  │   │
│  │  If buffer < batch_size: uniform random                         │   │
│  │  Otherwise:                                                      │   │
│  │    1. CNN forward pass → action_probs[5], coord_probs[64,64]   │   │
│  │    2. Simple actions: prob from action_probs, min 0.01          │   │
│  │    3. Click actions: prob from coord_probs at region centroids  │   │
│  │       + random click with p=0.02                                │   │
│  │    4. Normalize and sample from distribution                    │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  → action (type, x, y) → Game Engine                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Speed Optimizations (2026-02-12)

Six critical changes reduced action time from 3.0-4.6ms to 0.3-1.4ms:

1. **Hash comparison:** Replaced `np.array_equal(frame, prev_frame)` with `hash(frame) == prev_hash`
   - Frame change detection now O(1) instead of O(4096)

2. **No frame.copy():** Removed unnecessary frame copies in hot path
   - Memory allocation is expensive, buffer references are cheap

3. **Fast hash:** Replaced `blake2b(frame.tobytes()).hexdigest()` with `hash(frame.tobytes())`
   - All hash types changed: str → int
   - Python's SipHash is optimized and doesn't require hex encoding

4. **No CCL segmentation:** Eliminated scipy CCL entirely
   - CNN heatmap top-K pixels replace region centroids for click actions
   - Removed scipy dependency

5. **Pre-allocated one-hot tensors:** Use `scatter_` on pre-allocated buffers instead of `F.one_hot`
   - Avoids repeated tensor allocation in training loop

6. **GPU-only training indexing:** Keep sampled indices on GPU during batch creation
   - No CPU round-trips during training

**Result:** ls20=0.3ms, vc33=1.4ms, ft09=1.3ms (single-game speeds)

**Bug fix:** Click action type 6 was incorrectly pruned after 20 random misses. Fixed — click is never pruned.

---

## Novelty Signal (Key Innovation)

**Problem:** P(frame_change) is uninformative for many games:
- ls20: 99% of actions change the frame (player always moves) — CNN can't differentiate
- ft09: Game-over resets to start → frame "changed" but counterproductive

**Solution:** Target = `frame_changed AND state_novel`:
- ls20: Only ~15% of actions lead to never-seen states → CNN learns directional preferences
- ft09: Game-over resets to well-known start state → target=0 → CNN avoids game-over moves
- vc33: Similar to frame_change (each new click state is novel anyway)

**Non-stationarity:** As exploration progresses, novel rate decays toward 0 (all states seen).
This is expected and acceptable — by the time signal decays, CNN has learned enough to guide exploration.

---

## Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Adam |
| Batch size | 32 | Fast training steps, reduced from 64 |
| Train frequency | Every 20 actions | Ablation tested: 5/10/20, sweet spot at 20 |
| Buffer size | 50K | Deduped by (frame_hash, action) |
| Entropy coeff | 1e-4 | Prevents probability collapse |
| Dropout | 0.2 | Action head only |
| Restart threshold | 5000 steps | Reset if no novel state found |
| Max restarts/level | 3 | Prevent infinite restarts |

---

## Per-Step Flow

```
Step N:
  1. Receive frame from game
  2. Hash frame (blake2b)
  3. Record previous (state, action, novelty_target) in buffer
  4. Track novelty (add frame_hash to seen_states)
  5. If step % 10 == 0: train model (sample 64 from buffer)
  6. Check for restart (5000 steps without novel state)
  7. Select action (CNN prediction, stochastic sample)
  8. Execute action
```

---

## Level Management

On level complete:
1. Reset CNN model weights (fresh random init)
2. Clear experience buffer
3. Clear seen_states (new level = new state space)
4. Reset restart counter

Each level is a fresh learning problem. No transfer between levels.

---

## Key Differences from StochasticGoose

| Aspect | StochasticGoose (34M) | v4 (5.3M) |
|--------|----------------------|------------|
| Model size | 34M params | 5.3M params |
| Spatial features | Flatten(65536) | Flatten(16384) |
| Training target | P(frame_change) | P(novel_state) |
| State dedup | Hash set | Hash-deduped buffer |
| Click handling | Raw 64x64 | Region centroids (CCL) |
| Stuck recovery | None documented | Random restart (5K threshold) |
| Batch norm | No | Yes |
| Backbone | 5 conv layers | 4 conv layers |

---

## Files

```
src/v4/
├── __init__.py
├── __main__.py       # Entry point
├── agent.py          # Main agent (game loop, action selection, training)
├── model.py          # CNN model (backbone + action head + coord head)
└── (imports from v3)
    └── frame_processor.py # CCL segmentation for click targets
```

---

## Results

| Version | vc33 | ls20 | ft09 | Speed | Total |
|---------|------|------|------|-------|-------|
| v4.0 (P(frame_change)) | 2 levels | 0 | 0 | 2.6-3.5ms | 2 |
| v4.1 (novelty signal) | 2 levels | 1 level | 1 level | 3.0-4.6ms | 4 |
| v4.1 + speed opt | 2 levels | 1 level | 2 levels | 0.3-1.4ms | **5** |
| v4.1 ablation (goose 34M) | 2 levels | N/A | N/A | 2.9ms | 2 |
| v4.1 ablation (combined) | 2 levels | N/A | N/A | 14.1ms | 2 |

**Ablation study:** Speed > model sophistication. 34M params learns 19x faster per-action but is 10x slower wall-time. See [V4 Ablation Study](../findings/V4-ABLATION-STUDY.md).

**Baseline config:** 1.9M params, train every 20 steps, no persist, 0.3-1.4ms/act
