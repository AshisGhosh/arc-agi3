# v4 Architecture: Online P(state_novelty) CNN

**Status:** Active development
**Approach:** StochasticGoose-inspired online CNN predicting P(novel_state | state, action)
**Results:** 6 levels across 3 games (vc33=3, ls20=1, ft09=2)
**Speed:** 0.8-1.8ms/act (110K+ actions/game in 3 min budget)

---

## Design Principles

1. **One question:** P(novel_state | state, action) — nothing else
2. **Online-only learning:** Model starts from scratch each level, no pretraining
3. **CNN visual generalization:** Convolutional structure transfers knowledge spatially
4. **Hybrid action selection:** Graph BFS for nav-only (systematic), CNN for click/mixed
5. **Reset per level:** Fresh model for each game stage
6. **Random restart:** If stuck, reset CNN weights (keep novelty tracking)
7. **Softmax temperature:** Sharp sampling (temp=0.5) focuses on CNN's best predictions
8. **Epsilon-greedy:** Decaying random exploration (20%→5%) prevents overfitting
9. **Speed:** 0.8-1.8ms/act allows 110K+ actions/game in competition time budget

---

## Architecture

```
┌──────────────── v4 Agent (34M params, goose) ─────────────────────────┐
│                                                                         │
│  Game Engine → frame [64,64] (16 colors)                               │
│                                                                         │
│  ┌─ CNN Model (34M params, goose architecture) ──────────────────┐   │
│  │                                                                  │   │
│  │  Input: one_hot(frame) → [16, 64, 64]                          │   │
│  │                                                                  │   │
│  │  Backbone (4 conv at full 64x64 resolution):                    │   │
│  │    Conv2d(16→32, k=3, pad=1) + ReLU                             │   │
│  │    Conv2d(32→64, k=3, pad=1) + ReLU                             │   │
│  │    Conv2d(64→128, k=3, pad=1) + ReLU                            │   │
│  │    Conv2d(128→256, k=3, pad=1) + ReLU                           │   │
│  │    → [256, 64, 64] feature map (full resolution)                │   │
│  │                                                                  │   │
│  │  Action Head (for actions 1-5):                                  │   │
│  │    MaxPool(4) → [256, 16, 16]                                   │   │
│  │    Flatten → [65536] (preserves spatial info)                   │   │
│  │    Linear(65536→512) + ReLU + Dropout(0.2)                      │   │
│  │    Linear(512→5) → raw logits per action                        │   │
│  │                                                                  │   │
│  │  Coordinate Head (for action 6 = click):                        │   │
│  │    Conv2d(256→128, k=3, pad=1) + ReLU                           │   │
│  │    Conv2d(128→64, k=3, pad=1) + ReLU                            │   │
│  │    Conv2d(64→32, k=1) + ReLU                                    │   │
│  │    Conv2d(32→1, k=1) → [1, 64, 64] raw logits per pixel        │   │
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
│  ┌─ Action Selection ───────────────────────────────────────────┐    │
│  │                                                                  │   │
│  │  Nav-only games: Graph BFS (untested actions, frontier BFS)     │   │
│  │  Click/mixed games: CNN-guided softmax sampling                  │   │
│  │                                                                  │   │
│  │  CNN selection:                                                   │   │
│  │    0. Epsilon-greedy: random with P = max(0.05, 0.2*0.5^(t/200))│   │
│  │    1. CNN forward pass → action_logits[5], coord_logits[64,64]  │   │
│  │    2. Simple actions: raw logit per action (skip dead/pruned)    │   │
│  │    3. Click: top-K=16 pixels from coord heatmap + random click  │   │
│  │    4. Softmax(logits / temperature) → sample action             │   │
│  │                                                                  │   │
│  │  CNN always trains (even during graph BFS exploration)           │   │
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
| Model size | goose (34M) | Larger capacity enables L3+ on click games |
| Learning rate | 1e-4 | Adam |
| Batch size | 32 | Fast training steps, reduced from 64 |
| Train frequency | Every 20 actions | Ablation tested: 5/10/20, sweet spot at 20 |
| Buffer size | 50K | Deduped by (frame_hash, action) |
| Temperature | 0.5 | Softmax sharpness; 0.5 optimal (1.0 too uniform, 0.3 too greedy) |
| Epsilon | 0.2→0.05 decay | max(0.05, 0.2 * 0.5^(train_steps/200)) |
| Entropy coeff | 1e-4 | Prevents probability collapse |
| Dropout | 0.2 | Action head only (goose model) |
| Restart threshold | 5000 steps | Reset if no novel state found |
| Max restarts/level | 3 | Prevent infinite restarts |

---

## Per-Step Flow

```
Step N:
  1. Receive frame from game
  2. Hash frame (SipHash on tobytes)
  3. Record previous (state, action, novelty_target) in buffer
  4. Track novelty (add frame_hash to seen_states)
  5. If step % 20 == 0: train model (sample 32 from buffer)
  6. Check for restart (5000 steps without novel state)
  7. Select action:
     - Nav-only: graph BFS (untested → frontier → CNN fallback)
     - Click/mixed: epsilon-greedy → CNN softmax(logits/0.5)
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

| Aspect | StochasticGoose (34M) | v4.2 (34M goose) |
|--------|----------------------|-------------------|
| Model size | 34M params | 34M params (same arch) |
| Spatial features | Flatten(65536) | Flatten(65536) |
| Training target | P(frame_change) | P(novel_state) — continuous decay |
| Action selection | Sigmoid proportional | Softmax(logits/0.5) + epsilon-greedy |
| State dedup | Hash set | Hash-deduped buffer |
| Click handling | Raw 64x64 | CNN heatmap top-K=16 pixels |
| Stuck recovery | None documented | Random restart (5K threshold) |
| Nav games | CNN only | Graph BFS + CNN fallback |
| Normalization | No (plain Conv+ReLU) | No (same — goose arch) |

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
| v4.1 + speed opt | 2 levels | 1 level | 2 levels | 0.3-1.4ms | 5 |
| **v4.2 goose+temp+eps** | **3 levels** | **1 level** | **2 levels** | **0.8-1.8ms** | **6** |

**v4.2 config:** goose 34M, temp=0.5, epsilon-greedy (0.2→0.05), train every 20

**Key finding:** v4.1 ablation showed speed > sophistication with sigmoid sampling.
v4.2 overturned this: softmax temperature + epsilon-greedy unlocks larger model capacity.
See [V4 Ablation Study](../findings/V4-ABLATION-STUDY.md).
