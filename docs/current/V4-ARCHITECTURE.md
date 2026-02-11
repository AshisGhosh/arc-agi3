# v4 Architecture: Online P(frame_change) CNN

**Status:** Active development
**Approach:** StochasticGoose-inspired online CNN predicting P(frame_change | state, action)
**Goal:** >3 levels across 3 games, ~1ms/action

---

## Design Principles

1. **One question:** P(frame_change | state, action) — nothing else
2. **Online-only learning:** Model starts from scratch each game, no pretraining
3. **CNN visual generalization:** Convolutional structure transfers knowledge spatially
4. **State graph for structure:** Deduplication, dead-end pruning, frontier navigation
5. **Reset per level:** Prevents catastrophic forgetting between game stages

---

## Architecture

```
┌──────────────── v4 Agent ──────────────────────────────────────────────┐
│                                                                         │
│  Game Engine → frame [64,64] (16 colors)                               │
│                                                                         │
│  ┌─ CNN Model (~2.5M params) ──────────────────────────────────────┐   │
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
│  │    AdaptiveAvgPool → [256, 1, 1] → flatten                     │   │
│  │    Linear(256→128) + ReLU + Dropout(0.2)                        │   │
│  │    Linear(128→5) → sigmoid → P(frame_change) per action         │   │
│  │                                                                  │   │
│  │  Coordinate Head (for action 6 = click):                        │   │
│  │    ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + ReLU         │   │
│  │    ConvTranspose2d(128→64, k=4, s=2, p=1) + BN + ReLU          │   │
│  │    ConvTranspose2d(64→32, k=4, s=2, p=1) + BN + ReLU           │   │
│  │    Conv2d(32→1, k=1) → [1, 64, 64]                             │   │
│  │    → sigmoid → P(frame_change) per pixel                        │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Experience Buffer (max 50K unique) ────────────────────────────┐   │
│  │  Stores: (frame_hash, action_idx, frame_changed)                │   │
│  │  Hash dedup: same (state, action) not stored twice              │   │
│  │  Cleared on level complete                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ State Graph (from v3) ─────────────────────────────────────────┐   │
│  │  Nodes = frame hashes, Edges = (action, SUCCESS/DEAD/UNTESTED)  │   │
│  │  BFS frontier navigation with committed full paths              │   │
│  │  Dead-end backward propagation                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Action Selection ──────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. If frontier_plan has queued actions → pop next               │   │
│  │  2. Get CNN predictions for current frame                       │   │
│  │  3. Mask dead actions (from state graph)                        │   │
│  │  4. Boost untested actions (from state graph)                   │   │
│  │  5. Sample from distribution (sigmoid probs + exploration)      │   │
│  │  6. If no local untested → commit to BFS frontier path          │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  → action (type, x, y) → Game Engine                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Key Differences from StochasticGoose

| Aspect | StochasticGoose | v4 |
|--------|-----------------|-----|
| State dedup | Hash set in buffer only | State graph + hash buffer |
| Dead-end pruning | None (learns from data) | Backward propagation in graph |
| Frontier navigation | None (stochastic sampling) | BFS committed paths |
| Click space | Raw 64x64 sampling | Region centroids (CCL) + raw fallback |
| Batch norm | No | Yes (faster convergence) |
| Backbone pooling | MaxPool(4,4) in action head | MaxPool(2) in backbone + AdaptiveAvgPool |

Our additions (state graph, frontier navigation, region clicks) should help
on games that StochasticGoose struggles with — particularly click puzzles
where 4096 positions is too large to sample randomly.

---

## Training

**Loss:** Binary cross-entropy per action
```
L = BCE(predicted_change_prob, actual_changed) + entropy_reg
entropy_reg = -α * H(action_probs) - β * H(coord_probs)
```

**Hyperparameters:**
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning rate | 1e-4 | Adam, matches StochasticGoose |
| Batch size | 64 | Fits in single forward pass |
| Train frequency | Every 5 actions | Balance speed vs learning |
| Buffer size | 50K | Deduped, smaller than SG's 200K |
| Entropy coeff (action) | 1e-4 | Encourage exploration |
| Entropy coeff (coord) | 1e-5 | Lighter for coordinates |
| Dropout | 0.2 | Action head only |

**Training trigger:** Every 5 actions, sample batch of 64 from buffer, one
gradient step. ~0.5ms per training step on 4090.

---

## Action Selection

```python
def select_action(frame, available_actions, state_graph, model):
    # 1. Check frontier plan (committed BFS path)
    if frontier_plan:
        return frontier_plan.popleft()

    # 2. Get CNN predictions
    probs = model(frame)  # action_probs[5], coord_probs[64,64]

    # 3. Build candidate scores
    for each available action:
        if state_graph says DEAD → score = 0
        elif state_graph says UNTESTED → score = prob + exploration_bonus
        else → score = prob

    # 4. For click actions: use region centroids
    if action_6 available:
        for each region:
            click_prob = coord_probs[region.centroid_y, region.centroid_x]
            if untested in graph → click_prob += exploration_bonus

    # 5. Sample from distribution
    action = sample(scores)

    # 6. If all local actions tested → commit to frontier path
    if no untested actions locally:
        path = state_graph.get_full_path_to_frontier()
        if path: frontier_plan = deque(path)
```

---

## Per-Step Flow

```
Step N:
  1. Receive frame from game
  2. Hash frame, register in state graph
  3. Record previous (state, action, frame_changed) in buffer
  4. Update state graph edge (prev → current)
  5. If step % 5 == 0: train model (sample 64 from buffer)
  6. Select action (CNN prediction + graph info)
  7. Execute action
```

**Target timing:**
| Operation | Time |
|-----------|------|
| Frame hash | 0.02ms |
| State graph update | 0.01ms |
| CNN forward | 0.3ms |
| Training (amortized) | 0.1ms |
| Action selection | 0.05ms |
| **Total** | **~0.5ms** |

---

## Level Management

On level complete:
1. Reset CNN model weights (fresh random init)
2. Clear experience buffer
3. Clear state graph
4. Reset frontier plan

Each level is a fresh learning problem. No transfer between levels.

---

## Files

```
src/v4/
├── agent.py          # Main agent (game loop, action selection, training)
├── model.py          # CNN model (backbone + action head + coord head)
└── (uses from v3)
    ├── state_graph.py    # Imported from aria_v3
    └── frame_processor.py # Imported from aria_v3
```

---

## Success Criteria

| Metric | Target | v3.2 Best |
|--------|--------|-----------|
| vc33 levels | >=1 | 1 |
| ls20 levels | >=2 | 0-1 |
| ft09 levels | >=1 | 0 |
| Total levels | >3 | 1-2 |
| Speed | <1ms/action | 1.0-1.2ms |
| Actions per game | ~40K | ~20K |
