# v3 Post-Mortem: Online Learning Agents (v3, v3.1, v3.2)

**Date:** 2026-02-10
**Covers:** v3 Basic Agent, v3 Dreamer, v3.1 Three-Layer Agent, v3.2 Understanding Agent
**Verdict:** 1 level across all approaches. Fundamental architectural flaws identified.

---

## Executive Summary

Over 4 iterations (v3, v3 Dreamer, v3.1, v3.2), we built increasingly sophisticated game
understanding systems. Each iteration added more complexity — CNN change predictors, world
models with imagination, LLM reasoning oracles, pretrained understanding models with TTT —
and each produced 0-1 levels on 3 public games. Meanwhile, the competition leader
(StochasticGoose) achieved 18/20 levels with a simpler architecture: one online CNN
predicting P(frame_change | state, action).

The core mistake was confusing "understanding the game" with "completing levels." Every
iteration tried to build a better game comprehension system when the bottleneck was
exploration efficiency — specifically, visual generalization across similar states.

---

## Results Summary

| Approach | Params | Speed | vc33 | ls20 | ft09 | Total Levels |
|----------|--------|-------|------|------|------|:------------:|
| v3 Basic | ~200K | 3-16ms | **1** | 0 | 0 | **1** |
| v3 Dreamer | ~5M | 6-7ms | 0 | 0 | — | 0 |
| v3.1 Three-Layer | ~7B (LLM) | 4.2ms | **1** | — | — | **1** |
| v3.2 Understanding | ~4.3M | 1.0-1.6ms | **1** | 0-1* | 0 | **1-2** |
| **StochasticGoose (1st)** | ~34M | ~1ms | — | — | — | **18/20** |

*ls20 level 1 achieved once at action 13,830 but not reproducible.

---

## v3 Basic Agent

**Architecture:** State graph (frame hash nodes, action edges) + CNN change predictor.
**File:** `src/aria_v3/agent.py`

### What Worked
- State graph deduplication: avoided retrying known-dead actions
- BFS to frontier: found untested states via graph navigation
- Frame hashing: near-instant state identity check
- Achieved **first ever level** (vc33 level 1 at action 84)

### What Failed
- Frame hashing treats every pixel arrangement as completely novel
- No visual generalization: open space here ≠ open space there
- 5700 unique nodes explored on ls20 without learning transferable navigation knowledge
- CNN change predictor too narrow: only predicts "will frame change?" not "where can I go?"
- Speed: 3-16ms/action (frame processing dominated)

### Key Insight
The state graph was the right foundation for exploration structure, but frame hashing
prevents amortizing knowledge across visually similar states. A navigation game with 5000
reachable positions requires visiting all 5000 instead of learning "open space = passable."

---

## v3 Dreamer Agent

**Architecture:** Transformer world model + imagination rollouts + policy optimization.
**File:** `src/aria_v3/dreamer_agent.py`, `src/aria_v3/world_model.py`

### What Worked
- World model achieved 95.5% dynamics prediction accuracy
- Imagination rollouts were computationally feasible

### What Failed
- **No reward signal.** The ARC-AGI-3 API provides only level_complete (binary, sparse).
  Dreamer needs dense reward to optimize policy. Without it, the policy random-walks
  through imagined trajectories with no gradient toward the goal.
- World model accuracy was misleading — predicting "nothing changes" is accurate most
  of the time, but useless for learning what actions make progress.
- 6-7ms/action too slow for the exploration throughput needed.

### Key Insight
World model accuracy ≠ agent capability. Knowing "what will happen" doesn't help if you
don't know "what should happen." The competition's sparse reward makes model-based planning
with reward optimization infeasible.

---

## v3.1 Three-Layer Agent

**Architecture:** State graph (Layer 1) + hand-coded frame statistics (Layer 2) +
Qwen2.5-7B LLM reasoning oracle (Layer 3).
**File:** `src/aria_v3/three_layer_agent.py`

### What Worked
- LLM correctly identified vc33 as a "collection" game
- Layer structure was clean: reactive execution / statistical learning / strategic reasoning
- 25 LLM calls at ~530ms each = 13.25s total for a 5K action game (< 3 min budget)

### What Failed
- **LLM hallucinated invalid strategies.** On vc33 (click-only game), the LLM generated a
  movement_map with directional actions that don't exist. Required manual validation code.
- **Hand-coded Layer 2 limited to pre-defined patterns.** Only detects shift, counter,
  region changes. Games with novel mechanics (rotation, conditional effects, gravity) go
  undetected.
- **Speed: 4.2ms/action** (Layer 1 overhead from complex scoring)
- **Competition winners used NO LLMs.** The top 3 entries used fast online learning,
  not language model reasoning. An LLM that runs 25 times per game cannot match an online
  CNN that runs at every state.

### Key Insight
Game understanding via LLM is a dead end for this competition. The LLM's text generation
is too slow, too hallucination-prone, and operates on a fundamentally wrong abstraction
(natural language description of pixel-art game mechanics). Competition winners proved
that explicit reasoning about game rules is unnecessary — implicit learned P(frame_change)
is sufficient.

---

## v3.2 Understanding Agent

**Architecture:** Pretrained CNN transition encoder + temporal transformer + decoder heads
(game type, entity roles, action effects, confidence) + TTT (LoRA, self-supervised
next-frame prediction) + empirical fallbacks + state graph execution.

**Files:**
- `src/aria_v3/understanding_agent.py` (agent, ~760 lines)
- `src/aria_v3/understanding/` (model, training, TTT)
- `src/aria_v3/synthetic_games/` (data generation)

### Training Results (Synthetic Games)

| Metric | Value |
|--------|-------|
| Game type accuracy (val) | **100%** |
| Shift MAE (val) | **0.128** pixels |
| Entity role BCE | 0.19 |
| Frame prediction CE | 0.87 |
| Training time | 19 min (RTX 4090) |
| Total params | ~4.3M |
| Synthetic sequences | 3600 (3 archetypes × 200 configs × 6 variants) |

### Agent Results (Real Games)

| Game | Actions | Levels | Speed | Notes |
|------|---------|--------|-------|-------|
| vc33 | 5K | **1** | 1.0ms | Level 1 @ action 84. Walls=[5,9], collectibles=[4,7,14] |
| ls20 | 20K | **0-1** | 1.2ms | Level 1 once at action 13,830. Usually 0. 6300+ nodes explored. |
| ft09 | 10K | 0 | 1.6ms | 551 reachable nodes fully explored. 43% dead edges. |

### What Worked

1. **Speed: 1.0-1.2ms/action.** Fastest of all approaches. The CNN encoder + structured
   decoder heads eliminated LLM latency entirely.

2. **Empirical fallbacks compensated for model failure.** When the pretrained model
   predicted wrong entity roles and change probabilities, empirical tracking from actual
   observations provided correct signals:
   - Movement map: all 4 directions correct on ls20 within 100 steps
   - Entity detection: player, walls, collectibles identified from frame statistics
   - Change probability: Bayesian blending overrode model's 0.0 prediction for action 4
     on ls20 (real rate: 99%+)

3. **Committed frontier navigation.** Full BFS path execution (not just first step)
   eliminated oscillation between frontiers. Explored 6319 nodes in 20K actions on ls20
   with no saturation (vs 4910 with saturation before fix).

4. **Shift classification over regression.** Key architectural insight: predicting shift
   as 7-class classification {-16,-8,-4,0,+4,+8,+16} instead of MSE regression. With
   randomized action mappings, mean shift is (0,0), driving regression to zero. CE
   classification solved this (MAE: 1.5 → 0.13).

### What Failed

1. **Pretrained model outputs wrong on every real game.** Entity roles, change_prob, and
   game type predictions from the pretrained model were incorrect on all 3 real games.
   The agent worked DESPITE the model, not because of it. Every useful signal came from
   empirical fallbacks, which are cheaper to compute and don't need pretraining.

2. **Synthetic pretraining = wrong distribution.** 100% accuracy on synthetic games,
   0% useful transfer to real games. The 3 synthetic archetypes (navigation, click,
   collection) don't match the diversity of 150+ real games. Augmentations (color
   permutation, action remapping, spatial flips) help within-distribution but don't
   generalize to truly novel game mechanics.

3. **TTT couldn't fix wrong priors.** LoRA rank 4 (10.7K trainable params) fighting
   against 578K encoder params encoding incorrect assumptions about game structure.
   SGD lr=0.01 with ~500 updates per 5K actions wasn't enough to override pretrained
   biases. Starting from scratch (no pretrained weights) would be cheaper.

4. **State graph doesn't generalize visually.** The core exploration engine (frame hash →
   node) treats every pixel arrangement as unique. On ls20's spatial grid, this means
   visiting all 6000+ positions individually instead of learning "open space = passable."
   StochasticGoose's CNN generalizes: if a 5x5 patch looks like open space, movement
   succeeds there too — even if that specific frame was never visited.

5. **Unified scoring is a complexity tax.** Four hand-tuned weights (graph novelty,
   change_prob, blocked_prob, pathfinding) combining signals that are mostly wrong. Each
   additional signal required debugging and calibration. StochasticGoose uses one signal:
   P(frame_change). Simpler and more robust.

6. **ft09 state space fully explored but goal not reached.** 551 reachable nodes,
   all edges tested, 43% dead. The puzzle requires specific click sequences for color
   cycling that pure graph exploration can't systematically discover. Understanding the
   game mechanics (what each click does) would help, but the model couldn't learn this
   from synthetic data.

7. **ls20 level completion is stochastic.** Achieved level 1 once at action 13,830 but
   not reproducible. The game requires key-lock color matching — the agent must collect
   keys of the right color to open matching doors. Without understanding this mechanic,
   success is random walk through a large state space.

### Bugs Found and Fixed During Development

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| Action 4 suppressed on ls20 | Model change_prob=0.00, real=99%+ | Bayesian blending with empirical tracking |
| Understanding rapid-fire after level | step_count global, schedule re-triggered | Added level_step_count (per-level) |
| Movement map inverted | Tracked background color (largest), not player | Track smallest-area color |
| numpy division warning | color_present_count=0 | safe_count with np.where |
| Frontier oscillation | One step toward frontier, then redirect | Full BFS path commitment |
| Graph saturation at 4910 nodes | Pathfinding toward wrong targets | Reduced pathfinding weight 0.3→0.1 |

---

## The StochasticGoose Comparison

### What StochasticGoose Does

1. **One online CNN** (~34M params) predicting P(frame_change | state, action)
2. **No pretraining.** Model starts fresh for each game. Learns online from observations.
3. **CNN visual generalization.** Convolutional structure shares knowledge across visually
   similar states. Open space in top-left → open space in bottom-right.
4. **~42K actions per game** (3 min budget, ~1ms/action).
5. **Per-state-per-action prediction.** The CNN takes the current frame as input and
   outputs P(frame_change) for each action. Can predict blocked directions before trying
   them.
6. **Result: 18/20 levels** on leaderboard games.

### Why They Win and We Don't

| Dimension | Our Approach (v3.2) | StochasticGoose |
|-----------|---------------------|-----------------|
| Core question | "What IS this game?" | "Will frame change?" |
| State representation | Frame hash (no generalization) | CNN features (spatial generalization) |
| Learning | Pretrained + TTT fine-tune | Online-only (no wrong priors) |
| Model outputs | 6 decoder heads (entity roles, game type, shift, change_prob, blocked_prob, confidence) | 1 output: P(frame_change) per action |
| Signals used | 4 hand-tuned weights | 1 signal |
| Complexity | ~760 lines, 6 bugs found | Simple architecture, fewer failure modes |
| Visual generalization | None (per-hash) | Full CNN spatial transfer |
| Levels | 1-2 | 18/20 |

**The fundamental gap:** Our approach treats game understanding as a prerequisite for level
completion. StochasticGoose proves it isn't. Efficient exploration with visual generalization
is sufficient — you don't need to know the game rules to complete levels.

---

## Lessons Learned (Accumulated v3-v3.2)

### 1. Speed × Actions > Sophistication
All winning entries operate at ~1ms/action with ~42K actions/game. A simple model that makes
42K well-chosen actions beats a sophisticated model making 5K poorly-chosen actions. We
achieved the speed (1.0ms) but not the action quality.

### 2. Visual Generalization Is The Bottleneck
Frame hashing creates a unique node for every frame. A CNN shares knowledge across frames
with similar visual features. For spatial games like ls20 (grid navigation), this is the
difference between exploring 6000 unique nodes and knowing that open-space movement works
everywhere from a few examples.

### 3. One Clear Objective > Multiple Confused Objectives
StochasticGoose predicts one thing: P(frame_change). We predicted 6 things: entity roles,
game type, shift vectors, change probability, blocked probability, and confidence. Most
were wrong. Each additional output head is another potential failure mode.

### 4. Online Learning > Pretraining + Fine-tuning
Our pretrained model had 100% accuracy on synthetic games and 0% useful transfer to real
games. StochasticGoose learns from scratch on each game. No prior to override means no
distribution mismatch. The TTT approach (LoRA fine-tuning a pretrained model) is strictly
worse than training from scratch when the pretraining distribution is wrong.

### 5. Empirical Fallbacks Proved the Model Was Unnecessary
Every useful signal in v3.2 came from empirical observations (frame statistics, per-action
change rates, centroid displacement tracking), not from the pretrained model. This is
direct evidence that the 4.3M parameter understanding model was wasted complexity.

### 6. Exploration Efficiency > Game Understanding
The competition asks "complete levels", not "understand games." Understanding is one path
to completion, but it's not the only one and it's not the cheapest one. StochasticGoose
completes 18/20 levels without knowing what a "player" or "wall" is.

### 7. Each Iteration Added Complexity Without Adding Levels
- v3 Basic: State graph + CNN → 1 level
- v3 Dreamer: + world model + imagination → 0 levels
- v3.1: + hand-coded stats + LLM → 1 level
- v3.2: + pretrained encoder + transformer + decoder + TTT → 1-2 levels

Four iterations, 4 months of development, and the level count barely moved. The approach
was wrong from the start — more understanding doesn't produce more levels when the
bottleneck is exploration generalization.

---

## What to Do Next

### Recommended: v4 — Online P(frame_change) CNN

Based on StochasticGoose's proven approach:

1. **One CNN** predicting P(frame_change | frame, action) for all actions simultaneously
2. **Online-only learning** — model starts fresh each game, trains on observations
3. **Visual generalization** — convolutional structure transfers knowledge spatially
4. **State graph retained** — for deduplication and dead-end pruning (our strongest component)
5. **No pretraining, no entity detection, no game type classification**

### What to Keep from v3

- State graph (`state_graph.py`) — deduplication and frontier navigation are sound
- Frame processor (`frame_processor.py`) — hashing and segmentation still useful
- Competition API integration — game loop and action interface
- Speed discipline — 1ms/action target

### What to Discard

- Pretrained understanding model (`understanding/`)
- Synthetic game framework (`synthetic_games/`)
- TTT engine
- Entity role detection (model and empirical)
- Game type classification
- Unified scoring with multiple decoder heads
- Pathfinding with movement maps

---

## Appendix: File Inventory (v3.2)

### Active (to keep for v4)
| File | Lines | Purpose |
|------|-------|---------|
| `state_graph.py` | 276 | Graph exploration, BFS frontier navigation |
| `frame_processor.py` | ~200 | CCL segmentation, frame hashing |

### Superseded (archive after v4 launch)
| File | Lines | Purpose |
|------|-------|---------|
| `understanding_agent.py` | ~760 | v3.2 agent (this post-mortem) |
| `three_layer_agent.py` | ~400 | v3.1 agent |
| `learning_engine.py` | ~200 | v3.1 hand-coded Layer 2 |
| `reasoning_oracle.py` | ~300 | v3.1 LLM Layer 3 |
| `agent.py` | ~300 | v3 basic agent |
| `change_predictor.py` | ~150 | v3 online CNN |
| `world_model.py` | ~500 | v3 Dreamer world model |
| `dreamer_agent.py` | ~400 | v3 Dreamer agent |
| `understanding/encoder.py` | ~200 | v3.2 CNN encoder |
| `understanding/temporal.py` | ~200 | v3.2 temporal transformer |
| `understanding/decoder.py` | ~260 | v3.2 decoder heads |
| `understanding/model.py` | ~300 | v3.2 full model |
| `understanding/ttt.py` | ~200 | v3.2 TTT engine |
| `understanding/train.py` | ~300 | v3.2 training script |
| `understanding/dataset.py` | ~200 | v3.2 dataset class |
| `synthetic_games/*.py` | ~1000 | v3.2 synthetic game engines |

---

## Appendix: Competition Leaderboard Context

As of 2026-02-10:
- **1st place: StochasticGoose** — 18/20 levels
- Our best: 1-2 levels across 3 games

The top 3 competition entries share these traits:
1. Online-only learning (no pretraining)
2. CNN visual features (not frame hashing)
3. Single clear prediction target (P(frame_change) or similar)
4. No LLMs, no entity detection, no game type reasoning
5. ~42K actions per game at ~1ms/action
