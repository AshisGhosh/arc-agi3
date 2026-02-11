# Project Progress

## Current State
**Phase:** v4 — Online P(frame_change) CNN (StochasticGoose-inspired)
**Branch:** main
**Status:** v4 agent implemented and tested. 2 levels on vc33 (best ever). CNN-guided action selection.

## Immediate Next Step
**Optimize v4 for more levels:**
1. Improve ls20 — CNN can't differentiate productive vs unproductive movement (99% frame change rate)
2. Improve ft09 — high game-over rate (299 resets in 10K), need better click targeting
3. Speed optimization — currently 2.6-2.9ms/action, target <1.5ms
4. Try larger action budgets (3 min time budget allows ~70K actions)

---

## v4 Results (Online P(frame_change) CNN)

### v4 Agent (CNN-guided stochastic sampling, no pretraining)
| Game | Actions | Levels | Speed | Notes |
|------|---------|--------|-------|-------|
| **vc33** | 40K | **2** | 2.9ms/act | Level 1 @ action 86, Level 2 @ action 4462. Best ever. |
| ls20 | 40K | 0 | 2.6ms/act | 99% frame change rate — CNN can't differentiate movements |
| ft09 | 10K | 0 | 3.5ms/act | 299 resets, 1715 unique experiences. High game-over rate |

**Key improvements over v3.2:**
- vc33: 2 levels (was 1). CNN learns which regions to click.
- CNN-guided action selection replaces graph-based exploration
- Simpler architecture: no pretraining, no entity detection, no game type classification
- Clean online learning: model resets per level, learns from scratch

**Architecture:** `src/v4/` — [V4 Architecture](current/V4-ARCHITECTURE.md)
- 1.1M param CNN (16→32→64→128→256 backbone)
- Action head: 5 logits for simple actions 1-5
- Coordinate head: 64x64 spatial map for click positions
- Online training: BCE loss + entropy regularization, every 5 actions
- Hash-deduped experience buffer (200K max)

---

## v3.2 Architecture (Learned Understanding + TTT)

```
Per-transition CNN encoder (0.3ms/step, ~580K params)
  Input: one_hot(frame_t) + one_hot(frame_t+1) + action_embed
  Output: 256-dim transition embedding
  ↓ stored in rolling buffer (last 200)
Temporal Transformer (15ms every 100 actions, ~2M params)
  Input: last 100 transition embeddings
  16 learnable query tokens
  Output: 16 x 256-dim "understanding state"
  ↓
Understanding Decoder Heads (0.1ms, ~520K params)
  Action-Effect Head: per action → (shift_vector, change_prob, affected_color)
  Entity-Role Head: per color → (player, wall, collectible, background, counter)
  Game-Type Head: → classification
  Confidence Head: → scalar 0-1
  ↓ structured understanding
Layer 1: Reactive (<2ms, every action)
  State graph + plan executor. Uses understanding to guide exploration.
```

**TTT (Test-Time Training):** LoRA rank 4 on CNN encoder, SGD lr=0.01.
Self-supervised target: predict next frame. Adapts to each game during play.

**Key differences from v3.1:**
- Layer 2 (hand-coded stats) → learned CNN + transformer (discovers concepts)
- Layer 3 (Qwen2.5-7B LLM) → structured decoder heads (constrained, fast, no hallucination)
- Pre-trained on diverse synthetic games (not 3 demos)
- TTT adapts to each game online (not just prompting)

**Total: ~3M params, ~6MB VRAM, ~1.8ms/action amortized.**

---

## v3.2 Results (Understanding Agent)

### Understanding Agent (pretrained CNN+Transformer + TTT + empirical detection)
| Game | Actions | Levels | Speed | Notes |
|------|---------|--------|-------|-------|
| **vc33** | 5K | **1** | 1.0ms/act | Level 1 @ action 84. walls=[5,9], collectibles=[4,7,14] |
| **ls20** | 20K | **0-1** | 1.2ms/act | Level 1 sometimes found (~14K actions). Movement map correct. |
| ft09 | 10K | 0 | 1.6ms/act | State space saturated at 551 nodes, 43% dead edges |

**Key improvements over v3/v3.1:**
- Empirical entity detection: player, walls, collectibles identified from frame stats
- Empirical movement map: all 4 directions correct on ls20 by step ~100
- Blended model/empirical change_prob: fixes model suppressing valid actions
- Committed frontier navigation: full BFS path execution (no oscillation)
- Speed: 1.0-1.2ms/action (3x faster than v3 basic, 4x faster than v3.1)
- 6300+ nodes explored in 20K actions on ls20 (vs 5700 for v3 basic)

**Remaining issues:**
- ls20 level completion is stochastic (game requires key-lock matching, random exploration)
- ft09 state space is small (551 nodes) and fully explored but goal not found
- vc33 gets stuck after level 1 (level 2 requires more sophisticated click patterns)
- Model entity roles fail on real games (compensated by empirical fallback)

---

## v3.1 Results (Three-Layer Agent)

### Three-Layer Agent (state graph + hand-coded stats + Qwen2.5-7B LLM)
| Game | Actions | Levels | Speed | Notes |
|------|---------|--------|-------|-------|
| **vc33** | 5K | **1** | 4.2ms/act | LLM correctly identified "collection" game type |
| ls20 | — | — | — | Not yet tested |
| ft09 | — | — | — | Not yet tested |

**LLM oracle stats (vc33):** 25 calls, ~530ms/call, identified game_type=collection, plan=click_all_targets.
**Issue found:** LLM hallucinated movement_map for click-only game. Fixed with validation against available actions. Hand-coded Layer 2 works but limited to pre-defined patterns.

### Why pivot to learned understanding
- LLM hallucinated invalid strategies (movement_map for click-only game)
- Hand-coded Layer 2 can only detect pre-defined patterns (shifts, counters, region changes)
- Games with novel mechanics (rotation, conditional effects, gravity) won't be detected
- Winning competition entries used fast online learning, not LLMs
- Need: learned primitives that generalize across game types

---

## v3 Results (All Approaches Tested)

### v3 Basic Agent (state graph + CNN)
| Game | Actions | Levels | Speed | Notes |
|------|---------|--------|-------|-------|
| **vc33** | 5K | **1** | 16ms/act | **First level ever completed!** |
| ls20 | 20K | 0 | 6ms/act | 5700+ unique states, random walk |
| ft09 | 5K | 0 | 3ms/act | ~80 actions/state branching |

### v3 Dreamer Agent (Transformer world model + imagination)
| Game | Actions | Levels | Speed | Notes |
|------|---------|--------|-------|-------|
| vc33 | 5K | 0 | 6.6ms/act | Policy never converges — no reward signal |
| ls20 | 5K | 0 | 7.0ms/act | Random policy, no understanding |

---

## Previous Results

### v2 — World Model (Superseded)
| Metric | v2 Achieved |
|--------|-------------|
| VQ-VAE pixel accuracy | **99.85%** |
| World model frame prediction | **85.3%** |
| World model action type acc | **61.3%** |
| Policy type accuracy | **73.5%** |
| Policy location accuracy | **40.7%** |
| **Agent level completion** | **0 levels (all games)** |

### v1 — Behavioral Cloning / PPO (Failed)
- BC: 80% acc, 0% levels
- PPO: 0.18% success

---

## Code Status

### v4 — Online P(frame_change) CNN (Active Development)
| Component | File | Status |
|-----------|------|--------|
| CNN Model | `src/v4/model.py` | Done (1.1M params, 0.28ms forward) |
| Agent | `src/v4/agent.py` | Done (CNN-guided sampling, 2.6-3.5ms/act) |
| Experience Buffer | `src/v4/agent.py` | Done (hash-deduped, 200K max) |

### v3.2 — Learned Understanding (Concluded)
| Component | File | Status |
|-----------|------|--------|
| Synthetic Game Framework | `src/aria_v3/synthetic_games/` | Done (3 archetypes, 3600 sequences) |
| Data Generation Pipeline | `src/aria_v3/synthetic_games/generate.py` | Done (578K transitions) |
| Transition Encoder CNN | `src/aria_v3/understanding/encoder.py` | Done (578K params) |
| Temporal Transformer | `src/aria_v3/understanding/temporal.py` | Done (3.2M params) |
| Understanding Decoder | `src/aria_v3/understanding/decoder.py` | Done (shift classification, 85K params) |
| Full Model + Dataset | `src/aria_v3/understanding/model.py`, `dataset.py` | Done |
| Training Script | `src/aria_v3/understanding/train.py` | Done (30 epochs, 19 min) |
| TTT Engine | `src/aria_v3/understanding/ttt.py` | Done (LoRA, runs on real games) |
| Pretrained Checkpoint | `checkpoints/understanding/best.pt` | Done (100% game type, 0.13 shift MAE) |
| v3.2 Agent | `src/aria_v3/understanding_agent.py` | Done (1.0-1.2ms/act, unified scoring + empirical fallbacks) |

### v3.1 — Three-Layer Agent (Foundation)
| Component | File | Status |
|-----------|------|--------|
| Layer 1: Reactive | `src/aria_v3/three_layer_agent.py` | Done, 1 level vc33 |
| Layer 2: Learning | `src/aria_v3/learning_engine.py` | Done (hand-coded, to be replaced) |
| Layer 3: Reasoning | `src/aria_v3/reasoning_oracle.py` | Done (LLM, to be replaced) |

### v3 — Online Learning (Foundation)
| Component | File | Status |
|-----------|------|--------|
| Frame Processor | `src/aria_v3/frame_processor.py` | Done |
| State Graph | `src/aria_v3/state_graph.py` | Done |
| Change Predictor CNN | `src/aria_v3/change_predictor.py` | Done |
| Basic Agent | `src/aria_v3/agent.py` | Done, 1 level on vc33 |
| Dreamer World Model | `src/aria_v3/world_model.py` | Done, 0 levels |
| Dreamer Agent | `src/aria_v3/dreamer_agent.py` | Done, 0 levels |

### v2 — World Model (Superseded)
| Component | File | Status |
|-----------|------|--------|
| VQ-VAE | `src/aria_v2/tokenizer/frame_tokenizer.py` | Done (99.85% acc) |
| World Model + Policy | `src/aria_v2/world_model/` | Complete, 0 levels |

---

## v3.2 Training Results (30 epochs on 3600 synthetic sequences)

| Metric | Value |
|--------|-------|
| Game type accuracy (val) | **100%** |
| Shift MAE (val) | **0.128** pixels |
| Shift classification CE (dx/dy) | 0.013 / 0.015 |
| Frame prediction loss | 0.87 |
| Entity role BCE | 0.19 |
| Training time | ~19 min (RTX 4090) |
| Total params | ~4.3M (578K encoder + 3.2M temporal + 85K decoder + 392K frame predictor) |

**Key architectural insight:** Shift prediction must use CLASSIFICATION (7 bins: {-16,-8,-4,0,+4,+8,+16}) not regression. MSE regression fails because mean shift across randomized action mappings is (0,0), driving predictions to zero. CE classification solved this completely (MAE dropped from 1.5 → 0.13).

---

## Recent Completions

- **[2026-02-10] v4 Agent First Results**: vc33=2 levels (best ever), ls20=0, ft09=0. 2.6-3.5ms/action. CNN-guided action selection.
- **[2026-02-10] v4 Agent Implemented**: Online P(frame_change) CNN, StochasticGoose-inspired. No pretraining, no state graph for action selection.
- **[2026-02-10] v3.2 Post-Mortem**: Documented all v3/v3.1/v3.2 findings. Root cause: visual generalization (CNN) beats frame hashing. See [V3 Post-Mortem](findings/V3-POSTMORTEM.md).
- **[2026-02-10] StochasticGoose Gap Analysis**: 18/20 vs 1/20 levels. Online CNN P(frame_change) with visual generalization is the winning approach. No pretraining, no entity detection, no LLMs.
- **[2026-02-10] v3.2 Exploration Fixes**: Empirical change tracking, committed frontier navigation, empirical entity/movement detection. ls20 first ever level (stochastic). Movement map correct within 100 steps.
- **[2026-02-10] v3.2 Agent Tested on All 3 Games**: vc33=1 level, ls20=0-1 levels (stochastic), ft09=0 levels. 1.0-1.2ms/action.
- **[2026-02-10] v3.2 Understanding Agent Built**: Unified scoring (replaces 6-mode cascade), TTT integration, 0.66ms/step on realistic frames. Correctly identifies game type from observation.
- **[2026-02-10] Understanding Model Trained**: 100% game type acc, 0.13 shift MAE. CNN encoder + temporal transformer + shift classification decoder.
- **[2026-02-10] Synthetic Game Framework**: 3 archetypes (navigation, click, collection), 3600 sequences, 578K transitions with augmentations.
- **[2026-02-10] Three-Layer Agent Tested on vc33**: 1 level, 4.2ms/act. LLM hallucinated invalid strategy (fixed with validation).
- **[2026-02-10] Three-Layer Agent Built**: Reactive + Learning + Reasoning oracle integrated.
- **[2026-02-10] Three-Layer Architecture Design**: Reactive + Learning + Reasoning. LLM oracle for interpretation.
- **[2026-02-10] Dreamer Agent Tested**: 0 levels on vc33/ls20. No reward signal.
- **[2026-02-09] v3 Basic Agent First Run**: 1 level on vc33 (first ever!). 3-16ms/action.
- **[2026-02-09] Competition Rules Discovery**: No demos at test time. 150+ unknown games.
- **[2026-02-06] VQ-VAE + Trajectory Pipeline**: 99.85% pixel accuracy.

---

## What's Next

### v3.2 (Concluded)
1. ~~Synthetic game framework~~ Done (3 archetypes, 3600 sequences)
2. ~~Learned understanding model~~ Done (4.3M params, trained 30 epochs)
3. ~~Pretraining pipeline~~ Done (100% game type, 0.13 shift MAE)
4. ~~TTT integration~~ Done (LoRA adapts during play, 496+ updates per 5K actions)
5. ~~v3.2 Agent~~ Done (1.0-1.2ms/act, unified scoring + empirical fallbacks)
6. ~~Evaluation on 3 games~~ Done (vc33=1 level, ls20=0-1, ft09=0)
7. ~~Post-mortem~~ Done. See [V3 Post-Mortem](findings/V3-POSTMORTEM.md)

### v4 (Active)
1. ~~Design v4 architecture~~ Done. See [V4 Architecture](current/V4-ARCHITECTURE.md)
2. ~~Implement v4 agent~~ Done. `src/v4/agent.py`, `src/v4/model.py`
3. ~~First test on 3 games~~ Done. vc33=2 levels, ls20=0, ft09=0
4. **Optimize for more levels** — ls20 navigation, ft09 game-over rate, speed
5. **Competition submission** — ARC Prize 2026 (March 25, 2026)

---

## Links
- [Competition Rules](reference/COMPETITION-RULES.md) - **READ FIRST** — ground truth
- [V4 Architecture](current/V4-ARCHITECTURE.md) - v4 online P(frame_change) CNN (active)
- [Implementation Plan](current/IMPLEMENTATION-PLAN.md) - v3.2 learned understanding plan (concluded)
- [Game Mechanics](reference/GAME-MECHANICS.md) - ls20, vc33, ft09 analysis
- [V3 Post-Mortem](findings/V3-POSTMORTEM.md) - Why v3/v3.1/v3.2 failed, StochasticGoose comparison
- [ARIA v1 Report](findings/ARIA-V1-REPORT.md) - Why BC/PPO failed
- [V1 World Model Analysis](findings/V1-WORLD-MODEL-ANALYSIS.md) - World model structural flaws
