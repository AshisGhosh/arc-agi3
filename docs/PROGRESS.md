# Project Progress

## Current State
**Phase:** v3.2 — Learned Game Understanding (Synthetic Pretraining + TTT)
**Branch:** main
**Status:** Three-layer agent tested. Pivoting to learned understanding model with synthetic game pretraining.

## Immediate Next Step
**Build synthetic game engine + learned understanding model:**
1. Synthetic game framework (navigation, click, collection, mixed)
2. CNN transition encoder + temporal transformer + decoder heads (~3M params)
3. Pretraining on synthetic data with TTT (test-time training)
4. Integration with Layer 1 execution (state graph + pathfinding)

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

### v3.2 — Learned Understanding (Active Development)
| Component | File | Status |
|-----------|------|--------|
| Synthetic Game Framework | `src/aria_v3/synthetic_games/` | Planned |
| Transition Encoder CNN | `src/aria_v3/understanding/encoder.py` | Planned |
| Temporal Transformer | `src/aria_v3/understanding/temporal.py` | Planned |
| Understanding Decoder | `src/aria_v3/understanding/decoder.py` | Planned |
| TTT Loop | `src/aria_v3/understanding/ttt.py` | Planned |
| Data Generation Pipeline | `src/aria_v3/synthetic_games/generate.py` | Planned |

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

## Recent Completions

- **[2026-02-10] Three-Layer Agent Tested on vc33**: 1 level, 4.2ms/act. LLM hallucinated invalid strategy (fixed with validation).
- **[2026-02-10] Three-Layer Agent Built**: Reactive + Learning + Reasoning oracle integrated.
- **[2026-02-10] Three-Layer Architecture Design**: Reactive + Learning + Reasoning. LLM oracle for interpretation.
- **[2026-02-10] Dreamer Agent Tested**: 0 levels on vc33/ls20. No reward signal.
- **[2026-02-09] v3 Basic Agent First Run**: 1 level on vc33 (first ever!). 3-16ms/action.
- **[2026-02-09] Competition Rules Discovery**: No demos at test time. 150+ unknown games.
- **[2026-02-06] VQ-VAE + Trajectory Pipeline**: 99.85% pixel accuracy.

---

## What's Next

1. **Synthetic game framework** — navigation, click, collection, mixed, push, conditional
2. **Learned understanding model** — CNN encoder + temporal transformer + decoder heads
3. **Pretraining pipeline** — data generation, augmentation (color/action/spatial permutations)
4. **TTT integration** — LoRA adaptation during play
5. **Evaluation** — all 3 public games + synthetic holdout
6. **Competition submission** — ARC Prize 2026 (March 25, 2026)

---

## Links
- [Competition Rules](reference/COMPETITION-RULES.md) - **READ FIRST** — ground truth
- [Implementation Plan](current/IMPLEMENTATION-PLAN.md) - v3.2 learned understanding plan
- [Game Mechanics](reference/GAME-MECHANICS.md) - ls20, vc33, ft09 analysis
- [ARIA v1 Report](findings/ARIA-V1-REPORT.md) - Why BC/PPO failed
