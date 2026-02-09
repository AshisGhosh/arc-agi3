# Project Progress

## Current State
**Phase:** Game-Agnostic World Model + Separated Policy (v2 redesign)
**Branch:** main
**Status:** Code complete, needs retraining (world model + policy heads)

## Immediate Next Step
**Retrain world model with unified action tokenization:**
1. Delete old cache: `rm checkpoints/world_model/cache/trajectories.pt`
2. Retrain: `uv run python -m src.aria_v2.world_model.train --epochs 30`
3. Validate: `uv run python -m src.aria_v2.world_model.evaluate_world_model --mode all-games`
4. Train policy: `uv run python -m src.aria_v2.world_model.train_policy --epochs 50`
5. Run agent: `uv run python -m src.aria_v2.world_model.agent --game ls20`

---

## Architecture (v2)

```
Frame (64x64, 16 colors)
  → VQ-VAE encoder → 64 discrete tokens (8x8 grid, 512-code codebook)
  → Unified action tokenization: [ACT] <ACT_TYPE_i> <ACT_LOC_j> (69 tokens/step)
  → SmolLM2-360M (LoRA) → dynamics prediction (world model)
  → Masked context → PolicyHeads → action type + location (separated policy)
```

**Key changes from v1:**
1. **Unified action = (type, location).** Click coordinates tokenized as VQ cell indices.
2. **Dynamics separated from policy.** World model predicts next state; policy reads masked context.
3. **Action masking prevents mode collapse.** Policy cannot copy action history.

See [Architecture Details](current/ARCHITECTURE.md).

---

## Training Results (v1 — will be superseded by v2 retraining)

| Metric | Target | v1 Achieved | v2 Status |
|--------|--------|-------------|-----------|
| VQ-VAE pixel accuracy | >95% | **99.85%** | Unchanged (kept) |
| VQ-VAE codebook utilization | >50% | **44.73%** | Unchanged |
| World model frame prediction | >40% | 88.4% (v1) | Retraining needed |
| World model action type acc | >50% | N/A | New metric |
| World model action loc acc | >40% | N/A | New metric (click games) |
| World model perplexity | <20 | 1.8 (v1) | Retraining needed |
| Policy type accuracy | >50% | N/A | New (separate head) |
| Policy location accuracy | >40% | N/A | New (separate head) |
| Agent level completion | >0 levels | Not tested | Pending |

---

## Code Status

### Active Components (v2 Redesign)
| Component | File | Status |
|-----------|------|--------|
| VQ-VAE Frame Tokenizer | `src/aria_v2/tokenizer/frame_tokenizer.py` | Done (99.85% acc, unchanged) |
| VQ-VAE Training | `src/aria_v2/tokenizer/train_vqvae.py` | Done (unchanged) |
| Trajectory Dataset (v2) | `src/aria_v2/tokenizer/trajectory_dataset.py` | **Updated**: unified (type, loc) actions |
| World Model Config (v2) | `src/aria_v2/world_model/config.py` | **Updated**: new vocab, loss weights, PolicyConfig |
| SmolLM2 + LoRA (v2) | `src/aria_v2/world_model/game_transformer.py` | **Updated**: 589 tokens, location embedding init |
| Training Pipeline (v2) | `src/aria_v2/world_model/train.py` | **Updated**: type/loc metrics, v2 cache |
| Policy Heads | `src/aria_v2/world_model/policy_head.py` | **New**: ActionTypeHead + ActionLocationHead |
| Policy Training | `src/aria_v2/world_model/train_policy.py` | **New**: frozen backbone, masked training |
| Inference Agent (v2) | `src/aria_v2/world_model/agent.py` | **Rewritten**: masked policy, spatial actions |
| Evaluation (v2) | `src/aria_v2/world_model/evaluate_world_model.py` | **Updated**: type + loc accuracy |

### Earlier Components (Exploratory, not on critical path)
| Component | File | Notes |
|-----------|------|-------|
| Visual Grounding | `src/aria_v2/visual_grounding.py` | 100% accuracy on synthetic games |
| Synthetic Games | `src/aria_v2/pretraining/synthetic_games.py` | Training data generator |
| Abstract Learner | `src/aria_v2/core/abstract_learner.py` | Heuristic rule learning |
| Goal Induction | `src/aria_v2/core/goal_induction.py` | Hypothesis testing |
| Demonstration Learner | `src/aria_v2/core/demonstration_learner.py` | JSONL demo analysis |
| Heuristic Agent | `src/aria_v2/core/agent.py` | Older agent loop |
| Run Game | `src/aria_v2/run_game.py` | Game runner (arcengine) |

### Checkpoints
| Checkpoint | Path | Size | Status |
|------------|------|------|--------|
| VQ-VAE | `checkpoints/vqvae/best.pt` | ~2MB | Current |
| World Model (v1) | `checkpoints/world_model/best.pt` | 704MB | Outdated — needs retraining |
| Trajectory Cache (v1) | `checkpoints/world_model/cache/trajectories.pt` | ~3.5MB | Outdated — delete before retraining |
| Policy | `checkpoints/policy/best.pt` | TBD | Not yet trained |

---

## Recent Completions

- **[2026-02-09] v2 Redesign Implementation**: Complete rewrite of action tokenization, world model training, and agent architecture. Unified (type, location) action representation. Separated dynamics from policy with architectural action masking. New files: `policy_head.py`, `train_policy.py`. Updated all existing files for new token format.
- **[2026-02-08] Strategic Reasoning Generation (ft09)**: Generated 57 frame-by-frame reasoning entries for batch_0004.json.
- **[2026-02-06] World Model Training (v1)**: SmolLM2-360M + LoRA trained 30 epochs in 77min. Frame acc 88.4%, action acc 67.9%.
- **[2026-02-06] VQ-VAE + Trajectory Pipeline**: 99.85% pixel accuracy, 876K tokens from 28 demos.
- **[2026-02-05] Heuristic Approach**: Abstract learning, goal induction working on ls20. But brittle and won't generalize.
- **[2026-02-05] Visual Grounding**: 100% detection + classification on synthetic games.
- **[2026-02-04] ARIA v1**: BC (80% acc, 0% levels), PPO (0.18% success). Confirmed: puzzle games need understanding, not imitation.

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture shift v1 → v2 | VQ-VAE + transformer | BC/PPO failed (0% levels); need game understanding |
| Base model | SmolLM2-360M | Fits in 24GB VRAM with LoRA, good token prediction |
| Precision | bfloat16 | fp16 causes NaN with 49K vocab CE loss; bf16 has better range |
| LoRA rank | 16 on Q,K,V,O | 51M trainable params, sufficient for our data scale |
| Loss outside autocast | float32 CE | Prevents overflow in large-vocab softmax |
| **v2: Unified actions** | **(type, location) pairs** | **Enables click game dynamics; game-agnostic** |
| **v2: Separated policy** | **Masked context + heads** | **Prevents mode collapse; 500K trainable params** |
| **v2: Action masking** | **MASK token replaces action history** | **Policy can only learn from visual consequences** |

---

## What's Next

1. **Retrain World Model** — With new unified action tokenization (~80 min)
2. **Validate World Model** — Check frame prediction on all 3 games, especially click games
3. **Train Policy Heads** — On frozen backbone with masked context (~20 min)
4. **Agent Evaluation** — Run on ls20, vc33, ft09, measure levels completed
5. **Competition** — Submit to ARC-AGI-3

---

## Links
- [Design Document v2](current/DESIGN-DOC-V2.md) - Standalone architecture + data plan (start here)
- [Architecture](current/ARCHITECTURE.md) - v2 component-level architecture reference
- [Implementation Plan](current/IMPLEMENTATION-PLAN.md) - Phase-by-phase execution plan
- [v1 World Model Analysis](findings/V1-WORLD-MODEL-ANALYSIS.md) - Why v1 had structural flaws
- [Game Mechanics](reference/GAME-MECHANICS.md) - ls20, vc33, ft09 analysis
- [ARIA v1 Report](findings/ARIA-V1-REPORT.md) - Why BC/PPO failed
