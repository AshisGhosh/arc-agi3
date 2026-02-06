# Project Progress

## Current State
**Phase:** Learned World Model (VQ-VAE + SmolLM2-360M + LoRA)
**Branch:** main
**Status:** Training complete, ready for agent evaluation

## Immediate Next Step
**Run world model agent on real games and evaluate:**
- Run `uv run python -m src.aria_v2.world_model.agent --game ls20`
- Measure level completion rate vs heuristic baseline (0 levels) and human demos (11/12)
- Evaluate surprise-based exploration and goal inference behavior
- Profile inference speed (target: <100ms per action)

---

## Architecture

```
Frame (64x64, 16 colors)
  → VQ-VAE encoder → 64 discrete tokens (8x8 grid, 512-code codebook)
  → SmolLM2-360M (LoRA) → next-token prediction
  → World model (surprise) + Goal inference (P(LEVEL_COMPLETE)) + Action selection
```

**Why this approach:** Prediction error replaces all heuristics. "Background" = what the model predicts correctly. "Level transition" = what the model fails to predict. "Goal" = what precedes LEVEL_COMPLETE tokens.

See [Architecture Details](current/ARCHITECTURE.md).

---

## Training Results

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| VQ-VAE pixel accuracy | >95% | **99.85%** | Exceeded |
| VQ-VAE codebook utilization | >50% | **44.73%** | Close |
| World model frame prediction | >40% | **88.4%** | 2.2x target |
| World model action prediction | >30% | **67.9%** | 2.3x target |
| World model perplexity | <20 | **1.8** | 11x target |
| Level completion prediction | N/A | **99.5%** | Excellent |
| Best validation loss | N/A | **0.5721** | Converged |
| Total training time | N/A | **79 min** | VQ-VAE 1.9min + WM 77min |

---

## Code Status

### Active Components (Learned World Model)
| Component | File | Status |
|-----------|------|--------|
| VQ-VAE Frame Tokenizer | `src/aria_v2/tokenizer/frame_tokenizer.py` | Done (99.85% acc) |
| VQ-VAE Training | `src/aria_v2/tokenizer/train_vqvae.py` | Done |
| Trajectory Dataset | `src/aria_v2/tokenizer/trajectory_dataset.py` | Done (876K tokens, 28 trajectories) |
| World Model Config | `src/aria_v2/world_model/config.py` | Done |
| SmolLM2 + LoRA | `src/aria_v2/world_model/game_transformer.py` | Done |
| Training Pipeline | `src/aria_v2/world_model/train.py` | Done |
| Inference Agent | `src/aria_v2/world_model/agent.py` | Done |

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
| Checkpoint | Path | Size |
|------------|------|------|
| VQ-VAE | `checkpoints/vqvae/best.pt` | ~2MB |
| World Model | `checkpoints/world_model/best.pt` | 704MB |
| Trajectory Cache | `checkpoints/world_model/cache/trajectories.pt` | ~3.5MB |

---

## Recent Completions

- **[2026-02-06] World Model Training**: SmolLM2-360M + LoRA trained 30 epochs in 77min. Frame acc 88.4%, action acc 67.9%, level pred 99.5%, perplexity 1.8. Fixed NaN loss by switching fp16 to bfloat16.
- **[2026-02-06] VQ-VAE + Trajectory Pipeline**: 99.85% pixel accuracy, 876K tokens from 28 demos across 3 games (ls20, vc33, ft09). VQ-VAE trained in 1.9min.
- **[2026-02-05] Heuristic Approach**: Abstract learning, goal induction, demonstration learning all working on ls20. Color 9 = goal (95% confidence). But brittle and won't generalize.
- **[2026-02-05] Visual Grounding**: 100% detection + classification on synthetic games.
- **[2026-02-04] ARIA v1**: BC (80% acc, 0% levels), PPO (0.18% success). Confirmed: puzzle games need understanding, not imitation.
- **[2026-02-04] Human Demo Analysis**: 28 demos loaded, 12 for ls20 (11 won). Block-sliding puzzle requiring 29+ actions per level.

---

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture shift v1 → v2 | VQ-VAE + transformer | BC/PPO failed (0% levels); need game understanding |
| Base model | SmolLM2-360M | Fits in 24GB VRAM with LoRA, good token prediction |
| Precision | bfloat16 | fp16 causes NaN with 49K vocab CE loss; bf16 has better range |
| LoRA rank | 16 on Q,K,V,O | 51M trainable params, sufficient for our data scale |
| Loss outside autocast | float32 CE | Prevents overflow in large-vocab softmax |

---

## What's Next

1. **Agent Evaluation** - Run world model agent on ls20, measure levels completed
2. **Multi-game Testing** - Test on vc33 and ft09
3. **Iterate** - If agent struggles, consider: more training data, longer context, reward shaping
4. **Competition** - Submit to ARC-AGI-3 evaluation

---

## Links
- [Architecture](current/ARCHITECTURE.md) - VQ-VAE + SmolLM2 learned world model
- [Implementation Plan](current/IMPLEMENTATION-PLAN.md) - What was built and what's next
- [Game Mechanics](reference/GAME-MECHANICS.md) - ls20, vc33, ft09 analysis
- [ARIA v1 Report](findings/ARIA-V1-REPORT.md) - Why BC/PPO failed
