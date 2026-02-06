# Learned World Model Architecture

## Overview

The agent learns game dynamics, goals, and strategies from human demonstration trajectories via next-token prediction. A VQ-VAE tokenizes visual frames into discrete codes, and a LoRA-adapted SmolLM2-360M processes these as token sequences.

```
Frame (64x64, 16 colors)
  → VQ-VAE encoder → 64 discrete tokens (8x8 grid, 512-code codebook)
  → Trajectory: [GAME_START] [FRAME] v1..v64 [ACT] a [FRAME] v1..v64 [ACT] a ... [LEVEL_COMPLETE]
  → SmolLM2-360M (LoRA) → next-token prediction
  → Surprise (NLL) + Goal inference (P(LEVEL_COMPLETE)) + Learned policy
```

| Spec | Value |
|------|-------|
| **Approach** | VQ-VAE tokenization + autoregressive transformer |
| **Base model** | SmolLM2-360M (362M params, 960 hidden, 32 layers) |
| **Trainable params** | ~51M (3.9M LoRA + 47M embed/lm_head) |
| **Training data** | 28 human demos, 3 games, 876K tokens |
| **Precision** | bfloat16 (model) + float32 (loss) |
| **VRAM** | ~10GB during training |

---

## Component 1: VQ-VAE Frame Tokenizer

**File:** `src/aria_v2/tokenizer/frame_tokenizer.py`

Converts 64x64 16-color game frames into 64 discrete tokens.

```
Input: [B, 64, 64] int tensor (0-15)
  → nn.Embedding(16, 32)                              → [B, 64, 64, 32]
  → Encoder (4 conv layers, stride 2/2/2/1)           → [B, 128, 8, 8]
  → VectorQuantizer (512 codes, 128-dim, EMA updates) → [B, 8, 8] indices
  → Decoder (mirror of encoder)                        → [B, 16, 64, 64] logits
```

**Key details:**
- 512-code codebook with 128-dim vectors
- EMA codebook updates (decay=0.99) + dead code reset
- Loss: cross-entropy reconstruction + commitment loss (beta=0.25)
- ~1.3M parameters, trains in <2 minutes on 13K frames

**Results:** 99.85% pixel reconstruction accuracy, 44.73% codebook utilization.

---

## Component 2: Trajectory Dataset

**File:** `src/aria_v2/tokenizer/trajectory_dataset.py`

Converts JSONL human demonstrations into tokenized sequences.

### Token Vocabulary (523 new tokens added to SmolLM2's 49,152)

| Tokens | Count | IDs |
|--------|-------|-----|
| VQ codes (`<VQ_000>`..`<VQ_511>`) | 512 | 49152-49663 |
| Actions (`<ACT_0>`..`<ACT_6>`) | 7 | 49664-49670 |
| `<FRAME>` marker | 1 | 49671 |
| `<ACT>` marker | 1 | 49672 |
| `<LEVEL_COMPLETE>` | 1 | 49673 |
| `<GAME_START>` | 1 | 49674 |

### Sequence Format

```
<GAME_START> <FRAME> v1..v64 <ACT> a <FRAME> v1..v64 <ACT> a ... <LEVEL_COMPLETE> ...
```

Each step = 67 tokens (1 FRAME + 64 VQ + 1 ACT + 1 action). Context window of 2048 fits ~30 frames.

### Windowing
- Sliding window: size=2048, stride=670 (~10 frames)
- Windows containing `<LEVEL_COMPLETE>` oversampled 3x
- Total: ~1,987 training windows from 28 trajectories

### Level Transitions
Detected from JSONL ground truth: when `data.score[t+1] > data.score[t]`, insert `<LEVEL_COMPLETE>`. No heuristic needed.

---

## Component 3: SmolLM2 + LoRA Game Transformer

**File:** `src/aria_v2/world_model/game_transformer.py`

### Setup
1. Load SmolLM2-360M in **bfloat16** (fp16 causes NaN with 49K vocab)
2. Extend vocabulary by 523 tokens → 49,675 total
3. Initialize new embeddings from N(mean, std) of existing
4. Optionally project VQ-VAE codebook vectors to 960-dim for VQ token init
5. Apply LoRA rank=16 on Q, K, V, O projections
6. Mark embed_tokens + lm_head as trainable

### Architecture
```
SmolLM2-360M (LlamaForCausalLM):
  vocab_size: 49675 (49152 base + 523 game tokens)
  hidden_size: 960
  num_layers: 32
  num_attention_heads: 15
  intermediate_size: 2560
  max_position_embeddings: 8192

LoRA:
  rank: 16, alpha: 32, dropout: 0.05
  targets: q_proj, k_proj, v_proj, o_proj
  trainable: ~3.9M LoRA params
  + ~47M embed/lm_head params (for new tokens)
```

---

## Component 4: Training Pipeline

**File:** `src/aria_v2/world_model/train.py`

### Objective
Next-token prediction with per-token-type loss weights:
- VQ frame tokens: weight 1.0
- Action tokens: weight 2.0
- `<LEVEL_COMPLETE>`: weight 5.0
- Structural markers: weight 0.0

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| Batch size | 4 (effective 16 with grad accumulation) |
| Learning rate | 2e-4, cosine decay |
| Warmup | 10% of total steps |
| Epochs | 30 |
| Gradient checkpointing | Enabled |
| Mixed precision | bfloat16 autocast |
| Loss computation | float32 (outside autocast) |

### Critical Implementation Notes
- **Must load model in bfloat16**, NOT fp16. fp16 has limited dynamic range causing NaN with 49K+ vocabulary cross-entropy.
- **Compute CE loss in float32 outside `torch.amp.autocast`** to prevent overflow.
- Gradient clipping: max_norm=1.0

### Results (30 epochs, 77 minutes)
| Metric | Target | Achieved |
|--------|--------|----------|
| Frame prediction accuracy | >40% | **88.4%** |
| Action prediction accuracy | >30% | **67.9%** |
| Perplexity | <20 | **1.8** |
| Level completion prediction | N/A | **99.5%** |
| Best validation loss | N/A | **0.5721** |

---

## Component 5: Inference Agent

**File:** `src/aria_v2/world_model/agent.py`

Three capabilities from one model:

### 1. World Model (surprise measurement)
- Given context + action, predict next 64 VQ tokens
- Surprise = negative log-likelihood of actual vs predicted tokens
- Tracked against running EMA (no absolute thresholds)

### 2. Goal Inference
- For each candidate action: extend context, predict next frame, check P(LEVEL_COMPLETE)
- Rank actions by probability of leading to level completion

### 3. Action Selection
```
surprise = model_nll(actual_frame) vs surprise_ema
goal_scores = [P(LEVEL_COMPLETE | action=a) for a in actions]
policy_probs = model_action_distribution(context)

if max(goal_scores) > 2 * mean(goal_scores):   # one action clearly better
    action = argmax(goal_scores)                 # goal-directed
elif surprise > 2 * surprise_ema:                # something unexpected
    action = argmax(prediction_entropy)          # explore
else:
    action = sample(policy_probs)                # follow learned policy
```

### Integration
Uses arcengine for game execution, same interface as `run_game.py`.

---

## Why This Over the Previous Approaches

### vs. ARIA v1 (BC/PPO)
- BC learned action distributions, not game logic → 80% accuracy, 0% levels
- PPO couldn't discover goals with sparse reward → 0.18% success
- World model learns dynamics AND goals from the same next-token objective

### vs. Language-Guided Meta-Learning
- Visual grounding + event detection + LLM reasoning pipeline was too brittle
- Full of hand-coded heuristics (>12% = background, >30% = transition, step_size=5)
- Rule templates (MOVE, BLOCK, COUNTER) won't generalize to unknown games
- Learned model replaces ALL heuristics with prediction error

### vs. Heuristic Abstract Learning
- Abstract learner, goal induction, demonstration learner work on ls20
- But require game-specific thresholds and pattern matching
- World model learns its own vocabulary for describing game states

---

*Architecture version: 3.0*
*Approach: Learned World Model (VQ-VAE + SmolLM2-360M + LoRA)*
