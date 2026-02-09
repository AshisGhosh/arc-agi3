# Design Document: Game-Agnostic World Model v2

**Date:** 2026-02-09
**Status:** Code complete, pending training
**Predecessor:** [v1 World Model Analysis](../findings/V1-WORLD-MODEL-ANALYSIS.md)

---

## 1. Problem Statement

ARC-AGI-3 presents unknown interactive puzzle games. The agent must:
- Discover game mechanics from scratch (no game-specific code)
- Handle diverse action spaces: directional navigation (ls20), clicking with coordinates (vc33, ft09), and unknown future types
- Complete multiple levels per game with sparse rewards (only on level completion)

The v1 world model achieved 88.4% frame prediction accuracy but had three structural flaws that prevent game-agnostic behavior:
1. Click coordinates were discarded during tokenization
2. Action prediction mode-collapsed (copied history instead of reading visual state)
3. Policy was entangled with dynamics prediction

This document specifies the v2 redesign that fixes all three.

---

## 2. System Overview

```
                    64x64 frame (16 colors)
                           |
                    VQ-VAE Encoder
                           |
                  64 discrete tokens (8x8 grid)
                           |
              +-------------+-------------+
              |                           |
         World Model                  Policy
    (dynamics prediction)        (action selection)
              |                           |
    SmolLM2-360M + LoRA          Frozen backbone
    Full context with actions    Masked context (no actions)
              |                           |
    "What happens next?"         "What should I do?"
                                          |
                                 ActionTypeHead + ActionLocationHead
                                          |
                                  (type, location) action
                                          |
                                    Game API call
```

| Spec | Value |
|------|-------|
| Base model | SmolLM2-360M (362M params, 960 hidden, 32 layers) |
| World model trainable | ~51M (3.9M LoRA + 47M embed/lm_head) |
| Policy trainable | ~500K (two small heads) |
| Precision | bfloat16 (model) + float32 (loss) |
| Training VRAM | ~10GB |
| Inference | ~60ms/step, ~1GB VRAM |
| Context window | 2048 tokens = ~29 game steps |

---

## 3. Design Principles

1. **No game-specific heuristics.** Zero hardcoded knowledge of ls20 mechanics, color meanings, or action semantics. Everything is learned from demonstrations.

2. **Unified action = (type, location).** A single format that covers navigation (type=UP, loc=NULL), clicking (type=CLICK, loc=cell_37), and anything an unknown game might use.

3. **Dynamics separate from policy.** The world model answers "what happens if I do X?" The policy answers "what should I do?" They share a backbone but are trained with different objectives and different inputs.

4. **Architectural mode collapse prevention.** The policy physically cannot see action tokens (replaced with MASK). It must learn from visual consequences of actions, not from copying action history.

---

## 4. Data Pipeline

### 4.1 Source Data

28 human demonstrations across 3 games, stored as JSONL recordings.

| Game | Demos | Frames | Action Types | Spatial Actions |
|------|-------|--------|-------------|-----------------|
| ls20 | 12 | ~8,700 | 1-4 (directional) | No (ACT_LOC_NULL) |
| vc33 | ~8 | ~2,200 | 6 (click) | Yes (51/64 cells covered) |
| ft09 | ~8 | ~2,100 | 6 (click) | Yes (61/64 cells covered) |
| **Total** | **28** | **~13,000** | | |

**Source format (JSONL):**
```json
{
  "type": "action",
  "data": {
    "frame": [[0,0,3,...], ...],
    "action_input": {
      "id": 6,
      "data": {"x": 50, "y": 44}
    }
  }
}
```

- `data.frame[0]` = 64x64 grid (16-color pixel values)
- `data.action_input.id` = action type (0-7)
- `data.action_input.data.x/y` = click coordinates (present for action 6, absent for navigation)

### 4.2 Frame Tokenization (VQ-VAE)

Unchanged from v1. Converts 64x64 pixel frames into 64 discrete tokens.

```
Input: [B, 64, 64] int tensor (0-15)
  → nn.Embedding(16, 32)                              → [B, 64, 64, 32]
  → Encoder (4 conv layers, stride 2/2/2/1)           → [B, 128, 8, 8]
  → VectorQuantizer (512 codes, 128-dim, EMA updates) → [B, 8, 8] indices
  → Decoder (mirror of encoder)                        → [B, 16, 64, 64] logits
```

**Results:** 99.85% pixel accuracy, 44.73% codebook utilization (229/512 codes active).

**Checkpoint:** `checkpoints/vqvae/best.pt` (~2MB)

### 4.3 Action Tokenization

Every action becomes **2 tokens**: `<ACT_TYPE_i> <ACT_LOC_j>`.

**Spatial mapping** — action locations use the same 8x8 grid as the VQ-VAE:
```
Pixel → VQ cell:    cell = (y // 8) * 8 + (x // 8)
VQ cell → pixel:    x = (cell % 8) * 8 + 4,  y = (cell // 8) * 8 + 4
```

This creates a unified spatial reference: VQ code at cell 46 describes what's visually at cell 46, and ACT_LOC_46 means "click at cell 46." The model can directly associate location tokens with frame tokens.

**Per-game examples:**

| Game | Raw Action | Tokens |
|------|-----------|--------|
| ls20 (navigate right) | id=2, no coords | `<ACT_TYPE_2> <ACT_LOC_NULL>` |
| vc33 (click at 50,44) | id=6, x=50 y=44 | `<ACT_TYPE_6> <ACT_LOC_46>` |
| ft09 (click at 40,46) | id=6, x=40 y=46 | `<ACT_TYPE_6> <ACT_LOC_45>` |
| Unknown game | id=3, x=20 y=8 | `<ACT_TYPE_3> <ACT_LOC_9>` |

### 4.4 Token Vocabulary

589 new tokens appended to SmolLM2's base vocabulary (49,152 tokens):

| Token Range | Type | Count | IDs |
|---|---|---|---|
| `<VQ_000>` .. `<VQ_511>` | Frame codes | 512 | 49152-49663 |
| `<ACT_TYPE_0>` .. `<ACT_TYPE_7>` | Action type | 8 | 49664-49671 |
| `<ACT_LOC_0>` .. `<ACT_LOC_64>` | Action location | 65 | 49672-49736 |
| `<FRAME>` | Frame marker | 1 | 49737 |
| `<ACT>` | Action marker | 1 | 49738 |
| `<LEVEL_COMPLETE>` | Level boundary | 1 | 49739 |
| `<GAME_START>` | Trajectory start | 1 | 49740 |
| `<MASK>` | Policy mask (not in world model) | 1 | 49741 |
| **Total** | | **589** (+1 MASK) | |

**World model vocab:** 49,741. **Policy vocab:** 49,742 (adds MASK).

### 4.5 Sequence Format

**69 tokens per game step** (was 67 in v1):
```
<GAME_START>
<FRAME> vq_0 vq_1 ... vq_63 <ACT> <ACT_TYPE_i> <ACT_LOC_j>
<FRAME> vq_0 vq_1 ... vq_63 <ACT> <ACT_TYPE_i> <ACT_LOC_j>
...
<LEVEL_COMPLETE>
<FRAME> vq_0 vq_1 ... vq_63 <ACT> <ACT_TYPE_i> <ACT_LOC_j>
...
```

**Breakdown per step:** 1 (FRAME) + 64 (VQ codes) + 1 (ACT) + 1 (ACT_TYPE) + 1 (ACT_LOC) + 1 (next FRAME) = 69 tokens between adjacent frame markers.

**Context window:** 2048 tokens / 69 tokens per step ≈ 29 frames per training window.

### 4.6 Training Windows

Sliding window with stride 690 tokens (~10 steps) over concatenated trajectories:
- Windows: ~2000 (from 28 trajectories, ~900K total tokens)
- Each window is 2048 tokens
- Labels are right-shifted (standard causal LM)

**Token type labels** per position enable per-type loss weighting:
- `"vq"` — frame tokens
- `"action_type"` — action type tokens
- `"action_loc"` — action location tokens
- `"level_complete"` — level boundary tokens
- `"structural"` — FRAME, ACT, GAME_START markers

---

## 5. World Model (Dynamics Prediction)

### 5.1 Architecture

SmolLM2-360M with LoRA adaptation:
```
LlamaForCausalLM:
  vocab_size: 49741
  hidden_size: 960
  num_layers: 32
  num_attention_heads: 15
  max_position_embeddings: 8192

LoRA:
  rank: 16, alpha: 32, dropout: 0.05
  targets: q_proj, k_proj, v_proj, o_proj
  trainable: ~3.9M params

embed_tokens + lm_head:
  manually unfrozen (requires_grad=True)
  trainable: ~47M params

Total trainable: ~51M params
```

### 5.2 Embedding Initialization

New token embeddings are initialized to match the pretrained distribution:

- **VQ codes (512):** Projected from VQ-VAE codebook vectors (128-dim → 960-dim via learned linear layer), then scaled to match mean/std of existing embeddings.
- **ACT_LOC (64 cells):** Initialized from the same VQ codebook vectors as corresponding VQ codes. This creates spatial grounding — ACT_LOC_46 starts with the same representation as VQ_046, so the model can immediately associate click locations with frame content.
- **ACT_LOC_NULL (1):** Initialized as mean of all codebook vectors.
- **ACT_TYPE (8):** Random initialization, scaled to match distribution.
- **Structural tokens:** Random initialization, scaled.

### 5.3 Training Objective

Standard next-token prediction (causal language modeling) with per-token-type loss weights:

| Token Type | Weight | Rationale |
|-----------|--------|-----------|
| VQ frame tokens | 1.0 | Baseline; most tokens in sequence |
| Action type | 3.0 | Critical for policy; underrepresented (1/69 tokens) |
| Action location | 3.0 | Critical for spatial understanding; underrepresented |
| LEVEL_COMPLETE | 5.0 | Rare but essential for goal inference |
| Structural markers | 0.0 | Deterministic; no learning signal |

**Why weighted loss:** Without upweighting, action tokens contribute ~3% of total gradient (2/69 tokens). The 3.0 weight ensures ~9% gradient contribution, enough for the model to invest capacity in action prediction.

### 5.4 Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 30 |
| Batch size | 4 |
| Gradient accumulation | 4 (effective batch 16) |
| Learning rate | 2e-4 (cosine decay) |
| Weight decay | 0.01 |
| Warmup | 100 steps |
| Context window | 2048 |
| Precision | bfloat16 (model), float32 (CE loss) |
| VRAM | ~10GB |
| Time | ~80 min on 4090 |

**Critical: CE loss in float32.** The 49K+ vocab softmax in bfloat16 overflows. Loss computation must be outside `torch.amp.autocast`.

### 5.5 Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Frame prediction accuracy | >80% | % of VQ tokens correctly predicted |
| Action type accuracy | >50% | % of action type tokens correctly predicted |
| Action location accuracy | >40% (click games) | % of location tokens correctly predicted |
| Perplexity | <20 | exp(mean cross-entropy loss) |
| LEVEL_COMPLETE prediction | >90% | Recall of level boundary tokens |

Metrics are tracked per-game and per-token-type.

---

## 6. Policy (Action Selection)

### 6.1 Design: Separate from World Model

The key insight: the world model and policy need different inputs.

- **World model** sees full context including action tokens — it needs to know what actions were taken to predict what happens next.
- **Policy** must NOT see action tokens — otherwise it can mode-collapse by copying recent action history rather than reading visual state.

Both share the SmolLM2 backbone (frozen for policy), but the policy sees **masked** context:

```
World model:  [FRAME] vq... [ACT] TYPE LOC [FRAME] vq... [ACT] TYPE LOC [FRAME] vq...
Policy:       [FRAME] vq... [MASK][MASK][MASK] [FRAME] vq... [MASK][MASK][MASK] [FRAME] vq...
```

Every `<ACT>`, `<ACT_TYPE_*>`, and `<ACT_LOC_*>` token is replaced with a learned `<MASK>` embedding (960-dim). The policy can only infer what actions were taken from their visual consequences in subsequent frames.

### 6.2 ActionTypeHead

```
Input: h_frame [960]  (hidden state at last VQ token position)
  → Linear(960, 256)
  → GELU
  → Dropout(0.1)
  → Linear(256, 8)
Output: logits [8]  (one per action type)
```

~248K parameters. Answers "what kind of action?" from the overall frame representation.

### 6.3 ActionLocationHead

Spatial attention mechanism:

```
Input: h_frame [960], h_vq_cells [64, 960]
  → Query: Linear(960, 128)(h_frame)          → [128]
  → Keys:  Linear(960, 128)(h_vq_cells)       → [64, 128]
  → NULL:  learnable parameter                  → [1, 128]
  → Keys = concat(cell_keys, null_key)          → [65, 128]
  → Logits = (Keys @ Query) / sqrt(128)         → [65]
Output: logits [65]  (64 VQ cells + 1 NULL)
```

~250K parameters. Answers "where?" by attending to spatial positions in the current frame. For navigation games, it learns to predict NULL. For click games, it attends to the relevant cell.

### 6.4 Policy Training

| Parameter | Value |
|-----------|-------|
| Backbone | Frozen (world model weights locked) |
| Trainable | PolicyHeads only (~500K params) |
| Loss | type_CE + has_location * loc_CE |
| Optimizer | AdamW, lr=1e-3, weight_decay=0.01 |
| Epochs | 50 |
| Time | ~20 min on 4090 |

`has_location` is 1.0 for click actions (type=6) and 0.0 for navigation (types 1-4). Location loss is only applied when the action has spatial coordinates.

### 6.5 Metrics

| Metric | Target |
|--------|--------|
| Type accuracy | >50% overall, per-game breakdown |
| Location accuracy | >40% for click games (vc33, ft09) |
| Visual dependence | Predictions change with different visual inputs |

---

## 7. Inference Agent

### 7.1 Loop

```python
def act(frame, available_actions):
    # 1. Encode frame
    vq_codes = vqvae.encode(frame)        # [64]

    # 2. Append to world model context
    wm_context.append(FRAME_TOKEN, *vq_codes)

    # 3. Create masked context (replace action tokens with MASK)
    policy_context = mask_actions(wm_context)

    # 4. Forward pass on masked context (backbone frozen)
    hidden_states = backbone(policy_context)

    # 5. Extract hidden states
    h_frame = hidden_states[last_vq_position]       # [960]
    h_vq_cells = hidden_states[vq_positions]         # [64, 960]

    # 6. Predict action
    type_logits = action_type_head(h_frame)           # [8]
    type_logits = mask_unavailable(type_logits, available_actions)
    action_type = sample(softmax(type_logits))

    loc_logits = action_location_head(h_frame, h_vq_cells)  # [65]
    action_loc = sample(softmax(loc_logits))

    # 7. Convert to game API
    if action_loc != NULL:
        x = (action_loc % 8) * 8 + 4
        y = (action_loc // 8) * 8 + 4
        env.step(action_type, data={"x": x, "y": y})
    else:
        env.step(action_type)

    # 8. Record action in world model context
    wm_context.append(ACT_TOKEN, ACT_TYPE + action_type, ACT_LOC + action_loc)
```

### 7.2 Performance

- **Two backbone passes** per step: one for world model context maintenance, one for policy with masked input. ~30ms each on 4090.
- **Total latency:** ~60ms/step (well under 100ms target).
- **VRAM:** ~1GB inference (KV cache + model weights).
- **Context management:** Sliding window of 29 frames. Oldest frames are dropped when context fills.

---

## 8. Handling Unknown Games

The architecture is designed for zero-shot transfer to unseen games:

| Aspect | How It Generalizes |
|--------|-------------------|
| Frame encoding | VQ-VAE sees raw pixels; works on any 64x64 16-color frame |
| Action types | 8 slots (0-7) cover all known ARC-AGI-3 actions; expandable |
| Action locations | 64 VQ cells cover the full 8x8 spatial grid; NULL for non-spatial |
| World model | Learns dynamics from demonstration context; no game-specific weights |
| Policy | Reads visual state; no assumption about what actions "mean" |
| Level detection | LEVEL_COMPLETE predicted from score increases in context |

**For a completely new game:** Provide a few human demonstrations, retokenize with the same pipeline, and the model learns from context. No architecture or code changes needed.

---

## 9. Training and Evaluation Pipeline

### Step 1: Delete v1 cache
```bash
rm checkpoints/world_model/cache/trajectories.pt
```

### Step 2: Retrain world model (~80 min)
```bash
uv run python -m src.aria_v2.world_model.train --epochs 30
```
**Gate:** frame_acc >80%, action_type_acc >50%, action_loc_acc >40% (click games)

### Step 3: Validate world model
```bash
uv run python -m src.aria_v2.world_model.evaluate_world_model --mode all-games
```
Check: click-game frame prediction improves with spatial context.

### Step 4: Train policy heads (~20 min)
```bash
uv run python -m src.aria_v2.world_model.train_policy --epochs 50
```
**Gate:** type_acc >50%, loc_acc >40% (click games)

### Step 5: Agent evaluation
```bash
uv run python -m src.aria_v2.world_model.agent --game ls20
uv run python -m src.aria_v2.world_model.agent --game vc33
uv run python -m src.aria_v2.world_model.agent --game ft09
```
**Success:** Any level completion (baseline: 0 levels).

---

## 10. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| 8x8 spatial resolution too coarse for clicks | Low | Medium | 80-95% cell coverage in data; add sub-cell refinement later |
| Policy overfits on 13K frames | Medium | Medium | 500K params, dropout 0.1, early stopping |
| World model quality degrades with 2 extra tokens/step | Low | High | Validate frame acc before training policy |
| Two forward passes too slow at inference | Low | Low | ~60ms total; KV cache if needed |
| Unknown games have action types >7 | Low | Low | ACT_TYPE slots expandable at vocab extension |
| Training data distribution differs from test games | High | High | Architecture is game-agnostic; few-shot demos at test time |

---

## 11. File Map

```
src/aria_v2/
├── tokenizer/
│   ├── frame_tokenizer.py         # VQ-VAE (unchanged, 99.85% acc)
│   ├── train_vqvae.py             # VQ-VAE training
│   └── trajectory_dataset.py      # JSONL → (type, loc) tokens
│
├── world_model/
│   ├── config.py                  # WorldModelConfig, TrainingConfig, PolicyConfig
│   ├── game_transformer.py        # SmolLM2 + LoRA creation
│   ├── train.py                   # World model training (type/loc metrics)
│   ├── policy_head.py             # ActionTypeHead + ActionLocationHead
│   ├── train_policy.py            # Frozen backbone policy training
│   ├── agent.py                   # Inference agent (masked policy)
│   └── evaluate_world_model.py    # Type + location accuracy eval

checkpoints/
├── vqvae/best.pt                  # VQ-VAE (current, ~2MB)
├── world_model/best.pt            # SmolLM2 (needs retraining, 704MB)
└── policy/best.pt                 # Policy heads (not yet trained)
```

---

*Design document v2.0 — Game-Agnostic World Model with Separated Policy*
