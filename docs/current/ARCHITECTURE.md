# Learned World Model Architecture (v2)

## Overview

The agent learns game dynamics from human demonstrations via next-token prediction, then selects actions through a separate policy that operates on masked context. A VQ-VAE tokenizes visual frames, and a LoRA-adapted SmolLM2-360M serves as both dynamics predictor and feature backbone for the policy.

```
Frame (64x64, 16 colors)
  → VQ-VAE encoder → 64 discrete tokens (8x8 grid, 512-code codebook)
  → Trajectory: [GAME_START] [FRAME] v1..v64 [ACT] <TYPE> <LOC> [FRAME] v1..v64 ...
  → SmolLM2-360M (LoRA) → dynamics prediction (world model)
  → Masked context → PolicyHeads → action type + location (separated policy)
```

| Spec | Value |
|------|-------|
| **Approach** | VQ-VAE + autoregressive transformer + separated policy |
| **Base model** | SmolLM2-360M (362M params, 960 hidden, 32 layers) |
| **World model params** | ~51M trainable (3.9M LoRA + 47M embed/lm_head) |
| **Policy params** | ~500K (ActionTypeHead + ActionLocationHead) |
| **Training data** | 28 human demos, 3 games, ~900K tokens |
| **Precision** | bfloat16 (model) + float32 (loss) |
| **VRAM** | ~10GB training, ~1GB inference |
| **Inference speed** | ~60ms per step (two backbone passes) |

---

## Key Design Principles

1. **No game-specific heuristics.** No hardcoded knowledge of ls20 mechanics, color meanings, or action semantics.
2. **Unified action = (type, location).** Works for navigation, clicking, and unknown games.
3. **Dynamics separate from policy.** World model = "what happens?" Policy = "what should I do?"
4. **Architectural mode collapse prevention.** Policy input has action tokens masked.

---

## Component 1: VQ-VAE Frame Tokenizer

**File:** `src/aria_v2/tokenizer/frame_tokenizer.py`

Converts 64x64 16-color game frames into 64 discrete tokens. Unchanged from v1.

```
Input: [B, 64, 64] int tensor (0-15)
  → nn.Embedding(16, 32)                              → [B, 64, 64, 32]
  → Encoder (4 conv layers, stride 2/2/2/1)           → [B, 128, 8, 8]
  → VectorQuantizer (512 codes, 128-dim, EMA updates) → [B, 8, 8] indices
  → Decoder (mirror of encoder)                        → [B, 16, 64, 64] logits
```

**Results:** 99.85% pixel accuracy, 44.73% codebook utilization.

---

## Component 2: Unified Action Tokenization

**File:** `src/aria_v2/tokenizer/trajectory_dataset.py`

Every action becomes 2 tokens: `<ACT_TYPE_i> <ACT_LOC_j>`.

### Token Vocabulary (589 new tokens)

| Tokens | Count | IDs |
|--------|-------|-----|
| VQ codes (`<VQ_000>`..`<VQ_511>`) | 512 | 49152-49663 |
| Action types (`<ACT_TYPE_0>`..`<ACT_TYPE_7>`) | 8 | 49664-49671 |
| Action locations (`<ACT_LOC_0>`..`<ACT_LOC_64>`) | 65 | 49672-49736 |
| `<FRAME>` marker | 1 | 49737 |
| `<ACT>` marker | 1 | 49738 |
| `<LEVEL_COMPLETE>` | 1 | 49739 |
| `<GAME_START>` | 1 | 49740 |
| `<MASK>` (policy only) | 1 | 49741 |

### VQ Cell Mapping

Action locations map to the same 8x8 spatial grid as the VQ-VAE:
```
cell = (y // 8) * 8 + (x // 8)    # pixel → cell
x = (cell % 8) * 8 + 4             # cell → pixel center
y = (cell // 8) * 8 + 4
```

### Sequence Format (69 tokens per step)
```
<GAME_START> <FRAME> v1..v64 <ACT> <ACT_TYPE_i> <ACT_LOC_j> <FRAME> v1..v64 ...
```

### Per-Game Examples
| Game | Action | Tokens |
|------|--------|--------|
| ls20 (navigate right) | action_id=2, no coords | `<ACT_TYPE_2> <ACT_LOC_NULL>` |
| vc33 (click at 50,44) | action_id=6, x=50 y=44 | `<ACT_TYPE_6> <ACT_LOC_46>` |
| ft09 (click at 40,46) | action_id=6, x=40 y=46 | `<ACT_TYPE_6> <ACT_LOC_45>` |

---

## Component 3: World Model (Dynamics Only)

**File:** `src/aria_v2/world_model/game_transformer.py`

### Architecture
```
SmolLM2-360M (LlamaForCausalLM):
  vocab_size: 49741 (49152 base + 589 game tokens)
  hidden_size: 960
  num_layers: 32
  num_attention_heads: 15
  max_position_embeddings: 8192

LoRA:
  rank: 16, alpha: 32, dropout: 0.05
  targets: q_proj, k_proj, v_proj, o_proj
```

### Initialization
- VQ token embeddings: projected from VQ-VAE codebook (128→960)
- ACT_LOC embeddings: initialized from VQ codebook (spatial grounding)
- All new embeddings scaled to match existing distribution

### Training Objective
Next-token prediction with per-token-type loss weights:
- VQ frame tokens: weight 1.0
- Action type: weight 3.0
- Action location: weight 3.0
- `<LEVEL_COMPLETE>`: weight 5.0
- Structural markers: weight 0.0

**Why this fixes click games:** The world model now sees `[FRAME] vq... [ACT] TYPE_6 LOC_46 [FRAME] vq...` and can learn "clicking cell 46 changed cells 46 and 47." Before, it only saw `[ACT] 6` with no spatial info.

---

## Component 4: Policy (Separate from World Model)

**File:** `src/aria_v2/world_model/policy_head.py`

### Action Masking

The policy shares the SmolLM2 backbone but sees **masked** context where all action tokens are replaced with a learned `<MASK>` token:

```
World model sees:  [FRAME] vq... [ACT] TYPE LOC [FRAME] vq... [ACT] TYPE LOC [FRAME] vq...
Policy sees:       [FRAME] vq... [MASK][MASK][MASK] [FRAME] vq... [MASK][MASK][MASK] [FRAME] vq...
```

**Mode collapse is architecturally impossible.** The policy can only learn from visual consequences.

### Two-Headed Output

**ActionTypeHead** (Linear(960,256) → GELU → Dropout(0.1) → Linear(256,8)):
- Input: hidden state at last VQ token position
- Output: logits over 8 action types
- ~248K parameters

**ActionLocationHead** (spatial attention):
- Query: projection of frame summary hidden state → 128-dim
- Keys: projection of 64 VQ-cell hidden states + 1 learnable NULL key → 128-dim
- Output: scaled dot-product attention scores over 65 positions
- ~250K parameters

**Total policy: ~500K parameters.** Trains on frozen backbone.

---

## Component 5: Inference Loop

**File:** `src/aria_v2/world_model/agent.py`

```python
def act(frame, available_actions):
    # 1. Encode frame
    vq_codes = vqvae.encode(frame)

    # 2. Update world model context (includes real action tokens)
    wm_context.append(FRAME_TOKEN, *vq_codes)

    # 3. Create masked context for policy
    policy_context = mask_actions(wm_context)

    # 4. Run backbone on masked context
    hidden_states = backbone(policy_context)

    # 5. Predict action type + location
    type_logits = action_type_head(h_frame)
    action_type = sample(softmax(type_logits))
    loc_logits = action_location_head(h_frame, h_vq_cells)
    action_loc = sample(softmax(loc_logits))

    # 6. Convert to game API
    x, y = vq_cell_to_pixel(action_loc) if action_loc != NULL else (None, None)
    env.step(action, data={"x": x, "y": y} if x else None)

    # 7. Update context with chosen action
    wm_context.append(ACT_TOKEN, ACT_TYPE + action_type, ACT_LOC + action_loc)
```

---

## Why v2 Over v1

### v1 Flaws
1. **Click coordinates discarded.** For vc33/ft09 (99% action 6), only action ID tokenized. Model literally couldn't learn click dynamics.
2. **Action prediction mode-collapsed.** Actions = 1/67 tokens (~1.5% of gradient). Model learned to copy recent action history rather than read visual states.
3. **Policy and dynamics entangled.** Single `lm_head` served as both simulator and policy. Policy could shortcut via raw action token access.

### v2 Fixes
1. **Unified (type, location) tokens.** Click games now have spatial information. VQ cell mapping provides consistent spatial reference.
2. **Separate type/loc metrics.** Loss weight 3.0 on both (was 2.0 on combined action). Separate tracking reveals per-game quality.
3. **Architectural separation.** World model predicts dynamics, policy reads only visual consequences. MASK token ensures no action history leakage.

---

*Architecture version: 4.0 (v2 game-agnostic redesign)*
*Approach: VQ-VAE + SmolLM2-360M + LoRA + Separated Policy*
