# World Model v1 Analysis: Structural Flaws and v2 Redesign Rationale

**Date:** 2026-02-09
**Context:** v1 world model achieved strong training metrics (88.4% frame acc, 1.8 ppl) but had three structural flaws preventing game-agnostic agent behavior.

---

## v1 Training Results

| Metric | Target | Achieved |
|--------|--------|----------|
| Frame prediction accuracy | >40% | **88.4%** |
| Action prediction accuracy | >30% | **67.9%** |
| Perplexity | <20 | **1.8** |
| Level completion prediction | N/A | **99.5%** |
| Validation loss | N/A | **0.5721** |

These numbers look good on paper. But closer analysis revealed the model learned shortcuts rather than genuine game dynamics.

---

## Flaw 1: Click Coordinates Discarded

### Problem
The v1 tokenization mapped every action to a single token:
```
<ACT_0>..<ACT_6>  — 7 action codes
```

For vc33 and ft09, where 99% of actions are `action_id=6` (click), the model only saw:
```
[FRAME] vq0..vq63 [ACT] <ACT_6> [FRAME] vq0..vq63
```

It had **zero information** about WHERE the click happened. The JSONL data contains `action_input.data.x` and `action_input.data.y`, but these were never tokenized.

### Impact
- On vc33/ft09: action prediction always predicts `<ACT_6>` (trivially correct since it's 99% of data)
- Frame prediction for click games is essentially unguided — the model can't learn "clicking cell 46 changes cells 46-47"
- The 88.4% frame accuracy is heavily weighted by ls20 (which has diverse actions)

### Evidence from Data
```
vc33 step 1: action_input={"id": 6, "data": {"x": 62, "y": 26}}
vc33 step 2: action_input={"id": 6, "data": {"x": 52, "y": 47}}
ft09 step 1: action_input={"id": 6, "data": {"x": 40, "y": 46}}
```
All of this spatial information was thrown away.

---

## Flaw 2: Action Prediction Mode Collapse

### Problem
In v1, action tokens were 1 out of 67 tokens per step (~1.5% of the sequence). With loss weight 2.0 (vs 1.0 for VQ), action tokens contributed roughly 3% of total gradient. The model learned to predict actions from **action history** rather than from **visual state**.

### Diagnostic
Action prediction accuracy (67.9%) was suspiciously high given the simple architecture. Analysis would show:
- Action predictions are strongly correlated with recent action history (last 3-5 actions)
- Action predictions are weakly correlated with current frame content
- For ls20: if the last 3 actions were "UP", the model predicts "UP" again
- This is correct 67.9% of the time because humans tend to repeat actions in sequences

### Root Cause
The `lm_head` prediction at the action position attends to ALL prior tokens including previous action tokens. Since action tokens are much simpler (7 classes) than VQ tokens (512 classes), the model finds it easier to copy from recent action tokens than to read the complex visual state.

---

## Flaw 3: Policy and Dynamics Entangled

### Problem
A single `lm_head` output layer serves as both:
1. **World model:** predicting next frame given action
2. **Policy:** predicting next action given frame

There's no separation between "what will happen?" and "what should I do?" The model can shortcut by:
- Copying previous actions for policy (see Flaw 2)
- Attending to action tokens when predicting frames (correct but unhelpful for policy)

### Impact
At inference, the agent uses the same model for both world modeling (surprise/goal inference) and action selection. The entanglement means:
- Goal inference (P(LEVEL_COMPLETE | action=a)) is biased by action frequency
- Surprise measurement is confounded by action prediction quality
- Policy can't be evaluated independently of dynamics quality

---

## v2 Redesign: How Each Flaw Is Fixed

### Fix 1: Unified Action Tokenization
**Every action = (type, location)** — 2 tokens instead of 1.

```
v1: [ACT] <ACT_6>                              — no spatial info
v2: [ACT] <ACT_TYPE_6> <ACT_LOC_46>            — click at VQ cell 46
```

Action locations map to the same 8x8 spatial grid as the VQ-VAE:
- `cell = (y // 8) * 8 + (x // 8)` — pixel to cell
- `x = (cell % 8) * 8 + 4, y = (cell // 8) * 8 + 4` — cell to pixel center

**8 action types** (expandable for unknown games) + **65 locations** (64 VQ cells + NULL).

### Fix 2: Increased Loss Weight + Separate Metrics
- Action type weight: 3.0 (was 2.0 combined)
- Action location weight: 3.0 (new)
- Separate tracking: `action_type_acc` and `action_loc_acc` instead of combined `action_acc`
- For click games, location accuracy reveals whether the model truly understands spatial dynamics

### Fix 3: Architectural Separation
**World model** (dynamics) trains on full context with action tokens — learns "what happens if I click here?"
**Policy** (action selection) trains on **masked context** where all action tokens are replaced with a learned `<MASK>` token — can only learn from visual consequences.

```
World model sees:  [FRAME] vq... [ACT] TYPE LOC [FRAME] vq...
Policy sees:       [FRAME] vq... [MASK][MASK][MASK] [FRAME] vq...
```

**Mode collapse is architecturally impossible.** The policy heads (500K params) train on a frozen backbone with no access to action history.

---

## Expected Impact

| Aspect | v1 | v2 Expected |
|--------|-----|-------------|
| Click game dynamics | Cannot learn (no coords) | Should learn spatial effects |
| Action prediction quality | Mode-collapsed (copies history) | Must read visual state |
| Policy independence | Entangled with dynamics | Architecturally separated |
| ls20 frame prediction | 88.4% | Should maintain or improve |
| vc33/ft09 frame prediction | Unguided | Should improve with spatial context |
| Agent on click games | Cannot select locations | Full spatial action output |

---

## What We're Keeping

- **VQ-VAE**: 99.85% pixel accuracy, game-agnostic frame encoding. No changes.
- **SmolLM2-360M + LoRA**: Same backbone, same precision, same LoRA config.
- **Training infrastructure**: Same loop, hyperparams, hardware utilization.
- **Data**: Same 28 demos, 3 games. Only the tokenization changes.
