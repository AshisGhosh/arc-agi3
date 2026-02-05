# Training Validation

## Summary
**Goal:** Validate ARIA-Lite architecture can learn from expert demos (BC) and rewards (RL)
**Status:** ✅ Complete
**Progress:** 13/13 experiments (4 initial + 9 component validation)

## Current Focus
Branch complete. Transitioned to primitives-pretraining.

## Tasks
- [x] Run initial multi-phase training pipeline
- [x] Identify and fix encoder issues
- [x] Implement expert solvers (A*, greedy)
- [x] Validate BC on synthetic tasks (nav/collect/switches)
- [x] Validate RL (PPO) on synthetic tasks
- [x] Test ARC-like tasks with dense rewards
- [x] Implement position-aware cross-attention for spatial transforms
- [x] Fix curriculum learning memorization bug
- [x] Validate copy task works with randomization fix

## Key Results
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| BC Navigation | >90% | 100% | 🟢 |
| BC Collection | >90% | 100% | 🟢 |
| BC Switches | >90% | 96.5% | 🟢 |
| RL Navigation | >50% | 84.5% | 🟢 |
| RL Collection | >50% | 93.5% | 🟢 |
| RL Switches | >50% | 87.5% | 🟢 |
| ARC copy 5x5 | >50% | 79% | 🟢 |
| ARC reflect_h 5x5 | >50% | 17% | 🟡 |
| ARC color_swap | >50% | 63% | 🟢 |

## Experiments

### Phase 0: Initial Multi-Phase Training

These experiments tested the full ARIA-Lite training pipeline (world model → fast policy → slow policy → arbiter → joint) before component-level validation.

#### EXP-000a: Quick Validation Run
- **Date:** 2026-02-03
- **Device:** CPU
- **Duration:** 2m 59s
- **Config:** wm_epochs=5, fp_epochs=5, sp_epochs=5, arb_epochs=3, joint_epochs=3
- **Result:** 0% success, Mean reward -5.60
- **Conclusion:** Validation run completed without errors. Low success expected with minimal epochs.

#### EXP-000b: Full GPU Training
- **Date:** 2026-02-03
- **Device:** CUDA
- **Duration:** 5m 33s
- **Config:** wm_epochs=100, fp_epochs=50, sp_epochs=100, arb_epochs=20, joint_epochs=50
- **Result:** 0% success, Mean reward -4.77
- **Training Losses:** WM 0.0896, FP 2.056, SP 2.285, Joint 4.241
- **Conclusion:** World model converged but policies not translating to success. Need simpler environments.

#### EXP-000c: Single-Mechanic Training
- **Date:** 2026-02-03
- **Device:** CUDA
- **Config:** epochs=30 per phase, single mechanic per environment
- **Results:**
  | Mechanic | Success Rate |
  |----------|--------------|
  | Navigation | 6% |
  | Collection | 1% |
  | Switches | 1% |
- **Conclusion:** Navigation shows promise. Simpler environments help learning.

#### EXP-000d: Focused Navigation Training
- **Date:** 2026-02-03
- **Device:** CUDA
- **Config:** wm=50, fp=100, sp=100, joint=50 epochs, navigation only
- **Progress:**
  | Phase | Success |
  |-------|---------|
  | Before | 0% |
  | After SP | 2% |
  | After Joint | 8% |
- **Conclusion:** Slow policy training critical. Joint fine-tuning helps. But 8% still low → need expert demonstrations.

---

### Phase 1: Component Validation

After initial experiments showed learning was possible but slow, we switched to validating individual components with expert demonstrations.

### EXP-001: Initial Training Attempt
- **Date:** 2026-02-03
- **Goal:** Test if agent learns from random rollouts + PPO
- **Result:** FAIL (8% success = random)
- **Conclusion:** BC on random data doesn't work. Need expert demonstrations.

### EXP-002: Encoder Diagnosis
- **Date:** 2026-02-03
- **Goal:** Test if encoder produces differentiated states
- **Result:** FAIL (pairwise diff = 0.0001, expected >0.1)
- **Conclusion:** Aggressive 2x downsampling kills spatial info. 10x10→2x2 = lost.

### EXP-003: Simple Encoder + Expert BC
- **Date:** 2026-02-03
- **Goal:** Validate simpler encoder + expert data
- **Result:** PASS (100%/100%/96.5% on nav/collect/switches)
- **Conclusion:** Simple encoder without downsampling works. Architecture validated.

### EXP-004: PPO RL Training
- **Date:** 2026-02-03
- **Goal:** Validate learning from rewards (not just imitation)
- **Result:** PASS (84.5%/93.5%/87.5%)
- **Conclusion:** Full RL pipeline works. Slightly lower than BC (expected).

### EXP-005: ARC-like Tasks (Full Action Space)
- **Date:** 2026-02-03
- **Goal:** Learn ARC transforms with cursor+color actions
- **Result:** FAIL (0% on all tasks)
- **Conclusion:** 16-action space too large, reward too sparse.

### EXP-006: Simplified ARC (Dense Rewards)
- **Date:** 2026-02-03
- **Goal:** Test dense per-cell rewards
- **Result:** PARTIAL (color_swap 63%, copy/reflect 0%)
- **Conclusion:** LOCAL transforms learnable, SPATIAL need position-aware arch.

### EXP-007: Position-Aware Attention
- **Date:** 2026-02-03
- **Goal:** Test cross-attention for spatial transforms
- **Result:** PASS (copy 87%, reflect_h 92%)
- **Conclusion:** Cross-attention learns position→position mappings.

### EXP-008: Curriculum Learning v2
- **Date:** 2026-02-03
- **Goal:** Generalize to larger grids via 2x2→5x5 curriculum
- **Result:** FAIL (training 88%, eval 2% at 4x4+)
- **Conclusion:** Deterministic seeds caused memorization, not learning.

### EXP-009: Randomization Fix
- **Date:** 2026-02-03
- **Goal:** Fix memorization with `deterministic=False`
- **Result:** PASS for copy (100%/90%/79% on 3x3/4x4/5x5)
- **Conclusion:** Copy task fixed. Reflect_h at 5x5 needs more training.

## Decisions Made

### DEC-001: Expert BC as Pretraining
- **Context:** Needed to validate architecture before RL
- **Choice:** BC validation first, then RL
- **Rationale:** If BC fails, RL definitely fails. Found encoder bug early.

### DEC-002: Encoder Architecture Fix
- **Context:** Original encoder killed learning
- **Choice:** Simple encoder for validation
- **Rationale:** Quick validation > perfect architecture. Works for 10x10.

### DEC-003: Next Phase Selection
- **Context:** BC complete, choosing next step
- **Choice:** RL first, then ARC tasks
- **Rationale:** RL validates learning pipeline, ARC tasks bridge to goal.

### DEC-004: Architecture for ARC Transforms
- **Context:** Simple encoder failed on spatial tasks
- **Choice:** Position-aware cross-attention encoder
- **Rationale:** Spatial transforms need position→position mapping.

## Files Created
```
src/aria_lite/
├── experts/                    # A*, greedy solvers
├── encoder_simple.py           # No-downsampling encoder
├── training/
│   ├── expert_data.py          # Expert trajectory collection
│   ├── bc_trainer.py           # Behavioral cloning
│   ├── ppo_trainer.py          # PPO with GAE
│   ├── arc_like_env.py         # Full ARC environment
│   ├── arc_like_simple.py      # Simplified ARC (dense rewards)
│   └── arc_position_encoder.py # Cross-attention encoder
├── train_bc.py, train_rl.py, train_arc_*.py
```

## Links
- [Top level](../PROGRESS.md)
- [ARC-AGI-3 exploration](arc-agi3-exploration.md)
