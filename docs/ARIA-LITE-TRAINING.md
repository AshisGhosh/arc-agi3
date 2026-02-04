# ARIA-Lite Training Progress

**Project:** ARIA-Lite Training & Evaluation
**Last Updated:** 2026-02-03

---

## Executive Summary

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Synthetic Success Rate | >60% | 0% | 🟡 Training |
| Fast/Slow Benefit | >10% | - | ⚪ |
| Fast Policy Usage | >50% | 0% | 🟡 Training |
| World Model Error (5-step) | <30% | - | ⚪ |
| Training VRAM | <7GB | 1.34GB | 🟢 |

**Current Phase:** Initial Training
**Status:** Quick validation complete, full training pending

---

## Training Phases

### Phase 1: World Model Pretraining

| Metric | Target | Quick Run | Full Run |
|--------|--------|-----------|----------|
| Epochs | 100 | 5 | - |
| Final Loss | <0.01 | 0.0072 | - |
| Status | - | 🟢 Complete | ⚪ Pending |

### Phase 2: Fast Policy (BC + Entropy)

| Metric | Target | Quick Run | Full Run |
|--------|--------|-----------|----------|
| Epochs | 50 | 5 | - |
| Final Loss | <2.0 | 2.08 | - |
| Entropy | >1.5 | 2.06 | - |
| Status | - | 🟢 Complete | ⚪ Pending |

### Phase 3: Slow Policy Training

| Metric | Target | Quick Run | Full Run |
|--------|--------|-----------|----------|
| Epochs | 100 | 5 | - |
| Final Loss | <3.0 | 3.88 | - |
| Status | - | 🟢 Complete | ⚪ Pending |

### Phase 4: Arbiter Calibration

| Metric | Target | Quick Run | Full Run |
|--------|--------|-----------|----------|
| Epochs | 20 | 3 | - |
| Fast Accuracy | >50% | 0% | - |
| Status | - | 🟢 Complete | ⚪ Pending |

### Phase 5: Joint Fine-tuning

| Metric | Target | Quick Run | Full Run |
|--------|--------|-----------|----------|
| Epochs | 50 | 3 | - |
| Final Loss | <5.0 | 5.38 | - |
| Status | - | 🟢 Complete | ⚪ Pending |

---

## Experiment Log

### EXP-001: Quick Validation Run

**Date:** 2026-02-03
**Device:** CPU
**Duration:** 2m 59s

**Configuration:**
```
wm_epochs=5, fp_epochs=5, sp_epochs=5, arb_epochs=3, joint_epochs=3
buffer_capacity=5000, batch_sizes=16-32
```

**Results:**
| Metric | Value |
|--------|-------|
| Mean Reward | -5.60 |
| Mean Steps | 62.2 |
| Success Rate | 0.0% |
| Fast Usage Rate | 0.0% |

**Notes:**
- All training phases completed without errors
- Losses decreased as expected during training
- Low success rate expected with minimal epochs
- Fast usage 0% indicates arbiter choosing slow policy (low fast confidence)

**Status:** PASS (validation successful)

---

### EXP-002: Full GPU Training

**Date:** 2026-02-03
**Device:** CUDA (pending)
**Duration:** -

**Configuration:**
```
wm_epochs=100, fp_epochs=50, sp_epochs=100, arb_epochs=20, joint_epochs=50
buffer_capacity=100000, batch_sizes=32-128
```

**Results:**
| Metric | Value |
|--------|-------|
| Mean Reward | - |
| Mean Steps | - |
| Success Rate | - |
| Fast Usage Rate | - |

**Status:** ⚪ PENDING

---

## Hyperparameter Tuning

### Current Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| WM Learning Rate | 1e-4 | |
| FP Learning Rate | 3e-4 | |
| SP Learning Rate | 1e-4 | |
| Joint Learning Rate | 1e-5 | |
| Gradient Clip | 1.0 | |
| Entropy Coef | 0.01 | |
| Value Coef | 0.5 | |
| PPO Clip Range | 0.2 | |

### Tuning Log

| Date | Parameter | From | To | Result |
|------|-----------|------|-----|--------|
| - | - | - | - | - |

---

## Environment Statistics

### Synthetic Environment Distribution

| Mechanic | Frequency | Success Rate |
|----------|-----------|--------------|
| Navigation | - | - |
| Collection | - | - |
| Switches | - | - |
| Keys/Doors | - | - |
| Pushing | - | - |
| Patterns | - | - |

### Grid Size Distribution

| Size Range | Count | Success Rate |
|------------|-------|--------------|
| 8-12 | - | - |
| 13-20 | - | - |
| 21-32 | - | - |

---

## Checkpoints

| Checkpoint | Date | Epoch | Val Success | Notes |
|------------|------|-------|-------------|-------|
| quick_validation | 2026-02-03 | - | 0% | CPU quick run |

---

## Next Steps

1. [ ] Run full GPU training (EXP-002)
2. [ ] Analyze fast vs slow policy switching
3. [ ] Tune arbiter thresholds based on results
4. [ ] Evaluate on ARC-AGI-3 API
5. [ ] Profile VRAM usage during training

---

## Commands Reference

```bash
# Quick validation (CPU, ~3 min)
PYTHONPATH=src uv run python -m aria_lite.train --quick --device cpu

# Full training (GPU, ~2-3 hours estimated)
PYTHONPATH=src uv run python -m aria_lite.train --device cuda

# Resume from checkpoint
PYTHONPATH=src uv run python -m aria_lite.train --resume checkpoints/aria_lite/final_checkpoint.pt
```

---

## Links

- [ARIA-Lite Progress](./ARIA-LITE-PROGRESS.md) - Implementation tracker
- [ARIA-Lite Implementation](./ARIA-LITE-IMPLEMENTATION.md) - Architecture details
