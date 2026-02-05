# ARIA v1 Experiment Results Summary

## Overview

**Period:** 2026-02-03 to 2026-02-05
**Total Experiments:** 13 major experiments
**Final Outcome:** Architecture validated on synthetic tasks, failed on ARC-AGI-3 games

---

## Key Metrics

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Primitive BC accuracy | >90% | 100% | ✅ |
| Primitive RL accuracy | >50% | 87% | ✅ |
| Meta-learning (nav) | >70% | 76% | ✅ |
| Meta-learning (click) | >70% | 100% | ✅ |
| ARC-AGI-3 few-shot | >10% | 0% | ❌ |
| ARC-AGI-3 PPO | >10% | 0.18% | ❌ |

---

## Experiment Timeline

### Phase 1: Component Validation (Feb 3)

| ID | Goal | Result | Learning |
|----|------|--------|----------|
| EXP-001 | Random BC | 8% | Random data insufficient |
| EXP-002 | Encoder check | 0.0001 diff | Downsampling kills spatial info |
| EXP-003 | Expert BC | 100% | Simple encoder works |
| EXP-004 | PPO RL | 87% | RL pipeline validated |

### Phase 2: Primitives Pretraining (Feb 4)

| ID | Goal | Result | Learning |
|----|------|--------|----------|
| EXP-001 | Click primitive | 100% | Position-aware CNN works |
| EXP-002 | Pattern match | 100% | Conv template matching |
| EXP-003 | State tracking | 100% | Explicit memory needed |
| EXP-004 | Compositions | 83-100% | Primitives compose |
| EXP-005 | Meta-learning | 76-100% | FiLM conditioning works |

### Phase 3: ARC-AGI-3 Adaptation (Feb 4-5)

| ID | Goal | Result | Learning |
|----|------|--------|----------|
| EXP-006 | Zero-shot | 0% | Expected baseline |
| EXP-011 | Full ARIA BC | 79%/0% | Mode collapse |
| EXP-012 | World model | 0.033 loss | Learns dynamics not goals |
| EXP-013 | PPO | 0.18% | Sparse reward too hard |

---

## Checkpoints Produced

| File | Description | Performance |
|------|-------------|-------------|
| `aria_bc_simple.pt` | BC-trained encoder + fast policy | 79% accuracy |
| `world_model.pt` | 3-head ensemble world model | 0.033 state loss |
| `ppo_ls20.pt` | PPO trained on ls20 | 0.18% success |
| `bc_human_ls20.pt` | BC on human demos | 79% accuracy |
| `meta_human_200.pt` | Meta-learning model | 76% nav, 100% click |

---

## Human Demo Analysis

**Game:** ls20 (puzzle game)
**Demos:** 12 human recordings
**Success:** 11 won (7 levels), 1 got 6 levels

### Action Distribution
| Action | Count | Percentage |
|--------|-------|------------|
| UP | 2704 | 31.0% |
| DOWN | 2442 | 28.0% |
| LEFT | 1696 | 19.5% |
| RIGHT | 1830 | 21.0% |
| NOOP | 39 | 0.4% |

**Key Finding:** Balanced action distribution (not UP-biased like our model predicted)

---

## Root Cause Analysis

### Why BC Failed
1. **Mode collapse:** Model learns action distribution, not conditional policy
2. **No context:** Same observation can need different actions depending on game state
3. **Evaluation mismatch:** Training on trajectory continuation, evaluating on game play

### Why PPO Failed
1. **Sparse reward:** Only `levels_completed` signal available
2. **Complex goal:** ls20 requires state matching (sprite + color + rotation)
3. **Exploration insufficient:** Random actions can't discover puzzle mechanics

### Why World Model Wasn't Enough
1. **Dynamics ≠ Goals:** Predicting next state doesn't tell you what to aim for
2. **Curiosity insufficient:** Novel states ≠ goal states
3. **No reward shaping:** Game provides no intermediate feedback

---

## Lessons Learned

### Architecture
1. Downsampling kills spatial information - keep full resolution
2. Template matching is convolution - use right inductive bias
3. Explicit memory beats implicit (GRU) for state tracking
4. Cross-attention needed for spatial transforms

### Training
1. BC before RL - validate with perfect labels first
2. Expert data essential - random rollouts don't teach
3. Dense rewards help - sparse requires exploration tricks
4. Randomization crucial - deterministic causes memorization

### Meta-Learning
1. FiLM works for simple tasks (76-100%)
2. Mode collapse risk with few demos
3. Demo format must match train/eval setup

---

## Decision: Move to ARIA v2

**Reason:** End-to-end neural can't learn puzzle mechanics from sparse rewards

**New Approach:** Language-guided meta-learning
- Visual grounding → Language description
- LLM reasoning → Rule discovery
- Subgoal execution → Pretrained navigation

See `docs/ARIA-V2-ARCHITECTURE.md` for details.
