# ARIA-Lite Technical Report

## Executive Summary

**Project:** ARC-AGI-3 Competition Entry using ARIA-Lite Architecture
**Timeline:** 2026-02-03 to 2026-02-04 (ongoing)
**Current Status:** Primitives validated, human demos loaded, mode collapse identified

### Key Metrics
| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Primitive BC accuracy | >90% | 100% | ✅ |
| Primitive RL accuracy | >50% | 87% | ✅ |
| Meta-learning (nav) | >70% | 76% | ✅ |
| Meta-learning (click) | >70% | 100% | ✅ |
| ARC-AGI-3 few-shot | >10% | 0% | ❌ Blocked |

---

## 1. Problem Statement

### 1.1 ARC-AGI-3 Challenge
The ARC-AGI-3 challenge requires agents to play novel games with:
- 64x64 pixel observations
- 6 discrete actions (game-specific semantics)
- Sparse rewards (level completion only)
- Unknown rules (must be discovered through play)

### 1.2 Our Approach: ARIA-Lite
A 29M parameter dual-system architecture:
- **Fast system:** Neural habits learned from demonstrations
- **Slow system:** Deliberate planning with world model
- **Meta-learning:** Adapt to new games from few demonstrations

**Why this approach?**
- Small enough to iterate quickly (fits in 7GB VRAM)
- Dual-system allows both reactive and deliberate behavior
- Meta-learning enables rapid adaptation to new games

---

## 2. Architecture Decisions

### DEC-001: Expert BC Before RL
**Date:** 2026-02-03
**Context:** Need to validate architecture can learn at all
**Decision:** Test behavioral cloning (BC) with expert demonstrations first
**Rationale:**
- If BC fails, RL will definitely fail (BC has perfect labels)
- Allows isolating architecture bugs from exploration challenges
- Found encoder bug early (see DEC-002)
**Outcome:** Validated BC works (100% on nav/collect/switches)

### DEC-002: Simple Encoder Architecture
**Date:** 2026-02-03
**Context:** Original encoder killed learning (EXP-002 showed pairwise diff = 0.0001)
**Decision:** Remove aggressive downsampling, use simple CNN
**Rationale:**
- Original: 10x10 → 5x5 → 2x2 lost all spatial information
- Simple: 10x10 → 128-dim preserves position info
- Trade-off: Less abstraction but works
**Outcome:** Immediate improvement to 100% BC accuracy

### DEC-003: Position-Aware Cross-Attention for Spatial Tasks
**Date:** 2026-02-03
**Context:** Simple encoder failed on spatial transforms (copy, reflect)
**Decision:** Add cross-attention with position encodings
**Rationale:**
- Spatial transforms are position→position mappings
- Cross-attention can learn "output[i,j] = input[f(i,j)]"
- Inspired by Vision Transformer spatial reasoning
**Outcome:** 87% copy, 92% reflect_h accuracy

### DEC-004: Convolutional Template Matching for Patterns
**Date:** 2026-02-04
**Context:** MLP memorized patterns instead of learning matching (3% eval)
**Decision:** Use convolution operation for template matching
**Rationale:**
- Template matching IS convolution mathematically
- Sliding window comparison = conv kernel
- End-to-end differentiable
**Outcome:** 100% pattern matching accuracy

### DEC-005: Explicit Memory for State Tracking
**Date:** 2026-02-04
**Context:** GRU-based memory achieved only 33% on memory tasks
**Decision:** Add explicit memory buffer that stores previous observations
**Rationale:**
- GRU must learn what to remember implicitly
- Explicit memory: "store obs when trigger, compare later"
- More interpretable and reliable
**Outcome:** 100% memory task accuracy

### DEC-006: Meta-Learning with FiLM Conditioning
**Date:** 2026-02-04
**Context:** Need to adapt to new tasks from demonstrations
**Decision:** Use FiLM (Feature-wise Linear Modulation) conditioning
**Rationale:**
- Simple: γ, β parameters modulate hidden layers
- Demonstrated success in few-shot learning
- Lighter than full attention over demos
**Outcome:** 76% nav, 100% click few-shot accuracy

### DEC-007: JSONL Demo Loading over OCR Video Extraction
**Date:** 2026-02-04
**Context:** Human gameplay videos available, need to extract demos
**Decision:** Load directly from JSONL recording files instead of OCR
**Rationale:**
- JSONL files contain structured data (frame, action, state)
- OCR is error-prone and requires preprocessing
- Direct loading is faster and more reliable
**Outcome:** 28 human demos loaded (27 successful)

---

## 3. Experiments Timeline

### Phase 0: Initial Training (2026-02-03)

| ID | Goal | Result | Key Learning |
|----|------|--------|--------------|
| EXP-000a | Quick validation | 0% success | Pipeline works, needs epochs |
| EXP-000b | Full training | 0% success | World model converges, policies don't transfer |
| EXP-000c | Single mechanic | 6% nav | Simpler environments help |
| EXP-000d | Focused nav | 8% success | Slow policy critical, joint tuning helps |

**Key Learning:** Random rollouts insufficient. Need expert demonstrations.

### Phase 1: Component Validation (2026-02-03)

| ID | Goal | Result | Key Learning |
|----|------|--------|--------------|
| EXP-001 | Random BC | 8% (random) | BC on random data doesn't work |
| EXP-002 | Encoder check | 0.0001 diff | Downsampling kills spatial info |
| EXP-003 | Expert BC | 100% | Simple encoder + expert data works |
| EXP-004 | PPO RL | 87% | Full RL pipeline validated |
| EXP-005 | ARC full | 0% | 16-action space too large |
| EXP-006 | ARC dense | 63% color | Local transforms learnable |
| EXP-007 | Position attn | 87% copy | Cross-attention for spatial |
| EXP-008 | Curriculum | 2% eval | Memorization, not learning |
| EXP-009 | Randomization | 79% 5x5 | Fixed memorization bug |

**Key Learning:** Architecture works. Need proper training data and task design.

### Phase 2: Primitives Pretraining (2026-02-04)

| ID | Goal | Result | Key Learning |
|----|------|--------|--------------|
| EXP-001 | Click primitive | 100% | Position-aware CNN works |
| EXP-002 | Pattern match | 100% | Conv template matching |
| EXP-003 | State tracking | 100% | Explicit memory needed |
| EXP-004 | Compositions | 83-100% | Primitives compose |
| EXP-005 | Meta-learning | 76-100% | FiLM conditioning works |
| EXP-006 | ARC-AGI-3 zero | 0% | Expected, establishes baseline |

**Key Learning:** All primitives validated. Meta-learning works on synthetic tasks.

### Phase 3: ARC-AGI-3 Adaptation (2026-02-04)

| ID | Goal | Result | Key Learning |
|----|------|--------|--------------|
| Random demos | Collect 110 | 0 successful | Need expert trajectories |
| Human demos | Load 28 | 27 successful | Quality data available |
| Few-shot train | 92.9% train | 0% eval | **MODE COLLAPSE** |

**Key Learning:** Model collapses to predicting most common action (UP=74.5%).

### Phase 4: ARIA Architecture Integration (2026-02-05)

| ID | Goal | Result | Key Learning |
|----|------|--------|--------------|
| EXP-011a | Full encoder BC | 31% acc, 0 levels | 8.5M params too large for 5905 samples |
| EXP-011b | Simple encoder BC | 79% acc, 0 levels | Matches standalone BC accuracy |
| EXP-011c | Loop detection | 220+/300 loops | BC creates loops immediately |
| EXP-011d | Exploration (30%) | 0 levels | Random exploration insufficient |

**Key Learning:**
- Simple encoder achieves same accuracy as standalone BC (79%)
- Loop detection confirms policy gets stuck in 75% of steps
- Confidence is ~50% low - arbiter correctly triggers slow policy need
- But no trained slow policy exists yet to help when fast policy uncertain
- Need goal-directed exploration (A* expert) instead of random exploration

---

## 4. Current Blocker: Mode Collapse

### Problem Description
After training on 28 human demonstrations (27 successful):
- Training accuracy: 92.9%
- Evaluation on actual game: 0% levels completed
- Model predicts action 1 (UP) for 74.5% of inputs

### Root Cause Analysis
The training setup has a fundamental issue:
1. **Training:** Model sees (obs_t, obs_t+1, ..., obs_t+k), predicts action_t+k+1
2. **Evaluation:** Model sees (demo1_obs, demo2_obs, demo3_obs), predicts action for new_obs
3. **Mismatch:** Training learns trajectory continuation, not game understanding

### Action Distribution in Human Demos (ls20)
| Action | Count | Percentage |
|--------|-------|------------|
| 0 (NOOP) | 13 | 2.2% |
| 1 (UP) | 245 | 40.8% |
| 2 (DOWN) | 134 | 22.3% |
| 3 (LEFT) | 109 | 18.2% |
| 4 (RIGHT) | 99 | 16.5% |

The model learns the marginal distribution, not the conditional policy.

### Potential Solutions
1. **Observation-conditioned BC:** Train policy(obs) → action directly, ignore demo context
2. **Game-specific experts:** A* pathfinding for ls20 navigation
3. **RL with reward shaping:** Learn from game rewards with exploration bonus
4. **Different meta-learning:** MAML or gradient-based adaptation

---

## 5. Files and Components

### Core Architecture (src/aria_lite/)
| File | Params | Purpose |
|------|--------|---------|
| encoder.py | 8.3M | Grid observation encoding |
| encoder_simple.py | ~1M | Simplified encoder (validated) |
| world_model.py | 7.9M | Next-state prediction |
| belief.py | 0.8M | Belief state tracking |
| fast_policy.py | 0.4M | Reactive policy |
| slow_policy.py | 8.5M | Deliberate planning |
| arbiter.py | ~0.1M | Fast/slow switching |
| agent.py | - | Orchestration |

### Training Infrastructure (src/aria_lite/training/)
| File | Purpose |
|------|---------|
| bc_trainer.py | Behavioral cloning |
| ppo_trainer.py | PPO with GAE |
| expert_data.py | Expert trajectory collection |
| synthetic_env.py | Procedural environments |
| arc_like_env.py | ARC-style grid transforms |

### Primitives (src/aria_lite/primitives/)
| File | Families |
|------|----------|
| navigation.py | direct, obstacles, waypoints |
| click.py | target, sequence |
| pattern.py | match, difference, complete, cycle |
| state_tracking.py | memory, counter, multi_property |
| composition.py | nav_then_click, pattern_then_act, conditional |

### ARC-AGI-3 Integration
| File | Purpose |
|------|---------|
| arc_agent.py | Adapter for ARC-AGI-3 API |
| demo_collector.py | Collect game demonstrations |
| jsonl_demo_loader.py | Load human demo recordings |
| train_arc_fewshot.py | Few-shot training pipeline |

---

## 6. Lessons Learned

### Architecture
1. **Downsampling kills spatial info:** Keep full resolution for position-sensitive tasks
2. **Template matching is convolution:** Use the right inductive bias
3. **Explicit > implicit memory:** GRU can't learn what to remember
4. **Cross-attention for spatial:** Position→position mappings need attention

### Training
1. **BC before RL:** Validate architecture with perfect labels first
2. **Expert data essential:** Random rollouts don't teach useful behaviors
3. **Dense rewards help:** Sparse rewards require exploration tricks
4. **Randomization crucial:** Deterministic seeds cause memorization

### Meta-Learning
1. **FiLM works for simple tasks:** 76-100% on synthetic primitives
2. **Mode collapse risk:** Model can learn distribution, not policy
3. **Demo format matters:** Must match training and evaluation setup

---

## 7. Next Steps

### Immediate (Unblock Mode Collapse)
1. [ ] Try observation-conditioned BC (ignore demo context)
2. [ ] Implement A* expert for ls20 navigation
3. [ ] Test if expert policy transfers to game

### Short-term (Validate Approach)
1. [ ] Achieve >0% on one ARC-AGI-3 game
2. [ ] Compare few-shot vs. game-specific training
3. [ ] Measure sample efficiency

### Medium-term (Scale Up)
1. [ ] Extend to more games
2. [ ] Add slow deliberation for novel situations
3. [ ] Evaluate on full ARC-AGI-3 benchmark

---

## Appendix A: Hardware Configuration

- **GPU:** RTX 4090 (24GB VRAM)
- **Target VRAM:** <7GB training, <4GB inference
- **Measured VRAM:** ~3GB training (current models)

## Appendix B: Key Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| grid_size | 10 | Balance resolution/efficiency |
| hidden_dim | 128-256 | Fits VRAM budget |
| num_actions | 9 | Covers ARC-AGI-3 action space |
| num_colors | 16 | Standard ARC palette |
| learning_rate | 1e-3 | Standard for Adam |
| batch_size | 32-64 | Fits GPU memory |

## Appendix C: References

- ARC-AGI Challenge: https://arcprize.org/
- FiLM Conditioning: Perez et al. (2018)
- Cross-Attention: Vaswani et al. (2017)
- PPO: Schulman et al. (2017)
