# Project Progress

## Current State
**Phase:** Primitives Pretraining Complete → ARC-AGI-3 Few-Shot Adaptation
**Branch:** main
**Status:** 🟢 On Track

## Immediate Next Step
Implement A* expert for ls20 to generate goal-directed demonstrations that understand game mechanics.

## Latest Findings (EXP-011: ARIA Architecture Integration)
**Integrated BC training with full ARIA architecture (encoder + fast policy):**
1. Full ARIA encoder (8.5M params) with 64x64 inputs: 31% accuracy - too large for 5905 samples
2. Simple encoder (0.6M params) with 16x16 inputs: 79% accuracy - matches standalone BC
3. Loop detection reveals **220+ loops per 300 steps** - policy gets stuck immediately
4. Confidence now ~50% low (vs 100% before) - arbiter would trigger slow policy
5. Random exploration (30% epsilon + loop breaking) still gets 0 levels

**Key insight:** The dual-system architecture is working correctly - fast policy is uncertain and would trigger slow policy. But we don't have a trained slow policy yet!

**Root cause:** BC learns state→action mapping without understanding goals. The agent navigates in loops because it doesn't know "where to go" (key positions, lock positions).

**Next steps:**
1. Implement A* expert to generate goal-directed trajectories (knows where keys/locks are)
2. Use A* demonstrations to understand game mechanics
3. Train slow policy to plan toward goals when fast policy is uncertain

## Recent Completions
- [2026-02-04] **Human demos loaded**: 28 demos (27 successful) from JSONL recordings via `jsonl_demo_loader.py`
- [2026-02-04] **Mode collapse identified**: Model learns action distribution, not game logic (needs architecture fix)
- [2026-02-04] **Few-shot training pipeline**: Demo collector + train_arc_fewshot.py (92.9% train accuracy)
- [2026-02-04] **Demo collection**: 110 demos across 3 games (ls20, vc33, ft09)
- [2026-02-04] **ARC-AGI-3 integration complete**: Created adapter (`arc_agent.py`) and validation script
- [2026-02-04] **Zero-shot baseline**: 0/3 games (ls20, vc33, ft09) - establishes baseline for improvement
- [2026-02-04] Installed `arcengine` package for local game execution
- [2026-02-04] Meta-learning validated: Nav 76%, Click 100% (few-shot generalization)
- [2026-02-04] All primitives validated: Nav 100%, Click 100%, Pattern 100%, Memory 100%, Counter 100%, Compositions 83-100%
- [2026-02-04] Fixed pattern matching with convolutional approach
- [2026-02-04] Fixed memory with explicit memory storage
- [2026-02-04] Implemented `src/aria_lite/primitives/` with 5 primitive families

## Active Branches
| Branch | Status | Progress | Link |
|--------|--------|----------|------|
| primitives-pretraining | ✅ Complete | 5/5 | [details](progress/primitives-pretraining.md) |
| arc-agi3-exploration | ✅ Complete | 3/3 | [details](progress/arc-agi3-exploration.md) |
| training-validation | ✅ Complete | 9/9 experiments | [details](progress/training-validation.md) |

## Architecture Summary
| Component | Params | Status |
|-----------|--------|--------|
| GridEncoderLite | 8.3M | 🟢 Validated |
| WorldModelLite | 7.9M | 🟢 Built |
| BeliefState | 0.8M | 🟢 Built |
| FastPolicy | 0.4M | 🟢 Validated |
| SlowPolicy | 8.5M | 🟢 Built |
| **Total** | **25.9M** | Target: 29M |

## Key Decisions Made
- [x] **Primitive scope:** Navigation, Click, Pattern, Memory, Counter, Compositions (all validated)
- [x] **Meta-learning approach:** Context-conditioned with FiLM (Nav 76%, Click 100%)
- [x] **Evaluation strategy:** Using 3 local games (ls20, vc33, ft09) for offline testing

## Next Steps
- [x] Collect demonstrations from game exploration (110 demos collected)
- [x] Fine-tune meta-learning model on game-specific demos (92.9% train accuracy)
- [x] Test few-shot adaptation (K=3 demos) - 0% success (mode collapse)
- [x] Load human demos from JSONL recordings (28 demos, 27 successful)
- [x] Retrain meta-learning model with human demos
- [x] Integrate BC with ARIA architecture (encoder + fast policy + arbiter)
- [x] Add loop detection - confirmed 75% of steps are loops
- [ ] **Implement A* expert** for ls20 navigation (understands goal locations)
- [ ] Generate goal-directed demonstrations with A* expert
- [ ] Train slow policy for deliberate planning when fast policy uncertain
- [ ] Test full dual-system (fast BC + slow planning)

## Links
- [Technical Report](TECHNICAL-REPORT.md) - Comprehensive decisions, experiments, and learnings
- [Progress Guide](PROGRESS-GUIDE.md) - Format instructions for all trackers
- [ARC-AGI-3 Mechanics](ARC-AGI3-MECHANICS.md) - Game analysis
- [ARIA-Lite Implementation](ARIA-LITE-IMPLEMENTATION.md) - Component design
