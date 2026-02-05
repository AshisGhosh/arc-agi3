# Project Progress

## Current State
**Phase:** ARIA v2 - Phase 1: Visual Grounding
**Branch:** main
**Status:** 🟢 Ready to implement

## Immediate Next Step
**Phase 1.1:** Create synthetic game generator (`src/aria_v2/pretraining/synthetic_games.py`)
- Generate procedural games with labeled entities (player, goal, item, obstacle, trigger)
- Target: 10k labeled game states for training entity detector

## Code Status
| Component | File | Status |
|-----------|------|--------|
| Configuration | `src/aria_v2/config.py` | ✅ Done |
| Visual Grounding | `src/aria_v2/visual_grounding.py` | ⚪ Not started |
| Event Detector | `src/aria_v2/event_detector.py` | ⚪ Not started |
| LLM Reasoning | `src/aria_v2/llm_reasoning.py` | ⚪ Not started |
| Subgoal Executor | `src/aria_v2/subgoal_executor.py` | ⚪ Not started |
| Integrated Agent | `src/aria_v2/agent.py` | ⚪ Not started |

## Architecture Shift: ARIA v1 → ARIA v2

**Why the change:**
- ARIA v1 (end-to-end neural) couldn't learn ls20 puzzle mechanics
- BC achieves 80% accuracy but 0% level completion (wrong policy)
- PPO with sparse reward couldn't discover goals (0.18% success)
- Need meta-learning that understands game rules, not just mimics actions

**ARIA v2 Core Insight:**
> "Understand games in language, then act"

**New Architecture:**
```
Observation → Visual Grounding → Language Description
                    ↓
           Event Detection → "Player touched diamond, score +1"
                    ↓
           LLM Reasoning → "Diamonds are collectibles, goal is collect all"
                    ↓
           Subgoal Executor → Navigate to next diamond
```

See [ARIA v2 Architecture](current/ARCHITECTURE.md) for full details.

## Latest Findings (EXP-013: PPO Training)
**Implemented PPO training on ARC-AGI-3:**
1. **PPO v1**: 3/1712 episodes reached levels (0.18% success rate)
2. **PPO v2 (improved rewards)**: First test showed 10/500 success, but full run: 0/1700+ episodes
3. **BC with class weights**: Backfired - NOOP dominated (49.6%), worse than before

**Key discoveries about ls20:**
- ls20 is a PUZZLE game, not just navigation
- Requires: navigate to target + match state (sprite/color/rotation)
- Human demos: balanced actions (UP 31%, DOWN 28%, LEFT 20%, RIGHT 21%)
- Human average: 400-850 steps to complete all 7 levels
- All 12 human demos successful (11 won, 1 got 6 levels)

**What we tried:**
| Approach | Result | Issue |
|----------|--------|-------|
| BC (79% acc) | 0 levels | Mode collapse, loops |
| World Model | state_loss=0.033 | Learns dynamics, not goals |
| PPO v1 | 0.18% success | Sparse reward too hard |
| PPO v2 (stronger rewards) | 0% | High variance, no learning |
| BC + class weights | 0% | NOOP dominated |
| BC + random exploration | 0% | Ineffective |

**Root cause confirmed:**
- ls20 requires understanding game mechanics (state matching)
- Random/RL exploration insufficient for puzzle games
- BC learns imitation but not goal understanding

## Recent Completions
- [2026-02-05] **Docs reorganized**: Structured into current/, reference/, findings/, archive/
- [2026-02-05] **ARIA v2 config**: Created `src/aria_v2/config.py` with component configurations
- [2026-02-05] **ARIA v2 Architecture**: Designed language-guided meta-learning system
- [2026-02-05] **Architecture decision**: Shift from end-to-end neural to language-based reasoning
- [2026-02-05] **PPO training implemented**: train_ppo_arc.py with reward shaping, achieved 0.18% success rate
- [2026-02-05] **ls20 game analysis**: Identified as puzzle game requiring state matching (not just navigation)
- [2026-02-05] **Human demo analysis**: 12 demos, all successful, balanced action distribution
- [2026-02-05] **BC class weighting**: Attempted inverse frequency weighting, backfired (NOOP dominated)
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

## Archived Branches (ARIA v1)
| Branch | Status | Progress | Link |
|--------|--------|----------|------|
| primitives-pretraining | ✅ Complete | 5/5 | [details](archive/v1-progress/primitives-pretraining.md) |
| arc-agi3-exploration | ✅ Complete | 3/3 | [details](archive/v1-progress/arc-agi3-exploration.md) |
| training-validation | ✅ Complete | 9/9 experiments | [details](archive/v1-progress/training-validation.md) |

## Key Decisions Made
- [x] **Architecture shift:** ARIA v1 → v2 (end-to-end neural → language-guided reasoning)
- [x] **Evaluation strategy:** Using 3 local games (ls20, vc33, ft09) for offline testing
- [x] **LLM choice:** TinyLlama 1.1B (small enough to fit in VRAM alongside vision)

## Next Steps - ARIA v2 Implementation

### Phase 1: Visual Grounding (Pretraining)
- [ ] Create synthetic game generator with labeled entities
- [ ] Implement VisualGroundingModule (entity detection + classification)
- [ ] Train entity detector (player, goal, item, obstacle, trigger)
- [ ] Train movement correlator (what moves when action taken)
- [ ] Validate: >90% entity detection accuracy

### Phase 2: Event Detection
- [ ] Implement EventDetector (track state changes)
- [ ] Build cause-effect relationship detector
- [ ] Test on human demo recordings
- [ ] Validate: correctly identifies item collection, level completion

### Phase 3: LLM Reasoning
- [ ] Download Llama 3.2 1B (GGUF quantized)
- [ ] Implement LLMReasoningEngine
- [ ] Create prompts for: event interpretation, goal hypothesis, subgoal generation
- [ ] Add response caching for efficiency
- [ ] Validate: reasonable hypotheses on synthetic games

### Phase 4: Subgoal Executor
- [ ] Implement PretrainedNavigationPolicy (A* based)
- [ ] Train on synthetic navigation tasks
- [ ] Add obstacle avoidance
- [ ] Validate: >95% navigation success rate

### Phase 5: Integration
- [ ] Implement ARIAv2Agent (full pipeline)
- [ ] Test on ls20 with language trace logging
- [ ] Evaluate rule discovery rate
- [ ] Target: >10% level completion on ARC-AGI-3

## Completed (ARIA v1)
- [x] BC training - 80% accuracy, 0% eval
- [x] PPO training - 0.18% success
- [x] World model - learns dynamics
- [x] Human demo analysis - all 12 successful

## Links
- [**ARIA v2 Architecture**](current/ARCHITECTURE.md) - Language-guided meta-learning (CURRENT)
- [ARIA v2 Implementation Plan](current/IMPLEMENTATION-PLAN.md) - 5-phase build plan
- [ARIA v1 Technical Report](findings/ARIA-V1-REPORT.md) - Decisions, experiments, learnings
- [Game Mechanics Analysis](reference/GAME-MECHANICS.md) - ls20, vc33, ft09 analysis
- [ARIA v1 Experiment Results](../experiments/aria_v1/results/summary.md) - Metrics and outcomes
- [Archived Architectures](archive/architectures/) - v1 approaches (deprecated)
