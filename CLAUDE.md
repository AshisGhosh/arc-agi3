# Claude Instructions for ARC-AGI-3 Project

## Project Context

This is an ARC-AGI-3 competition project implementing the ARIA (Adaptive Reasoning with Integrated Abstractions) architecture.

## Current Focus: ARIA v2 - Language-Guided Meta-Learning

We shifted from ARIA-Lite (end-to-end neural) to ARIA v2 (language-guided reasoning) because:
- BC/PPO couldn't learn puzzle game mechanics from sparse rewards
- Need meta-learning that understands game rules, not just mimics actions
- Language provides transferable representation for game understanding

**Core Architecture:**
```
Observation → Visual Grounding → Language Description
                    ↓
           Event Detection → "Player touched diamond, score +1"
                    ↓
           LLM Reasoning → "Diamonds are collectibles"
                    ↓
           Subgoal Executor → Navigate to next diamond
```

## Progress Tracking (READ FIRST)

**ALWAYS start a session by reading `docs/PROGRESS.md`** - this is the single source of truth for project state.

### Hierarchy
```
docs/PROGRESS.md              # TOP LEVEL - Current state & next steps
docs/PROGRESS-GUIDE.md        # Meta instructions for trackers
docs/progress/
├── training-validation.md    # BC/PPO experiments (complete)
├── arc-agi3-exploration.md   # Game mechanics (complete)
└── primitives-pretraining.md # Current work
```

### Update Protocol
1. **At session start:** Read `docs/PROGRESS.md` to orient
2. **After completing work:** Update the relevant branch tracker
3. **After experiments:** Add results with EXP-XXX format
4. **After decisions:** Document with DEC-XXX format
5. **At session end:** Ensure state is captured for next session

See `docs/PROGRESS-GUIDE.md` for full format specifications.

## Key Documents

- **Progress Tracker:** `docs/PROGRESS.md` (READ FIRST, UPDATE REGULARLY)
- **ARIA v2 Architecture:** `docs/ARIA-V2-ARCHITECTURE.md` (CURRENT APPROACH)
- **ARIA v2 Implementation Plan:** `docs/ARIA-V2-IMPLEMENTATION-PLAN.md`
- **Technical Report:** `docs/TECHNICAL-REPORT.md` (comprehensive decisions & learnings)
- **Progress Guide:** `docs/PROGRESS-GUIDE.md` (format instructions)
- **ARC-AGI-3 Mechanics:** `docs/ARC-AGI3-MECHANICS.md`
- **ARIA-Lite (deprecated):** `docs/ARIA-LITE-IMPLEMENTATION.md`

## Continuous Documentation (CRITICAL)

**Every significant action must be documented.** This project requires detailed tracking for reproducibility and learning.

### What to Document

1. **Decisions (DEC-XXX format):**
   - Context: What problem were we solving?
   - Options considered: What alternatives existed?
   - Decision: What did we choose?
   - Rationale: WHY did we choose it? (most important)
   - Outcome: What happened?

2. **Experiments (EXP-XXX format):**
   - Goal: What hypothesis are we testing?
   - Method: Exact setup, hyperparameters, data
   - Result: Metrics, success/failure
   - Key Learning: What did we learn?

3. **Blockers:**
   - Problem description
   - Root cause analysis
   - Potential solutions with trade-offs

4. **Architecture changes:**
   - What changed and why
   - Trade-offs made
   - Impact on other components

### Where to Document

| Type | Location |
|------|----------|
| Current state | `docs/PROGRESS.md` |
| Experiment details | `docs/progress/*.md` |
| Decisions & learnings | `docs/TECHNICAL-REPORT.md` |
| Architecture rationale | Code comments + TECHNICAL-REPORT.md |

### When to Document

- **Before starting:** Read existing docs to avoid repeating work
- **During work:** Note unexpected findings immediately
- **After experiments:** Record results even if negative
- **After decisions:** Document rationale while fresh
- **End of session:** Update PROGRESS.md with current state

## Specialized Agents

Use these agents for their designated tasks:
- `python-ml-architect` → Code structure, ML patterns
- `multimodal-action-architect` → Architecture decisions
- `rl-training-expert` → Training strategy
- `experiment-manager` → Metrics, experiments, tracking
- `arc-agi-evaluator` → Component scoring

## Gate Decisions

For each component, make explicit PASS/ITERATE/BLOCKED decisions:
- **PASS:** Metrics met, proceed to next component
- **ITERATE:** Partially met, document fixes needed
- **BLOCKED:** Escalate to user with options

## Code Location

```
src/aria_v2/            # ARIA v2 implementation (CURRENT)
├── visual_grounding.py # Entity detection
├── event_detector.py   # Change tracking
├── llm_reasoning.py    # LLM integration
├── subgoal_executor.py # Navigation
└── agent.py            # Integrated agent

src/aria_lite/          # ARIA v1 (deprecated, reference only)
src/aria/               # Original ARIA (reference)
src/arc_dreamer_v2/     # World model, belief tracking
```

## Success Criteria (ARIA v2)

| Metric | Target |
|--------|--------|
| Entity detection accuracy | >90% |
| Navigation success rate | >95% |
| Rule discovery rate | >50% |
| ARC level completion | >10% |
| Adaptation speed | <50 steps to correct hypothesis |

## Git Commits

**IMPORTANT:**
- All commits must be authored by AshisGhosh only
- Never mention Claude, add co-authors, or use any other author identity
- Never modify git config user.name or user.email
- When using git-manager agent, ensure SSH key is for AshisGhosh (not AshisBotCo or others)
- Keep commit messages focused on the changes made

## Commands

```bash
# Lint code
uv run ruff check --fix src/aria_lite/

# Run tests (when available)
uv run pytest src/aria_lite/tests/

# Check VRAM usage
nvidia-smi
```
