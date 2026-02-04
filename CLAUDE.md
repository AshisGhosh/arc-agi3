# Claude Instructions for ARC-AGI-3 Project

## Project Context

This is an ARC-AGI-3 competition project implementing the ARIA (Adaptive Reasoning with Integrated Abstractions) architecture.

## Current Focus: ARIA-Lite

We are implementing ARIA-Lite, a minimal 29M parameter dual-system architecture to validate the core hypothesis before scaling.

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
- **Progress Guide:** `docs/PROGRESS-GUIDE.md` (format instructions)
- **Implementation Guide:** `docs/ARIA-LITE-IMPLEMENTATION.md`
- **ARC-AGI-3 Mechanics:** `docs/ARC-AGI3-MECHANICS.md`
- **Variants Comparison:** `docs/ARIA-VARIANTS.md`

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
src/aria_lite/          # ARIA-Lite implementation (to be created)
src/aria/               # Original ARIA (reference, some reuse)
src/arc_dreamer_v2/     # World model, belief tracking (reuse)
src/arc_neurosymbolic_v2/  # DSL, reasoning (reference)
```

## Success Criteria

| Metric | Target |
|--------|--------|
| Level completion (seen) | >60% |
| Fast/slow switching benefit | >10% |
| World model error (5-step) | <30% |
| Training VRAM | <7GB |

## Commands

```bash
# Lint code
uv run ruff check --fix src/aria_lite/

# Run tests (when available)
uv run pytest src/aria_lite/tests/

# Check VRAM usage
nvidia-smi
```
