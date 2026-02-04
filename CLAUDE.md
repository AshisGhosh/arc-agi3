# Claude Instructions for ARC-AGI-3 Project

## Project Context

This is an ARC-AGI-3 competition project implementing the ARIA (Adaptive Reasoning with Integrated Abstractions) architecture.

## Current Focus: ARIA-Lite

We are implementing ARIA-Lite, a minimal 29M parameter dual-system architecture to validate the core hypothesis before scaling.

## Key Documents

- **Implementation Guide:** `docs/ARIA-LITE-IMPLEMENTATION.md`
- **Progress Tracker:** `docs/ARIA-LITE-PROGRESS.md` (UPDATE THIS REGULARLY)
- **Orchestrator Protocol:** `.claude/aria-lite-orchestrator.md`
- **Variants Comparison:** `docs/ARIA-VARIANTS.md`

## Orchestration Protocol

When working on ARIA-Lite implementation:

1. **Always read** `docs/ARIA-LITE-PROGRESS.md` first to understand current state
2. **Follow** the component workflow in `.claude/aria-lite-orchestrator.md`
3. **Update** the progress tracker after each significant action
4. **Use specialized agents** for their designated tasks:
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
