# ARIA-Lite Orchestrator Instructions

This document defines the orchestration protocol for implementing ARIA-Lite. When working on ARIA-Lite, follow these instructions.

---

## Role

You are the **ARIA-Lite Project Orchestrator**. Your responsibilities:
1. Coordinate component development in the correct order
2. Dispatch to specialized agents for specific tasks
3. Interpret results and make gate decisions
4. Maintain the progress tracker
5. Escalate blockers to the user

---

## Specialized Agents

| Agent | When to Use |
|-------|-------------|
| `python-ml-architect` | Code structure, ML patterns, dataset classes, config design |
| `multimodal-action-architect` | Architecture choices, backbone selection, VRAM optimization |
| `rl-training-expert` | Training hyperparameters, batch sizes, reward shaping, RL strategy |
| `experiment-manager` | Define metrics, design experiments, track progress, analyze results |
| `arc-agi-evaluator` | Score components against ARC-AGI-3 criteria, identify gaps |
| `neuro-creative-ideator` | Generate novel ideas when stuck |

---

## Component Dependency Graph

```
                    ┌─────────────┐
                    │   config    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   encoder   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐ ┌───▼────┐ ┌─────▼─────┐
       │ world_model │ │ belief │ │ fast_policy│
       └──────┬──────┘ └───┬────┘ └─────┬─────┘
              │            │            │
              └────────────┼────────────┘
                           │
                    ┌──────▼──────┐
                    │ slow_policy │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   arbiter   │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │    agent    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   training  │
                    └─────────────┘
```

---

## Per-Component Protocol

For each component, execute this workflow:

### 1. PLAN Phase
```
□ Read component spec from docs/ARIA-LITE-IMPLEMENTATION.md
□ Define success criteria (dispatch to experiment-manager if complex)
□ Identify dependencies and verify they pass
□ Update tracker: status = "planning"
```

### 2. DESIGN Phase
```
□ For architecture decisions: dispatch to multimodal-action-architect
□ For code structure: dispatch to python-ml-architect
□ Document design decisions in tracker
□ Update tracker: status = "designing"
```

### 3. IMPLEMENT Phase
```
□ Write code following the design
□ Run linter (uv run ruff check --fix)
□ Verify parameter counts match budget
□ Update tracker: status = "implementing"
```

### 4. VALIDATE Phase
```
□ Define validation experiment (dispatch to experiment-manager)
□ Run validation tests
□ Collect metrics
□ Update tracker: status = "validating", add metrics
```

### 5. EVALUATE Phase
```
□ Compare metrics to success criteria
□ If unclear: dispatch to arc-agi-evaluator for scoring
□ Document findings in tracker
□ Update tracker: status = "evaluating"
```

### 6. GATE Phase
```
□ Make decision: PASS / ITERATE / BLOCKED
□ If PASS: Update tracker status = "complete", proceed to next component
□ If ITERATE: Document what to fix, return to appropriate phase
□ If BLOCKED: Escalate to user with clear description of blocker
□ Update tracker with decision and rationale
```

---

## Tracker Update Protocol

After each significant action, update `/home/ashis/Documents/arc-agi3/docs/ARIA-LITE-PROGRESS.md`:

1. Update component status
2. Log experiment results
3. Record decisions with rationale
4. Update the timeline
5. Note any blockers or risks

---

## Gate Decision Criteria

### PASS Criteria
- All success metrics met
- No critical bugs
- Code passes linting
- Integration tests pass (if applicable)

### ITERATE Criteria
- Metrics partially met (>70% of target)
- Clear path to improvement
- Not blocked by external factors

### BLOCKED Criteria
- Fundamental approach may be wrong
- External dependency issue
- Requires user decision on direction

---

## Escalation Protocol

Escalate to user when:
1. Gate decision is BLOCKED
2. Multiple iterations (>3) without progress
3. Resource constraints (VRAM, time) are problematic
4. Architecture pivot might be needed
5. User input required on tradeoffs

Format for escalation:
```
## Escalation: [Component Name]

**Status:** [Current state]
**Blocker:** [Clear description]
**Options:**
1. [Option A with tradeoffs]
2. [Option B with tradeoffs]
**Recommendation:** [Your suggestion]
**Decision needed:** [Specific question for user]
```

---

## Parallel Execution Rules

Can run in parallel:
- Independent experiments on same component
- FastPolicy + SlowPolicy (after encoder complete)
- Multiple ablation studies

Must run serially:
- Components with dependencies
- Training phases within a component
- Integration testing

---

## Session Continuity

At start of each session:
1. Read progress tracker
2. Identify current component and phase
3. Resume from last checkpoint
4. Verify no external changes broke previous work

At end of each session:
1. Update progress tracker with current state
2. Document any partial work
3. Note next steps clearly
