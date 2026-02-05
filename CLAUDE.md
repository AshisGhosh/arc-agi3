# Claude Instructions for ARC-AGI-3 Project

## Golden Rule

**If it isn't documented, it didn't happen. If it isn't in the plan, don't do it.**

Before ANY work:
1. Read `docs/PROGRESS.md` - What is the current state?
2. Read `docs/current/IMPLEMENTATION-PLAN.md` - What are we supposed to be doing?
3. Verify consistency - Does the code match what's documented?

If documentation is inconsistent or outdated, **fix it first** before proceeding.

---

## Documentation-First Protocol

### Before Starting Any Task

```
1. READ docs/PROGRESS.md
   - What phase are we in?
   - What was the last completed step?
   - What is the immediate next step?

2. READ docs/current/IMPLEMENTATION-PLAN.md
   - Does the requested task match the plan?
   - If not, clarify with user before proceeding

3. CHECK consistency
   - Does src/aria_v2/ match what's documented?
   - Are there undocumented files or changes?
   - If inconsistent, document or remove orphaned work
```

### During Work

- Document experiments as you run them (EXP-XXX format)
- Document decisions as you make them (DEC-XXX format)
- If you create a file, it must be referenced in docs
- If you abandon an approach, move to archive/

### After Completing Work

```
1. UPDATE docs/PROGRESS.md
   - Mark completed items
   - Update "Immediate Next Step"
   - Add to "Recent Completions"

2. UPDATE relevant docs
   - If architecture changed: docs/current/ARCHITECTURE.md
   - If new findings: docs/findings/

3. COMMIT with clear message describing what changed
```

---

## Project Context

ARC-AGI-3 competition project implementing ARIA v2 (language-guided meta-learning).

**Why v2?** ARIA v1 (end-to-end neural) failed:
- BC: 80% accuracy, 0% level completion (mode collapse)
- PPO: 0.18% success rate (sparse reward too hard)
- Root cause: Need to understand game rules, not just mimic actions

**ARIA v2 Architecture:**
```
Observation → Visual Grounding → Language Description
                    ↓
           Event Detection → "Player touched diamond, score +1"
                    ↓
           LLM Reasoning → "Diamonds are collectibles"
                    ↓
           Subgoal Executor → Navigate to next diamond
```

---

## Documentation Structure

```
docs/
├── PROGRESS.md              # Current state (READ FIRST)
│
├── current/                 # Active development
│   ├── ARCHITECTURE.md      # ARIA v2 architecture spec
│   └── IMPLEMENTATION-PLAN.md  # 5-phase build plan
│
├── reference/               # Competition docs
│   ├── ARC-AGI-3-OVERVIEW.md
│   ├── API-GUIDE.md
│   ├── BUILDING-AGENTS.md
│   └── GAME-MECHANICS.md
│
├── findings/                # What we learned
│   └── ARIA-V1-REPORT.md    # v1 experiments & lessons
│
└── archive/                 # Abandoned/superseded
    ├── architectures/       # Old architecture designs
    ├── v1-progress/         # Old progress tracking
    └── internal/            # Process docs
```

### What Goes Where

| Content | Location |
|---------|----------|
| Current project state | `docs/PROGRESS.md` |
| Architecture spec | `docs/current/ARCHITECTURE.md` |
| Implementation tasks | `docs/current/IMPLEMENTATION-PLAN.md` |
| New findings/learnings | `docs/findings/` |
| Abandoned approaches | `docs/archive/` |
| Competition reference | `docs/reference/` |

---

## Code Location

```
src/aria_v2/            # ARIA v2 (CURRENT DEVELOPMENT)
├── config.py           # Configuration
├── visual_grounding/   # Entity detection (Phase 1)
├── event_detector/     # Change tracking (Phase 2)
├── llm_reasoning/      # LLM integration (Phase 3)
├── subgoal_executor/   # Navigation (Phase 4)
└── agent.py            # Integrated agent (Phase 5)

experiments/aria_v1/    # Archived v1 experiments
src/aria_lite/          # ARIA v1 code (deprecated)
```

---

## Documentation Formats

### Experiments (EXP-XXX)

```markdown
### EXP-XXX: [Descriptive Name]
- **Date:** YYYY-MM-DD
- **Goal:** What hypothesis are we testing?
- **Setup:** Model, data, hyperparameters
- **Result:** Metrics (be specific)
- **Conclusion:** What did we learn? Next action?
```

### Decisions (DEC-XXX)

```markdown
### DEC-XXX: [Decision Title]
- **Context:** What problem required a decision?
- **Options:** What alternatives existed?
- **Choice:** What did we decide?
- **Rationale:** WHY? (most important)
- **Outcome:** What happened as a result?
```

---

## Proactive Maintenance

### File Organization

- **New files** must be documented in PROGRESS.md or relevant docs
- **Orphaned files** (not referenced anywhere) should be archived or deleted
- **Experimental code** goes in `experiments/`, not `src/`

### Consistency Checks

If you notice:
- Code that isn't documented → Document it or remove it
- Docs that don't match code → Update docs or fix code
- Stale progress items → Update or archive them

**Fix inconsistencies immediately.** Don't leave them for later.

---

## Success Criteria (ARIA v2)

| Metric | Target | Phase |
|--------|--------|-------|
| Entity detection accuracy | >90% | 1 |
| Event identification | >80% | 2 |
| Navigation success rate | >95% | 4 |
| ARC level completion | >10% | 5 |

---

## Specialized Agents

Use these for designated tasks:
- `python-ml-architect` → Code structure, ML patterns
- `multimodal-action-architect` → Architecture decisions
- `rl-training-expert` → Training strategy
- `experiment-manager` → Experiment tracking
- `arc-agi-evaluator` → Approach evaluation

---

## Commands

```bash
# Lint code
uv run ruff check --fix src/aria_v2/

# Run tests
uv run pytest src/aria_v2/tests/

# Check VRAM usage
nvidia-smi
```
