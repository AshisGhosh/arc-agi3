# Progress Tracking Guide

**Purpose:** Standardized format for tracking project progress across all agents and sessions.

---

## Hierarchy

```
docs/
├── PROGRESS.md                    # TOP LEVEL - Current state & next steps
├── PROGRESS-GUIDE.md              # THIS FILE - Instructions & formats
└── progress/
    ├── training-validation.md     # BC/PPO validation experiments
    ├── arc-agi3-exploration.md    # Game mechanics analysis
    ├── primitives-pretraining.md  # Primitive task pretraining
    └── meta-learning.md           # Meta-learning architecture
```

---

## Top-Level Format (PROGRESS.md)

The top-level tracker must be **glanceable**. Anyone should understand project state in 30 seconds.

```markdown
# Project Progress

## Current State
**Phase:** [Current major phase]
**Branch:** [Active work branch]
**Status:** [🟢 On Track | 🟡 Blocked | 🔴 Off Track]

## Immediate Next Step
[Single clear action - what to do RIGHT NOW]

## Recent Completions
- [Date] [What was completed]

## Active Branches
| Branch | Status | Progress | Link |
|--------|--------|----------|------|
| [name] | [status emoji] | [X/Y or %] | [link] |

## Decisions Pending
- [ ] [Decision needed with context]
```

---

## Branch Tracker Format

Each branch tracker follows this structure:

```markdown
# [Branch Name]

## Summary
**Goal:** [What this branch aims to achieve]
**Status:** [🟢 Active | 🟡 Paused | ⚪ Not Started | ✅ Complete]
**Progress:** [X/Y tasks or percentage]

## Current Focus
[What's being worked on right now]

## Tasks
- [x] Completed task
- [ ] Pending task
- [ ] **NEXT:** Immediate next task

## Key Results
| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| [name] | [value] | [value] | [emoji] |

## Experiments
### EXP-XXX: [Name]
- **Date:** YYYY-MM-DD
- **Goal:** [What we tested]
- **Result:** [PASS/FAIL/PARTIAL]
- **Conclusion:** [Actionable insight]

## Decisions Made
### DEC-XXX: [Title]
- **Context:** [Why decision was needed]
- **Choice:** [What was decided]
- **Rationale:** [Why]

## Blockers
- [ ] [Blocker description] → [Proposed resolution]

## Links
- [Related doc](path)
```

---

## Status Indicators

| Emoji | Meaning |
|-------|---------|
| 🟢 | On track / Passing / Active |
| 🟡 | Needs attention / In progress / Partial |
| 🔴 | Blocked / Failing / Critical |
| ⚪ | Not started / Pending |
| ✅ | Complete / Done |

---

## Naming Conventions

### Experiments
- Format: `EXP-XXX` where XXX is sequential number
- Reset numbering per branch (EXP-001 in each branch)

### Decisions
- Format: `DEC-XXX` where XXX is sequential number
- Include date and rationale

### Tasks
- Use checkbox format: `- [ ]` or `- [x]`
- Mark immediate next with: `**NEXT:**`

---

## Update Protocol

### When to Update

1. **After completing a task** - Check it off, add to "Recent Completions"
2. **After an experiment** - Add results with conclusion
3. **After a decision** - Document context and rationale
4. **When blocked** - Add to Blockers with proposed resolution
5. **At session start** - Read top-level to orient
6. **At session end** - Ensure state is captured

### Who Updates

- Any agent working on the project
- Always update the branch tracker you're working on
- Only update top-level when phase/branch changes

---

## Migration Notes

### From Old Format
The previous `ARIA-LITE-PROGRESS.md` contained:
- Component status (now complete - archived)
- Experiment log (migrate to branch trackers)
- Decision log (migrate to branch trackers)

Archive old content, don't delete - may need historical reference.

---

## Quick Reference

**Starting a session:**
1. Read `PROGRESS.md` (top-level)
2. Find your branch, read its tracker
3. Look for `**NEXT:**` task

**Ending a session:**
1. Update branch tracker with progress
2. If phase changed, update `PROGRESS.md`
3. Ensure "Immediate Next Step" is clear for next session

**Adding an experiment:**
1. Add to branch tracker under "## Experiments"
2. Include Date, Goal, Result, Conclusion
3. If result changes direction, update "## Current Focus"

**Making a decision:**
1. Add to branch tracker under "## Decisions Made"
2. Include Context, Choice, Rationale
3. If decision affects other branches, note in top-level
