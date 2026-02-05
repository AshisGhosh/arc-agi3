# Documentation Index

## Quick Navigation

| What you need | Where to look |
|---------------|---------------|
| Current architecture & plan | `current/` |
| How to use ARC-AGI-3 API | `reference/` |
| What we learned from v1 | `findings/` |
| Old/abandoned approaches | `archive/` |

---

## Current Development (ARIA v2)

**Status:** Active development

| Document | Description |
|----------|-------------|
| [PROGRESS.md](PROGRESS.md) | Project state, next steps, recent completions |
| [current/ARCHITECTURE.md](current/ARCHITECTURE.md) | Language-guided meta-learning architecture |
| [current/IMPLEMENTATION-PLAN.md](current/IMPLEMENTATION-PLAN.md) | 5-phase implementation plan |

---

## Reference Material

**Status:** Always useful, competition-specific

| Document | Description |
|----------|-------------|
| [reference/ARC-AGI-3-OVERVIEW.md](reference/ARC-AGI-3-OVERVIEW.md) | Competition overview, scoring, timeline |
| [reference/API-GUIDE.md](reference/API-GUIDE.md) | `arc-agi` package usage, GameAction, FrameData |
| [reference/BUILDING-AGENTS.md](reference/BUILDING-AGENTS.md) | Agent interface, strategies, templates |
| [reference/GAME-MECHANICS.md](reference/GAME-MECHANICS.md) | ls20, vc33, ft09 game analysis |

---

## Historical Findings

**Status:** Valuable learnings from completed experiments

| Document | Description |
|----------|-------------|
| [findings/ARIA-V1-REPORT.md](findings/ARIA-V1-REPORT.md) | ARIA v1 experiments, decisions, lessons learned |
| [experiments/aria_v1/results/summary.md](../experiments/aria_v1/results/summary.md) | V1 experiment metrics and outcomes |

### Key Takeaways from v1

- BC achieves 80% accuracy but 0% level completion (mode collapse)
- PPO with sparse reward: 0.18% success rate
- World model learns dynamics (0.033 loss) but not goals
- ls20 is a puzzle game requiring state matching, not just navigation
- Need language-based reasoning to understand game rules

---

## Archive

**Status:** Superseded or abandoned approaches

| Folder | Contents |
|--------|----------|
| [archive/architectures/](archive/architectures/) | Old architecture designs (Dreamer, Hybrid, Neurosymbolic) |
| [archive/v1-progress/](archive/v1-progress/) | V1 progress tracking files |
| [archive/internal/](archive/internal/) | Process documentation |

---

## Directory Structure

```
docs/
├── README.md              # This file
├── PROGRESS.md            # Current state (start here)
│
├── current/               # ARIA v2 (what we're building)
│   ├── ARCHITECTURE.md
│   └── IMPLEMENTATION-PLAN.md
│
├── reference/             # Competition & API docs
│   ├── ARC-AGI-3-OVERVIEW.md
│   ├── API-GUIDE.md
│   ├── BUILDING-AGENTS.md
│   └── GAME-MECHANICS.md
│
├── findings/              # What we learned
│   └── ARIA-V1-REPORT.md
│
└── archive/               # Old/abandoned work
    ├── architectures/
    ├── v1-progress/
    └── internal/
```
