# Documentation Index

## Quick Navigation

| What you need | Where to look |
|---------------|---------------|
| Current state & next steps | [PROGRESS.md](PROGRESS.md) |
| Architecture details | [current/ARCHITECTURE.md](current/ARCHITECTURE.md) |
| What's built & what's next | [current/IMPLEMENTATION-PLAN.md](current/IMPLEMENTATION-PLAN.md) |
| ARC-AGI-3 API reference | `reference/` |
| What we learned from past approaches | `findings/` |
| Old/abandoned approaches | `archive/` |

---

## Current Approach: Learned World Model

**Status:** Training complete, agent evaluation next

```
Frame (64x64, 16 colors)
  → VQ-VAE encoder → 64 discrete tokens (512-code codebook)
  → SmolLM2-360M (LoRA) → next-token prediction
  → Surprise + Goal inference + Learned policy
```

| Document | Description |
|----------|-------------|
| [PROGRESS.md](PROGRESS.md) | Training results, code status, next steps |
| [current/ARCHITECTURE.md](current/ARCHITECTURE.md) | VQ-VAE + SmolLM2 + LoRA architecture |
| [current/IMPLEMENTATION-PLAN.md](current/IMPLEMENTATION-PLAN.md) | 5 stages (all done) + evaluation plan |

---

## Reference Material

| Document | Description |
|----------|-------------|
| [reference/ARC-AGI-3-OVERVIEW.md](reference/ARC-AGI-3-OVERVIEW.md) | Competition overview, scoring, timeline |
| [reference/API-GUIDE.md](reference/API-GUIDE.md) | `arc-agi` package usage, GameAction, FrameData |
| [reference/BUILDING-AGENTS.md](reference/BUILDING-AGENTS.md) | Agent interface, strategies, templates |
| [reference/GAME-MECHANICS.md](reference/GAME-MECHANICS.md) | ls20, vc33, ft09 game analysis |

---

## Historical Findings

| Document | Description |
|----------|-------------|
| [findings/ARIA-V1-REPORT.md](findings/ARIA-V1-REPORT.md) | BC/PPO experiments, why they failed |

**Key takeaways:**
- BC: 80% accuracy, 0% levels (mode collapse)
- PPO: 0.18% success (sparse reward too hard)
- Puzzle games need understanding, not imitation

---

## Archive

| Folder | Contents |
|--------|----------|
| [archive/architectures/](archive/architectures/) | Old architecture designs (Dreamer, Hybrid, Neurosymbolic) |
| [archive/v1-progress/](archive/v1-progress/) | V1 progress tracking |
| [archive/ABSTRACT-LEARNING.md](archive/ABSTRACT-LEARNING.md) | Heuristic rule learning approach |
| [archive/internal/](archive/internal/) | Process documentation |
