# ARC-AGI-3 Exploration

## Summary
**Goal:** Understand ARC-AGI-3 game mechanics and identify gaps vs our training
**Status:** ✅ Complete
**Progress:** 3/3 tasks

## Current Focus
Branch complete. Findings inform primitives-pretraining design.

## Tasks
- [x] Set up local offline game execution
- [x] Download and analyze 3 preview games (ls20, vc33, ft09)
- [x] Document gap analysis: our training vs ARC-AGI-3 requirements

## Key Results
| Metric | Our Training | ARC-AGI-3 | Gap |
|--------|--------------|-----------|-----|
| Grid size | 5x5 | 64x64 | 13x larger |
| Actions | Color selection | Navigate/Click+coords | Different space |
| Rewards | Dense (+1/cell) | Sparse (level complete) | Major |
| Rules | Known (copy input) | Must discover | Major |
| Planning | Greedy works | Multi-step required | Major |

## Games Analyzed

### ls20 (Locksmith)
- **Type:** Navigation + pattern matching
- **Actions:** 1-4 (UDLR, 5 pixels each)
- **Mechanics:** Navigate sprite, configure key (shape/color/rotation), unlock doors
- **Skills:** Pathfinding, state tracking, pattern matching

### vc33 (Budget Logic)
- **Type:** Click-based constraint puzzle
- **Actions:** 6 (click at x,y)
- **Mechanics:** Toggle cells within budget to satisfy constraints
- **Skills:** Constraint satisfaction, resource optimization

### ft09 (Pattern Matching)
- **Type:** Click-based color cycling
- **Actions:** 1-5 (patterns), 6 (click x,y)
- **Mechanics:** Cycle colors to match same/different constraints
- **Skills:** Pattern recognition, color cycle tracking

## Decisions Made

### DEC-001: Offline Mode Setup
- **Context:** API throttling blocked iteration
- **Choice:** Download games locally, use `OPERATION_MODE=offline`
- **Rationale:** Fast local iteration without rate limits

### DEC-002: Architecture Adaptations Needed
- **Context:** Major gaps between our training and ARC-AGI-3
- **Choice:** Build primitive pretraining layer before meta-learning
- **Rationale:** Can't jump directly to complex games. Need foundation.

## Recommended Adaptations

1. **Encoder:** Handle 64x64 inputs, extract objects/patterns
2. **Policy:** Variable action spaces (nav vs click), coordinate prediction
3. **World Model:** Learn game dynamics, enable planning
4. **Meta-Learning:** Quick adaptation to new mechanics

## Local Setup
```bash
# .env configuration
OPERATION_MODE=offline
ENVIRONMENTS_DIR=environment_files

# Downloaded games
environment_files/
├── ls20/cb3b57cc/  # Locksmith
├── vc33/9851e02b/  # Budget logic
└── ft09/9ab2447a/  # Pattern matching
```

## Links
- [Top level](../PROGRESS.md)
- [Full mechanics doc](../ARC-AGI3-MECHANICS.md)
- [Training validation](training-validation.md)
