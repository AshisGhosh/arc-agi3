# ARIA v1 Experiments Archive

This directory contains archived experiments from the ARIA v1 (end-to-end neural) approach.

## Summary

**Period:** 2026-02-03 to 2026-02-05
**Outcome:** Architecture validated on synthetic tasks, but failed on ARC-AGI-3 games
**Reason for deprecation:** End-to-end neural approach couldn't learn puzzle game mechanics from sparse rewards

## Key Findings

### What Worked
- BC achieves 80% accuracy on human demos
- World model learns dynamics (state_loss=0.033)
- Primitives pretraining validated (nav 100%, click 100%, pattern 100%)
- Meta-learning works on synthetic tasks (FiLM conditioning)

### What Didn't Work
- BC creates loops during evaluation (0 levels)
- PPO with sparse reward: 0.18% success rate (3/1712 episodes)
- Mode collapse: model learns action distribution, not game logic
- Curiosity-driven exploration insufficient for puzzle games

## Experiments

| ID | Goal | Result | Key Learning |
|----|------|--------|--------------|
| EXP-001 | Random BC | 8% | Random data doesn't work |
| EXP-003 | Expert BC | 100% | Simple encoder + expert works |
| EXP-004 | PPO RL | 87% | RL pipeline validated |
| EXP-011 | Full ARIA BC | 79% train, 0 eval | Loops, mode collapse |
| EXP-012 | World Model | 0.033 state_loss | Learns dynamics not goals |
| EXP-013 | PPO on ARC | 0.18% | Sparse reward too hard |

## Files

### Training Scripts (training/)
- `train_aria_bc.py` - BC with ARIA encoder
- `train_ppo_arc.py` - PPO on ARC-AGI-3
- `train_world_model.py` - World model training
- `train_meta.py` - Meta-learning experiments
- Various primitive training scripts

### Checkpoints (../../checkpoints/aria_v1/)
- `aria_bc_simple.pt` - Best BC model (79% accuracy)
- `world_model.pt` - Trained world model
- `ppo_ls20.pt` - PPO checkpoint

## Lessons Learned

1. **End-to-end neural needs lots of data** - We only had 12 human demos
2. **Sparse reward is very hard** - PPO couldn't discover puzzle mechanics
3. **BC learns distribution, not policy** - Same observation can need different actions
4. **Puzzle games need reasoning** - Navigation alone isn't enough for ls20

## Why We Moved to ARIA v2

ARIA v2 uses language-guided reasoning instead of end-to-end neural:
- Visual grounding → Language description
- LLM reasoning → Rule discovery
- Subgoal execution → Pretrained navigation

This allows meta-learning through language understanding rather than gradient-based adaptation.

See `docs/ARIA-V2-ARCHITECTURE.md` for the new approach.
