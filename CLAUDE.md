# Claude Instructions for ARC-AGI-3 Project

## Golden Rule

**If it isn't documented, it didn't happen. If it isn't in the plan, don't do it.**

Before ANY work:
1. Read `docs/PROGRESS.md` - What is the current state?
2. Read `docs/current/IMPLEMENTATION-PLAN.md` - What are we supposed to be doing?
3. Verify consistency - Does the code match what's documented?

If documentation is inconsistent or outdated, **fix it first** before proceeding.

---

## Project Context

ARC-AGI-3 competition project. The agent learns game dynamics from human demonstrations via a learned world model.

**Current approach:** VQ-VAE + SmolLM2-360M + LoRA
```
Frame (64x64, 16 colors)
  → VQ-VAE encoder → 64 discrete tokens (512-code codebook)
  → SmolLM2-360M (LoRA) → next-token prediction
  → Surprise + Goal inference (P(LEVEL_COMPLETE)) + Learned policy
```

**Status:** Training complete (88.4% frame acc, 67.9% action acc, 1.8 ppl). Next: agent evaluation.

**Previous approaches (superseded):**
- ARIA v1: BC (80% acc, 0% levels) and PPO (0.18% success) - failed
- Language-guided meta-learning: visual grounding + LLM reasoning - too brittle
- Heuristic abstract learning: rule templates - won't generalize

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

## Code Location

```
src/aria_v2/
├── tokenizer/                  # VQ-VAE frame tokenization
│   ├── frame_tokenizer.py      #   VQ-VAE model (99.85% acc)
│   ├── train_vqvae.py          #   VQ-VAE training script
│   └── trajectory_dataset.py   #   JSONL → tokenized sequences (876K tokens)
│
├── world_model/                # SmolLM2 + LoRA world model
│   ├── config.py               #   All config dataclasses
│   ├── game_transformer.py     #   Model creation (bfloat16, LoRA)
│   ├── train.py                #   Training pipeline (88.4% frame acc)
│   └── agent.py                #   Inference agent (surprise + goal + policy)
│
├── core/                       # Earlier heuristic approach
│   ├── abstract_learner.py     #   Rule learning from observations
│   ├── goal_induction.py       #   Hypothesis testing
│   ├── demonstration_learner.py#   JSONL demo analysis
│   └── agent.py                #   Heuristic agent loop
│
├── pretraining/                # Synthetic training data
│   └── synthetic_games.py
│
├── config.py                   # Old v2 config
├── visual_grounding.py         # Entity detection (100% acc)
└── run_game.py                 # Game runner (arcengine)

checkpoints/
├── vqvae/best.pt               # VQ-VAE checkpoint (~2MB)
└── world_model/best.pt         # SmolLM2 checkpoint (704MB)
```

---

## Documentation Structure

```
docs/
├── PROGRESS.md              # Current state (READ FIRST)
├── current/
│   ├── ARCHITECTURE.md      # VQ-VAE + SmolLM2 architecture
│   └── IMPLEMENTATION-PLAN.md  # What's built + next steps
├── reference/               # Competition docs
│   └── GAME-MECHANICS.md    # ls20, vc33, ft09 analysis
├── findings/                # What we learned
│   └── ARIA-V1-REPORT.md   # v1 experiments & lessons
└── archive/                 # Abandoned approaches
```

---

## Critical Technical Notes

- **bfloat16, not fp16**: SmolLM2 must be loaded in bfloat16. fp16 causes NaN with 49K vocab CE loss.
- **Loss in float32**: Compute cross-entropy outside `torch.amp.autocast` block.
- **PEFT**: embed_tokens and lm_head need manual `requires_grad=True` after applying LoRA.
- **JSONL format**: `data.frame[0]` is the grid, `data.action_input.id` is the action.
- **VQ-VAE EMA**: Must wrap updates in `torch.no_grad()` and detach `z_flat`.

---

## Success Criteria

| Metric | Target | Current |
|--------|--------|---------|
| VQ-VAE pixel accuracy | >95% | 99.85% |
| World model frame prediction | >40% | 88.4% |
| World model action prediction | >30% | 67.9% |
| World model perplexity | <20 | 1.8 |
| Agent level completion | >0 levels | Not yet tested |
| Competition target | >10% levels | Pending |

---

## Specialized Agents

Use these for designated tasks:
- `python-ml-architect` - Code structure, ML patterns
- `multimodal-action-architect` - Architecture decisions
- `rl-training-expert` - Training strategy
- `experiment-manager` - Experiment tracking
- `arc-agi-evaluator` - Approach evaluation

---

## Commands

```bash
# Run agent on game
uv run python -m src.aria_v2.world_model.agent --game ls20

# Train VQ-VAE
uv run python -m src.aria_v2.tokenizer.train_vqvae

# Train world model
uv run python -m src.aria_v2.world_model.train --epochs 30

# Lint
uv run ruff check --fix src/aria_v2/

# VRAM check
nvidia-smi
```
