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

ARC-AGI-3 competition project. The agent plays 150+ unknown games with no demos or
instructions, learning rules online during play.

**Current approach (v3.2):** Learned understanding model + TTT + state graph execution
```
(frame_t, action, frame_t+1)
  → CNN transition encoder (0.3ms) → 256-dim embedding
  → Rolling buffer of last 200 embeddings
  → Temporal transformer (15ms every 100 actions) → understanding state
  → Decoder heads → action effects, entity roles, game type, confidence
  → Layer 1: State graph + plan executor (<2ms) → action
  TTT: LoRA on CNN, self-supervised next-frame prediction during play
```

**Status:** v3.1 three-layer agent tested (1 level on vc33). Building learned understanding model with synthetic game pretraining.

**Previous approaches (superseded):**
- v3.1: Three-layer agent (state graph + hand-coded stats + Qwen2.5-7B LLM) — 1 level vc33
- v3: Basic agent (state graph + CNN change predictor) — 1 level vc33
- v3 Dreamer: Transformer world model + imagination — 0 levels, no reward signal
- v2: VQ-VAE + SmolLM2 world model — 0 levels, offline, too slow
- v1: BC/PPO — 0 levels

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
   - Does src/aria_v3/ match what's documented?
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
   - If architecture changed: docs/current/IMPLEMENTATION-PLAN.md
   - If new findings: docs/findings/

3. COMMIT with clear message describing what changed
```

---

## Code Location

```
src/aria_v3/                    # Active development
├── synthetic_games/            # Synthetic game engines (v3.2)
│   ├── base.py                 #   SyntheticGame base class
│   ├── navigation.py           #   Grid navigation game
│   ├── click_puzzle.py         #   Click/toggle game
│   ├── collection.py           #   Navigate + collect
│   └── generate.py             #   Data generation pipeline
│
├── understanding/              # Learned understanding model (v3.2)
│   ├── encoder.py              #   CNN transition encoder (~580K params)
│   ├── temporal.py             #   Temporal transformer (~2M params)
│   ├── decoder.py              #   Understanding decoder heads (~520K params)
│   ├── model.py                #   Full model
│   ├── ttt.py                  #   Test-time training loop
│   ├── train.py                #   Pretraining script
│   └── dataset.py              #   Dataset class
│
├── frame_processor.py          # CCL, hashing, regions
├── state_graph.py              # Graph exploration, dead-end pruning
├── three_layer_agent.py        # v3.1 integrated agent (1 level vc33)
├── learning_engine.py          # v3.1 hand-coded Layer 2
├── reasoning_oracle.py         # v3.1 LLM Layer 3
├── agent.py                    # v3 basic agent (1 level vc33)
├── change_predictor.py         # v3 online CNN
├── world_model.py              # v3 Dreamer (0 levels)
├── dreamer_agent.py            # v3 Dreamer agent (0 levels)
└── pretrain.py                 # Dreamer pretraining

src/aria_v2/                    # Superseded
├── tokenizer/                  # VQ-VAE (99.85% acc, still useful for hashing)
├── world_model/                # SmolLM2 world model (0 levels)
└── ...

checkpoints/
├── vqvae/best.pt               # VQ-VAE checkpoint (~2MB)
├── world_model/best.pt         # SmolLM2 checkpoint (v2, superseded)
└── understanding/              # v3.2 model checkpoints (planned)
```

---

## Documentation Structure

```
docs/
├── PROGRESS.md              # Current state (READ FIRST)
├── current/
│   └── IMPLEMENTATION-PLAN.md  # v3.2 learned understanding plan
├── reference/
│   ├── COMPETITION-RULES.md # ARC-AGI-3 competition ground truth
│   └── GAME-MECHANICS.md    # ls20, vc33, ft09 analysis
├── findings/
│   └── ARIA-V1-REPORT.md   # v1 experiments & lessons
└── archive/
```

---

## Critical Technical Notes

- **bfloat16 everywhere**: Use bfloat16, never fp16 (NaN issues with large vocab CE loss).
- **Action indexing**: Game API uses 1-indexed action IDs (1-7). Do NOT convert to 0-indexed.
- **64x64 frames, 16 colors**: All games use this format. One-hot encoding = [16, 64, 64].
- **Available actions vary per game**: Some games have only click (vc33=[6]), others only directional (ls20=[1,2,3,4]).
- **uv run**: Always use `uv run` for Python commands. VIRTUAL_ENV warning is expected.
- **CUDA multinomial NaN**: Always clamp probs, check for NaN before sampling.

---

## Success Criteria

| Metric | Target | Best So Far |
|--------|--------|-------------|
| Levels completed (3 games) | >3 total | 1 (vc33) |
| Games with >0 levels | >=2 of 3 | 1 of 3 |
| Action speed (average) | <2ms/act | 4.2ms (v3.1) |
| Understanding accuracy (holdout) | >70% | N/A |
| Competition target | >10% levels | Pending |

---

## Specialized Agents

Use these for designated tasks:
- `python-ml-architect` - Code structure, ML patterns
- `multimodal-action-architect` - Architecture decisions
- `rl-training-expert` - Training strategy
- `experiment-manager` - Experiment tracking
- `arc-agi-evaluator` - Approach evaluation
- `devils-advocate` - Challenge proposals before committing

---

## Commands

```bash
# Run v3.1 three-layer agent
uv run python -m src.aria_v3.three_layer_agent --game vc33

# Run v3.1 without LLM (faster baseline)
uv run python -m src.aria_v3.three_layer_agent --game vc33 --no-llm

# Run v3 basic agent
uv run python -m src.aria_v3.agent --game vc33

# Lint
uv run ruff check --fix src/aria_v3/

# VRAM check
nvidia-smi
```
