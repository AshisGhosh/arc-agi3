# ARC-AGI-3 Competition Rules & Understanding

**This is the ground truth document. All architecture decisions must be justified against these constraints.**

---

## The Challenge

The agent is dropped into an unknown video game with **no instructions, no demos, no rules**. It must:
1. Explore to discover what the game is
2. Learn the rules from interaction
3. Complete levels efficiently (fewer actions = higher score)

**150+ games** in the full benchmark, each hand-crafted and novel. The agent cannot memorize its way to success.

---

## Observation Space

Each `env.step()` returns:

| Field | Type | Description |
|-------|------|-------------|
| `frame` | `list[list[list[int]]]` | 64x64 grid, values 0-15 (16 colors). Origin (0,0) = top-left |
| `state` | `GameState` | `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER` |
| `levels_completed` | `int` | How many levels completed so far (0-254) |
| `win_levels` | `int` | Total levels needed to win |
| `available_actions` | `list[int]` | Which action IDs (1-7) work for this game. **Fixed per game, not per step** |
| `game_id` | `str` | Game identifier |

**What the agent knows before playing:**
- Which actions are available (e.g., `[1,2,3,4]` for navigation, `[6]` for click-only)
- How many levels to complete
- The game ID

**What the agent does NOT know:**
- What the actions do
- What the colors mean
- What the goal is
- Any training examples or demonstrations

---

## Action Space

| Action | ID | Type | Data |
|--------|----|------|------|
| RESET | 0 | Simple | None. Resets current level (or full game if at start/after win) |
| ACTION1-5 | 1-5 | Simple | None. Game-specific (directional, select, rotate, etc.) |
| ACTION6 | 6 | Complex | `{"x": int, "y": int}` where x,y in [0,63]. Click at pixel coordinate |
| ACTION7 | 7 | Simple | Undo (disabled during competition) |

- **Simple actions**: Just the ID, no extra data
- **Complex actions** (ACTION6 only): Requires x,y coordinates in the 64x64 display space
- **Effective click space**: 4,096 positions (64x64), but games typically have ~50-100 meaningful click targets (contiguous same-color regions)

**Per-game action spaces (public games):**
| Game | Available | Type | Notes |
|------|-----------|------|-------|
| ls20 | [1,2,3,4] | Navigation | 4 directions, 5px movement per step |
| vc33 | [6] | Click only | Budget-based cell manipulation |
| ft09 | [1,2,3,4,5,6] | Mixed | Directional + click, color cycling |

---

## Scoring: RHAE (Relative Human Action Efficiency)

**Per level:**
```
level_score = min(human_baseline_actions / ai_actions, 1.0) * 100%
```
- Capped at 100% (no bonus for beating humans)
- **Uncompleted levels score 0%**

**Per game:**
```
game_score = average(all level_scores)
```

**Overall:**
```
total_score = sum(all game_scores) / number_of_games
```

**Human baseline**: 2nd-best human from ~10 controlled first-time testers per game.

**Example baselines (public games):**
| Game | Levels | Baseline Actions per Level |
|------|--------|---------------------------|
| ls20 | 7 | [29, 41, 172, 49, 53, 62, 82] |
| ft09 | 6 | [15, 7, 15, 16, 21, 17] |
| vc33 | 7 | [6, 13, 31, 59, 92, 24, 82] |

**Key implication**: The agent is NOT penalized for exploration actions. Only actions *after discovering the solution* count against efficiency. The scoring compares total actions used by the agent to complete a level vs the human baseline.

**However**: Using fewer total actions (including exploration) gives a better score. An agent that completes ls20 level 1 in 100 actions scores `min(29/100, 1) = 29%`. An agent completing it in 29 actions scores 100%.

---

## Compute Constraints

| Resource | Limit |
|----------|-------|
| GPU | RTX 5090 (32GB VRAM) |
| Time | 8 hours for ALL games (150+) |
| API budget | $1,000 if using cloud APIs |

**Time budget per game**: ~8 hours / 150 games = **~3 minutes per game**

**arcengine speed**: 2,000+ FPS without rendering. The bottleneck is the agent's neural network, not the game.

**Action budget**: No hard limit on actions, but more actions = lower score. Preview winner (StochasticGoose) used ~255,964 total actions across 6 games (~42K per game).

**Practical inference budget**: At 200ms/action (our current speed), 3 minutes = 900 actions. At 1ms/action (pure game steps), 3 minutes = 360K actions. **Speed matters enormously.**

---

## What Works (Preview Competition Results)

| Rank | Team | Score | Levels | Approach |
|------|------|-------|--------|----------|
| 1st | StochasticGoose | 12.58% | 18/20 | CNN learns online which actions change frames |
| 2nd | Blind Squirrel | 6.71% | 13/20 | State graph + ResNet18 value model |
| 3rd | Just Explore | ~5% | 12-17/20 | Frame segmentation, no ML |

### StochasticGoose (1st place) - Key Details
- **CNN predicts whether (state, action) causes a frame change**
- 16-channel one-hot encoding of 64x64 grid
- Separate action head + convolutional coordinate head (spatial bias for clicks)
- **Learns online during gameplay** — no pre-training on other games
- Hash-based experience deduplication
- Model reset per level to prevent catastrophic forgetting

### Blind Squirrel (2nd place) - Key Details
- Builds **state graph** from observed frames
- Tracks transitions: (state_hash, action) -> next_state_hash
- Prunes ineffective actions (no frame change = no-op)
- **Shape-based action reduction**: segments frame into contiguous same-color regions, clicks region centers. Reduces 4,096 click positions to ~50-100 targets.
- ResNet18 value model for state evaluation

### Just Explore (3rd place) - Key Details
- **No ML at all** — pure heuristic exploration
- Segments frames into colored regions
- Categorizes regions by "button likelihood"
- Systematic exploration of clickable regions

### What Did NOT Work
- **LLM-based agents**: Vision-based LLM agents showed poor performance
- **World models for planning**: The approach we built. Predicting next frames doesn't directly help discover game rules.
- **Pure random search**: Works on easy games, fails on harder ones
- **Pre-trained models without online adaptation**: Can't generalize to 150+ unseen games

---

## Critical Implications for Our Architecture

### 1. Online Learning is Mandatory
The agent must learn **during play**, not from pre-collected demos. There are no demos for unseen games.

### 2. Speed > Accuracy
At 200ms/action, we get ~900 actions per game in 3 min. The winner used ~42K actions. **We need 1-10ms per action** to be competitive. This rules out full transformer forward passes per step.

### 3. Frame Change Detection is the Key Signal
The winning approach learned "does this action change the frame?" This is the most informative signal for exploration — it distinguishes productive actions from no-ops.

### 4. Click Space Reduction is Critical
4,096 possible click positions is too large to explore randomly. Segmenting the frame into ~50-100 colored regions and clicking region centers is essential.

### 5. State Graphs Prevent Loops
Hashing frames and tracking state transitions prevents the agent from repeating failed actions. Simple but effective.

### 6. Model-per-Level, Not Model-per-Game
Reset the learned model between levels to prevent catastrophic forgetting. Each level may have different rules.

### 7. Pre-training Value is Limited
Pre-training on public games teaches visual patterns (shapes, colors) but NOT game rules. The real learning happens online. Pre-training should be lightweight — a visual encoder, not a full world model.

---

## Our Previous Approach vs What's Needed

| Aspect | What We Built | What's Needed |
|--------|--------------|---------------|
| Learning | Offline (human demos) | Online (during play) |
| Model | 360M param transformer | Lightweight CNN (~1-10M params) |
| Inference | 200ms/action | 1-10ms/action |
| Action selection | Hand-tuned scoring heuristics | Learned frame-change predictor |
| Pre-training | Critical (all learning from demos) | Optional (visual encoder only) |
| Generalization | 3 games (public only) | 150+ unseen games |
| Click handling | VQ cell grid (8x8 = 64 positions) | Convolutional coordinate head or region segmentation |
| State tracking | Token context window | Frame hashing + state graph |

---

## References

- [ARC-AGI-3 Official Page](https://arcprize.org/arc-agi/3/)
- [ARC-AGI-3 Preview: 30-Day Learnings](https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings)
- [Preview Competition](https://arcprize.org/competitions/arc-agi-3-preview-agents/)
- [1st Place: StochasticGoose](https://github.com/DriesSmit/ARC3-solution)
- [2nd Place: Blind Squirrel](https://github.com/wd13ca/ARC-AGI-3-Agents)
- [3rd Place: Just Explore](https://github.com/dolphin-in-a-coma/arc-agi-3-just-explore)
