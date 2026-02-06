# Abstract Learning for ARC-AGI Games

## Overview

This document describes the Abstract Game Learner architecture that discovers game rules and goals from observation without prior assumptions.

## Problem Statement

ARC-AGI games have diverse mechanics:
- Player-navigation games (move character through maze)
- Block-sliding puzzles (push blocks around)
- Pattern-matching games (match shapes)
- Rule-inference games (discover hidden rules)

A general agent must **learn the game mechanics from scratch** rather than assuming a specific type.

## Architecture

### 1. Object Detection Layer

```python
class ObjectDetector:
    def detect_objects(frame) -> list[GameObject]
    def track_objects(prev_objects, curr_objects) -> dict[int, int]
```

- Finds connected components (objects) in frames
- Tracks objects across time using color and proximity
- No assumptions about what objects represent

### 2. Rule Induction Engine

```python
class RuleInducer:
    def observe(prev_frame, action, curr_frame, level_completed, game_over)
    def get_rules() -> list[LearnedRule]
```

Learns rules by observing state transitions:

| Rule Type | Example |
|-----------|---------|
| MOVE | "Action 1 moves Color 12 UP by 5" |
| BLOCKED | "Color 12 blocked by Color 4" |
| COUNTER | "Color 11 decreases by 2 each action" |
| TRANSFORM | "Color A becomes Color B when touched" |
| WIN | "Level completes when X happens" |
| LOSE | "Game over when Color 8 reaches 0" |

### 3. Goal Induction (Hypothesis Testing)

```python
class GoalInducer:
    def analyze_frame(frame) -> list[GoalHypothesis]
    def get_action_toward_goal(frame, goal) -> Optional[int]
    def next_hypothesis() -> Optional[GoalHypothesis]
```

Generates and tests hypotheses about win conditions:
- Structural analysis (small isolated regions are targets)
- Behavioral analysis (what is the agent moving toward?)
- Level-transition analysis (what changes at level completion?)
- UCB-like hypothesis cycling when stuck

### 4. Demonstration Learning

```python
class DemonstrationLearner:
    def load_from_jsonl(jsonl_path) -> Optional[Trajectory]
    def load_from_jsonl_folder(folder_path) -> int
    def get_likely_goals() -> list[tuple[int, float]]
```

Learns goals from human gameplay recordings:
- **Player identification**: Finds which color responds to actions
- **Level-transition analysis**: Detects level boundaries via frame diffs, tracks what colors are near the player at completion
- **Approach patterns**: Tracks net movement direction, identifies colors player moves toward
- **Disappearance tracking**: Colors that vanish at level completion are likely goals
- **Background filtering**: Automatically excludes large (>12% pixel) colors from goal candidates
- **GoalInducer integration**: Exports findings as high-confidence hypotheses

### 5. World Model Simulator

```python
class WorldModelSimulator:
    def simulate(frame, action) -> next_frame
    def plan(start, goal_check, max_depth) -> action_sequence
```

Uses learned rules to:
- Predict future states without taking actions
- Search for action sequences that achieve goals

### 6. Abstract Game Agent

```python
class AbstractGameAgent:
    def act(frame, level_completed, game_over) -> action
```

Phases:
1. **Demonstration**: Learn goals from human recordings (if available)
2. **Exploration**: Systematically try actions to learn rules
3. **Hypothesis**: Generate/refine hypotheses about goals
4. **Planning**: Search for action sequences
5. **Exploitation**: Execute plans

## Results on ls20

### Rule Induction (from agent exploration):
- Color 12 is the controllable object
- Movement rules: UP/DOWN/LEFT/RIGHT by 5 pixels
- Blocking relationships between colors
- Color 11 is a counter (decreases each action)

### Goal Induction (from 12 human demos):
- **Color 9: 45% goal confidence** (top candidate - correct target regions)
- **Color 14: 21%** (disappears on level complete)
- Player identified as Color 12
- Background colors (3, 4) filtered out automatically
- 90 insights extracted from 8711 total frames

### Integration Test:
GoalInducer receives demonstration-bootstrapped hypotheses:
- Color 9 at 95% confidence (top)
- Color 14 at 64%
- Color 8 at 43%

## Key Challenges

### 1. Goal Discovery is Hard (Partially Solved)
Without observing a level completion, we can't infer the win condition.
**Solution**: Demonstration learning analyzes human recordings to bootstrap goal understanding before agent exploration.

### 2. Sparse Rewards
Level completion is a sparse reward signal.
Many actions give no feedback about progress.

### 3. Combinatorial Explosion
With 4 actions and 100+ possible positions, brute-force search is infeasible.
Need heuristics to guide exploration.

## Files

| File | Description |
|------|-------------|
| `src/aria_v2/core/abstract_learner.py` | Rule induction, object detection, world model |
| `src/aria_v2/core/goal_induction.py` | Goal hypothesis generation and testing |
| `src/aria_v2/core/demonstration_learner.py` | Learn goals from human gameplay recordings |
| `src/aria_v2/core/observation_tracker.py` | Change detection |
| `src/aria_v2/core/belief_state.py` | State tracking |

## Usage

### Rule Learning (exploration-based)
```python
from src.aria_v2.core.abstract_learner import AbstractGameAgent

agent = AbstractGameAgent(exploration_budget=16)

while not done:
    action = agent.act(frame, level_completed, game_over)
    frame, reward, done, info = env.step(action)

print(agent.rule_inducer.describe_rules())
```

### Goal Learning (demonstration-based)
```python
from src.aria_v2.core.demonstration_learner import DemonstrationLearner, integrate_with_goal_inducer
from src.aria_v2.core.goal_induction import GoalInducer

# Learn from human demos
learner = DemonstrationLearner()
learner.load_from_jsonl_folder("videos/ARC-AGI-3 Human Performance/ls20")
print(learner.describe())

# Bootstrap goal inducer
inducer = GoalInducer()
integrate_with_goal_inducer(learner, inducer)
# inducer now has high-confidence hypotheses about game goals
```

## Conclusion

The Abstract Game Learner successfully discovers game mechanics without prior assumptions. Goal discovery, previously the hardest challenge, is now addressable through demonstration learning:

| Component | Approach | Data Needed | Status |
|-----------|----------|-------------|--------|
| Rules | Observation (~20 samples) | Agent exploration | Working |
| Goals | Demonstration analysis | Human recordings | Working |
| Planning | World model + BFS | Learned rules + goals | Prototype |

The key insight is that **rules and goals require different learning strategies**:
- **Rules**: Inferrable from (state, action, next_state) triples
- **Goals**: Require success signals, best obtained from demonstrations

This architecture provides a foundation for combining rule learning with goal inference to build general game-playing agents.
