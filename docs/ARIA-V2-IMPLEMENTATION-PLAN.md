# ARIA v2 Implementation Plan

## Overview

This document outlines the step-by-step implementation plan for ARIA v2: Language-Guided Meta-Learning architecture.

**Goal:** Build a system that understands games through language reasoning, not end-to-end neural prediction.

**Timeline:** 5 phases, each building on the previous.

---

## Phase 1: Visual Grounding (Days 1-2)

### Objective
Build a module that converts pixel observations to structured entity descriptions.

### Tasks

#### 1.1 Synthetic Game Generator
**File:** `src/aria_v2/pretraining/synthetic_games.py`

Create procedural games with known entity labels:
- Player: single controllable entity
- Goals: level completion triggers
- Items: collectibles (disappear on contact)
- Obstacles: movement blockers
- Triggers: buttons/switches

```python
@dataclass
class SyntheticGameConfig:
    grid_size: int = 32
    num_items: int = 5
    num_obstacles: int = 10
    has_goal: bool = True
    mechanics: list[str] = ["navigation", "collection"]
```

**Deliverable:** Generate 10k labeled game states for training.

#### 1.2 Entity Detector CNN
**File:** `src/aria_v2/visual_grounding.py`

```python
class EntityDetectorCNN(nn.Module):
    """Detect non-background entities in grid."""
    # Input: [B, H, W] color indices
    # Output: [B, H, W] entity mask + [B, N, 6] entity features
```

**Deliverable:** >95% entity detection recall.

#### 1.3 Entity Classifier
**File:** `src/aria_v2/visual_grounding.py`

```python
class EntityClassifier(nn.Module):
    """Classify detected entities: player/goal/item/obstacle/trigger/unknown"""
    # Input: entity crops
    # Output: class probabilities
```

**Deliverable:** >85% classification accuracy.

#### 1.4 Movement Correlator
**File:** `src/aria_v2/visual_grounding.py`

Self-supervised: identify player by what moves when action taken.

```python
def identify_player(obs_t, action, obs_t1) -> tuple[int, int]:
    """Find entity that moved in response to action."""
    diff = find_differences(obs_t, obs_t1)
    # Entity that moved is the player
    return diff.moved_entity_position
```

**Deliverable:** >98% player identification accuracy.

### Validation Checkpoint
- [ ] Entity detection recall >95%
- [ ] Entity classification accuracy >85%
- [ ] Player identification >98%
- [ ] Language description generation works

---

## Phase 2: Event Detection (Days 2-3)

### Objective
Track changes between frames and identify meaningful events.

### Tasks

#### 2.1 State Differencing
**File:** `src/aria_v2/event_detector.py`

```python
class StateDifferencer:
    """Track what changed between frames."""

    def diff(self, prev_entities: dict, curr_entities: dict) -> list[Change]:
        changes = []
        # Position changes
        # Entity appearances/disappearances
        # Property changes (color, state)
        return changes
```

#### 2.2 Event Classifier
**File:** `src/aria_v2/event_detector.py`

Classify changes into semantic events:
- Movement: player position changed
- Collection: item disappeared + player at that position
- Collision: action taken but no movement
- Trigger: interaction caused distant change
- Damage: negative signal (lives, game over)
- Level complete: positive terminal signal

#### 2.3 Cause-Effect Tracker
**File:** `src/aria_v2/event_detector.py`

```python
class CauseEffectTracker:
    """Track temporal correlations to infer causation."""

    def observe(self, action: int, events: list[str]):
        # Build action → event correlation model
        pass

    def infer_rules(self) -> list[str]:
        # "Touching blue diamonds increases score"
        # "Red areas cause damage"
        pass
```

### Validation Checkpoint
- [ ] Correctly detects item collection events
- [ ] Correctly detects collision events
- [ ] Correctly detects level completion
- [ ] Generates reasonable event descriptions

---

## Phase 3: LLM Reasoning (Days 3-4)

### Objective
Integrate LLM for game understanding and planning.

### Tasks

#### 3.1 LLM Setup
**File:** `src/aria_v2/llm_reasoning.py`

```bash
# Download quantized Llama 3.2 1B
# huggingface-cli download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
# Or use smaller model: TinyLlama, Phi-2, etc.
```

```python
from llama_cpp import Llama

llm = Llama(
    model_path="models/llama-3.2-1b.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # Full GPU offload
)
```

#### 3.2 Prompt Engineering
**File:** `src/aria_v2/llm_reasoning.py`

Create structured prompts for:

**Event Interpretation:**
```
You are analyzing a grid puzzle game.
Recent events: [list]
Known rules: [list]
What new rules can you infer? (Be specific, only high-confidence)
```

**Goal Hypothesis:**
```
You are playing a grid puzzle game.
Scene: [description]
Known rules: [list]
What is the likely goal? (One sentence)
```

**Subgoal Generation:**
```
Current scene: [description]
Goal: [hypothesis]
Generate 1-3 immediate subgoals.
Format: "Go to (x,y)" or "Interact with [entity]"
```

#### 3.3 Response Caching
**File:** `src/aria_v2/llm_reasoning.py`

```python
class LLMCache:
    """Cache LLM responses to avoid redundant queries."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}

    def get_or_query(self, prompt: str, llm: Llama) -> str:
        key = hash(prompt)
        if key not in self.cache:
            self.cache[key] = llm(prompt)
        return self.cache[key]
```

#### 3.4 Game Memory
**File:** `src/aria_v2/game_memory.py`

```python
@dataclass
class GameStateMemory:
    rules: list[str]              # Discovered rules
    goal_hypothesis: str          # Current goal belief
    subgoals_pending: list[str]   # Active subgoals
    failed_attempts: list[dict]   # Learning from failure
    entity_knowledge: dict        # Per-entity knowledge
```

### Validation Checkpoint
- [ ] LLM loads and runs on GPU
- [ ] Reasonable rule interpretations
- [ ] Sensible goal hypotheses
- [ ] Actionable subgoal generation
- [ ] Response caching works (<50ms repeated queries)

---

## Phase 4: Subgoal Executor (Days 4-5)

### Objective
Execute LLM-generated subgoals using pretrained navigation.

### Tasks

#### 4.1 Navigation Policy
**File:** `src/aria_v2/subgoal_executor.py`

Simple A*-based navigation:

```python
class NavigationPolicy:
    """Navigate to target avoiding obstacles."""

    def plan_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: set[tuple[int, int]],
    ) -> list[int]:
        """A* pathfinding, return action sequence."""
        pass

    def get_next_action(self, ...) -> int:
        """Get immediate next action."""
        pass
```

#### 4.2 Subgoal Parser
**File:** `src/aria_v2/subgoal_executor.py`

Parse LLM subgoals into navigation targets:

```python
def parse_subgoal(subgoal: str, entities: dict) -> tuple[int, int] | None:
    """
    Parse subgoal string to target position.

    "Go to (5, 3)" → (5, 3)
    "Collect the blue diamond" → position of blue diamond
    "Interact with button" → position of button
    """
    pass
```

#### 4.3 Exploration Mode
**File:** `src/aria_v2/subgoal_executor.py`

When no specific subgoal, explore systematically:

```python
class ExplorationPolicy:
    """Systematic exploration when no subgoal."""

    def __init__(self, grid_size: int):
        self.visited = set()

    def get_exploration_target(self, current_pos: tuple) -> tuple:
        """Return next unvisited position to explore."""
        pass
```

### Validation Checkpoint
- [ ] A* navigation works on test grids
- [ ] Subgoal parsing handles common formats
- [ ] Exploration covers grid systematically
- [ ] >95% navigation success rate

---

## Phase 5: Integration & Evaluation (Days 5-6)

### Objective
Integrate all components and evaluate on ARC-AGI-3.

### Tasks

#### 5.1 ARIAv2Agent
**File:** `src/aria_v2/agent.py`

```python
class ARIAv2Agent:
    """Integrated language-guided agent."""

    def __init__(self, llm_model_path: str):
        self.visual_grounder = VisualGroundingModule()
        self.event_detector = EventDetector()
        self.memory = GameStateMemory()
        self.llm = LLMReasoningEngine(llm_model_path)
        self.executor = SubgoalExecutor()

    def act(self, observation, game_signals) -> tuple[int, dict]:
        # Full pipeline
        pass
```

#### 5.2 Language Trace Logging
**File:** `src/aria_v2/agent.py`

Log all reasoning for debugging:

```python
def log_trace(self, step: int, info: dict):
    """Log language reasoning trace."""
    trace = {
        "step": step,
        "scene": info["scene"],
        "events": info["events"],
        "rules": self.memory.rules,
        "hypothesis": self.memory.goal_hypothesis,
        "subgoal": info["subgoal"],
        "action": info["action"],
    }
    self.trace_log.append(trace)
```

#### 5.3 ARC-AGI-3 Evaluation
**File:** `src/aria_v2/evaluation/arc_evaluator.py`

```python
def evaluate_on_arc(agent: ARIAv2Agent, game_ids: list[str]) -> dict:
    """Evaluate agent on ARC-AGI-3 games."""
    results = {}
    for game_id in game_ids:
        levels, steps, trace = run_episode(agent, game_id)
        results[game_id] = {
            "levels": levels,
            "steps": steps,
            "rules_discovered": len(agent.memory.rules),
            "trace": trace,
        }
    return results
```

### Validation Checkpoint
- [ ] Full pipeline runs on ls20
- [ ] Language traces are interpretable
- [ ] >10% level completion on test games
- [ ] Discovers at least some correct rules

---

## File Structure

```
src/aria_v2/
├── __init__.py
├── config.py
├── visual_grounding.py          # Phase 1
├── event_detector.py            # Phase 2
├── game_memory.py               # Phase 3
├── llm_reasoning.py             # Phase 3
├── subgoal_executor.py          # Phase 4
├── agent.py                     # Phase 5
├── pretraining/
│   ├── __init__.py
│   ├── synthetic_games.py       # Phase 1
│   ├── visual_grounding_trainer.py
│   └── navigation_trainer.py    # Phase 4
└── evaluation/
    ├── __init__.py
    └── arc_evaluator.py         # Phase 5
```

---

## Dependencies

```toml
# Add to pyproject.toml
[project.optional-dependencies]
aria-v2 = [
    "llama-cpp-python>=0.2.0",   # LLM inference
    "opencv-python>=4.8.0",       # Image processing
]
```

**External downloads:**
- Llama 3.2 1B GGUF (~1.2GB): `huggingface-cli download`
- Or alternative: TinyLlama, Phi-2

---

## Success Metrics

| Phase | Metric | Target |
|-------|--------|--------|
| 1 | Entity detection recall | >95% |
| 1 | Player identification | >98% |
| 2 | Event detection accuracy | >90% |
| 3 | LLM latency (cached) | <50ms |
| 4 | Navigation success rate | >95% |
| 5 | ARC level completion | >10% |
| 5 | Rule discovery rate | >50% |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM too slow | Aggressive caching, smaller model, batch queries |
| Visual grounding fails | Fallback to color-based heuristics |
| Wrong goal hypothesis | Update frequently, try alternatives |
| Navigation gets stuck | Exploration mode fallback |

---

## Quick Start Commands

```bash
# Create aria_v2 directory
mkdir -p src/aria_v2/pretraining src/aria_v2/evaluation

# Download LLM (after implementation)
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models/

# Run evaluation (after implementation)
uv run python -m src.aria_v2.evaluation.arc_evaluator --game ls20
```

---

*Plan version: 1.0*
*Estimated effort: 5-6 days*
*Next action: Start Phase 1 - Create synthetic game generator*
