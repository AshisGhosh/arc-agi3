# ARIA v2: Language-Guided Meta-Learning Architecture

## Executive Summary

ARIA v2 shifts from "predict action from pixels" to "understand games in language, then act." The core insight is that meta-learning requires understanding game mechanics in a transferable representation - natural language.

| Specification | Value |
|---------------|-------|
| **Approach** | Visual grounding → Language description → LLM reasoning → Action |
| **Key Innovation** | Pretrained visual concepts + runtime language reasoning |
| **Target** | Learn new games from observation, not massive data |

---

## Core Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OBSERVATION (64x64 grid)                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  1. VISUAL GROUNDING MODULE (Pretrained)                                 │
│  ────────────────────────────────────────                                │
│  Detect and label game entities:                                         │
│  • Player: "Green square at (5,3)"                                       │
│  • Goals: "Yellow star at (12,8)"                                        │
│  • Items: "Blue diamond at (7,4)"                                        │
│  • Obstacles: "Red walls at (3,5), (3,6), (3,7)"                        │
│  • Triggers: "Button at (9,2)"                                           │
│                                                                          │
│  Output: Structured scene description (language)                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. EVENT DETECTOR                                                       │
│  ─────────────────                                                       │
│  Track changes between frames:                                           │
│  • "Player moved from (5,3) to (5,4)"                                   │
│  • "Blue diamond at (7,4) DISAPPEARED"                                   │
│  • "Score changed: 0 → 1"                                               │
│  • "Door at (10,5) OPENED"                                              │
│                                                                          │
│  Output: Event log with cause-effect relationships                       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. GAME STATE MEMORY                                                    │
│  ────────────────────                                                    │
│  Maintain understanding of:                                              │
│  • Discovered rules: "Blue diamonds are collectibles"                    │
│  • Current hypothesis: "Goal is to collect all diamonds"                 │
│  • Subgoal progress: "Collected 2/5 diamonds"                           │
│  • Failed attempts: "Red areas cause damage"                            │
│                                                                          │
│  Output: Persistent game understanding                                   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. LLM REASONING ENGINE                                                 │
│  ────────────────────────                                                │
│  Input: Scene + Events + Memory                                          │
│                                                                          │
│  Capabilities:                                                           │
│  • Interpret events: "Touching diamond increased score → collectible"    │
│  • Form hypotheses: "This is a collection game with obstacles"          │
│  • Set subgoals: "Navigate to diamond at (7,4), avoid red at (6,4)"    │
│  • Update beliefs: "Buttons seem to open doors"                         │
│                                                                          │
│  Output: Current subgoal + reasoning trace                               │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  5. SUBGOAL EXECUTOR (Pretrained)                                        │
│  ────────────────────────────────                                        │
│  Execute LLM subgoals:                                                   │
│  • "Go to (7,4)" → pathfinding + collision avoidance                    │
│  • "Interact with button" → navigate + action                           │
│  • "Avoid red areas" → constrained navigation                           │
│                                                                          │
│  Output: Low-level actions (UP/DOWN/LEFT/RIGHT/ACT)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Visual Grounding Module

**Purpose:** Convert pixel observations to language descriptions of game entities.

**Pretraining Objectives:**
| Concept | How to Detect | Training Signal |
|---------|---------------|-----------------|
| **Player** | Entity that moves when actions taken | Action→movement correlation |
| **Goal/Exit** | Area that triggers level completion | `levels_completed` signal |
| **Collectibles** | Entities that disappear on contact + score | Entity vanish + reward |
| **Obstacles** | Positions that block movement | Action taken but no movement |
| **Triggers** | Entities whose interaction changes other entities | Cause-effect correlation |
| **Hazards** | Contact causes negative outcome | Lives decrease, game over |

**Architecture:**
```python
class VisualGroundingModule(nn.Module):
    """
    Pretrained module to detect and describe game entities.

    Output format:
    {
        "player": {"pos": (5, 3), "color": "green", "shape": "square"},
        "goals": [{"pos": (12, 8), "color": "yellow", "shape": "star"}],
        "items": [{"pos": (7, 4), "color": "blue", "shape": "diamond"}],
        "obstacles": [{"pos": (3, 5), "color": "red", "shape": "wall"}],
        "triggers": [{"pos": (9, 2), "color": "gray", "shape": "button"}],
    }
    """

    def __init__(self, grid_size: int = 64, num_colors: int = 16):
        super().__init__()
        # Entity detection CNN
        self.entity_detector = EntityDetectorCNN(num_colors)
        # Entity classifier (player, goal, item, obstacle, trigger, hazard)
        self.entity_classifier = EntityClassifier(num_classes=6)
        # Spatial descriptor
        self.spatial_encoder = SpatialEncoder(grid_size)

    def forward(self, observation: torch.Tensor) -> dict:
        # Detect entities
        entity_mask = self.entity_detector(observation)
        # Classify each entity
        classifications = self.entity_classifier(observation, entity_mask)
        # Generate spatial descriptions
        descriptions = self.spatial_encoder(classifications)
        return descriptions

    def to_language(self, entities: dict) -> str:
        """Convert structured entities to natural language."""
        lines = []
        if entities["player"]:
            p = entities["player"]
            lines.append(f"Player ({p['color']} {p['shape']}) at {p['pos']}")
        for item in entities["items"]:
            lines.append(f"Item ({item['color']} {item['shape']}) at {item['pos']}")
        for goal in entities["goals"]:
            lines.append(f"Goal ({goal['color']} {goal['shape']}) at {goal['pos']}")
        for obs in entities["obstacles"]:
            lines.append(f"Obstacle ({obs['color']}) at {obs['pos']}")
        return ". ".join(lines)
```

**Pretraining Data:**
- Synthetic games with known entity labels
- Self-supervised: predict what moves when action taken
- Reward correlation: what entity contact causes score change

---

### 2. Event Detector

**Purpose:** Track state changes and identify cause-effect relationships.

**Event Types:**
| Event | Detection | Example |
|-------|-----------|---------|
| **Movement** | Player position changed | "Player moved UP to (5,4)" |
| **Collection** | Item disappeared + reward | "Collected blue diamond, score +1" |
| **Unlock** | Obstacle state changed | "Door opened after button press" |
| **Damage** | Lives/health decreased | "Hit red area, lives 3→2" |
| **Level Complete** | levels_completed increased | "Reached goal, level complete!" |
| **Collision** | Action taken but no movement | "Blocked by wall at (3,5)" |

**Architecture:**
```python
class EventDetector:
    """Track changes between frames and identify events."""

    def __init__(self):
        self.prev_state = None
        self.prev_entities = None
        self.event_history = []

    def detect_events(
        self,
        current_entities: dict,
        action_taken: int,
        game_signals: dict,  # levels_completed, score, lives, etc.
    ) -> list[str]:
        events = []

        if self.prev_entities is None:
            self.prev_entities = current_entities
            return events

        # Movement detection
        if current_entities["player"]["pos"] != self.prev_entities["player"]["pos"]:
            old_pos = self.prev_entities["player"]["pos"]
            new_pos = current_entities["player"]["pos"]
            events.append(f"Player moved from {old_pos} to {new_pos}")
        elif action_taken in [1, 2, 3, 4]:  # Movement action but no movement
            events.append(f"Movement blocked (action {action_taken})")

        # Item collection detection
        prev_items = {tuple(i["pos"]) for i in self.prev_entities["items"]}
        curr_items = {tuple(i["pos"]) for i in current_entities["items"]}
        collected = prev_items - curr_items
        for pos in collected:
            events.append(f"Item at {pos} collected/disappeared")

        # Score/reward changes
        if game_signals.get("score_delta", 0) > 0:
            events.append(f"Score increased by {game_signals['score_delta']}")
        if game_signals.get("score_delta", 0) < 0:
            events.append(f"Score decreased by {abs(game_signals['score_delta'])}")

        # Level completion
        if game_signals.get("level_completed", False):
            events.append("LEVEL COMPLETED!")

        # Lives/damage
        if game_signals.get("lives_delta", 0) < 0:
            events.append(f"Took damage, lives decreased")

        self.prev_entities = current_entities
        self.event_history.extend(events)
        return events
```

---

### 3. Game State Memory

**Purpose:** Maintain persistent understanding of discovered game rules and progress.

**Memory Structure:**
```python
@dataclass
class GameStateMemory:
    """Persistent memory of game understanding."""

    # Discovered rules (high confidence observations)
    rules: list[str] = field(default_factory=list)
    # Examples:
    # - "Blue diamonds are collectibles (score +1 each)"
    # - "Red areas cause damage"
    # - "Gray buttons open doors"

    # Current hypothesis about game goal
    goal_hypothesis: str = ""
    # Example: "Collect all diamonds and reach the yellow star"

    # Subgoal progress
    subgoals_completed: list[str] = field(default_factory=list)
    subgoals_pending: list[str] = field(default_factory=list)

    # Failed attempts (for learning)
    failed_actions: list[dict] = field(default_factory=list)
    # Example: {"action": "went to (6,4)", "result": "took damage from red area"}

    # Entity memory (what we know about specific entities)
    entity_knowledge: dict[tuple, str] = field(default_factory=dict)
    # Example: {(7, 4): "blue diamond, collectible", (10, 5): "door, needs key"}

    def add_rule(self, rule: str, confidence: float = 1.0):
        if confidence > 0.7 and rule not in self.rules:
            self.rules.append(rule)

    def update_hypothesis(self, new_hypothesis: str):
        self.goal_hypothesis = new_hypothesis

    def to_context(self) -> str:
        """Convert memory to context string for LLM."""
        context = []
        if self.rules:
            context.append("Known rules:\n" + "\n".join(f"- {r}" for r in self.rules))
        if self.goal_hypothesis:
            context.append(f"Current goal: {self.goal_hypothesis}")
        if self.subgoals_pending:
            context.append("Pending subgoals:\n" + "\n".join(f"- {s}" for s in self.subgoals_pending))
        if self.failed_actions:
            recent_failures = self.failed_actions[-3:]
            context.append("Recent failures:\n" + "\n".join(f"- {f}" for f in recent_failures))
        return "\n\n".join(context)
```

---

### 4. LLM Reasoning Engine

**Purpose:** Interpret observations, form hypotheses, and set actionable subgoals.

**Capabilities:**
1. **Event Interpretation:** "Player touched blue diamond → diamond disappeared → score increased" → "Blue diamonds are collectibles"
2. **Goal Hypothesis:** Based on game elements, hypothesize the win condition
3. **Subgoal Generation:** Break down goal into executable steps
4. **Belief Update:** Refine understanding based on new evidence

**Architecture:**
```python
class LLMReasoningEngine:
    """
    LLM-based reasoning for game understanding and planning.

    Uses structured prompts to:
    1. Interpret events and form rules
    2. Hypothesize game goals
    3. Generate actionable subgoals
    """

    def __init__(self, model_path: str):
        from llama_cpp import Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1,
        )
        self.response_cache = {}

    def interpret_events(
        self,
        events: list[str],
        memory: GameStateMemory,
    ) -> list[str]:
        """Interpret events to discover rules."""

        prompt = f"""You are analyzing a grid-based puzzle game.

Recent events:
{chr(10).join(f"- {e}" for e in events[-10:])}

{memory.to_context()}

Based on these events, what rules can you infer about the game mechanics?
List only HIGH CONFIDENCE rules (things you observed directly).

Rules (one per line):"""

        response = self._query(prompt, max_tokens=200)
        rules = [line.strip("- ") for line in response.strip().split("\n") if line.strip()]
        return rules

    def hypothesize_goal(
        self,
        scene_description: str,
        memory: GameStateMemory,
    ) -> str:
        """Generate hypothesis about game goal."""

        prompt = f"""You are analyzing a grid-based puzzle game.

Current scene:
{scene_description}

{memory.to_context()}

What is the likely goal of this game? Be specific but concise.

Goal:"""

        response = self._query(prompt, max_tokens=100)
        return response.strip()

    def generate_subgoals(
        self,
        scene_description: str,
        memory: GameStateMemory,
    ) -> list[str]:
        """Generate actionable subgoals to achieve the hypothesized goal."""

        prompt = f"""You are playing a grid-based puzzle game.

Current scene:
{scene_description}

{memory.to_context()}

Generate 1-3 immediate subgoals (specific, actionable steps).
Format: "Go to (x,y)" or "Interact with [entity]" or "Avoid [hazard]"

Subgoals:"""

        response = self._query(prompt, max_tokens=150)
        subgoals = [line.strip("- 1234567890.") for line in response.strip().split("\n") if line.strip()]
        return subgoals[:3]

    def _query(self, prompt: str, max_tokens: int = 100) -> str:
        """Query LLM with caching."""
        cache_key = hash(prompt)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]

        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.3,
            stop=["\n\n", "---"],
        )
        result = response["choices"][0]["text"]
        self.response_cache[cache_key] = result
        return result
```

---

### 5. Subgoal Executor

**Purpose:** Execute LLM-generated subgoals using pretrained navigation skills.

**Pretrained Skills:**
| Skill | Input | Output |
|-------|-------|--------|
| **Navigate to position** | target (x, y) | Sequence of UP/DOWN/LEFT/RIGHT |
| **Avoid obstacles** | obstacle positions | Path that avoids |
| **Interact with entity** | entity position | Navigate + ACT |
| **Explore area** | region bounds | Systematic coverage |

**Architecture:**
```python
class SubgoalExecutor:
    """
    Execute high-level subgoals using pretrained navigation.

    Pretrained on synthetic navigation tasks:
    - Point-to-point navigation
    - Obstacle avoidance
    - Entity interaction
    """

    def __init__(self, grid_size: int = 64):
        self.grid_size = grid_size
        # Pretrained navigation policy (simple, fast)
        self.nav_policy = PretrainedNavigationPolicy(grid_size)
        # Current subgoal
        self.current_subgoal = None
        self.subgoal_target = None

    def set_subgoal(self, subgoal: str, entities: dict):
        """Parse subgoal and set navigation target."""
        self.current_subgoal = subgoal

        # Parse "Go to (x, y)" format
        if "go to" in subgoal.lower():
            import re
            match = re.search(r'\((\d+),\s*(\d+)\)', subgoal)
            if match:
                self.subgoal_target = (int(match.group(1)), int(match.group(2)))
                return

        # Parse "Go to [entity]" format
        if "diamond" in subgoal.lower():
            for item in entities.get("items", []):
                if "diamond" in item.get("shape", "").lower():
                    self.subgoal_target = item["pos"]
                    return

        # Parse "Interact with button" format
        if "button" in subgoal.lower():
            for trigger in entities.get("triggers", []):
                if "button" in trigger.get("shape", "").lower():
                    self.subgoal_target = trigger["pos"]
                    return

        # Default: explore
        self.subgoal_target = None

    def get_action(
        self,
        player_pos: tuple[int, int],
        obstacles: list[tuple[int, int]],
    ) -> int:
        """Get next action to achieve current subgoal."""

        if self.subgoal_target is None:
            # Exploration mode: random walk avoiding obstacles
            return self._explore_action(player_pos, obstacles)

        # Navigation mode: move toward target
        return self.nav_policy.get_action(
            player_pos,
            self.subgoal_target,
            obstacles,
        )

    def subgoal_achieved(self, player_pos: tuple[int, int]) -> bool:
        """Check if current subgoal is achieved."""
        if self.subgoal_target is None:
            return False
        # Within 1 cell of target
        return abs(player_pos[0] - self.subgoal_target[0]) <= 1 and \
               abs(player_pos[1] - self.subgoal_target[1]) <= 1

    def _explore_action(self, player_pos: tuple, obstacles: list) -> int:
        """Random exploration avoiding obstacles."""
        import random
        valid_actions = []
        for action, (dx, dy) in enumerate([(0,0), (0,-1), (0,1), (-1,0), (1,0)]):
            new_pos = (player_pos[0] + dx, player_pos[1] + dy)
            if new_pos not in obstacles and 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
                valid_actions.append(action)
        return random.choice(valid_actions) if valid_actions else 0


class PretrainedNavigationPolicy:
    """Simple A* or learned navigation policy."""

    def __init__(self, grid_size: int):
        self.grid_size = grid_size

    def get_action(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        obstacles: list[tuple[int, int]],
    ) -> int:
        """Get action to move from start toward goal avoiding obstacles."""
        # Simple greedy: move in direction that reduces distance
        dx = goal[0] - start[0]
        dy = goal[1] - start[1]

        # Prefer larger delta
        if abs(dx) > abs(dy):
            # Try horizontal first
            if dx > 0 and (start[0] + 1, start[1]) not in obstacles:
                return 4  # RIGHT
            elif dx < 0 and (start[0] - 1, start[1]) not in obstacles:
                return 3  # LEFT

        # Try vertical
        if dy > 0 and (start[0], start[1] + 1) not in obstacles:
            return 2  # DOWN
        elif dy < 0 and (start[0], start[1] - 1) not in obstacles:
            return 1  # UP

        # Fallback: try any valid move
        for action, (ddx, ddy) in [(4, (1, 0)), (3, (-1, 0)), (2, (0, 1)), (1, (0, -1))]:
            new_pos = (start[0] + ddx, start[1] + ddy)
            if new_pos not in obstacles:
                return action

        return 0  # NOOP
```

---

## Integrated Agent

```python
class ARIAv2Agent:
    """
    Language-Guided Meta-Learning Agent.

    Flow:
    1. Observe → Visual grounding → Scene description
    2. Detect events → Update memory
    3. LLM reasoning → Interpret, hypothesize, plan
    4. Execute subgoal → Low-level actions
    """

    def __init__(
        self,
        llm_model_path: str,
        grid_size: int = 64,
    ):
        self.visual_grounder = VisualGroundingModule(grid_size)
        self.event_detector = EventDetector()
        self.memory = GameStateMemory()
        self.llm = LLMReasoningEngine(llm_model_path)
        self.executor = SubgoalExecutor(grid_size)

        self.reasoning_interval = 10  # Reason every N steps
        self.step_count = 0

    def reset(self):
        """Reset for new game."""
        self.event_detector = EventDetector()
        self.memory = GameStateMemory()
        self.executor.current_subgoal = None
        self.step_count = 0

    def act(
        self,
        observation: torch.Tensor,
        game_signals: dict,
        action_taken: int = None,
    ) -> tuple[int, dict]:
        """
        Select action based on observation.

        Returns: (action_id, info_dict)
        """
        self.step_count += 1

        # 1. Visual grounding
        entities = self.visual_grounder(observation)
        scene_description = self.visual_grounder.to_language(entities)

        # 2. Detect events
        events = self.event_detector.detect_events(
            entities, action_taken or 0, game_signals
        )

        # 3. LLM reasoning (periodic or when subgoal achieved)
        should_reason = (
            self.step_count % self.reasoning_interval == 0 or
            self.executor.subgoal_achieved(entities["player"]["pos"]) or
            len(events) > 0  # Something interesting happened
        )

        if should_reason and events:
            # Interpret events
            new_rules = self.llm.interpret_events(events, self.memory)
            for rule in new_rules:
                self.memory.add_rule(rule)

            # Update goal hypothesis
            if not self.memory.goal_hypothesis or "LEVEL COMPLETED" in str(events):
                self.memory.goal_hypothesis = self.llm.hypothesize_goal(
                    scene_description, self.memory
                )

            # Generate subgoals
            subgoals = self.llm.generate_subgoals(scene_description, self.memory)
            if subgoals:
                self.memory.subgoals_pending = subgoals
                self.executor.set_subgoal(subgoals[0], entities)

        # 4. Execute current subgoal
        player_pos = entities["player"]["pos"]
        obstacles = [tuple(o["pos"]) for o in entities.get("obstacles", [])]

        action = self.executor.get_action(player_pos, obstacles)

        return action, {
            "scene": scene_description,
            "events": events,
            "memory": self.memory.to_context(),
            "subgoal": self.executor.current_subgoal,
        }
```

---

## Pretraining Strategy

### Phase 1: Visual Grounding Pretraining

**Data:** Synthetic games with labeled entities
**Tasks:**
1. Entity detection: Find all non-background pixels
2. Entity classification: Player vs item vs obstacle vs goal
3. Movement correlation: What moves when action taken?

**Training:**
```python
# Self-supervised: correlate actions with movement
for episode in synthetic_episodes:
    for obs, action, next_obs in transitions:
        # Predict what entity moved
        moved_entity = detect_movement(obs, next_obs)
        loss = classify_as_player(moved_entity)  # The thing that moved is player

# Supervised: synthetic games with labels
for obs, entity_labels in labeled_data:
    predictions = visual_grounder(obs)
    loss = cross_entropy(predictions, entity_labels)
```

### Phase 2: Navigation Pretraining

**Data:** Simple grid navigation tasks
**Tasks:**
1. Point-to-point navigation
2. Obstacle avoidance
3. Multi-target collection (shortest path)

**Training:**
```python
# Imitation learning from A* demonstrations
for task in navigation_tasks:
    optimal_path = astar(start, goal, obstacles)
    for state, action in optimal_path:
        predicted = nav_policy(state, goal, obstacles)
        loss = cross_entropy(predicted, action)
```

### Phase 3: Event Understanding (Optional LLM fine-tuning)

**Data:** Event sequences paired with rule descriptions
**Tasks:**
- Event → Rule inference
- Scene → Goal hypothesis
- State → Subgoal generation

**Training:**
```python
# Optional: fine-tune LLM on game reasoning
for event_sequence, ground_truth_rule in training_data:
    prompt = format_events(event_sequence)
    generated = llm.generate(prompt)
    loss = compare(generated, ground_truth_rule)
```

---

## Implementation Plan

### File Structure

```
src/aria_v2/
├── __init__.py
├── config.py                    # Configuration
├── visual_grounding.py          # Entity detection and description
├── event_detector.py            # Change tracking
├── game_memory.py               # Persistent game understanding
├── llm_reasoning.py             # LLM-based reasoning
├── subgoal_executor.py          # Low-level action execution
├── agent.py                     # Integrated ARIAv2Agent
├── pretraining/
│   ├── __init__.py
│   ├── visual_grounding_trainer.py
│   ├── navigation_trainer.py
│   └── synthetic_games.py       # Synthetic game generator
└── evaluation/
    ├── __init__.py
    └── arc_evaluator.py         # ARC-AGI-3 evaluation
```

### Build Order

```
1. config.py                    ← No dependencies
2. synthetic_games.py           ← Generate training data
3. visual_grounding.py          ← Core perception
4. pretraining/visual_*         ← Train visual grounding
5. event_detector.py            ← Depends on visual grounding
6. game_memory.py               ← Data structures
7. subgoal_executor.py          ← Navigation policy
8. pretraining/navigation_*     ← Train navigation
9. llm_reasoning.py             ← LLM integration
10. agent.py                    ← Integrate all components
11. evaluation/arc_evaluator.py ← Test on ARC-AGI-3
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Entity detection accuracy | >90% | On synthetic games with labels |
| Navigation success rate | >95% | Point-to-point tasks |
| Rule discovery rate | >70% | Correct rules after 100 steps |
| Goal hypothesis accuracy | >60% | Matches true goal |
| ARC-AGI-3 level completion | >10% | On test games |
| Adaptation speed | <50 steps | Steps to form correct hypothesis |

---

## Key Differences from ARIA v1

| Aspect | ARIA v1 | ARIA v2 |
|--------|---------|---------|
| **Core representation** | Latent vectors | Natural language |
| **Learning** | End-to-end neural | Modular + LLM reasoning |
| **Adaptation** | Gradient-based | In-context (LLM) |
| **Interpretability** | Low | High (language traces) |
| **Data efficiency** | Needs lots of data | Few-shot via reasoning |
| **Fast policy** | MLP habits | Pretrained navigation |
| **Slow policy** | Transformer + planning | LLM reasoning |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| LLM hallucinations | Ground all reasoning in observed events |
| LLM latency | Cache responses, batch queries, reason periodically |
| Visual grounding errors | Use confidence thresholds, cross-validate |
| Goal hypothesis wrong | Update based on new evidence, try alternatives |
| Navigation failures | Fall back to exploration mode |

---

*Document version: 2.0*
*Architecture: Language-Guided Meta-Learning*
*Target: ARC-AGI-3 Competition*
