# Building ARC-AGI-3 Agents

## Agent Architecture

Every ARC-AGI-3 agent must implement two core methods:

1. **`is_done()`** - Decide when to stop playing
2. **`choose_action()`** - Select the next action

## Base Agent Class

```python
from abc import ABC, abstractmethod
from arcengine import FrameData, GameAction, GameState

class Agent(ABC):
    """Interface for an agent that plays one ARC-AGI-3 game."""

    MAX_ACTIONS: int = 80  # Prevent infinite loops

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Decide if the agent is done playing."""
        raise NotImplementedError

    @abstractmethod
    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose the next action to take."""
        raise NotImplementedError
```

## Simple Random Agent

```python
import random
from arcengine import FrameData, GameAction, GameState

class RandomAgent:
    """An agent that selects random actions."""

    MAX_ACTIONS = 80

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Stop when we win."""
        return latest_frame.state is GameState.WIN

    def choose_action(
        self, frames: list[FrameData], latest_frame: FrameData
    ) -> GameAction:
        """Choose a random action."""
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            # Need to reset to start/restart the game
            return GameAction.RESET

        # Choose a random action (excluding RESET)
        action = random.choice([
            a for a in GameAction if a is not GameAction.RESET
        ])

        # If the action requires coordinates, set them
        if action.is_complex():
            action.set_data({
                "x": random.randint(0, 63),
                "y": random.randint(0, 63),
            })

        return action
```

## Understanding FrameData

Each frame contains important information:

```python
def choose_action(self, frames: list[FrameData], latest_frame: FrameData):
    # Access frame data
    current_state = latest_frame.state
    grid = latest_frame.frame  # 2D list of integers
    completed = latest_frame.levels_completed
    needed_to_win = latest_frame.win_levels
    valid_actions = latest_frame.available_actions

    # Use history
    previous_frames = frames[:-1]
    action_count = len(frames)
```

## Action Types

### Simple Actions

Actions that don't require additional data:

```python
action = GameAction.ACTION1
action.reasoning = "Moving forward"  # Optional explanation
```

### Complex Actions

Actions that require coordinates:

```python
action = GameAction.ACTION1
action.set_data({"x": 10, "y": 20})
action.reasoning = {"action": "click", "reason": "Clicking on target"}
```

### Available Actions

- `GameAction.ACTION1` through `GameAction.ACTION7`
- `GameAction.RESET` - Reset/restart the game

Check if an action is simple or complex:

```python
if action.is_simple():
    # No additional data needed
    pass
elif action.is_complex():
    # Must call action.set_data()
    action.set_data({"x": x, "y": y})
```

## Agent Strategies

### 1. Random Exploration

Good for understanding environment mechanics:

```python
def choose_action(self, frames, latest_frame):
    return random.choice(list(GameAction))
```

### 2. Greedy Strategy

Maximize immediate reward:

```python
def choose_action(self, frames, latest_frame):
    best_action = None
    best_score = -1

    for action in latest_frame.available_actions:
        # Simulate or estimate the outcome
        estimated_score = self.estimate_reward(action, latest_frame)
        if estimated_score > best_score:
            best_score = estimated_score
            best_action = action

    return best_action
```

### 3. LLM-Powered Agent

Use language models for reasoning:

```python
import openai

def choose_action(self, frames, latest_frame):
    # Convert frame to text description
    state_description = self.describe_state(latest_frame)

    # Ask LLM for action
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[{
            "role": "user",
            "content": f"Given this game state: {state_description}\n"
                      f"Available actions: {latest_frame.available_actions}\n"
                      f"What action should I take and why?"
        }]
    )

    # Parse and return action
    return self.parse_action(response.choices[0].message.content)
```

## Running Your Agent

### Using the Official Framework

```bash
# Clone the agents repository
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents

# Run your agent
uv run main.py --agent=random --game=ls20
```

### Standalone Script

```python
import arc_agi
from arcengine import GameAction, GameState

class MyAgent:
    def run(self, game_id: str):
        arc = arc_agi.Arcade()
        env = arc.make(game_id)

        while True:
            frame = env.observation_space
            if frame.state == GameState.WIN:
                break

            action = self.choose_action(frame)
            env.step(action)

        return arc.get_scorecard()
```

## Testing Your Agent

### Unit Tests

```python
import pytest
from arcengine import FrameData, GameState

def test_agent_handles_not_played():
    agent = MyAgent()
    frame = FrameData(state=GameState.NOT_PLAYED)

    action = agent.choose_action([], frame)
    assert action == GameAction.RESET

def test_agent_stops_on_win():
    agent = MyAgent()
    frame = FrameData(state=GameState.WIN)

    assert agent.is_done([], frame) is True
```

### Run Tests

```bash
pytest tests/
```

## Observability with AgentOps

Track your agent's behavior:

```bash
# Install agentops
uv sync --extra agentops

# Set API key
export AGENTOPS_API_KEY=your_key_here
```

The `@trace_agent_session` decorator automatically logs:
- Actions taken
- Frame states
- Performance metrics

## Competition Submission

1. Ensure your agent works with the official framework
2. Test on all available games
3. Submit via: https://forms.gle/wMLZrEFGDh33DhzV9

## Best Practices

1. **Handle all game states** - NOT_PLAYED, PLAYING, WIN, GAME_OVER
2. **Respect MAX_ACTIONS** - Prevent infinite loops
3. **Log your reasoning** - Use `action.reasoning` for debugging
4. **Test extensively** - Use recordings for reproducibility
5. **Optimize for efficiency** - Fewer actions = better score

## Agent Templates

Check the `agents-reference/agents/templates/` directory for:

- `random_agent.py` - Basic random action agent
- `llm_agents.py` - LLM-powered agents
- `reasoning_agent.py` - Multi-step reasoning
- `multimodal.py` - Vision-capable agents
- `langgraph_*.py` - LangChain/LangGraph integration
- `smolagents.py` - HuggingFace integration

## Resources

- [Official Documentation](https://docs.arcprize.org/)
- [API Guide](API-GUIDE.md)
- [ARC-AGI-3 Overview](ARC-AGI-3-OVERVIEW.md)
