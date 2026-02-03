# ARC-AGI-3 API Guide

## Installation

### Using UV (Recommended)

```bash
uv init
uv add arc-agi
```

### Using pip

```bash
pip install arc-agi
```

## API Key Setup

1. Get your API key from [three.arcprize.org](https://three.arcprize.org/)
2. Set it as an environment variable:

```bash
export ARC_API_KEY="your-api-key-here"
```

Or use a `.env` file:

```env
ARC_API_KEY=your-api-key-here
```

## Quick Start

### Basic Game Interaction

```python
import arc_agi
from arcengine import GameAction

# Create an arcade instance
arc = arc_agi.Arcade()

# Create an environment for a specific game
env = arc.make("ls20", render_mode="terminal")

# Take some actions
for _ in range(10):
    env.step(GameAction.ACTION1)

# Get the scorecard
print(arc.get_scorecard())
```

### Performance Mode (No Rendering)

Remove `render_mode` to achieve 2,000+ FPS for faster training:

```python
env = arc.make("ls20")  # No render_mode = faster execution
```

## Core Concepts

### Arcade

The `Arcade` class is the main entry point:

```python
arc = arc_agi.Arcade()
```

### Environment

Environments represent individual games:

```python
env = arc.make("game_id", render_mode="terminal")
```

Available render modes:
- `"terminal"` - ASCII art in terminal
- `None` - No rendering (fastest)

### GameAction

Actions the agent can take:

```python
from arcengine import GameAction

# Simple actions (no arguments)
GameAction.ACTION1
GameAction.ACTION2
GameAction.ACTION3
GameAction.ACTION4
GameAction.ACTION5
GameAction.ACTION6
GameAction.ACTION7
GameAction.RESET  # Reset the game

# Complex actions (require coordinates)
action = GameAction.ACTION1
action.set_data({"x": 10, "y": 20})
```

### FrameData

Data returned after each action:

```python
from arcengine import FrameData, GameState

# FrameData fields:
frame.game_id           # Current game ID
frame.frame             # 2D grid representation
frame.state             # GameState enum
frame.levels_completed  # Number of completed levels
frame.win_levels        # Levels needed to win
frame.guid              # Unique identifier
frame.available_actions # List of valid actions
```

### GameState

Current state of the game:

```python
from arcengine import GameState

GameState.NOT_PLAYED   # Game hasn't started
GameState.PLAYING      # Game in progress
GameState.WIN          # Level completed
GameState.GAME_OVER    # Game ended (can restart)
```

## Environment Wrapper

The `EnvironmentWrapper` provides direct access to environments:

```python
from arc_agi import EnvironmentWrapper

# Create wrapper
env = EnvironmentWrapper(game_id="ls20")

# Step through the environment
raw_data = env.step(action, data=action_data, reasoning={})

# Access observation space
observation = env.observation_space
```

## Scorecard

Track performance across games:

```python
# Get overall scorecard
scorecard = arc.get_scorecard()
print(scorecard)

# Scorecard includes:
# - Games played
# - Levels completed
# - Actions taken
# - Efficiency metrics
```

## Operation Modes

### Online Mode (Default)

Uses the ARC-AGI-3 API server:

```python
# Set in .env or environment
OPERATION_MODE=online
```

### Local Mode

Run environments locally for faster execution:

```python
OPERATION_MODE=local
```

Benefits:
- Up to 2,000+ FPS
- No network latency
- Good for training

## Error Handling

```python
from arcengine import FrameData
from pydantic import ValidationError

try:
    frame = env.step(action)
except ValidationError as e:
    print(f"Invalid frame data: {e}")
except ConnectionError as e:
    print(f"API connection failed: {e}")
```

## Logging

Enable debug logging:

```bash
export DEBUG=True
```

Or in code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Recording Gameplay

The framework supports recording and playback:

```python
# Recordings are stored in RECORDINGS_DIR
# Default: recordings/

# Set custom directory in .env
RECORDINGS_DIR=my_recordings
```

## Available Games

| Game ID | Description |
|---------|-------------|
| `ls20` | Locksmith - Conditional interactions |
| `vc33` | Budget puzzle logic |
| `ft09` | Pattern matching |

More games available at [three.arcprize.org](https://three.arcprize.org/)

## Next Steps

- See [Building Agents](BUILDING-AGENTS.md) for creating custom agents
- Check the [official documentation](https://docs.arcprize.org/) for updates
- Join the Discord community for support
