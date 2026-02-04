# ARC-AGI-3 Challenge Environment

A development environment for the ARC-AGI-3 interactive reasoning benchmark, designed to measure AI agents' ability to explore, learn, plan, and adapt in novel environments.

## What is ARC-AGI-3?

ARC-AGI-3 is the first **interactive reasoning benchmark** for AI agents, launching March 25, 2026. Unlike its predecessors (ARC-AGI-1 and ARC-AGI-2) which featured static input/output grid transformations, ARC-AGI-3 presents video-game-like environments where agents must:

- **Explore** - Discover mechanics through interaction without documentation
- **Learn** - Understand rules and patterns from experience
- **Plan** - Strategize multi-step objectives
- **Adapt** - Apply learned knowledge to novel situations

**Key Stats:**
- 1,000+ levels across 150+ environments
- All environments are human-solvable
- Scoring based on **action efficiency** (not just success)

## Requirements

- **Python 3.12+** (required by arc-agi package)
- **uv** package manager (recommended)

## Quick Start

### 1. Set Up Environment

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure Python 3.12+ is used
uv python pin 3.12

# Sync dependencies
uv sync
```

### 2. Configure API Key

Get your API key from [three.arcprize.org](https://three.arcprize.org/) and add it to `.env`:

```bash
cp .env.example .env
# Edit .env and add your ARC_API_KEY
```

### 3. Run Your First Agent

```bash
# Run the basic agent example
uv run python examples/basic_agent.py

# Or run the random agent
uv run python examples/random_agent.py
```

## Directory Structure

```
arc-agi3/
├── README.md                    # This file
├── pyproject.toml               # UV project configuration
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore rules
├── main.py                      # Simple entry point
├── docs/
│   ├── ARC-AGI-3-OVERVIEW.md    # Challenge overview
│   ├── API-GUIDE.md             # Python API documentation
│   ├── BUILDING-AGENTS.md       # Agent development guide
│   ├── SOLUTION-PROPOSALS.md    # Solution approaches & evaluation
│   ├── ARIA-VARIANTS.md         # ARIA implementation variants (Lite/Standard/Max)
│   ├── ARIA-LITE-IMPLEMENTATION.md  # Detailed ARIA-Lite implementation guide
│   ├── ARIA-LITE-PROGRESS.md    # Live progress tracker
│   ├── ARC-DREAMER-V2.md        # ARC-Dreamer v2 architecture
│   ├── NEUROSYMBOLIC-V2-ARCHITECTURE.md  # Neurosymbolic v2 architecture
│   └── HYBRID-ARCHITECTURE.md   # ARIA hybrid architecture
├── src/
│   ├── arc_dreamer_v2/          # ARC-Dreamer v2 implementation
│   ├── arc_neurosymbolic_v2/    # Neurosymbolic v2 implementation
│   └── aria/                    # ARIA hybrid implementation
├── examples/
│   ├── basic_agent.py           # Simple agent example
│   ├── random_agent.py          # Random action agent
│   └── arc_dreamer_demo.py      # ARC-Dreamer v2 demo
└── agents-reference/            # Cloned official ARC-AGI-3-Agents repo
    ├── agents/                  # Agent framework
    │   ├── agent.py             # Base Agent class
    │   ├── swarm.py             # Multi-agent orchestration
    │   └── templates/           # Agent templates
    └── main.py                  # Entry point for running agents
```

## Available Games (Preview)

| Game ID | Description |
|---------|-------------|
| `ls20` | Conditional interactions with latent state |
| `vc33` | Budget-based puzzle logic |
| `ft09` | Pattern matching with abstract mechanics |

## Documentation

- [ARC-AGI-3 Overview](docs/ARC-AGI-3-OVERVIEW.md) - What is ARC-AGI-3 and why it matters
- [API Guide](docs/API-GUIDE.md) - Using the `arc-agi` Python package
- [Building Agents](docs/BUILDING-AGENTS.md) - Creating custom agents

## Official Resources

- **Website**: https://arcprize.org/arc-agi/3/
- **Preview**: https://three.arcprize.org/
- **Documentation**: https://docs.arcprize.org/
- **GitHub**: https://github.com/arcprize/ARC-AGI-3-Agents
- **Discord**: Community discussion and support

## License

This project is for educational and research purposes. See the official ARC Prize website for competition terms.
