# ARC-AGI-3 Overview

## What is ARC-AGI?

The **Abstraction and Reasoning Corpus (ARC)** is a benchmark created by François Chollet (creator of Keras) to measure artificial general intelligence. It's designed to be "easy for humans, hard for AI" - testing the fundamental gaps in AI's reasoning and adaptability.

### Why ARC Matters

Current AI systems excel at pattern matching and memorization but struggle with:
- Novel problem-solving
- Generalizing from few examples
- Understanding abstract concepts
- Adapting to unseen situations

ARC directly tests these capabilities by presenting tasks that require **fluid intelligence** - the ability to solve problems you've never seen before.

## Evolution of ARC

### ARC-AGI-1 (Original, 2019)
- **Format**: Static 2D grid transformations
- **Tasks**: 400 training + 400 evaluation tasks
- **Input**: 2-10 example input/output pairs
- **Challenge**: Determine the transformation rule and apply it

### ARC-AGI-2 (2024)
- **Format**: Same as ARC-AGI-1
- **Tasks**: 1,000 training + 120 evaluation tasks
- **Used For**: ARC Prize 2025 competition ($700,000 Grand Prize)
- **Best Score**: 53% (2024)

### ARC-AGI-3 (2026)
- **Format**: Interactive video-game-like environments
- **Tasks**: 1,000+ levels across 150+ environments
- **Key Change**: Agents must explore, learn, and adapt in real-time
- **Scoring**: Based on action efficiency, not just success

## ARC-AGI-3: The Interactive Evolution

### What's Different

Unlike ARC-AGI-1/2 where you analyze static examples, ARC-AGI-3 requires agents to:

1. **Explore Environments** - No documentation provided; learn through interaction
2. **Discover Mechanics** - Understand how the environment works
3. **Set Goals** - Determine what needs to be accomplished
4. **Plan Actions** - Strategize multi-step solutions
5. **Execute Efficiently** - Minimize actions to achieve goals

### Scoring Methodology

The key metric is **action efficiency**:
- The question isn't IF you solve the environment
- It's HOW EFFICIENTLY you solve it
- Compares AI performance to human baseline
- First formal human vs AI action efficiency comparison

### Core Capabilities Tested

| Capability | Description |
|------------|-------------|
| **Exploration** | Discovering environment mechanics through interaction |
| **Planning** | Strategizing multi-step objectives |
| **Memory** | Retaining learned information across episodes |
| **Goal Acquisition** | Understanding what needs to be achieved |
| **Alignment** | Following implicit objectives without explicit instructions |

## Available Games (Preview)

### ls20 - Locksmith
- **Focus**: Conditional interactions with latent state
- **Challenge**: Understanding hidden state changes

### vc33 - Budget Puzzle
- **Focus**: Resource management and puzzle logic
- **Challenge**: Optimizing limited resources

### ft09 - Pattern Matching
- **Focus**: Abstract pattern recognition
- **Challenge**: Identifying and applying patterns

## Getting Started

1. **Play Manually**: Visit https://three.arcprize.org to play games yourself
2. **Build Agents**: Use the `arc-agi` Python package
3. **Submit**: Participate in the competition via the official form

## Resources

- **Official Documentation**: https://docs.arcprize.org/
- **ARC Prize Website**: https://arcprize.org/arc-agi/3/
- **GitHub Repository**: https://github.com/arcprize/ARC-AGI-3-Agents
- **Preview Site**: https://three.arcprize.org/

## Competition Timeline

- **Preview Launch**: January 29, 2026
- **Full Launch**: March 25, 2026
- **Prize Pool**: TBA

## Core Knowledge Priors

ARC-AGI restricts content to **core knowledge priors** - cognitive building blocks present at birth or acquired early with minimal instruction:

- **Objectness** - Understanding that the world contains objects
- **Numbers & Counting** - Basic numeracy
- **Basic Geometry** - Lines, shapes, symmetry
- **Goal-directedness** - Understanding that agents pursue goals
- **Elementary Physics** - Basic concepts like containment, occlusion

This ensures fair comparison by avoiding:
- Language-specific content
- Cultural artifacts
- Specialized domain knowledge
- Task-specific prior knowledge
