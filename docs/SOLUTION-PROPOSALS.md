# ARC-AGI-3 Solution Proposals

A comprehensive analysis of solution approaches for the ARC-AGI-3 interactive reasoning benchmark.

---

## Executive Summary

ARC-AGI-3 represents a fundamental shift from static puzzles to interactive reasoning. Unlike previous versions where agents analyzed input/output pairs, ARC-AGI-3 requires agents to:

1. **Explore** environments without documentation
2. **Discover** mechanics through interaction
3. **Plan** multi-step strategies
4. **Execute efficiently** (scoring based on action count vs humans)

This document presents four distinct solution architectures, each with detailed technical specifications.

---

## Table of Contents

1. [Approach 1: Vision-Language-Action (VLA) Model](#approach-1-vision-language-action-vla-model)
2. [Approach 2: Reinforcement Learning with World Models](#approach-2-reinforcement-learning-with-world-models)
3. [Approach 3: Neurosymbolic Program Synthesis](#approach-3-neurosymbolic-program-synthesis)
4. [Approach 4: Neuroscience-Inspired Architectures](#approach-4-neuroscience-inspired-architectures)
5. [Comparative Analysis](#comparative-analysis)
6. [Implementation Recommendations](#implementation-recommendations)

---

## Approach 1: Vision-Language-Action (VLA) Model

### Overview

A multimodal architecture combining visual grid understanding, language-based reasoning, and action prediction in a unified model.

### High-Level Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    VLA Architecture                          │
├─────────────────────────────────────────────────────────────┤
│  [Grid Frame] → [Grid Encoder] → [Temporal Fusion]          │
│                                         ↓                    │
│  [Action History] → [History Encoder] → [Cross-Attention]   │
│                                         ↓                    │
│                              [World Model Head]              │
│                                    ↓                         │
│                    [Policy Head] → [Action + Coordinates]    │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

**1. Grid Encoder (Custom, not pretrained)**
- Input: 2D grid (1x1 to 30x30), values 0-9
- Cell embedding: `nn.Embedding(10, 32)` for color codes
- Positional encoding: Learned 2D sinusoidal
- Architecture: CNN + Transformer hybrid
  - Conv layers for local patterns (shapes, lines)
  - Self-attention for global relationships

**2. Temporal Fusion Module**
- Maintains history of last K frames (K=8-16)
- Learns what changed between frames
- Uses FiLM conditioning for dynamic modulation

**3. World Model**
- Predicts next frame given current state + action
- Enables planning without environment interaction
- Uncertainty estimation for exploration guidance

**4. Action Head**
- Factorized output: Action type → Coordinates (if needed)
- 8 action types (ACTION1-7 + RESET)
- 64x64 coordinate prediction for complex actions

### Training Pipeline

```python
# Phase 1: Self-Play Data Collection
- Run random/curiosity agents across all games
- Store (state, action, next_state, reward) tuples
- Build diverse experience replay buffer

# Phase 2: World Model Pre-training
- Train frame prediction: MSE + perceptual loss
- Learn action-conditioned dynamics
- Estimate prediction uncertainty

# Phase 3: Policy Training
- Actor-Critic with world model rollouts
- Intrinsic motivation: curiosity + disagreement
- Efficiency penalty: -0.01 per action

# Phase 4: Meta-Learning
- MAML-style adaptation across games
- Learn to learn new mechanics quickly
```

### Hardware Requirements

| Component | RTX 4090 (24GB) | A100 (80GB) |
|-----------|-----------------|-------------|
| Model Size | 20-30M params | 100M+ params |
| Batch Size | 64-128 | 256-512 |
| Training FPS | ~2000 | ~5000 |
| Full Training | 24-48 hours | 6-12 hours |

### Strengths
- End-to-end differentiable
- World model enables efficient planning
- Scalable with more compute

### Weaknesses
- Requires significant training data
- World model errors compound in long horizons
- Coordinate prediction is high-dimensional

---

## Approach 2: Reinforcement Learning with World Models

### Overview

A DreamerV3-inspired architecture combining model-based RL with meta-learning for rapid adaptation to new environments.

### High-Level Strategy

```
┌──────────────────────────────────────────────────────────────┐
│                   ARC-Dreamer Architecture                    │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  REAL ENVIRONMENT                    IMAGINATION              │
│  ┌────────────┐                    ┌────────────────┐        │
│  │ Observe    │──→ World Model ──→│ Imagine Futures │        │
│  │ Act        │←── Policy     ←───│ Evaluate Plans  │        │
│  └────────────┘                    └────────────────┘        │
│        ↓                                   ↓                  │
│  Experience Buffer              Imagined Trajectories         │
│        ↓                                   ↓                  │
│        └───────────→ Joint Training ←──────┘                 │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Algorithm: ARC-Dreamer

```
OFFLINE PHASE:
1. Train world model M₀ on human demonstrations
2. Train initial policy π₀ via behavioral cloning
3. Extract environment embeddings for meta-learning

ONLINE META-TRAINING:
For each epoch:
    1. Sample environment e from curriculum
    2. Collect trajectory using π with exploration
    3. Update world model M with new data
    4. Generate imagined trajectories from M
    5. Update policy using imagined returns + intrinsic rewards
    6. MAML outer loop update on meta-parameters
    7. Update curriculum based on performance

TEST-TIME ADAPTATION:
1. Initialize from meta-parameters
2. Explore new environment for K steps
3. Update world model
4. Plan using updated model
5. Execute plan, minimizing action count
```

### Key Components

**1. State Encoder: CNN + Transformer**
```python
class StateEncoder(nn.Module):
    def __init__(self):
        self.cell_embed = nn.Embedding(10, 32)
        self.cnn = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=128, nhead=8),
            num_layers=4
        )
```

**2. Intrinsic Motivation: Ensemble Disagreement + RND + Counts**
```python
intrinsic_reward = (
    0.5 * ensemble_disagreement +  # Epistemic uncertainty
    0.3 * rnd_novelty +            # State novelty
    0.2 * episodic_count_bonus     # Within-episode coverage
)
```

**3. Hierarchical Policy**
- High-level: Sets subgoals every N steps
- Low-level: Executes primitive actions to reach subgoals
- Trained with Hindsight Experience Replay

**4. Factorized Action Space**
```
P(action, x, y | state) = P(action | state) × P(x | state, action) × P(y | state, action, x)
```

### Reward Shaping

```python
total_reward = (
    +10.0 * level_completion +      # Sparse task reward
    -0.01 * action_penalty +        # Efficiency pressure
    +intrinsic_motivation +         # Exploration bonus
    +potential_shaping              # Dense progress signal
)
```

### Training Configuration

```yaml
world_model:
  latent_dim: 256
  hidden_dim: 512
  imagination_horizon: 15

policy:
  learning_rate: 3e-4
  gamma: 0.99
  lambda_gae: 0.95
  entropy_coef: 0.01

training:
  batch_size: 64
  sequence_length: 50
  replay_buffer_size: 1_000_000

exploration:
  intrinsic_coef: 0.1
  intrinsic_decay: 0.9999
```

### Strengths
- Sample efficient via imagination
- Native multi-step planning
- Strong exploration mechanisms

### Weaknesses
- Complex implementation
- World model errors accumulate
- Requires careful hyperparameter tuning

---

## Approach 3: Neurosymbolic Program Synthesis

### Overview

Combine neural perception with symbolic reasoning and LLM-guided program synthesis. This approach explicitly discovers environment rules and synthesizes executable programs.

### High-Level Strategy

```
┌────────────────────────────────────────────────────────────────┐
│                 Neurosymbolic Architecture                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Grid Frame] → [Perception] → [Symbolic State]                │
│                                       ↓                         │
│  [Action History] → [Rule Induction] → [Hypothesis Set]        │
│                                       ↓                         │
│  [Goal Inference] → [Program Synthesis] → [Executable Program] │
│                                       ↓                         │
│  [DSL Interpreter] → [Action Selection] → [Execute]            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Domain-Specific Language (DSL)

```python
# Movement Primitives
move(direction: UP|DOWN|LEFT|RIGHT)
move_to(target: ObjectRef)
move_until(condition: Predicate)

# Object Interaction
interact(target: ObjectRef)
pick_up(target: ObjectRef)
use(item: ObjectRef, target: ObjectRef)

# Control Flow
seq(action1, action2, ...)           # Sequential execution
if_then_else(cond, true_branch, false_branch)
while_do(condition, body)
repeat(n, action)

# Predicates
adjacent_to(obj1, obj2)
color_of(obj) == color
path_exists(from, to)
```

### Key Components

**1. Symbolic Perception**
```python
class SymbolicState:
    objects: List[GridObject]      # Detected objects
    agent_position: Tuple[int,int] # Player location
    relations: List[Relation]      # Spatial relationships

class GridObject:
    color: int
    pixels: Set[Tuple[int,int]]
    bounding_box: Tuple[int,int,int,int]
    shape: str  # 'line', 'rectangle', 'L-shape', etc.
```

**2. Rule Induction Engine**
```python
class RuleInductionEngine:
    def observe(self, prev_state, action, next_state):
        # Extract what changed
        changes = self.diff_states(prev_state, next_state)

        # Generate rule hypotheses
        hypotheses = self.generate_hypotheses(prev_state, action, changes)

        # Update confidence based on evidence
        for h in hypotheses:
            self.update_confidence(h, changes)

        return self.get_confirmed_rules()
```

**3. LLM-Guided Program Synthesis**
```python
def synthesize_program(state, rules, goal):
    prompt = f"""
    Current state: {state.to_description()}
    Known rules: {format_rules(rules)}
    Inferred goal: {goal}

    Synthesize a program in the DSL to achieve the goal.
    Available primitives: {DSL_PRIMITIVES}
    """

    candidates = llm.generate(prompt, n=16, temperature=0.7)

    # Verify each candidate
    valid_programs = []
    for program in candidates:
        if self.verify_program(program, state, rules):
            valid_programs.append(program)

    # Return best by expected efficiency
    return min(valid_programs, key=lambda p: p.estimated_actions)
```

### Project Structure

```
src/arc_solver/
├── perception/
│   ├── connected.py       # Connected component segmentation
│   └── symbolic.py        # Symbolic state representation
├── reasoning/
│   ├── rules.py           # Rule data structures
│   ├── induction.py       # Rule learning from observations
│   └── templates.py       # Hypothesis templates
├── synthesis/
│   ├── dsl.py             # DSL definition
│   ├── interpreter.py     # Program execution
│   ├── llm.py             # LLM integration
│   └── synthesizer.py     # Program synthesis
├── planning/
│   ├── state_graph.py     # State space exploration
│   └── action_planner.py  # Action selection
└── agent/
    └── neurosymbolic.py   # Main agent
```

### Strengths
- Highly interpretable
- Sample efficient (hypothesis-driven)
- Discovered rules transfer across levels

### Weaknesses
- DSL may miss novel mechanics
- LLM latency impacts real-time performance
- Rule explosion in complex environments

---

## Approach 4: Neuroscience-Inspired Architectures

### Overview

Four radical approaches inspired by cognitive science and neuroscience, targeting the fundamental capabilities that make humans good at novel problem-solving.

### 4A: Predictive Processing Engine (PPE)

**Core Insight**: The brain learns from prediction errors, not raw observations.

```
┌─────────────────────────────────────────────────────────────┐
│               Predictive Processing Engine                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [World Model] → [Predict Next State]                       │
│       ↑                    ↓                                 │
│       │           [Prediction Error] ← [Actual State]       │
│       │                    ↓                                 │
│  [Model Update] ← [Precision Weighting]                     │
│                                                              │
│  [Action Selection]: argmax (expected information gain)      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation**: Actions are selected to maximize *expected prediction error* (information gain), not reward. This drives targeted exploration of uncertain mechanics.

### 4B: Hippocampal Replay and Dreaming (HRD)

**Core Insight**: The brain replays and recombines experiences to enable planning without acting.

```
┌─────────────────────────────────────────────────────────────┐
│               Hippocampal Replay System                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Experience Buffer] → [Relational Encoder]                 │
│                              ↓                               │
│                    [Compressed Codes]                        │
│                              ↓                               │
│  [Replay Generator] ← recombine → [Novel Trajectories]      │
│                              ↓                               │
│  [Prefrontal Evaluator] → score against goals               │
│                              ↓                               │
│  [Best Trajectory] → execute first action                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation**: Plan in *compressed relational space*, not raw state space. This enables generalization and efficient mental simulation.

### 4C: Embodied Affordance Network (EAN)

**Core Insight**: Perceive action possibilities (affordances), not features.

```
┌─────────────────────────────────────────────────────────────┐
│               Affordance-First Perception                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Grid Frame] → [Affordance Detector]                       │
│                        ↓                                     │
│  [Affordance Map]: per-location action possibilities         │
│    - can_move_to: 0.9                                        │
│    - blocks_movement: 0.8                                    │
│    - unlocks_something: 0.6                                  │
│                        ↓                                     │
│  [Action = activate most relevant affordance]                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation**: Affordances learned through *sensorimotor contingencies* - detecting reliable action→effect relationships.

### 4D: Oscillatory Binding Machine (OBM) [Speculative]

**Core Insight**: Solve the binding problem through temporal synchrony.

```
┌─────────────────────────────────────────────────────────────┐
│               Oscillatory Binding Machine                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Features] → [Oscillatory Layer]                           │
│                     ↓                                        │
│  Features of same object → synchronized phase               │
│  Features of different objects → different phases            │
│                     ↓                                        │
│  [Synchrony Detection] → emergent object tokens              │
│                     ↓                                        │
│  [Object-Level Transformer] → reasoning                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Key Innovation**: Native compositionality without learning. Objects emerge from synchrony patterns, enabling systematic generalization.

### Synthesis: PARA (Predictive Affordance Replay Architecture)

The most promising path combines insights from all four:

```
┌─────────────────────────────────────────────────────────────┐
│                         PARA                                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [Frame] → [Affordance Detector] → [Affordance State]       │
│                                          ↓                   │
│  [Prediction Engine] → predict affordance changes            │
│                                          ↓                   │
│  [Prediction Error] → learn from surprises                   │
│                                          ↓                   │
│  [Hippocampal Memory] → store compressed experiences         │
│                                          ↓                   │
│  [Replay/Preplay] → simulate in affordance space             │
│                                          ↓                   │
│  [Active Inference] → maximize (goal + information)          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Comparative Analysis

### Approach Comparison Matrix

| Criterion | VLA | RL/World Model | Neurosymbolic | Neuro-Inspired |
|-----------|-----|----------------|---------------|----------------|
| **Sample Efficiency** | Medium | High | Very High | Very High |
| **Interpretability** | Low | Low | Very High | Medium |
| **Generalization** | Medium | Medium | High | Very High |
| **Implementation Complexity** | Medium | High | High | Very High |
| **Compute Requirements** | High | Medium | Low-Medium | Medium |
| **Risk Level** | Low | Medium | Medium | High |

### Scoring by ARC-AGI-3 Capabilities

| Capability | VLA | RL | Neurosymbolic | Neuro-Inspired |
|------------|-----|-----|---------------|----------------|
| Exploration | ★★★☆ | ★★★★ | ★★★☆ | ★★★★★ |
| Planning | ★★★☆ | ★★★★ | ★★★★ | ★★★★ |
| Memory | ★★★☆ | ★★★☆ | ★★★★ | ★★★★★ |
| Goal Acquisition | ★★☆☆ | ★★★☆ | ★★★★ | ★★★★ |
| Action Efficiency | ★★★☆ | ★★★★ | ★★★★★ | ★★★★ |

### Risk Assessment

| Approach | Primary Risk | Mitigation |
|----------|-------------|------------|
| VLA | Training data scarcity | Self-play + data augmentation |
| RL | Sample inefficiency | World model + meta-learning |
| Neurosymbolic | DSL expressiveness | Hybrid with neural fallback |
| Neuro-Inspired | Implementation difficulty | Start simple, iterate |

---

## Implementation Recommendations

### Phase 1: Baseline (Weeks 1-2)
- Implement **Neurosymbolic perception** (connected components, object detection)
- Build **basic DSL interpreter**
- Create **LLM integration** for program synthesis
- Target: Working agent on single game

### Phase 2: Core Architecture (Weeks 3-4)
- Add **world model** for frame prediction
- Implement **rule induction engine**
- Build **replay buffer** and basic RL loop
- Target: Multi-game training

### Phase 3: Advanced Features (Weeks 5-6)
- Add **intrinsic motivation** (curiosity, disagreement)
- Implement **hierarchical planning**
- Add **meta-learning** (MAML or context encoding)
- Target: Competitive performance

### Phase 4: Optimization (Weeks 7-8)
- Profile and optimize inference
- Implement **CUDA graphs** for fast inference
- Add **caching** for LLM calls
- Target: Real-time performance at 2000+ FPS

### Recommended Starting Point

**Primary**: Neurosymbolic + RL Hybrid
- Use symbolic perception for interpretability
- Add world model for efficient planning
- Use LLM for program synthesis when stuck
- Fall back to RL for fine-grained control

```python
class HybridAgent:
    def choose_action(self, state):
        # 1. Symbolic perception
        symbolic_state = self.perceive(state)

        # 2. Check if we have confident rules
        if self.rules.confidence > 0.8:
            # Plan with DSL
            program = self.synthesize_program(symbolic_state)
            return self.interpret(program)

        # 3. Otherwise, explore with RL
        return self.rl_policy.act(state, explore=True)
```

---

## ARC-DREAMER v2: Enhanced RL with World Models

Based on critical evaluation of Approach 2, we have developed **ARC-DREAMER v2** which addresses
all identified weaknesses:

| Original Weakness | v2 Solution | Impact |
|-------------------|-------------|--------|
| 54% error at 15 steps | Ensemble + consistency + grounding every 5 steps | <25% peak error |
| Arbitrary intrinsic weights | Information-theoretic formulation | Principled, adaptive |
| Undefined hierarchy | Object-centric subgoals + options | Clear 3-level structure |
| No goal discovery | Contrastive learning from transitions | Automatic goal identification |
| No hidden state inference | Particle filter belief tracking | Handle POMDPs |
| Uninterpretable latent space | Slot attention + disentanglement | Symbolic grounding |

**Target Score: 9/10** (up from 5.7/10)

See [ARC-DREAMER-V2.md](ARC-DREAMER-V2.md) for complete architecture details.

Implementation available at: `examples/arc_dreamer_v2/`

---

## V2 Evaluation Results and Blockers

### Final Scores

After iterative refinement with expert agents and comprehensive evaluation, the v2 architectures achieved:

| Approach | Score | Strengths | Key Gap |
|----------|-------|-----------|---------|
| **NEUROSYMBOLIC v2** | 6.4/10 | Interpretability, sample efficiency | LLM latency, DSL completeness |
| **ARC-DREAMER v2** | 7.4/10 | World model, intrinsic motivation | Compounding errors, goal discovery |
| **ARIA Hybrid** | 8.4/10 | Dual-system, 100k FPS fast path | Unknown unknowns, cold start |

**None achieved the 9/10 target.**

### Five Fundamental Blockers

These are structural challenges that prevent any current approach from reaching 9/10:

#### 1. Sample Efficiency vs. Generalization Tradeoff
- **Problem**: Learning fast in a new environment (few-shot) while still generalizing to truly novel mechanics
- **Why it's hard**: Meta-learning biases toward seen patterns; pure exploration is too slow
- **Current best**: ARIA's hybrid approach with neural habit learning (~70% of cases)

#### 2. Goal Discovery from Sparse Signals
- **Problem**: Inferring what "winning" means when there's no explicit reward until WIN state
- **Why it's hard**: Many environments have complex win conditions (sequences, patterns, hidden state)
- **Current best**: Contrastive learning from positive/negative transitions (still requires seeing wins)

#### 3. Hidden State Inference with Unknown Cardinality
- **Problem**: Detecting and tracking hidden variables (like lock states in Locksmith) when we don't know how many exist
- **Why it's hard**: Can't enumerate hypotheses without knowing the state space size
- **Current best**: Particle filters with adaptive resampling (struggles with high-dimensional hidden state)

#### 4. Latency vs. Intelligence Tradeoff
- **Problem**: Sophisticated reasoning (LLM, search) is slow; fast reactions miss complex patterns
- **Why it's hard**: ARC-AGI-3 requires both speed (2000+ FPS target) AND complex reasoning
- **Current best**: ARIA's tiered system (neural habits → local model → cloud LLM)

#### 5. Transfer vs. Memorization
- **Problem**: Agents that memorize solutions don't transfer; agents that abstract too much miss details
- **Why it's hard**: The right level of abstraction varies by environment
- **Current best**: Hierarchical representations with multiple granularities

### Evaluator's Recommended Path to 9/10

The arc-agi-evaluator proposed a merged architecture combining the best elements:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED ARCHITECTURE (Target: 9/10)               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  FROM ARIA:                                                          │
│  ├─ Dual-system fast/slow reasoning                                  │
│  ├─ Neural habit cache (100k FPS)                                    │
│  └─ Belief state tracking (RSSM)                                     │
│                                                                      │
│  FROM ARC-DREAMER v2:                                                │
│  ├─ Error-correcting ensemble world model                            │
│  ├─ Information-theoretic intrinsic motivation                       │
│  └─ Object-centric slot attention                                    │
│                                                                      │
│  FROM NEUROSYMBOLIC v2:                                              │
│  ├─ DSL primitives for interpretable plans                           │
│  ├─ Goal inference via contrastive learning                          │
│  └─ Hypothesis-driven exploration                                    │
│                                                                      │
│  MISSING (Required for 9/10):                                        │
│  ├─ Causal intervention for hidden state discovery                   │
│  ├─ Compositional generalization guarantees                          │
│  ├─ Proven sample complexity bounds                                  │
│  ├─ Adaptive abstraction level selection                             │
│  └─ Theoretical framework unifying fast/slow systems                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Conclusion

**The 9/10 target cannot be achieved with current approaches** because:

1. We lack theoretical frameworks that guarantee compositional generalization
2. Hidden state discovery in POMDPs with unknown state cardinality remains unsolved
3. The exploration-exploitation tradeoff in sparse-reward interactive settings has no optimal solution
4. Real-time constraints conflict with the reasoning depth needed for novel mechanics

**Recommended strategy**: Proceed with ARIA Hybrid (8.4/10) as the primary implementation while continuing research on the fundamental blockers. This provides:
- Competitive baseline for competition entry
- Practical system that handles 70-80% of environments well
- Clear failure modes that guide future research

See [ARIA-VARIANTS.md](ARIA-VARIANTS.md) for concrete implementation variants with specific parameter budgets, pretrained models, and data requirements:
- **ARIA-Lite** (7.0/10): Validate core ideas on RTX 4090 with 29M params
- **ARIA-Standard** (8.5/10): Production candidate on RTX 4090 with 75M params
- **ARIA-Max** (9.2/10): Competition-ready on A100 with 213M params

---

## References

1. Hafner et al. (2023). "Mastering Diverse Domains through World Models" (DreamerV3)
2. Chollet (2019). "On the Measure of Intelligence"
3. ARC Prize 2025 Technical Report
4. Jolicoeur-Martineau et al. "Tiny Recursive Reasoning Models"
5. Friston (2010). "The Free-Energy Principle"
6. Gibson (1979). "The Ecological Approach to Visual Perception"
7. Lake et al. (2017). "Building Machines That Learn and Think Like People"
8. Locatello et al. (2020). "Object-Centric Learning with Slot Attention"
9. Bacon et al. (2017). "The Option-Critic Architecture"
10. Pitis et al. (2020). "Maximum Entropy Gain Exploration"

---

*Document generated for ARC-AGI-3 competition preparation. See individual approach sections for detailed implementation guidance.*
