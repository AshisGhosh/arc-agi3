# ARIA-Lite Implementation Guide

A comprehensive guide for implementing ARIA-Lite, a minimal dual-system architecture for the ARC-AGI-3 challenge.

---

## Executive Summary

ARIA-Lite is a **validation prototype** designed to prove the dual-system hypothesis (fast neural habits + slow deliberate planning) with minimal compute before scaling to ARIA-Standard and ARIA-Max.

| Specification | Value |
|---------------|-------|
| **Total Parameters** | 29M trainable |
| **Training VRAM** | ~7GB (fits RTX 4090) |
| **Expected Score** | 7.0/10 |
| **Purpose** | Validate core architecture |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARIA-Lite Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐   │
│  │ Grid Input   │───►│ GridEncoder  │───►│ State Representation │   │
│  │ [H, W]       │    │ Lite (5M)    │    │ [256 dim]            │   │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘   │
│                                                      │               │
│                      ┌───────────────────────────────┼───────────┐   │
│                      │                               ▼           │   │
│                      │    ┌──────────────────────────────────┐   │   │
│                      │    │         DUAL SYSTEM               │   │   │
│                      │    ├──────────────┬───────────────────┤   │   │
│                      │    │              │                   │   │   │
│                      │    │  ┌────────┐  │  ┌─────────────┐  │   │   │
│                      │    │  │ FAST   │  │  │    SLOW     │  │   │   │
│                      │    │  │ Policy │  │  │   Policy    │  │   │   │
│                      │    │  │ (1M)   │  │  │    (5M)     │  │   │   │
│                      │    │  │        │  │  │             │  │   │   │
│                      │    │  │ MLP    │  │  │ Transformer │  │   │   │
│                      │    │  │ Habits │  │  │ + Planning  │  │   │   │
│                      │    │  └───┬────┘  │  └──────┬──────┘  │   │   │
│                      │    │      │       │         │         │   │   │
│                      │    │      ▼       │         ▼         │   │   │
│                      │    │  confidence  │    uncertainty    │   │   │
│                      │    │      │       │         │         │   │   │
│                      │    └──────┼───────┴─────────┼─────────┘   │   │
│                      │           │                 │             │   │
│                      │           └────────┬────────┘             │   │
│                      │                    ▼                      │   │
│                      │           ┌──────────────┐                │   │
│                      │           │   ARBITER    │                │   │
│                      │           │   (Switch)   │                │   │
│                      │           └──────┬───────┘                │   │
│                      │                  │                        │   │
│                      │                  ▼                        │   │
│                      │              ACTION                       │   │
│                      └───────────────────────────────────────────┘   │
│                                                                      │
│  Supporting Components:                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ World Model  │  │ Belief State │  │ LLM (Llama)  │               │
│  │ 3-head (15M) │  │ RSSM (3M)    │  │ 1B int4      │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. GridEncoderLite (5M params)

**Purpose:** Encode 2D grid observations into a compact latent representation.

**Architecture:**
```
Input: [B, H, W] int tensor (0-15 colors)
    │
    ▼
Color Embedding: 16 colors → 32 dimensions (~512 params)
    │
    ▼
2D Positional Encoding: sinusoidal, 32 dimensions (~4K params)
    │
    ▼
CNN Block 1: Conv3x3(64→128), GroupNorm, GELU, MaxPool2x2 (~75K params)
    │
    ▼
CNN Block 2: Conv3x3(128→256), GroupNorm, GELU, MaxPool2x2 (~300K params)
    │
    ▼
CNN Block 3: Conv3x3(256→256), GroupNorm, GELU (~600K params)
    │
    ▼
Transformer Encoder: 2 layers, 4 heads, 256 dim (~3.2M params)
    │
    ▼
AdaptiveAvgPool2d → Linear(256, 256) (~65K params)
    │
    ▼
Output: [B, 256] latent state

TOTAL: ~4.3M params (within 5M budget)
```

**Key Design Decisions:**
- Uses GroupNorm instead of BatchNorm for small batch compatibility
- 2-layer transformer (vs 4 in full ARIA) for parameter efficiency
- Residual connections in CNN blocks for gradient flow

**Reuse from existing code:**
- Positional encoding pattern from `src/aria/perception.py:52-55`
- CNN architecture inspiration from `src/aria/perception.py:100-150`

---

### 2. World Model (15M params = 3 × 5M)

**Purpose:** Predict next state given current state and action. Ensemble provides uncertainty estimation.

**Architecture per head:**
```
Input: state [256] + action_onehot [8] = [264]
    │
    ▼
Linear(264, 640) + LayerNorm + ReLU
    │
    ▼
Linear(640, 640) + LayerNorm + ReLU
    │
    ▼
├── State Predictor: Linear(640, 256) with residual connection
├── Reward Predictor: Linear(640, 1)
└── Done Predictor: Linear(640, 1) + Sigmoid

Params per head: ~5M
Total (3 heads): ~15M
```

**Uncertainty Estimation:**
```python
def get_uncertainty(state, action):
    predictions = [head(state, action) for head in ensemble]
    mean = torch.stack(predictions).mean(dim=0)
    variance = torch.stack(predictions).var(dim=0)
    uncertainty = variance.mean()  # Epistemic uncertainty
    return mean, uncertainty
```

**Adaptation from existing code:**
- Base implementation: `src/arc_dreamer_v2/world_model.py:68` (EnsembleWorldModel)
- Changes: `num_ensemble=3` (from 5), `hidden_dim=640` (from 512)

---

### 3. Belief State Tracker (3M params)

**Purpose:** Maintain belief over hidden state using particle filtering (RSSM-style).

**Architecture:**
```
Hidden Transition Model:
    Input: prev_belief [64] + action_onehot [8] = [72]
    → Linear(72, 128) + ReLU + Linear(128, 64)
    → Output: belief_delta [64]

Observation Model:
    Input: belief [64] + observation [256] = [320]
    → Linear(320, 128) + ReLU + Linear(128, 64)
    → Output: likelihood [64]

Particle Filter:
    - num_particles: 50
    - resampling: systematic when ESS < threshold
```

**Key Operations:**
```python
def update(action, observation):
    # 1. Predict: propagate particles through transition model
    predicted_particles = transition_model(particles, action)

    # 2. Update: weight particles by observation likelihood
    weights = observation_model(predicted_particles, observation)

    # 3. Resample if effective sample size is low
    if effective_sample_size(weights) < threshold:
        particles = resample(predicted_particles, weights)

    # 4. Return mean belief
    return particles.mean(dim=0)
```

**Reuse directly:** `src/arc_dreamer_v2/belief_tracking.py` (BeliefStateTracker class)

---

### 4. Fast Policy (1M params)

**Purpose:** Quick, habitual action selection for familiar situations.

**Architecture:**
```
Input: state [256] (from encoder)
    │
    ▼
Compress: AdaptiveAvgPool2d(8,8) → Flatten → Linear(256*64, 128) → GELU
    │
    ├── Action Head: Linear(128, 128) → GELU → Linear(128, 8)
    │       → action_logits [8]
    │
    ├── Confidence Head: Linear(128, 64) → GELU → Linear(64, 1) → Sigmoid
    │       → confidence [1]  (used by arbiter)
    │
    └── Coordinate Heads (factorized):
            P(x|state,action): Linear(128+32, 128) → GELU → Linear(128, 64)
            P(y|state,action,x): Linear(128+32+64, 128) → GELU → Linear(128, 64)

TOTAL: ~0.8M params (within 1M budget)
```

**Inference:** <0.1ms on GPU (>10k FPS)

**Reuse from existing code:**
- Full implementation: `src/aria/policy.py:19` (FastPolicy class)
- Configure with `policy_hidden_dim=128` for parameter budget

---

### 5. Slow Policy (5M params) [NEW]

**Purpose:** Deliberate reasoning with lookahead for novel/difficult situations.

**Architecture:**
```
Input: state [256] + belief [64] + goal [64] = [384]
    │
    ▼
Input Projection: Linear(384, 256)
    │
    ▼
Transformer Encoder:
    - 4 layers
    - 4 attention heads
    - 256 hidden dimension
    - 512 feedforward dimension
    - dropout: 0.1
    (~4.2M params)
    │
    ▼
├── Policy Head: Linear(256, 8) → action_logits
├── Value Head: Linear(256, 1) → state value
└── Uncertainty Head: Linear(256, 1) → Sigmoid → planning uncertainty

TOTAL: ~4.5M params (within 5M budget)
```

**Key Differences from Fast Policy:**
- Takes belief state as input (handles partial observability)
- Takes goal embedding as input (goal-conditioned)
- Includes uncertainty estimation for planning
- Can be combined with MCTS for tree search

---

### 6. Arbiter (Metacognitive Switcher) [NEW]

**Purpose:** Decide when to use fast vs slow system.

**Logic:**
```python
class Arbiter:
    def __init__(
        self,
        confidence_threshold: float = 0.7,
        uncertainty_threshold: float = 0.3,
        novelty_threshold: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.novelty_threshold = novelty_threshold

    def should_use_slow(
        self,
        fast_confidence: float,      # From fast policy
        world_model_uncertainty: float,  # From ensemble disagreement
        belief_entropy: float,       # From particle filter
        novelty_score: float,        # From state hashing
    ) -> bool:
        """
        Use slow system when ANY of:
        1. Fast policy is not confident
        2. World model is uncertain (novel dynamics)
        3. Belief state has high entropy (hidden state unclear)
        4. State is novel (not seen during training)
        """
        if fast_confidence < self.confidence_threshold:
            return True  # Fast policy unsure
        if world_model_uncertainty > self.uncertainty_threshold:
            return True  # Dynamics unclear
        if novelty_score > self.novelty_threshold:
            return True  # Novel situation
        return False  # Use fast policy
```

**Calibration:** Thresholds are tuned during training Phase 4 to optimize:
```
efficiency = success_rate / weighted_compute_cost
```

---

### 7. LLM Integration (Llama 3.2 1B)

**Purpose:** Generate goal hypotheses when exploration guidance is needed.

**Setup:**
```python
from llama_cpp import Llama

class LLMGoalHypothesizer:
    def __init__(self, model_path: str):
        self.llm = Llama(
            model_path=model_path,  # Path to GGUF file
            n_ctx=512,              # Context window
            n_gpu_layers=-1,        # Full GPU offload
        )
        self.cache = {}  # Response caching

    def hypothesize_goals(
        self,
        state_description: str,
        recent_observations: list[str],
    ) -> list[str]:
        """Generate 3 goal hypotheses for current state."""

        # Check cache first
        cache_key = hash(state_description)
        if cache_key in self.cache:
            return self.cache[cache_key]

        prompt = f"""Analyze this grid puzzle and suggest 3 possible goals.

State: {state_description}
Recent: {recent_observations[-3:]}

Goals (one per line):"""

        response = self.llm(prompt, max_tokens=100, temperature=0.3)
        goals = response["choices"][0]["text"].strip().split("\n")[:3]

        self.cache[cache_key] = goals
        return goals
```

**Resource Usage:**
- VRAM: ~1GB (int4 quantization)
- Latency: ~50ms per query
- Caching reduces redundant queries by ~80%

---

## Data Pipeline

### Data Sources

| Source | Episodes | Purpose |
|--------|----------|---------|
| ARC-AGI-3 API (self-play) | 50k | Real environment dynamics |
| Synthetic environments | 200k | Diverse mechanics, known ground truth |
| **Total** | **250k** | ~10GB on disk |

### Synthetic Environment Generator

```python
class SyntheticEnvGenerator:
    """
    Generate procedural ARC-AGI-3-like environments.

    Mechanics library:
    1. Navigation - Agent movement on grid
    2. Collection - Pick up colored objects
    3. Switches - Toggle states with interactions
    4. Keys/Doors - Unlock mechanisms (hidden state)
    5. Pushing - Move objects by contact
    6. Patterns - Match/create color patterns
    """

    def generate(self) -> SyntheticEnv:
        # Select 1-3 mechanics to compose
        mechanics = random.sample(self.mechanics_library, k=random.randint(1, 3))

        # Generate grid
        grid_size = random.randint(16, 64)
        grid = self._generate_grid(grid_size, mechanics)

        # Define win condition
        win_condition = self._compose_win_condition(mechanics)

        return SyntheticEnv(
            grid=grid,
            mechanics=mechanics,
            win_condition=win_condition,
            max_steps=random.randint(20, 100),
        )
```

### Episode Format

```python
@dataclass
class Episode:
    game_id: str
    observations: list[torch.Tensor]  # Grid states
    actions: list[int]
    rewards: list[float]
    dones: list[bool]
    info: dict  # Metadata (mechanics, win condition, etc.)
```

---

## Training Pipeline

### Phase 1: World Model Pretraining

**Goal:** Learn accurate dynamics prediction

**Data:** Random policy rollouts (10k episodes)

**Training:**
```python
for batch in dataloader:
    states, actions, next_states = batch

    # Forward pass through all ensemble members
    predictions = world_model(states, actions)

    # MSE loss on next state prediction
    prediction_loss = F.mse_loss(predictions, next_states)

    # Consistency loss (forward-backward agreement)
    reconstructed = backward_model(predictions, actions)
    consistency_loss = F.mse_loss(reconstructed, states)

    loss = prediction_loss + 0.1 * consistency_loss
    loss.backward()
    optimizer.step()
```

**Success Criterion:** Prediction error < 30% at 5-step horizon

### Phase 2: Fast Policy Training

**Goal:** Learn habitual actions via imitation + reinforcement

**Stage 2a - Behavioral Cloning:**
```python
# Clone from random + exploratory data
for batch in expert_data:
    states, expert_actions = batch
    predicted_actions = fast_policy(states)
    loss = F.cross_entropy(predicted_actions, expert_actions)
```

**Stage 2b - PPO Fine-tuning:**
```python
for episode in environment:
    # Collect trajectory
    states, actions, rewards = rollout(fast_policy, environment)

    # Compute advantages
    advantages = compute_gae(rewards, values)

    # PPO update
    for _ in range(ppo_epochs):
        ratio = new_probs / old_probs
        clipped = torch.clamp(ratio, 1-eps, 1+eps)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        value_loss = F.mse_loss(values, returns)
        entropy_loss = -policy.entropy().mean()

        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_loss
```

### Phase 3: Slow Policy Training

**Goal:** Learn deliberate reasoning for difficult situations

**Data Collection:**
```python
# Identify states where fast policy struggles
difficult_states = []
for episode in evaluation_episodes:
    for state, fast_output in zip(states, fast_outputs):
        if fast_output.confidence < 0.5:
            difficult_states.append(state)
```

**MCTS Supervision:**
```python
for state in difficult_states:
    # Run MCTS to find good actions
    mcts_policy = mcts.search(state, world_model, num_simulations=100)

    # Train slow policy to match MCTS
    slow_output = slow_policy(state, belief, goal)
    loss = F.kl_div(slow_output.log_softmax(), mcts_policy)
```

### Phase 4: Arbiter Calibration

**Goal:** Optimize fast/slow switching thresholds

**Procedure:**
```python
# Collect statistics
for episode in validation_episodes:
    for step in episode:
        # Record outcomes for both systems
        fast_action = fast_policy(state)
        slow_action = slow_policy(state, belief, goal)

        # Ground truth from environment
        fast_correct = (fast_action == optimal_action)
        slow_correct = (slow_action == optimal_action)

        # Record switching signals
        signals = {
            'confidence': fast_policy.confidence,
            'uncertainty': world_model.uncertainty,
            'entropy': belief_tracker.entropy,
            'novelty': novelty_detector.score,
            'fast_correct': fast_correct,
            'slow_correct': slow_correct,
        }
        statistics.append(signals)

# Optimize thresholds
best_thresholds = grid_search(
    statistics,
    metric=lambda s: efficiency(s.success_rate, s.compute_cost)
)
```

### Phase 5: Joint Fine-tuning

**Goal:** End-to-end optimization with all components

**Training Loop:**
```python
for epoch in range(num_epochs):
    for episode in mixed_episodes:
        # Use arbiter to select system
        for state in episode:
            if arbiter.should_use_slow(...):
                action = slow_policy(state, belief, goal)
                system = 'slow'
            else:
                action = fast_policy(state)
                system = 'fast'

            # Environment step
            next_state, reward, done = env.step(action)

            # Update all components
            world_model.update(state, action, next_state)
            belief_tracker.update(action, next_state)

            if system == 'fast':
                fast_policy.update(state, action, reward)
            else:
                slow_policy.update(state, belief, goal, action, reward)
```

---

## File Structure

```
src/aria_lite/
├── __init__.py                  # Package exports
├── config.py                    # ARIALiteConfig dataclass
├── encoder.py                   # GridEncoderLite
├── world_model.py               # EnsembleWorldModelLite (3 heads)
├── belief.py                    # BeliefStateTracker adapter
├── fast_policy.py               # FastPolicy adapter
├── slow_policy.py               # SlowPolicy (transformer)
├── arbiter.py                   # Metacognitive switcher
├── agent.py                     # ARIALiteAgent
├── llm.py                       # LLMGoalHypothesizer
└── training/
    ├── __init__.py
    ├── data_collector.py        # Episode collection utilities
    ├── synthetic_env.py         # Procedural environment generator
    ├── replay_buffer.py         # Experience storage
    └── trainer.py               # Multi-phase training orchestration
```

---

## Dependencies

### New packages to add to pyproject.toml:

```toml
[project.optional-dependencies]
aria-lite = [
    "llama-cpp-python>=0.2.0",   # Local LLM inference
    "wandb>=0.16.0",              # Experiment tracking
    "gymnasium>=0.29.0",          # Environment interface
]
```

### External model downloads:

| Model | Size | Source |
|-------|------|--------|
| Llama 3.2 1B (GGUF, Q4) | ~1.2GB | HuggingFace (TheBloke or official) |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Level completion (seen envs) | >60% | Evaluation on training game distribution |
| Fast/slow switching benefit | >10% | Compare to fast-only baseline |
| World model error (5-step) | <30% | MSE on held-out trajectories |
| Fast policy usage rate | >50% | Fraction of steps using fast system |
| Training VRAM | <7GB | nvidia-smi during training |
| Inference FPS | >10k | Benchmark on synthetic episodes |

---

## Risk Mitigation

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| World model errors compound | Medium | Increase grounding frequency, use shorter planning horizons |
| Fast policy overconfident | Medium | Calibrate confidence via temperature scaling |
| Slow policy too slow | Low | Reduce transformer layers, use caching |
| LLM latency impacts gameplay | Medium | Aggressive caching, async queries |
| VRAM exceeded | Low | Gradient checkpointing, reduce batch size |

---

## Implementation Order

```
Phase 1: Core (files 1-4)
├── 1. config.py              ← No dependencies
├── 2. encoder.py             ← Depends on config
├── 3. world_model.py         ← Depends on config
└── 4. belief.py              ← Import from arc_dreamer_v2

Phase 2: Policies (files 5-7)
├── 5. fast_policy.py         ← Import from aria
├── 6. slow_policy.py         ← Depends on config
└── 7. arbiter.py             ← Depends on config

Phase 3: Integration (files 8-12)
├── 8. llm.py                 ← External dep: llama-cpp-python
├── 9. agent.py               ← Depends on all above
├── 10. training/replay_buffer.py
├── 11. training/synthetic_env.py
└── 12. training/trainer.py   ← Depends on agent + buffer + env
```

---

## Validation Checkpoints

### Checkpoint 1: Core Components
- [ ] GridEncoderLite produces [B, 256] output
- [ ] World model prediction error < 50% at 1-step
- [ ] Belief tracker maintains particle distribution

### Checkpoint 2: Policies
- [ ] Fast policy inference < 0.1ms
- [ ] Slow policy produces valid action distributions
- [ ] Arbiter switching logic is correct

### Checkpoint 3: Integration
- [ ] Full agent runs through synthetic environment
- [ ] VRAM usage < 7GB during training
- [ ] Wandb logging captures all metrics

### Checkpoint 4: Training
- [ ] World model converges (Phase 1)
- [ ] Fast policy achieves >40% on easy tasks (Phase 2)
- [ ] Slow policy improves difficult cases (Phase 3)
- [ ] Combined system achieves >60% (Phase 5)

---

## Next Steps After ARIA-Lite

If ARIA-Lite achieves success criteria, proceed to:

1. **ARIA-Standard** (75M params, 16GB VRAM)
   - Add DINOv2-Small backbone
   - Scale world model to 5 heads
   - Upgrade to Llama 3.2 3B

2. **ARIA-Max** (213M params, 54GB VRAM)
   - Move to A100
   - Add goal/hidden state detectors
   - Full distillation pipeline

---

*Document version: 1.0*
*Target: ARC-AGI-3 Competition (March 2026)*
