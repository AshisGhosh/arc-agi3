# ARIA Architecture Variants

Concrete implementation variants of the ARIA (Adaptive Reasoning with Integrated Abstractions) hybrid architecture, designed for incremental validation from RTX 4090 to A100.

---

## Hardware Constraints

| Resource | RTX 4090 | A100 (80GB) |
|----------|----------|-------------|
| VRAM | 24GB | 80GB |
| Training budget (fp16) | ~60M trainable params | ~300M trainable params |
| Inference budget (int4) | ~12B params | ~40B params |
| Batch size (typical) | 32-64 | 128-256 |

---

## Variant A: ARIA-Lite

**Purpose:** Validate core dual-system architecture with minimal compute.

**Expected Score:** 7.0/10

### Architecture

| Component | Model | Params | VRAM (train) |
|-----------|-------|--------|--------------|
| Grid Encoder | Custom CNN+Transformer | 5M | 200MB |
| Belief State (RSSM) | Custom | 3M | 120MB |
| Fast Policy (habits) | MLP | 1M | 40MB |
| Slow Policy (planner) | Small Transformer | 5M | 200MB |
| World Model | Ensemble of 3 | 15M | 600MB |
| **Total Trainable** | | **29M** | **~1.2GB** |

### Pretrained Components

None - fully trained from scratch.

### LLM Integration

| Use Case | Model | Quantization | VRAM | Latency |
|----------|-------|--------------|------|---------|
| Goal hypotheses | Llama 3.2 1B | int4 | 1GB | 50ms |
| Fallback reasoning | API (Claude Haiku) | - | 0 | 300ms |

### VRAM Budget

```
Training mode:
  Trainable models:         1.2GB
  Activations (batch 32):   2.0GB
  Optimizer states:         2.4GB
  Llama 3.2 1B (int4):      1.0GB
  ─────────────────────────────────
  Total:                   ~7GB ✓ (fits easily on 4090)

Inference mode:
  Models (fp16):            0.6GB
  Llama 3.2 1B (int4):      1.0GB
  ─────────────────────────────────
  Total:                   ~2GB ✓
```

### Data Requirements

| Dataset | Size | Source |
|---------|------|--------|
| ARC-AGI-3 episodes | 50k episodes | Self-play on real API |
| Synthetic environments | 200k episodes | Procedural generator |
| **Total** | **250k episodes** | ~10GB on disk |

### What This Validates

| Hypothesis | Validated |
|------------|-----------|
| Dual-system fast/slow switching works | Yes |
| World model predicts well enough for planning | Yes |
| Belief tracking helps with hidden state | Yes |
| Pretrained vision helps | No |
| Full scale performance | No |

### Expected Outcomes

- **Strengths:** Fast iteration, proves architecture viability
- **Limitations:** No transfer learning, limited generalization
- **Success criteria:** >60% level completion on seen environments, measurable fast/slow switching benefit

---

## Variant B: ARIA-Standard

**Purpose:** Production candidate with best achievable performance on RTX 4090.

**Expected Score:** 8.5/10

### Architecture

| Component | Model | Params | VRAM (train) |
|-----------|-------|--------|--------------|
| Grid Encoder | DINOv2-Small (frozen) + adapter | 22M + 2M | 180MB |
| Belief State (RSSM) | Custom | 8M | 320MB |
| Fast Policy (habits) | Transformer-XS | 5M | 200MB |
| Slow Policy (planner) | Transformer-S | 15M | 600MB |
| World Model | Ensemble of 5 | 40M | 1.6GB |
| Goal Detector | Contrastive head | 3M | 120MB |
| Hidden State Detector | Bayesian network | 2M | 80MB |
| **Total Trainable** | | **75M** | **~3GB** |

### Pretrained Components (Frozen)

| Component | Model | Params | VRAM | Source |
|-----------|-------|--------|------|--------|
| Vision backbone | DINOv2-Small | 22M | 180MB | Meta (facebookresearch/dinov2) |
| Grid understanding | Custom | 10M | 80MB | Pretrained on ARC-AGI-1/2 |

### LLM Integration

| Use Case | Model | Quantization | VRAM | Latency |
|----------|-------|--------------|------|---------|
| Goal hypotheses | Llama 3.2 3B | int4 | 2.5GB | 80ms |
| Rule induction | Llama 3.2 3B | int4 | (shared) | 100ms |
| Complex planning | API (Claude Haiku) | - | 0 | 300ms |

### VRAM Budget

```
Training mode:
  Trainable models:         3.0GB
  Frozen backbones:         0.3GB
  Activations (batch 64):   4.0GB
  Optimizer states:         6.0GB
  Llama 3.2 3B (int4):      2.5GB
  ─────────────────────────────────
  Total:                  ~16GB ✓ (fits on 4090)

Inference mode:
  Models (fp16):            1.5GB
  Frozen backbones:         0.3GB
  Llama 3.2 3B (int4):      2.5GB
  ─────────────────────────────────
  Total:                   ~5GB ✓
```

### Data Requirements

| Dataset | Size | Source |
|---------|------|--------|
| ARC-AGI-1 training | 400 tasks | Public dataset |
| ARC-AGI-2 training | 400 tasks | Public dataset |
| ARC-AGI-3 episodes | 100k episodes | Self-play + API |
| Synthetic environments | 500k episodes | Procedural generator |
| Hidden state detection | 100k episodes | Synthetic with labels |
| **Total** | **~700k episodes** | ~30GB on disk |

### What This Validates

| Hypothesis | Validated |
|------------|-----------|
| Dual-system fast/slow switching works | Yes |
| World model predicts well enough for planning | Yes |
| Belief tracking helps with hidden state | Yes |
| Pretrained vision helps | Yes |
| Local LLM is fast enough | Yes |
| ARC-1/2 pretraining transfers | Yes |
| Full competition performance | Partial |

### Expected Outcomes

- **Strengths:** Strong baseline, validates all core hypotheses, competitive performance
- **Limitations:** May struggle with most complex environments, limited planning depth
- **Success criteria:** >75% level completion, <2x human action count on solved levels

---

## Variant C: ARIA-Max

**Purpose:** Maximum performance for competition, requires A100.

**Expected Score:** 9.2/10

### Architecture

| Component | Model | Params | VRAM (train) |
|-----------|-------|--------|--------------|
| Grid Encoder | DINOv2-Base (frozen) + adapter | 86M + 5M | 700MB |
| Belief State (RSSM) | Large | 20M | 800MB |
| Fast Policy (habits) | Transformer-M | 20M | 800MB |
| Slow Policy (planner) | Transformer-L | 50M | 2GB |
| World Model | Ensemble of 7 | 100M | 4GB |
| Goal Detector | Large contrastive | 10M | 400MB |
| Hidden State Detector | Ensemble Bayesian | 8M | 320MB |
| **Total Trainable** | | **213M** | **~9GB** |

### Pretrained Components (Frozen)

| Component | Model | Params | VRAM | Source |
|-----------|-------|--------|------|--------|
| Vision backbone | DINOv2-Base | 86M | 700MB | Meta (facebookresearch/dinov2) |
| Grid reasoning | Custom | 30M | 240MB | Pretrained on ARC-1/2 |
| Action primitives | From ARIA-Standard | 10M | 80MB | Transfer from Variant B |

### LLM Integration

| Use Case | Model | Quantization | VRAM | Latency |
|----------|-------|--------------|------|---------|
| All local tasks | Llama 3.1 8B | int4 | 6GB | 150ms |
| Complex reasoning | API (Claude Sonnet) | - | 0 | 800ms |

### VRAM Budget (A100 80GB)

```
Training mode:
  Trainable models:          9.0GB
  Frozen backbones:          1.0GB
  Activations (batch 256):  20.0GB
  Optimizer states:         18.0GB
  Llama 3.1 8B (int4):       6.0GB
  ─────────────────────────────────
  Total:                   ~54GB ✓ (fits on A100)

Inference mode:
  Models (fp16):             4.5GB
  Frozen backbones:          1.0GB
  Llama 3.1 8B (int4):       6.0GB
  ─────────────────────────────────
  Total:                   ~12GB ✓
```

### Data Requirements

| Dataset | Size | Source |
|---------|------|--------|
| Everything from Variant B | 700k episodes | Inherited |
| Extended self-play | 500k episodes | Additional API usage |
| LLM-generated environments | 200k episodes | Procedural + LLM creativity |
| Distillation data (slow→fast) | 1M state-action pairs | Offline slow policy rollouts |
| **Total** | **~2M+ episodes** | ~80GB on disk |

### What This Validates

| Hypothesis | Validated |
|------------|-----------|
| All ARIA-Standard hypotheses | Yes |
| Larger models improve performance | Yes |
| Distillation preserves intelligence | Yes |
| Competition-ready performance | Yes |

### Expected Outcomes

- **Strengths:** Best possible performance, handles edge cases, robust planning
- **Limitations:** Requires A100, longer training, higher API costs
- **Success criteria:** >85% level completion, <1.5x human action count, top-tier competition placement

---

## Variant Comparison

| Aspect | ARIA-Lite | ARIA-Standard | ARIA-Max |
|--------|-----------|---------------|----------|
| **Trainable Params** | 29M | 75M | 213M |
| **Total Params (w/ frozen)** | 29M | 107M | 339M |
| **Training VRAM** | 7GB | 16GB | 54GB |
| **GPU Required** | RTX 4090 | RTX 4090 | A100 80GB |
| **Data Size** | 250k episodes | 700k episodes | 2M+ episodes |
| **Local LLM** | Llama 3.2 1B | Llama 3.2 3B | Llama 3.1 8B |
| **Expected Score** | 7.0/10 | 8.5/10 | 9.2/10 |
| **Purpose** | Validate ideas | Production candidate | Competition |

---

## Recommended Implementation Path

### Phase 1: Foundation (ARIA-Lite)

**Objective:** Prove the dual-system architecture works.

1. **Set up infrastructure**
   - Configure 4090 training environment
   - Set up experiment tracking (weights & biases or similar)
   - Create data pipeline for ARC-AGI-3 API

2. **Build core components**
   - Implement custom grid encoder (CNN + Transformer)
   - Build RSSM belief state tracker
   - Create world model with 3-member ensemble

3. **Implement dual-system policy**
   - Fast policy: MLP habit network
   - Slow policy: Small transformer planner
   - Switching mechanism based on uncertainty

4. **Integrate local LLM**
   - Set up Llama 3.2 1B with int4 quantization
   - Implement goal hypothesis generation
   - Add caching for repeated queries

5. **Train and validate**
   - Collect 50k episodes via self-play
   - Generate 200k synthetic episodes
   - Train full pipeline
   - Evaluate on held-out environments

6. **Success gate:** Proceed to Phase 2 if:
   - Fast/slow switching shows measurable benefit (>10% improvement)
   - World model prediction error <30% at 5-step horizon
   - >60% level completion on seen environments

### Phase 2: Scale Up (ARIA-Standard)

**Objective:** Achieve best performance on 4090 with transfer learning.

1. **Add pretrained vision**
   - Integrate DINOv2-Small as frozen backbone
   - Train lightweight adapter layers
   - Verify improved sample efficiency

2. **Pretrain on ARC-AGI-1/2**
   - Download public ARC-AGI-1 and ARC-AGI-2 datasets
   - Train grid understanding module on transformation tasks
   - Extract reusable primitive library

3. **Build specialized detectors**
   - Implement goal detector with contrastive learning
   - Build hidden state detector with Bayesian network
   - Generate labeled synthetic data for supervision

4. **Scale world model**
   - Increase ensemble to 5 members
   - Add consistency loss between members
   - Implement periodic grounding to real observations

5. **Upgrade LLM integration**
   - Switch to Llama 3.2 3B
   - Add rule induction capability
   - Implement semantic caching

6. **Extended training**
   - Collect 100k ARC-AGI-3 episodes
   - Generate 500k synthetic episodes
   - Generate 100k hidden state detection episodes
   - Full training with all components

7. **Success gate:** Proceed to Phase 3 if:
   - >75% level completion on evaluation set
   - Action efficiency <2x human baseline
   - Hidden state detection accuracy >70%

### Phase 3: Competition Ready (ARIA-Max)

**Objective:** Maximum performance on A100 for competition.

1. **Migrate to A100**
   - Transfer trained weights from ARIA-Standard
   - Verify reproducibility on new hardware

2. **Scale architecture**
   - Upgrade to DINOv2-Base backbone
   - Increase all component sizes per spec
   - Expand world model to 7-member ensemble

3. **Generate distillation data**
   - Run slow policy extensively on all environments
   - Collect 1M+ state-action pairs with reasoning traces
   - Distill into fast policy

4. **Expand data**
   - Extended self-play (500k additional episodes)
   - LLM-generated novel environments (200k episodes)
   - Hard case mining from failure analysis

5. **Upgrade LLM**
   - Switch to Llama 3.1 8B for local inference
   - Add Claude Sonnet for complex reasoning fallback
   - Optimize inference pipeline

6. **Final optimization**
   - Hyperparameter sweep
   - Ensemble multiple training runs
   - Profile and optimize inference latency

7. **Competition preparation**
   - Full evaluation on held-out test set
   - Failure mode analysis
   - Documentation and submission

---

## Key Dependencies

### Software

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.2+ | Core framework |
| transformers | 4.36+ | DINOv2, Llama integration |
| llama-cpp-python | 0.2+ | Local LLM inference |
| arc-agi | 0.9.1+ | ARC-AGI-3 API |
| wandb | latest | Experiment tracking |

### Models to Download

| Model | Size | Source |
|-------|------|--------|
| DINOv2-Small | 88MB | `facebookresearch/dinov2` |
| DINOv2-Base | 344MB | `facebookresearch/dinov2` |
| Llama 3.2 1B (GGUF) | 1.2GB | `TheBloke/Llama-3.2-1B-GGUF` |
| Llama 3.2 3B (GGUF) | 2.5GB | `TheBloke/Llama-3.2-3B-GGUF` |
| Llama 3.1 8B (GGUF) | 6GB | `TheBloke/Llama-3.1-8B-GGUF` |

### Data Sources

| Dataset | URL |
|---------|-----|
| ARC-AGI-1 | https://github.com/fchollet/ARC |
| ARC-AGI-2 | https://github.com/fchollet/ARC-AGI |
| ARC-AGI-3 API | https://three.arcprize.org/ |

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| DINOv2 doesn't transfer to grids | Fall back to custom encoder, fine-tune DINOv2 |
| Local LLM too slow | Aggressive caching, reduce LLM calls, use smaller model |
| World model errors compound | Increase grounding frequency, use shorter horizons |
| Hidden state detection fails | Increase synthetic data diversity, add more supervision |
| 4090 VRAM insufficient | Gradient checkpointing, reduce batch size, mixed precision |

---

## Success Metrics

| Metric | ARIA-Lite | ARIA-Standard | ARIA-Max |
|--------|-----------|---------------|----------|
| Level completion rate | >60% | >75% | >85% |
| Action efficiency (vs human) | <3x | <2x | <1.5x |
| World model error (5-step) | <30% | <25% | <20% |
| Hidden state detection | N/A | >70% | >80% |
| Fast policy usage rate | >50% | >60% | >70% |
| Inference FPS | >10k | >5k | >2k |

---

*Document version: 1.0*
*Architecture: ARIA (Adaptive Reasoning with Integrated Abstractions)*
*Target: ARC-AGI-3 Competition (March 2026)*
