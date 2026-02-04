"""
ARIA-Lite Configuration

Self-contained configuration for the ARIA-Lite architecture.
No external dependencies beyond stdlib and torch.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class EncoderConfig:
    """Grid encoder configuration."""

    # Input
    max_grid_size: int = 64
    num_colors: int = 16

    # Embedding
    color_embed_dim: int = 64
    pos_embed_dim: int = 64

    # CNN (increased for 5M target)
    cnn_channels: tuple[int, ...] = (128, 256, 512)
    cnn_kernel_size: int = 3

    # Transformer (increased for 5M target)
    transformer_dim: int = 512
    transformer_heads: int = 8
    transformer_layers: int = 3
    transformer_ff_dim: int = 1024
    transformer_dropout: float = 0.1

    # Output
    output_dim: int = 256

    def estimate_params(self) -> int:
        """Estimate total parameters."""
        params = 0

        # Color embedding: 16 colors -> 32 dim
        params += self.num_colors * self.color_embed_dim  # ~512

        # Positional encoding (learned)
        params += self.max_grid_size * self.max_grid_size * self.pos_embed_dim  # ~130K

        # CNN layers
        in_channels = self.color_embed_dim + self.pos_embed_dim  # 64
        for out_channels in self.cnn_channels:
            # Conv weight + bias
            params += in_channels * out_channels * self.cnn_kernel_size**2
            params += out_channels
            # GroupNorm
            params += out_channels * 2
            in_channels = out_channels

        # Transformer
        d = self.transformer_dim
        for _ in range(self.transformer_layers):
            # Self-attention: Q, K, V projections + output
            params += 4 * d * d
            # Feedforward
            params += d * self.transformer_ff_dim + self.transformer_ff_dim
            params += self.transformer_ff_dim * d + d
            # Layer norms
            params += 4 * d

        # Output projection
        params += self.cnn_channels[-1] * self.output_dim + self.output_dim

        return params


@dataclass
class WorldModelConfig:
    """World model ensemble configuration."""

    state_dim: int = 256
    action_dim: int = 8
    hidden_dim: int = 1024  # Increased for 5M per head
    num_layers: int = 3  # More layers
    num_ensemble: int = 3

    def estimate_params(self) -> int:
        """Estimate parameters for ensemble."""
        params_per_head = 0

        # Input layer
        input_dim = self.state_dim + self.action_dim
        params_per_head += input_dim * self.hidden_dim + self.hidden_dim
        params_per_head += self.hidden_dim * 2  # LayerNorm

        # Hidden layers
        for _ in range(self.num_layers - 1):
            params_per_head += self.hidden_dim * self.hidden_dim + self.hidden_dim
            params_per_head += self.hidden_dim * 2  # LayerNorm

        # Predictors
        params_per_head += self.hidden_dim * self.state_dim + self.state_dim  # State
        params_per_head += self.hidden_dim * 1 + 1  # Reward
        params_per_head += self.hidden_dim * 1 + 1  # Done

        return params_per_head * self.num_ensemble


@dataclass
class BeliefConfig:
    """Belief state tracker (RSSM) configuration."""

    hidden_dim: int = 256  # Increased for 3M target
    stochastic_dim: int = 64
    observation_dim: int = 256
    action_dim: int = 8
    num_particles: int = 50
    num_layers: int = 3

    def estimate_params(self) -> int:
        """Estimate parameters."""
        params = 0

        # Transition model (GRU-like): 3 gates x (input + hidden -> hidden)
        input_dim = self.hidden_dim + self.action_dim
        params += 3 * (input_dim * self.hidden_dim + self.hidden_dim)  # GRU gates

        # Stochastic layer
        params += self.hidden_dim * self.stochastic_dim * 2 + self.stochastic_dim * 2  # mean, std

        # Observation model: encoder
        params += self.observation_dim * self.hidden_dim + self.hidden_dim
        for _ in range(self.num_layers - 1):
            params += self.hidden_dim * self.hidden_dim + self.hidden_dim

        # Posterior (combine observation with prior)
        params += (self.hidden_dim + self.observation_dim) * self.hidden_dim + self.hidden_dim
        params += self.hidden_dim * self.stochastic_dim * 2 + self.stochastic_dim * 2

        # Decoder (hidden -> observation)
        params += (self.hidden_dim + self.stochastic_dim) * self.hidden_dim + self.hidden_dim
        params += self.hidden_dim * self.observation_dim + self.observation_dim

        return params


@dataclass
class FastPolicyConfig:
    """Fast policy (habit network) configuration."""

    state_dim: int = 256
    hidden_dim: int = 256  # Increased for 1M target
    num_actions: int = 8
    grid_size: int = 64
    num_layers: int = 3

    def estimate_params(self) -> int:
        """Estimate parameters."""
        params = 0

        # Input compression
        params += self.state_dim * self.hidden_dim + self.hidden_dim

        # Hidden layers
        for _ in range(self.num_layers - 1):
            params += self.hidden_dim * self.hidden_dim + self.hidden_dim

        # Action head
        params += self.hidden_dim * self.hidden_dim + self.hidden_dim
        params += self.hidden_dim * self.num_actions + self.num_actions

        # Confidence head
        params += self.hidden_dim * (self.hidden_dim // 2) + self.hidden_dim // 2
        params += (self.hidden_dim // 2) * 1 + 1

        # Coordinate heads (factorized)
        coord_hidden = self.hidden_dim // 2
        # X head
        params += (self.hidden_dim + coord_hidden) * coord_hidden + coord_hidden
        params += coord_hidden * self.grid_size + self.grid_size
        # Y head
        params += (self.hidden_dim + coord_hidden + self.grid_size) * coord_hidden + coord_hidden
        params += coord_hidden * self.grid_size + self.grid_size

        return params


@dataclass
class SlowPolicyConfig:
    """Slow policy (transformer planner) configuration."""

    state_dim: int = 256
    belief_dim: int = 256  # Match belief hidden_dim
    goal_dim: int = 64
    hidden_dim: int = 384  # Increased for 5M target
    num_heads: int = 6
    num_layers: int = 6  # More layers
    ff_dim: int = 1024  # Larger feedforward
    dropout: float = 0.1
    num_actions: int = 8

    def estimate_params(self) -> int:
        """Estimate parameters."""
        params = 0

        # Input projection
        input_dim = self.state_dim + self.belief_dim + self.goal_dim
        params += input_dim * self.hidden_dim + self.hidden_dim

        # Transformer layers
        d = self.hidden_dim
        for _ in range(self.num_layers):
            # Self-attention: Q, K, V, out projections
            params += 4 * d * d + 4 * d
            # Feedforward
            params += d * self.ff_dim + self.ff_dim
            params += self.ff_dim * d + d
            # Layer norms (2 per layer)
            params += 4 * d

        # Output heads
        params += d * self.num_actions + self.num_actions  # Policy
        params += d * 1 + 1  # Value
        params += d * 1 + 1  # Uncertainty

        return params


@dataclass
class ArbiterConfig:
    """Arbiter (metacognitive switcher) configuration."""

    confidence_threshold: float = 0.7
    uncertainty_threshold: float = 0.3
    novelty_threshold: float = 0.5
    use_learned_switching: bool = False
    switching_hidden_dim: int = 32

    def estimate_params(self) -> int:
        """Estimate parameters (minimal if using heuristics)."""
        if not self.use_learned_switching:
            return 0
        # Small MLP for learned switching
        params = 4 * self.switching_hidden_dim + self.switching_hidden_dim
        params += self.switching_hidden_dim * 1 + 1
        return params


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Optimization
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    gradient_clip: float = 1.0

    # RL
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5

    # PPO
    ppo_epochs: int = 4
    ppo_clip: float = 0.2

    # World model
    world_model_lr: float = 1e-4
    imagination_horizon: int = 15

    # Data
    replay_buffer_size: int = 100_000
    min_buffer_size: int = 1000

    # Phases
    world_model_pretrain_steps: int = 10_000
    fast_policy_bc_steps: int = 5_000
    fast_policy_rl_steps: int = 20_000
    slow_policy_steps: int = 10_000
    joint_finetune_steps: int = 10_000


@dataclass
class LLMConfig:
    """LLM integration configuration."""

    model_path: str = ""  # Path to GGUF file
    model_name: str = "llama-3.2-1b"
    context_length: int = 512
    max_tokens: int = 100
    temperature: float = 0.3
    n_gpu_layers: int = -1  # All layers on GPU
    cache_size: int = 1000


@dataclass
class ARIALiteConfig:
    """
    Complete ARIA-Lite configuration.

    Parameter Budget: 29M
    VRAM Budget: 7GB

    Components:
    - Encoder: 5M
    - World Model: 15M (3 heads x 5M)
    - Belief State: 3M
    - Fast Policy: 1M
    - Slow Policy: 5M
    - Arbiter: ~0 (heuristic)
    """

    # Component configs
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    belief: BeliefConfig = field(default_factory=BeliefConfig)
    fast_policy: FastPolicyConfig = field(default_factory=FastPolicyConfig)
    slow_policy: SlowPolicyConfig = field(default_factory=SlowPolicyConfig)
    arbiter: ArbiterConfig = field(default_factory=ArbiterConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Global settings
    device: Literal["cuda", "cpu", "auto"] = "auto"
    seed: int = 42
    dtype: Literal["float32", "float16", "bfloat16"] = "float16"

    # Logging
    log_interval: int = 100
    eval_interval: int = 1000
    save_interval: int = 5000
    wandb_project: str = "aria-lite"
    wandb_run_name: str = ""

    def total_params(self) -> int:
        """Calculate total trainable parameters."""
        return (
            self.encoder.estimate_params()
            + self.world_model.estimate_params()
            + self.belief.estimate_params()
            + self.fast_policy.estimate_params()
            + self.slow_policy.estimate_params()
            + self.arbiter.estimate_params()
        )

    def params_breakdown(self) -> dict[str, int]:
        """Get parameter count per component."""
        return {
            "encoder": self.encoder.estimate_params(),
            "world_model": self.world_model.estimate_params(),
            "belief": self.belief.estimate_params(),
            "fast_policy": self.fast_policy.estimate_params(),
            "slow_policy": self.slow_policy.estimate_params(),
            "arbiter": self.arbiter.estimate_params(),
        }

    def estimate_vram_gb(self, batch_size: int | None = None) -> float:
        """
        Estimate VRAM usage in GB.

        Includes:
        - Model parameters (fp16)
        - Optimizer states (Adam: 2x params)
        - Activations (estimated)
        - Gradient buffers
        """
        if batch_size is None:
            batch_size = self.training.batch_size

        total_params = self.total_params()

        # Model weights (fp16 = 2 bytes per param)
        weights_bytes = total_params * 2

        # Optimizer states (Adam: m and v, each same size as params)
        # In fp32 for stability = 4 bytes each
        optimizer_bytes = total_params * 8

        # Gradients (fp16)
        gradient_bytes = total_params * 2

        # Activations (rough estimate: 2x params * batch_size / 32)
        activation_bytes = total_params * 2 * (batch_size / 32)

        # LLM (Llama 3.2 1B int4 ~ 1GB)
        llm_bytes = 1 * 1024**3

        total_bytes = (
            weights_bytes + optimizer_bytes + gradient_bytes + activation_bytes + llm_bytes
        )

        return total_bytes / (1024**3)

    def validate(self) -> list[str]:
        """Validate configuration against constraints."""
        issues = []

        # Parameter budget
        total = self.total_params()
        if total > 30_000_000:
            issues.append(f"Parameter budget exceeded: {total:,} > 30M")

        # VRAM budget
        vram = self.estimate_vram_gb()
        if vram > 7.5:
            issues.append(f"VRAM budget exceeded: {vram:.1f}GB > 7.5GB")

        # Dimension consistency
        if self.encoder.output_dim != self.fast_policy.state_dim:
            issues.append(
                f"Encoder output ({self.encoder.output_dim}) != "
                f"FastPolicy input ({self.fast_policy.state_dim})"
            )

        if self.encoder.output_dim != self.slow_policy.state_dim:
            issues.append(
                f"Encoder output ({self.encoder.output_dim}) != "
                f"SlowPolicy input ({self.slow_policy.state_dim})"
            )

        if self.belief.hidden_dim != self.slow_policy.belief_dim:
            issues.append(
                f"Belief dim ({self.belief.hidden_dim}) != "
                f"SlowPolicy belief_dim ({self.slow_policy.belief_dim})"
            )

        return issues

    def summary(self) -> str:
        """Generate configuration summary."""
        breakdown = self.params_breakdown()
        total = self.total_params()
        vram = self.estimate_vram_gb()

        lines = [
            "ARIA-Lite Configuration Summary",
            "=" * 40,
            "",
            "Parameter Breakdown:",
        ]

        for name, count in breakdown.items():
            pct = count / total * 100
            lines.append(f"  {name:15s}: {count:>10,} ({pct:5.1f}%)")

        lines.extend(
            [
                f"  {'TOTAL':15s}: {total:>10,}",
                "",
                f"Estimated VRAM: {vram:.2f} GB",
                f"Batch Size: {self.training.batch_size}",
                f"Device: {self.device}",
                f"Dtype: {self.dtype}",
                "",
            ]
        )

        issues = self.validate()
        if issues:
            lines.append("VALIDATION ISSUES:")
            for issue in issues:
                lines.append(f"  - {issue}")
        else:
            lines.append("Validation: PASSED")

        return "\n".join(lines)


# Convenience function
def create_config(**overrides) -> ARIALiteConfig:
    """Create config with optional overrides."""
    config = ARIALiteConfig()

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")

    return config
