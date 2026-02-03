"""
ARIA Configuration
"""

from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class ARIAConfig:
    """
    Configuration for ARIA architecture.

    Provides presets for different hardware targets:
    - RTX 4090 (24GB): Prototype/development
    - A100 (80GB): Full scale training
    """

    # === Environment ===
    grid_size: int = 64
    num_colors: int = 16
    num_actions: int = 8

    # === Perception Tower ===
    embed_dim: int = 64
    hidden_dim: int = 256
    num_transformer_layers: int = 4
    num_attention_heads: int = 4

    # === Fast Policy ===
    policy_hidden_dim: int = 256

    # === Slow Planner ===
    max_search_depth: int = 15
    max_search_expansions: int = 1000
    use_llm_heuristic: bool = False  # Enable for better planning, slower

    # === Belief State Tracker ===
    belief_hidden_dim: int = 256
    belief_stoch_dim: int = 32
    belief_num_categories: int = 32

    # === Goal Discovery ===
    num_goal_prototypes: int = 32

    # === Memory ===
    replay_capacity: int = 500_000
    rule_library_size: int = 500
    max_context_length: int = 32

    # === Training ===
    train_batch_size: int = 256
    sequence_length: int = 50
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    kl_weight: float = 1.0

    # === Meta-Learning ===
    meta_batch_size: int = 4
    inner_steps: int = 5
    inner_lr: float = 1e-3
    meta_iterations: int = 10000

    # === World Model ===
    world_model_epochs: int = 100
    imagination_horizon: int = 15

    # === Inference ===
    inference_batch_size: int = 64
    confidence_threshold: float = 0.7
    novelty_threshold: float = 0.5

    # === Optimization ===
    use_amp: bool = True
    use_cuda_graphs: bool = True
    compile_mode: Optional[Literal["default", "reduce-overhead", "max-autotune"]] = (
        "reduce-overhead"
    )

    # === Device ===
    device: str = "cuda"

    @classmethod
    def for_4090(cls) -> "ARIAConfig":
        """
        Optimized configuration for RTX 4090 (24GB VRAM).
        Target: ~5000 FPS, 10GB VRAM usage.
        """
        return cls(
            # Smaller model
            embed_dim=64,
            hidden_dim=256,
            num_transformer_layers=4,
            policy_hidden_dim=256,
            belief_hidden_dim=256,
            belief_stoch_dim=32,
            # Smaller memory
            replay_capacity=500_000,
            rule_library_size=500,
            # Smaller batches
            train_batch_size=256,
            inference_batch_size=64,
            # Full optimization
            use_amp=True,
            use_cuda_graphs=True,
            compile_mode="reduce-overhead",
        )

    @classmethod
    def for_a100(cls) -> "ARIAConfig":
        """
        Optimized configuration for A100 (80GB VRAM).
        Target: ~20000 FPS, 40GB VRAM usage.
        """
        return cls(
            # Larger model
            embed_dim=128,
            hidden_dim=512,
            num_transformer_layers=8,
            num_attention_heads=8,
            policy_hidden_dim=512,
            belief_hidden_dim=512,
            belief_stoch_dim=64,
            # Larger memory
            replay_capacity=2_000_000,
            rule_library_size=2000,
            # Larger batches
            train_batch_size=1024,
            inference_batch_size=256,
            # Max optimization
            use_amp=True,
            use_cuda_graphs=True,
            compile_mode="max-autotune",
        )

    @classmethod
    def for_cpu(cls) -> "ARIAConfig":
        """
        Configuration for CPU-only inference (testing).
        """
        return cls(
            # Minimal model
            embed_dim=32,
            hidden_dim=128,
            num_transformer_layers=2,
            policy_hidden_dim=128,
            belief_hidden_dim=128,
            belief_stoch_dim=16,
            # Minimal memory
            replay_capacity=10_000,
            rule_library_size=100,
            # Small batches
            train_batch_size=32,
            inference_batch_size=1,
            # No GPU optimization
            use_amp=False,
            use_cuda_graphs=False,
            compile_mode=None,
            device="cpu",
        )

    def estimate_vram_gb(self) -> float:
        """
        Estimate VRAM usage in GB.
        """
        # Rough estimates based on parameter counts
        perception_params = (
            self.embed_dim * self.num_colors
            + self.hidden_dim * self.hidden_dim * self.num_transformer_layers * 12
        )
        policy_params = self.policy_hidden_dim * self.policy_hidden_dim * 4
        belief_params = (
            self.belief_hidden_dim * self.belief_hidden_dim * 4
            + self.belief_stoch_dim * self.belief_num_categories * self.belief_hidden_dim * 2
        )

        total_params = perception_params + policy_params + belief_params

        # 2 bytes per param (fp16) + optimizer state (2x for Adam)
        model_memory = total_params * 2 * 3 / 1e9

        # Replay buffer (assuming 256 bytes per transition)
        replay_memory = self.replay_capacity * 256 / 1e9

        # Activation memory (rough estimate)
        activation_memory = (
            self.train_batch_size * self.hidden_dim * self.grid_size * self.grid_size * 2 / 1e9
        )

        return model_memory + replay_memory + activation_memory
