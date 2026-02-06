"""Configuration dataclasses for the learned world model."""

from dataclasses import dataclass, field


@dataclass
class VQVAEConfig:
    """VQ-VAE frame tokenizer config."""
    num_colors: int = 16
    color_embed_dim: int = 32
    hidden_dim: int = 128
    codebook_size: int = 512
    code_dim: int = 128
    commitment_beta: float = 0.25
    ema_decay: float = 0.99
    dead_code_threshold: int = 2


@dataclass
class WorldModelConfig:
    """SmolLM2-based world model config."""
    # Base model
    model_name: str = "HuggingFaceTB/SmolLM2-360M"
    base_vocab_size: int = 49152
    total_new_tokens: int = 523
    total_vocab_size: int = 49675

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Token IDs
    vq_offset: int = 49152
    act_offset: int = 49664
    frame_token: int = 49671
    act_token: int = 49672
    level_complete_token: int = 49673
    game_start_token: int = 49674

    # Architecture
    hidden_size: int = 960  # SmolLM2-360M hidden size
    max_context: int = 2048


@dataclass
class TrainingConfig:
    """Training pipeline config."""
    # Optimization
    batch_size: int = 4
    grad_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 30
    max_grad_norm: float = 1.0

    # Token loss weights
    vq_weight: float = 1.0
    action_weight: float = 2.0
    level_complete_weight: float = 5.0
    structural_weight: float = 0.0  # FRAME, ACT, GAME_START markers

    # Context
    max_seq_len: int = 2048
    window_stride: int = 670

    # Evaluation
    eval_every_epochs: int = 2
    save_every_epochs: int = 5

    # Hardware
    fp16: bool = True
    gradient_checkpointing: bool = True

    # Paths
    vqvae_checkpoint: str = "checkpoints/vqvae/best.pt"
    demo_dir: str = "videos/ARC-AGI-3 Human Performance"
    cache_dir: str = "checkpoints/world_model/cache"
    output_dir: str = "checkpoints/world_model"


@dataclass
class AgentConfig:
    """Inference agent config."""
    # Model paths
    world_model_checkpoint: str = "checkpoints/world_model/best.pt"
    vqvae_checkpoint: str = "checkpoints/vqvae/best.pt"

    # Decision making
    surprise_ema_decay: float = 0.95
    goal_threshold_factor: float = 2.0  # action must be 2x better than mean
    surprise_threshold_factor: float = 2.0

    # Inference
    num_candidate_actions: int = 7  # Actions 0-6
    temperature: float = 1.0
    max_context_frames: int = 30  # ~2048 tokens
