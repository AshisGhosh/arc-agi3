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
    total_new_tokens: int = 589
    total_vocab_size: int = 49741  # World model vocab (without MASK)

    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # Token IDs (unified action tokenization)
    vq_offset: int = 49152        # 49152-49663 (512 VQ codes)
    act_type_offset: int = 49664  # 49664-49671 (8 action types)
    act_loc_offset: int = 49672   # 49672-49736 (65 locations: 64 cells + NULL)
    act_loc_null: int = 49736     # NULL location for non-spatial actions
    frame_token: int = 49737
    act_token: int = 49738
    level_complete_token: int = 49739
    game_start_token: int = 49740
    mask_token: int = 49741       # Used by policy only (not in world model vocab)

    # Architecture
    hidden_size: int = 960  # SmolLM2-360M hidden size
    max_context: int = 2048
    num_action_types: int = 8
    num_action_locs: int = 65  # 64 VQ cells + 1 NULL


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
    action_type_weight: float = 3.0
    spatial_loc_weight: float = 10.0   # Spatial click locations (cells 0-63)
    null_loc_weight: float = 0.0       # NULL location — no gradient (learned from context)
    level_complete_weight: float = 5.0
    structural_weight: float = 0.0  # FRAME, ACT, GAME_START markers

    # Game-balanced sampling
    game_balanced: bool = True  # Equalize sampling across games

    # Context
    max_seq_len: int = 2048
    window_stride: int = 690  # ~10 frames * 69 tokens/frame

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
class PolicyConfig:
    """Policy head training config."""
    # Architecture
    hidden_size: int = 960
    type_head_hidden: int = 256
    loc_head_dim: int = 128
    num_action_types: int = 8
    num_action_locs: int = 65  # 64 VQ cells + 1 NULL
    dropout: float = 0.1

    # Training
    batch_size: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 50
    max_grad_norm: float = 1.0

    # Paths
    world_model_checkpoint: str = "checkpoints/world_model/best.pt"
    vqvae_checkpoint: str = "checkpoints/vqvae/best.pt"
    output_dir: str = "checkpoints/policy"
    demo_dir: str = "videos/ARC-AGI-3 Human Performance"
    cache_dir: str = "checkpoints/world_model/cache"


@dataclass
class AgentConfig:
    """Inference agent config."""
    # Model paths
    world_model_checkpoint: str = "checkpoints/world_model/best.pt"
    policy_checkpoint: str = "checkpoints/policy/best.pt"
    vqvae_checkpoint: str = "checkpoints/vqvae/best.pt"

    # Decision making
    surprise_ema_decay: float = 0.95
    goal_threshold_factor: float = 2.0
    surprise_threshold_factor: float = 2.0

    # Inference
    num_action_types: int = 8
    temperature: float = 1.0
    max_context_frames: int = 29  # ~2048 tokens at 69 tokens/frame


@dataclass
class PlanningConfig:
    """KV-cache lookahead planning agent config."""
    # Model paths (no policy checkpoint needed — uses lm_head directly)
    world_model_checkpoint: str = "checkpoints/world_model/best.pt"
    vqvae_checkpoint: str = "checkpoints/vqvae/best.pt"

    # Planning parameters
    top_k_types: int = 3          # Candidate action types to explore
    top_k_locs: int = 3           # Candidate locations per type
    level_complete_weight: float = 10.0   # Weight for P(LEVEL_COMPLETE)
    frame_change_weight: float = 1.0      # Weight for frame difference
    type_prior_weight: float = 0.5        # Weight for lm_head type prior

    # Inference
    temperature: float = 0.8      # Sampling temperature for tie-breaking
    max_context_frames: int = 29  # ~2048 tokens at 69 tokens/frame

    # Exploration
    repeat_penalty: float = 0.5   # Penalty for repeating recent actions
    repeat_window: int = 5        # How many recent actions to penalize
