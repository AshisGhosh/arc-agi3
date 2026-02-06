"""
SmolLM2-based game transformer with extended vocabulary and LoRA.

Loads a pretrained SmolLM2-360M, extends its vocabulary with 523 game-specific
tokens (VQ codes, actions, structural markers), and applies LoRA for
parameter-efficient fine-tuning.

Trainable parameters: ~4.9M (3.9M LoRA + 1M new embeddings).
Base 362M parameters are frozen.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from .config import WorldModelConfig
from ..tokenizer.trajectory_dataset import (
    VQ_OFFSET, ACT_OFFSET, FRAME_TOKEN, ACT_TOKEN,
    LEVEL_COMPLETE, GAME_START, TOTAL_NEW_TOKENS,
)


def create_game_transformer(
    config: WorldModelConfig | None = None,
    vqvae_codebook: torch.Tensor | None = None,
) -> nn.Module:
    """
    Create a SmolLM2 model extended for game token prediction.

    Args:
        config: Model configuration.
        vqvae_codebook: Optional [512, 128] VQ-VAE codebook vectors
            for initializing VQ token embeddings.

    Returns:
        PEFT model ready for training.
    """
    config = config or WorldModelConfig()

    # 1. Load base model in bfloat16 (better dynamic range than fp16, native 4090 support)
    print(f"Loading {config.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # 2. Extend vocabulary
    old_vocab_size = model.config.vocab_size
    new_vocab_size = old_vocab_size + TOTAL_NEW_TOKENS
    print(f"Extending vocabulary: {old_vocab_size} → {new_vocab_size} (+{TOTAL_NEW_TOKENS})")

    model.resize_token_embeddings(new_vocab_size)

    # Initialize new embeddings from existing distribution
    with torch.no_grad():
        old_embeds = model.get_input_embeddings().weight[:old_vocab_size]
        mean = old_embeds.mean(dim=0)
        std = old_embeds.std(dim=0)

        new_embeds = model.get_input_embeddings().weight[old_vocab_size:]
        nn.init.normal_(new_embeds, 0, 0.02)
        # Scale to match existing distribution
        new_embeds.mul_(std.unsqueeze(0)).add_(mean.unsqueeze(0))

        # Optionally initialize VQ tokens from VQ-VAE codebook
        if vqvae_codebook is not None:
            # Project 128-dim codebook to hidden_size (960)
            device = old_embeds.device
            codebook_cpu = vqvae_codebook.float().cpu()
            projection = torch.randn(128, config.hidden_size) * 0.01
            projected = codebook_cpu @ projection
            projected = projected.to(new_embeds.dtype)
            # Normalize to match embedding scale
            projected = projected * (std.norm() / projected.norm(dim=1, keepdim=True).mean())
            vq_start = VQ_OFFSET - old_vocab_size
            vq_end = vq_start + 512
            model.get_input_embeddings().weight[VQ_OFFSET:VQ_OFFSET + 512] = projected

        # Initialize output head for new tokens similarly
        if hasattr(model, "lm_head"):
            lm_weight = model.lm_head.weight
            old_lm = lm_weight[:old_vocab_size]
            lm_mean = old_lm.mean(dim=0)
            lm_std = old_lm.std(dim=0)
            new_lm = lm_weight[old_vocab_size:]
            nn.init.normal_(new_lm, 0, 0.02)
            new_lm.mul_(lm_std.unsqueeze(0)).add_(lm_mean.unsqueeze(0))

    # 3. Apply LoRA
    print(f"Applying LoRA (rank={config.lora_rank}, targets={config.lora_target_modules})")
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Make embedding/lm_head layers trainable (needed for new token embeddings)
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True

    # Print parameter counts
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} ({trainable/1e6:.1f}M) / Total: {total:,} ({total/1e6:.1f}M)")

    return model


def load_game_transformer(
    checkpoint_path: str,
    config: WorldModelConfig | None = None,
    device: str = "cuda",
) -> nn.Module:
    """Load a trained game transformer from checkpoint."""
    config = config or WorldModelConfig()

    # Create model structure
    model = create_game_transformer(config)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def get_tokenizer(model_name: str = "HuggingFaceTB/SmolLM2-360M") -> AutoTokenizer:
    """Get tokenizer (mainly for reference - we use custom token IDs)."""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
