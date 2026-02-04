#!/usr/bin/env python3
"""
ARIA-Lite Training Script

Run the multi-phase training pipeline on synthetic environments.

Usage:
    python -m aria_lite.train [--quick] [--device cuda]
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import torch

from .agent import create_agent
from .config import ARIALiteConfig
from .training.trainer import ARIALiteTrainer, TrainerConfig, create_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train ARIA-Lite agent")

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick training for validation (reduced epochs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/aria_lite",
        help="Directory for checkpoints",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args()


def get_quick_config() -> TrainerConfig:
    """Get configuration for quick validation run."""
    return TrainerConfig(
        # Reduced epochs for quick testing
        wm_epochs=5,
        fp_epochs=5,
        sp_epochs=5,
        arb_epochs=3,
        joint_epochs=3,

        # Smaller batches
        wm_batch_size=16,
        fp_batch_size=32,
        sp_batch_size=16,
        joint_batch_size=16,

        # Smaller buffer
        buffer_capacity=5000,
        min_buffer_size=500,

        # Fewer environments
        num_train_envs=100,
        num_val_envs=20,

        # More frequent logging
        log_every=1,
        checkpoint_every=5,
    )


def get_full_config() -> TrainerConfig:
    """Get configuration for full training run."""
    return TrainerConfig(
        # Full training epochs
        wm_epochs=100,
        fp_epochs=50,
        sp_epochs=100,
        arb_epochs=20,
        joint_epochs=50,

        # Standard batches
        wm_batch_size=64,
        fp_batch_size=128,
        sp_batch_size=32,
        joint_batch_size=64,

        # Large buffer
        buffer_capacity=100_000,
        min_buffer_size=1000,

        # Environment settings
        num_train_envs=1000,
        num_val_envs=100,

        # Logging
        log_every=10,
        checkpoint_every=10,
    )


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print("=" * 60)
    print("ARIA-Lite Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Quick mode: {args.quick}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print()

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Get config
    trainer_config = get_quick_config() if args.quick else get_full_config()
    trainer_config.device = args.device
    trainer_config.seed = args.seed
    trainer_config.checkpoint_dir = str(checkpoint_dir)

    # Create ARIA config
    aria_config = ARIALiteConfig()

    # Print parameter summary
    print("Model Configuration:")
    print(f"  Total parameters: {aria_config.total_params() / 1e6:.1f}M")
    print(f"  Estimated VRAM: {aria_config.estimate_vram_gb():.2f} GB")
    print()

    # Create agent and trainer
    print("Creating agent and trainer...")
    agent = create_agent(aria_config)
    trainer = ARIALiteTrainer(agent, trainer_config, aria_config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training
    print("\nStarting training...")
    start_time = datetime.now()

    try:
        results = trainer.train_full()

        # Save final checkpoint
        final_path = checkpoint_dir / "final_checkpoint.pt"
        trainer.save_checkpoint(str(final_path))

        # Evaluate
        print("\nRunning final evaluation...")
        eval_results = trainer.evaluate(num_episodes=50 if args.quick else 200)

        # Print results
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Duration: {datetime.now() - start_time}")
        print()
        print("Evaluation Results:")
        print(f"  Mean reward: {eval_results['mean_reward']:.2f}")
        print(f"  Mean steps: {eval_results['mean_steps']:.1f}")
        print(f"  Success rate: {eval_results['success_rate']:.1%}")
        print(f"  Fast usage rate: {eval_results['fast_usage_rate']:.1%}")

        # Save results
        results_path = checkpoint_dir / "training_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "config": {
                    "quick_mode": args.quick,
                    "device": args.device,
                    "seed": args.seed,
                },
                "evaluation": eval_results,
                "duration_seconds": (datetime.now() - start_time).total_seconds(),
            }, f, indent=2)

        print(f"\nResults saved to {results_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        interrupt_path = checkpoint_dir / "interrupted_checkpoint.pt"
        trainer.save_checkpoint(str(interrupt_path))
        print(f"Checkpoint saved to {interrupt_path}")


if __name__ == "__main__":
    main()
