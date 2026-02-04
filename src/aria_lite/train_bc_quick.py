#!/usr/bin/env python3
"""Quick BC training test."""

import sys
import torch

from .config import ARIALiteConfig
from .agent import create_agent
from .training.bc_trainer import BCTrainer, BCConfig
from .training.expert_data import collect_expert_dataset

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    mechanic = "navigation"
    print(f"\nCollecting expert data for {mechanic}...", flush=True)

    dataset = collect_expert_dataset(
        mechanic=mechanic,
        num_trajectories=1000,  # Smaller for quick test
        grid_size=10,
        max_steps=50,
    )

    print(f"Collected: {len(dataset.trajectories)} trajectories, "
          f"{dataset.num_transitions} transitions", flush=True)

    config = BCConfig(
        epochs=50,  # Fewer epochs
        batch_size=64,
        learning_rate=1e-3,  # Higher LR
        device=device,
        log_every=5,
        eval_every=10,
    )

    aria_config = ARIALiteConfig()
    agent = create_agent(aria_config)
    trainer = BCTrainer(agent, config)

    print("\nTraining...", flush=True)
    metrics = trainer.train(dataset, mechanic)

    print("\nFinal evaluation (100 episodes)...", flush=True)
    final_success = trainer.evaluate(mechanic, num_episodes=100)
    print(f"\nFINAL SUCCESS RATE: {final_success:.1%}", flush=True)

    if final_success > 0.5:
        print("SUCCESS: BC is working!")
    else:
        print("FAILURE: BC not learning effectively")


if __name__ == "__main__":
    main()
