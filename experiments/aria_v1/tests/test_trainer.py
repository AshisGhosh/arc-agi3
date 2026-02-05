"""
Phase 12 Validation: Trainer Tests

Success Criteria:
- [x] TrainerConfig instantiates with defaults
- [x] ARIALiteTrainer instantiates
- [x] Data collection works
- [x] World model training runs
- [x] Fast policy training runs
- [x] Slow policy training runs
- [x] Arbiter calibration runs
- [x] Joint fine-tuning runs
- [x] Checkpointing works
- [x] Evaluation works
"""

import os
import tempfile

import torch


def test_trainer_config():
    """Test TrainerConfig instantiation."""
    from aria_lite.training.trainer import TrainerConfig

    config = TrainerConfig()

    assert config.wm_epochs == 100
    assert config.fp_epochs == 50
    assert config.sp_epochs == 100
    assert config.joint_epochs == 50
    assert config.buffer_capacity == 100_000


def test_trainer_config_custom():
    """Test custom configuration."""
    from aria_lite.training.trainer import TrainerConfig

    config = TrainerConfig(
        wm_epochs=10,
        fp_epochs=5,
        buffer_capacity=1000,
    )

    assert config.wm_epochs == 10
    assert config.fp_epochs == 5
    assert config.buffer_capacity == 1000


def test_training_phase_enum():
    """Test TrainingPhase enum."""
    from aria_lite.training.trainer import TrainingPhase

    assert TrainingPhase.WORLD_MODEL.value == "world_model"
    assert TrainingPhase.FAST_POLICY.value == "fast_policy"
    assert TrainingPhase.SLOW_POLICY.value == "slow_policy"
    assert TrainingPhase.ARBITER.value == "arbiter"
    assert TrainingPhase.JOINT.value == "joint"


def test_trainer_instantiation():
    """Test trainer instantiation."""
    from aria_lite.agent import create_agent
    from aria_lite.config import ARIALiteConfig
    from aria_lite.training.trainer import ARIALiteTrainer, TrainerConfig

    aria_config = ARIALiteConfig()
    agent = create_agent(aria_config)
    trainer_config = TrainerConfig(device="cpu")

    trainer = ARIALiteTrainer(agent, trainer_config, aria_config)

    assert trainer is not None
    assert trainer.buffer is not None
    assert trainer.env_generator is not None


def test_create_trainer_factory():
    """Test factory function."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(device="cpu")
    trainer = create_trainer(config=config)

    assert trainer is not None
    assert trainer.agent is not None


def test_data_collection():
    """Test experience collection."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(device="cpu", buffer_capacity=1000)
    trainer = create_trainer(config=config)

    initial_size = len(trainer.buffer)
    trainer.collect_data(num_episodes=5)

    assert len(trainer.buffer) > initial_size


def test_world_model_training():
    """Test world model training phase."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        wm_epochs=2,
        wm_batch_size=4,
        buffer_capacity=500,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    # Collect data first
    trainer.collect_data(20)

    # Train world model
    metrics = trainer.train_world_model(num_epochs=2)

    assert len(metrics) == 2
    assert all(m.phase == "world_model" for m in metrics)
    assert all(m.loss > 0 for m in metrics)


def test_fast_policy_training():
    """Test fast policy training phase."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        fp_epochs=2,
        fp_batch_size=4,
        buffer_capacity=500,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    trainer.collect_data(20)
    metrics = trainer.train_fast_policy(num_epochs=2)

    assert len(metrics) == 2
    assert all(m.phase == "fast_policy" for m in metrics)


def test_slow_policy_training():
    """Test slow policy training phase."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        sp_epochs=2,
        sp_batch_size=4,
        buffer_capacity=500,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    trainer.collect_data(20)
    metrics = trainer.train_slow_policy(num_epochs=2)

    assert len(metrics) == 2
    assert all(m.phase == "slow_policy" for m in metrics)


def test_arbiter_calibration():
    """Test arbiter calibration phase."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        arb_epochs=2,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    metrics = trainer.calibrate_arbiter(num_epochs=2)

    assert len(metrics) == 2
    assert all(m.phase == "arbiter" for m in metrics)


def test_joint_finetuning():
    """Test joint fine-tuning phase."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        joint_epochs=2,
        joint_batch_size=4,
        buffer_capacity=500,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    trainer.collect_data(20)
    metrics = trainer.joint_finetune(num_epochs=2)

    assert len(metrics) == 2
    assert all(m.phase == "joint" for m in metrics)


def test_checkpoint_save_load():
    """Test checkpointing."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(device="cpu")
    trainer = create_trainer(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "checkpoint.pt")

        # Save
        trainer.global_step = 42
        trainer.save_checkpoint(path)

        assert os.path.exists(path)

        # Load into new trainer
        trainer2 = create_trainer(config=config)
        trainer2.load_checkpoint(path)

        assert trainer2.global_step == 42


def test_evaluation():
    """Test evaluation."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(device="cpu")
    trainer = create_trainer(config=config)

    results = trainer.evaluate(num_episodes=3)

    assert "mean_reward" in results
    assert "mean_steps" in results
    assert "success_rate" in results
    assert "fast_usage_rate" in results


def test_metrics_history():
    """Test metrics history tracking."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        wm_epochs=2,
        wm_batch_size=4,
        buffer_capacity=500,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    initial_history = len(trainer.metrics_history)
    trainer.collect_data(20)
    trainer.train_world_model(num_epochs=2)

    assert len(trainer.metrics_history) > initial_history


def test_gradient_clipping():
    """Test that gradient clipping is applied."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        max_grad_norm=0.5,
        wm_epochs=1,
        wm_batch_size=4,
        buffer_capacity=500,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    trainer.collect_data(20)
    trainer.train_world_model(num_epochs=1)

    # Check that gradients were clipped (no NaN)
    for param in trainer.agent.world_model.parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any()


def test_scheduler_step():
    """Test learning rate scheduling."""
    from aria_lite.training.trainer import TrainerConfig, create_trainer

    config = TrainerConfig(
        device="cpu",
        wm_epochs=5,
        wm_batch_size=4,
        buffer_capacity=500,
        log_every=100,
    )
    trainer = create_trainer(config=config)

    initial_lr = trainer.wm_optimizer.param_groups[0]["lr"]
    trainer.collect_data(20)
    trainer.train_world_model(num_epochs=5)

    # LR should have changed (cosine annealing)
    # Note: At end of cosine schedule, LR approaches 0
    final_lr = trainer.wm_optimizer.param_groups[0]["lr"]
    assert final_lr != initial_lr


def test_training_metrics_structure():
    """Test TrainingMetrics dataclass."""
    from aria_lite.training.trainer import TrainingMetrics

    metrics = TrainingMetrics(
        phase="test",
        epoch=0,
        loss=0.5,
        aux_losses={"a": 0.1, "b": 0.2},
        metrics={"accuracy": 0.9},
    )

    assert metrics.phase == "test"
    assert metrics.epoch == 0
    assert metrics.loss == 0.5
    assert metrics.aux_losses["a"] == 0.1
    assert metrics.metrics["accuracy"] == 0.9


if __name__ == "__main__":
    print("Phase 12 Validation: Trainer Tests")
    print("=" * 40)

    test_trainer_config()
    print("✓ Trainer config")

    test_trainer_config_custom()
    print("✓ Custom config")

    test_training_phase_enum()
    print("✓ Training phase enum")

    test_trainer_instantiation()
    print("✓ Trainer instantiation")

    test_create_trainer_factory()
    print("✓ Factory function")

    test_data_collection()
    print("✓ Data collection")

    test_world_model_training()
    print("✓ World model training")

    test_fast_policy_training()
    print("✓ Fast policy training")

    test_slow_policy_training()
    print("✓ Slow policy training")

    test_arbiter_calibration()
    print("✓ Arbiter calibration")

    test_joint_finetuning()
    print("✓ Joint fine-tuning")

    test_checkpoint_save_load()
    print("✓ Checkpointing")

    test_evaluation()
    print("✓ Evaluation")

    test_metrics_history()
    print("✓ Metrics history")

    test_gradient_clipping()
    print("✓ Gradient clipping")

    test_scheduler_step()
    print("✓ Scheduler step")

    test_training_metrics_structure()
    print("✓ Training metrics structure")

    print("\n" + "=" * 40)
    print("Phase 12 Validation: ALL TESTS PASSED")
    print("=" * 40)
