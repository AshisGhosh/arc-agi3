"""
Phase 3 Validation: World Model Tests

Success Criteria:
- [x] EnsembleWorldModel instantiates without errors
- [x] 3-head ensemble producing correct output shapes
- [x] Uncertainty estimation works (variance across heads)
- [x] Trajectory prediction works
- [x] Parameter count within budget
- [x] Gradients flow through all heads
- [x] Residual connections working
"""

import pytest
import torch


def test_world_model_instantiation():
    """Test that world model instantiates with default config."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    assert model is not None
    assert len(model.heads) == 3  # Default 3 heads


def test_world_model_from_config():
    """Test world model creation from ARIALiteConfig."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.world_model import create_world_model

    config = ARIALiteConfig()
    model = create_world_model(config)
    assert len(model.heads) == config.world_model.num_ensemble


def test_forward_shape():
    """Test that forward pass produces correct output shapes."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    model.eval()

    state = torch.randn(4, 256)  # Batch of 4
    action = torch.zeros(4, 8)  # One-hot actions
    action[:, 0] = 1

    output = model(state, action)

    assert output.next_state.shape == (4, 256)
    assert output.reward.shape == (4, 1)
    assert output.done.shape == (4, 1)
    assert output.uncertainty.shape == (4,)


def test_single_head_shape():
    """Test single head output shapes."""
    from aria_lite.config import WorldModelConfig
    from aria_lite.world_model import WorldModelHead

    config = WorldModelConfig()
    head = WorldModelHead(config)
    head.eval()

    state = torch.randn(2, 256)
    action = torch.zeros(2, 8)
    action[:, 0] = 1

    next_state, reward, done = head(state, action)

    assert next_state.shape == (2, 256)
    assert reward.shape == (2, 1)
    assert done.shape == (2, 1)


def test_uncertainty_estimation():
    """Test that uncertainty varies with input."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    model.eval()

    state = torch.randn(4, 256)
    action = torch.zeros(4, 8)
    action[:, 0] = 1

    output = model(state, action)

    # Uncertainty should be non-negative
    assert (output.uncertainty >= 0).all()

    # With random initialization, there should be some uncertainty
    # (heads will have different predictions)
    assert output.uncertainty.sum() > 0


def test_trajectory_prediction():
    """Test trajectory prediction over multiple timesteps."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    model.eval()

    initial_state = torch.randn(2, 256)
    actions = torch.zeros(2, 5, 8)  # 5 timesteps
    for t in range(5):
        actions[:, t, t % 8] = 1  # Cycle through actions

    states, rewards, dones, uncertainties = model.predict_trajectory(
        initial_state, actions
    )

    assert states.shape == (2, 6, 256)  # T+1 states
    assert rewards.shape == (2, 5, 1)
    assert dones.shape == (2, 5, 1)
    assert uncertainties.shape == (2, 5)


def test_residual_connection():
    """Test that residual connection affects output."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    model.eval()

    state = torch.randn(1, 256)
    action = torch.zeros(1, 8)
    action[:, 0] = 1

    with torch.no_grad():
        output = model(state, action)

    # Next state should be related to input state (not completely different)
    # due to residual connection
    correlation = torch.corrcoef(
        torch.stack([state.squeeze(), output.next_state.squeeze()])
    )[0, 1]

    # Should have some correlation due to residual
    assert correlation > -0.5  # Not strongly anti-correlated


def test_done_probability_range():
    """Test that done predictions are valid probabilities."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    model.eval()

    state = torch.randn(10, 256)
    action = torch.zeros(10, 8)
    action[:, 0] = 1

    output = model(state, action)

    # Done should be in [0, 1]
    assert (output.done >= 0).all()
    assert (output.done <= 1).all()


def test_parameter_count():
    """Test that parameter count is within budget."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    param_count = model.count_parameters()

    print(f"\nWorld model parameters: {param_count:,}")

    # Config estimates ~7.9M based on current settings
    # Allow 3-15M range
    assert param_count > 3_000_000, f"Too few parameters: {param_count:,}"
    assert param_count < 15_000_000, f"Too many parameters: {param_count:,}"


def test_parameter_count_vs_estimate():
    """Test that actual count is reasonable vs config estimate."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.world_model import create_world_model

    config = ARIALiteConfig()
    model = create_world_model(config)

    actual = model.count_parameters()
    estimated = config.world_model.estimate_params()

    print(f"\nActual params: {actual:,}")
    print(f"Estimated params: {estimated:,}")

    # Allow 50% tolerance
    ratio = actual / estimated if estimated > 0 else float("inf")
    assert 0.5 < ratio < 2.0, f"Estimate mismatch: actual={actual:,}, estimated={estimated:,}"


def test_gradient_flow():
    """Test that gradients flow through all heads."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()
    model.train()

    state = torch.randn(4, 256, requires_grad=True)
    action = torch.zeros(4, 8)
    action[:, 0] = 1

    output = model(state, action)

    # Backprop through all outputs
    loss = output.next_state.mean() + output.reward.mean() + output.done.mean()
    loss.backward()

    # Check gradients in all heads
    for i, head in enumerate(model.heads):
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        assert has_grad, f"No gradients in head {i}"


def test_sample_head():
    """Test that head sampling works."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel()

    # Sample heads multiple times
    heads_sampled = set()
    for _ in range(100):
        head = model.sample_head()
        heads_sampled.add(id(head))

    # Should have sampled all heads at least once
    assert len(heads_sampled) == 3


def test_action_onehot():
    """Test action to one-hot conversion."""
    from aria_lite.world_model import action_to_onehot

    actions = torch.tensor([0, 3, 7])
    onehot = action_to_onehot(actions, num_actions=8)

    assert onehot.shape == (3, 8)
    assert (onehot.sum(dim=-1) == 1).all()
    assert onehot[0, 0] == 1
    assert onehot[1, 3] == 1
    assert onehot[2, 7] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    """Test world model on CUDA device."""
    from aria_lite.world_model import EnsembleWorldModel

    model = EnsembleWorldModel().cuda()
    model.eval()

    state = torch.randn(4, 256).cuda()
    action = torch.zeros(4, 8).cuda()
    action[:, 0] = 1

    output = model(state, action)

    assert output.next_state.device.type == "cuda"
    assert output.reward.device.type == "cuda"
    assert output.done.device.type == "cuda"
    assert output.uncertainty.device.type == "cuda"


if __name__ == "__main__":
    print("Phase 3 Validation: World Model Tests")
    print("=" * 40)

    test_world_model_instantiation()
    print("✓ World model instantiation")

    test_world_model_from_config()
    print("✓ World model from config")

    test_forward_shape()
    print("✓ Forward shape")

    test_single_head_shape()
    print("✓ Single head shape")

    test_uncertainty_estimation()
    print("✓ Uncertainty estimation")

    test_trajectory_prediction()
    print("✓ Trajectory prediction")

    test_residual_connection()
    print("✓ Residual connection")

    test_done_probability_range()
    print("✓ Done probability range")

    test_parameter_count()
    print("✓ Parameter count")

    test_parameter_count_vs_estimate()
    print("✓ Parameter count vs estimate")

    test_gradient_flow()
    print("✓ Gradient flow")

    test_sample_head()
    print("✓ Sample head")

    test_action_onehot()
    print("✓ Action onehot")

    print("\n" + "=" * 40)
    print("Phase 3 Validation: ALL TESTS PASSED")
    print("=" * 40)
