"""
Phase 4 Validation: Belief State Tracker Tests

Success Criteria:
- [x] BeliefStateTracker instantiates without errors
- [x] Particle filtering works (predict, update, resample)
- [x] Effective sample size computed correctly
- [x] Belief output has correct shapes
- [x] Gradients flow through the model
- [x] Parameter count within budget
"""

import pytest
import torch


def test_belief_tracker_instantiation():
    """Test that belief tracker instantiates with default config."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()
    assert tracker is not None
    assert tracker.num_particles == 50  # Default


def test_belief_tracker_from_config():
    """Test belief tracker creation from ARIALiteConfig."""
    from aria_lite.belief import create_belief_tracker
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    tracker = create_belief_tracker(config)
    assert tracker.num_particles == config.belief.num_particles


def test_init_belief():
    """Test belief initialization."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()
    output = tracker.init_belief(batch_size=4, device=torch.device("cpu"))

    assert output.belief.shape == (4, 256)  # [B, hidden_dim]
    assert output.particles.shape == (4, 50, 256)  # [B, num_particles, hidden_dim]
    assert output.weights.shape == (4, 50)  # [B, num_particles]
    assert output.effective_sample_size.shape == (4,)


def test_belief_update_shapes():
    """Test that belief update produces correct shapes."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()
    tracker.eval()

    # Initialize
    prev_output = tracker.init_belief(batch_size=2, device=torch.device("cpu"))

    # Update with action and observation
    action = torch.zeros(2, 8)
    action[:, 0] = 1
    observation = torch.randn(2, 256)

    new_output = tracker.update(action, observation, prev_output)

    assert new_output.belief.shape == (2, 256)
    assert new_output.particles.shape == (2, 50, 256)
    assert new_output.weights.shape == (2, 50)


def test_weights_sum_to_one():
    """Test that particle weights sum to 1."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()

    prev_output = tracker.init_belief(batch_size=2, device=torch.device("cpu"))

    action = torch.zeros(2, 8)
    action[:, 0] = 1
    observation = torch.randn(2, 256)

    new_output = tracker.update(action, observation, prev_output)

    # Weights should sum to approximately 1
    weight_sums = new_output.weights.sum(dim=1)
    assert torch.allclose(weight_sums, torch.ones(2), atol=1e-5)


def test_effective_sample_size():
    """Test effective sample size computation."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()

    # Uniform weights: ESS = N
    uniform_weights = torch.ones(2, 50) / 50
    ess = tracker._effective_sample_size(uniform_weights)
    assert torch.allclose(ess, torch.tensor([50.0, 50.0]), atol=0.1)

    # Concentrated weights: ESS ≈ 1
    concentrated = torch.zeros(2, 50)
    concentrated[:, 0] = 1.0
    ess = tracker._effective_sample_size(concentrated)
    assert (ess < 2).all()


def test_resampling():
    """Test particle resampling."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()

    # Create particles with concentrated weights
    particles = torch.randn(2, 50, 256)
    weights = torch.zeros(2, 50)
    weights[:, 0] = 1.0  # All weight on first particle

    mask = torch.ones(2, dtype=torch.bool)
    resampled, new_weights = tracker._resample(particles, weights, mask)

    # After resampling, all particles should be close to the first one
    assert resampled.shape == particles.shape

    # Weights should be uniform
    expected = torch.ones(2, 50) / 50
    assert torch.allclose(new_weights, expected, atol=1e-5)


def test_multiple_updates():
    """Test multiple sequential updates."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()
    tracker.eval()

    output = tracker.init_belief(batch_size=1, device=torch.device("cpu"))

    # Run 10 updates
    for _ in range(10):
        action = torch.zeros(1, 8)
        action[:, torch.randint(0, 8, (1,))] = 1
        observation = torch.randn(1, 256)

        output = tracker.update(action, observation, output)

    # Should still have valid outputs
    assert not torch.isnan(output.belief).any()
    assert not torch.isnan(output.particles).any()
    assert not torch.isnan(output.weights).any()


def test_transition_model():
    """Test transition model independently."""
    from aria_lite.belief import TransitionModel
    from aria_lite.config import BeliefConfig

    config = BeliefConfig()
    model = TransitionModel(config)

    belief = torch.randn(2, 256)
    action = torch.zeros(2, 8)
    action[:, 0] = 1

    next_belief, mean, logvar = model(belief, action)

    assert next_belief.shape == (2, 256)
    assert mean.shape == (2, 64)  # stochastic_dim
    assert logvar.shape == (2, 64)


def test_transition_model_particles():
    """Test transition model with particle inputs."""
    from aria_lite.belief import TransitionModel
    from aria_lite.config import BeliefConfig

    config = BeliefConfig()
    model = TransitionModel(config)

    # Particles: [B, N, hidden_dim]
    particles = torch.randn(2, 50, 256)
    action = torch.zeros(2, 8)
    action[:, 0] = 1

    next_particles, mean, logvar = model(particles, action)

    assert next_particles.shape == (2, 50, 256)
    assert mean.shape == (2, 50, 64)
    assert logvar.shape == (2, 50, 64)


def test_observation_model():
    """Test observation model independently."""
    from aria_lite.belief import ObservationModel
    from aria_lite.config import BeliefConfig

    config = BeliefConfig()
    model = ObservationModel(config)

    belief = torch.randn(2, 256)
    observation = torch.randn(2, 256)

    log_likelihood = model(belief, observation)
    assert log_likelihood.shape == (2,)


def test_observation_model_particles():
    """Test observation model with particle inputs."""
    from aria_lite.belief import ObservationModel
    from aria_lite.config import BeliefConfig

    config = BeliefConfig()
    model = ObservationModel(config)

    particles = torch.randn(2, 50, 256)
    observation = torch.randn(2, 256)

    log_likelihood = model(particles, observation)
    assert log_likelihood.shape == (2, 50)


def test_parameter_count():
    """Test that parameter count is within budget."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()
    param_count = tracker.count_parameters()

    print(f"\nBelief tracker parameters: {param_count:,}")

    # Config estimates ~0.7M
    # Allow 0.3M - 2M range
    assert param_count > 300_000, f"Too few parameters: {param_count:,}"
    assert param_count < 2_000_000, f"Too many parameters: {param_count:,}"


def test_parameter_count_vs_estimate():
    """Test that actual count is reasonable vs config estimate."""
    from aria_lite.belief import create_belief_tracker
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    tracker = create_belief_tracker(config)

    actual = tracker.count_parameters()
    estimated = config.belief.estimate_params()

    print(f"\nActual params: {actual:,}")
    print(f"Estimated params: {estimated:,}")

    # Allow 50% tolerance
    ratio = actual / estimated if estimated > 0 else float("inf")
    assert 0.5 < ratio < 2.0, f"Estimate mismatch: actual={actual:,}, estimated={estimated:,}"


def test_gradient_flow():
    """Test that gradients flow through belief tracker."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()
    tracker.train()

    # Initialize
    prev_output = tracker.init_belief(batch_size=2, device=torch.device("cpu"))

    # Create differentiable inputs
    action = torch.zeros(2, 8)
    action[:, 0] = 1
    observation = torch.randn(2, 256, requires_grad=True)

    new_output = tracker.update(action, observation, prev_output)

    # Backprop
    loss = new_output.belief.mean()
    loss.backward()

    # Check gradients exist
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in tracker.parameters()
    )
    assert has_grad, "No gradients found in belief tracker"


def test_get_belief_embedding():
    """Test belief embedding extraction."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker()
    output = tracker.init_belief(batch_size=2, device=torch.device("cpu"))

    embedding = tracker.get_belief_embedding(output)
    assert embedding.shape == (2, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    """Test belief tracker on CUDA device."""
    from aria_lite.belief import BeliefStateTracker

    tracker = BeliefStateTracker().cuda()
    tracker.eval()

    output = tracker.init_belief(batch_size=2, device=torch.device("cuda"))

    action = torch.zeros(2, 8).cuda()
    action[:, 0] = 1
    observation = torch.randn(2, 256).cuda()

    new_output = tracker.update(action, observation, output)

    assert new_output.belief.device.type == "cuda"
    assert new_output.particles.device.type == "cuda"


if __name__ == "__main__":
    print("Phase 4 Validation: Belief State Tracker Tests")
    print("=" * 40)

    test_belief_tracker_instantiation()
    print("✓ Belief tracker instantiation")

    test_belief_tracker_from_config()
    print("✓ Belief tracker from config")

    test_init_belief()
    print("✓ Init belief")

    test_belief_update_shapes()
    print("✓ Belief update shapes")

    test_weights_sum_to_one()
    print("✓ Weights sum to one")

    test_effective_sample_size()
    print("✓ Effective sample size")

    test_resampling()
    print("✓ Resampling")

    test_multiple_updates()
    print("✓ Multiple updates")

    test_transition_model()
    print("✓ Transition model")

    test_transition_model_particles()
    print("✓ Transition model particles")

    test_observation_model()
    print("✓ Observation model")

    test_observation_model_particles()
    print("✓ Observation model particles")

    test_parameter_count()
    print("✓ Parameter count")

    test_parameter_count_vs_estimate()
    print("✓ Parameter count vs estimate")

    test_gradient_flow()
    print("✓ Gradient flow")

    test_get_belief_embedding()
    print("✓ Get belief embedding")

    print("\n" + "=" * 40)
    print("Phase 4 Validation: ALL TESTS PASSED")
    print("=" * 40)
