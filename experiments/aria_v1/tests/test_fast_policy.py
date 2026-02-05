"""
Phase 5 Validation: Fast Policy Tests

Success Criteria:
- [x] FastPolicy instantiates without errors
- [x] Forward pass produces correct shapes
- [x] Action sampling works (deterministic and stochastic)
- [x] Confidence output is in [0, 1]
- [x] Coordinate prediction works
- [x] Log probability computation is correct
- [x] Entropy computation is correct
- [x] Parameter count within budget
- [x] Gradients flow properly
"""

import pytest
import torch


def test_fast_policy_instantiation():
    """Test that fast policy instantiates with default config."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    assert policy is not None
    assert policy.num_actions == 8


def test_fast_policy_from_config():
    """Test fast policy creation from ARIALiteConfig."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.fast_policy import create_fast_policy

    config = ARIALiteConfig()
    policy = create_fast_policy(config)
    assert policy.num_actions == config.fast_policy.num_actions


def test_forward_shape():
    """Test that forward pass produces correct output shapes."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    output = policy(state)

    assert output.action_logits.shape == (4, 8)
    assert output.action_probs.shape == (4, 8)
    assert output.confidence.shape == (4,)
    assert output.x_logits.shape == (4, 64)  # grid_size
    assert output.y_logits.shape == (4, 64)


def test_action_probs_sum_to_one():
    """Test that action probabilities sum to 1."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    output = policy(state)

    sums = output.action_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_confidence_range():
    """Test that confidence is in [0, 1]."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(10, 256)
    output = policy(state)

    assert (output.confidence >= 0).all()
    assert (output.confidence <= 1).all()


def test_get_action_deterministic():
    """Test deterministic action selection."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)

    # Deterministic should give same result
    with torch.no_grad():
        action1, _ = policy.get_action(state, deterministic=True)
        action2, _ = policy.get_action(state, deterministic=True)

    assert (action1 == action2).all()


def test_get_action_stochastic():
    """Test stochastic action selection."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(100, 256)

    # Stochastic should produce variety
    with torch.no_grad():
        actions, _ = policy.get_action(state, deterministic=False)

    # Should have multiple different actions
    unique_actions = actions.unique()
    assert len(unique_actions) > 1


def test_get_coordinates():
    """Test coordinate prediction."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)

    with torch.no_grad():
        x, y = policy.get_coordinates(state)

    assert x.shape == (4,)
    assert y.shape == (4,)
    assert (x >= 0).all() and (x < 64).all()
    assert (y >= 0).all() and (y < 64).all()


def test_log_prob():
    """Test log probability computation."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    action = torch.randint(0, 8, (4,))

    log_prob = policy.log_prob(state, action)

    assert log_prob.shape == (4,)
    # Log probs should be negative
    assert (log_prob <= 0).all()


def test_log_prob_consistency():
    """Test that log_prob is consistent with action_probs."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    action = torch.randint(0, 8, (4,))

    with torch.no_grad():
        output = policy(state)
        log_prob = policy.log_prob(state, action)

    # log_prob should match log of gathered action_prob
    expected_log_prob = torch.log(output.action_probs.gather(1, action.unsqueeze(-1))).squeeze(-1)
    assert torch.allclose(log_prob, expected_log_prob, atol=1e-5)


def test_entropy():
    """Test entropy computation."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)

    entropy = policy.entropy(state)

    assert entropy.shape == (4,)
    # Entropy should be non-negative
    assert (entropy >= 0).all()


def test_entropy_bounds():
    """Test that entropy is within valid bounds."""
    import math

    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    entropy = policy.entropy(state)

    # Max entropy for 8 actions is log(8)
    max_entropy = math.log(8)
    assert (entropy <= max_entropy + 0.01).all()  # Small tolerance


def test_temperature_effect():
    """Test that temperature affects output distribution."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.eval()

    state = torch.randn(4, 256)

    with torch.no_grad():
        output_low_temp = policy(state, temperature=0.1)
        output_high_temp = policy(state, temperature=2.0)

    # Lower temperature should be more peaked (lower entropy)
    low_entropy = -(output_low_temp.action_probs * torch.log(output_low_temp.action_probs + 1e-10)).sum(dim=-1)
    high_entropy = -(output_high_temp.action_probs * torch.log(output_high_temp.action_probs + 1e-10)).sum(dim=-1)

    assert (low_entropy < high_entropy).all()


def test_parameter_count():
    """Test that parameter count is within budget."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    param_count = policy.count_parameters()

    print(f"\nFast policy parameters: {param_count:,}")

    # Config estimates ~0.4M
    # Allow 0.1M - 1M range
    assert param_count > 100_000, f"Too few parameters: {param_count:,}"
    assert param_count < 1_000_000, f"Too many parameters: {param_count:,}"


def test_parameter_count_vs_estimate():
    """Test that actual count is reasonable vs config estimate."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.fast_policy import create_fast_policy

    config = ARIALiteConfig()
    policy = create_fast_policy(config)

    actual = policy.count_parameters()
    estimated = config.fast_policy.estimate_params()

    print(f"\nActual params: {actual:,}")
    print(f"Estimated params: {estimated:,}")

    # Allow 50% tolerance
    ratio = actual / estimated if estimated > 0 else float("inf")
    assert 0.5 < ratio < 2.0, f"Estimate mismatch: actual={actual:,}, estimated={estimated:,}"


def test_gradient_flow():
    """Test that gradients flow through fast policy."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy()
    policy.train()

    state = torch.randn(4, 256, requires_grad=True)
    output = policy(state)

    # Backprop through multiple outputs
    loss = (
        output.action_logits.mean()
        + output.confidence.mean()
        + output.x_logits.mean()
        + output.y_logits.mean()
    )
    loss.backward()

    # Check gradients exist
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in policy.parameters()
    )
    assert has_grad, "No gradients found in fast policy"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    """Test fast policy on CUDA device."""
    from aria_lite.fast_policy import FastPolicy

    policy = FastPolicy().cuda()
    policy.eval()

    state = torch.randn(4, 256).cuda()
    output = policy(state)

    assert output.action_logits.device.type == "cuda"
    assert output.confidence.device.type == "cuda"


if __name__ == "__main__":
    print("Phase 5 Validation: Fast Policy Tests")
    print("=" * 40)

    test_fast_policy_instantiation()
    print("✓ Fast policy instantiation")

    test_fast_policy_from_config()
    print("✓ Fast policy from config")

    test_forward_shape()
    print("✓ Forward shape")

    test_action_probs_sum_to_one()
    print("✓ Action probs sum to one")

    test_confidence_range()
    print("✓ Confidence range")

    test_get_action_deterministic()
    print("✓ Get action deterministic")

    test_get_action_stochastic()
    print("✓ Get action stochastic")

    test_get_coordinates()
    print("✓ Get coordinates")

    test_log_prob()
    print("✓ Log prob")

    test_log_prob_consistency()
    print("✓ Log prob consistency")

    test_entropy()
    print("✓ Entropy")

    test_entropy_bounds()
    print("✓ Entropy bounds")

    test_temperature_effect()
    print("✓ Temperature effect")

    test_parameter_count()
    print("✓ Parameter count")

    test_parameter_count_vs_estimate()
    print("✓ Parameter count vs estimate")

    test_gradient_flow()
    print("✓ Gradient flow")

    print("\n" + "=" * 40)
    print("Phase 5 Validation: ALL TESTS PASSED")
    print("=" * 40)
