"""
Phase 6 Validation: Slow Policy Tests

Success Criteria:
- [x] SlowPolicy instantiates without errors
- [x] Forward pass produces correct shapes
- [x] Transformer layers work correctly
- [x] Value and uncertainty heads work
- [x] Action sampling works
- [x] PPO-style evaluation works
- [x] Parameter count within budget
- [x] Gradients flow through transformer
"""

import pytest
import torch


def test_slow_policy_instantiation():
    """Test that slow policy instantiates with default config."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    assert policy is not None
    assert policy.num_actions == 8


def test_slow_policy_from_config():
    """Test slow policy creation from ARIALiteConfig."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.slow_policy import create_slow_policy

    config = ARIALiteConfig()
    policy = create_slow_policy(config)
    assert policy.num_actions == config.slow_policy.num_actions


def test_forward_shape():
    """Test that forward pass produces correct output shapes."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)

    output = policy(state, belief)

    assert output.action_logits.shape == (4, 8)
    assert output.action_probs.shape == (4, 8)
    assert output.value.shape == (4,)
    assert output.uncertainty.shape == (4,)


def test_forward_with_goal():
    """Test forward pass with goal input."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)
    goal = torch.randn(4, 64)

    output = policy(state, belief, goal)

    assert output.action_logits.shape == (4, 8)
    assert output.value.shape == (4,)


def test_action_probs_sum_to_one():
    """Test that action probabilities sum to 1."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)

    output = policy(state, belief)

    sums = output.action_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_value_and_uncertainty_ranges():
    """Test value and uncertainty outputs."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(10, 256)
    belief = torch.randn(10, 256)

    output = policy(state, belief)

    # Uncertainty should be in [0, 1] (sigmoid)
    assert (output.uncertainty >= 0).all()
    assert (output.uncertainty <= 1).all()

    # Value can be any real number
    assert not torch.isnan(output.value).any()


def test_get_action_deterministic():
    """Test deterministic action selection."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)

    with torch.no_grad():
        action1, _ = policy.get_action(state, belief, deterministic=True)
        action2, _ = policy.get_action(state, belief, deterministic=True)

    assert (action1 == action2).all()


def test_get_action_stochastic():
    """Test stochastic action selection."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(100, 256)
    belief = torch.randn(100, 256)

    with torch.no_grad():
        actions, _ = policy.get_action(state, belief, deterministic=False)

    unique_actions = actions.unique()
    assert len(unique_actions) > 1


def test_plan():
    """Test planning method."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)
    goal = torch.randn(4, 64)

    action, output = policy.plan(state, belief, goal)

    assert action.shape == (4,)
    assert (action >= 0).all() and (action < 8).all()


def test_evaluate_action():
    """Test PPO-style action evaluation."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)
    action = torch.randint(0, 8, (4,))

    log_prob, value, entropy = policy.evaluate_action(state, belief, action)

    assert log_prob.shape == (4,)
    assert value.shape == (4,)
    assert entropy.shape == (4,)

    # Log probs should be negative
    assert (log_prob <= 0).all()
    # Entropy should be non-negative
    assert (entropy >= 0).all()


def test_log_prob():
    """Test log probability computation."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)
    action = torch.randint(0, 8, (4,))

    log_prob = policy.log_prob(state, belief, action)

    assert log_prob.shape == (4,)
    assert (log_prob <= 0).all()


def test_entropy():
    """Test entropy computation."""
    import math

    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)

    entropy = policy.entropy(state, belief)

    assert entropy.shape == (4,)
    assert (entropy >= 0).all()
    assert (entropy <= math.log(8) + 0.01).all()


def test_return_hidden_states():
    """Test returning hidden states."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)

    output = policy(state, belief, return_hidden=True)

    assert output.hidden_states is not None
    assert output.hidden_states.shape[0] == 4


def test_temperature_effect():
    """Test that temperature affects output distribution."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.eval()

    state = torch.randn(4, 256)
    belief = torch.randn(4, 256)

    with torch.no_grad():
        output_low_temp = policy(state, belief, temperature=0.1)
        output_high_temp = policy(state, belief, temperature=2.0)

    # Lower temperature should be more peaked
    low_entropy = -(output_low_temp.action_probs * torch.log(output_low_temp.action_probs + 1e-10)).sum(dim=-1)
    high_entropy = -(output_high_temp.action_probs * torch.log(output_high_temp.action_probs + 1e-10)).sum(dim=-1)

    assert (low_entropy < high_entropy).all()


def test_parameter_count():
    """Test that parameter count is within budget."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    param_count = policy.count_parameters()

    print(f"\nSlow policy parameters: {param_count:,}")

    # Config estimates ~8.5M
    # Allow 4M - 15M range
    assert param_count > 4_000_000, f"Too few parameters: {param_count:,}"
    assert param_count < 15_000_000, f"Too many parameters: {param_count:,}"


def test_parameter_count_vs_estimate():
    """Test that actual count is reasonable vs config estimate."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.slow_policy import create_slow_policy

    config = ARIALiteConfig()
    policy = create_slow_policy(config)

    actual = policy.count_parameters()
    estimated = config.slow_policy.estimate_params()

    print(f"\nActual params: {actual:,}")
    print(f"Estimated params: {estimated:,}")

    # Allow 50% tolerance
    ratio = actual / estimated if estimated > 0 else float("inf")
    assert 0.5 < ratio < 2.0, f"Estimate mismatch: actual={actual:,}, estimated={estimated:,}"


def test_gradient_flow():
    """Test that gradients flow through slow policy."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy()
    policy.train()

    state = torch.randn(4, 256, requires_grad=True)
    belief = torch.randn(4, 256, requires_grad=True)
    goal = torch.randn(4, 64, requires_grad=True)

    output = policy(state, belief, goal)

    # Backprop through all outputs
    loss = (
        output.action_logits.mean()
        + output.value.mean()
        + output.uncertainty.mean()
    )
    loss.backward()

    # Check gradients exist in transformer
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.transformer.parameters()
    )
    assert has_grad, "No gradients found in transformer"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    """Test slow policy on CUDA device."""
    from aria_lite.slow_policy import SlowPolicy

    policy = SlowPolicy().cuda()
    policy.eval()

    state = torch.randn(4, 256).cuda()
    belief = torch.randn(4, 256).cuda()

    output = policy(state, belief)

    assert output.action_logits.device.type == "cuda"
    assert output.value.device.type == "cuda"


if __name__ == "__main__":
    print("Phase 6 Validation: Slow Policy Tests")
    print("=" * 40)

    test_slow_policy_instantiation()
    print("✓ Slow policy instantiation")

    test_slow_policy_from_config()
    print("✓ Slow policy from config")

    test_forward_shape()
    print("✓ Forward shape")

    test_forward_with_goal()
    print("✓ Forward with goal")

    test_action_probs_sum_to_one()
    print("✓ Action probs sum to one")

    test_value_and_uncertainty_ranges()
    print("✓ Value and uncertainty ranges")

    test_get_action_deterministic()
    print("✓ Get action deterministic")

    test_get_action_stochastic()
    print("✓ Get action stochastic")

    test_plan()
    print("✓ Plan")

    test_evaluate_action()
    print("✓ Evaluate action")

    test_log_prob()
    print("✓ Log prob")

    test_entropy()
    print("✓ Entropy")

    test_return_hidden_states()
    print("✓ Return hidden states")

    test_temperature_effect()
    print("✓ Temperature effect")

    test_parameter_count()
    print("✓ Parameter count")

    test_parameter_count_vs_estimate()
    print("✓ Parameter count vs estimate")

    test_gradient_flow()
    print("✓ Gradient flow")

    print("\n" + "=" * 40)
    print("Phase 6 Validation: ALL TESTS PASSED")
    print("=" * 40)
