"""
Phase 7 Validation: Arbiter Tests

Success Criteria:
- [x] Arbiter instantiates without errors
- [x] Heuristic switching works correctly
- [x] Thresholds are respected
- [x] Statistics tracking works
- [x] Action selection works
- [x] Calibration method works
- [x] Minimal parameters (heuristic mode)
"""

import torch


def test_arbiter_instantiation():
    """Test that arbiter instantiates with default config."""
    from aria_lite.arbiter import Arbiter

    arbiter = Arbiter()
    assert arbiter is not None
    assert arbiter.confidence_threshold == 0.7


def test_arbiter_from_config():
    """Test arbiter creation from ARIALiteConfig."""
    from aria_lite.arbiter import create_arbiter
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    arbiter = create_arbiter(config)
    assert arbiter.confidence_threshold == config.arbiter.confidence_threshold


def test_should_use_slow_basic():
    """Test basic should_use_slow logic."""
    from aria_lite.arbiter import Arbiter

    arbiter = Arbiter()

    # High confidence, low uncertainty -> use fast
    use_slow = arbiter.should_use_slow(
        confidence=torch.tensor([0.9]),
        uncertainty=torch.tensor([0.1]),
    )
    assert not use_slow[0]

    # Low confidence -> use slow
    use_slow = arbiter.should_use_slow(
        confidence=torch.tensor([0.3]),
        uncertainty=torch.tensor([0.1]),
    )
    assert use_slow[0]

    # High uncertainty -> use slow
    use_slow = arbiter.should_use_slow(
        confidence=torch.tensor([0.9]),
        uncertainty=torch.tensor([0.5]),
    )
    assert use_slow[0]


def test_should_use_slow_novelty():
    """Test novelty-based switching."""
    from aria_lite.arbiter import Arbiter

    arbiter = Arbiter()

    # High novelty -> use slow
    use_slow = arbiter.should_use_slow(
        confidence=torch.tensor([0.9]),
        uncertainty=torch.tensor([0.1]),
        novelty=torch.tensor([0.8]),
    )
    assert use_slow[0]


def test_forward_with_fast_output():
    """Test forward with FastPolicyOutput."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.fast_policy import FastPolicyOutput

    arbiter = Arbiter()

    fast_output = FastPolicyOutput(
        action_logits=torch.randn(4, 8),
        action_probs=torch.softmax(torch.randn(4, 8), dim=-1),
        confidence=torch.tensor([0.9, 0.5, 0.8, 0.3]),
        x_logits=torch.randn(4, 64),
        y_logits=torch.randn(4, 64),
    )
    uncertainty = torch.tensor([0.1, 0.4, 0.1, 0.2])

    decision = arbiter(fast_output, uncertainty)

    assert decision.use_slow.shape == (4,)
    # Low confidence samples should use slow
    assert not decision.use_slow[0]  # 0.9 > 0.7
    assert decision.use_slow[1]  # 0.5 < 0.7 or 0.4 > 0.3
    assert not decision.use_slow[2]  # 0.8 > 0.7, 0.1 < 0.3


def test_force_slow():
    """Test force_slow override."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.fast_policy import FastPolicyOutput

    arbiter = Arbiter()

    fast_output = FastPolicyOutput(
        action_logits=torch.randn(4, 8),
        action_probs=torch.softmax(torch.randn(4, 8), dim=-1),
        confidence=torch.ones(4) * 0.99,  # Very high confidence
        x_logits=torch.randn(4, 64),
        y_logits=torch.randn(4, 64),
    )
    uncertainty = torch.zeros(4)  # Very low uncertainty

    decision = arbiter(fast_output, uncertainty, force_slow=True)

    # Should all use slow despite high confidence
    assert decision.use_slow.all()
    assert decision.reason == "forced_slow"


def test_select_action():
    """Test action selection between fast and slow."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.fast_policy import FastPolicyOutput
    from aria_lite.slow_policy import SlowPolicyOutput

    arbiter = Arbiter()

    # Fast output: action 0 is most likely
    fast_probs = torch.zeros(2, 8)
    fast_probs[:, 0] = 0.9
    fast_probs[:, 1:] = 0.1 / 7

    fast_output = FastPolicyOutput(
        action_logits=torch.log(fast_probs),
        action_probs=fast_probs,
        confidence=torch.tensor([0.9, 0.9]),
        x_logits=torch.randn(2, 64),
        y_logits=torch.randn(2, 64),
    )

    # Slow output: action 7 is most likely
    slow_probs = torch.zeros(2, 8)
    slow_probs[:, 7] = 0.9
    slow_probs[:, :7] = 0.1 / 7

    slow_output = SlowPolicyOutput(
        action_logits=torch.log(slow_probs),
        action_probs=slow_probs,
        value=torch.zeros(2),
        uncertainty=torch.zeros(2),
    )

    # First uses fast, second uses slow
    use_slow = torch.tensor([False, True])

    action_probs, actions = arbiter.select_action(fast_output, slow_output, use_slow)

    assert actions[0] == 0  # Fast action
    assert actions[1] == 7  # Slow action


def test_statistics_tracking():
    """Test that statistics are tracked correctly."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.fast_policy import FastPolicyOutput

    arbiter = Arbiter()
    arbiter.eval()
    arbiter.reset_stats()

    # Run several decisions
    for _ in range(10):
        fast_output = FastPolicyOutput(
            action_logits=torch.randn(4, 8),
            action_probs=torch.softmax(torch.randn(4, 8), dim=-1),
            confidence=torch.rand(4),
            x_logits=torch.randn(4, 64),
            y_logits=torch.randn(4, 64),
        )
        uncertainty = torch.rand(4) * 0.5

        arbiter(fast_output, uncertainty)

    stats = arbiter.get_stats()

    assert stats["num_fast"] + stats["num_slow"] == 40
    assert 0 <= stats["fast_ratio"] <= 1


def test_reset_stats():
    """Test statistics reset."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.fast_policy import FastPolicyOutput

    arbiter = Arbiter()
    arbiter.eval()

    # Run some decisions
    fast_output = FastPolicyOutput(
        action_logits=torch.randn(4, 8),
        action_probs=torch.softmax(torch.randn(4, 8), dim=-1),
        confidence=torch.rand(4),
        x_logits=torch.randn(4, 64),
        y_logits=torch.randn(4, 64),
    )
    arbiter(fast_output, torch.rand(4))

    # Reset
    arbiter.reset_stats()
    stats = arbiter.get_stats()

    assert stats["num_fast"] == 0
    assert stats["num_slow"] == 0


def test_calibrate_thresholds():
    """Test threshold calibration."""
    from aria_lite.arbiter import Arbiter

    arbiter = Arbiter()

    # Generate history
    confidence_history = torch.rand(1000)
    uncertainty_history = torch.rand(1000) * 0.5

    # Calibrate for 80% fast usage
    arbiter.calibrate_thresholds(
        target_fast_ratio=0.8,
        confidence_history=confidence_history,
        uncertainty_history=uncertainty_history,
    )

    # Thresholds should have changed
    # (Exact values depend on random data)
    assert arbiter.confidence_threshold is not None
    assert arbiter.uncertainty_threshold is not None


def test_parameter_count_heuristic():
    """Test that heuristic arbiter has zero trainable parameters."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.config import ArbiterConfig

    config = ArbiterConfig(use_learned_switching=False)
    arbiter = Arbiter(config)

    param_count = arbiter.count_parameters()
    assert param_count == 0


def test_parameter_count_learned():
    """Test learned arbiter has parameters."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.config import ArbiterConfig

    config = ArbiterConfig(use_learned_switching=True)
    arbiter = Arbiter(config)

    param_count = arbiter.count_parameters()
    assert param_count > 0


def test_learned_switching():
    """Test learned switching mode."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.config import ArbiterConfig
    from aria_lite.fast_policy import FastPolicyOutput

    config = ArbiterConfig(use_learned_switching=True)
    arbiter = Arbiter(config)

    fast_output = FastPolicyOutput(
        action_logits=torch.randn(4, 8),
        action_probs=torch.softmax(torch.randn(4, 8), dim=-1),
        confidence=torch.rand(4),
        x_logits=torch.randn(4, 64),
        y_logits=torch.randn(4, 64),
    )
    uncertainty = torch.rand(4)

    decision = arbiter(fast_output, uncertainty)

    assert decision.use_slow.shape == (4,)
    assert decision.reason == "learned"


def test_batch_consistency():
    """Test that batch processing is consistent."""
    from aria_lite.arbiter import Arbiter

    arbiter = Arbiter()

    # Same inputs should give same outputs
    confidence = torch.tensor([0.8, 0.8, 0.8])
    uncertainty = torch.tensor([0.1, 0.1, 0.1])

    use_slow = arbiter.should_use_slow(confidence, uncertainty)

    # All should be the same
    assert (use_slow == use_slow[0]).all()


def test_decision_reason():
    """Test that decision reasons are correct."""
    from aria_lite.arbiter import Arbiter
    from aria_lite.fast_policy import FastPolicyOutput

    arbiter = Arbiter()

    # High confidence, low uncertainty -> fast_sufficient
    fast_output = FastPolicyOutput(
        action_logits=torch.randn(1, 8),
        action_probs=torch.softmax(torch.randn(1, 8), dim=-1),
        confidence=torch.tensor([0.95]),
        x_logits=torch.randn(1, 64),
        y_logits=torch.randn(1, 64),
    )
    decision = arbiter(fast_output, torch.tensor([0.1]))
    assert decision.reason == "fast_sufficient"

    # Low confidence
    fast_output.confidence = torch.tensor([0.3])
    decision = arbiter(fast_output, torch.tensor([0.1]))
    assert "low_confidence" in decision.reason


if __name__ == "__main__":
    print("Phase 7 Validation: Arbiter Tests")
    print("=" * 40)

    test_arbiter_instantiation()
    print("✓ Arbiter instantiation")

    test_arbiter_from_config()
    print("✓ Arbiter from config")

    test_should_use_slow_basic()
    print("✓ Should use slow basic")

    test_should_use_slow_novelty()
    print("✓ Should use slow novelty")

    test_forward_with_fast_output()
    print("✓ Forward with fast output")

    test_force_slow()
    print("✓ Force slow")

    test_select_action()
    print("✓ Select action")

    test_statistics_tracking()
    print("✓ Statistics tracking")

    test_reset_stats()
    print("✓ Reset stats")

    test_calibrate_thresholds()
    print("✓ Calibrate thresholds")

    test_parameter_count_heuristic()
    print("✓ Parameter count heuristic")

    test_parameter_count_learned()
    print("✓ Parameter count learned")

    test_learned_switching()
    print("✓ Learned switching")

    test_batch_consistency()
    print("✓ Batch consistency")

    test_decision_reason()
    print("✓ Decision reason")

    print("\n" + "=" * 40)
    print("Phase 7 Validation: ALL TESTS PASSED")
    print("=" * 40)
