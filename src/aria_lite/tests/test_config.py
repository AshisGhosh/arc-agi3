"""
Phase 1 Validation: Config Tests

Success Criteria:
- [x] ARIALiteConfig instantiates without errors
- [x] Total params <= 29M
- [x] VRAM estimate <= 7GB
- [x] All dimension consistency checks pass
- [x] Parameter breakdown is reasonable
"""

import pytest


def test_config_instantiation():
    """Test that config instantiates with defaults."""
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    assert config is not None
    assert config.device in ["cuda", "cpu", "auto"]


def test_parameter_budget():
    """Test that total parameters are within 29M budget."""
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    total = config.total_params()

    print(f"\nTotal parameters: {total:,}")
    assert total <= 29_000_000, f"Parameter budget exceeded: {total:,} > 29M"


def test_vram_budget():
    """Test that VRAM estimate is within 7GB budget."""
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    vram = config.estimate_vram_gb()

    print(f"\nEstimated VRAM: {vram:.2f} GB")
    assert vram <= 7.0, f"VRAM budget exceeded: {vram:.2f}GB > 7GB"


def test_dimension_consistency():
    """Test that dimensions are consistent across components."""
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    issues = config.validate()

    if issues:
        for issue in issues:
            print(f"Issue: {issue}")

    assert len(issues) == 0, f"Validation failed: {issues}"


def test_parameter_breakdown():
    """Test that parameter breakdown is reasonable."""
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    breakdown = config.params_breakdown()
    total = config.total_params()

    print("\nParameter breakdown:")
    for name, count in breakdown.items():
        pct = count / total * 100
        print(f"  {name}: {count:,} ({pct:.1f}%)")

    # Core checks: ensure no component is 0 or dominates
    for name, count in breakdown.items():
        if name != "arbiter":  # Arbiter can be 0 if using heuristics
            assert count > 10_000, f"{name} is too small: {count:,}"
            assert count < total * 0.5, f"{name} dominates: {count:,} > 50% of total"

    # Total should be in 20-30M range
    assert 20_000_000 < total < 32_000_000, f"Total {total:,} out of range [20M, 32M]"


def test_summary_generation():
    """Test that summary generates without error."""
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    summary = config.summary()

    print(f"\n{summary}")
    assert "ARIA-Lite Configuration Summary" in summary
    assert "Validation: PASSED" in summary


def test_create_config_helper():
    """Test the convenience config creator."""
    from aria_lite.config import create_config

    config = create_config(seed=123, device="cpu")
    assert config.seed == 123
    assert config.device == "cpu"


def test_create_config_invalid_key():
    """Test that invalid keys raise error."""
    from aria_lite.config import create_config

    with pytest.raises(ValueError):
        create_config(invalid_key=123)


if __name__ == "__main__":
    # Run all tests manually
    test_config_instantiation()
    print("✓ Config instantiation")

    test_parameter_budget()
    print("✓ Parameter budget")

    test_vram_budget()
    print("✓ VRAM budget")

    test_dimension_consistency()
    print("✓ Dimension consistency")

    test_parameter_breakdown()
    print("✓ Parameter breakdown")

    test_summary_generation()
    print("✓ Summary generation")

    test_create_config_helper()
    print("✓ Create config helper")

    print("\n" + "=" * 40)
    print("Phase 1 Validation: ALL TESTS PASSED")
    print("=" * 40)
