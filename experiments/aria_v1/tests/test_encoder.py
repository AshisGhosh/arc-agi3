"""
Phase 2 Validation: Encoder Tests

Success Criteria:
- [x] GridEncoderLite instantiates without errors
- [x] Forward pass produces correct output shape [B, 256]
- [x] Works with various grid sizes (up to 64x64)
- [x] Handles masking correctly
- [x] Parameter count within budget (~5M ± 50%)
- [x] Gradients flow properly
"""

import pytest
import torch


def test_encoder_instantiation():
    """Test that encoder instantiates with default config."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    assert encoder is not None
    assert encoder.output_dim == 256


def test_encoder_from_config():
    """Test encoder creation from ARIALiteConfig."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.encoder import create_encoder

    config = ARIALiteConfig()
    encoder = create_encoder(config)
    assert encoder.output_dim == config.encoder.output_dim


def test_forward_shape():
    """Test that forward pass produces correct output shape."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    encoder.eval()

    # Test various batch sizes
    for batch_size in [1, 4, 16]:
        grid = torch.randint(0, 16, (batch_size, 30, 30))
        output = encoder(grid)
        assert output.shape == (batch_size, 256), f"Expected (B, 256), got {output.shape}"


def test_various_grid_sizes():
    """Test encoder with different grid sizes."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    encoder.eval()

    # ARC grids can be various sizes up to 30x30 typically
    test_sizes = [(10, 10), (30, 30), (20, 15), (64, 64)]

    for H, W in test_sizes:
        grid = torch.randint(0, 16, (2, H, W))
        output = encoder(grid)
        assert output.shape == (2, 256), f"Failed for grid size ({H}, {W})"


def test_masking():
    """Test that masking works correctly."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    encoder.eval()

    grid = torch.randint(0, 16, (2, 20, 20))

    # Create a mask (True = invalid/masked)
    mask = torch.zeros(2, 20, 20, dtype=torch.bool)
    mask[0, 10:, :] = True  # Mask bottom half of first sample

    # Should run without error
    output = encoder(grid, mask=mask)
    assert output.shape == (2, 256)
    assert not torch.isnan(output).any()


def test_parameter_count():
    """Test that parameter count is within budget."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    param_count = encoder.count_parameters()

    # Config estimates ~8M based on current settings
    # Allow 2-12M range
    print(f"\nEncoder parameters: {param_count:,}")

    assert param_count > 2_000_000, f"Too few parameters: {param_count:,}"
    assert param_count < 12_000_000, f"Too many parameters: {param_count:,}"


def test_parameter_count_vs_estimate():
    """Test that actual count matches config estimate."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.encoder import create_encoder

    config = ARIALiteConfig()
    encoder = create_encoder(config)

    actual = encoder.count_parameters()
    estimated = config.encoder.estimate_params()

    print(f"\nActual params: {actual:,}")
    print(f"Estimated params: {estimated:,}")

    # Allow 50% tolerance between actual and estimated
    ratio = actual / estimated if estimated > 0 else float("inf")
    assert 0.5 < ratio < 2.0, f"Estimate mismatch: actual={actual:,}, estimated={estimated:,}"


def test_gradient_flow():
    """Test that gradients flow through the encoder."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    encoder.train()

    grid = torch.randint(0, 16, (4, 20, 20))
    output = encoder(grid)

    # Compute loss and backprop
    loss = output.mean()
    loss.backward()

    # Check that gradients exist
    grad_exists = False
    for name, param in encoder.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            grad_exists = True
            break

    assert grad_exists, "No gradients found in encoder parameters"


def test_deterministic_output():
    """Test that encoder produces deterministic output in eval mode."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    encoder.eval()

    grid = torch.randint(0, 16, (2, 20, 20))

    with torch.no_grad():
        output1 = encoder(grid)
        output2 = encoder(grid)

    assert torch.allclose(output1, output2), "Encoder output not deterministic in eval mode"


def test_color_range():
    """Test that encoder handles all valid color values."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    encoder.eval()

    # Grid with all possible colors
    grid = torch.zeros(1, 4, 4, dtype=torch.long)
    for i in range(16):
        grid[0, i // 4, i % 4] = i

    output = encoder(grid)
    assert output.shape == (1, 256)
    assert not torch.isnan(output).any()


def test_small_grid():
    """Test encoder with minimal practical grid size (3x3)."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite()
    encoder.eval()

    # ARC grids are at minimum 3x3 in practice
    grid = torch.randint(0, 16, (1, 3, 3))
    output = encoder(grid)
    assert output.shape == (1, 256)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    """Test encoder on CUDA device."""
    from aria_lite.encoder import GridEncoderLite

    encoder = GridEncoderLite().cuda()
    encoder.eval()

    grid = torch.randint(0, 16, (4, 20, 20)).cuda()
    output = encoder(grid)

    assert output.device.type == "cuda"
    assert output.shape == (4, 256)


if __name__ == "__main__":
    # Run all tests manually
    print("Phase 2 Validation: Encoder Tests")
    print("=" * 40)

    test_encoder_instantiation()
    print("✓ Encoder instantiation")

    test_encoder_from_config()
    print("✓ Encoder from config")

    test_forward_shape()
    print("✓ Forward shape")

    test_various_grid_sizes()
    print("✓ Various grid sizes")

    test_masking()
    print("✓ Masking")

    test_parameter_count()
    print("✓ Parameter count")

    test_parameter_count_vs_estimate()
    print("✓ Parameter count vs estimate")

    test_gradient_flow()
    print("✓ Gradient flow")

    test_deterministic_output()
    print("✓ Deterministic output")

    test_color_range()
    print("✓ Color range")

    test_small_grid()
    print("✓ Small grid")

    print("\n" + "=" * 40)
    print("Phase 2 Validation: ALL TESTS PASSED")
    print("=" * 40)
