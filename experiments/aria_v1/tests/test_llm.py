"""
Phase 8 Validation: LLM Integration Tests

Success Criteria:
- [x] LLMInterface instantiates without errors
- [x] Fallback mode works when model unavailable
- [x] Cache functions correctly
- [x] Goal embedding has correct shape
- [x] Grid-to-text conversion works
- [x] Graceful handling of missing dependencies
"""

import torch


def test_llm_interface_instantiation():
    """Test that LLM interface instantiates (fallback mode)."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()
    assert llm is not None
    # Without model path, should be in fallback mode
    assert not llm.is_available()


def test_llm_interface_from_config():
    """Test LLM interface creation from ARIALiteConfig."""
    from aria_lite.config import ARIALiteConfig
    from aria_lite.llm import create_llm_interface

    config = ARIALiteConfig()
    llm = create_llm_interface(config)
    assert llm is not None


def test_generate_fallback():
    """Test fallback goal generation."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    # Simple grid
    grid = torch.zeros(10, 10, dtype=torch.long)
    grid[2:8, 2:8] = 1  # Blue square in center

    response = llm.generate_goal_hypothesis(grid)

    assert response.text is not None
    assert len(response.text) > 0
    assert response.goal_embedding is not None
    assert response.goal_embedding.shape == (64,)
    assert response.tokens_used == 0  # Fallback doesn't use tokens


def test_generate_with_output():
    """Test goal generation with output grid."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    input_grid = torch.zeros(10, 10, dtype=torch.long)
    input_grid[2:8, 2:8] = 1

    # Smaller output
    output_grid = torch.ones(6, 6, dtype=torch.long)

    response = llm.generate_goal_hypothesis(input_grid, output_grid)

    assert "extract" in response.text.lower() or "crop" in response.text.lower() or "Goal" in response.text


def test_cache_functionality():
    """Test that caching works correctly."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    grid = torch.randint(0, 5, (8, 8))

    # First call
    response1 = llm.generate_goal_hypothesis(grid)
    assert not response1.cached

    # Second call with same input should be cached
    response2 = llm.generate_goal_hypothesis(grid)
    assert response2.cached
    assert response2.text == response1.text


def test_cache_stats():
    """Test cache statistics."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()
    llm.clear_cache()

    # Generate some responses
    for i in range(5):
        grid = torch.randint(0, 16, (5 + i, 5 + i))
        llm.generate_goal_hypothesis(grid)

    stats = llm.get_cache_stats()
    assert stats["size"] == 5
    assert stats["max_size"] == llm.config.cache_size


def test_cache_clear():
    """Test cache clearing."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    grid = torch.randint(0, 5, (8, 8))
    llm.generate_goal_hypothesis(grid)

    assert llm.get_cache_stats()["size"] > 0

    llm.clear_cache()
    assert llm.get_cache_stats()["size"] == 0


def test_goal_encoder():
    """Test goal text encoder."""
    from aria_lite.llm import GoalEncoder

    encoder = GoalEncoder(embed_dim=64)

    text1 = "rotate the grid 90 degrees"
    text2 = "extract the blue region"

    emb1 = encoder(text1)
    emb2 = encoder(text2)

    assert emb1.shape == (64,)
    assert emb2.shape == (64,)

    # Different texts should give different embeddings
    assert not torch.allclose(emb1, emb2)


def test_goal_encoder_deterministic():
    """Test that goal encoder is deterministic."""
    from aria_lite.llm import GoalEncoder

    encoder = GoalEncoder(embed_dim=64)
    encoder.eval()

    text = "find the pattern"

    emb1 = encoder(text)
    emb2 = encoder(text)

    assert torch.allclose(emb1, emb2)


def test_grid_to_text():
    """Test grid to text conversion."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    # Grid with known colors
    grid = torch.zeros(5, 5, dtype=torch.long)
    grid[0, 0] = 1  # blue
    grid[1, 1] = 2  # red
    grid[2, 2] = 3  # green

    text = llm._grid_to_text(grid)

    assert "5x5" in text
    assert "black" in text.lower()
    assert "blue" in text.lower()


def test_batched_grid():
    """Test with batched grid input."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    # Batched grid [B, H, W]
    grid = torch.randint(0, 10, (4, 8, 8))

    # Should handle first element of batch
    response = llm.generate_goal_hypothesis(grid)

    assert response.text is not None
    assert response.goal_embedding.shape == (64,)


def test_symmetric_detection():
    """Test symmetric pattern detection in fallback."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    # Create symmetric grid
    grid = torch.zeros(6, 6, dtype=torch.long)
    grid[1:5, 1:5] = 1
    grid[2:4, 2:4] = 2

    response = llm.generate_goal_hypothesis(grid)

    # Should detect symmetry or pattern
    assert response.text is not None


def test_lru_cache():
    """Test LRU cache behavior."""
    from aria_lite.llm import LRUCache

    cache = LRUCache(maxsize=3)

    # Fill cache
    cache.put("a", "1")
    cache.put("b", "2")
    cache.put("c", "3")

    assert len(cache) == 3

    # Access 'a' to make it recently used
    cache.get("a")

    # Add new item, should evict 'b' (least recently used)
    cache.put("d", "4")

    assert cache.get("a") == "1"
    assert cache.get("b") is None  # Evicted
    assert cache.get("c") == "3"
    assert cache.get("d") == "4"


def test_empty_grid():
    """Test with empty (all zeros) grid."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    grid = torch.zeros(5, 5, dtype=torch.long)
    response = llm.generate_goal_hypothesis(grid)

    assert response.text is not None
    assert response.goal_embedding is not None


def test_many_colors():
    """Test with grid containing many colors."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    # Grid with all 16 colors
    grid = torch.arange(16).reshape(4, 4)
    response = llm.generate_goal_hypothesis(grid)

    assert "pattern" in response.text.lower() or "Goal" in response.text


def test_context_parameter():
    """Test with additional context."""
    from aria_lite.llm import LLMInterface

    llm = LLMInterface()

    grid = torch.randint(0, 5, (8, 8))
    response = llm.generate_goal_hypothesis(
        grid,
        context="This is a rotation task"
    )

    assert response.text is not None


if __name__ == "__main__":
    print("Phase 8 Validation: LLM Integration Tests")
    print("=" * 40)

    test_llm_interface_instantiation()
    print("✓ LLM interface instantiation")

    test_llm_interface_from_config()
    print("✓ LLM interface from config")

    test_generate_fallback()
    print("✓ Generate fallback")

    test_generate_with_output()
    print("✓ Generate with output")

    test_cache_functionality()
    print("✓ Cache functionality")

    test_cache_stats()
    print("✓ Cache stats")

    test_cache_clear()
    print("✓ Cache clear")

    test_goal_encoder()
    print("✓ Goal encoder")

    test_goal_encoder_deterministic()
    print("✓ Goal encoder deterministic")

    test_grid_to_text()
    print("✓ Grid to text")

    test_batched_grid()
    print("✓ Batched grid")

    test_symmetric_detection()
    print("✓ Symmetric detection")

    test_lru_cache()
    print("✓ LRU cache")

    test_empty_grid()
    print("✓ Empty grid")

    test_many_colors()
    print("✓ Many colors")

    test_context_parameter()
    print("✓ Context parameter")

    print("\n" + "=" * 40)
    print("Phase 8 Validation: ALL TESTS PASSED")
    print("=" * 40)
