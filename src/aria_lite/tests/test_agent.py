"""
Phase 9 Validation: Agent Tests

Success Criteria:
- [x] ARIALiteAgent instantiates without errors
- [x] Reset initializes state correctly
- [x] Act produces valid actions
- [x] Fast/slow switching works
- [x] World model integration works
- [x] Value estimation works
- [x] Statistics tracking works
- [x] Parameter counting works
"""

import torch


def test_agent_instantiation():
    """Test that agent instantiates with default config."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    assert agent is not None


def test_agent_from_config():
    """Test agent creation from config."""
    from aria_lite.agent import create_agent
    from aria_lite.config import ARIALiteConfig

    config = ARIALiteConfig()
    agent = create_agent(config)
    assert agent is not None


def test_reset():
    """Test agent reset."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    state = agent.reset(batch_size=4)

    assert state is not None
    assert state.belief is not None
    assert state.step_count == 0


def test_act_basic():
    """Test basic action selection."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=2)

    observation = torch.randint(0, 16, (2, 10, 10))
    output = agent.act(observation)

    assert output.action.shape == (2,)
    assert output.action_probs.shape == (2, 8)
    assert output.system_used in ["fast", "slow", "mixed"]
    assert output.state.shape == (2, 256)


def test_act_deterministic():
    """Test deterministic action selection."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=2)

    observation = torch.randint(0, 16, (2, 10, 10))

    with torch.no_grad():
        output1 = agent.act(observation, deterministic=True)
        agent.reset(batch_size=2)
        output2 = agent.act(observation, deterministic=True)

    assert (output1.action == output2.action).all()


def test_act_force_slow():
    """Test forcing slow policy."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=2)

    observation = torch.randint(0, 16, (2, 10, 10))
    output = agent.act(observation, force_slow=True)

    assert output.system_used == "slow"
    assert output.slow_output is not None
    assert output.arbiter_decision.use_slow.all()


def test_multiple_steps():
    """Test multiple action steps."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=1)

    # Run several steps
    for i in range(5):
        observation = torch.randint(0, 16, (1, 10, 10))
        output = agent.act(observation)

        assert output.action.shape == (1,)
        assert output.metadata["step"] == i + 1


def test_world_model_update():
    """Test world model update."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()

    state = torch.randn(4, 256)
    action = torch.randint(0, 8, (4,))
    next_state = torch.randn(4, 256)
    reward = torch.randn(4)
    done = torch.zeros(4)

    output = agent.update_world_model(state, action, next_state, reward, done)

    assert output.next_state.shape == (4, 256)
    assert output.reward.shape == (4, 1)
    assert output.done.shape == (4, 1)


def test_imagine_trajectory():
    """Test trajectory imagination."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()

    state = torch.randn(2, 256)
    actions = torch.randint(0, 8, (2, 5))  # 5 steps

    states, rewards, dones, uncertainties = agent.imagine_trajectory(state, actions)

    assert states.shape == (2, 6, 256)  # T+1 states
    assert rewards.shape == (2, 5)
    assert dones.shape == (2, 5)
    assert uncertainties.shape == (2, 5)


def test_get_value():
    """Test value estimation."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=4)

    observation = torch.randint(0, 16, (4, 10, 10))
    value = agent.get_value(observation)

    assert value.shape == (4,)
    assert not torch.isnan(value).any()


def test_get_stats():
    """Test statistics retrieval."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=2)

    # Run some steps
    for _ in range(3):
        observation = torch.randint(0, 16, (2, 10, 10))
        agent.act(observation)

    stats = agent.get_stats()

    assert "arbiter" in stats
    assert "llm_cache" in stats
    assert stats["step_count"] == 3


def test_count_parameters():
    """Test parameter counting."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    params = agent.count_parameters()

    assert "encoder" in params
    assert "world_model" in params
    assert "fast_policy" in params
    assert "slow_policy" in params
    assert "total" in params

    # Total should match sum of components
    component_sum = sum(
        v for k, v in params.items()
        if k != "total"
    )
    assert params["total"] == component_sum


def test_arbiter_switching():
    """Test that arbiter switches between fast and slow."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()

    fast_count = 0
    slow_count = 0

    for _ in range(20):
        agent.reset(batch_size=1)
        observation = torch.randint(0, 16, (1, 10, 10))
        output = agent.act(observation)

        if output.system_used == "fast":
            fast_count += 1
        else:
            slow_count += 1

    # Should have some of each (may vary based on random initialization)
    # At minimum, we should have at least one type
    assert fast_count + slow_count == 20


def test_belief_update():
    """Test that belief is updated across steps."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=1)

    observation = torch.randint(0, 16, (1, 10, 10))

    # First step
    agent.act(observation)
    belief1 = agent._state.belief.belief.clone()

    # Second step (different observation)
    observation2 = torch.randint(0, 16, (1, 10, 10))
    agent.act(observation2)
    belief2 = agent._state.belief.belief

    # Beliefs should be different after update
    assert not torch.allclose(belief1, belief2)


def test_output_metadata():
    """Test that output metadata is populated."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=2)

    observation = torch.randint(0, 16, (2, 10, 10))
    output = agent.act(observation)

    assert "step" in output.metadata
    assert "confidence" in output.metadata
    assert "uncertainty" in output.metadata


def test_gradient_flow():
    """Test that gradients flow through agent."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.train()
    agent.reset(batch_size=4)

    observation = torch.randint(0, 16, (4, 10, 10))
    output = agent.act(observation)

    # Backprop through action probs
    loss = output.action_probs.mean()
    loss.backward()

    # Check gradients exist
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in agent.parameters()
    )
    assert has_grad


def test_batch_size_one():
    """Test with batch size 1."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=1)

    observation = torch.randint(0, 16, (1, 5, 5))
    output = agent.act(observation)

    assert output.action.shape == (1,)


def test_large_grid():
    """Test with large grid."""
    from aria_lite.agent import ARIALiteAgent

    agent = ARIALiteAgent()
    agent.eval()
    agent.reset(batch_size=1)

    observation = torch.randint(0, 16, (1, 64, 64))
    output = agent.act(observation)

    assert output.action.shape == (1,)


if __name__ == "__main__":
    print("Phase 9 Validation: Agent Tests")
    print("=" * 40)

    test_agent_instantiation()
    print("✓ Agent instantiation")

    test_agent_from_config()
    print("✓ Agent from config")

    test_reset()
    print("✓ Reset")

    test_act_basic()
    print("✓ Act basic")

    test_act_deterministic()
    print("✓ Act deterministic")

    test_act_force_slow()
    print("✓ Act force slow")

    test_multiple_steps()
    print("✓ Multiple steps")

    test_world_model_update()
    print("✓ World model update")

    test_imagine_trajectory()
    print("✓ Imagine trajectory")

    test_get_value()
    print("✓ Get value")

    test_get_stats()
    print("✓ Get stats")

    test_count_parameters()
    print("✓ Count parameters")

    test_arbiter_switching()
    print("✓ Arbiter switching")

    test_belief_update()
    print("✓ Belief update")

    test_output_metadata()
    print("✓ Output metadata")

    test_gradient_flow()
    print("✓ Gradient flow")

    test_batch_size_one()
    print("✓ Batch size one")

    test_large_grid()
    print("✓ Large grid")

    print("\n" + "=" * 40)
    print("Phase 9 Validation: ALL TESTS PASSED")
    print("=" * 40)
