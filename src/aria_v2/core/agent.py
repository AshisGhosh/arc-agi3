"""
Integrated Agent - Ties all core components together.

This is the main entry point for playing ARC-AGI-3 games.
Supports A/B testing of exploration policies.
"""

from dataclasses import dataclass, field
from typing import Optional
import time

import numpy as np

from .observation_tracker import ObservationTracker, Observation
from .belief_state import BeliefState
from .action_selector import ActionSelector, ActionDecision
from .exploration import (
    ExplorationPolicy,
    RandomExplorationPolicy,
    SystematicExplorationPolicy,
    LearnedExplorationPolicy,
)


@dataclass
class EpisodeStats:
    """Statistics for one episode."""
    actions_taken: int = 0
    levels_completed: int = 0
    fast_decisions: int = 0
    evidence_decisions: int = 0
    exploration_decisions: int = 0
    player_found_at_action: Optional[int] = None
    duration_seconds: float = 0.0


@dataclass
class AgentConfig:
    """Configuration for the agent."""
    # Exploration policy type: "random", "systematic", "learned"
    exploration_type: str = "systematic"

    # Step size for navigation (pixels)
    step_size: int = 5

    # Maximum actions per episode
    max_actions: int = 1000

    # Confidence thresholds
    goal_confidence: float = 0.5
    collectible_confidence: float = 0.6
    blocker_confidence: float = 0.6

    # LLM advisor
    use_llm: bool = False
    llm_provider: str = "auto"  # "anthropic", "openai", "ollama", "heuristic", "auto"
    llm_model: str = None

    # Random seed for reproducibility
    seed: int = 42


class ARIAAgent:
    """
    ARIA v2 Agent - Evidence-based game playing.

    No pre-training. Learns during play through observation.
    Supports A/B testing of exploration strategies.
    Optionally uses LLM for strategic advice.
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()

        # Core components
        self.tracker = ObservationTracker()
        self.belief_state = BeliefState()

        # Create exploration policy based on config
        self.exploration_policy = self._create_exploration_policy()

        # Action selector
        self.action_selector = ActionSelector(
            exploration_policy=self.exploration_policy,
            step_size=self.config.step_size,
        )

        # LLM advisor (optional)
        self.llm_advisor = None
        if self.config.use_llm:
            from .llm_advisor import LLMAdvisor
            self.llm_advisor = LLMAdvisor(
                provider=self.config.llm_provider,
                model=self.config.llm_model,
            )

        # Episode tracking
        self.stats = EpisodeStats()
        self.action_history: list[ActionDecision] = []
        self.observation_history: list[Observation] = []
        self.llm_advice_history: list = []

    def _create_exploration_policy(self) -> ExplorationPolicy:
        """Create exploration policy based on config."""
        if self.config.exploration_type == "random":
            return RandomExplorationPolicy(seed=self.config.seed)
        elif self.config.exploration_type == "systematic":
            return SystematicExplorationPolicy()
        elif self.config.exploration_type == "learned":
            return LearnedExplorationPolicy()
        else:
            raise ValueError(f"Unknown exploration type: {self.config.exploration_type}")

    def reset(self):
        """Reset for new episode."""
        self.tracker.reset()
        self.belief_state = BeliefState()
        self.stats = EpisodeStats()
        self.action_history = []
        self.observation_history = []

    def act(
        self,
        frame: np.ndarray,
        level_completed: bool = False,
        game_over: bool = False,
    ) -> int:
        """
        Choose action based on current frame.

        Args:
            frame: Current game frame [H, W] of color indices
            level_completed: Signal from game
            game_over: Signal from game

        Returns:
            Action to take (0-7)
        """
        start_time = time.time()

        # Get previous action (or 0 for first frame)
        prev_action = 0
        if self.action_history:
            prev_action = self.action_history[-1].action

        # Observe changes
        observation = self.tracker.observe(
            frame, prev_action, level_completed, game_over
        )
        self.observation_history.append(observation)

        # Update belief state
        self.belief_state.update(observation)

        # Update timer from frame
        self.belief_state.update_timer(frame)

        # Track when player is found
        if self.belief_state.player_identified and self.stats.player_found_at_action is None:
            self.stats.player_found_at_action = self.stats.actions_taken

        # Check for level completion
        if level_completed:
            self.stats.levels_completed += 1

        # Get LLM advice if enabled (periodically, not every action)
        llm_advice = None
        if self.llm_advisor and self.stats.actions_taken % 10 == 0:
            llm_advice = self.llm_advisor.get_advice(self.belief_state)
            self.llm_advice_history.append(llm_advice)
            # Print LLM advice
            print(f"  LLM [{llm_advice.source}]: {llm_advice.strategy} - {llm_advice.reasoning}")

        # Select action
        decision = self.action_selector.select_action(
            self.belief_state,
            frame,
        )

        # Optionally modify decision based on LLM advice
        if llm_advice and llm_advice.confidence > 0.7:
            decision.reasoning += f" [LLM: {llm_advice.strategy}]"

        self.action_history.append(decision)

        # Update stats
        self.stats.actions_taken += 1
        if decision.source == "evidence":
            self.stats.evidence_decisions += 1
        elif decision.source == "rule":
            self.stats.fast_decisions += 1
        else:
            self.stats.exploration_decisions += 1

        self.stats.duration_seconds += time.time() - start_time

        return decision.action

    def get_last_decision(self) -> Optional[ActionDecision]:
        """Get the last action decision with reasoning."""
        if self.action_history:
            return self.action_history[-1]
        return None

    def get_stats(self) -> EpisodeStats:
        """Get episode statistics."""
        return self.stats

    def get_belief_summary(self) -> str:
        """Get human-readable belief state summary."""
        return self.belief_state.to_summary()


class ABTestRunner:
    """
    Runs A/B tests comparing exploration policies.

    Usage:
        runner = ABTestRunner()
        results = runner.run_test(env, num_episodes=100)
        print(results.summary())
    """

    @dataclass
    class PolicyResult:
        """Results for one policy."""
        policy_type: str
        episodes: int = 0
        total_actions: int = 0
        total_levels: int = 0
        player_found_rate: float = 0.0
        avg_actions_to_find_player: float = 0.0
        avg_actions_per_level: float = 0.0

    @dataclass
    class ABResults:
        """A/B test results."""
        baseline: "ABTestRunner.PolicyResult"
        treatment: "ABTestRunner.PolicyResult"

        def summary(self) -> str:
            """Generate summary report."""
            lines = [
                "=== A/B Test Results ===",
                "",
                f"Baseline ({self.baseline.policy_type}):",
                f"  Episodes: {self.baseline.episodes}",
                f"  Levels completed: {self.baseline.total_levels}",
                f"  Avg actions/level: {self.baseline.avg_actions_per_level:.1f}",
                f"  Player found rate: {self.baseline.player_found_rate:.1%}",
                "",
                f"Treatment ({self.treatment.policy_type}):",
                f"  Episodes: {self.treatment.episodes}",
                f"  Levels completed: {self.treatment.total_levels}",
                f"  Avg actions/level: {self.treatment.avg_actions_per_level:.1f}",
                f"  Player found rate: {self.treatment.player_found_rate:.1%}",
                "",
            ]

            # Calculate improvement
            if self.baseline.avg_actions_per_level > 0:
                improvement = (
                    self.baseline.avg_actions_per_level - self.treatment.avg_actions_per_level
                ) / self.baseline.avg_actions_per_level
                lines.append(f"Improvement: {improvement:.1%}")

            return "\n".join(lines)

    def __init__(
        self,
        baseline_type: str = "random",
        treatment_type: str = "learned",
    ):
        self.baseline_type = baseline_type
        self.treatment_type = treatment_type

    def run_test(
        self,
        env,  # Environment with reset() and step(action) methods
        num_episodes: int = 100,
        max_actions: int = 500,
    ) -> ABResults:
        """
        Run A/B test.

        Args:
            env: Environment to test on
            num_episodes: Number of episodes per policy
            max_actions: Max actions per episode

        Returns:
            ABResults with comparison
        """
        baseline_stats = self._run_policy(
            env, self.baseline_type, num_episodes, max_actions
        )
        treatment_stats = self._run_policy(
            env, self.treatment_type, num_episodes, max_actions
        )

        return self.ABResults(baseline=baseline_stats, treatment=treatment_stats)

    def _run_policy(
        self,
        env,
        policy_type: str,
        num_episodes: int,
        max_actions: int,
    ) -> PolicyResult:
        """Run episodes with one policy."""
        config = AgentConfig(
            exploration_type=policy_type,
            max_actions=max_actions,
        )
        agent = ARIAAgent(config)

        total_actions = 0
        total_levels = 0
        player_found_count = 0
        actions_to_find_player = []

        for _ in range(num_episodes):
            agent.reset()
            frame = env.reset()
            done = False
            level_completed = False
            actions = 0

            while not done and actions < max_actions:
                action = agent.act(frame, level_completed=level_completed)
                frame, reward, done, info = env.step(action)
                level_completed = info.get("level_completed", False)
                actions += 1

            stats = agent.get_stats()
            total_actions += stats.actions_taken
            total_levels += stats.levels_completed

            if stats.player_found_at_action is not None:
                player_found_count += 1
                actions_to_find_player.append(stats.player_found_at_action)

        return self.PolicyResult(
            policy_type=policy_type,
            episodes=num_episodes,
            total_actions=total_actions,
            total_levels=total_levels,
            player_found_rate=player_found_count / num_episodes if num_episodes > 0 else 0,
            avg_actions_to_find_player=(
                sum(actions_to_find_player) / len(actions_to_find_player)
                if actions_to_find_player else 0
            ),
            avg_actions_per_level=(
                total_actions / total_levels if total_levels > 0 else float('inf')
            ),
        )


def test_agent():
    """Test the integrated agent."""
    # Create agent
    agent = ARIAAgent(AgentConfig(exploration_type="systematic"))

    # Create simple test frames
    frame1 = np.zeros((64, 64), dtype=np.int32)
    frame1[32, 32] = 1  # Player
    frame1[10:20, 10:20] = 4  # Collectible
    frame1[40:50, 40:50] = 5  # Wall

    # Simulate a few actions
    for i in range(10):
        action = agent.act(frame1)
        decision = agent.get_last_decision()
        print(f"Step {i+1}: action={action}, reasoning='{decision.reasoning}', source={decision.source}")

    # Show stats
    stats = agent.get_stats()
    print(f"\nStats: {stats.actions_taken} actions")
    print(f"  Evidence decisions: {stats.evidence_decisions}")
    print(f"  Fast decisions: {stats.fast_decisions}")
    print(f"  Exploration decisions: {stats.exploration_decisions}")

    print("\nBelief Summary:")
    print(agent.get_belief_summary())


if __name__ == "__main__":
    test_agent()
