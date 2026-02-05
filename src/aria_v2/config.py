"""
ARIA v2 Configuration

Configuration for the Language-Guided Meta-Learning architecture.
"""

from dataclasses import dataclass, field


@dataclass
class VisualGroundingConfig:
    """Configuration for visual grounding module."""
    grid_size: int = 64
    num_colors: int = 16
    num_entity_classes: int = 6  # player, goal, item, obstacle, trigger, unknown
    embed_dim: int = 32
    hidden_dim: int = 128


@dataclass
class EventDetectorConfig:
    """Configuration for event detection."""
    history_length: int = 10  # Number of recent events to track
    correlation_threshold: float = 0.7  # For cause-effect inference


@dataclass
class LLMConfig:
    """Configuration for LLM reasoning engine."""
    model_path: str = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    n_ctx: int = 2048
    n_gpu_layers: int = -1  # -1 for full GPU offload
    temperature: float = 0.3
    max_tokens: int = 200
    cache_size: int = 1000


@dataclass
class SubgoalExecutorConfig:
    """Configuration for subgoal execution."""
    grid_size: int = 64
    max_path_length: int = 100
    exploration_probability: float = 0.1


@dataclass
class ARIAv2Config:
    """Main configuration for ARIA v2 agent."""

    # Component configs
    visual_grounding: VisualGroundingConfig = field(default_factory=VisualGroundingConfig)
    event_detector: EventDetectorConfig = field(default_factory=EventDetectorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    subgoal_executor: SubgoalExecutorConfig = field(default_factory=SubgoalExecutorConfig)

    # Agent behavior
    reasoning_interval: int = 10  # Reason every N steps
    max_subgoals: int = 3  # Max subgoals to track

    # Debug
    log_traces: bool = True
    verbose: bool = False

    def __post_init__(self):
        """Ensure nested configs are instantiated."""
        if isinstance(self.visual_grounding, dict):
            self.visual_grounding = VisualGroundingConfig(**self.visual_grounding)
        if isinstance(self.event_detector, dict):
            self.event_detector = EventDetectorConfig(**self.event_detector)
        if isinstance(self.llm, dict):
            self.llm = LLMConfig(**self.llm)
        if isinstance(self.subgoal_executor, dict):
            self.subgoal_executor = SubgoalExecutorConfig(**self.subgoal_executor)
