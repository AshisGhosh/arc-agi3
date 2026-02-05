"""
ARIA-Lite Agent for ARC-AGI-3 Games.

This module provides an adapter between ARIA-Lite and the ARC-AGI-3 framework.
It converts frame data to grid observations and maps actions appropriately.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from arcengine import FrameData, GameAction, GameState

from .agent import create_agent
from .config import ARIALiteConfig
from .meta import MetaLearningAgent


class ARCAgentConfig:
    """Configuration for ARC-AGI-3 agent adapter."""

    # Frame processing
    frame_layer: int = 0  # Which layer of frame data to use
    grid_size: int = 10  # Target grid size after downsampling
    num_colors: int = 16  # Number of distinct colors

    # Action mapping: our actions -> GameAction
    action_mapping: dict[int, GameAction] = {
        0: GameAction.RESET,  # NOOP maps to RESET (will be filtered)
        1: GameAction.ACTION1,  # UP
        2: GameAction.ACTION2,  # DOWN
        3: GameAction.ACTION3,  # LEFT
        4: GameAction.ACTION4,  # RIGHT
        5: GameAction.ACTION5,  # INTERACT
        6: GameAction.ACTION6,  # CONFIRM
        7: GameAction.ACTION7,  # CANCEL (if exists)
    }

    # Meta-learning settings
    use_meta_learning: bool = True
    demo_collection_steps: int = 5  # Steps to collect as demonstrations

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ARCAGIAgent:
    """
    ARC-AGI-3 compatible agent using ARIA-Lite.

    This adapter:
    1. Converts 64x64 pixel frames to grid observations
    2. Maps our internal actions to GameAction enum
    3. Collects demonstrations for meta-learning
    4. Interfaces with the agents-reference Agent class

    Usage:
        config = ARCAgentConfig()
        agent = ARCAGIAgent(config)

        # In the game loop:
        action = agent.choose_action(frames, latest_frame)
    """

    def __init__(
        self,
        config: Optional[ARCAgentConfig] = None,
        aria_config: Optional[ARIALiteConfig] = None,
        meta_model: Optional[MetaLearningAgent] = None,
    ):
        self.config = config or ARCAgentConfig()

        # Initialize ARIA-Lite agent
        if aria_config is None:
            aria_config = ARIALiteConfig()

        self.device = torch.device(self.config.device)

        # Use meta-learning agent if provided, else use standard agent
        self.meta_agent = None
        self.aria_agent = None

        if meta_model is not None:
            self.meta_agent = meta_model.to(self.device)
        elif self.config.use_meta_learning:
            self.meta_agent = self._create_meta_agent()
        else:
            self.aria_agent = create_agent(aria_config).to(self.device)

        # Demo collection for meta-learning
        self.demo_observations: list[torch.Tensor] = []
        self.demo_actions: list[int] = []

        # Track state
        self.step_count = 0
        self.last_game_state: Optional[GameState] = None

    def _create_meta_agent(self) -> MetaLearningAgent:
        """Create meta-learning agent."""
        return MetaLearningAgent(
            hidden_dim=128,
            task_dim=64,
            num_colors=self.config.num_colors,
            num_actions=8,  # Match our action space
            max_grid=self.config.grid_size,
        ).to(self.device)

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.demo_observations = []
        self.demo_actions = []
        self.step_count = 0
        self.last_game_state = None

        if self.aria_agent is not None:
            self.aria_agent.reset(batch_size=1)

    def _frame_to_grid(self, frame: list) -> torch.Tensor:
        """
        Convert frame data to grid observation.

        Args:
            frame: List of frame layers (can be numpy arrays or lists)

        Returns:
            grid: [H, W] tensor with color indices (0-15)
        """
        if not frame:
            return torch.zeros(
                self.config.grid_size, self.config.grid_size,
                dtype=torch.long, device=self.device
            )

        # Use specified frame layer (usually 0)
        layer = frame[self.config.frame_layer]

        # Convert to numpy array if it's a list
        if isinstance(layer, list):
            layer = np.array(layer)

        # Frame is typically [64, 64] or similar with pixel values
        # Convert to grid by downsampling and color quantization

        # If frame is RGB, convert to single channel via max
        if len(layer.shape) == 3:
            layer = layer.max(axis=-1)

        # Convert to tensor
        frame_tensor = torch.from_numpy(layer).float().to(self.device)

        # Downsample to target grid size
        H, W = frame_tensor.shape
        if H != self.config.grid_size or W != self.config.grid_size:
            # Nearest neighbor downsampling to preserve colors
            frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            frame_tensor = F.interpolate(
                frame_tensor,
                size=(self.config.grid_size, self.config.grid_size),
                mode='nearest'
            )
            frame_tensor = frame_tensor.squeeze(0).squeeze(0)  # [H, W]

        # Quantize colors to 0-15 range
        grid = (frame_tensor / 256 * self.config.num_colors).long()
        grid = grid.clamp(0, self.config.num_colors - 1)

        return grid

    def _action_to_game_action(self, action: int) -> GameAction:
        """Map internal action to GameAction."""
        if action in self.config.action_mapping:
            return self.config.action_mapping[action]
        # Default to ACTION1 if unknown
        return GameAction.ACTION1

    def _add_demonstration(
        self,
        observation: torch.Tensor,
        action: int,
    ) -> None:
        """Add observation-action pair to demonstrations."""
        self.demo_observations.append(observation.clone())
        self.demo_actions.append(action)

        # Limit demo length
        max_demos = self.config.demo_collection_steps * 3
        if len(self.demo_observations) > max_demos:
            self.demo_observations = self.demo_observations[-max_demos:]
            self.demo_actions = self.demo_actions[-max_demos:]

    def _get_demo_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get demonstration tensors for meta-learning."""
        if len(self.demo_observations) == 0:
            # Return dummy demos
            H = W = self.config.grid_size
            return (
                torch.zeros(1, 1, H, W, device=self.device),
                torch.zeros(1, 1, dtype=torch.long, device=self.device),
            )

        # Stack demonstrations
        demo_obs = torch.stack(self.demo_observations, dim=0)  # [K, H, W]
        demo_actions = torch.tensor(self.demo_actions, device=self.device)  # [K]

        # Add batch dimension
        return demo_obs.unsqueeze(0), demo_actions.unsqueeze(0)  # [1, K, H, W], [1, K]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Check if agent should stop."""
        if latest_frame.state == GameState.WIN:
            return True
        # Optional: stop on GAME_OVER after certain attempts
        return False

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        """
        Choose action based on current game state.

        This is the main interface method called by the game loop.

        Args:
            frames: History of all frames
            latest_frame: Most recent frame

        Returns:
            GameAction to take
        """
        # Handle game state transitions
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            self.reset()
            return GameAction.RESET

        # Convert frame to grid observation
        observation = self._frame_to_grid(latest_frame.frame)

        # Get action from appropriate agent
        if self.meta_agent is not None:
            action = self._choose_action_meta(observation)
        else:
            action = self._choose_action_aria(observation)

        # Record for demonstration
        self._add_demonstration(observation, action)

        # Update state
        self.step_count += 1
        self.last_game_state = latest_frame.state

        # Convert to GameAction
        game_action = self._action_to_game_action(action)

        # Add reasoning metadata
        game_action.reasoning = {
            "step": self.step_count,
            "internal_action": action,
            "agent_type": "meta" if self.meta_agent else "aria",
        }

        return game_action

    def _choose_action_meta(self, observation: torch.Tensor) -> int:
        """Choose action using meta-learning agent."""
        # Get demonstrations
        demo_obs, demo_actions = self._get_demo_tensors()

        # Add batch dimension to observation
        obs = observation.unsqueeze(0)  # [1, H, W]

        with torch.no_grad():
            output = self.meta_agent.act(
                obs=obs,
                demo_obs=demo_obs,
                demo_actions=demo_actions,
                grid_size=self.config.grid_size,
            )

            # Get action from logits
            action = output["action_logits"].argmax(-1).item()

        return action

    def _choose_action_aria(self, observation: torch.Tensor) -> int:
        """Choose action using ARIA-Lite agent."""
        # Add batch dimension
        obs = observation.unsqueeze(0)  # [1, H, W]

        with torch.no_grad():
            output = self.aria_agent.act(obs, deterministic=True)
            action = output.action.item()

        return action


def create_arc_agent(
    config: Optional[ARCAgentConfig] = None,
    meta_model_path: Optional[str] = None,
) -> ARCAGIAgent:
    """
    Factory function to create ARC-AGI-3 compatible agent.

    Args:
        config: Agent configuration
        meta_model_path: Optional path to load trained meta-learning model

    Returns:
        Configured ARCAGIAgent
    """
    meta_model = None

    if meta_model_path is not None:
        meta_model = MetaLearningAgent(
            hidden_dim=128,
            task_dim=64,
            num_colors=16,
            num_actions=8,
            max_grid=10,
        )
        checkpoint = torch.load(meta_model_path, weights_only=True)
        meta_model.load_state_dict(checkpoint)

    return ARCAGIAgent(config=config, meta_model=meta_model)
