"""
Visual Grounding Module for ARIA v2.

Converts pixel observations to structured entity descriptions.
Contains:
- EntityDetectorCNN: Detects non-background entities
- EntityClassifier: Classifies entities (player, goal, item, obstacle, trigger)
- MovementCorrelator: Identifies player by what moves with actions
- VisualGroundingModule: Integrated module for full pipeline
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretraining.synthetic_games import EntityType


@dataclass
class DetectedEntity:
    """A detected entity with position and classification."""
    entity_type: EntityType
    x: int
    y: int
    width: int
    height: int
    color: int
    confidence: float

    def to_description(self) -> str:
        """Generate language description of entity."""
        type_name = self.entity_type.name.lower()
        return f"{type_name} at ({self.x}, {self.y})"


@dataclass
class SceneDescription:
    """Structured description of a game scene."""
    entities: list[DetectedEntity]
    player_position: Optional[tuple[int, int]]
    grid_size: tuple[int, int]

    def to_language(self) -> str:
        """Generate natural language description."""
        lines = [f"Scene: {self.grid_size[0]}x{self.grid_size[1]} grid"]

        if self.player_position:
            lines.append(f"Player at ({self.player_position[0]}, {self.player_position[1]})")

        # Group entities by type
        by_type: dict[EntityType, list[DetectedEntity]] = {}
        for e in self.entities:
            if e.entity_type not in by_type:
                by_type[e.entity_type] = []
            by_type[e.entity_type].append(e)

        for etype, ents in by_type.items():
            if etype == EntityType.PLAYER:
                continue  # Already mentioned
            if etype == EntityType.BACKGROUND:
                continue
            positions = [(e.x, e.y) for e in ents]
            lines.append(f"{len(ents)} {etype.name.lower()}(s) at {positions}")

        return "\n".join(lines)


class EntityDetectorCNN(nn.Module):
    """
    Detect non-background entities in grid.

    Input: [B, H, W] color indices (0-15)
    Output:
        - entity_mask: [B, H, W] probability of being an entity
        - features: [B, H, W, C] per-pixel features for classification
    """

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        feature_dim: int = 32,
    ):
        super().__init__()

        self.num_colors = num_colors
        self.embed_dim = embed_dim

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Entity mask prediction (binary: entity or not)
        self.mask_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

        # Feature extraction for classification
        self.feature_head = nn.Sequential(
            nn.Conv2d(hidden_dim, feature_dim, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: [B, H, W] color indices

        Returns:
            mask: [B, H, W] entity probability
            features: [B, H, W, C] per-pixel features
        """
        # Embed colors: [B, H, W] -> [B, H, W, E]
        embedded = self.color_embed(x)

        # Reshape for conv: [B, H, W, E] -> [B, E, H, W]
        embedded = embedded.permute(0, 3, 1, 2)

        # Convolutional layers with residual-like connections
        h = F.relu(self.conv1(embedded))
        h = F.relu(self.conv2(h)) + h
        h = F.relu(self.conv3(h)) + h

        # Entity mask: [B, 1, H, W] -> [B, H, W]
        mask = self.mask_head(h).squeeze(1)
        mask = torch.sigmoid(mask)

        # Features: [B, C, H, W] -> [B, H, W, C]
        features = self.feature_head(h)
        features = features.permute(0, 2, 3, 1)

        return mask, features


class EntityClassifier(nn.Module):
    """
    Classify detected entities.

    Input: [B, N, C] entity features (from crops or pooled regions)
    Output: [B, N, num_classes] class probabilities
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 64,
        num_classes: int = 6,  # player, goal, item, obstacle, trigger, unknown
    ):
        super().__init__()

        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify entities.

        Args:
            x: [B, N, C] entity features

        Returns:
            logits: [B, N, num_classes]
        """
        return self.classifier(x)


class MovementCorrelator:
    """
    Identify player by detecting what moves when action is taken.

    This is a self-supervised signal: the entity that moves in response
    to an action is the player.
    """

    def __init__(self, movement_threshold: float = 0.5):
        self.movement_threshold = movement_threshold

    def identify_player(
        self,
        obs_t: np.ndarray,
        action: int,
        obs_t1: np.ndarray,
    ) -> Optional[tuple[int, int]]:
        """
        Find entity that moved in response to action.

        Args:
            obs_t: Grid before action [H, W]
            action: Action taken (1=up, 2=down, 3=left, 4=right)
            obs_t1: Grid after action [H, W]

        Returns:
            (x, y) position of player in obs_t1, or None if no movement
        """
        if action == 0:  # NOOP
            return None

        # Expected movement direction
        dx, dy = 0, 0
        if action == 1:    # up
            dy = -1
        elif action == 2:  # down
            dy = 1
        elif action == 3:  # left
            dx = -1
        elif action == 4:  # right
            dx = 1

        # Find differences between frames
        diff = (obs_t != obs_t1)

        if not diff.any():
            return None  # No change

        # Find positions that changed
        changed_positions = np.argwhere(diff)  # [N, 2] as (y, x)

        if len(changed_positions) < 2:
            return None

        # Look for a pattern: something disappeared at one position,
        # appeared at an adjacent position in the direction of movement
        for y_new, x_new in changed_positions:
            y_old = y_new - dy
            x_old = x_new - dx

            # Check if old position is valid and changed
            if 0 <= y_old < obs_t.shape[0] and 0 <= x_old < obs_t.shape[1]:
                # Check if color moved from old to new
                if obs_t[y_old, x_old] == obs_t1[y_new, x_new] and obs_t[y_old, x_old] != 0:
                    # This looks like the player moved
                    return (x_new, y_new)

        return None

    def find_player_from_transitions(
        self,
        transitions: list[tuple[np.ndarray, int, np.ndarray]],
    ) -> Optional[tuple[int, int]]:
        """
        Find player position from multiple transitions.

        Args:
            transitions: List of (obs_t, action, obs_t1) tuples

        Returns:
            Most likely player position, or None
        """
        position_votes: dict[tuple[int, int], int] = {}

        for obs_t, action, obs_t1 in transitions:
            pos = self.identify_player(obs_t, action, obs_t1)
            if pos:
                position_votes[pos] = position_votes.get(pos, 0) + 1

        if not position_votes:
            return None

        # Return most voted position
        return max(position_votes, key=position_votes.get)


class VisualGroundingModule(nn.Module):
    """
    Integrated visual grounding module.

    Combines entity detection, classification, and player identification
    to produce structured scene descriptions.
    """

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        feature_dim: int = 32,
        num_classes: int = 6,
    ):
        super().__init__()

        self.detector = EntityDetectorCNN(
            num_colors=num_colors,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
        )

        self.classifier = EntityClassifier(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
        )

        self.movement_correlator = MovementCorrelator()

        # Threshold for entity detection
        self.detection_threshold = 0.5

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            x: [B, H, W] color indices

        Returns:
            mask: [B, H, W] entity probability
            features: [B, H, W, C] per-pixel features
            class_logits: [B, H, W, num_classes] per-pixel class logits
        """
        mask, features = self.detector(x)

        # Classify every pixel (inefficient but simple)
        B, H, W, C = features.shape
        flat_features = features.view(B, H * W, C)
        class_logits = self.classifier(flat_features)
        class_logits = class_logits.view(B, H, W, -1)

        return mask, features, class_logits

    @torch.no_grad()
    def extract_entities(
        self,
        grid: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> list[DetectedEntity]:
        """
        Extract entities from a grid observation.

        Args:
            grid: [H, W] color indices
            device: Torch device

        Returns:
            List of detected entities
        """
        self.eval()

        # Convert to tensor
        x = torch.from_numpy(grid).long().unsqueeze(0).to(device)

        # Forward pass
        mask, features, class_logits = self.forward(x)

        # Get predictions
        mask = mask.squeeze(0).cpu().numpy()
        class_probs = F.softmax(class_logits, dim=-1).squeeze(0).cpu().numpy()
        class_preds = class_logits.argmax(dim=-1).squeeze(0).cpu().numpy()

        # Find connected components of entities
        entities = []
        visited = np.zeros_like(mask, dtype=bool)

        for y in range(mask.shape[0]):
            for x_pos in range(mask.shape[1]):
                if visited[y, x_pos]:
                    continue
                if mask[y, x_pos] < self.detection_threshold:
                    continue

                # Found entity pixel - flood fill to find extent
                entity_pixels = self._flood_fill(
                    mask, visited, x_pos, y, self.detection_threshold
                )

                if not entity_pixels:
                    continue

                # Get bounding box
                xs = [p[0] for p in entity_pixels]
                ys = [p[1] for p in entity_pixels]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Get most common class prediction
                class_votes = {}
                for px, py in entity_pixels:
                    c = int(class_preds[py, px])
                    class_votes[c] = class_votes.get(c, 0) + 1
                entity_class = max(class_votes, key=class_votes.get)

                # Get confidence (average probability)
                confidences = [class_probs[py, px, entity_class] for px, py in entity_pixels]
                avg_confidence = sum(confidences) / len(confidences)

                # Get color (most common)
                colors = [int(grid[py, px]) for px, py in entity_pixels]
                entity_color = max(set(colors), key=colors.count)

                entities.append(DetectedEntity(
                    entity_type=EntityType(entity_class),
                    x=min_x,
                    y=min_y,
                    width=max_x - min_x + 1,
                    height=max_y - min_y + 1,
                    color=entity_color,
                    confidence=avg_confidence,
                ))

        return entities

    def _flood_fill(
        self,
        mask: np.ndarray,
        visited: np.ndarray,
        start_x: int,
        start_y: int,
        threshold: float,
    ) -> list[tuple[int, int]]:
        """Flood fill to find connected entity pixels."""
        h, w = mask.shape
        stack = [(start_x, start_y)]
        pixels = []

        while stack:
            x, y = stack.pop()
            if x < 0 or x >= w or y < 0 or y >= h:
                continue
            if visited[y, x]:
                continue
            if mask[y, x] < threshold:
                continue

            visited[y, x] = True
            pixels.append((x, y))

            # Add neighbors (4-connected)
            stack.extend([
                (x + 1, y),
                (x - 1, y),
                (x, y + 1),
                (x, y - 1),
            ])

        return pixels

    @torch.no_grad()
    def describe_scene(
        self,
        grid: np.ndarray,
        device: torch.device = torch.device("cpu"),
    ) -> SceneDescription:
        """
        Generate structured scene description.

        Args:
            grid: [H, W] color indices
            device: Torch device

        Returns:
            SceneDescription with entities and player position
        """
        entities = self.extract_entities(grid, device)

        # Find player
        player_position = None
        for e in entities:
            if e.entity_type == EntityType.PLAYER:
                player_position = (e.x, e.y)
                break

        return SceneDescription(
            entities=entities,
            player_position=player_position,
            grid_size=(grid.shape[1], grid.shape[0]),
        )


def create_visual_grounding_module(
    num_colors: int = 16,
    grid_size: int = 64,
) -> VisualGroundingModule:
    """Factory function to create visual grounding module with default config."""
    return VisualGroundingModule(
        num_colors=num_colors,
        embed_dim=32,
        hidden_dim=64,
        feature_dim=32,
        num_classes=6,  # player, goal, item, obstacle, trigger, unknown
    )


if __name__ == "__main__":
    # Test the module
    print("Testing VisualGroundingModule...")

    module = create_visual_grounding_module()

    # Count parameters
    num_params = sum(p.numel() for p in module.parameters())
    print(f"Total parameters: {num_params:,}")

    # Test forward pass
    batch_size = 4
    grid_size = 32
    x = torch.randint(0, 16, (batch_size, grid_size, grid_size))

    mask, features, class_logits = module(x)
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Class logits shape: {class_logits.shape}")

    # Test entity extraction
    from .pretraining.synthetic_games import SyntheticGameConfig, SyntheticGameGenerator

    config = SyntheticGameConfig(grid_size=32, num_items=3, num_obstacles=5)
    generator = SyntheticGameGenerator(config)
    generator.seed(42)

    state = generator.generate()
    print(f"\nGenerated state with {len(state.entities)} entities")

    # Extract entities (untrained, so results will be poor)
    detected = module.extract_entities(state.grid)
    print(f"Detected {len(detected)} entities (untrained model)")

    # Generate scene description
    scene = module.describe_scene(state.grid)
    print(f"\nScene description:\n{scene.to_language()}")

    # Test movement correlator
    correlator = MovementCorrelator()
    next_state, reward, event = generator.generate_transition(state, action=4)  # right
    player_pos = correlator.identify_player(state.grid, 4, next_state.grid)
    print(f"\nMovement correlator found player at: {player_pos}")
    print(f"Actual player position: {next_state.player_position}")
