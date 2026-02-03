"""
ARIA Perception Tower

Parallel neural + symbolic encoding of grid states.
"""

from collections import deque
from typing import List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import ARIAConfig
from .types import (
    GridObject,
    PerceptionOutput,
    SpatialRelation,
    SymbolicState,
)


class PerceptionTower(nn.Module):
    """
    Parallel encoding: neural for gradients, symbolic for structure.

    Neural path:
    - Color embedding + 2D positional encoding
    - Multi-scale CNN for local patterns
    - Lightweight transformer for global context

    Symbolic path:
    - Connected component detection
    - Spatial relation extraction
    - Agent localization

    Memory: ~50MB total (fits easily on 4090)
    """

    def __init__(self, config: ARIAConfig):
        super().__init__()

        self.config = config
        self.grid_size = config.grid_size
        self.num_colors = config.num_colors
        self.embed_dim = config.embed_dim
        self.hidden_dim = config.hidden_dim

        # === NEURAL PATH ===
        # Color embedding
        self.color_embed = nn.Embedding(config.num_colors, config.embed_dim)

        # 2D positional encoding (learned, factorized)
        self.pos_embed_x = nn.Embedding(config.grid_size, config.embed_dim // 2)
        self.pos_embed_y = nn.Embedding(config.grid_size, config.embed_dim // 2)

        # Multi-scale local pattern extractor
        self.local_cnn = nn.Sequential(
            # 3x3 receptive field
            nn.Conv2d(config.embed_dim, config.hidden_dim, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.GELU(),
            # 5x5 effective receptive field
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.GELU(),
            # 7x7 effective receptive field
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.GELU(),
        )

        # Global context transformer
        self.global_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,  # Pre-LN for stability
            ),
            num_layers=config.num_transformer_layers,
        )

        # Final projection
        self.output_proj = nn.Conv2d(config.hidden_dim, config.hidden_dim, 1)

        # === SYMBOLIC PATH ===
        self.object_detector = ConnectedComponentDetector()
        self.relation_extractor = SpatialRelationExtractor()

        # Register position indices for efficient lookup
        self.register_buffer(
            "x_indices",
            torch.arange(config.grid_size),
        )
        self.register_buffer(
            "y_indices",
            torch.arange(config.grid_size),
        )

    def forward(
        self,
        grid: torch.Tensor,  # [B, H, W] or [B, T, H, W] int
        return_symbolic: bool = True,
    ) -> PerceptionOutput:
        """
        Process grid observation.

        Args:
            grid: Integer grid [B, H, W] or [B, T, H, W] for temporal stacks
            return_symbolic: Whether to extract symbolic state (can be disabled for speed)

        Returns:
            PerceptionOutput with neural features and optional symbolic state
        """
        # Handle temporal dimension
        has_temporal = grid.dim() == 4
        if has_temporal:
            B, T, H, W = grid.shape
            # Process last frame (or could do temporal fusion here)
            grid = grid[:, -1]  # [B, H, W]
        else:
            B, H, W = grid.shape

        # === NEURAL ENCODING ===
        # Color embedding
        color_feats = self.color_embed(grid)  # [B, H, W, embed_dim]

        # Add 2D positional encoding
        x_pos = self.pos_embed_x(self.x_indices[:W])  # [W, embed_dim/2]
        y_pos = self.pos_embed_y(self.y_indices[:H])  # [H, embed_dim/2]

        # Broadcast positional embeddings
        pos_feats = torch.cat(
            [
                x_pos.unsqueeze(0).expand(H, -1, -1),  # [H, W, embed_dim/2]
                y_pos.unsqueeze(1).expand(-1, W, -1),  # [H, W, embed_dim/2]
            ],
            dim=-1,
        )  # [H, W, embed_dim]

        feats = color_feats + pos_feats.unsqueeze(0)  # [B, H, W, embed_dim]

        # Permute for Conv2d: [B, embed_dim, H, W]
        feats = feats.permute(0, 3, 1, 2)

        # Local CNN
        local_feats = self.local_cnn(feats)  # [B, hidden_dim, H, W]

        # Global transformer
        # Flatten spatial dims: [B, H*W, hidden_dim]
        flat_feats = local_feats.flatten(2).transpose(1, 2)
        global_feats = self.global_attn(flat_feats)

        # Reshape back: [B, hidden_dim, H, W]
        neural_features = global_feats.transpose(1, 2).view(B, -1, H, W)

        # Final projection
        neural_features = self.output_proj(neural_features)

        # === SYMBOLIC EXTRACTION ===
        symbolic_state = None
        if return_symbolic:
            # Run on CPU (can be parallelized)
            grid_np = grid.cpu().numpy()
            symbolic_state = self._extract_symbolic(grid_np)

        return PerceptionOutput(
            neural_features=neural_features,
            symbolic_state=symbolic_state,
            grid=grid,
        )

    def _extract_symbolic(self, grid_np: np.ndarray) -> SymbolicState:
        """Extract symbolic state from numpy grid."""
        # Assume single batch for now
        if grid_np.ndim == 3:
            grid_np = grid_np[0]

        objects = self.object_detector(grid_np)
        relations = self.relation_extractor(objects)
        agent_pos = self._find_agent(objects)

        return SymbolicState(
            objects=objects,
            relations=relations,
            agent_pos=agent_pos,
            raw_grid=grid_np,
        )

    def _find_agent(self, objects: List[GridObject]) -> Optional[Tuple[int, int]]:
        """Find agent position from detected objects."""
        for obj in objects:
            if obj.is_agent:
                return (int(obj.centroid[0]), int(obj.centroid[1]))

        # Heuristic: agent often has specific colors (blue body, orange head in ARC-AGI-3)
        # Color 9 = Blue, Color 12 = Orange based on the palette
        agent_colors = {9, 12}
        for obj in objects:
            if obj.color in agent_colors and obj.area < 20:  # Agent is usually small
                return (int(obj.centroid[0]), int(obj.centroid[1]))

        return None


class ConnectedComponentDetector:
    """
    Fast connected component detection using flood fill.
    Extracts objects from grid based on color connectivity.
    """

    def __init__(self, min_area: int = 1, max_objects: int = 100):
        self.min_area = min_area
        self.max_objects = max_objects

        # Agent color heuristics (can be learned)
        self.agent_colors = {9, 12}  # Blue, Orange

    def __call__(self, grid: np.ndarray) -> List[GridObject]:
        """
        Detect connected components in grid.

        Args:
            grid: 2D numpy array of color indices

        Returns:
            List of GridObject
        """
        H, W = grid.shape
        visited = np.zeros((H, W), dtype=bool)
        objects = []
        obj_id = 0

        for y in range(H):
            for x in range(W):
                if visited[y, x]:
                    continue

                color = grid[y, x]
                if color == 0:  # Skip background (white)
                    visited[y, x] = True
                    continue

                # Flood fill to find connected component
                pixels = self._flood_fill(grid, x, y, color, visited)

                if len(pixels) >= self.min_area:
                    obj = self._create_object(obj_id, color, pixels)
                    objects.append(obj)
                    obj_id += 1

                    if obj_id >= self.max_objects:
                        return objects

        return objects

    def _flood_fill(
        self,
        grid: np.ndarray,
        start_x: int,
        start_y: int,
        color: int,
        visited: np.ndarray,
    ) -> Set[Tuple[int, int]]:
        """BFS flood fill to find connected pixels."""
        H, W = grid.shape
        pixels = set()
        queue = deque([(start_x, start_y)])

        while queue:
            x, y = queue.popleft()

            if x < 0 or x >= W or y < 0 or y >= H:
                continue
            if visited[y, x]:
                continue
            if grid[y, x] != color:
                continue

            visited[y, x] = True
            pixels.add((x, y))

            # 4-connectivity
            queue.extend(
                [
                    (x + 1, y),
                    (x - 1, y),
                    (x, y + 1),
                    (x, y - 1),
                ]
            )

        return pixels

    def _create_object(
        self,
        obj_id: int,
        color: int,
        pixels: Set[Tuple[int, int]],
    ) -> GridObject:
        """Create GridObject from pixels."""
        xs = [p[0] for p in pixels]
        ys = [p[1] for p in pixels]

        bbox = (min(xs), min(ys), max(xs), max(ys))
        centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        area = len(pixels)

        # Detect shape
        shape = self._detect_shape(pixels, bbox)

        # Detect if agent
        is_agent = color in self.agent_colors and area < 20

        return GridObject(
            id=obj_id,
            color=color,
            pixels=pixels,
            bbox=bbox,
            centroid=centroid,
            shape_signature=shape,
            area=area,
            is_agent=is_agent,
        )

    def _detect_shape(
        self,
        pixels: Set[Tuple[int, int]],
        bbox: Tuple[int, int, int, int],
    ) -> str:
        """Classify shape based on pixel pattern."""
        x1, y1, x2, y2 = bbox
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        area = len(pixels)
        bbox_area = width * height

        # Fill ratio
        fill_ratio = area / bbox_area if bbox_area > 0 else 0

        if fill_ratio > 0.9:
            if width == 1:
                return "line_v"
            elif height == 1:
                return "line_h"
            elif abs(width - height) <= 2:
                return "square"
            else:
                return "rectangle"
        elif fill_ratio > 0.4:
            # Could be L-shape, T-shape, etc.
            return "L"
        else:
            return "custom"


class SpatialRelationExtractor:
    """
    Extract spatial relations between objects.
    """

    def __init__(self, adjacency_threshold: int = 2):
        self.adj_threshold = adjacency_threshold

    def __call__(self, objects: List[GridObject]) -> List[SpatialRelation]:
        """
        Extract all pairwise spatial relations.

        Args:
            objects: List of GridObject

        Returns:
            List of SpatialRelation
        """
        relations = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:
                    continue

                # Compute relation
                rel = self._compute_relation(obj1, obj2)
                if rel:
                    relations.append(rel)

        return relations

    def _compute_relation(
        self,
        obj1: GridObject,
        obj2: GridObject,
    ) -> Optional[SpatialRelation]:
        """Compute relation between two objects."""
        c1 = obj1.centroid
        c2 = obj2.centroid

        dx = c2[0] - c1[0]
        dy = c2[1] - c1[1]
        distance = (dx * dx + dy * dy) ** 0.5

        # Determine primary relation based on relative position
        if abs(dx) > abs(dy):
            relation = "left_of" if dx > 0 else "right_of"
        else:
            relation = "above" if dy > 0 else "below"

        # Check for adjacency
        if self._are_adjacent(obj1, obj2):
            relation = "adjacent"

        return SpatialRelation(
            subject_id=obj1.id,
            relation=relation,
            object_id=obj2.id,
            distance=distance,
        )

    def _are_adjacent(self, obj1: GridObject, obj2: GridObject) -> bool:
        """Check if objects are adjacent (within threshold)."""
        for x1, y1 in obj1.pixels:
            for x2, y2 in obj2.pixels:
                if abs(x1 - x2) + abs(y1 - y2) <= self.adj_threshold:
                    return True
        return False


# Torchscript-compatible wrapper for JIT compilation
class ScriptablePerceptionTower(nn.Module):
    """
    Simplified perception tower for TorchScript compilation.
    Only neural path, no symbolic extraction.
    """

    def __init__(self, config: ARIAConfig):
        super().__init__()

        self.color_embed = nn.Embedding(config.num_colors, config.embed_dim)
        self.pos_embed_x = nn.Embedding(config.grid_size, config.embed_dim // 2)
        self.pos_embed_y = nn.Embedding(config.grid_size, config.embed_dim // 2)

        self.local_cnn = nn.Sequential(
            nn.Conv2d(config.embed_dim, config.hidden_dim, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.GELU(),
            nn.Conv2d(config.hidden_dim, config.hidden_dim, 3, padding=1),
            nn.GroupNorm(8, config.hidden_dim),
            nn.GELU(),
        )

        self.global_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=config.num_transformer_layers,
        )

        self.output_proj = nn.Conv2d(config.hidden_dim, config.hidden_dim, 1)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Args:
            grid: [B, H, W] int tensor

        Returns:
            features: [B, hidden_dim, H, W]
        """
        B, H, W = grid.shape
        device = grid.device

        # Color embedding
        color_feats = self.color_embed(grid)  # [B, H, W, embed]

        # Position embedding
        x_idx = torch.arange(W, device=device)
        y_idx = torch.arange(H, device=device)
        x_pos = self.pos_embed_x(x_idx)  # [W, embed/2]
        y_pos = self.pos_embed_y(y_idx)  # [H, embed/2]

        pos_feats = torch.cat(
            [
                x_pos.unsqueeze(0).expand(H, -1, -1),
                y_pos.unsqueeze(1).expand(-1, W, -1),
            ],
            dim=-1,
        )

        feats = color_feats + pos_feats.unsqueeze(0)
        feats = feats.permute(0, 3, 1, 2)

        local_feats = self.local_cnn(feats)
        flat_feats = local_feats.flatten(2).transpose(1, 2)
        global_feats = self.global_attn(flat_feats)
        neural_features = global_feats.transpose(1, 2).view(B, -1, H, W)

        return self.output_proj(neural_features)
