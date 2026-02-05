"""
Visual Grounding Training V2 - Entity-Level Classification.

Key improvements over V1:
1. Entity-level classification (not per-pixel)
2. Focal loss for class imbalance
3. Entity-specific features (size, color, position)
4. More visually distinct entity patterns

Usage:
    uv run python -m src.aria_v2.pretraining.visual_grounding_trainer_v2 --num_samples 10000
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# Improved Synthetic Data with Distinct Entity Patterns
# ============================================================================

class EntityType:
    """Entity type labels."""
    BACKGROUND = 0
    PLAYER = 1
    GOAL = 2
    ITEM = 3
    OBSTACLE = 4
    TRIGGER = 5


@dataclass
class DistinctEntity:
    """Entity with distinctive visual pattern."""
    entity_type: int
    x: int
    y: int
    width: int
    height: int
    color: int
    pattern: str  # "solid", "hollow", "cross", "dot"


@dataclass
class DistinctGameConfig:
    """Config for generating visually distinct entities."""
    grid_size: int = 32
    num_items: int = 5
    num_obstacles: int = 8
    num_triggers: int = 2
    has_goal: bool = True
    min_spacing: int = 2


class DistinctGameGenerator:
    """Generate games with visually distinct entity patterns."""

    # Each entity type has a distinct visual pattern
    ENTITY_PATTERNS = {
        EntityType.PLAYER: ("solid", 1, 1),      # Single pixel, unique color
        EntityType.GOAL: ("hollow", 3, 3),       # Hollow square
        EntityType.ITEM: ("dot", 1, 1),          # Small dots
        EntityType.OBSTACLE: ("solid", 2, 3),    # Solid rectangles
        EntityType.TRIGGER: ("cross", 3, 3),     # Cross pattern
    }

    # Distinct colors for each type
    ENTITY_COLORS = {
        EntityType.PLAYER: 1,   # Blue
        EntityType.GOAL: 3,     # Green
        EntityType.ITEM: 4,     # Yellow
        EntityType.OBSTACLE: 5, # Grey
        EntityType.TRIGGER: 7,  # Orange
    }

    def __init__(self, config: DistinctGameConfig, seed: int = 42):
        self.config = config
        self.rng = np.random.RandomState(seed)

    def _draw_pattern(
        self,
        grid: np.ndarray,
        labels: np.ndarray,
        mask: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        color: int,
        pattern: str,
        entity_type: int,
    ):
        """Draw entity pattern on grid."""
        if pattern == "solid":
            grid[y:y+height, x:x+width] = color
            labels[y:y+height, x:x+width] = entity_type
            mask[y:y+height, x:x+width] = 1

        elif pattern == "hollow":
            # Draw border only
            for dy in range(height):
                for dx in range(width):
                    if dy == 0 or dy == height-1 or dx == 0 or dx == width-1:
                        if 0 <= y+dy < grid.shape[0] and 0 <= x+dx < grid.shape[1]:
                            grid[y+dy, x+dx] = color
                            labels[y+dy, x+dx] = entity_type
                            mask[y+dy, x+dx] = 1

        elif pattern == "cross":
            # Draw + shape
            mid_x, mid_y = width // 2, height // 2
            for dy in range(height):
                if 0 <= y+dy < grid.shape[0] and 0 <= x+mid_x < grid.shape[1]:
                    grid[y+dy, x+mid_x] = color
                    labels[y+dy, x+mid_x] = entity_type
                    mask[y+dy, x+mid_x] = 1
            for dx in range(width):
                if 0 <= y+mid_y < grid.shape[0] and 0 <= x+dx < grid.shape[1]:
                    grid[y+mid_y, x+dx] = color
                    labels[y+mid_y, x+dx] = entity_type
                    mask[y+mid_y, x+dx] = 1

        elif pattern == "dot":
            grid[y, x] = color
            labels[y, x] = entity_type
            mask[y, x] = 1

    def generate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[DistinctEntity]]:
        """Generate a game state with distinct entity patterns."""
        cfg = self.config
        size = cfg.grid_size

        grid = np.zeros((size, size), dtype=np.int32)
        labels = np.zeros((size, size), dtype=np.int32)
        mask = np.zeros((size, size), dtype=np.int32)
        entities: list[DistinctEntity] = []
        occupied: set[tuple[int, int]] = set()

        def can_place(x: int, y: int, w: int, h: int) -> bool:
            """Check if position is available."""
            for dy in range(-cfg.min_spacing, h + cfg.min_spacing):
                for dx in range(-cfg.min_spacing, w + cfg.min_spacing):
                    if (x + dx, y + dy) in occupied:
                        return False
            return 0 <= x and x + w <= size and 0 <= y and y + h <= size

        def place_entity(entity_type: int) -> Optional[DistinctEntity]:
            """Try to place entity with its distinct pattern."""
            pattern, base_w, base_h = self.ENTITY_PATTERNS[entity_type]
            color = self.ENTITY_COLORS[entity_type]

            # Add some size variation for obstacles
            if entity_type == EntityType.OBSTACLE:
                w = self.rng.randint(1, 4)
                h = self.rng.randint(1, 4)
            else:
                w, h = base_w, base_h

            for _ in range(100):
                x = self.rng.randint(0, size - w)
                y = self.rng.randint(0, size - h)

                if can_place(x, y, w, h):
                    self._draw_pattern(grid, labels, mask, x, y, w, h, color, pattern, entity_type)

                    # Mark as occupied
                    for dy in range(h):
                        for dx in range(w):
                            occupied.add((x + dx, y + dy))

                    return DistinctEntity(
                        entity_type=entity_type,
                        x=x, y=y, width=w, height=h,
                        color=color, pattern=pattern
                    )
            return None

        # Place entities in order of importance
        player = place_entity(EntityType.PLAYER)
        if player:
            entities.append(player)

        if cfg.has_goal:
            goal = place_entity(EntityType.GOAL)
            if goal:
                entities.append(goal)

        for _ in range(cfg.num_items):
            item = place_entity(EntityType.ITEM)
            if item:
                entities.append(item)

        for _ in range(cfg.num_obstacles):
            obs = place_entity(EntityType.OBSTACLE)
            if obs:
                entities.append(obs)

        for _ in range(cfg.num_triggers):
            trig = place_entity(EntityType.TRIGGER)
            if trig:
                entities.append(trig)

        return grid, labels, mask, entities


# ============================================================================
# Entity-Level Dataset
# ============================================================================

class EntityLevelDataset(Dataset):
    """Dataset that extracts individual entities for classification."""

    def __init__(
        self,
        num_samples: int,
        grid_size: int = 32,
        seed: int = 42,
        crop_size: int = 5,  # Size of context around entity
    ):
        self.crop_size = crop_size
        self.samples = []  # List of (crop, label) tuples

        generator = DistinctGameGenerator(
            DistinctGameConfig(grid_size=grid_size),
            seed=seed
        )

        for i in range(num_samples):
            generator.rng = np.random.RandomState(seed + i)
            grid, labels, mask, entities = generator.generate()

            for entity in entities:
                # Extract crop centered on entity
                cx = entity.x + entity.width // 2
                cy = entity.y + entity.height // 2

                x1 = max(0, cx - crop_size // 2)
                y1 = max(0, cy - crop_size // 2)
                x2 = min(grid_size, x1 + crop_size)
                y2 = min(grid_size, y1 + crop_size)

                crop = np.zeros((crop_size, crop_size), dtype=np.int32)
                crop_h = y2 - y1
                crop_w = x2 - x1
                crop[:crop_h, :crop_w] = grid[y1:y2, x1:x2]

                self.samples.append({
                    "crop": crop,
                    "label": entity.entity_type,
                    "size": (entity.width, entity.height),
                    "color": entity.color,
                })

        # Also add background crops
        for i in range(num_samples // 2):
            generator.rng = np.random.RandomState(seed + num_samples + i)
            grid, labels, mask, entities = generator.generate()

            # Find random background location
            for _ in range(10):
                x = np.random.randint(0, grid_size - crop_size)
                y = np.random.randint(0, grid_size - crop_size)
                if mask[y:y+crop_size, x:x+crop_size].sum() == 0:
                    self.samples.append({
                        "crop": grid[y:y+crop_size, x:x+crop_size].copy(),
                        "label": EntityType.BACKGROUND,
                        "size": (0, 0),
                        "color": 0,
                    })
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        return {
            "crop": torch.from_numpy(sample["crop"]).long(),
            "label": torch.tensor(sample["label"]).long(),
            "size": torch.tensor(sample["size"]).float(),
            "color": torch.tensor(sample["color"]).long(),
        }


# ============================================================================
# Entity Classifier with Better Features
# ============================================================================

class EntityClassifierV2(nn.Module):
    """
    Entity-level classifier with better features.

    Input: 5x5 crop around entity center
    Output: 6-class probabilities
    """

    def __init__(
        self,
        num_colors: int = 16,
        embed_dim: int = 16,
        hidden_dim: int = 64,
        num_classes: int = 6,
        crop_size: int = 5,
    ):
        super().__init__()

        self.num_colors = num_colors
        self.crop_size = crop_size

        # Color embedding
        self.color_embed = nn.Embedding(num_colors, embed_dim)

        # Conv layers for spatial patterns
        self.conv1 = nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

        # Global pooling + classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for size (2) and color (1)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        crop: torch.Tensor,
        size: torch.Tensor,
        color: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            crop: [B, H, W] color indices
            size: [B, 2] entity size (width, height)
            color: [B] entity color

        Returns:
            logits: [B, num_classes]
        """
        # Embed colors: [B, H, W] -> [B, H, W, E]
        x = self.color_embed(crop)

        # Conv: [B, H, W, E] -> [B, E, H, W]
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Global pool: [B, E, H, W] -> [B, E]
        x = x.mean(dim=(2, 3))

        # Normalize size
        size_norm = size / 5.0  # Normalize by max expected size

        # Color one-hot (simplified to single value)
        color_feat = color.float().unsqueeze(1) / self.num_colors

        # Concat features
        features = torch.cat([x, size_norm, color_feat], dim=1)

        return self.classifier(features)


# ============================================================================
# Training with Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for class imbalance."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


@dataclass
class TrainingConfigV2:
    """Training configuration."""
    num_samples: int = 10000
    batch_size: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 30
    grid_size: int = 32
    val_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    focal_gamma: float = 2.0


def train_entity_classifier(config: TrainingConfigV2) -> EntityClassifierV2:
    """Train entity-level classifier."""
    print(f"Training entity classifier with {config.num_samples} game states")
    print(f"Device: {config.device}")

    device = torch.device(config.device)

    # Create dataset
    print("Generating entity-level dataset...")
    full_dataset = EntityLevelDataset(
        num_samples=config.num_samples,
        grid_size=config.grid_size,
    )
    print(f"Total entity samples: {len(full_dataset)}")

    # Count class distribution
    class_counts = {}
    for i in range(len(full_dataset)):
        label = full_dataset.samples[i]["label"]
        class_counts[label] = class_counts.get(label, 0) + 1
    print("Class distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  Class {label}: {count} ({count/len(full_dataset)*100:.1f}%)")

    # Split
    split_idx = int(len(full_dataset) * (1 - config.val_split))
    train_dataset = torch.utils.data.Subset(full_dataset, range(split_idx))
    val_dataset = torch.utils.data.Subset(full_dataset, range(split_idx, len(full_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Model
    model = EntityClassifierV2().to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Class weights for focal loss (inverse frequency)
    total = sum(class_counts.values())
    weights = torch.tensor([total / (class_counts.get(i, 1) * 6) for i in range(6)]).to(device)
    weights = weights / weights.sum() * 6  # Normalize
    print(f"Class weights: {weights.cpu().numpy()}")

    criterion = FocalLoss(gamma=config.focal_gamma, alpha=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    best_acc = 0.0
    best_state = None

    for epoch in range(config.num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            crop = batch["crop"].to(device)
            label = batch["label"].to(device)
            size = batch["size"].to(device)
            color = batch["color"].to(device)

            optimizer.zero_grad()
            logits = model(crop, size, color)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == label).sum().item()
            train_total += label.size(0)

        scheduler.step()

        # Evaluate
        model.eval()
        val_correct = 0
        val_total = 0
        class_correct = {i: 0 for i in range(6)}
        class_total = {i: 0 for i in range(6)}

        with torch.no_grad():
            for batch in val_loader:
                crop = batch["crop"].to(device)
                label = batch["label"].to(device)
                size = batch["size"].to(device)
                color = batch["color"].to(device)

                logits = model(crop, size, color)
                preds = logits.argmax(1)

                val_correct += (preds == label).sum().item()
                val_total += label.size(0)

                for i in range(6):
                    mask = label == i
                    class_total[i] += mask.sum().item()
                    class_correct[i] += ((preds == label) & mask).sum().item()

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{config.num_epochs}: "
              f"train_loss={train_loss/len(train_loader):.4f}, "
              f"train_acc={train_acc:.3f}, val_acc={val_acc:.3f}")

        # Per-class accuracy
        if (epoch + 1) % 10 == 0 or epoch == config.num_epochs - 1:
            print("  Per-class accuracy:")
            for i in range(6):
                if class_total[i] > 0:
                    acc = class_correct[i] / class_total[i]
                    print(f"    Class {i}: {acc:.3f} ({class_correct[i]}/{class_total[i]})")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()

    print(f"\nTraining complete. Best accuracy: {best_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)

    return model


def main():
    parser = argparse.ArgumentParser(description="Train entity classifier V2")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="checkpoints/entity_classifier_v2.pt")

    args = parser.parse_args()

    config = TrainingConfigV2(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )

    model = train_entity_classifier(config)

    # Save
    import os
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, args.save_path)
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
