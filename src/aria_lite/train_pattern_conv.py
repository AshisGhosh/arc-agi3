"""
Convolutional template matching.

Key insight: Template matching is inherently a convolution operation.
Slide template over grid and compute similarity at each position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.aria_lite.primitives import PatternEnv
from src.aria_lite.primitives.base import Action


class ConvTemplateMatching(nn.Module):
    """
    Template matching using convolution.

    1. Embed grid and template to feature space
    2. Use template features as conv kernel
    3. Slide over grid to get match scores
    4. Argmax of match scores = template location
    """

    def __init__(self, num_colors: int = 16, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim

        # Embedding for cell values
        self.embed = nn.Embedding(num_colors, embed_dim)

        # Project embeddings to matching space
        self.grid_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

        self.template_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, 1),
        )

    def forward(self, grid: torch.Tensor, template: torch.Tensor) -> dict:
        """
        Find template in grid using convolution.

        Args:
            grid: [B, H, W] main grid
            template: [B, Th, Tw] template

        Returns:
            dict with match_scores [B, H-Th+1, W-Tw+1] and predicted x, y
        """
        B = grid.shape[0]
        H, W = grid.shape[1], grid.shape[2]
        Th, Tw = template.shape[1], template.shape[2]

        # Embed
        grid_embed = self.embed(grid.long().clamp(0, 15))  # [B, H, W, D]
        grid_embed = grid_embed.permute(0, 3, 1, 2)  # [B, D, H, W]

        template_embed = self.embed(template.long().clamp(0, 15))  # [B, Th, Tw, D]
        template_embed = template_embed.permute(0, 3, 1, 2)  # [B, D, Th, Tw]

        # Project
        grid_feat = self.grid_proj(grid_embed)  # [B, D, H, W]
        template_feat = self.template_proj(template_embed)  # [B, D, Th, Tw]

        # Compute match scores via convolution
        # For each sample, use its template as the conv kernel
        match_scores = []
        for b in range(B):
            g = grid_feat[b:b+1]  # [1, D, H, W]
            t = template_feat[b]  # [D, Th, Tw]

            # Convolution: slide template over grid
            # Output: [1, 1, H-Th+1, W-Tw+1]
            score = F.conv2d(g, t.unsqueeze(0))
            match_scores.append(score)

        match_scores = torch.cat(match_scores, dim=0)  # [B, 1, H-Th+1, W-Tw+1]
        match_scores = match_scores.squeeze(1)  # [B, H-Th+1, W-Tw+1]

        # Find argmax
        flat_scores = match_scores.view(B, -1)
        max_idx = flat_scores.argmax(dim=-1)

        out_h = H - Th + 1
        out_w = W - Tw + 1

        pred_y = max_idx // out_w
        pred_x = max_idx % out_w

        return {
            "match_scores": match_scores,
            "pred_x": pred_x,
            "pred_y": pred_y,
        }


def train_conv_matching():
    """Train convolutional template matching."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # Collect data
    print("Collecting data...")
    data = []
    for _ in range(2000):
        env = PatternEnv(
            grid_size=10,
            pattern_size=3,
            num_colors=4,
            max_steps=1,
            variant="match",
        )
        obs = env.reset()

        grid = obs[:10, :]
        template = obs[10:13, :3]
        y, x = env.target_pos

        data.append({
            "grid": grid.clone(),
            "template": template.clone(),
            "x": x,
            "y": y,
        })

    print(f"Collected {len(data)} samples")

    # Stack
    grids = torch.stack([d["grid"] for d in data])
    templates = torch.stack([d["template"] for d in data])
    xs = torch.tensor([d["x"] for d in data], dtype=torch.long)
    ys = torch.tensor([d["y"] for d in data], dtype=torch.long)

    # Model
    model = ConvTemplateMatching(num_colors=16, embed_dim=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 64
    num_epochs = 100

    for epoch in range(num_epochs):
        perm = torch.randperm(len(grids))
        grids = grids[perm]
        templates = templates[perm]
        xs = xs[perm]
        ys = ys[perm]

        total_loss = 0
        correct = 0
        total = 0

        for i in range(0, len(grids), batch_size):
            grid_batch = grids[i:i+batch_size].to(device)
            template_batch = templates[i:i+batch_size].to(device)
            x_batch = xs[i:i+batch_size].to(device)
            y_batch = ys[i:i+batch_size].to(device)

            out = model(grid_batch, template_batch)

            # Loss: cross-entropy on flattened match scores
            B = grid_batch.shape[0]
            scores_flat = out["match_scores"].view(B, -1)  # [B, 8*8=64]

            # Target index
            out_w = 8  # 10 - 3 + 1
            target_idx = y_batch * out_w + x_batch

            loss = F.cross_entropy(scores_flat, target_idx)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accuracy
            correct += ((out["pred_x"] == x_batch) & (out["pred_y"] == y_batch)).sum().item()
            total += B

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/(i//batch_size+1):.4f}, acc={correct/total:.1%}")

    # Evaluate
    print("\nEvaluating...")
    successes = []
    for _ in range(200):
        env = PatternEnv(grid_size=10, pattern_size=3, num_colors=4, variant="match")
        obs = env.reset()

        grid = obs[:10, :]
        template = obs[10:13, :3]

        with torch.no_grad():
            out = model(grid.unsqueeze(0).to(device), template.unsqueeze(0).to(device))
            pred_x = out["pred_x"].item()
            pred_y = out["pred_y"].item()

        result = env.step(Action.CLICK, pred_x, pred_y)
        successes.append(1.0 if result.success else 0.0)

    eval_success = sum(successes) / len(successes)
    print(f"\nConv template matching eval: {eval_success:.1%}")
    return eval_success


def train_direct_comparison():
    """
    Simpler approach: directly compare template to each grid region.

    No learning needed - just compare cell values directly.
    """
    print("\n--- Direct Comparison Baseline ---")

    successes = []
    for _ in range(200):
        env = PatternEnv(grid_size=10, pattern_size=3, num_colors=4, variant="match")
        obs = env.reset()

        grid = obs[:10, :]  # [10, 10]
        template = obs[10:13, :3]  # [3, 3]

        # Slide template over grid, count matches
        best_score = -1
        best_x, best_y = 0, 0

        for y in range(8):  # 10 - 3 + 1
            for x in range(8):
                region = grid[y:y+3, x:x+3]
                score = (region == template).float().sum().item()
                if score > best_score:
                    best_score = score
                    best_x, best_y = x, y

        result = env.step(Action.CLICK, best_x, best_y)
        successes.append(1.0 if result.success else 0.0)

    success_rate = sum(successes) / len(successes)
    print(f"Direct comparison: {success_rate:.1%}")
    return success_rate


if __name__ == "__main__":
    print("=" * 60)
    print("CONVOLUTIONAL TEMPLATE MATCHING")
    print("=" * 60)

    # First, test direct comparison (no learning)
    direct_success = train_direct_comparison()

    # Then, train learned matching
    print("\n--- Learned Conv Matching ---")
    conv_success = train_conv_matching()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Direct comparison: {direct_success:.1%} {'PASS' if direct_success > 0.5 else 'FAIL'}")
    print(f"Learned conv: {conv_success:.1%} {'PASS' if conv_success > 0.5 else 'FAIL'}")
