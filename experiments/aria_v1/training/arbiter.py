"""
ARIA-Lite Arbiter

Metacognitive controller that decides when to use the fast policy (System 1)
versus the slow policy (System 2) based on confidence and uncertainty signals.

Decision Logic:
    - Use slow policy when:
        - Fast policy confidence < threshold (0.7)
        - World model uncertainty > threshold (0.3)
        - High novelty detected
        - Value estimate indicates high-stakes decision

Target: ~0 parameters (heuristic) or ~1K (learned)
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .config import ArbiterConfig, ARIALiteConfig
from .fast_policy import FastPolicyOutput
from .slow_policy import SlowPolicyOutput


@dataclass
class ArbiterDecision:
    """Decision from the arbiter."""

    use_slow: torch.Tensor  # [B] bool tensor
    confidence: torch.Tensor  # [B] fast policy confidence
    uncertainty: torch.Tensor  # [B] world model uncertainty
    novelty: torch.Tensor  # [B] novelty score
    reason: str  # Human-readable reason


class Arbiter(nn.Module):
    """
    Metacognitive arbiter for fast/slow policy switching.

    Uses heuristic rules by default, can optionally learn switching.
    """

    def __init__(self, config: Optional[ArbiterConfig] = None):
        super().__init__()

        if config is None:
            config = ArbiterConfig()

        self.config = config
        self.confidence_threshold = config.confidence_threshold
        self.uncertainty_threshold = config.uncertainty_threshold
        self.novelty_threshold = config.novelty_threshold

        # Optional learned switching MLP
        if config.use_learned_switching:
            self.switch_network = nn.Sequential(
                nn.Linear(4, config.switching_hidden_dim),
                nn.ReLU(),
                nn.Linear(config.switching_hidden_dim, 1),
            )
        else:
            self.switch_network = None

        # Track statistics for calibration
        self.register_buffer("num_fast", torch.tensor(0))
        self.register_buffer("num_slow", torch.tensor(0))
        self.register_buffer("running_confidence_mean", torch.tensor(0.5))
        self.register_buffer("running_uncertainty_mean", torch.tensor(0.3))

    def forward(
        self,
        fast_output: FastPolicyOutput,
        world_model_uncertainty: torch.Tensor,
        novelty: Optional[torch.Tensor] = None,
        force_slow: bool = False,
    ) -> ArbiterDecision:
        """
        Decide whether to use slow policy.

        Args:
            fast_output: Output from fast policy
            world_model_uncertainty: [B] uncertainty from world model
            novelty: [B] optional novelty score
            force_slow: if True, always use slow policy

        Returns:
            ArbiterDecision with use_slow tensor and metadata
        """
        B = fast_output.confidence.shape[0]
        device = fast_output.confidence.device

        confidence = fast_output.confidence
        uncertainty = world_model_uncertainty

        if novelty is None:
            novelty = torch.zeros(B, device=device)

        if force_slow:
            use_slow = torch.ones(B, dtype=torch.bool, device=device)
            reason = "forced_slow"
        elif self.switch_network is not None:
            # Learned switching
            features = torch.stack([
                confidence,
                uncertainty,
                novelty,
                fast_output.action_probs.max(dim=-1).values,
            ], dim=-1)
            logits = self.switch_network(features).squeeze(-1)
            use_slow = torch.sigmoid(logits) > 0.5
            reason = "learned"
        else:
            # Heuristic rules
            low_confidence = confidence < self.confidence_threshold
            high_uncertainty = uncertainty > self.uncertainty_threshold
            high_novelty = novelty > self.novelty_threshold

            use_slow = low_confidence | high_uncertainty | high_novelty

            # Build reason string
            reasons = []
            if low_confidence.any():
                reasons.append("low_confidence")
            if high_uncertainty.any():
                reasons.append("high_uncertainty")
            if high_novelty.any():
                reasons.append("high_novelty")
            reason = ",".join(reasons) if reasons else "fast_sufficient"

        # Update statistics
        self._update_stats(use_slow, confidence, uncertainty)

        return ArbiterDecision(
            use_slow=use_slow,
            confidence=confidence,
            uncertainty=uncertainty,
            novelty=novelty,
            reason=reason,
        )

    def should_use_slow(
        self,
        confidence: torch.Tensor,
        uncertainty: torch.Tensor,
        novelty: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Simple check if slow policy should be used.

        Args:
            confidence: [B] fast policy confidence
            uncertainty: [B] world model uncertainty
            novelty: [B] optional novelty score

        Returns:
            [B] bool tensor indicating slow policy usage
        """
        device = confidence.device
        B = confidence.shape[0]

        if novelty is None:
            novelty = torch.zeros(B, device=device)

        low_confidence = confidence < self.confidence_threshold
        high_uncertainty = uncertainty > self.uncertainty_threshold
        high_novelty = novelty > self.novelty_threshold

        return low_confidence | high_uncertainty | high_novelty

    def select_action(
        self,
        fast_output: FastPolicyOutput,
        slow_output: Optional[SlowPolicyOutput],
        use_slow: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select action based on arbiter decision.

        Args:
            fast_output: Output from fast policy
            slow_output: Output from slow policy (only used where use_slow=True)
            use_slow: [B] bool tensor from arbiter decision

        Returns:
            action_probs: [B, num_actions] selected action probabilities
            actions: [B] selected action indices
        """
        if slow_output is None:
            # Only fast policy available
            return fast_output.action_probs, fast_output.action_probs.argmax(dim=-1)

        # Use where to select between fast and slow
        action_probs = torch.where(
            use_slow.unsqueeze(-1),
            slow_output.action_probs,
            fast_output.action_probs,
        )

        actions = action_probs.argmax(dim=-1)

        return action_probs, actions

    def _update_stats(
        self,
        use_slow: torch.Tensor,
        confidence: torch.Tensor,
        uncertainty: torch.Tensor,
    ):
        """Update running statistics."""
        if self.training:
            return  # Don't update during training

        num_slow = use_slow.sum().item()
        num_fast = (~use_slow).sum().item()

        self.num_slow = self.num_slow + int(num_slow)
        self.num_fast = self.num_fast + int(num_fast)

        # Exponential moving average
        alpha = 0.01
        self.running_confidence_mean = (
            (1 - alpha) * self.running_confidence_mean
            + alpha * confidence.mean()
        )
        self.running_uncertainty_mean = (
            (1 - alpha) * self.running_uncertainty_mean
            + alpha * uncertainty.mean()
        )

    def get_stats(self) -> dict:
        """Get arbiter statistics."""
        total = self.num_fast.item() + self.num_slow.item()
        if total == 0:
            fast_ratio = 0.5
        else:
            fast_ratio = self.num_fast.item() / total

        return {
            "num_fast": self.num_fast.item(),
            "num_slow": self.num_slow.item(),
            "fast_ratio": fast_ratio,
            "running_confidence": self.running_confidence_mean.item(),
            "running_uncertainty": self.running_uncertainty_mean.item(),
        }

    def reset_stats(self):
        """Reset statistics."""
        self.num_fast.zero_()
        self.num_slow.zero_()
        self.running_confidence_mean.fill_(0.5)
        self.running_uncertainty_mean.fill_(0.3)

    def calibrate_thresholds(
        self,
        target_fast_ratio: float = 0.7,
        confidence_history: Optional[torch.Tensor] = None,
        uncertainty_history: Optional[torch.Tensor] = None,
    ):
        """
        Calibrate thresholds based on observed distributions.

        Args:
            target_fast_ratio: desired ratio of fast policy usage
            confidence_history: [N] historical confidence values
            uncertainty_history: [N] historical uncertainty values
        """
        if confidence_history is not None:
            # Set confidence threshold to achieve target_fast_ratio
            sorted_conf, _ = torch.sort(confidence_history)
            idx = int((1 - target_fast_ratio) * len(sorted_conf))
            self.confidence_threshold = sorted_conf[idx].item()

        if uncertainty_history is not None:
            # Set uncertainty threshold
            sorted_unc, _ = torch.sort(uncertainty_history, descending=True)
            idx = int((1 - target_fast_ratio) * len(sorted_unc))
            self.uncertainty_threshold = sorted_unc[idx].item()

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_arbiter(config: Optional[ARIALiteConfig] = None) -> Arbiter:
    """Factory function to create arbiter from full config."""
    if config is None:
        config = ARIALiteConfig()
    return Arbiter(config.arbiter)
