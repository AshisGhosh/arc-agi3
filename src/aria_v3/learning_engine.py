"""
Layer 2: Learning Engine.

Observes (frame, action, next_frame) transitions and produces structured
game reports. Does NOT interpret what it finds — just records facts.

Reports include:
- Per-action statistics (change rate, consistent pixel shifts)
- Region tracking (which colored regions moved/appeared/disappeared)
- Notable events (counter changes, object disappearances)
- State coverage (unique frames, dead ends, game overs)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.ndimage


@dataclass
class RegionSnapshot:
    """A connected-component region at a moment in time."""
    color: int
    area: int
    centroid_y: float
    centroid_x: float
    bbox: tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max)


@dataclass
class TransitionRecord:
    """What happened when an action was taken."""
    action: int
    frame_changed: bool
    changed_pixels: int
    shift_vector: tuple[int, int] | None  # (dy, dx) of the moved cluster
    regions_before: list[RegionSnapshot]
    regions_after: list[RegionSnapshot]
    appeared: list[RegionSnapshot]  # regions in after but not before
    disappeared: list[RegionSnapshot]  # regions in before but not after
    moved: list[tuple[RegionSnapshot, RegionSnapshot, int, int]]  # (before, after, dy, dx)
    color_changed: list[tuple[RegionSnapshot, RegionSnapshot]]  # same pos, different color


@dataclass
class ActionStats:
    """Accumulated statistics for a single action type."""
    count: int = 0
    frame_changed_count: int = 0
    pixel_diff_counts: list[int] = field(default_factory=list)
    shift_vectors: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class NotableEvent:
    """Something interesting that happened during play."""
    step: int
    event_type: str  # "counter_change", "region_disappeared", "region_appeared", "large_reset"
    location: tuple[float, float] | None  # (y, x) centroid
    color: int | None
    color_from: int | None  # for counter_change
    color_to: int | None  # for counter_change
    trigger_action: int
    details: str


class LearningEngine:
    """Layer 2: Observes transitions and builds structured game reports."""

    def __init__(self):
        self.action_stats: dict[int, ActionStats] = defaultdict(ActionStats)
        self.notable_events: list[NotableEvent] = []
        self.step_count: int = 0
        self.game_overs: int = 0
        self.level_completions: int = 0
        self.unique_frame_hashes: set[str] = set()
        self.recent_transitions: list[TransitionRecord] = []
        self.available_actions: list[int] = []

        # For tracking regions across the game (not just per-transition)
        self._prev_frame: np.ndarray | None = None
        self._prev_regions: list[RegionSnapshot] | None = None

    def setup(self, available_actions: list[int]) -> None:
        self.available_actions = available_actions

    def record(
        self,
        frame: np.ndarray,
        action: int,
        next_frame: np.ndarray,
        frame_hash: str,
        next_frame_hash: str,
        game_over: bool = False,
        level_complete: bool = False,
    ) -> None:
        """Record a single (frame, action, next_frame) transition."""
        self.step_count += 1
        self.unique_frame_hashes.add(frame_hash)
        self.unique_frame_hashes.add(next_frame_hash)

        if game_over:
            self.game_overs += 1
        if level_complete:
            self.level_completions += 1

        # Compute transition details
        transition = self._analyze_transition(frame, action, next_frame)

        # Update per-action stats
        stats = self.action_stats[action]
        stats.count += 1
        if transition.frame_changed:
            stats.frame_changed_count += 1
        stats.pixel_diff_counts.append(transition.changed_pixels)
        if transition.shift_vector is not None:
            stats.shift_vectors.append(transition.shift_vector)

        # Detect notable events
        self._detect_events(transition, game_over, level_complete)

        # Keep recent transitions (capped)
        self.recent_transitions.append(transition)
        if len(self.recent_transitions) > 500:
            self.recent_transitions = self.recent_transitions[-500:]

    def _analyze_transition(
        self, frame: np.ndarray, action: int, next_frame: np.ndarray
    ) -> TransitionRecord:
        """Compute detailed diff between two frames."""
        frame_changed = not np.array_equal(frame, next_frame)
        diff_mask = frame != next_frame
        changed_pixels = int(diff_mask.sum())

        # Compute shift vector from the changed pixel cluster
        shift_vector = None
        if changed_pixels > 0 and changed_pixels < 2000:
            shift_vector = self._compute_shift(frame, next_frame, diff_mask)

        # Region analysis
        regions_before = self._extract_regions_fast(frame)
        regions_after = self._extract_regions_fast(next_frame)

        appeared, disappeared, moved, color_changed = self._match_regions(
            regions_before, regions_after
        )

        return TransitionRecord(
            action=action,
            frame_changed=frame_changed,
            changed_pixels=changed_pixels,
            shift_vector=shift_vector,
            regions_before=regions_before,
            regions_after=regions_after,
            appeared=appeared,
            disappeared=disappeared,
            moved=moved,
            color_changed=color_changed,
        )

    def _compute_shift(
        self, frame: np.ndarray, next_frame: np.ndarray, diff_mask: np.ndarray
    ) -> tuple[int, int] | None:
        """Compute displacement vector of the changed pixel cluster.

        Strategy: find the largest single-color region that moved, and compute
        its centroid displacement.
        """
        # Find pixels that changed FROM a specific color
        ys, xs = np.where(diff_mask)
        if len(ys) == 0:
            return None

        # Group changed pixels by their old color
        old_colors = frame[ys, xs]
        new_colors = next_frame[ys, xs]

        # For each old color, find where those pixels went
        best_shift = None
        best_count = 0

        for color in np.unique(old_colors):
            # Pixels that WERE this color and changed
            was_mask = (frame == color) & diff_mask
            was_ys, was_xs = np.where(was_mask)
            if len(was_ys) < 2:
                continue

            # Pixels that ARE this color now and changed
            now_mask = (next_frame == color) & diff_mask
            now_ys, now_xs = np.where(now_mask)
            if len(now_ys) < 2:
                continue

            # Centroid displacement
            dy = int(round(now_ys.mean() - was_ys.mean()))
            dx = int(round(now_xs.mean() - was_xs.mean()))

            if (dy != 0 or dx != 0) and len(was_ys) > best_count:
                best_shift = (dy, dx)
                best_count = len(was_ys)

        return best_shift

    def _extract_regions_fast(self, frame: np.ndarray) -> list[RegionSnapshot]:
        """Extract connected-component regions (lightweight version)."""
        regions = []
        for color in range(16):
            binary = (frame == color)
            if not binary.any():
                continue

            labeled, n = scipy.ndimage.label(binary)
            for i in range(1, n + 1):
                ys, xs = np.where(labeled == i)
                if len(ys) < 2:  # skip single pixels
                    continue

                regions.append(RegionSnapshot(
                    color=color,
                    area=len(ys),
                    centroid_y=float(ys.mean()),
                    centroid_x=float(xs.mean()),
                    bbox=(int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())),
                ))
        return regions

    def _match_regions(
        self,
        before: list[RegionSnapshot],
        after: list[RegionSnapshot],
    ) -> tuple[
        list[RegionSnapshot],  # appeared
        list[RegionSnapshot],  # disappeared
        list[tuple[RegionSnapshot, RegionSnapshot, int, int]],  # moved (before, after, dy, dx)
        list[tuple[RegionSnapshot, RegionSnapshot]],  # color_changed (before, after)
    ]:
        """Match regions across frames by color and spatial proximity."""
        appeared = []
        disappeared = []
        moved = []
        color_changed = []

        matched_after = set()
        proximity_threshold = 16.0  # pixels

        for rb in before:
            best_match = None
            best_dist = float("inf")

            # First try same-color match (movement detection)
            for j, ra in enumerate(after):
                if j in matched_after:
                    continue
                if ra.color != rb.color:
                    continue
                dist = ((ra.centroid_y - rb.centroid_y) ** 2 +
                        (ra.centroid_x - rb.centroid_x) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_match = j

            if best_match is not None and best_dist < proximity_threshold:
                ra = after[best_match]
                matched_after.add(best_match)
                dy = int(round(ra.centroid_y - rb.centroid_y))
                dx = int(round(ra.centroid_x - rb.centroid_x))
                if dy != 0 or dx != 0:
                    moved.append((rb, ra, dy, dx))
                continue

            # Try position match with different color (color change detection)
            for j, ra in enumerate(after):
                if j in matched_after:
                    continue
                dist = ((ra.centroid_y - rb.centroid_y) ** 2 +
                        (ra.centroid_x - rb.centroid_x) ** 2) ** 0.5
                if dist < 4.0 and abs(ra.area - rb.area) < max(ra.area, rb.area) * 0.3:
                    matched_after.add(j)
                    if ra.color != rb.color:
                        color_changed.append((rb, ra))
                    break
            else:
                disappeared.append(rb)

        # Unmatched after regions = appeared
        for j, ra in enumerate(after):
            if j not in matched_after:
                appeared.append(ra)

        return appeared, disappeared, moved, color_changed

    def _detect_events(
        self, transition: TransitionRecord, game_over: bool, level_complete: bool
    ) -> None:
        """Detect notable events from a transition."""
        # Large reset (game over or level change)
        if transition.changed_pixels > 2000:
            self.notable_events.append(NotableEvent(
                step=self.step_count,
                event_type="large_reset",
                location=None,
                color=None,
                color_from=None,
                color_to=None,
                trigger_action=transition.action,
                details=f"{'game_over' if game_over else 'level_complete' if level_complete else 'unknown'}, "
                        f"{transition.changed_pixels} pixels changed",
            ))
            return

        # Region disappeared
        for r in transition.disappeared:
            if r.area >= 4:  # ignore tiny regions
                self.notable_events.append(NotableEvent(
                    step=self.step_count,
                    event_type="region_disappeared",
                    location=(r.centroid_y, r.centroid_x),
                    color=r.color,
                    color_from=None,
                    color_to=None,
                    trigger_action=transition.action,
                    details=f"color={r.color}, area={r.area}, pos=({r.centroid_y:.0f},{r.centroid_x:.0f})",
                ))

        # Region appeared
        for r in transition.appeared:
            if r.area >= 4:
                self.notable_events.append(NotableEvent(
                    step=self.step_count,
                    event_type="region_appeared",
                    location=(r.centroid_y, r.centroid_x),
                    color=r.color,
                    color_from=None,
                    color_to=None,
                    trigger_action=transition.action,
                    details=f"color={r.color}, area={r.area}, pos=({r.centroid_y:.0f},{r.centroid_x:.0f})",
                ))

        # Color changed (counter-like)
        for rb, ra in transition.color_changed:
            if rb.area < 200:  # small regions changing color = likely counter/indicator
                self.notable_events.append(NotableEvent(
                    step=self.step_count,
                    event_type="counter_change",
                    location=(rb.centroid_y, rb.centroid_x),
                    color=None,
                    color_from=rb.color,
                    color_to=ra.color,
                    trigger_action=transition.action,
                    details=f"pos=({rb.centroid_y:.0f},{rb.centroid_x:.0f}), "
                            f"color {rb.color}→{ra.color}, area={rb.area}",
                ))

    def generate_report(self) -> dict:
        """Generate structured game report for Layer 3.

        Returns a dict with all accumulated evidence.
        """
        action_report = {}
        for action_id, stats in sorted(self.action_stats.items()):
            change_rate = stats.frame_changed_count / max(stats.count, 1)
            mean_pixels = (
                float(np.mean(stats.pixel_diff_counts)) if stats.pixel_diff_counts else 0.0
            )

            # Compute consistent shift
            consistent_shift = None
            shift_confidence = 0.0
            if len(stats.shift_vectors) >= 3:
                shifts = np.array(stats.shift_vectors)
                # Find most common shift
                unique_shifts, counts = np.unique(shifts, axis=0, return_counts=True)
                best_idx = counts.argmax()
                best_count = counts[best_idx]
                shift_confidence = best_count / len(stats.shift_vectors)
                if shift_confidence >= 0.5:
                    consistent_shift = tuple(int(x) for x in unique_shifts[best_idx])

            # Find most commonly affected regions (by color)
            moved_colors: dict[int, int] = defaultdict(int)
            for t in self.recent_transitions:
                if t.action == action_id:
                    for _, _, _, _ in t.moved:
                        pass  # counted via shift_vectors
                    for rb, ra, dy, dx in t.moved:
                        moved_colors[rb.color] += 1

            affected_regions = [
                {"color": color, "move_count": count}
                for color, count in sorted(moved_colors.items(), key=lambda x: -x[1])[:3]
            ]

            action_report[action_id] = {
                "count": stats.count,
                "frame_changed": stats.frame_changed_count,
                "change_rate": round(change_rate, 3),
                "mean_changed_pixels": round(mean_pixels, 1),
                "consistent_shift": consistent_shift,
                "shift_confidence": round(shift_confidence, 3),
                "affected_regions": affected_regions,
            }

        # Deduplicate notable events (keep most recent per type+location)
        deduped_events = []
        seen = set()
        for event in reversed(self.notable_events[-50:]):
            key = (event.event_type, event.color, event.color_from, event.color_to)
            if event.location:
                key = key + (round(event.location[0], -1), round(event.location[1], -1))
            if key not in seen:
                seen.add(key)
                deduped_events.append(event)
        deduped_events.reverse()

        events_report = [
            {
                "step": e.step,
                "type": e.event_type,
                "location": e.location,
                "color": e.color,
                "color_from": e.color_from,
                "color_to": e.color_to,
                "trigger_action": e.trigger_action,
                "details": e.details,
            }
            for e in deduped_events[:20]  # cap at 20 events
        ]

        return {
            "actions_taken": self.step_count,
            "available_actions": self.available_actions,
            "per_action": action_report,
            "notable_events": events_report,
            "coverage": {
                "unique_frames": len(self.unique_frame_hashes),
                "game_overs": self.game_overs,
                "level_completions": self.level_completions,
            },
        }

    def report_to_text(self, report: dict | None = None) -> str:
        """Convert a game report to natural language text for the LLM."""
        if report is None:
            report = self.generate_report()

        lines = [f"GAME OBSERVATION REPORT (actions 0-{report['actions_taken']}):", ""]

        # Available actions
        actions = report["available_actions"]
        has_click = 6 in actions
        simple = [a for a in actions if 1 <= a <= 5]
        if simple and has_click:
            lines.append(f"Available actions: {actions} (simple: {simple}, click: action 6)")
        elif simple:
            lines.append(f"Available actions: {simple} (directional/simple, no click)")
        elif has_click:
            lines.append("Available actions: [6] (click only)")
        lines.append("")

        # Per-action stats
        lines.append("Per-action statistics:")
        for action_id, stats in sorted(report["per_action"].items()):
            cr = stats["change_rate"]
            lines.append(
                f"- Action {action_id}: used {stats['count']} times, "
                f"frame changed {stats['frame_changed']} times ({cr:.0%})."
            )
            if stats["consistent_shift"]:
                dy, dx = stats["consistent_shift"]
                conf = stats["shift_confidence"]
                lines.append(
                    f"  Consistent pixel shift: ({dx}, {dy}) with {conf:.0%} confidence."
                )
            if stats["affected_regions"]:
                regions_str = ", ".join(
                    f"color {r['color']} (moved {r['move_count']}x)"
                    for r in stats["affected_regions"]
                )
                lines.append(f"  Affected regions: {regions_str}")
            if cr < 0.5:
                lines.append(f"  Low effectiveness: {1-cr:.0%} of attempts had no effect.")
        lines.append("")

        # Notable events
        if report["notable_events"]:
            lines.append("Notable events:")
            for event in report["notable_events"]:
                loc = ""
                if event["location"]:
                    y, x = event["location"]
                    loc = f" at ({x:.0f}, {y:.0f})"

                if event["type"] == "counter_change":
                    lines.append(
                        f"- Step {event['step']}: Counter{loc} changed color "
                        f"from {event['color_from']} to {event['color_to']}. "
                        f"Triggered by action {event['trigger_action']}."
                    )
                elif event["type"] == "region_disappeared":
                    lines.append(
                        f"- Step {event['step']}: Region{loc} of color {event['color']} "
                        f"disappeared. Triggered by action {event['trigger_action']}."
                    )
                elif event["type"] == "region_appeared":
                    lines.append(
                        f"- Step {event['step']}: Region{loc} of color {event['color']} "
                        f"appeared. Triggered by action {event['trigger_action']}."
                    )
                elif event["type"] == "large_reset":
                    lines.append(
                        f"- Step {event['step']}: Large frame reset. {event['details']}."
                    )
            lines.append("")

        # Coverage
        cov = report["coverage"]
        lines.append(
            f"State coverage: {cov['unique_frames']} unique frames. "
            f"Game overs: {cov['game_overs']}. "
            f"Level completions: {cov['level_completions']}."
        )

        return "\n".join(lines)

    def on_level_complete(self) -> None:
        """Reset per-level state. Keep notable events and overall stats."""
        # Reset per-action counters (new level = new layout)
        for stats in self.action_stats.values():
            stats.count = 0
            stats.frame_changed_count = 0
            stats.pixel_diff_counts.clear()
            stats.shift_vectors.clear()
        self.recent_transitions.clear()
        self.unique_frame_hashes.clear()
