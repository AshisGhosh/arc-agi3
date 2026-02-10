"""
Frame processor: region segmentation, frame hashing.

Takes a 64x64 frame with 16 colors and produces:
- Connected-component regions (for click-space reduction)
- Frame hash (for state deduplication)
- One-hot tensor (for CNN input)
"""

import hashlib
from dataclasses import dataclass

import numpy as np
import scipy.ndimage


@dataclass
class Region:
    """A connected component in the frame."""
    region_id: int
    color: int
    area: int
    bbox: tuple[int, int, int, int]  # (y_min, x_min, y_max, x_max)
    centroid_y: float
    centroid_x: float
    mask: np.ndarray  # [H, W] boolean mask of this region's pixels


class FrameProcessor:
    """Processes 64x64 game frames into regions and hashes."""

    def segment(self, frame: np.ndarray) -> list[Region]:
        """Segment frame into connected-component regions.

        Args:
            frame: [64, 64] array with values 0-15

        Returns:
            List of Region objects sorted by area descending
        """
        regions = []
        region_id = 0

        for color in range(16):
            binary_mask = (frame == color)
            if not binary_mask.any():
                continue

            labeled, num_features = scipy.ndimage.label(binary_mask)

            for i in range(1, num_features + 1):
                component_mask = (labeled == i)
                ys, xs = np.where(component_mask)

                if len(ys) == 0:
                    continue

                y_min, y_max = int(ys.min()), int(ys.max())
                x_min, x_max = int(xs.min()), int(xs.max())
                area = int(component_mask.sum())

                regions.append(Region(
                    region_id=region_id,
                    color=color,
                    area=area,
                    bbox=(y_min, x_min, y_max, x_max),
                    centroid_y=float(ys.mean()),
                    centroid_x=float(xs.mean()),
                    mask=component_mask,
                ))
                region_id += 1

        # Sort by area descending (larger regions first)
        regions.sort(key=lambda r: -r.area)
        return regions

    def hash_frame(self, frame: np.ndarray) -> str:
        """Compute collision-resistant hash of the frame.

        Args:
            frame: [64, 64] array with values 0-15

        Returns:
            32-character hex string (Blake2b 128-bit)
        """
        flat = frame.ravel()
        packed = (flat[0::2].astype(np.uint8) << 4) | (flat[1::2].astype(np.uint8) & 0x0F)
        return hashlib.blake2b(packed.tobytes(), digest_size=16).hexdigest()

    def get_click_point(self, region: Region) -> tuple[int, int]:
        """Get centroid click point for a region.

        Returns:
            (x, y) pixel coordinates for the game API
        """
        x = max(0, min(63, int(round(region.centroid_x))))
        y = max(0, min(63, int(round(region.centroid_y))))
        return x, y
