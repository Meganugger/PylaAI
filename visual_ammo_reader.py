# reads ammo count from screen by checking the yellow bars under the attack button

from __future__ import annotations

import numpy as np
import cv2
from typing import Tuple, Optional


class AmmoReader:
    """Read ammo count from the game UI by scanning ammo indicators."""

    # base coordinates (1920×1080 reference)
    # These define the three ammo bar sampling regions below the attack button.
    # Bars are ordered left to right: ammo_1, ammo_2, ammo_3
    # Each entry: (x_center, y_center, half_width, half_height)
    AMMO_BARS_BASE = [
        (1660, 870, 22, 10),   # Left ammo bar (ammo 1)
        (1725, 870, 22, 10),   # Center ammo bar (ammo 2)
        (1790, 870, 22, 10),   # Right ammo bar (ammo 3)
    ]

    # HSV range for "full" ammo (bright yellow-orange glow)
    HSV_AMMO_FULL_LO = np.array([15, 100, 150])
    HSV_AMMO_FULL_HI = np.array([40, 255, 255])

    # Alternative: some brawlers have white/bright ammo bars
    HSV_AMMO_WHITE_LO = np.array([0, 0, 200])
    HSV_AMMO_WHITE_HI = np.array([180, 40, 255])

    # Minimum percentage of bright pixels within a bar region to count as "full"
    FILL_THRESHOLD = 0.20  # 20% of the crop must be bright yellow/white

    def __init__(self, window_controller=None):
        self._width_ratio = 1.0
        self._height_ratio = 1.0
        if window_controller is not None:
            self._width_ratio = getattr(window_controller, 'width_ratio', 1.0)
            self._height_ratio = getattr(window_controller, 'height_ratio', 1.0)

        # Pre-compute scaled bar regions
        self._bar_regions = self._compute_bar_regions()

        # Confidence tracking (rolling accuracy)
        self._last_readings = []  # last N readings for temporal smoothing
        self._smoothing_window = 5

    def _compute_bar_regions(self):
        """Compute pixel regions for each ammo bar, scaled to actual resolution."""
        regions = []
        wr = self._width_ratio
        hr = self._height_ratio
        for (bx, by, hw, hh) in self.AMMO_BARS_BASE:
            x1 = int((bx - hw) * wr)
            y1 = int((by - hh) * hr)
            x2 = int((bx + hw) * wr)
            y2 = int((by + hh) * hr)
            regions.append((x1, y1, x2, y2))
        return regions

    def update_ratios(self, width_ratio: float, height_ratio: float):
        """Update scaling ratios if resolution changed."""
        self._width_ratio = width_ratio
        self._height_ratio = height_ratio
        self._bar_regions = self._compute_bar_regions()

    def read_ammo(self, frame) -> int:
        """Read ammo count from the current frame."""
        try:
            arr = np.asarray(frame)
            fh, fw = arr.shape[:2]
            ammo_count = 0

            for (x1, y1, x2, y2) in self._bar_regions:
                # Bounds check
                if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
                    continue
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = arr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                total_pixels = crop_hsv.shape[0] * crop_hsv.shape[1]
                if total_pixels == 0:
                    continue

                # Check yellow/orange ammo glow
                mask_yellow = cv2.inRange(crop_hsv, self.HSV_AMMO_FULL_LO,
                                           self.HSV_AMMO_FULL_HI)
                yellow_ratio = np.count_nonzero(mask_yellow) / total_pixels

                # Check white/bright variant
                mask_white = cv2.inRange(crop_hsv, self.HSV_AMMO_WHITE_LO,
                                          self.HSV_AMMO_WHITE_HI)
                white_ratio = np.count_nonzero(mask_white) / total_pixels

                # Bar is "full" if either color is above threshold
                if yellow_ratio >= self.FILL_THRESHOLD or white_ratio >= self.FILL_THRESHOLD:
                    ammo_count += 1

            # Temporal smoothing: majority vote over last N frames
            self._last_readings.append(ammo_count)
            if len(self._last_readings) > self._smoothing_window:
                self._last_readings.pop(0)

            if len(self._last_readings) >= 3:
                # Majority vote
                from collections import Counter
                counts = Counter(self._last_readings)
                smoothed = counts.most_common(1)[0][0]
                return smoothed

            return ammo_count

        except Exception:
            return -1

    def read_ammo_detailed(self, frame) -> dict:
        """Read ammo with per-bar confidence -- for debug overlay."""
        result = {"ammo_count": 0, "bars": []}
        try:
            arr = np.array(frame) if not isinstance(frame, np.ndarray) else frame
            for (x1, y1, x2, y2) in self._bar_regions:
                bar_info = {
                    "filled": False,
                    "yellow_pct": 0.0,
                    "white_pct": 0.0,
                    "region": (x1, y1, x2, y2),
                }

                fh, fw = arr.shape[:2]
                if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh or x2 <= x1 or y2 <= y1:
                    result["bars"].append(bar_info)
                    continue

                crop = arr[y1:y2, x1:x2]
                if crop.size == 0:
                    result["bars"].append(bar_info)
                    continue

                crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                total = crop_hsv.shape[0] * crop_hsv.shape[1]
                if total == 0:
                    result["bars"].append(bar_info)
                    continue

                mask_y = cv2.inRange(crop_hsv, self.HSV_AMMO_FULL_LO,
                                      self.HSV_AMMO_FULL_HI)
                mask_w = cv2.inRange(crop_hsv, self.HSV_AMMO_WHITE_LO,
                                      self.HSV_AMMO_WHITE_HI)

                bar_info["yellow_pct"] = np.count_nonzero(mask_y) / total
                bar_info["white_pct"] = np.count_nonzero(mask_w) / total
                bar_info["filled"] = (bar_info["yellow_pct"] >= self.FILL_THRESHOLD or
                                       bar_info["white_pct"] >= self.FILL_THRESHOLD)
                if bar_info["filled"]:
                    result["ammo_count"] += 1

                result["bars"].append(bar_info)

        except Exception:
            pass

        return result

    def reset(self):
        """Reset smoothing state (call on new match)."""
        self._last_readings.clear()
