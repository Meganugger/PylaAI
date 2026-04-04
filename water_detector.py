"""
Water / Hazard Tile Detector -- HSV color-based detection.

The ONNX tile model does NOT have a "water" class, so water tiles are
invisible to the existing wall detection. This module fills that gap
by scanning for water-colored regions using HSV thresholds and returning
them as bounding boxes that can be treated as impassable walls.

Brawl Stars water tiles have a distinctive saturated blue/teal hue that
is very different from sky, UI elements, and ground textures.

Usage:
    detector = WaterDetector()
    water_bboxes = detector.detect(frame, player_bbox=None)
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional


class WaterDetector:
    """Detect water tiles in Brawl Stars frames using HSV color analysis."""

    # HSV ranges for water in Brawl Stars (multiple variants across maps):
    # - Standard water: deep blue/teal, H=95-125, high saturation
    # - Shallow water: lighter blue, H=90-120, medium saturation
    # - Dark water/lava: can vary, but we focus on blue water
    _WATER_HSV_RANGES = [
        # (h_lo, s_lo, v_lo, h_hi, s_hi, v_hi)
        (90, 80, 50, 130, 255, 220),    # Deep blue/teal water
        (85, 60, 40, 125, 255, 180),    # Darker water variants
    ]

    # Minimum area (in px²) for a water region to be valid
    _MIN_AREA = 800
    # Minimum contour width/height
    _MIN_DIM = 20

    # Morphological kernels
    _KERN_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    _KERN_OPEN = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    # Exclusion zones (relative to frame size) -- avoid detecting blue UI elements
    # Format: (x_frac_min, y_frac_min, x_frac_max, y_frac_max)
    _UI_EXCLUSION_ZONES = [
        (0.0, 0.0, 1.0, 0.06),     # Top HUD bar
        (0.0, 0.90, 1.0, 1.0),     # Bottom controls area
        (0.0, 0.0, 0.05, 1.0),     # Left edge
        (0.95, 0.0, 1.0, 1.0),     # Right edge
        (0.70, 0.70, 1.0, 1.0),    # Joystick/attack area (bottom-right)
    ]

    def __init__(self):
        self._last_water_bboxes: List[List[int]] = []
        self._detection_count = 0

    def detect(self, frame, player_bbox: Optional[List[int]] = None) -> List[List[int]]:
        """Detect water regions in the frame."""
        try:
            arr = np.array(frame) if not isinstance(frame, np.ndarray) else frame
            fh, fw = arr.shape[:2]

            # Convert to HSV (OpenCV expects BGR, PIL gives RGB)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # Build combined water mask from all HSV ranges
            water_mask = np.zeros((fh, fw), dtype=np.uint8)
            for h_lo, s_lo, v_lo, h_hi, s_hi, v_hi in self._WATER_HSV_RANGES:
                mask = cv2.inRange(hsv,
                                   np.array([h_lo, s_lo, v_lo]),
                                   np.array([h_hi, s_hi, v_hi]))
                water_mask = cv2.bitwise_or(water_mask, mask)

            # Exclude UI zones (blue buttons, HUD elements, etc.)
            for xf_min, yf_min, xf_max, yf_max in self._UI_EXCLUSION_ZONES:
                x1 = int(xf_min * fw)
                y1 = int(yf_min * fh)
                x2 = int(xf_max * fw)
                y2 = int(yf_max * fh)
                water_mask[y1:y2, x1:x2] = 0

            # Exclude player bbox area (player effects can be blue)
            if player_bbox:
                px1, py1, px2, py2 = [int(v) for v in player_bbox[:4]]
                # Expand by 30px margin
                px1 = max(0, px1 - 30)
                py1 = max(0, py1 - 30)
                px2 = min(fw, px2 + 30)
                py2 = min(fh, py2 + 30)
                water_mask[py1:py2, px1:px2] = 0

            # Morphological cleanup -- close gaps, then remove noise
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, self._KERN_CLOSE)
            water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, self._KERN_OPEN)

            # Find contours and extract bounding boxes
            contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            water_bboxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self._MIN_AREA:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if w < self._MIN_DIM or h < self._MIN_DIM:
                    continue
                # Additional validation: check density of water pixels in bbox
                roi = water_mask[y:y+h, x:x+w]
                density = np.count_nonzero(roi) / (w * h) if (w * h) > 0 else 0
                if density < 0.25:
                    continue  # Too sparse -- probably not a solid water tile
                water_bboxes.append([x, y, x + w, y + h])

            self._last_water_bboxes = water_bboxes
            self._detection_count += 1

            # Debug: print first time and every 50 detections
            if self._detection_count == 1 or (water_bboxes and self._detection_count % 50 == 0):
                print(f"[WATER-DETECT] Found {len(water_bboxes)} water regions "
                      f"(frame {fw}x{fh})")

            return water_bboxes

        except Exception as e:
            return self._last_water_bboxes
