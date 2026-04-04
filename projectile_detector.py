# HSV-based projectile detection + frame-to-frame tracking for dodge decisions

from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class ProjectileDetector:
    """HSV-based projectile detection and tracking.

    Detects colored blobs in the game area and tracks them across frames
    to determine velocity and collision predictions.
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # Detection region (exclude HUD areas)
        self.roi_x1 = int(screen_width * 0.05)
        self.roi_y1 = int(screen_height * 0.05)
        self.roi_x2 = int(screen_width * 0.80)  # Exclude right-side buttons
        self.roi_y2 = int(screen_height * 0.85)  # Exclude bottom HUD

        # HSV ranges for common projectile colors
        # Brawl Stars projectiles are typically bright warm colors
        self.hsv_ranges = [
            # Yellow projectiles (most common): Shelly, Colt, Rico, etc.
            {"low": (15, 120, 150), "high": (35, 255, 255), "name": "yellow"},
            # Orange projectiles: Amber, some supers
            {"low": (8, 150, 150), "high": (18, 255, 255), "name": "orange"},
            # Red/pink projectiles: Edgar, some supers
            {"low": (0, 130, 150), "high": (8, 255, 255), "name": "red_low"},
            {"low": (165, 130, 150), "high": (180, 255, 255), "name": "red_high"},
            # Blue projectiles: Brock rockets, some abilities
            {"low": (100, 130, 150), "high": (125, 255, 255), "name": "blue"},
            # Bright white/cyan projectiles: some supers, electric attacks
            {"low": (80, 50, 200), "high": (100, 160, 255), "name": "cyan"},
            # Purple projectiles: Gene, Frank super indicator
            {"low": (125, 100, 150), "high": (155, 255, 255), "name": "purple"},
        ]

        # Blob size filters (in pixels at base resolution)
        self.min_blob_area = 80       # Minimum blob area (very small = noise)
        self.max_blob_area = 8000     # Maximum blob area (too big = terrain/HUD)
        self.min_circularity = 0.15   # Projectiles are somewhat round

        # Tracking across frames
        self._prev_blobs: List[Dict] = []  # Blobs from previous frame
        self._prev_frame_time: float = 0.0
        self._tracking_history: List[List[Dict]] = []  # Last N frames of blobs
        self._max_history = 4

        # Association parameters
        self._max_match_distance = 200  # px -- max distance to associate same projectile
        self._min_velocity = 50          # px/s -- below this, probably not a projectile

        # Player exclusion zone (don't detect player's own attacks near them)
        self._player_exclusion_radius = 80  # px around player to ignore

    def detect(self, frame: np.ndarray, player_pos: Tuple[float, float],
               enemy_positions: Optional[List[Tuple[float, float]]] = None
               ) -> List[Dict]:
        """Detect projectiles in the current frame."""

        # Compute delta-time from previous frame
        import time as _time
        now = _time.time()
        dt = now - self._prev_frame_time if self._prev_frame_time > 0 else 0.033
        self._prev_frame_time = now

        # Step 1: Detect colored blobs in ROI
        blobs = self._detect_blobs(frame, player_pos)

        # Step 2: Associate with previous frame blobs -> compute velocity
        tracked = self._track_blobs(blobs, dt)

        # Step 3: Predict threats to player
        projectiles = self._evaluate_threats(tracked, player_pos, dt)

        # Update history
        self._prev_blobs = blobs
        self._tracking_history.append(blobs)
        if len(self._tracking_history) > self._max_history:
            self._tracking_history.pop(0)

        return projectiles

    def _detect_blobs(self, frame: np.ndarray, player_pos: Tuple[float, float]) -> List[Dict]:
        """Detect colored blobs that could be projectiles."""
        h, w = frame.shape[:2]

        # Scale ROI to actual frame size
        scale_x = w / self.screen_width
        scale_y = h / self.screen_height
        x1 = int(self.roi_x1 * scale_x)
        y1 = int(self.roi_y1 * scale_y)
        x2 = int(self.roi_x2 * scale_x)
        y2 = int(self.roi_y2 * scale_y)

        roi = frame[y1:y2, x1:x2]
        
        # PERFORMANCE: Downsample to 50% for faster HSV processing
        # Projectiles are large enough to be detected at half res
        roi_small = cv2.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(roi_small, cv2.COLOR_RGB2HSV)

        # Combine all HSV masks
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        color_masks = {}

        for hr in self.hsv_ranges:
            mask = cv2.inRange(hsv, np.array(hr["low"]), np.array(hr["high"]))
            color_masks[hr["name"]] = mask
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        # Adjust player pos for downsampled image (0.5 scale)
        px_scaled = (player_pos[0] * scale_x - x1) * 0.5
        py_scaled = (player_pos[1] * scale_y - y1) * 0.5
        exclusion_r = self._player_exclusion_radius * min(scale_x, scale_y) * 0.5

        for contour in contours:
            area = cv2.contourArea(contour)
            # Scale area back: multiply by 4 for 0.5 downscale
            area_scaled = area * 4 / (scale_x * scale_y)  # Scale to base resolution

            if area_scaled < self.min_blob_area or area_scaled > self.max_blob_area:
                continue

            # Circularity check
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                continue

            # Get center and radius
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]

            # Skip if too close to player (our own attacks)
            dist_to_player = math.sqrt((cx - px_scaled) ** 2 + (cy - py_scaled) ** 2)
            if dist_to_player < exclusion_r:
                continue

            # Convert back to full-frame coordinates (account for 0.5 downsample)
            full_cx = (cx * 2 + x1) / scale_x
            full_cy = (cy * 2 + y1) / scale_y
            radius = math.sqrt(area_scaled / math.pi)

            # Determine color
            color = "unknown"
            max_pixels = 0
            for name, mask in color_masks.items():
                # Count pixels of this color within the contour
                contour_mask = np.zeros(mask.shape, dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], -1, 255, -1)
                pixel_count = cv2.countNonZero(cv2.bitwise_and(mask, contour_mask))
                if pixel_count > max_pixels:
                    max_pixels = pixel_count
                    color = name

            blobs.append({
                "pos": (full_cx, full_cy),
                "radius": radius,
                "area": area_scaled,
                "circularity": circularity,
                "color": color,
            })

        return blobs

    def _track_blobs(self, current_blobs: List[Dict], dt: float) -> List[Dict]:
        """Associate current blobs with previous frame blobs to compute velocity."""
        tracked = []

        if not self._prev_blobs or dt <= 0:
            # No history -- return blobs without velocity
            for blob in current_blobs:
                tracked.append({
                    **blob,
                    "velocity": (0.0, 0.0),
                    "speed": 0.0,
                    "tracked": False,
                })
            return tracked

        # Greedy nearest-neighbor matching
        used_prev = set()

        for blob in current_blobs:
            bx, by = blob["pos"]
            best_prev = None
            best_dist = self._max_match_distance

            for j, prev in enumerate(self._prev_blobs):
                if j in used_prev:
                    continue
                px, py = prev["pos"]
                dist = math.sqrt((bx - px) ** 2 + (by - py) ** 2)

                # Also check color match for better association
                color_bonus = 0 if blob["color"] == prev["color"] else 50
                effective_dist = dist + color_bonus

                if effective_dist < best_dist:
                    best_dist = effective_dist
                    best_prev = j

            if best_prev is not None:
                used_prev.add(best_prev)
                prev = self._prev_blobs[best_prev]
                vx = (bx - prev["pos"][0]) / dt
                vy = (by - prev["pos"][1]) / dt
                speed = math.sqrt(vx * vx + vy * vy)

                tracked.append({
                    **blob,
                    "velocity": (vx, vy),
                    "speed": speed,
                    "tracked": True,
                })
            else:
                tracked.append({
                    **blob,
                    "velocity": (0.0, 0.0),
                    "speed": 0.0,
                    "tracked": False,
                })

        return tracked

    def _evaluate_threats(self, tracked: List[Dict],
                           player_pos: Tuple[float, float],
                           dt: float) -> List[Dict]:
        """Evaluate which tracked blobs are threatening projectiles aimed at the player."""
        projectiles = []
        ppx, ppy = player_pos

        for blob in tracked:
            speed = blob["speed"]

            # Must be moving fast enough to be a projectile
            if speed < self._min_velocity:
                continue

            bx, by = blob["pos"]
            vx, vy = blob["velocity"]

            # Predict future position
            dist_to_player = math.sqrt((bx - ppx) ** 2 + (by - ppy) ** 2)

            # Check if projectile is heading toward player
            # Compute closest approach distance
            # Direction from blob to player
            to_player_x = ppx - bx
            to_player_y = ppy - by

            # Dot product: how much of velocity is toward player
            if speed > 0:
                dot = (vx * to_player_x + vy * to_player_y) / speed
            else:
                dot = 0

            # If dot < 0, projectile is moving away from player
            if dot < 0:
                continue

            # Perpendicular distance (closest approach)
            # Using cross product: |v × (p - b)| / |v|
            cross = abs(vx * to_player_y - vy * to_player_x) / speed
            closest_approach = cross

            # Player hitbox radius (approximate)
            player_radius = 40

            # Will it hit the player?
            if closest_approach > player_radius + blob["radius"] + 30:
                # Adding margin for safety
                continue

            # Time to impact
            time_to_impact = dist_to_player / speed if speed > 0 else 999
            frames_to_impact = max(1, int(time_to_impact / max(dt, 0.016)))

            # Threat level: closer + faster + more direct = more threatening
            directness = max(0, dot / dist_to_player) if dist_to_player > 0 else 1.0
            proximity_factor = max(0, 1 - dist_to_player / 600)  # 600px = max threat dist
            threat_level = min(1.0, directness * proximity_factor * (speed / 500))

            projectiles.append({
                "pos": (bx, by),
                "velocity": (vx, vy),
                "speed": speed,
                "radius": blob["radius"],
                "color": blob["color"],
                "threat_level": threat_level,
                "frames_to_impact": frames_to_impact,
                "closest_approach": closest_approach,
                "time_to_impact": time_to_impact,
            })

        # Sort by threat level (most dangerous first)
        projectiles.sort(key=lambda p: p["threat_level"], reverse=True)

        return projectiles

    def get_dodge_direction(self, projectile: Dict,
                             player_pos: Tuple[float, float]) -> Tuple[float, float]:
        """Calculate the optimal dodge direction for an incoming projectile."""
        vx, vy = projectile["velocity"]
        speed = projectile["speed"]

        if speed < 1:
            return (0.0, 0.0)

        # Perpendicular directions to projectile velocity
        perp1 = (-vy / speed, vx / speed)
        perp2 = (vy / speed, -vx / speed)

        # Choose the perpendicular that moves us further from the projectile's path
        bx, by = projectile["pos"]
        ppx, ppy = player_pos

        # Test both directions
        test_dist = 50
        pos1 = (ppx + perp1[0] * test_dist, ppy + perp1[1] * test_dist)
        pos2 = (ppx + perp2[0] * test_dist, ppy + perp2[1] * test_dist)

        # Distance from predicted projectile path for each option
        future_bx = bx + vx * 0.1  # 100ms into future
        future_by = by + vy * 0.1
        d1 = math.sqrt((pos1[0] - future_bx) ** 2 + (pos1[1] - future_by) ** 2)
        d2 = math.sqrt((pos2[0] - future_bx) ** 2 + (pos2[1] - future_by) ** 2)

        return perp1 if d1 > d2 else perp2

    def direction_to_keys(self, dx: float, dy: float) -> str:
        """Convert a direction vector to movement key string."""
        keys = ""
        threshold = 0.3

        if dy < -threshold:
            keys += "W"
        elif dy > threshold:
            keys += "S"
        if dx < -threshold:
            keys += "A"
        elif dx > threshold:
            keys += "D"

        return keys or "W"  # Default: move up if ambiguous

    def reset(self):
        """Reset tracking state for new match."""
        self._prev_blobs.clear()
        self._prev_frame_time = 0.0
        self._tracking_history.clear()
