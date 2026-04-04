# persistent occupancy grid for the match, danger heatmap, LoS raycasting

from __future__ import annotations

import math
import time
from typing import List, Tuple, Optional, Dict

import numpy as np


# cell Types
UNKNOWN   = 0
EMPTY     = 1
WALL      = 2
BUSH      = 3
DESTROYED = 4
WATER     = 5
GAS       = 6

CELL_NAMES = {
    UNKNOWN: "unknown",
    EMPTY: "empty",
    WALL: "wall",
    BUSH: "bush",
    DESTROYED: "destroyed",
    WATER: "water",
    GAS: "gas",
}


class CellType:
    """Convenient enum-like accessor for cell type constants."""
    UNKNOWN   = UNKNOWN
    EMPTY     = EMPTY
    WALL      = WALL
    BUSH      = BUSH
    DESTROYED = DESTROYED
    WATER     = WATER
    GAS       = GAS


# spatial Memory
class SpatialMemory:
    """Occupancy grid for the game map.

    Converts pixel coordinates to grid cells and maintains a persistent
    map representation that is updated each frame with new detections.

    """

    def __init__(self, width: int = 1920, height: int = 1080, cell_size: int = 40):
        self.screen_width = width
        self.screen_height = height
        self.cell_size = cell_size
        self.cols = width // cell_size
        self.rows = height // cell_size

        # Main occupancy grid (rows × cols)
        self.grid = np.full((self.rows, self.cols), UNKNOWN, dtype=np.uint8)

        # Danger heatmap -- accumulated danger score per cell (float)
        self.danger_map = np.zeros((self.rows, self.cols), dtype=np.float32)

        # Damage history -- where player took damage (decays over time)
        self._damage_events: List[Tuple[int, int, float, float]] = []  # (row, col, amount, timestamp)

        # Fog-of-war: cells we've observed this match
        self.observed = np.zeros((self.rows, self.cols), dtype=bool)

        # Time tracking
        self._last_update_time = 0.0
        self._match_start_time = time.time()

        # Gas zone tracking
        self._gas_center = (self.cols // 2, self.rows // 2)  # Grid coords
        self._gas_radius_cells = max(self.cols, self.rows)    # Starts huge (no gas)
        self._gas_shrink_rate = 0.0                            # Cells per second

        # Ghost walls -- temporary walls injected by stuck detection
        # Maps (row, col) -> expire_time. Cleared back to EMPTY when expired.
        self._ghost_walls: Dict[Tuple[int, int], float] = {}

    # property Aliases
    @property
    def grid_w(self) -> int:
        """Alias for cols (grid width in cells)."""
        return self.cols

    @property
    def grid_h(self) -> int:
        """Alias for rows (grid height in cells)."""
        return self.rows

    # convenience Methods
    def update_from_detections(self, walls: List[List[int]], bushes: List[List[int]],
                               player_pos: Tuple[float, float] = (960, 540), **kwargs):
        """Convenience wrapper for update(). Accepts keyword-only args."""
        self.update(walls, bushes, player_pos, **kwargs)

    def update_visibility(self, player_pos: Tuple[float, float], view_radius_cells: int = 8):
        """Public wrapper for _update_visibility()."""
        self._update_visibility(player_pos, view_radius_cells)

    def add_danger(self, pos: Tuple[float, float], radius: int = 3, intensity: float = 1.0):
        """Add a danger value at a pixel position in a radius of grid cells."""
        row, col = self.pixel_to_grid(pos[0], pos[1])
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    dist = math.sqrt(dr * dr + dc * dc)
                    if dist <= radius:
                        falloff = max(0.0, 1.0 - dist / (radius + 0.01))
                        self.danger_map[r, c] += intensity * falloff

    def inject_ghost_wall(self, px: float, py: float, duration: float = 5.0):
        """Inject a temporary ghost wall at pixel position.
        
        Ghost walls automatically expire after `duration` seconds and are
        reverted to EMPTY, preventing permanent grid pollution.
        """
        row, col = self.pixel_to_grid(px, py)
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row, col] = WALL
            self._ghost_walls[(row, col)] = time.time() + duration
            print(f"[GHOST] Injected ghost wall at cell ({row},{col}) expires in {duration:.1f}s")

    def clear_expired_ghosts(self):
        """Remove ghost walls whose expiry time has passed."""
        if not self._ghost_walls:
            return
        now = time.time()
        expired = [k for k, exp in self._ghost_walls.items() if now >= exp]
        for row, col in expired:
            # Only revert if still marked as WALL (might have been overwritten by ONNX detection)
            if self.grid[row, col] == WALL:
                self.grid[row, col] = EMPTY
            del self._ghost_walls[(row, col)]
        if expired:
            print(f"[GHOST] Cleared {len(expired)} expired ghost walls")

    # coordinate Conversion
    def pixel_to_grid(self, px: float, py: float) -> Tuple[int, int]:
        """Convert pixel coordinates to grid (row, col)."""
        col = max(0, min(int(px / self.cell_size), self.cols - 1))
        row = max(0, min(int(py / self.cell_size), self.rows - 1))
        return row, col

    def grid_to_pixel(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid (row, col) to pixel center coordinates."""
        px = (col + 0.5) * self.cell_size
        py = (row + 0.5) * self.cell_size
        return px, py

    def bbox_to_grid_cells(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Get all grid cells covered by a bounding box."""
        r1, c1 = self.pixel_to_grid(x1, y1)
        r2, c2 = self.pixel_to_grid(x2, y2)
        cells = []
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                cells.append((r, c))
        return cells

    # grid Updates
    def update(self, walls: List[List[int]], bushes: List[List[int]],
               player_pos: Tuple[float, float],
               destroyed_zones: Optional[List[List[int]]] = None,
               gas_active: bool = False,
               storm_center: Optional[Tuple[float, float]] = None,
               storm_radius: Optional[float] = None):
        """Update the grid with current frame detections.

        """
        now = time.time()
        self._last_update_time = now

        # Expire temporary ghost walls before applying new detections
        self.clear_expired_ghosts()

        # Mark player's visible area as observed + empty (if not wall/bush)
        self._update_visibility(player_pos)

        # Update walls
        for bbox in walls:
            x1, y1, x2, y2 = bbox[:4]
            for r, c in self.bbox_to_grid_cells(x1, y1, x2, y2):
                self.grid[r, c] = WALL
                self.observed[r, c] = True

        # Update bushes
        for bbox in bushes:
            x1, y1, x2, y2 = bbox[:4]
            for r, c in self.bbox_to_grid_cells(x1, y1, x2, y2):
                if self.grid[r, c] != WALL:  # Walls take priority
                    self.grid[r, c] = BUSH
                    self.observed[r, c] = True

        # Update destroyed wall zones
        if destroyed_zones:
            for zone in destroyed_zones:
                x1, y1, x2, y2 = zone[:4]
                for r, c in self.bbox_to_grid_cells(x1, y1, x2, y2):
                    if self.grid[r, c] == WALL:
                        self.grid[r, c] = DESTROYED
                    self.observed[r, c] = True

        # Update gas/storm zone
        if gas_active and storm_center and storm_radius:
            self._update_gas_zone(storm_center, storm_radius)

        # Decay danger map (half-life of 10 seconds)
        dt = now - self._last_update_time if self._last_update_time > 0 else 0.033
        if dt > 0:
            decay = 0.5 ** (dt / 10.0)
            self.danger_map *= decay

    def _update_visibility(self, player_pos: Tuple[float, float], view_radius_cells: int = 4):
        """Mark cells around the player as observed. Clear un-walled cells to EMPTY.
        Uses numpy slicing for speed instead of Python loops.
        Radius reduced from 8 to 4 to avoid aggressively clearing undetected walls."""
        pr, pc = self.pixel_to_grid(player_pos[0], player_pos[1])

        r_lo = max(0, pr - view_radius_cells)
        r_hi = min(self.rows, pr + view_radius_cells + 1)
        c_lo = max(0, pc - view_radius_cells)
        c_hi = min(self.cols, pc + view_radius_cells + 1)

        # Build local coordinate arrays
        rr = np.arange(r_lo, r_hi)
        cc = np.arange(c_lo, c_hi)
        rg, cg = np.meshgrid(rr, cc, indexing='ij')
        dist_sq = (rg - pr) ** 2 + (cg - pc) ** 2
        in_circle = dist_sq <= view_radius_cells * view_radius_cells

        self.observed[r_lo:r_hi, c_lo:c_hi] |= in_circle
        # Only overwrite UNKNOWN -> EMPTY
        unknown_mask = (self.grid[r_lo:r_hi, c_lo:c_hi] == UNKNOWN) & in_circle
        self.grid[r_lo:r_hi, c_lo:c_hi][unknown_mask] = EMPTY

    def _update_gas_zone(self, storm_center: Tuple[float, float], storm_radius: float):
        """Mark cells outside the safe zone as GAS. Uses numpy for speed."""
        cr, cc = self.pixel_to_grid(storm_center[0], storm_center[1])
        radius_cells = storm_radius / self.cell_size

        rr = np.arange(self.rows)
        ccs = np.arange(self.cols)
        rg, cg = np.meshgrid(rr, ccs, indexing='ij')
        dist_sq = (rg - cr) ** 2 + (cg - cc) ** 2
        outside = dist_sq > radius_cells * radius_cells
        passable = (self.grid == EMPTY) | (self.grid == UNKNOWN)
        self.grid[outside & passable] = GAS

        self._gas_center = (cc, cr)
        self._gas_radius_cells = radius_cells

    # damage Tracking
    def record_damage(self, px: float, py: float, amount: float = 1.0):
        """Record that the player took damage at a position.
        Increases danger score for that area."""
        r, c = self.pixel_to_grid(px, py)
        now = time.time()
        self._damage_events.append((r, c, amount, now))

        # Apply to danger map with Gaussian spread
        for dr in range(-3, 4):
            for dc in range(-3, 4):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    dist = math.sqrt(dr * dr + dc * dc)
                    falloff = math.exp(-dist * dist / 4.0)  # σ = 2 cells
                    self.danger_map[nr, nc] += amount * falloff

        # Prune old events (>60s)
        self._damage_events = [(r, c, a, t) for r, c, a, t in self._damage_events
                                if now - t < 60.0]
        # Safety cap in case pruning falls behind
        if len(self._damage_events) > 2000:
            self._damage_events = self._damage_events[-1000:]

    def get_danger_at(self, px: float, py: float) -> float:
        """Get the danger score at a pixel position."""
        r, c = self.pixel_to_grid(px, py)
        return float(self.danger_map[r, c])

    # queries
    def is_walkable(self, px: float, py: float) -> bool:
        """Check if a pixel position is walkable (not wall, water, or gas)."""
        r, c = self.pixel_to_grid(px, py)
        return self.grid[r, c] in (EMPTY, BUSH, DESTROYED, UNKNOWN)

    def is_wall_at(self, px: float, py: float) -> bool:
        """Check if a wall exists at pixel position."""
        r, c = self.pixel_to_grid(px, py)
        return self.grid[r, c] == WALL

    def is_bush_at(self, px: float, py: float) -> bool:
        """Check if a bush exists at pixel position."""
        r, c = self.pixel_to_grid(px, py)
        return self.grid[r, c] == BUSH

    def is_gas_at(self, px: float, py: float) -> bool:
        """Check if gas/storm is at pixel position."""
        r, c = self.pixel_to_grid(px, py)
        return self.grid[r, c] == GAS

    def get_cell_type(self, px: float, py: float) -> int:
        """Get the cell type at a pixel position."""
        r, c = self.pixel_to_grid(px, py)
        return int(self.grid[r, c])

    # raycasting (Line of Sight)
    def raycast(self, start_px: float, start_py: float,
                end_px: float, end_py: float,
                ignore_bushes: bool = False) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Cast a ray from start to end, checking for wall obstructions."""

        # Convert pixel coordinates to grid coordinates
        r0, c0 = self.pixel_to_grid(start_px, start_py)
        r1, c1 = self.pixel_to_grid(end_px, end_py)

        # Bresenham's line algorithm
        cells = self._bresenham(r0, c0, r1, c1)

        for r, c in cells:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                cell = self.grid[r, c]
                if cell == WALL:
                    px, py = self.grid_to_pixel(r, c)
                    return False, (px, py)
                if cell == BUSH and not ignore_bushes:
                    # Bushes block LoS but are passable
                    pass  # Don't block for basic LoS, only for visibility
            else:
                # Out of bounds = blocked
                return False, None

        return True, None

    def _bresenham(self, r0: int, c0: int, r1: int, c1: int) -> List[Tuple[int, int]]:
        """Bresenham's line algorithm for grid raycasting."""
        cells = []
        dr = abs(r1 - r0)
        dc = abs(c1 - c0)
        sr = 1 if r0 < r1 else -1
        sc = 1 if c0 < c1 else -1
        err = dr - dc

        while True:
            cells.append((r0, c0))
            if r0 == r1 and c0 == c1:
                break
            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r0 += sr
            if e2 < dr:
                err += dc
                c0 += sc

        return cells

    def directional_raycasts(self, px: float, py: float, n_rays: int = 8,
                              max_dist_px: float = 400) -> List[float]:
        """Cast rays in N evenly-spaced directions from a point.

        Returns list of distances (in pixels) to the nearest wall in each direction.
        If no wall found, returns max_dist_px.
        """
        distances = []
        for i in range(n_rays):
            angle = (2 * math.pi * i) / n_rays
            end_x = px + max_dist_px * math.cos(angle)
            end_y = py + max_dist_px * math.sin(angle)

            clear, hit_point = self.raycast(px, py, end_x, end_y)
            if clear or hit_point is None:
                distances.append(max_dist_px)
            else:
                hx, hy = hit_point
                dist = math.sqrt((hx - px) ** 2 + (hy - py) ** 2)
                distances.append(dist)

        return distances

    # pathfinding Helpers
    def get_safe_direction(self, px: float, py: float, n_rays: int = 8,
                            max_dist_px: float = 300) -> Optional[Tuple[float, float]]:
        """Find the direction with lowest danger score.
        Returns (dx, dy) normalized direction vector, or None."""
        distances = self.directional_raycasts(px, py, n_rays, max_dist_px)
        best_score = float('inf')
        best_dir = None

        for i in range(n_rays):
            angle = (2 * math.pi * i) / n_rays
            dx = math.cos(angle)
            dy = math.sin(angle)

            # Check point along this direction
            check_dist = min(distances[i], 200)
            check_x = px + dx * check_dist
            check_y = py + dy * check_dist

            danger = self.get_danger_at(check_x, check_y)
            # Penalize short distances (walls nearby)
            wall_penalty = max(0, (100 - distances[i]) / 100) * 5.0
            score = danger + wall_penalty

            if score < best_score and distances[i] > 50:  # Must have room to move
                best_score = score
                best_dir = (dx, dy)

        return best_dir

    def get_nearby_cover(self, px: float, py: float, search_radius_px: float = 200
                          ) -> List[Tuple[float, float, str]]:
        """Find nearby cover positions (walls/bushes the player can hide behind).

        Returns: List of (px, py, type) where type is 'wall' or 'bush'.
        """
        covers = []
        r, c = self.pixel_to_grid(px, py)
        radius_cells = int(search_radius_px / self.cell_size)

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    cell = self.grid[nr, nc]
                    if cell in (WALL, BUSH):
                        cpx, cpy = self.grid_to_pixel(nr, nc)
                        dist = math.sqrt((cpx - px) ** 2 + (cpy - py) ** 2)
                        if dist <= search_radius_px:
                            covers.append((cpx, cpy, CELL_NAMES[cell]))

        return covers

    def get_occupancy_around(self, px: float, py: float, radius: int = 2) -> np.ndarray:
        """Get a (2*radius+1)×(2*radius+1) sub-grid centered on pixel position.
        Used for RL state encoding."""
        r, c = self.pixel_to_grid(px, py)
        size = 2 * radius + 1
        result = np.full((size, size), UNKNOWN, dtype=np.uint8)

        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    result[dr + radius, dc + radius] = self.grid[nr, nc]

        return result

    # match Reset
    def reset(self):
        """Reset all map data for a new match."""
        self.grid.fill(UNKNOWN)
        self.danger_map.fill(0.0)
        self.observed.fill(False)
        self._damage_events.clear()
        self._match_start_time = time.time()
        self._last_update_time = 0.0
        self._gas_center = (self.cols // 2, self.rows // 2)
        self._gas_radius_cells = max(self.cols, self.rows)

    # debug / Visualization
    def get_grid_image(self, scale: int = 4) -> np.ndarray:
        """Generate a color-coded image of the grid for debugging."""
        colors = {
            UNKNOWN:   (40, 40, 40),
            EMPTY:     (200, 200, 200),
            WALL:      (80, 60, 40),
            BUSH:      (30, 150, 30),
            DESTROYED: (150, 100, 50),
            WATER:     (50, 100, 200),
            GAS:       (150, 30, 150),
        }

        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        for cell_type, color in colors.items():
            mask = self.grid == cell_type
            img[mask] = color

        # Scale up for visibility
        if scale > 1:
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        return img

    def get_danger_overlay(self, scale: int = 4) -> np.ndarray:
        """Generate a red-channel overlay of the danger heatmap."""
        max_danger = self.danger_map.max()
        if max_danger <= 0:
            return np.zeros((self.rows * scale, self.cols * scale, 3), dtype=np.uint8)

        normalized = (self.danger_map / max_danger * 255).astype(np.uint8)
        img = np.zeros((self.rows, self.cols, 3), dtype=np.uint8)
        img[:, :, 0] = normalized  # Red channel = danger

        if scale > 1:
            img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

        return img

    def __repr__(self):
        wall_count = int(np.sum(self.grid == WALL))
        bush_count = int(np.sum(self.grid == BUSH))
        observed_pct = int(np.sum(self.observed) / self.observed.size * 100)
        return (f"SpatialMemory({self.cols}×{self.rows}, "
                f"walls={wall_count}, bushes={bush_count}, "
                f"observed={observed_pct}%)")
