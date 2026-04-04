# A* pathfinding on the spatial memory grid, outputs WASD strings

from __future__ import annotations

import heapq
import math
import time
from typing import List, Tuple, Optional

import numpy as np

# Cell types from spatial_memory
UNKNOWN   = 0
EMPTY     = 1
WALL      = 2
BUSH      = 3
DESTROYED = 4
WATER     = 5
GAS       = 6

# Movement cost multipliers
_SQRT2 = math.sqrt(2)

# 8-directional neighbors: (drow, dcol, cost)
_NEIGHBORS = [
    (-1,  0, 1.0),   # N
    ( 1,  0, 1.0),   # S
    ( 0, -1, 1.0),   # W
    ( 0,  1, 1.0),   # E
    (-1, -1, _SQRT2), # NW
    (-1,  1, _SQRT2), # NE
    ( 1, -1, _SQRT2), # SW
    ( 1,  1, _SQRT2), # SE
]


def _octile_heuristic(r0: int, c0: int, r1: int, c1: int) -> float:
    """Octile distance heuristic - consistent for 8-directional grids."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    return max(dr, dc) + (_SQRT2 - 1) * min(dr, dc)


def inflate_walls(grid: np.ndarray, radius: int = 1) -> np.ndarray:
    """Inflate wall cells by `radius` cells in all directions.
    
    Creates a copy of the grid where cells adjacent to WALLs are also
    marked as WALL, providing clearance for the brawler's collision radius.
    Uses scipy.ndimage for speed when available, falls back to numpy.
    """
    wall_mask = (grid == WALL) | (grid == WATER)
    try:
        from scipy.ndimage import binary_dilation
        struct = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
        expanded = binary_dilation(wall_mask, structure=struct)
    except ImportError:
        # Fallback: pad-based approach (no np.roll loops)
        rows, cols = grid.shape
        padded = np.pad(wall_mask, radius, mode='constant', constant_values=False)
        expanded = np.zeros_like(wall_mask)
        for dr in range(2 * radius + 1):
            for dc in range(2 * radius + 1):
                expanded |= padded[dr:dr + rows, dc:dc + cols]
    inflated = grid.copy()
    inflated[expanded & (grid != WALL) & (grid != WATER)] = WALL
    return inflated


def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
          danger_map: Optional[np.ndarray] = None,
          gas_cost: float = 50.0,
          danger_weight: float = 2.0,
          max_iterations: int = 1500) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding on a 2D grid."""
    rows, cols = grid.shape
    sr, sc = start
    gr, gc = goal

    # Clamp to grid bounds
    sr = max(0, min(rows - 1, sr))
    sc = max(0, min(cols - 1, sc))
    gr = max(0, min(rows - 1, gr))
    gc = max(0, min(cols - 1, gc))

    # If start or goal is in a wall, snap to nearest walkable
    if grid[sr, sc] in (WALL, WATER):
        snap = _find_nearest_walkable(grid, sr, sc)
        if snap is None:
            return None
        sr, sc = snap
    if grid[gr, gc] in (WALL, WATER):
        snap = _find_nearest_walkable(grid, gr, gc)
        if snap is None:
            return None
        gr, gc = snap

    start = (sr, sc)
    goal = (gr, gc)

    if start == goal:
        return [start]

    # Open set: (f_score, counter, (row, col))
    counter = 0
    open_set = [(0.0, counter, start)]
    came_from: dict = {}
    g_score = {start: 0.0}
    iterations = 0

    walkable = {EMPTY, BUSH, DESTROYED, UNKNOWN}

    while open_set and iterations < max_iterations:
        iterations += 1
        f, _, current = heapq.heappop(open_set)
        cr, cc = current

        if current == goal:
            return _reconstruct_path(came_from, current)

        for dr, dc, base_cost in _NEIGHBORS:
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            cell = grid[nr, nc]
            if cell == WALL or cell == WATER:
                continue
            if cell == GAS:
                move_cost = base_cost + gas_cost
            elif cell not in walkable:
                continue
            else:
                move_cost = base_cost

            # Danger cost
            if danger_map is not None:
                move_cost += danger_map[nr, nc] * danger_weight

            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score.get((nr, nc), float('inf')):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative_g
                h = _octile_heuristic(nr, nc, gr, gc)
                counter += 1
                heapq.heappush(open_set, (tentative_g + h, counter, (nr, nc)))

    # No path found (or max iterations hit)
    return None


def _find_nearest_walkable(grid: np.ndarray, r: int, c: int, max_radius: int = 5
                           ) -> Optional[Tuple[int, int]]:
    """Find the nearest walkable cell to (r, c) within max_radius."""
    rows, cols = grid.shape
    for radius in range(1, max_radius + 1):
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) != radius and abs(dc) != radius:
                    continue  # Only check ring, not filled area
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if grid[nr, nc] in (EMPTY, BUSH, DESTROYED, UNKNOWN):
                        return (nr, nc)
    return None


def _reconstruct_path(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Trace back from goal to start."""
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def smooth_path(path: List[Tuple[int, int]], grid: np.ndarray) -> List[Tuple[int, int]]:
    """Remove unnecessary waypoints via line-of-sight checks on the grid.
    
    If you can walk straight from waypoint A to waypoint C without hitting
    a wall, skip waypoint B.
    """
    if len(path) <= 2:
        return path
    
    smoothed = [path[0]]
    i = 0
    
    while i < len(path) - 1:
        # Try to skip as far ahead as possible
        furthest = i + 1
        for j in range(len(path) - 1, i + 1, -1):
            if _grid_line_clear(grid, path[i], path[j]):
                furthest = j
                break
        smoothed.append(path[furthest])
        i = furthest
    
    return smoothed


def _grid_line_clear(grid: np.ndarray, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Check if a straight line between two grid cells is clear of walls.
    Uses Bresenham's algorithm."""
    r0, c0 = a
    r1, c1 = b
    rows, cols = grid.shape
    
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    
    while True:
        if r0 < 0 or r0 >= rows or c0 < 0 or c0 >= cols:
            return False
        if grid[r0, c0] in (WALL, WATER):
            return False
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dc
            c0 += sc
    
    return True


def grid_to_wasd(current_pos: Tuple[float, float], 
                 target_pos: Tuple[float, float],
                 threshold: float = 15.0) -> str:
    """Convert a pixel-space direction into a WASD string."""
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    if abs(dx) < threshold and abs(dy) < threshold:
        return ""
    keys = ""
    if dy < -threshold:
        keys += "W"
    elif dy > threshold:
        keys += "S"
    if dx < -threshold:
        keys += "A"
    elif dx > threshold:
        keys += "D"
    return keys


class PathPlanner:
    """Manages A* pathfinding with caching, replanning, and wall inflation.
    
    Sits between the SpatialMemory grid and the movement decision code.
    Call `get_movement_toward(player_pos, goal_pos, spatial_memory)` to get
    a WASD string that navigates around obstacles.
    
    Features:
      - Automatic wall inflation for clearance
      - Path caching with smart replanning triggers
      - Danger-aware routing (avoids high-danger cells)
      - Stuck detection triggers immediate replan
      - Path visualization data for overlay
    """
    
    def __init__(self, cell_size: int = 40):
        self.cell_size = cell_size
        
        # Current path state
        self._path: Optional[List[Tuple[int, int]]] = None  # Grid coords
        self._path_px: Optional[List[Tuple[float, float]]] = None  # Pixel coords
        self._waypoint_idx: int = 0
        self._goal_grid: Optional[Tuple[int, int]] = None
        self._goal_px: Optional[Tuple[float, float]] = None
        
        # Replanning
        self._path_time: float = 0.0  # When path was computed
        self._replan_interval: float = 1.0  # Max age before forced replan
        self._min_replan_gap: float = 0.20  # Minimum 200ms between A* runs
        self._last_grid_hash: int = 0  # Detect grid changes
        self._inflated_grid: Optional[np.ndarray] = None
        self._inflated_grid_time: float = 0.0
        self._inflate_interval: float = 1.5  # 1.5s cache - react faster to new walls
        
        # Waypoint following
        self._waypoint_reach_dist: float = 35.0  # Pixels to consider waypoint reached
        self._path_deviation_dist: float = 100.0  # Max deviation before replan
        self._goal_move_dist: float = 200.0  # Goal moved by this much -> replan
        
        # WASD result cache - avoid recomputation between frames
        self._cached_wasd: str = ""
        self._cached_wasd_time: float = 0.0
        self._wasd_cache_ttl: float = 0.30  # 300ms WASD cache (prevents direction jitter)
        
        # Stuck detection integration
        self._stuck_replan_cost: float = 5.0  # Extra cost around stuck position
        self._stuck_positions: List[Tuple[float, float]] = []  # Recent stuck spots
        self._stuck_expire: float = 3.0  # Seconds to remember stuck positions
        
        # Stats
        self._total_plans: int = 0
        self._total_plan_time: float = 0.0
        self._last_plan_ms: float = 0.0
    
    def get_movement_toward(self, player_pos: Tuple[float, float],
                            goal_pos: Tuple[float, float],
                            spatial_memory,
                            force_replan: bool = False) -> str:
        """Get WASD movement toward goal using A* pathfinding."""
        now = time.time()

        # Quick WASD cache - same result for 100ms
        if (not force_replan
                and now - self._cached_wasd_time < self._wasd_cache_ttl
                and self._cached_wasd):
            return self._cached_wasd

        if spatial_memory is None:
            return grid_to_wasd(player_pos, goal_pos)

        # Convert positions to grid coords
        pr, pc = spatial_memory.pixel_to_grid(player_pos[0], player_pos[1])
        gr, gc = spatial_memory.pixel_to_grid(goal_pos[0], goal_pos[1])
        start = (pr, pc)
        goal_g = (gr, gc)

        # Decide whether to replan
        need_replan = force_replan
        if self._path is None or self._goal_grid is None:
            need_replan = True
        elif goal_g != self._goal_grid:
            # Goal moved significantly?
            if self._goal_px:
                gdist = math.hypot(goal_pos[0] - self._goal_px[0],
                                   goal_pos[1] - self._goal_px[1])
                if gdist > self._goal_move_dist:
                    need_replan = True
            else:
                need_replan = True
        if not need_replan and now - self._path_time > self._replan_interval:
            need_replan = True

        if need_replan:
            self._plan(spatial_memory, start, goal_g)
            self._goal_grid = goal_g
            self._goal_px = goal_pos

        # Follow the current path
        if self._path_px and self._waypoint_idx < len(self._path_px):
            # Advance waypoint if close enough
            while self._waypoint_idx < len(self._path_px) - 1:
                wp = self._path_px[self._waypoint_idx]
                d = math.hypot(player_pos[0] - wp[0], player_pos[1] - wp[1])
                if d < self._waypoint_reach_dist:
                    self._waypoint_idx += 1
                else:
                    break
            target = self._path_px[self._waypoint_idx]
            keys = grid_to_wasd(player_pos, target)
        else:
            keys = grid_to_wasd(player_pos, goal_pos)

        self._cached_wasd = keys
        self._cached_wasd_time = now
        return keys

    def get_movement_away(self, player_pos: Tuple[float, float],
                          threat_pos: Tuple[float, float],
                          spatial_memory) -> str:
        """Get WASD movement away from a threat using A* to a safe point."""
        # Compute a flee target: mirror of threat around player, clamped to screen
        dx = player_pos[0] - threat_pos[0]
        dy = player_pos[1] - threat_pos[1]
        mag = math.hypot(dx, dy)
        if mag < 1.0:
            dx, dy = 0, -1  # Default: flee upward
            mag = 1.0
        flee_dist = 250.0
        fx = player_pos[0] + dx / mag * flee_dist
        fy = player_pos[1] + dy / mag * flee_dist
        fx = max(50, min(1870, fx))
        fy = max(50, min(1030, fy))
        return self.get_movement_toward(player_pos, (fx, fy), spatial_memory)

    def report_stuck(self, pos: Tuple[float, float]):
        """Record a stuck position - increases cost around it for future plans."""
        now = time.time()
        self._stuck_positions.append((pos[0], pos[1], now))
        # Expire old ones
        self._stuck_positions = [(x, y, t) for x, y, t in self._stuck_positions
                                 if now - t < self._stuck_expire]
        # Force immediate replan on next call
        self._path = None
        self._path_px = None

    def reset(self):
        """Reset all pathfinding state for a new match."""
        self._path = None
        self._path_px = None
        self._waypoint_idx = 0
        self._goal_grid = None
        self._goal_px = None
        self._path_time = 0.0
        self._inflated_grid = None
        self._inflated_grid_time = 0.0
        self._stuck_positions = []
        self._cached_wasd = ""
        self._cached_wasd_time = 0.0

    @property
    def current_path_px(self) -> Optional[List[Tuple[float, float]]]:
        """Pixel-space path for overlay visualization."""
        return self._path_px

    @property
    def last_plan_ms(self) -> float:
        return self._last_plan_ms

    # ---- internal ----

    def _plan(self, spatial_memory, start: Tuple[int, int],
              goal: Tuple[int, int]):
        """Run A* on the (inflated) spatial memory grid."""
        t0 = time.time()
        now = t0

        grid = spatial_memory.grid

        # Re-inflate grid if needed (cached for 1.5s)
        if (self._inflated_grid is None
                or now - self._inflated_grid_time > self._inflate_interval):
            self._inflated_grid = inflate_walls(grid, radius=1)
            self._inflated_grid_time = now

        # Build danger map from stuck positions
        danger = None
        live_stuck = [(x, y, t) for x, y, t in self._stuck_positions
                      if now - t < self._stuck_expire]
        self._stuck_positions = live_stuck
        if live_stuck:
            danger = np.zeros(grid.shape, dtype=np.float32)
            for sx, sy, _ in live_stuck:
                sr, sc = spatial_memory.pixel_to_grid(sx, sy)
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        nr, nc = sr + dr, sc + dc
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                            danger[nr, nc] += self._stuck_replan_cost

        path = astar(self._inflated_grid, start, goal, danger_map=danger)

        if path and len(path) > 2:
            path = smooth_path(path, self._inflated_grid)

        self._path = path
        self._waypoint_idx = 0

        # Convert to pixel coords
        if path:
            self._path_px = [
                (c * spatial_memory.cell_size + spatial_memory.cell_size // 2,
                 r * spatial_memory.cell_size + spatial_memory.cell_size // 2)
                for r, c in path
            ]
        else:
            self._path_px = None

        self._path_time = now
        self._total_plans += 1
        elapsed = (time.time() - t0) * 1000
        self._last_plan_ms = elapsed
        self._total_plan_time += elapsed
