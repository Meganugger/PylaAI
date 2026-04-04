# per-enemy tracking: velocity, hp, cooldowns, brawler identification

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple


# per-Entity Track
class EntityTrack:
    """Persistent track for a single detected entity (enemy/teammate)."""

    __slots__ = (
        "track_id", "positions", "velocities", "hp_history",
        "last_seen", "first_seen", "hits_on_us", "attack_times",
        "last_attack_time", "avg_attack_interval", "brawler_class",
        "brawler_confidence", "is_alive", "death_time",
        "estimated_reload_sec", "estimated_ammo",
        "_miss_count", "_vel_smooth",
        "_accel_smooth", "_prev_vel", "_prev_vel_time",
        "_dir_change_times", "_last_dir",
    )

    def __init__(self, track_id: int, pos: Tuple[float, float], now: float):
        self.track_id = track_id
        # Position history: deque of (cx, cy, timestamp)
        self.positions: deque = deque(maxlen=30)
        self.positions.append((*pos, now))
        self.velocities: deque = deque(maxlen=15)  # (vx, vy) per-frame

        # HP
        self.hp_history: deque = deque(maxlen=20)  # (hp_percent, timestamp)

        # Timing
        self.last_seen: float = now
        self.first_seen: float = now

        # Attack tracking (inferred from player HP drops near this enemy)
        self.hits_on_us: int = 0
        self.attack_times: deque = deque(maxlen=20)
        self.last_attack_time: float = 0.0
        self.avg_attack_interval: float = 2.0  # Default ~2 sec reload cycle

        # Brawler classification
        self.brawler_class: str = "unknown"
        self.brawler_confidence: float = 0.0

        # Alive / respawn
        self.is_alive: bool = True
        self.death_time: float = 0.0

        # Reload estimation
        self.estimated_reload_sec: float = 1.6  # Default reload speed
        self.estimated_ammo: int = 3  # Full ammo assumed initially

        # Internal
        self._miss_count: int = 0  # Frames not seen consecutively
        self._vel_smooth: Tuple[float, float] = (0.0, 0.0)
        # Acceleration tracking for curved prediction
        self._accel_smooth: Tuple[float, float] = (0.0, 0.0)
        self._prev_vel: Tuple[float, float] = (0.0, 0.0)
        self._prev_vel_time: float = 0.0
        # Direction change tracking (juke detection)
        self._dir_change_times: list = []
        self._last_dir: str = 'none'

    @property
    def pos(self) -> Tuple[float, float]:
        if self.positions:
            p = self.positions[-1]
            return (p[0], p[1])
        return (0.0, 0.0)

    @property
    def velocity(self) -> Tuple[float, float]:
        return self._vel_smooth

    @property
    def speed(self) -> float:
        return math.hypot(self._vel_smooth[0], self._vel_smooth[1])

    @property
    def hp(self) -> float:
        if self.hp_history:
            return self.hp_history[-1][0]
        return -1  # Unknown

    @property
    def time_since_attack(self) -> float:
        if self.last_attack_time > 0:
            return time.time() - self.last_attack_time
        return 999.0

    @property
    def is_likely_reloading(self) -> bool:
        """True if this enemy recently fired and is probably reloading."""
        return 0 < self.time_since_attack < self.estimated_reload_sec

    def update_position(self, pos: Tuple[float, float], now: float):
        """Add new position observation."""
        self.positions.append((*pos, now))
        self.last_seen = now
        self._miss_count = 0
        self.is_alive = True

        # Compute velocity from last two positions
        if len(self.positions) >= 2:
            prev = self.positions[-2]
            dt = now - prev[2]
            if dt > 0.01:
                vx = (pos[0] - prev[0]) / dt
                vy = (pos[1] - prev[1]) / dt
                self.velocities.append((vx, vy))

                # EMA smoothing (alpha=0.6)
                alpha = 0.6
                self._vel_smooth = (
                    alpha * vx + (1 - alpha) * self._vel_smooth[0],
                    alpha * vy + (1 - alpha) * self._vel_smooth[1],
                )

                # Acceleration estimation
                if self._prev_vel_time > 0:
                    dt_vel = now - self._prev_vel_time
                    if dt_vel > 0.03:
                        ax = (self._vel_smooth[0] - self._prev_vel[0]) / dt_vel
                        ay = (self._vel_smooth[1] - self._prev_vel[1]) / dt_vel
                        a_alpha = 0.4
                        self._accel_smooth = (
                            a_alpha * ax + (1 - a_alpha) * self._accel_smooth[0],
                            a_alpha * ay + (1 - a_alpha) * self._accel_smooth[1],
                        )
                self._prev_vel = self._vel_smooth
                self._prev_vel_time = now

                # Direction change tracking
                spd = math.hypot(self._vel_smooth[0], self._vel_smooth[1])
                if spd > 40:
                    if abs(self._vel_smooth[0]) > abs(self._vel_smooth[1]):
                        cur_dir = 'r' if self._vel_smooth[0] > 0 else 'l'
                    else:
                        cur_dir = 'd' if self._vel_smooth[1] > 0 else 'u'
                    if cur_dir != self._last_dir and self._last_dir != 'none':
                        self._dir_change_times.append(now)
                    self._last_dir = cur_dir
                # Prune old direction changes
                self._dir_change_times = [t for t in self._dir_change_times if now - t < 1.5]

    def update_hp(self, hp_percent: float, now: float):
        """Record HP reading."""
        if hp_percent >= 0:
            self.hp_history.append((hp_percent, now))

    def record_attack(self, now: float):
        """Record that this enemy fired (inferred from player HP drop)."""
        if now - self.last_attack_time > 0.3:  # Debounce
            if self.last_attack_time > 0:
                interval = now - self.last_attack_time
                self.attack_times.append(interval)
                # Update average interval
                if self.attack_times:
                    self.avg_attack_interval = (
                        sum(self.attack_times) / len(self.attack_times)
                    )
            self.last_attack_time = now
            self.hits_on_us += 1
            self.estimated_ammo = max(0, self.estimated_ammo - 1)

    def tick_ammo(self, now: float):
        """Estimate ammo regeneration over time."""
        if self.estimated_ammo < 3 and self.last_attack_time > 0:
            time_since = now - self.last_attack_time
            reloaded = int(time_since / self.estimated_reload_sec)
            self.estimated_ammo = min(3, self.estimated_ammo + reloaded)

    def mark_dead(self, now: float):
        """Mark this entity as dead (disappeared from detection)."""
        self.is_alive = False
        self.death_time = now

    def predict_position(self, dt: float) -> Tuple[float, float]:
        """Predict position dt seconds into the future.
        Uses velocity + acceleration for curved prediction.
        Dampens prediction when entity is juking frequently."""
        cx, cy = self.pos
        vx, vy = self._vel_smooth
        ax, ay = self._accel_smooth

        # Check for juke: reduce prediction if direction changed a lot recently
        juke_count = len(self._dir_change_times)
        if juke_count >= 4:
            juke_factor = 0.2  # Heavy juking - barely lead
        elif juke_count >= 2:
            juke_factor = 0.5  # Moderate strafing
        else:
            juke_factor = 1.0  # Steady movement

        # Acceleration-aware: s = v*t + 0.5*a*t²
        accel_mag = math.hypot(ax, ay)
        if accel_mag > 30:  # Meaningful acceleration
            px = cx + (vx * dt + 0.5 * ax * dt * dt) * juke_factor
            py = cy + (vy * dt + 0.5 * ay * dt * dt) * juke_factor
        else:
            px = cx + vx * dt * juke_factor
            py = cy + vy * dt * juke_factor
        return (px, py)

    def get_safe_window(self) -> float:
        """Estimated seconds of safety before this enemy fires again.

        Returns 0 if enemy could fire at any time, positive if reloading.
        """
        if self.last_attack_time == 0:
            return 0  # Unknown - assume dangerous
        time_since = time.time() - self.last_attack_time
        remaining_reload = self.estimated_reload_sec - time_since
        return max(0, remaining_reload)


# main Enemy Tracker
class EnemyTracker:
    """Track all enemies across frames with persistent IDs.

    Associates new detections with existing tracks via position proximity
    (Hungarian-style greedy matching).
    """

    def __init__(self, association_radius: float = 180.0,
                 max_miss_frames: int = 15):
        self.association_radius = association_radius
        self.max_miss_frames = max_miss_frames

        self._tracks: Dict[int, EntityTrack] = {}
        self._next_id: int = 0

        # Player HP tracking for attack detection
        self._prev_player_hp: float = 100
        self._prev_hp_time: float = 0.0

        # Statistics
        self.total_enemies_ever: int = 0
        self.total_attacks_detected: int = 0

    @property
    def active_tracks(self) -> List[EntityTrack]:
        """Return all currently alive, recently-seen tracks."""
        return [t for t in self._tracks.values() if t.is_alive]

    def update(self, detected_enemies: list, player_pos: Tuple[float, float],
               player_hp: float, frame=None, now: float = None):
        """Main update - call once per frame.

        """
        if now is None:
            now = time.time()

        # Convert bboxes to center positions
        enemy_centers = []
        for bbox in detected_enemies:
            if len(bbox) >= 4:
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                enemy_centers.append((cx, cy, bbox))
            elif len(bbox) == 2:
                enemy_centers.append((bbox[0], bbox[1], None))

        # greedy assignment: match detections -> existing tracks
        used_tracks = set()
        used_detections = set()

        # Build distance matrix
        assignments = []
        for di, (ecx, ecy, bbox) in enumerate(enemy_centers):
            for tid, track in self._tracks.items():
                if not track.is_alive:
                    continue
                dx = ecx - track.pos[0]
                dy = ecy - track.pos[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < self.association_radius:
                    assignments.append((dist, di, tid))

        # Sort by distance and greedily assign
        assignments.sort(key=lambda x: x[0])
        for dist, di, tid in assignments:
            if di in used_detections or tid in used_tracks:
                continue
            ecx, ecy, bbox = enemy_centers[di]
            self._tracks[tid].update_position((ecx, ecy), now)
            used_tracks.add(tid)
            used_detections.add(di)

        # create new tracks for unmatched detections
        for di, (ecx, ecy, bbox) in enumerate(enemy_centers):
            if di not in used_detections:
                new_track = EntityTrack(self._next_id, (ecx, ecy), now)
                self._tracks[self._next_id] = new_track
                self._next_id += 1
                self.total_enemies_ever += 1

        # increment miss count for unmatched tracks
        for tid, track in self._tracks.items():
            if tid not in used_tracks and track.is_alive:
                track._miss_count += 1
                if track._miss_count >= self.max_miss_frames:
                    track.mark_dead(now)

        # Prune dead tracks older than 5 seconds to prevent unbounded memory growth
        stale_ids = [
            tid for tid, t in self._tracks.items()
            if not t.is_alive and (now - t.death_time) > 5.0
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        # detect attacks: if player HP dropped + enemy nearby
        hp_dropped = (player_hp < self._prev_player_hp - 3 and
                      self._prev_player_hp > 0 and player_hp > 0)
        if hp_dropped:
            self._attribute_attack(player_pos, now)
            self.total_attacks_detected += 1
        self._prev_player_hp = player_hp
        self._prev_hp_time = now

        # tick ammo regen for all tracks
        for track in self._tracks.values():
            if track.is_alive:
                track.tick_ammo(now)

    def _attribute_attack(self, player_pos: Tuple[float, float], now: float):
        """When player HP drops, attribute to nearest alive enemy."""
        best_track = None
        best_dist = 500  # Only consider enemies within 500px

        for track in self._tracks.values():
            if not track.is_alive:
                continue
            dx = track.pos[0] - player_pos[0]
            dy = track.pos[1] - player_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < best_dist:
                best_dist = dist
                best_track = track

        if best_track is not None:
            best_track.record_attack(now)

    def get_tracked_enemies(self) -> List[Dict]:
        """Return enriched enemy data for blackboard / decision-making.

        Returns list of dicts:
            track_id, pos, velocity, speed, hp, is_reloading,
            safe_window, hits_on_us, brawler_class, attack_interval
        """
        result = []
        for track in self._tracks.values():
            if not track.is_alive:
                continue
            result.append({
                "track_id": track.track_id,
                "pos": track.pos,
                "velocity": track.velocity,
                "speed": track.speed,
                "hp": track.hp,
                "is_reloading": track.is_likely_reloading,
                "safe_window": track.get_safe_window(),
                "hits_on_us": track.hits_on_us,
                "brawler_class": track.brawler_class,
                "attack_interval": track.avg_attack_interval,
                "estimated_ammo": track.estimated_ammo,
                "time_visible": time.time() - track.first_seen,
            })
        return result

    def get_safest_target(self, player_pos: Tuple[float, float] = None) -> Optional[Dict]:
        """Return the enemy that is safest to attack right now.

        Prioritizes:
          1. Enemies currently reloading (safe window > 0)
          2. Low HP enemies
          3. Closest enemies (if equally safe)
        """
        enemies = self.get_tracked_enemies()
        if not enemies:
            return None

        def safety_score(e):
            score = 0
            # Reloading = +100 safety
            if e["is_reloading"]:
                score += 100 + e["safe_window"] * 50
            # Low HP = +50 per missing %
            if e["hp"] > 0:
                score += (100 - e["hp"]) * 0.5
            # Closer = slightly better (inverse distance)
            if player_pos is not None:
                dist = math.hypot(
                    e["pos"][0] - player_pos[0],
                    e["pos"][1] - player_pos[1]
                )
                score += max(0, 500 - dist) * 0.1
            return score

        return max(enemies, key=safety_score)

    def get_most_dangerous(self) -> Optional[Dict]:
        """Return the most threatening enemy (most attacks on us, closest)."""
        enemies = self.get_tracked_enemies()
        if not enemies:
            return None

        def threat_score(e):
            score = e["hits_on_us"] * 10
            # Enemies not reloading are more dangerous right now
            if not e["is_reloading"]:
                score += 20
            if e["estimated_ammo"] > 0:
                score += e["estimated_ammo"] * 5
            return score

        return max(enemies, key=threat_score)

    def get_predicted_positions(self, dt: float = 0.5) -> Dict[int, Tuple[float, float]]:
        """Get predicted positions for all alive enemies dt seconds ahead."""
        predictions = {}
        for track in self._tracks.values():
            if track.is_alive:
                predictions[track.track_id] = track.predict_position(dt)
        return predictions

    def get_enemy_with_reload_window(self) -> Optional[EntityTrack]:
        """Find an enemy in a reload window (best time to rush)."""
        best_track = None
        best_window = 0.0
        for track in self._tracks.values():
            if track.is_alive and track.is_likely_reloading:
                window = track.get_safe_window()
                if window > best_window:
                    best_window = window
                    best_track = track
        return best_track

    def reset(self):
        """Reset all tracking for a new match."""
        self._tracks.clear()
        self._next_id = 0
        self._prev_player_hp = 100
        self._prev_hp_time = 0.0
        self.total_enemies_ever = 0
        self.total_attacks_detected = 0

    def get_summary(self) -> str:
        """Debug summary of all tracks."""
        alive = [t for t in self._tracks.values() if t.is_alive]
        if not alive:
            return "No enemies tracked"
        lines = [f"Tracking {len(alive)} enemies:"]
        for t in alive:
            status = "RELOAD" if t.is_likely_reloading else "ARMED"
            lines.append(
                f"  [{t.track_id}] pos=({t.pos[0]:.0f},{t.pos[1]:.0f}) "
                f"spd={t.speed:.0f} hp={t.hp:.0f}% {status} "
                f"ammo≈{t.estimated_ammo} hits={t.hits_on_us}"
            )
        return "\n".join(lines)
