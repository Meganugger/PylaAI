# converts blackboard game state into a flat float32 vector for the RL network

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np


# constants
MAX_ENEMIES = 3
MAX_TEAMMATES = 2
N_RAYCASTS = 8
GRID_RADIUS = 2  # 5×5 grid
GRID_SIZE = (2 * GRID_RADIUS + 1) ** 2

# Feature dimensions
PLAYER_FEATURES = 7
ENEMY_FEATURES = 9 * MAX_ENEMIES
TEAMMATE_FEATURES = 4 * MAX_TEAMMATES
MAP_FEATURES = N_RAYCASTS + 2  # raycasts + bush_count + gas_distance
MATCH_FEATURES = 3
SPATIAL_FEATURES = GRID_SIZE
COMBAT_FEATURES = 9  # was 5; +4 threat/risk features

TOTAL_FEATURES = (
    PLAYER_FEATURES
    + ENEMY_FEATURES
    + TEAMMATE_FEATURES
    + MAP_FEATURES
    + MATCH_FEATURES
    + SPATIAL_FEATURES
    + COMBAT_FEATURES
)

# Normalization constants
MAX_SCREEN_X = 1920.0
MAX_SCREEN_Y = 1080.0
MAX_DISTANCE = 1200.0
MAX_SPEED = 1000.0
MAX_MATCH_TIME = 180.0


class StateEncoder:
    """Encodes game state from Blackboard into a fixed-size feature vector."""

    def __init__(self):
        self._feature_dim = TOTAL_FEATURES
        self._prev_state: Optional[np.ndarray] = None
        self._prev_player_hp: float = 100.0
        self._last_hit_ts: float = time.time()

        self._projectile_event_times: deque[float] = deque(maxlen=120)
        self._projectile_last_seen: Dict[Tuple[int, ...], float] = {}

        self._norm_path = os.path.join("rl_models", "feature_normalizer_state.json")
        self._norm_count: int = 0
        self._running_mean = np.zeros(self._feature_dim, dtype=np.float64)
        self._running_m2 = np.zeros(self._feature_dim, dtype=np.float64)
        self._save_every = 250
        self._encode_calls = 0
        self._load_normalizer_state()

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def state_dim(self) -> int:
        return self._feature_dim

    def _load_normalizer_state(self):
        try:
            if not os.path.exists(self._norm_path):
                return
            with open(self._norm_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            mean = np.asarray(data.get("mean", []), dtype=np.float64)
            m2 = np.asarray(data.get("m2", []), dtype=np.float64)
            count = int(data.get("count", 0))
            if len(mean) == self._feature_dim and len(m2) == self._feature_dim and count >= 0:
                self._running_mean = mean
                self._running_m2 = m2
                self._norm_count = count
        except Exception:
            self._running_mean = np.zeros(self._feature_dim, dtype=np.float64)
            self._running_m2 = np.zeros(self._feature_dim, dtype=np.float64)
            self._norm_count = 0

    def _save_normalizer_state(self):
        try:
            os.makedirs("rl_models", exist_ok=True)
            std = self._running_std().tolist()
            payload = {
                "count": int(self._norm_count),
                "mean": self._running_mean.tolist(),
                "std": std,
                "m2": self._running_m2.tolist(),
            }
            with open(self._norm_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception:
            pass

    def reset_feature_normalizer_state(self):
        self._norm_count = 0
        self._running_mean = np.zeros(self._feature_dim, dtype=np.float64)
        self._running_m2 = np.zeros(self._feature_dim, dtype=np.float64)
        self._save_normalizer_state()

    def _running_std(self) -> np.ndarray:
        if self._norm_count < 2:
            return np.ones(self._feature_dim, dtype=np.float64)
        var = self._running_m2 / max(1, self._norm_count - 1)
        return np.sqrt(np.maximum(var, 1e-6))

    def _update_running_stats(self, features: np.ndarray):
        self._norm_count += 1
        delta = features - self._running_mean
        self._running_mean += delta / self._norm_count
        delta2 = features - self._running_mean
        self._running_m2 += delta * delta2

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        # x_norm = (x - running_mean[i]) / (running_std[i] + 1e-8)
        std = self._running_std()
        x_norm = (features.astype(np.float64) - self._running_mean) / (std + 1e-8)
        x_norm = np.clip(x_norm, -5.0, 5.0)
        return x_norm.astype(np.float32)

    @staticmethod
    def _projectile_signature(projectile: Dict[str, float]) -> Tuple[int, ...]:
        pos_raw = projectile.get("pos", projectile.get("center", (0, 0))) or (0, 0)
        vel_raw = projectile.get("velocity", (0, 0)) or (0, 0)
        if isinstance(pos_raw, (list, tuple)) and len(pos_raw) >= 2:
            pos = (float(pos_raw[0]), float(pos_raw[1]))
        else:
            pos = (0.0, 0.0)
        if isinstance(vel_raw, (list, tuple)) and len(vel_raw) >= 2:
            vel = (float(vel_raw[0]), float(vel_raw[1]))
        else:
            vel = (0.0, 0.0)
        threat = projectile.get("threat_level", 0.0)
        return (
            int(pos[0] // 12),
            int(pos[1] // 12),
            int(vel[0] // 30),
            int(vel[1] // 30),
            int(float(threat) * 10),
        )

    def _update_incoming_projectile_count(self, projectiles: List[dict], now_ts: float) -> float:
        # Deduplicated event counting in the last 1 second.
        for p in projectiles:
            threat = float(p.get("threat_level", 0.0) or 0.0)
            eta = float(p.get("frames_to_impact", 999.0) or 999.0)
            is_incoming = threat >= 0.35 or eta <= 20
            if not is_incoming:
                continue
            sig = self._projectile_signature(p)
            last_seen = self._projectile_last_seen.get(sig, -999.0)
            if now_ts - last_seen > 0.40:
                self._projectile_event_times.append(now_ts)
            self._projectile_last_seen[sig] = now_ts

        cutoff = now_ts - 1.0
        while self._projectile_event_times and self._projectile_event_times[0] < cutoff:
            self._projectile_event_times.popleft()

        stale = [k for k, t in self._projectile_last_seen.items() if now_ts - t > 2.0]
        for key in stale:
            self._projectile_last_seen.pop(key, None)

        count_last_1s = len(self._projectile_event_times)
        return float(np.clip(count_last_1s / 10.0, 0.0, 1.0))

    def encode(self, blackboard) -> np.ndarray:
        """Encode the full game state into a normalized feature vector."""

        def _bb_dict(key: str) -> dict:
            try:
                v = blackboard[key]
                return v if isinstance(v, dict) else {}
            except Exception:
                return {}

        def _bb_list(key: str) -> list:
            try:
                v = blackboard[key]
                return v if isinstance(v, list) else []
            except Exception:
                return []

        def _bb_get(key: str, default=None):
            try:
                return blackboard.get(key, default)
            except Exception:
                return default

        raw = np.zeros(self._feature_dim, dtype=np.float32)
        idx = 0
        now_ts = time.time()

        # === PLAYER STATE ===
        player = _bb_dict("player")
        player_pos = player.get("pos", (0, 0)) or (0, 0)
        raw[idx] = player_pos[0] / MAX_SCREEN_X
        raw[idx + 1] = player_pos[1] / MAX_SCREEN_Y
        raw[idx + 2] = max(0, player.get("hp", 100)) / 100.0
        raw[idx + 3] = player.get("ammo", 3) / max(1, player.get("max_ammo", 3))
        raw[idx + 4] = 1.0 if player.get("super_ready", player.get("has_super", False)) else 0.0
        raw[idx + 5] = 1.0 if player.get("gadget_ready", player.get("has_gadget", False)) else 0.0
        raw[idx + 6] = 1.0 if player.get("hypercharge_ready", player.get("has_hypercharge", False)) else 0.0

        current_hp = float(player.get("hp", 100) or 100)
        if current_hp < self._prev_player_hp:
            self._last_hit_ts = now_ts
        self._prev_player_hp = current_hp
        idx += PLAYER_FEATURES

        # === ENEMY STATES ===
        enemies = _bb_list("enemies")
        sorted_enemies = sorted(enemies, key=lambda e: e.get("distance", 9999)) if enemies else []

        for i in range(MAX_ENEMIES):
            if i < len(sorted_enemies):
                e = sorted_enemies[i]
                epos = e.get("pos", (0, 0)) or (0, 0)

                rel_x = (epos[0] - player_pos[0]) / MAX_DISTANCE * 0.5 + 0.5
                rel_y = (epos[1] - player_pos[1]) / MAX_DISTANCE * 0.5 + 0.5

                raw[idx] = np.clip(rel_x, 0, 1)
                raw[idx + 1] = np.clip(rel_y, 0, 1)
                raw[idx + 2] = max(0, e.get("hp", 100)) / 100.0
                raw[idx + 3] = min(e.get("distance", MAX_DISTANCE), MAX_DISTANCE) / MAX_DISTANCE

                dx = epos[0] - player_pos[0]
                dy = epos[1] - player_pos[1]
                angle = (math.atan2(dy, dx) + math.pi) / (2 * math.pi)
                raw[idx + 4] = angle

                vel = e.get("velocity", (0, 0)) or (0, 0)
                raw[idx + 5] = np.clip(vel[0] / MAX_SPEED * 0.5 + 0.5, 0, 1)
                raw[idx + 6] = np.clip(vel[1] / MAX_SPEED * 0.5 + 0.5, 0, 1)

                raw[idx + 7] = 1.0 if e.get("is_hittable", e.get("hittable", False)) else 0.0
                raw[idx + 8] = min(e.get("threat_score", 50), 100) / 100.0
            idx += 9

        # === TEAMMATE STATES ===
        teammates = _bb_list("teammates")
        for i in range(MAX_TEAMMATES):
            if i < len(teammates):
                t = teammates[i]
                tpos = t.get("pos", (0, 0)) or (0, 0)

                rel_x = (tpos[0] - player_pos[0]) / MAX_DISTANCE * 0.5 + 0.5
                rel_y = (tpos[1] - player_pos[1]) / MAX_DISTANCE * 0.5 + 0.5

                raw[idx] = np.clip(rel_x, 0, 1)
                raw[idx + 1] = np.clip(rel_y, 0, 1)
                raw[idx + 2] = max(0, t.get("hp", 100)) / 100.0
                raw[idx + 3] = min(t.get("distance", MAX_DISTANCE), MAX_DISTANCE) / MAX_DISTANCE
            idx += 4

        # === MAP INFO ===
        map_data = _bb_dict("map")
        grid = map_data.get("grid")
        raycasts = [400.0] * N_RAYCASTS
        if grid is not None and hasattr(grid, "directional_raycasts"):
            try:
                raycasts = grid.directional_raycasts(player_pos[0], player_pos[1], N_RAYCASTS, 400)
                for j, dist in enumerate(raycasts):
                    raw[idx + j] = min(dist, 400) / 400.0
            except Exception:
                pass
        idx += N_RAYCASTS

        bushes = map_data.get("bushes", []) or []
        nearby_bushes = 0
        for b in bushes:
            try:
                if self._bbox_distance(player_pos, b) < 200:
                    nearby_bushes += 1
            except Exception:
                pass
        raw[idx] = min(nearby_bushes, 10) / 10.0
        idx += 1

        storm_center = map_data.get("storm_center", (960, 540)) or (960, 540)
        storm_radius = map_data.get("storm_radius", 9999) or 9999
        dist_from_center = math.sqrt(
            (player_pos[0] - storm_center[0]) ** 2 + (player_pos[1] - storm_center[1]) ** 2
        )
        gas_proximity = max(0, dist_from_center - storm_radius) / 500.0
        raw[idx] = min(gas_proximity, 1.0)
        idx += 1

        # === MATCH STATE ===
        match_data = _bb_dict("match")
        phase_map = {"early": 0.0, "mid": 0.5, "late": 1.0}
        raw[idx] = phase_map.get(match_data.get("phase", "early"), 0.0)

        our = match_data.get("our_score", 0) or 0
        their = match_data.get("their_score", 0) or 0
        score_diff = our - their
        raw[idx + 1] = np.clip(score_diff / 3.0 * 0.5 + 0.5, 0, 1)

        start_t = match_data.get("start_time", 0) or 0
        now_t = _bb_get("current_time", now_ts) or now_ts
        time_elapsed = max(0, now_t - start_t) if start_t > 0 else 0
        raw[idx + 2] = min(time_elapsed / MAX_MATCH_TIME, 1.0)
        idx += MATCH_FEATURES

        # === SPATIAL GRID ===
        if grid is not None and hasattr(grid, "get_occupancy_around"):
            try:
                occ = grid.get_occupancy_around(player_pos[0], player_pos[1], GRID_RADIUS)
                flat = occ.flatten().astype(np.float32) / 6.0
                raw[idx:idx + len(flat)] = flat[:SPATIAL_FEATURES]
            except Exception:
                pass
        idx += SPATIAL_FEATURES

        # === COMBAT STATE (9) ===
        decision = _bb_dict("decision")
        raw[idx] = 1.0 if decision.get("combo") else 0.0
        raw[idx + 1] = (_bb_get("aggression", 1.0) or 1.0) / 2.0
        raw[idx + 2] = 1.0 if player.get("shield_active", player.get("respawn_shield", False)) else 0.0
        raw[idx + 3] = 1.0 if player.get("is_regenerating", False) else 0.0

        projectiles = _bb_list("projectiles")
        max_threat = max((p.get("threat_level", 0) for p in projectiles), default=0) if projectiles else 0
        raw[idx + 4] = min(max_threat, 1.0)

        # New feature 1: cover_score
        cover_score = player.get("cover_score", None)
        if cover_score is None:
            blocked_rays = sum(1 for d in raycasts if d < 120)
            cover_from_rays = blocked_rays / max(1, len(raycasts))
            cover_from_bush = min(nearby_bushes / 4.0, 1.0)
            cover_score = 0.6 * cover_from_bush + 0.4 * cover_from_rays
        water_nearby = float(_bb_get("map.water_nearby", 0) or 0)
        water_tiles = float(map_data.get("water_tiles", _bb_get("map.water_tiles", 0)) or 0)
        water_signal = np.clip((water_nearby / 3.0) + (water_tiles / 20.0), 0.0, 1.0)
        cover_with_hazard = float(np.clip(float(cover_score) * (1.0 - 0.35 * water_signal), 0.0, 1.0))
        raw[idx + 5] = cover_with_hazard

        # New feature 2: damage_trade_ratio
        dealt_recent = float(player.get("damage_dealt_recent", _bb_get("damage_dealt_recent", 0.0)) or 0.0)
        taken_recent = float(player.get("damage_taken_recent", _bb_get("damage_taken_recent", 0.0)) or 0.0)
        if dealt_recent <= 0 and taken_recent <= 0:
            trade_ratio = 1.0
        else:
            trade_ratio = dealt_recent / max(1.0, taken_recent)
        trade_norm = float(np.clip(trade_ratio / 2.0, 0.0, 1.0))

        peek_phase = str(player.get("peek_phase", _bb_get("player.peek_phase", "idle")) or "idle").lower()
        peek_score = {
            "idle": 0.0,
            "expose": 0.6,
            "fire": 1.0,
            "hide": 0.3,
        }.get(peek_phase, 0.0)

        enemy_move_dir = str(_bb_get("enemy.move_direction", "none") or "none").lower()
        enemy_speed = float(_bb_get("enemy.speed", 0.0) or 0.0)
        strafe_like = 1.0 if enemy_move_dir in ("left", "right", "strafe", "strafe_left", "strafe_right") else 0.0
        model_pattern_pressure = float(_bb_get("enemy.pattern_pressure", 0.0) or 0.0)
        model_strafe_ratio = float(_bb_get("enemy.strafe_ratio", 0.0) or 0.0)
        model_approach_ratio = float(_bb_get("enemy.approach_ratio", 0.0) or 0.0)
        movement_pattern_pressure = float(np.clip(
            (enemy_speed / 250.0) * (0.45 + 0.35 * strafe_like + 0.2 * np.clip(model_strafe_ratio, 0.0, 1.0))
            + 0.35 * np.clip(model_pattern_pressure, 0.0, 1.0)
            + 0.15 * np.clip(model_approach_ratio, 0.0, 1.0),
            0.0,
            1.0,
        ))

        tactical_tempo = 0.5 * trade_norm + 0.3 * peek_score + 0.2 * movement_pattern_pressure
        raw[idx + 6] = float(np.clip(tactical_tempo, 0.0, 1.0))

        # New feature 3: time_since_last_hit
        tslh = player.get("time_since_last_hit", None)
        if tslh is None:
            tslh = _bb_get("time_since_last_hit", None)
        if tslh is None:
            tslh = now_ts - self._last_hit_ts
        team_aggr = float(_bb_get("enemy.team_aggression", 0.5) or 0.5)
        predicted_attack_in = float(_bb_get("enemy.predicted_attack_in", 999.0) or 999.0)
        predicted_attack_soon = float(_bb_get("enemy.predicted_attack_soon", 0.0) or 0.0)
        attack_window_risk = float(np.clip(1.0 - (predicted_attack_in / 2.5), 0.0, 1.0))
        hit_stability = float(np.clip(float(tslh) / 10.0, 0.0, 1.0))
        stability_adjusted = hit_stability * (
            1.0
            - 0.28 * float(np.clip(team_aggr, 0.0, 1.0))
            - 0.22 * float(np.clip(attack_window_risk + 0.4 * predicted_attack_soon, 0.0, 1.0))
        )
        raw[idx + 7] = float(np.clip(stability_adjusted, 0.0, 1.0))

        # New feature 4: incoming_projectiles_last_1s
        incoming_proj = self._update_incoming_projectile_count(projectiles, now_ts)
        dangerous_bushes = _bb_get("map.dangerous_bushes", []) or []
        bush_risk = float(np.clip(len(dangerous_bushes) / 3.0, 0.0, 1.0))
        terrain_trap_risk = float(np.clip(0.7 * water_signal + 0.3 * bush_risk, 0.0, 1.0))
        incoming_threat = float(np.clip(max(incoming_proj, terrain_trap_risk, attack_window_risk), 0.0, 1.0))
        raw[idx + 8] = incoming_threat
        idx += COMBAT_FEATURES

        raw = np.clip(raw, 0.0, 1.0)

        features = self._normalize_features(raw)
        self._update_running_stats(raw)

        self._encode_calls += 1
        if self._encode_calls % self._save_every == 0:
            self._save_normalizer_state()

        self._prev_state = features.copy()
        return features

    def encode_delta(self, blackboard) -> Optional[np.ndarray]:
        """Encode the state change from previous frame (for temporal features)."""
        prev = self._prev_state.copy() if self._prev_state is not None else None
        current = self.encode(blackboard)
        if prev is None:
            return None
        return current - prev

    @staticmethod
    def _bbox_distance(pos: Tuple[float, float], bbox: list) -> float:
        if len(bbox) < 4:
            return 9999.0
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        return math.sqrt((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2)

    def get_feature_names(self) -> List[str]:
        names = []

        names.extend(["p_x", "p_y", "p_hp", "p_ammo", "p_super", "p_gadget", "p_hyper"])

        for i in range(MAX_ENEMIES):
            prefix = f"e{i}_"
            names.extend([
                f"{prefix}rx", f"{prefix}ry", f"{prefix}hp", f"{prefix}dist",
                f"{prefix}angle", f"{prefix}vx", f"{prefix}vy",
                f"{prefix}hittable", f"{prefix}threat"
            ])

        for i in range(MAX_TEAMMATES):
            prefix = f"t{i}_"
            names.extend([f"{prefix}rx", f"{prefix}ry", f"{prefix}hp", f"{prefix}dist"])

        for i in range(N_RAYCASTS):
            names.append(f"ray_{i}")
        names.extend(["bushes_near", "gas_dist"])

        names.extend(["match_phase", "score_diff", "time_elapsed"])

        for r in range(2 * GRID_RADIUS + 1):
            for c in range(2 * GRID_RADIUS + 1):
                names.append(f"grid_{r}_{c}")

        names.extend([
            "combo_active",
            "aggression",
            "shield",
            "regen",
            "projectile_threat",
            "cover_with_hazard",
            "tactical_tempo",
            "hit_stability",
            "incoming_threat",
        ])

        return names

    def reset(self):
        self._prev_state = None
        self._prev_player_hp = 100.0
        self._last_hit_ts = time.time()
        self._projectile_event_times.clear()
        self._projectile_last_seen.clear()
        self._save_normalizer_state()
