# computes per-frame and end-of-match rewards for RL training

from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple


# reward Weights (tunable) - SURVIVAL-FIRST V2
REWARD_WEIGHTS = {
    # === FRAME-LEVEL ===
    "damage_dealt_per_1000": 1.5,
    "damage_taken_per_1000": -2.5,
    "kill": 20.0,
    "death": -50.0,
    "gas_death": -40.0,
    "assist": 5.0,
    "kill_streak_bonus": 10.0,
    "projectile_dodged": 2.0,
    "projectile_hit": -0.6,
    "near_miss_bonus": 1.0,
    "super_charged": 2.0,
    "ammo_wasted": -0.4,
    "good_super_use": 8.0,
    "bad_super_use": -3.0,
    "in_storm_per_sec": -4.0,
    "in_danger_zone": -2.0,
    "stuck_per_sec": -2.0,
    "hp_regenerated": 1.0,
    "safe_positioning": 0.4,
    "chase_low_hp_enemy": 0.6,
    "kill_secured": 10.0,
    "attack_while_close": 0.4,
    "passive_penalty": -0.2,
    "enemy_in_range_no_attack": -0.3,
    "exposed_penalty": -0.3,
    "overextend_penalty": -0.9,
    "low_hp_engage_penalty": -1.3,
    "peek_success_bonus": 1.8,
    "bad_peek_penalty": -1.6,
    "repeated_bad_peek_penalty": -2.2,
    "predict_dodge": 2.0,
    "outplay_kill": 6.0,
    "bait_success": 4.0,
    "teammate_healed": 0.5,

    # === MATCH-LEVEL ===
    "match_win": 55.0,
    "match_loss": -30.0,
    "match_draw": 0.0,
    "trophy_gained": 3.0,
    "trophy_lost": -2.0,
    "kda_bonus_per_point": 5.0,
    "zero_deaths_bonus": 25.0,
    "multi_kill_bonus": 8.0,
    "high_kill_game": 15.0,
    "flawless_game": 25.0,
}


class RewardCalculator:
    """Computes rewards from game state transitions."""

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = {**REWARD_WEIGHTS, **(weights or {})}

        self._normalizer_path = os.path.join("rl_models", "reward_normalizer_state.json")
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_count = 0
        self._load_reward_normalizer_state()

        self._usage_frames = 0
        self._reward_usage_counter: Dict[str, int] = {k: 0 for k in self.weights.keys()}
        self._disabled_low_usage_signals: Dict[str, float] = {}

        # Tracking state
        self._prev_player_hp: float = 100.0
        self._prev_enemy_hps: Dict[int, float] = {}
        self._prev_enemy_count: int = 0
        self._prev_ammo: int = 3
        self._prev_super: bool = False
        self._prev_pos: Tuple[float, float] = (0, 0)
        self._stuck_duration: float = 0.0
        self._storm_duration: float = 0.0
        self._shots_fired: List[float] = []
        self._damage_dealt_after_shot: Dict[float, bool] = {}
        self._last_frame_time: float = 0.0
        self._match_start_time: float = 0.0

        # Accumulators for match summary
        self._total_damage_dealt: float = 0.0
        self._total_damage_taken: float = 0.0
        self._kills: int = 0
        self._deaths: int = 0
        self._dodges: int = 0
        self._shots_total: int = 0
        self._shots_hit: int = 0
        self._kill_streak: int = 0
        self._best_kill_streak: int = 0
        self._overextend_events: int = 0
        self._low_hp_engage_events: int = 0
        self._peek_success_events: int = 0
        self._bad_peek_events: int = 0
        self._recent_bad_peek_times: deque[float] = deque(maxlen=200)

        self._reward_components: Dict[str, float] = {}
        self._total_episode_reward: float = 0.0
        self._total_episode_raw_reward: float = 0.0
        self._frame_count: int = 0

    def _load_reward_normalizer_state(self):
        try:
            if not os.path.exists(self._normalizer_path):
                return
            with open(self._normalizer_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._reward_mean = float(data.get("mean", 0.0))
            self._reward_m2 = float(data.get("m2", 0.0))
            self._reward_count = int(data.get("count", 0))
        except Exception:
            self._reward_mean = 0.0
            self._reward_m2 = 0.0
            self._reward_count = 0

    def _save_reward_normalizer_state(self):
        try:
            os.makedirs("rl_models", exist_ok=True)
            with open(self._normalizer_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mean": self._reward_mean,
                        "m2": self._reward_m2,
                        "count": self._reward_count,
                    },
                    f,
                    indent=2,
                )
        except Exception:
            pass

    def reset_reward_normalizer_state(self):
        """Hard reset reward normalizer stats (used for training reset)."""
        self._reward_mean = 0.0
        self._reward_m2 = 0.0
        self._reward_count = 0
        self._save_reward_normalizer_state()

    def _reward_std(self) -> float:
        if self._reward_count < 2:
            return 1.0
        var = self._reward_m2 / max(1, self._reward_count - 1)
        return max(math.sqrt(var), 1e-3)

    def _normalize_reward(self, raw_reward: float) -> float:
        std = self._reward_std()
        norm = (raw_reward - self._reward_mean) / (std + 1e-8)
        norm = max(-20.0, min(20.0, norm))

        self._reward_count += 1
        delta = raw_reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = raw_reward - self._reward_mean
        self._reward_m2 += delta * delta2

        if self._reward_count % 200 == 0:
            self._save_reward_normalizer_state()

        return norm

    def _estimate_cover_score(self, player: Dict[str, Any], map_data: Dict[str, Any]) -> float:
        if "cover_score" in player:
            try:
                return float(player.get("cover_score", 0.5))
            except Exception:
                return 0.5

        player_pos = player.get("pos", (0, 0)) or (0, 0)
        bushes = map_data.get("bushes", []) or []
        nearby_bushes = 0
        for b in bushes:
            if isinstance(b, (list, tuple)) and len(b) >= 4:
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                d = ((player_pos[0] - cx) ** 2 + (player_pos[1] - cy) ** 2) ** 0.5
                if d < 180:
                    nearby_bushes += 1

        bush_cover = min(nearby_bushes / 4.0, 1.0)

        ray_cover = 0.0
        grid = map_data.get("grid")
        if grid is not None and hasattr(grid, "directional_raycasts"):
            try:
                rays = grid.directional_raycasts(player_pos[0], player_pos[1], 8, 220)
                if rays:
                    blocked = sum(1 for r in rays if r < 120)
                    ray_cover = blocked / max(1, len(rays))
            except Exception:
                pass

        return float(max(0.0, min((0.65 * bush_cover + 0.35 * ray_cover), 1.0)))

    def _mark_component(self, name: str, value: float):
        if abs(value) < 1e-9:
            return
        self._reward_components[name] = self._reward_components.get(name, 0.0) + value

    def _track_usage(self):
        self._usage_frames += 1
        active = {k for k, v in self._reward_components.items() if abs(v) > 1e-9}
        for key in active:
            if key in self._reward_usage_counter:
                self._reward_usage_counter[key] += 1

        if self._usage_frames > 0 and self._usage_frames % 5000 == 0:
            self._auto_disable_low_usage_signals()

    def _auto_disable_low_usage_signals(self):
        protected = {
            "kill", "death", "damage_dealt_per_1000", "damage_taken_per_1000",
            "match_win", "match_loss", "zero_deaths_bonus", "passive_penalty",
            "enemy_in_range_no_attack", "exposed_penalty", "in_storm_per_sec",
            "in_danger_zone", "stuck_per_sec",
        }
        for key, count in self._reward_usage_counter.items():
            if key in protected:
                continue
            if key not in self.weights:
                continue
            if key in self._disabled_low_usage_signals:
                continue
            usage_rate = count / max(1, self._usage_frames)
            if usage_rate < 0.001 and abs(self.weights.get(key, 0.0)) > 0.0:
                self._disabled_low_usage_signals[key] = self.weights[key]
                self.weights[key] = 0.0
                print(f"[RL-REWARD] Disabled low-usage signal '{key}' (usage={usage_rate:.4%})")

    def compute_frame_reward(self, blackboard) -> float:
        """Compute normalized frame reward based on state transition."""
        now = time.time()
        dt = now - self._last_frame_time if self._last_frame_time > 0 else 0.033
        dt = max(0.001, min(dt, 0.25))
        self._last_frame_time = now

        reward = 0.0
        self._reward_components = {}
        player = blackboard["player"] if isinstance(blackboard, dict) else blackboard.get("player", {})
        enemies = blackboard["enemies"] if isinstance(blackboard, dict) else blackboard.get("enemies", [])

        current_hp = player.get("hp", 100)
        hp_delta = current_hp - self._prev_player_hp

        if hp_delta < 0:
            damage_taken = abs(hp_delta) * 32
            r = self.weights["damage_taken_per_1000"] * (damage_taken / 1000)
            reward += r
            self._mark_component("damage_taken_per_1000", r)
            self._total_damage_taken += damage_taken
        elif hp_delta > 0 and self._prev_player_hp < 90:
            r = self.weights["hp_regenerated"]
            reward += r
            self._mark_component("hp_regenerated", r)

        for i, enemy in enumerate(enemies):
            enemy_hp = enemy.get("hp", -1)
            if enemy_hp < 0:
                continue
            prev_hp = self._prev_enemy_hps.get(i, 100)
            if enemy_hp < prev_hp:
                damage_est = (prev_hp - enemy_hp) * 32
                r = self.weights["damage_dealt_per_1000"] * (damage_est / 1000)
                reward += r
                self._mark_component("damage_dealt_per_1000", r)
                self._total_damage_dealt += damage_est

                for shot_time in list(self._damage_dealt_after_shot.keys()):
                    if now - shot_time < 1.5 and not self._damage_dealt_after_shot[shot_time]:
                        self._damage_dealt_after_shot[shot_time] = True
                        self._shots_hit += 1

        current_enemy_count = len(enemies)
        if current_enemy_count < self._prev_enemy_count and self._prev_enemy_count > 0:
            for _, prev_hp in self._prev_enemy_hps.items():
                if prev_hp < 80:
                    self._kills += 1
                    self._kill_streak += 1
                    self._best_kill_streak = max(self._best_kill_streak, self._kill_streak)
                    r = self.weights["kill"]
                    if self._kill_streak >= 2:
                        streak_bonus = self.weights.get("kill_streak_bonus", 5.0) * (self._kill_streak - 1)
                        r += streak_bonus
                        self._mark_component("kill_streak_bonus", streak_bonus)
                    reward += r
                    self._mark_component("kill", r)
                    break

        if current_hp <= 0 and self._prev_player_hp > 0:
            r = self.weights["death"]
            reward += r
            self._mark_component("death", r)
            self._deaths += 1
            self._kill_streak = 0

        current_ammo = player.get("ammo", 3)
        if current_ammo < self._prev_ammo:
            shots = self._prev_ammo - current_ammo
            self._shots_total += shots
            for _ in range(shots):
                shot_t = now + (_ * 1e-4)
                self._shots_fired.append(shot_t)
                self._damage_dealt_after_shot[shot_t] = False

        wasted = [t for t, hit in self._damage_dealt_after_shot.items() if now - t > 1.5 and not hit]
        for t in wasted:
            r = self.weights["ammo_wasted"]
            reward += r
            self._mark_component("ammo_wasted", r)
            del self._damage_dealt_after_shot[t]

        self._shots_fired = [t for t in self._shots_fired if now - t < 5.0]
        self._damage_dealt_after_shot = {
            t: h for t, h in self._damage_dealt_after_shot.items() if now - t < 5.0
        }

        current_super = player.get("has_super", False)
        if current_super and not self._prev_super:
            r = self.weights["super_charged"]
            reward += r
            self._mark_component("super_charged", r)

        projectiles = blackboard.get("projectiles", []) if hasattr(blackboard, "get") else []
        if projectiles and hp_delta >= 0:
            high_threat = sum(1 for p in projectiles if p.get("threat_level", 0) > 0.6)
            if high_threat > 0:
                r = self.weights["projectile_dodged"] * min(high_threat, 3)
                reward += r
                self._mark_component("projectile_dodged", r)
                self._dodges += high_threat

        if player.get("in_storm", False):
            self._storm_duration += dt
            r = self.weights["in_storm_per_sec"] * dt
            reward += r
            self._mark_component("in_storm_per_sec", r)
        else:
            self._storm_duration = 0.0

        if player.get("in_danger_zone", False) or player.get("in_gas", False):
            r = self.weights.get("in_danger_zone", -2.0) * dt
            reward += r
            self._mark_component("in_danger_zone", r)

        closest_enemy_dist = player.get("closest_enemy_dist", 9999)
        if closest_enemy_dist >= 9999 and enemies:
            closest_enemy_dist = min(e.get("distance", 9999) for e in enemies)

        is_attacking = player.get("is_attacking", False) or current_ammo < self._prev_ammo
        decision = blackboard.get("decision", {}) if hasattr(blackboard, "get") else {}
        attack_range = blackboard.get("brawler", {}).get("attack_range", 400) if hasattr(blackboard, "get") else 400
        immediate_fight_intent = (
            is_attacking
            or bool(decision.get("combo", False))
            or bool(player.get("is_aiming", False))
            or closest_enemy_dist < attack_range * 0.8
        )

        if is_attacking and closest_enemy_dist < 350:
            dist_factor = max(0.2, 1.0 - closest_enemy_dist / 350)
            r = self.weights.get("attack_while_close", 0.3) * dist_factor
            reward += r
            self._mark_component("attack_while_close", r)

        # Passive penalty only in true combat-near range.
        if enemies and closest_enemy_dist < 1.5 * attack_range and not is_attacking:
            r1 = self.weights.get("passive_penalty", -0.2) * dt
            r2 = self.weights.get("enemy_in_range_no_attack", -0.3) * dt
            reward += r1 + r2
            self._mark_component("passive_penalty", r1)
            self._mark_component("enemy_in_range_no_attack", r2)

        # Exposed penalty halved and only if no immediate fight intent.
        map_data = blackboard.get("map", {}) if hasattr(blackboard, "get") else {}
        cover_score = self._estimate_cover_score(player, map_data)
        if (cover_score < 0.35 and closest_enemy_dist < attack_range * 1.8 and not immediate_fight_intent):
            r = self.weights.get("exposed_penalty", -0.3) * dt
            reward += r
            self._mark_component("exposed_penalty", r)

        retreat_hp = blackboard.get("style", {}).get("hp_retreat", 45) if hasattr(blackboard, "get") else 45
        try:
            retreat_hp = float(retreat_hp)
        except Exception:
            retreat_hp = 45.0

        if (
            enemies
            and closest_enemy_dist < attack_range * 0.75
            and cover_score < 0.28
            and hp_delta < 0
        ):
            intensity = min(2.0, 1.0 + abs(float(hp_delta)) / 10.0)
            r = self.weights.get("overextend_penalty", -0.9) * dt * intensity
            reward += r
            self._mark_component("overextend_penalty", r)
            self._overextend_events += 1

        if (
            enemies
            and current_hp <= retreat_hp
            and immediate_fight_intent
            and closest_enemy_dist < attack_range * 1.1
        ):
            r = self.weights.get("low_hp_engage_penalty", -1.3) * dt
            reward += r
            self._mark_component("low_hp_engage_penalty", r)
            self._low_hp_engage_events += 1

        peek_phase = str(player.get("peek_phase", "idle") or "idle").lower()
        if peek_phase in ("expose", "fire") and closest_enemy_dist < attack_range * 1.2:
            if hp_delta < 0 and cover_score < 0.35:
                r = self.weights.get("bad_peek_penalty", -1.6)
                reward += r
                self._mark_component("bad_peek_penalty", r)
                self._bad_peek_events += 1

                self._recent_bad_peek_times.append(now)
                while self._recent_bad_peek_times and (now - self._recent_bad_peek_times[0]) > 6.0:
                    self._recent_bad_peek_times.popleft()
                if len(self._recent_bad_peek_times) >= 3:
                    r_rep = self.weights.get("repeated_bad_peek_penalty", -2.2)
                    reward += r_rep
                    self._mark_component("repeated_bad_peek_penalty", r_rep)
            elif (is_attacking or hp_delta >= 0) and cover_score >= 0.35:
                r = self.weights.get("peek_success_bonus", 1.8)
                reward += r
                self._mark_component("peek_success_bonus", r)
                self._peek_success_events += 1

        current_pos = player.get("pos", (0, 0))
        pos_delta = ((current_pos[0] - self._prev_pos[0]) ** 2 + (current_pos[1] - self._prev_pos[1]) ** 2) ** 0.5
        if pos_delta < 20:
            self._stuck_duration += dt
            if self._stuck_duration > 3.0:
                r = self.weights["stuck_per_sec"] * dt
                reward += r
                self._mark_component("stuck_per_sec", r)
        else:
            self._stuck_duration = 0.0

        if enemies:
            safe_range = blackboard.get("brawler", {}).get("safe_range", 300) if hasattr(blackboard, "get") else 300
            if safe_range < closest_enemy_dist < attack_range * 1.2:
                r = self.weights["safe_positioning"] * dt
                reward += r
                self._mark_component("safe_positioning", r)

        self._prev_player_hp = current_hp
        self._prev_enemy_hps = {i: e.get("hp", 100) for i, e in enumerate(enemies)}
        self._prev_enemy_count = current_enemy_count
        self._prev_ammo = current_ammo
        self._prev_super = current_super
        self._prev_pos = current_pos

        self._track_usage()
        normalized_reward = self._normalize_reward(reward)
        self._reward_components["raw_reward"] = reward
        self._reward_components["normalized_reward"] = normalized_reward

        self._total_episode_raw_reward += reward
        self._total_episode_reward += normalized_reward
        self._frame_count += 1
        return normalized_reward

    def compute_match_reward(self, result: str, trophy_delta: int = 0) -> float:
        """Compute normalized end-of-match reward."""
        reward = 0.0
        components: Dict[str, float] = {}

        if result == "win":
            if self._kills == 0:
                passive_win = self.weights["match_win"] * 0.25
                reward += passive_win
                components["match_win"] = passive_win
            else:
                reward += self.weights["match_win"]
                components["match_win"] = self.weights["match_win"]
        elif result == "loss":
            reward += self.weights["match_loss"]
            components["match_loss"] = self.weights["match_loss"]
        else:
            reward += self.weights.get("match_draw", 0)

        if trophy_delta > 0:
            r = self.weights["trophy_gained"] * trophy_delta
            reward += r
            components["trophies"] = r
        elif trophy_delta < 0:
            r = self.weights["trophy_lost"] * abs(trophy_delta)
            reward += r
            components["trophies"] = r

        kda_diff = self._kills - self._deaths
        kda_weight = self.weights.get("kda_bonus_per_point", 3.0)
        if kda_diff != 0:
            r = kda_weight * kda_diff
            reward += r
            components["kda_bonus"] = r

        if self._deaths == 0 and self._kills > 0:
            r = self.weights.get("zero_deaths_bonus", 25.0)
            reward += r
            components["zero_deaths_bonus"] = r

        if self._kills >= 3:
            r = self.weights.get("multi_kill_bonus", 5.0)
            reward += r
            components["multi_kill_bonus"] = r

        self._reward_components.update(components)
        self._track_usage()

        normalized_reward = self._normalize_reward(reward)
        self._reward_components["raw_match_reward"] = reward
        self._reward_components["normalized_match_reward"] = normalized_reward

        self._total_episode_raw_reward += reward
        self._total_episode_reward += normalized_reward
        self._save_reward_normalizer_state()

        print(
            f"[RL-REWARD] Match {result}: K={self._kills} D={self._deaths} "
            f"streak={self._best_kill_streak} kda_diff={kda_diff:+d} "
            f"raw={reward:+.2f} norm={normalized_reward:+.2f} "
            f"total_ep_norm={self._total_episode_reward:.2f}"
        )
        return normalized_reward

    def get_episode_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the current episode (match)."""
        accuracy = min(1.0, max(0.0, self._shots_hit / max(1, self._shots_total)))

        usage_rates = {
            k: (v / max(1, self._usage_frames))
            for k, v in self._reward_usage_counter.items()
        }

        return {
            "total_reward": self._total_episode_reward,
            "total_raw_reward": self._total_episode_raw_reward,
            "frames": self._frame_count,
            "damage_dealt": self._total_damage_dealt,
            "damage_taken": self._total_damage_taken,
            "kills": self._kills,
            "deaths": self._deaths,
            "kill_streak": self._best_kill_streak,
            "kda": self._kills - self._deaths,
            "dodges": self._dodges,
            "shots_total": self._shots_total,
            "shots_hit": self._shots_hit,
            "accuracy": accuracy,
            "overextend_events": self._overextend_events,
            "low_hp_engage_events": self._low_hp_engage_events,
            "peek_success_events": self._peek_success_events,
            "bad_peek_events": self._bad_peek_events,
            "reward_usage_counter": dict(self._reward_usage_counter),
            "reward_usage_rates": usage_rates,
            "disabled_reward_signals": dict(self._disabled_low_usage_signals),
            "last_components": dict(self._reward_components),
        }

    def reset(self):
        """Reset all per-episode state for a new match/episode."""
        self._prev_player_hp = 100.0
        self._prev_enemy_hps = {}
        self._prev_enemy_count = 0
        self._prev_ammo = 3
        self._prev_super = False
        self._prev_pos = (0, 0)
        self._stuck_duration = 0.0
        self._stuck_window_pos = (0, 0)
        self._stuck_window_time = 0.0
        self._storm_duration = 0.0
        self._shots_fired = []
        self._damage_dealt_after_shot = {}
        self._last_frame_time = 0.0
        self._match_start_time = 0.0
        self._total_damage_dealt = 0.0
        self._total_damage_taken = 0.0
        self._kills = 0
        self._deaths = 0
        self._dodges = 0
        self._shots_total = 0
        self._shots_hit = 0
        self._kill_streak = 0
        self._best_kill_streak = 0
        self._overextend_events = 0
        self._low_hp_engage_events = 0
        self._peek_success_events = 0
        self._bad_peek_events = 0
        self._recent_bad_peek_times.clear()
        self._reward_components = {}
        self._total_episode_reward = 0.0
        self._total_episode_raw_reward = 0.0
        self._frame_count = 0

    # aliases
    def calculate(self, blackboard=None, **kwargs) -> float:
        """Calculate reward - supports both blackboard and keyword args."""
        if blackboard is not None and hasattr(blackboard, "get"):
            return self.compute_frame_reward(blackboard)

        reward = 0.0
        self._reward_components = {}

        damage_dealt = kwargs.get("damage_dealt", 0) or 0
        enemy_hp_diff = kwargs.get("enemy_hp_diff", 0) or 0
        if enemy_hp_diff > 0:
            damage_dealt = max(damage_dealt, enemy_hp_diff * 32)
        if damage_dealt > 0:
            r = self.weights["damage_dealt_per_1000"] * (damage_dealt / 1000)
            reward += r
            self._mark_component("damage_dealt_per_1000", r)
            self._total_damage_dealt += damage_dealt
            self._shots_hit += 1

        damage_taken = kwargs.get("damage_taken", 0) or 0
        hp_diff = kwargs.get("hp_diff", 0) or 0
        if hp_diff < 0:
            damage_taken = max(damage_taken, abs(hp_diff) * 32)
        if damage_taken > 0:
            r = self.weights["damage_taken_per_1000"] * (damage_taken / 1000)
            reward += r
            self._mark_component("damage_taken_per_1000", r)
            self._total_damage_taken += damage_taken

        if kwargs.get("killed", False):
            r = self.weights["kill"]
            self._kills += 1
            self._kill_streak += 1
            self._best_kill_streak = max(self._best_kill_streak, self._kill_streak)
            if self._kill_streak >= 2:
                streak_bonus = self.weights.get("kill_streak_bonus", 5.0) * (self._kill_streak - 1)
                r += streak_bonus
                self._mark_component("kill_streak_bonus", streak_bonus)
            reward += r
            self._mark_component("kill", r)

        if kwargs.get("died", False):
            r = self.weights["death"]
            self._deaths += 1
            self._kill_streak = 0
            reward += r
            self._mark_component("death", r)
            if kwargs.get("in_storm", False):
                gas_r = self.weights.get("gas_death", -25.0)
                reward += gas_r
                self._mark_component("gas_death", gas_r)

        if kwargs.get("attacked", False):
            enemy_dist = kwargs.get("closest_enemy_dist", 9999)
            if enemy_dist < 400:
                prox_reward = self.weights.get("attack_while_close", 0.4) * max(0.0, 1.0 - enemy_dist / 400)
                reward += prox_reward
                self._mark_component("attack_while_close", prox_reward)
            self._shots_total += 1

        enemy_hp_pct = kwargs.get("closest_enemy_hp_pct", 100)
        if 0 < enemy_hp_pct < 40:
            enemy_dist = kwargs.get("closest_enemy_dist", 9999)
            if enemy_dist < 500:
                chase_reward = self.weights.get("chase_low_hp_enemy", 0.3)
                reward += chase_reward
                self._mark_component("chase_low_hp_enemy", chase_reward)

        player_hp_pct = float(kwargs.get("player_hp_pct", 100) or 100)
        retreat_hp = float(kwargs.get("retreat_hp_threshold", 45) or 45)
        attack_range = float(kwargs.get("attack_range", 400) or 400)
        enemy_dist = float(kwargs.get("closest_enemy_dist", 9999) or 9999)

        if kwargs.get("attacked", False):
            if player_hp_pct <= (retreat_hp + 3) and enemy_dist < attack_range * 1.1:
                r = self.weights.get("low_hp_engage_penalty", -1.3)
                reward += r
                self._mark_component("low_hp_engage_penalty", r)
                self._low_hp_engage_events += 1

            if damage_taken > (damage_dealt * 1.2) and enemy_dist < attack_range * 0.85:
                r = self.weights.get("overextend_penalty", -0.9)
                reward += r
                self._mark_component("overextend_penalty", r)
                self._overextend_events += 1

        now = time.time()
        peek_phase = str(kwargs.get("peek_phase", "idle") or "idle").lower()
        cover_score_hint = float(kwargs.get("cover_score", 0.5) or 0.5)
        if peek_phase in ("expose", "fire") and enemy_dist < attack_range * 1.2:
            if damage_taken > max(1.0, damage_dealt * 1.15) and cover_score_hint < 0.35:
                r = self.weights.get("bad_peek_penalty", -1.6)
                reward += r
                self._mark_component("bad_peek_penalty", r)
                self._bad_peek_events += 1

                self._recent_bad_peek_times.append(now)
                while self._recent_bad_peek_times and (now - self._recent_bad_peek_times[0]) > 6.0:
                    self._recent_bad_peek_times.popleft()
                if len(self._recent_bad_peek_times) >= 3:
                    r_rep = self.weights.get("repeated_bad_peek_penalty", -2.2)
                    reward += r_rep
                    self._mark_component("repeated_bad_peek_penalty", r_rep)
            elif kwargs.get("attacked", False) and cover_score_hint >= 0.35 and damage_dealt >= damage_taken:
                r = self.weights.get("peek_success_bonus", 1.8)
                reward += r
                self._mark_component("peek_success_bonus", r)
                self._peek_success_events += 1

        self._track_usage()
        normalized_reward = self._normalize_reward(reward)
        self._reward_components["raw_reward"] = reward
        self._reward_components["normalized_reward"] = normalized_reward

        self._total_episode_raw_reward += reward
        self._total_episode_reward += normalized_reward
        self._frame_count += 1
        return normalized_reward

    def add_match_result(self, won: bool = False, trophy_delta: int = 0) -> float:
        """Add end-of-match reward."""
        result = "win" if won else "loss"
        return self.compute_match_reward(result, trophy_delta)

    def episode_summary(self, won: bool = False) -> Dict[str, Any]:
        """Alias for get_episode_summary() with optional won flag."""
        summary = self.get_episode_summary()
        summary["won"] = won
        return summary
