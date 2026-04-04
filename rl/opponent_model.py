# tracks enemy behavior over a match, classifies style, suggests counters

from __future__ import annotations

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple


# opponent Styles
OPPONENT_STYLES = {
    "aggressive": {
        "description": "Pushes forward, engages at close range, high fire rate",
        "counter": {
            "hp_retreat_mod": 1.3,     # Retreat earlier
            "dodge_chance_mod": 1.5,   # Dodge more
            "range_preference": 1.2,   # Keep more distance
            "attack_timing": "reactive",  # Attack after they attack (punish windows)
        },
    },
    "defensive": {
        "description": "Stays back, maintains range, fires at max range",
        "counter": {
            "hp_retreat_mod": 0.8,     # Can be more aggressive
            "dodge_chance_mod": 0.8,   # Less dodging needed
            "range_preference": 0.7,   # Close gap
            "attack_timing": "pressure",  # Constant pressure to force them back
        },
    },
    "passive": {
        "description": "Barely fights, avoids combat, stays near teammates",
        "counter": {
            "hp_retreat_mod": 0.6,     # Very aggressive
            "dodge_chance_mod": 0.5,   # Low dodge needed
            "range_preference": 0.5,   # Rush in
            "attack_timing": "pressure",
        },
    },
    "unpredictable": {
        "description": "Alternates between aggressive and defensive, hard to read",
        "counter": {
            "hp_retreat_mod": 1.1,     # Slightly cautious
            "dodge_chance_mod": 1.3,   # More dodging
            "range_preference": 1.0,   # Neutral range
            "attack_timing": "reactive",
        },
    },
}


# opponent Tracker (per enemy)
class EnemyProfile:
    """Tracks behavioral data for a single opponent."""

    def __init__(self):
        # Position tracking
        self.position_history: deque = deque(maxlen=100)  # (x, y, timestamp)
        self.velocity_history: deque = deque(maxlen=30)

        # Behavior metrics
        self.approach_count: int = 0     # Times enemy moved toward us
        self.retreat_count: int = 0      # Times enemy moved away
        self.strafe_count: int = 0       # Times enemy moved sideways
        self.attack_count: int = 0       # Estimated attacks (HP drops detected)
        self.idle_count: int = 0         # Frames with minimal movement

        # Distance tracking
        self.distances: deque = deque(maxlen=50)  # distances to player over time
        self.avg_engagement_distance: float = 0.0
        self.min_distance_seen: float = 9999
        self.max_distance_seen: float = 0

        # Timing
        self.last_attack_time: float = 0.0
        self.attack_intervals: deque = deque(maxlen=20)  # Time between attacks
        self.reaction_times: deque = deque(maxlen=10)     # Time to respond to our actions
        self._first_seen: float = 0.0
        self._total_visible_time: float = 0.0

        # Predicted style
        self.classified_style: str = "unknown"
        self.style_confidence: float = 0.0

        # Last known state
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_hp: float = 100
        self.last_seen_time: float = 0.0

    def update(self, pos: Tuple[float, float], hp: float,
               player_pos: Tuple[float, float], current_time: float,
               was_attack_event: bool = False):
        """Update profile with new observation."""
        if self._first_seen == 0:
            self._first_seen = current_time

        # Distance to player
        dist = math.sqrt((pos[0] - player_pos[0]) ** 2 + (pos[1] - player_pos[1]) ** 2)
        self.distances.append(dist)
        self.min_distance_seen = min(self.min_distance_seen, dist)
        self.max_distance_seen = max(self.max_distance_seen, dist)

        # Movement classification (approach/retreat/strafe)
        if self.last_pos is not None:
            prev_dist = math.sqrt(
                (self.last_pos[0] - player_pos[0]) ** 2 +
                (self.last_pos[1] - player_pos[1]) ** 2
            )
            # Direction relative to player
            dx = pos[0] - self.last_pos[0]
            dy = pos[1] - self.last_pos[1]
            move_speed = math.sqrt(dx * dx + dy * dy)

            if move_speed > 5:  # Moving meaningfully
                # Approach: getting closer to player
                if dist < prev_dist - 10:
                    self.approach_count += 1
                elif dist > prev_dist + 10:
                    self.retreat_count += 1
                else:
                    self.strafe_count += 1

                # Velocity tracking
                dt = current_time - self.last_seen_time if self.last_seen_time > 0 else 0.033
                if dt > 0:
                    vx = dx / dt
                    vy = dy / dt
                    self.velocity_history.append((vx, vy, current_time))
            else:
                self.idle_count += 1

        # Attack detection (event-driven): inferred from player damage events
        # and associated to likely threatening enemies by OpponentModel.update().
        if was_attack_event:
            attack_time_diff = current_time - self.last_attack_time
            if attack_time_diff > 0.3:  # Debounce
                self.attack_count += 1
                if self.last_attack_time > 0:
                    self.attack_intervals.append(attack_time_diff)
                self.last_attack_time = current_time

        # Update engagement distance
        if self.distances:
            self.avg_engagement_distance = sum(self.distances) / len(self.distances)

        self._total_visible_time += (current_time - self.last_seen_time) if self.last_seen_time > 0 else 0

        self.last_pos = pos
        self.last_hp = hp
        self.last_seen_time = current_time

    def classify(self) -> Tuple[str, float]:
        """Classify this enemy's playstyle based on accumulated data.

        Returns: (style_name, confidence 0-1)
        """
        total_actions = self.approach_count + self.retreat_count + self.strafe_count + self.idle_count
        if total_actions < 10:
            return "unknown", 0.0

        approach_ratio = self.approach_count / total_actions
        retreat_ratio = self.retreat_count / total_actions
        idle_ratio = self.idle_count / total_actions

        # Scoring for each style
        scores = {}

        # Aggressive: high approach, low retreat, close distances
        scores["aggressive"] = (
            approach_ratio * 3.0 +
            (1 - retreat_ratio) * 1.5 +
            (1 - self.avg_engagement_distance / 600) * 2.0 +
            min(self.attack_count / max(self._total_visible_time, 1), 1) * 2.0
        )

        # Defensive: high retreat, maintains distance, low approach
        scores["defensive"] = (
            retreat_ratio * 3.0 +
            (1 - approach_ratio) * 1.5 +
            (self.avg_engagement_distance / 600) * 2.0 +
            (1 - min(self.attack_count / max(self._total_visible_time, 1), 1)) * 1.5
        )

        # Passive: high idle, very low attack, avoids combat
        scores["passive"] = (
            idle_ratio * 4.0 +
            (1 - approach_ratio) * 2.0 +
            max(0, 1 - self.attack_count / 5) * 2.0
        )

        # Unpredictable: balanced approach/retreat, variable distances
        approach_retreat_balance = 1 - abs(approach_ratio - retreat_ratio)
        dist_variance = 0
        if len(self.distances) > 5:
            dists = list(self.distances)
            mean_d = sum(dists) / len(dists)
            dist_variance = sum((d - mean_d) ** 2 for d in dists) / len(dists)
            dist_variance = min(dist_variance / 10000, 1)  # Normalize

        scores["unpredictable"] = (
            approach_retreat_balance * 3.0 +
            dist_variance * 3.0
        )

        # Pick highest score
        best_style = max(scores, key=scores.get)
        max_score = scores[best_style]
        total_score = sum(scores.values())

        confidence = max_score / total_score if total_score > 0 else 0.0

        self.classified_style = best_style
        self.style_confidence = confidence
        return best_style, confidence


# opponent Model
class OpponentModel:
    """Manages opponent profiling across all visible enemies.

    Tracks up to 6 enemy profiles (3v3 with respawns = may see more).
    Associates detections with profiles using position proximity.
    """

    def __init__(self, max_profiles: int = 6):
        self.max_profiles = max_profiles
        self.profiles: Dict[int, EnemyProfile] = {}
        self._next_id = 0
        self._association_radius = 150  # px -- max distance to associate same enemy
        self._reclassify_interval = 3.0  # Reclassify every 3 seconds
        self._last_reclassify = 0.0

        # Global opponent tendencies (aggregated)
        self.team_style: str = "unknown"
        self.team_aggression: float = 0.5  # 0=passive, 1=aggressive

    def update(self, enemies: List[Dict], player_pos: Tuple[float, float],
               current_time: float, player_damage_taken: float = 0.0):
        """Update all opponent profiles with current enemy detections.

        """
        # Associate detected enemies with existing profiles
        used_profiles = set()
        profile_distances: Dict[int, float] = {}

        for enemy in enemies:
            epos = enemy.get("pos", (0, 0))
            ehp = enemy.get("hp", -1)
            if ehp < 0:
                ehp = 100  # Unknown HP

            # Find closest matching profile
            best_profile_id = None
            best_dist = self._association_radius

            for pid, profile in self.profiles.items():
                if pid in used_profiles:
                    continue
                if profile.last_pos is None:
                    continue
                dist = math.sqrt(
                    (epos[0] - profile.last_pos[0]) ** 2 +
                    (epos[1] - profile.last_pos[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_profile_id = pid

            if best_profile_id is not None:
                dist_to_player = math.sqrt(
                    (epos[0] - player_pos[0]) ** 2 +
                    (epos[1] - player_pos[1]) ** 2
                )
                profile_distances[best_profile_id] = dist_to_player
                self.profiles[best_profile_id].update(epos, ehp, player_pos, current_time)
                used_profiles.add(best_profile_id)
            else:
                # New enemy -- create profile
                if len(self.profiles) < self.max_profiles:
                    new_id = self._next_id
                    self._next_id += 1
                    self.profiles[new_id] = EnemyProfile()
                    dist_to_player = math.sqrt(
                        (epos[0] - player_pos[0]) ** 2 +
                        (epos[1] - player_pos[1]) ** 2
                    )
                    profile_distances[new_id] = dist_to_player
                    self.profiles[new_id].update(epos, ehp, player_pos, current_time)
                    used_profiles.add(new_id)

        # Attack-event attribution: when player took meaningful damage this frame,
        # attribute to nearest visible enemy profile(s).
        if player_damage_taken > 0 and profile_distances:
            nearest = sorted(profile_distances.items(), key=lambda kv: kv[1])
            nearest_pid, nearest_dist = nearest[0]
            # Primary attacker: nearest profile in plausible combat range.
            if nearest_dist <= 520:
                profile = self.profiles.get(nearest_pid)
                if profile is not None:
                    profile.update(
                        profile.last_pos or player_pos,
                        profile.last_hp,
                        player_pos,
                        current_time,
                        was_attack_event=True,
                    )

        # Periodic reclassification
        if current_time - self._last_reclassify > self._reclassify_interval:
            self._reclassify_all()
            self._update_team_style()
            self._last_reclassify = current_time

    def _reclassify_all(self):
        """Reclassify all opponent profiles."""
        for profile in self.profiles.values():
            profile.classify()

    def _update_team_style(self):
        """Compute aggregate team style from individual profiles."""
        if not self.profiles:
            return

        styles = [p.classified_style for p in self.profiles.values()
                   if p.classified_style != "unknown"]

        if not styles:
            return

        # Count styles
        from collections import Counter
        counts = Counter(styles)
        most_common = counts.most_common(1)[0][0]
        self.team_style = most_common

        # Compute team aggression
        aggression_scores = []
        for p in self.profiles.values():
            total = p.approach_count + p.retreat_count + p.strafe_count + p.idle_count
            if total > 0:
                aggression_scores.append(p.approach_count / total)

        if aggression_scores:
            self.team_aggression = sum(aggression_scores) / len(aggression_scores)

    def get_opponent_style(self, enemy_idx: int = 0) -> str:
        """Get the classified style of a specific enemy.

        """
        profiles = list(self.profiles.values())
        if enemy_idx < len(profiles):
            return profiles[enemy_idx].classified_style
        return "unknown"

    def get_counter_strategy(self, style: Optional[str] = None) -> Dict:
        """Get counter-strategy modifiers for an opponent style.

        Returns dict with multipliers:
            hp_retreat_mod: multiply retreat threshold
            dodge_chance_mod: multiply dodge probability
            range_preference: multiply preferred range
            attack_timing: "reactive" or "pressure"
        """
        if style is None:
            style = self.team_style

        if style in OPPONENT_STYLES:
            return OPPONENT_STYLES[style]["counter"]

        # Default: neutral
        return {
            "hp_retreat_mod": 1.0,
            "dodge_chance_mod": 1.0,
            "range_preference": 1.0,
            "attack_timing": "normal",
        }

    def get_enemy_prediction(self, enemy_idx: int = 0) -> Dict:
        """Predict enemy's next likely action based on their pattern.

        Returns dict with:
            likely_direction: (dx, dy) predicted movement direction
            attack_in_seconds: estimated time until next attack
            safety_window: seconds of relative safety
        """
        profiles = list(self.profiles.values())
        if enemy_idx >= len(profiles):
            return {"likely_direction": (0, 0), "attack_in_seconds": 999, "safety_window": 0}

        profile = profiles[enemy_idx]

        # Predict direction from recent velocity
        likely_dir = (0.0, 0.0)
        if profile.velocity_history:
            recent = list(profile.velocity_history)[-5:]
            avg_vx = sum(v[0] for v in recent) / len(recent)
            avg_vy = sum(v[1] for v in recent) / len(recent)
            speed = math.sqrt(avg_vx ** 2 + avg_vy ** 2)
            if speed > 10:
                likely_dir = (avg_vx / speed, avg_vy / speed)

        # Predict attack timing
        attack_in = 999.0
        safety_window = 0.0
        if profile.attack_intervals:
            avg_interval = sum(profile.attack_intervals) / len(profile.attack_intervals)
            time_since_last = time.time() - profile.last_attack_time
            attack_in = max(0, avg_interval - time_since_last)
            safety_window = max(0, avg_interval * 0.3)  # 30% of interval = safe window

        return {
            "likely_direction": likely_dir,
            "attack_in_seconds": attack_in,
            "safety_window": safety_window,
        }

    def get_summary(self) -> str:
        """Human-readable summary for debug overlay."""
        if not self.profiles:
            return "No opponents tracked"

        lines = [f"Team: {self.team_style} (agg={self.team_aggression:.1%})"]
        for i, (pid, p) in enumerate(self.profiles.items()):
            lines.append(f"  E{i}: {p.classified_style} ({p.style_confidence:.0%}) "
                          f"d={p.avg_engagement_distance:.0f}px atk={p.attack_count}")
        return "\n".join(lines)

    def reset(self):
        """Reset all profiles for a new match."""
        self.profiles.clear()
        self._next_id = 0
        self.team_style = "unknown"
        self.team_aggression = 0.5
        self._last_reclassify = 0.0
