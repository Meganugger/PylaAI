"""Adaptive combat tuning engine for strongest-bot-rl.

Per-brawler parameter tuning system with real hit attribution.

Features:
  - Per-brawler parameter sets: each brawler has its own tuned parameters
    stored in state["brawlers"][key]["params"]. New brawlers inherit from
    the global baseline.
  - Real hit attribution: play.py tracks enemy count deltas between frames
    after each shot. A decrease = confirmed hit/kill. Unchanged = miss.
    This replaces the old "clear-fire proxy" with ground-truth signals.
  - EMA smoothing on parameter adjustments (alpha = 0.12) to prevent noise-
    driven jumps.
  - Decay-to-baseline when a brawler hasn't been played recently, so stale
    tuning doesn't persist indefinitely (half-life ~ 40 matches total).
  - Hard-clamped parameter bounds to prevent runaway drift.
  - Atomic JSON persistence (write to temp file + rename).
  - Thread-safe state access for live dashboard reads.

Provenance:
  - Parameter schema and basic win-rate logic: from PylaAI-OP adaptive_brain.
  - Per-brawler tracking, EMA, decay, hit tracking, threading,
    atomic persistence: designed for this branch.
"""

import json
import math
import os
import tempfile
import threading
import time


ADAPTIVE_STATE_PATH = "cfg/adaptive_state.json"

# Parameter schema
# These are the tunable parameters. Each has a default, hard limits, and
# a per-update step size. The system adjusts them based on recent win-rate
# and per-brawler hit-rate data.

DEFAULTS = {
    "safe_range_multiplier": 1.0,
    "strafe_blend": 0.35,
    "strafe_interval": 1.6,
    "attack_cooldown": 0.16,
}

LIMITS = {
    "safe_range_multiplier": (0.80, 1.30),
    "strafe_blend": (0.12, 0.65),
    "strafe_interval": (0.8, 2.5),
    "attack_cooldown": (0.10, 0.24),
}

STEP = {
    "safe_range_multiplier": 0.025,
    "strafe_blend": 0.025,
    "strafe_interval": 0.06,
    "attack_cooldown": 0.004,
}

# EMA smoothing factor for parameter updates (lower = more conservative)
EMA_ALPHA = 0.12

# Decay factor per match of any brawler; pulls brawler parameters toward
# the global baseline. After ~40 matches of NOT playing a brawler, its
# params will be roughly halfway back to baseline.
DECAY_PER_MATCH = 0.017

# Minimum fires before the per-brawler hit-rate is trusted enough
# to influence pacing.
MIN_ACCURACY_SAMPLES = 12

# Rolling window size for match results (win/loss/draw history).
DEFAULT_WINDOW_SIZE = 20

# Rolling window for fire-event tracking per brawler.
FIRE_WINDOW_SIZE = 60


class AdaptiveBrain:
    """Win-rate and real hit attribution driven per-brawler parameter tuner."""

    def __init__(self, enabled=True, state_path=ADAPTIVE_STATE_PATH,
                 window_size=DEFAULT_WINDOW_SIZE):
        self.enabled = enabled
        self.state_path = state_path
        self.window_size = max(5, int(window_size))
        self._lock = threading.Lock()
        self.state = self._load()
        self._active_brawler_key = None
        # In-memory fire-event tracking (not persisted to avoid disk churn).
        # Keyed by brawler slug -> list of {"hit": bool, "time": float}
        # "hit" means the bot confirmed damage dealt (enemy count decreased
        # after the shot within the hit-check window).
        self._fire_log = {}
        self._last_save_time = time.time()

    # Public properties

    @property
    def params(self):
        """Current tuned parameters for the active brawler (or global)."""
        with self._lock:
            return dict(self._active_params_locked())

    @property
    def total_matches(self):
        with self._lock:
            return int(self.state.get("total_matches", 0))

    # Per-brawler parameter access

    def _active_params_locked(self):
        """Return the active brawler's params, falling back to global."""
        if self._active_brawler_key:
            bdata = self.state.get("brawlers", {}).get(self._active_brawler_key, {})
            bparams = bdata.get("params")
            if bparams:
                return bparams
        return self.state["params"]

    def _ensure_brawler_params(self, brawler_key):
        """Ensure per-brawler params exist, inheriting from global baseline."""
        bdata = self._ensure_brawler_data(brawler_key)
        if "params" not in bdata:
            # New brawler: inherit current global params
            bdata["params"] = dict(self.state["params"])
        return bdata["params"]

    # Match result recording

    def record_result(self, result, brawler=None):
        """Record a match result and update parameters.

        Args:
            result: "victory", "defeat", "draw", "1st", "2nd", "3rd", "4th"
            brawler: current brawler name (for per-brawler tracking)
        """
        if not self.enabled:
            return

        with self._lock:
            bucket = self._result_to_bucket(result)
            now = time.time()
            self.state["history"].append({"bucket": bucket, "time": now})
            self.state["history"] = self.state["history"][-self.window_size:]

            win_rate = self._win_rate_locked()

            # Per-brawler hit-rate signal
            brawler_key = self._brawler_key(brawler)
            brawler_hit_rate = None
            if brawler_key:
                brawler_hit_rate = self._get_brawler_hit_rate(brawler_key)
                self._decay_stale_brawlers(brawler_key)
                bdata = self._ensure_brawler_data(brawler_key)
                bdata["matches"] = bdata.get("matches", 0) + 1
                bdata["last_played"] = now
                # Ensure this brawler has its own param set
                bparams = self._ensure_brawler_params(brawler_key)
                self._adjust(win_rate, brawler_hit_rate, bparams)
            else:
                # No brawler context: adjust global params
                self._adjust(win_rate, brawler_hit_rate, self.state["params"])

            self.state["last_win_rate"] = round(win_rate, 3)
            self.state["total_matches"] = int(self.state.get("total_matches", 0)) + 1
            self._save_locked()

        hit_str = f"{brawler_hit_rate:.1%}" if brawler_hit_rate is not None else "n/a"
        active_p = self._active_params_locked() if brawler_key else self.state["params"]
        print(
            f"Adaptive brain: result={result} win_rate={win_rate:.1%} "
            f"hit_rate={hit_str} "
            f"safe_mult={active_p['safe_range_multiplier']:.3f} "
            f"strafe={active_p['strafe_blend']:.3f} "
            f"brawler={brawler_key or 'global'}"
        )

    # Fire-event tracking (real hit attribution)

    def record_fire(self, brawler, hit):
        """Record a fire event with real hit attribution.

        Args:
            brawler: current brawler name
            hit: True if the shot confirmed damage (enemy count decreased
                 within the hit-check window after firing). False if miss.
                 None to skip recording.
        """
        if not self.enabled or hit is None:
            return

        brawler_key = self._brawler_key(brawler)
        if not brawler_key:
            return

        with self._lock:
            log = self._fire_log.setdefault(brawler_key, [])
            log.append({"hit": bool(hit), "time": time.time()})
            # Keep only recent fires
            if len(log) > FIRE_WINDOW_SIZE:
                self._fire_log[brawler_key] = log[-FIRE_WINDOW_SIZE:]

    def get_fire_stats(self, brawler):
        """Return (total_fires, hits, hit_rate_or_none) for a brawler.

        hit_rate is the fraction of fires that confirmed damage, or None if
        fewer than MIN_ACCURACY_SAMPLES fires have been recorded.
        """
        brawler_key = self._brawler_key(brawler)
        if not brawler_key:
            return 0, 0, None

        with self._lock:
            log = self._fire_log.get(brawler_key, [])
            if not log:
                return 0, 0, None
            total = len(log)
            hits = sum(1 for s in log if s["hit"])
            rate = hits / total if total >= MIN_ACCURACY_SAMPLES else None
            return total, hits, rate

    # Apply parameters to play instance

    def apply_to_play(self, play_instance, brawler=None):
        """Apply current tuned parameters to a Play instance.

        Uses per-brawler params if available, falling back to global.
        Also stores a reference to this brain on the play instance so that
        play.py can call record_fire() during combat.
        """
        brawler_key = self._brawler_key(brawler)
        with self._lock:
            self._active_brawler_key = brawler_key
            if brawler_key:
                bdata = self.state.get("brawlers", {}).get(brawler_key, {})
                bparams = bdata.get("params")
                params = bparams if bparams else self.state["params"]
            else:
                params = self.state["params"]
            params = dict(params) if self.enabled else dict(DEFAULTS)

        play_instance.adaptive_safe_range_multiplier = params["safe_range_multiplier"]
        play_instance._adaptive_brain = self
        if hasattr(play_instance, "strafe_blend"):
            play_instance.strafe_blend = params["strafe_blend"]
        if hasattr(play_instance, "strafe_interval"):
            play_instance.strafe_interval = params["strafe_interval"]
        if hasattr(play_instance, "attack_cooldown"):
            play_instance.attack_cooldown = params["attack_cooldown"]

    # Live dashboard data

    def get_live_state(self, current_brawler=None):
        """Return a snapshot dict suitable for the dashboard live-data feed."""
        with self._lock:
            brawler_key = self._brawler_key(current_brawler)
            self._active_brawler_key = brawler_key

            # Get per-brawler params if available, otherwise global
            global_params = dict(self.state["params"])
            bdata = self.state.get("brawlers", {}).get(brawler_key, {}) if brawler_key else {}
            bparams = bdata.get("params")
            active_params = dict(bparams) if bparams else global_params

            # Fire-event stats (real hits)
            fire_log = self._fire_log.get(brawler_key, []) if brawler_key else []
            total_fires = len(fire_log)
            hit_fires = sum(1 for s in fire_log if s["hit"])

            # Parameter offsets from baseline (global defaults)
            offsets = {}
            clamped = {}
            for key, default_val in DEFAULTS.items():
                current_val = active_params.get(key, default_val)
                offsets[key] = round(current_val - default_val, 4)
                lo, hi = LIMITS[key]
                clamped[key] = (current_val <= lo + 0.001) or (current_val >= hi - 0.001)

            hit_rate = hit_fires / total_fires if total_fires >= MIN_ACCURACY_SAMPLES else None
            sample_threshold_met = total_fires >= MIN_ACCURACY_SAMPLES

            # Is the system currently updating or in a dead zone?
            win_rate = self._win_rate_locked()
            is_updating = win_rate > 0.62 or win_rate < 0.35
            is_decaying = bool(brawler_key and bdata.get("matches", 0) > 0)
            has_brawler_params = bparams is not None

            return {
                "adaptive_enabled": self.enabled,
                "total_matches": int(self.state.get("total_matches", 0)),
                "window_win_rate": round(win_rate, 3),
                "is_updating": is_updating,
                "is_decaying": is_decaying,
                "brawler": brawler_key or "",
                "brawler_matches": int(bdata.get("matches", 0)),
                "has_brawler_params": has_brawler_params,
                "total_fires": total_fires,
                "hit_fires": hit_fires,
                "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
                "sample_threshold_met": sample_threshold_met,
                "min_accuracy_samples": MIN_ACCURACY_SAMPLES,
                "params": active_params,
                "global_params": global_params,
                "defaults": dict(DEFAULTS),
                "offsets": offsets,
                "clamped": clamped,
                "limits": {k: list(v) for k, v in LIMITS.items()},
            }

    # Summary

    def win_rate(self):
        with self._lock:
            return self._win_rate_locked()

    def summary(self):
        with self._lock:
            brawler_count = len(self.state.get("brawlers", {}))
            brawlers_with_params = sum(
                1 for b in self.state.get("brawlers", {}).values()
                if b.get("params")
            )
            return (
                f"Adaptive brain: enabled={self.enabled}, "
                f"matches={self.state.get('total_matches', 0)}, "
                f"win_rate={self.state.get('last_win_rate', 'n/a')}, "
                f"brawlers_tracked={brawler_count}, "
                f"brawlers_with_params={brawlers_with_params}, "
                f"global_params={self.state['params']}"
            )

    # Internal

    @staticmethod
    def _brawler_key(brawler):
        if not brawler:
            return None
        return str(brawler).strip().lower().replace(" ", "").replace("-", "")

    @staticmethod
    def _result_to_bucket(result):
        normalized = str(result).lower().strip()
        if normalized in ("1st", "2nd", "victory"):
            return "win"
        if normalized in ("4th", "defeat"):
            return "loss"
        return "draw"

    @staticmethod
    def _clamp(key, value):
        lo, hi = LIMITS[key]
        return max(lo, min(hi, value))

    def _win_rate_locked(self):
        """Calculate win rate from recent history. Must hold self._lock."""
        history = self.state.get("history", [])
        wins = sum(1 for item in history if item.get("bucket") == "win")
        losses = sum(1 for item in history if item.get("bucket") == "loss")
        total = wins + losses
        return 0.5 if total <= 0 else wins / total

    def _get_brawler_hit_rate(self, brawler_key):
        """Return hit-rate or None if not enough samples.

        Uses real hit attribution from enemy count deltas.
        """
        log = self._fire_log.get(brawler_key, [])
        if len(log) < MIN_ACCURACY_SAMPLES:
            return None
        hits = sum(1 for s in log if s["hit"])
        return hits / len(log)

    def _ensure_brawler_data(self, brawler_key):
        """Ensure per-brawler metadata exists."""
        brawlers = self.state.setdefault("brawlers", {})
        if brawler_key not in brawlers:
            brawlers[brawler_key] = {
                "matches": 0,
                "last_played": 0.0,
            }
        return brawlers[brawler_key]

    def _decay_stale_brawlers(self, current_brawler_key):
        """Decay inactive brawler params toward the global baseline."""
        global_params = self.state["params"]
        brawlers = self.state.get("brawlers", {})
        for bkey, bdata in brawlers.items():
            if bkey == current_brawler_key:
                continue
            bparams = bdata.get("params")
            if not bparams:
                continue
            # Pull this inactive brawler's params toward the global baseline
            for key, global_val in global_params.items():
                current = bparams.get(key, global_val)
                diff = current - global_val
                if abs(diff) > 0.001:
                    bparams[key] = self._clamp(key, current - diff * DECAY_PER_MATCH)

    def _adjust(self, win_rate, brawler_hit_rate=None, params=None):
        """Adjust parameters based on win rate and hit-rate using EMA.

        win_rate: primary signal (real match outcomes)
        brawler_hit_rate: secondary signal from real hit attribution
            (fraction of fires that confirmed damage via enemy count delta).
        params: the parameter dict to modify (brawler-specific or global)
        """
        if params is None:
            params = self.state["params"]

        # -- Win-rate driven adjustments (primary, real signal) -------
        # High win rate -> push aggressive (lower safe range, more strafe,
        # faster attacks).
        # Low win rate -> pull defensive (higher safe range, less strafe,
        # slower attacks).

        if win_rate > 0.62:
            targets = {
                "safe_range_multiplier": params["safe_range_multiplier"] - STEP["safe_range_multiplier"],
                "strafe_blend": params["strafe_blend"] + STEP["strafe_blend"],
                "strafe_interval": params["strafe_interval"] - STEP["strafe_interval"],
                "attack_cooldown": params["attack_cooldown"] - STEP["attack_cooldown"],
            }
        elif win_rate < 0.35:
            targets = {
                "safe_range_multiplier": params["safe_range_multiplier"] + STEP["safe_range_multiplier"],
                "strafe_blend": params["strafe_blend"] - STEP["strafe_blend"],
                "strafe_interval": params["strafe_interval"] + STEP["strafe_interval"],
                "attack_cooldown": params["attack_cooldown"] + STEP["attack_cooldown"],
            }
        else:
            # Neutral zone: no win-rate driven adjustment, just EMA hold
            targets = dict(params)

        # -- Hit-rate adjustment (secondary, real signal) ----
        # Real hit attribution: hit_rate measures confirmed damage from
        # enemy count deltas after each shot.
        # High hit rate (>0.40) = good aim -> fire faster.
        # Low hit rate (<0.15) = poor accuracy -> pace shots more.
        if brawler_hit_rate is not None:
            if brawler_hit_rate > 0.40:
                # Good accuracy: tighten attack cooldown slightly
                targets["attack_cooldown"] = min(
                    targets["attack_cooldown"],
                    params["attack_cooldown"] - STEP["attack_cooldown"] * 0.5,
                )
            elif brawler_hit_rate < 0.15:
                # Poor accuracy: increase cooldown to conserve ammo
                targets["attack_cooldown"] = max(
                    targets["attack_cooldown"],
                    params["attack_cooldown"] + STEP["attack_cooldown"] * 0.5,
                )

        # Apply EMA smoothing and clamp
        for key in DEFAULTS:
            raw_target = targets.get(key, params[key])
            smoothed = params[key] + EMA_ALPHA * (raw_target - params[key])
            params[key] = self._clamp(key, smoothed)

    def _load(self):
        """Load state from disk, merging with defaults for forward compat."""
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data.setdefault("history", [])
                data.setdefault("total_matches", 0)
                data.setdefault("last_win_rate", None)
                data.setdefault("brawlers", {})
                params = data.setdefault("params", {})
                for key, value in DEFAULTS.items():
                    params.setdefault(key, value)
                    params[key] = self._clamp(key, float(params[key]))
                # Migrate per-brawler params
                for bkey, bdata in data.get("brawlers", {}).items():
                    if "params" in bdata:
                        for key, value in DEFAULTS.items():
                            bdata["params"].setdefault(key, value)
                            bdata["params"][key] = self._clamp(
                                key, float(bdata["params"][key])
                            )
                return data
            except Exception as e:
                print(f"Adaptive brain: could not load state ({e}), starting fresh.")
        return {
            "params": dict(DEFAULTS),
            "history": [],
            "total_matches": 0,
            "last_win_rate": None,
            "brawlers": {},
        }

    def _save_locked(self):
        """Atomic save: write to temp file then rename."""
        try:
            folder = os.path.dirname(self.state_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                suffix=".tmp",
                dir=folder or ".",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=2)
                # Atomic rename (Windows: os.replace is atomic within same volume)
                os.replace(tmp_path, self.state_path)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            self._last_save_time = time.time()
        except Exception as e:
            print(f"Adaptive brain: could not save state: {e}")
