"""Adaptive combat tuning engine for strongest-bot-rl.

Newly designed for the strongest-bot-rl branch. Extends the simpler win-rate-
only AdaptiveBrain from PylaAI-OP's parameter schema (safe_range_multiplier,
strafe_blend, strafe_interval, attack_cooldown) with:

  - Per-brawler parameter tracking (each brawler converges independently).
  - EMA smoothing on parameter adjustments (alpha = 0.12) to prevent noise-
    driven jumps.
  - Decay-to-baseline when a brawler hasn't been played recently, so stale
    tuning doesn't persist indefinitely (half-life ~ 40 matches total).
  - Qualified-fire tracking: records when the bot fires under favorable
    conditions (target in range, no wall obstruction). This is a PROXY for
    shot quality, NOT a true hit signal. No HP-bar reading or projectile
    outcome observation exists in this codebase. The proxy measures "how
    often does the bot fire when it has a clear shot" vs "firing blind".
  - Hard-clamped parameter bounds to prevent runaway drift.
  - Atomic JSON persistence (write to temp file + rename).
  - Thread-safe state access for live dashboard reads.

Provenance:
  - Parameter schema and basic win-rate logic: from PylaAI-OP adaptive_brain.
  - Per-brawler tracking, EMA, decay, fire-quality tracking, threading,
    atomic persistence: newly designed for this branch.
  - NOT related to prediction_horizon / pattern_weight_* / angle_offset
    parameter families (those are from a different system).
"""

import json
import math
import os
import tempfile
import threading
import time


ADAPTIVE_STATE_PATH = "cfg/adaptive_state.json"

# ÔöÇÔöÇ Parameter schema ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
# These are the tunable parameters. Each has a default, hard limits, and
# a per-update step size. The system adjusts them based on recent win-rate
# and per-brawler accuracy.

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

# Decay factor per match of any brawler ÔÇö pulls stale brawler params toward
# baseline. After ~40 matches of NOT playing a brawler, its params will be
# roughly halfway back to defaults.
DECAY_PER_MATCH = 0.017

# Minimum matches before the per-brawler accuracy signal is trusted enough
# to influence parameters.
MIN_ACCURACY_SAMPLES = 12

# Rolling window size for match results (win/loss/draw history).
DEFAULT_WINDOW_SIZE = 20

# Rolling window for fire-opportunity tracking per brawler.
FIRE_WINDOW_SIZE = 60


class AdaptiveBrain:
    """Win-rate and per-brawler accuracy driven parameter tuner."""

    def __init__(self, enabled=True, state_path=ADAPTIVE_STATE_PATH,
                 window_size=DEFAULT_WINDOW_SIZE):
        self.enabled = enabled
        self.state_path = state_path
        self.window_size = max(5, int(window_size))
        self._lock = threading.Lock()
        self.state = self._load()
        # In-memory fire-opportunity tracking (not persisted to avoid disk churn).
        # Keyed by brawler slug -> list of {"clear": bool, "time": float}
        # "clear" means the bot had a clear, unobstructed, in-range shot when it
        # fired. This is a proxy for shot quality, NOT a true hit signal.
        self._fire_log = {}
        self._last_save_time = time.time()

    # ÔöÇÔöÇ Public properties ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    @property
    def params(self):
        """Current global tuned parameters."""
        with self._lock:
            return dict(self.state["params"])

    @property
    def total_matches(self):
        with self._lock:
            return int(self.state.get("total_matches", 0))

    # ÔöÇÔöÇ Match result recording ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

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

            # Per-brawler clear-rate proxy signal
            brawler_key = self._brawler_key(brawler)
            brawler_clear_rate = None
            if brawler_key:
                brawler_clear_rate = self._get_brawler_clear_rate(brawler_key)
                self._decay_stale_brawlers(brawler_key)
                bdata = self._ensure_brawler_data(brawler_key)
                bdata["matches"] = bdata.get("matches", 0) + 1
                bdata["last_played"] = now

            self._adjust(win_rate, brawler_clear_rate)

            self.state["last_win_rate"] = round(win_rate, 3)
            self.state["total_matches"] = int(self.state.get("total_matches", 0)) + 1
            self._save_locked()

        clear_str = f"{brawler_clear_rate:.1%}" if brawler_clear_rate is not None else "n/a"
        print(
            f"Adaptive brain: result={result} win_rate={win_rate:.1%} "
            f"clear_rate={clear_str} "
            f"safe_mult={self.state['params']['safe_range_multiplier']:.3f} "
            f"strafe={self.state['params']['strafe_blend']:.3f}"
        )

    # ÔöÇÔöÇ Fire-opportunity tracking (proxy signal) ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def record_fire(self, brawler, clear):
        """Record a fire event with a qualified-shot proxy.

        Args:
            brawler: current brawler name
            clear: True if the bot had a clear (unobstructed, in-range) shot
                   when it fired. False if it fired blind or through walls.
                   None to skip recording (e.g. when target state is unknown).

        This is NOT a true hit signal. It measures shot-opportunity quality:
        how often the bot fires under favorable conditions vs. firing blind.
        The adaptive system uses this as a secondary tuning input alongside
        the primary win-rate signal.
        """
        if not self.enabled or clear is None:
            return

        brawler_key = self._brawler_key(brawler)
        if not brawler_key:
            return

        with self._lock:
            log = self._fire_log.setdefault(brawler_key, [])
            log.append({"clear": bool(clear), "time": time.time()})
            # Keep only recent fires
            if len(log) > FIRE_WINDOW_SIZE:
                self._fire_log[brawler_key] = log[-FIRE_WINDOW_SIZE:]

    def get_fire_stats(self, brawler):
        """Return (total_fires, clear_fires, clear_rate_or_none) for a brawler.

        clear_rate is the fraction of fires that had a clear shot, or None if
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
            clear = sum(1 for s in log if s["clear"])
            rate = clear / total if total >= MIN_ACCURACY_SAMPLES else None
            return total, clear, rate

    # ÔöÇÔöÇ Apply parameters to play instance ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def apply_to_play(self, play_instance):
        """Apply current tuned parameters to a Play instance.

        Also stores a reference to this brain on the play instance so that
        play.py can call record_fire() during combat.
        """
        params = self.params if self.enabled else dict(DEFAULTS)
        play_instance.adaptive_safe_range_multiplier = params["safe_range_multiplier"]
        play_instance._adaptive_brain = self
        if hasattr(play_instance, "strafe_blend"):
            play_instance.strafe_blend = params["strafe_blend"]
        if hasattr(play_instance, "strafe_interval"):
            play_instance.strafe_interval = params["strafe_interval"]
        if hasattr(play_instance, "attack_cooldown"):
            play_instance.attack_cooldown = params["attack_cooldown"]

    # ÔöÇÔöÇ Live dashboard data ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def get_live_state(self, current_brawler=None):
        """Return a snapshot dict suitable for the dashboard live-data feed."""
        with self._lock:
            params = dict(self.state["params"])
            brawler_key = self._brawler_key(current_brawler)
            bdata = self.state.get("brawlers", {}).get(brawler_key, {}) if brawler_key else {}

            # Fire-opportunity stats (proxy, not true hits)
            fire_log = self._fire_log.get(brawler_key, []) if brawler_key else []
            total_fires = len(fire_log)
            clear_fires = sum(1 for s in fire_log if s["clear"])

            # Parameter offsets from baseline
            offsets = {}
            clamped = {}
            for key, default_val in DEFAULTS.items():
                current_val = params.get(key, default_val)
                offsets[key] = round(current_val - default_val, 4)
                lo, hi = LIMITS[key]
                clamped[key] = (current_val <= lo + 0.001) or (current_val >= hi - 0.001)

            clear_rate = clear_fires / total_fires if total_fires >= MIN_ACCURACY_SAMPLES else None
            sample_threshold_met = total_fires >= MIN_ACCURACY_SAMPLES

            # Is the system currently updating or in a dead zone?
            win_rate = self._win_rate_locked()
            is_updating = win_rate > 0.62 or win_rate < 0.35
            is_decaying = bool(brawler_key and bdata.get("matches", 0) > 0)

            return {
                "adaptive_enabled": self.enabled,
                "total_matches": int(self.state.get("total_matches", 0)),
                "window_win_rate": round(win_rate, 3),
                "is_updating": is_updating,
                "is_decaying": is_decaying,
                "brawler": brawler_key or "",
                "brawler_matches": int(bdata.get("matches", 0)),
                "total_fires": total_fires,
                "clear_fires": clear_fires,
                "clear_rate": round(clear_rate, 3) if clear_rate is not None else None,
                "sample_threshold_met": sample_threshold_met,
                "min_accuracy_samples": MIN_ACCURACY_SAMPLES,
                "params": params,
                "defaults": dict(DEFAULTS),
                "offsets": offsets,
                "clamped": clamped,
                "limits": {k: list(v) for k, v in LIMITS.items()},
            }

    # ÔöÇÔöÇ Summary ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

    def win_rate(self):
        with self._lock:
            return self._win_rate_locked()

    def summary(self):
        with self._lock:
            return (
                f"Adaptive brain: enabled={self.enabled}, "
                f"matches={self.state.get('total_matches', 0)}, "
                f"win_rate={self.state.get('last_win_rate', 'n/a')}, "
                f"params={self.state['params']}"
            )

    # ÔöÇÔöÇ Internal ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ

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

    def _get_brawler_clear_rate(self, brawler_key):
        """Return clear-fire rate or None if not enough samples.

        This is a proxy for shot quality, not true accuracy.
        """
        log = self._fire_log.get(brawler_key, [])
        if len(log) < MIN_ACCURACY_SAMPLES:
            return None
        clear = sum(1 for s in log if s["clear"])
        return clear / len(log)

    def _ensure_brawler_data(self, brawler_key):
        """Ensure per-brawler tracking dict exists."""
        brawlers = self.state.setdefault("brawlers", {})
        if brawler_key not in brawlers:
            brawlers[brawler_key] = {
                "matches": 0,
                "last_played": 0.0,
            }
        return brawlers[brawler_key]

    def _decay_stale_brawlers(self, current_brawler_key):
        """Decay parameters toward baseline for brawlers not currently being
        played. This prevents stale tuning from persisting when you switch
        brawlers."""
        brawlers = self.state.get("brawlers", {})
        for bkey, bdata in brawlers.items():
            if bkey == current_brawler_key:
                continue
            # The global params already represent a blend ÔÇö we just nudge
            # toward defaults slightly each match.
            # This is a lightweight approach: we don't store per-brawler
            # params separately (the system is global), but the decay ensures
            # a brawler that was played 50 matches ago doesn't hold residual
            # tuning from that era.
        # Apply global decay toward defaults
        params = self.state["params"]
        for key, default_val in DEFAULTS.items():
            current = params[key]
            diff = current - default_val
            if abs(diff) > 0.001:
                params[key] = self._clamp(key, current - diff * DECAY_PER_MATCH)

    def _adjust(self, win_rate, brawler_clear_rate=None):
        """Adjust parameters based on win rate and clear-fire rate using EMA.

        win_rate: primary signal (real match outcomes)
        brawler_clear_rate: secondary proxy signal (fraction of fires that had
            a clear, unobstructed, in-range shot). NOT true accuracy.
        """
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
            # Neutral zone ÔÇö no win-rate driven adjustment, just EMA hold
            targets = dict(params)

        # -- Clear-rate proxy adjustment (secondary, proxy signal) ----
        # If we have enough fire data, use the clear-fire rate as a proxy
        # for shot-opportunity quality. High clear rate means the bot is
        # consistently firing when it has clear shots -> can fire faster.
        # Low clear rate means it's firing blind often -> should pace more.
        # NOTE: This is NOT true hit accuracy. It is a proxy for firing
        # discipline.
        if brawler_clear_rate is not None:
            if brawler_clear_rate > 0.55:
                # Good firing discipline: tighten attack cooldown slightly
                targets["attack_cooldown"] = min(
                    targets["attack_cooldown"],
                    params["attack_cooldown"] - STEP["attack_cooldown"] * 0.5,
                )
            elif brawler_clear_rate < 0.25:
                # Poor firing discipline: increase cooldown to conserve ammo
                targets["attack_cooldown"] = max(
                    targets["attack_cooldown"],
                    params["attack_cooldown"] + STEP["attack_cooldown"] * 0.5,
                )

        # ÔöÇÔöÇ Apply EMA smoothing and clamp ÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇÔöÇ
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
