# reads HP bars from screen using HSV masks, handles per-entity smoothing

from __future__ import annotations

import time
import traceback
import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple, Dict


class HPEstimator:
    """Accurate HP bar reader with temporal smoothing and confidence."""

    # morphology kernels (created once)
    _KERN_CLOSE = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))   # close 1-2px horizontal gaps
    _KERN_OPEN  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))   # remove salt noise

    def __init__(self, smoothing_frames: int = 8):
        self._smoothing_frames = smoothing_frames
        # Per-entity rolling HP values: key = entity_key, value = deque of (hp, confidence, ts)
        self._history: Dict[str, deque] = {}
        # Track max bar width seen per entity for stable 100% reference
        self._max_bar_widths: Dict[str, deque] = {}  # deque of recent widths
        # Track last smoothed HP per entity (for fast-descent logic)
        self._last_smoothed: Dict[str, int] = {}
        # Last confident reading per entity
        self._last_confident: Dict[str, Tuple[int, float]] = {}  # (hp, timestamp)
        # Debug info from last _raw_estimate (search region, rows found, etc.)
        self._last_raw_debug: Dict[str, dict] = {}
        # --- LEN-priority tracking ---
        # Store last successful LEN reading per entity: (hp, conf, timestamp)
        self._last_len_hp: Dict[str, Tuple[int, float, float]] = {}
        # Count consecutive LEN failures per entity
        self._len_miss_count: Dict[str, int] = {}
        # --- Damage-model HP (stable value anchored to damage events) ---
        self._model_hp: Dict[str, float] = {}           # modeled HP 0-100
        self._model_last_hit: Dict[str, float] = {}     # timestamp of last confirmed damage
        self._model_last_update: Dict[str, float] = {}  # timestamp of last model update
        self._consecutive_low: Dict[str, int] = {}      # frames where visual < model
        self._consecutive_high: Dict[str, int] = {}     # frames where visual > model (regen/noise)
        self._raw_damage_streak: Dict[str, int] = {}   # consecutive RAW-indicates-damage frames

    # --- pUBLIC API ---

    _hp_dbg_ctr = 0

    def estimate(self, frame, bbox, is_player: bool = False,
                 entity_key: str = None) -> Tuple[int, float]:
        """Estimate HP percentage with confidence score.

        Architecture:  DAMAGE MODEL (authoritative) ← fed ONLY by LEN readings
          - LEN succeeds (~30% of frames): update model directly
          - LEN fails: return the model's current value unchanged
          - RAW / HOLD: shown in debug only, never affect the model
          - Respawn: register_respawn() forces model to 100%

        This makes HP completely immune to RAW noise.
        """
        # --- Step 1: Try LEN method ---
        raw_hp, raw_conf, bar_width = self._length_based_estimate(frame, bbox, is_player)
        method = "LEN"
        len_succeeded = False
        len_hp = -1  # The LEN-derived HP value (if available)

        # Sentinel -2 = valid health bar measured, need max-width conversion
        if raw_hp == -2 and entity_key:
            health_length = bar_width
            # Reject impossibly wide bars (> 150% of character bbox = env glow)
            # HP bars in Brawl Stars can easily be 100-140% of char_w
            char_w = int(bbox[2]) - int(bbox[0])
            if health_length > char_w * 1.50:
                raw_hp = -1   # Reject
                raw_conf = 0.0
            else:
                if entity_key not in self._max_bar_widths:
                    self._max_bar_widths[entity_key] = deque(maxlen=40)
                # Only add to max-width reference if this measurement is
                # close to the current maximum (within 85%).  When the
                # player is at low HP the coloured bar is short — storing
                # those values would drag the reference DOWN and make
                # every future reading appear as ~100%.  By filtering,
                # the reference stays anchored to the true 100%-HP bar
                # width observed at match start / after respawn.
                cur_max = max(self._max_bar_widths[entity_key]) if self._max_bar_widths[entity_key] else 0
                if cur_max == 0 or health_length >= cur_max * 0.85:
                    self._max_bar_widths[entity_key].append(health_length)
                # 90th-percentile reference (rejects outlier max values)
                if self._max_bar_widths[entity_key]:
                    sorted_w = sorted(self._max_bar_widths[entity_key])
                    ref_len = sorted_w[max(0, int(len(sorted_w) * 0.9) - 1)]
                else:
                    ref_len = 0
                if ref_len >= 15:
                    raw_hp = int((health_length / ref_len) * 100)
                    raw_hp = max(1, min(100, raw_hp))
                    bar_width = health_length
                else:
                    raw_hp = 100
                    raw_conf = 0.4
                len_succeeded = True
                len_hp = raw_hp
        elif raw_hp == -2:
            raw_hp = 100
            raw_conf = 0.4
            len_succeeded = True
            len_hp = 100

        # Store successful LEN reading for HOLD fallback (debug/display only)
        if len_succeeded and entity_key:
            self._last_len_hp[entity_key] = (raw_hp, raw_conf, time.time())
            self._len_miss_count[entity_key] = 0

        # --- For debug display: still compute HOLD/RAW (but they don't affect output) ---
        display_hp = raw_hp
        display_conf = raw_conf
        if raw_hp < 0 and entity_key:
            self._len_miss_count[entity_key] = self._len_miss_count.get(entity_key, 0) + 1
            misses = self._len_miss_count[entity_key]
            if misses < 30 and entity_key in self._last_len_hp:
                held_hp, held_conf, held_ts = self._last_len_hp[entity_key]
                age = time.time() - held_ts
                if age < 1.5:
                    display_hp = held_hp
                    display_conf = max(0.2, held_conf - age * 0.15)
                    method = "HOLD"
        if display_hp < 0:
            display_hp, display_conf, bar_width = self._raw_estimate(frame, bbox, is_player)
            method = "RAW"
            display_conf *= 0.3

        # periodic debug (every 5 calls)
        HPEstimator._hp_dbg_ctr += 1
        if HPEstimator._hp_dbg_ctr % 5 == 0:
            tag = "P" if is_player else "E"
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            ref_w = 0
            if entity_key and entity_key in self._max_bar_widths:
                sw = sorted(self._max_bar_widths[entity_key])
                ref_w = sw[max(0, int(len(sw) * 0.9) - 1)]
            mdl = int(self._model_hp.get(entity_key, 100)) if entity_key else -1
            print(f"[HP] {tag} {method} hp={display_hp} conf={display_conf:.2f} bw={bar_width} ref={ref_w} mdl={mdl} bbox={x2-x1}x{y2-y1}")

        # === DAMAGE MODEL is the SOLE AUTHORITY for returned HP ===
        if entity_key:
            if len_succeeded:
                # LEN fired → feed into damage model
                self._raw_damage_streak[entity_key] = 0
                final_hp = self._apply_damage_model(entity_key, len_hp, raw_conf)
            elif method == "RAW" and display_hp > 0:
                # RAW fired but LEN didn't – track damage streaks
                model_val = self._model_hp.get(entity_key, 100.0)
                if display_hp < model_val - 10:
                    self._raw_damage_streak[entity_key] = self._raw_damage_streak.get(entity_key, 0) + 1
                else:
                    self._raw_damage_streak[entity_key] = 0

                if self._raw_damage_streak.get(entity_key, 0) >= 3:
                    # 3+ consecutive RAW readings show significant damage → trust it
                    final_hp = self._apply_damage_model(entity_key, display_hp, display_conf)
                    self._raw_damage_streak[entity_key] = 0
                else:
                    final_hp = self._apply_damage_model_tick(entity_key)
            else:
                # Neither LEN nor RAW → passive regen/decay only
                self._raw_damage_streak[entity_key] = 0
                final_hp = self._apply_damage_model_tick(entity_key)

            self._last_smoothed[entity_key] = final_hp
            self._last_confident[entity_key] = (final_hp, time.time())
            return final_hp, 0.8 if len_succeeded else 0.5

        # No entity key - return raw best-effort
        return max(1, display_hp) if display_hp > 0 else 100, 0.3
    
    def register_respawn(self, entity_key: str):
        """Called when entity respawns — force HP to 100% and clear history."""
        self._model_hp[entity_key] = 100.0
        self._model_last_hit[entity_key] = time.time()   # anchor to now so regen doesn't fire early
        self._model_last_update[entity_key] = time.time()
        self._consecutive_low[entity_key] = 0
        self._consecutive_high[entity_key] = 0
        self._history.pop(entity_key, None)
        self._max_bar_widths.pop(entity_key, None)
        self._last_smoothed[entity_key] = 100
        self._last_confident[entity_key] = (100, time.time())
        self._last_len_hp.pop(entity_key, None)
        self._len_miss_count[entity_key] = 0
        self._raw_damage_streak.pop(entity_key, None)
        print(f"[HP-MODEL] {entity_key} respawned -> HP=100%")

    def _apply_damage_model(self, entity_key: str, len_hp: int, len_conf: float) -> int:
        """LEN-only damage model: update model HP based on a trusted LEN reading.

        Called ONLY when LEN method succeeds (~30% of frames).
        LEN readings are accurate (conf~1.0), so the logic is simpler:

        Rules:
          - 2 consecutive LEN lows (diff < -8) → confirm damage, snap down
          - 2 consecutive LEN highs (len_hp > model+5) → snap UP immediately
            (catches regen, healer, respawn edge cases)
          - Within ±8 of model → gently blend toward LEN (0.3 lerp)
          - Respawn: register_respawn() forces 100% externally
        """
        now = time.time()

        # Initialize on first call — assume full HP
        if entity_key not in self._model_hp:
            self._model_hp[entity_key] = 100.0
            self._model_last_hit[entity_key] = now   # use current time, NOT 0 (epoch)
            self._model_last_update[entity_key] = now
            self._consecutive_low[entity_key] = 0
            self._consecutive_high[entity_key] = 0
            return 100

        model_hp = self._model_hp[entity_key]
        diff = len_hp - model_hp   # negative = LEN reads lower

        # === SNAP UP: LEN consistently reads higher → accept recovery ===
        if diff > 5:
            ch = self._consecutive_high.get(entity_key, 0) + 1
            self._consecutive_high[entity_key] = ch
            if ch >= 2:
                # 2 consecutive LEN highs → real regen
                model_hp = float(len_hp)
                self._consecutive_high[entity_key] = 0
                self._consecutive_low[entity_key] = 0
        else:
            self._consecutive_high[entity_key] = 0

        # === DAMAGE: LEN reads significantly lower → accept drop ===
        # Require 2 consecutive LEN lows to confirm real damage.
        # Single LEN readings can be noisy (bar partially occluded,
        # glow effects, map color bleed) so one reading alone is not
        # reliable enough.  The threshold is -5 (was -8, too strict).
        if diff < -5:
            cl = self._consecutive_low.get(entity_key, 0) + 1
            self._consecutive_low[entity_key] = cl
            if cl >= 2:
                # 2 consecutive LEN lows → confirmed damage
                model_hp = float(len_hp)
                self._model_last_hit[entity_key] = now
                self._consecutive_low[entity_key] = 0
                self._consecutive_high[entity_key] = 0
        else:
            self._consecutive_low[entity_key] = 0

        # === GENTLE BLEND: within ±5 of model → small correction ===
        if -5 <= diff <= 5:
            model_hp = model_hp + diff * 0.15

        model_hp = max(1.0, min(100.0, model_hp))
        self._model_hp[entity_key] = model_hp
        self._model_last_update[entity_key] = now
        return int(model_hp)

    def _apply_damage_model_tick(self, entity_key: str) -> int:
        """Tick the damage model on frames where LEN didn't fire.

        Only applies passive regen: after 2s of no damage,
        gradually rise at 25%/s toward 100%.
        Otherwise returns the stored model HP unchanged.
        """
        now = time.time()

        if entity_key not in self._model_hp:
            self._model_hp[entity_key] = 100.0
            self._model_last_hit[entity_key] = now   # use current time, NOT 0 (epoch)
            self._model_last_update[entity_key] = now
            return 100

        model_hp = self._model_hp[entity_key]

        # Passive regen: after 3s of no damage, rise at 13%/s toward 100
        # (matches Brawl Stars passive regen roughly)
        if model_hp < 100.0:
            since_hit = now - self._model_last_hit.get(entity_key, now)
            if since_hit > 3.0:
                last_upd = self._model_last_update.get(entity_key, now)
                dt = max(0.001, now - last_upd)
                max_rise = dt * 13.0
                model_hp = min(100.0, model_hp + max_rise)

        model_hp = max(1.0, min(100.0, model_hp))
        self._model_hp[entity_key] = model_hp
        self._model_last_update[entity_key] = now
        return int(model_hp)

    def _length_based_estimate(self, frame, bbox, is_player: bool) -> Tuple[int, float, float]:
        """
        MAX-WIDTH LENGTH-BASED HP estimation.

        Measures ONLY the health-coloured bar length. Returns a sentinel (-2)
        so the caller (estimate()) can convert to % via max-width tracking.
        This completely avoids detecting depleted bar colours.

        Returns:
          (-2, confidence, health_length)  on success (sentinel for max-width)
          (-1, 0.0, 0)                    on failure
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            char_width = x2 - x1
            char_height = y2 - y1
            if char_width < 15 or char_height < 15:
                return -1, 0.0, 0

            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return -1, 0.0, 0
            fh, fw = arr.shape[:2]

            # HP bar is above the character – extend slightly into bbox
            # so bars near / overlapping y1 are not missed (matches RAW region)
            search_above = max(60, int(char_height * 0.60))
            search_below = max(8, int(char_height * 0.10))
            search_y1 = max(0, y1 - search_above)
            search_y2 = min(fh, y1 + search_below)
            pad_x = max(25, int(char_width * 0.4))
            search_x1 = max(0, x1 - pad_x)
            search_x2 = min(fw, x2 + pad_x)

            if search_y2 - search_y1 < 5 or search_x2 - search_x1 < 20:
                return -1, 0.0, 0

            crop = arr[search_y1:search_y2, search_x1:search_x2]
            if crop.size == 0:
                return -1, 0.0, 0

            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)

            # ── Step 1: health-coloured pixels ──────────────────────────
            health_mask = self._build_health_mask(crop_hsv, is_player)
            # Morphological closing to bridge 1-2px gaps in the bar
            health_mask = cv2.morphologyEx(health_mask, cv2.MORPH_CLOSE, self._KERN_CLOSE)

            # ── Step 2: find BAR ROWS ───────────────────────────────────
            health_per_row = np.sum(health_mask > 0, axis=1)
            active_rows = np.where(health_per_row >= 2)[0]

            if len(active_rows) < 1:
                return -1, 0.0, 0

            row_start, row_end = self._find_longest_extent(active_rows, gap_tol=2)
            bar_height = row_end - row_start + 1

            if bar_height < 1 or bar_height > 25:
                return -1, 0.0, 0

            # ── Step 3: measure health extent in bar band ───────────────
            bar_band = health_mask[row_start:row_end + 1, :]
            col_prof = np.sum(bar_band > 0, axis=0)
            health_cols = np.where(col_prof >= 1)[0]

            if len(health_cols) < 2:
                return -1, 0.0, 0

            h_start, h_end = self._find_longest_extent(health_cols, gap_tol=3)
            health_length = h_end - h_start + 1

            if health_length < 5:
                return -1, 0.0, 0

            # Confidence from bar density and height
            bar_density = np.mean(col_prof[h_start:h_end + 1])
            confidence = min(1.0, (bar_density / 2.0) * (bar_height / 3.0))
            confidence = max(0.3, min(1.0, confidence))

            return -2, confidence, health_length

        except Exception:
            return -1, 0.0, 0

    @staticmethod
    def _find_longest_extent(cols: np.ndarray, gap_tol: int = 2) -> Tuple[int, int]:
        """Find the start and end of the longest contiguous segment."""
        if len(cols) == 0:
            return 0, 0
        if len(cols) == 1:
            return cols[0], cols[0]
        
        cols = np.sort(cols)
        gaps = np.diff(cols) > gap_tol
        seg_ids = np.concatenate([[0], np.cumsum(gaps)])
        
        best_start, best_end = cols[0], cols[0]
        best_length = 1
        
        for sid in range(seg_ids[-1] + 1):
            seg_cols = cols[seg_ids == sid]
            if len(seg_cols) > 0:
                s, e = seg_cols[0], seg_cols[-1]
                length = e - s + 1
                if length > best_length:
                    best_length = length
                    best_start, best_end = s, e
        
        return best_start, best_end

    # --- cORE RAW ESTIMATION ---

    def _raw_estimate(self, frame, bbox, is_player: bool) -> Tuple[int, float, float]:
        """Core HP estimation using multi-row consensus + morphological cleanup.

        Returns (hp_percent, confidence, bar_width) or (-1, 0, 0) on failure.
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            char_width = x2 - x1
            char_height = y2 - y1
            if char_width < 10 or char_height < 10:
                return -1, 0.0, 0

            arr = np.asarray(frame)
            if arr.ndim != 3 or arr.shape[2] < 3:
                return -1, 0.0, 0
            fh, fw = arr.shape[:2]

            # resolution-independent search region
            # HP bar sits above the name text, which is above the character
            # head.  Total offset from bbox-top to bar is typically 20-50px.
            # INCREASED to 0.50 to reliably capture the bar
            # on all brawlers regardless of model size variations.
            search_above = max(30, int(char_height * 0.50))
            search_below = max(8, int(char_height * 0.08))
            pad_x = max(15, int(char_width * 0.20))

            search_y1 = max(0, y1 - search_above)
            search_y2 = min(fh, y1 + search_below)
            search_x1 = max(0, x1 - pad_x)
            search_x2 = min(fw, x2 + pad_x)

            if search_y2 - search_y1 < 3 or search_x2 - search_x1 < 10:
                return -1, 0.0, 0

            crop = arr[search_y1:search_y2, search_x1:search_x2]
            if crop.size == 0:
                return -1, 0.0, 0

            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            crop_h, crop_w = crop_hsv.shape[:2]

            # build health mask (colored portion of bar)
            health_mask = self._build_health_mask(crop_hsv, is_player)

            # morphological cleanup
            # Close tiny horizontal gaps (character model cuts through bar)
            health_mask = cv2.morphologyEx(health_mask, cv2.MORPH_CLOSE, self._KERN_CLOSE)
            # Remove isolated noise pixels
            health_mask = cv2.morphologyEx(health_mask, cv2.MORPH_OPEN, self._KERN_OPEN)

            health_px = int(np.count_nonzero(health_mask))
            self._debug_log(f"[RAW] {'P' if is_player else 'E'} crop={crop_h}x{crop_w} "
                           f"health_px={health_px} bbox=({x1},{y1},{x2},{y2}) "
                           f"search_above={search_above} search_below={search_below}")

            # build depleted bar background mask
            # Dark grey background of the depleted portion.
            # RELAXED MASK: Wider gray range to catch depleted bar reliably
            # V=15-120 catches dark to medium gray, S=0-60 allows some color tint
            bar_bg_mask = cv2.inRange(crop_hsv,
                                       np.array([0, 0, 15]),
                                       np.array([180, 60, 120]))
            # Dark depleted bar
            dark_bg = cv2.inRange(crop_hsv,
                                   np.array([0, 0, 5]),
                                   np.array([180, 80, 50]))
            # Cyan/Teal depleted bar (seen in some modes)
            cyan_bg = cv2.inRange(crop_hsv,
                                   np.array([80, 20, 30]),
                                   np.array([115, 150, 120]))
            bar_bg_mask = cv2.bitwise_or(bar_bg_mask, dark_bg)
            bar_bg_mask = cv2.bitwise_or(bar_bg_mask, cyan_bg)
            bar_bg_mask = cv2.morphologyEx(bar_bg_mask, cv2.MORPH_CLOSE, self._KERN_CLOSE)

            # full bar = health + adjacent background
            full_bar_mask = cv2.bitwise_or(health_mask, bar_bg_mask)

            min_bar_pixels = max(4, int(char_width * 0.08))
            # Bar width sanity: at most 150% of char_width (filters env glow).
            # NOTE: we do NOT filter on a minimum health-segment width here,
            # because at low HP the colored portion can be very narrow (5-10px).
            # The min_bar_pixels check (4-6px) already rejects noise.
            max_bar_w = int(char_width * 1.50)

            # multi-row scanning
            row_estimates = []
            bar_widths = []
            bar_rows = []   # track which rows had valid bar readings

            binary_health = (health_mask > 0).astype(np.uint8)
            binary_full = (full_bar_mask > 0).astype(np.uint8)
            health_sums = binary_health.sum(axis=1)
            full_sums = binary_full.sum(axis=1)

            for r in range(crop_h):
                if health_sums[r] < min_bar_pixels:
                    continue

                h_cols = np.where(binary_health[r])[0]
                if len(h_cols) < min_bar_pixels:
                    continue

                h_width, h_start, h_end = self._longest_segment(h_cols, gap_tol=2)
                if h_width < min_bar_pixels:
                    continue

                # bAR SHAPE VALIDATION
                # Reject rows where the health segment is impossibly wide
                # (green/red glow spanning the whole crop).  Do NOT reject
                # narrow segments - at low HP the colored bar is legitimately
                # very small (5-15px).
                if h_width > max_bar_w:
                    continue

                # full bar width: only background pixels adjacent to health
                nearby_full_widths = []
                for nr in range(max(0, r - 2), min(crop_h, r + 3)):
                    if full_sums[nr] < min_bar_pixels:
                        continue
                    f_cols = np.where(binary_full[nr])[0]

                    # Only keep columns within reasonable range of health segment.
                    # The full bar cannot be wider than max_bar_w, so limit
                    # the search to h_start + max_bar_w (+ small margin).
                    bar_margin = max(10, int(char_width * 0.15))
                    valid_f_cols = f_cols[
                        (f_cols >= h_start - bar_margin) &
                        (f_cols <= h_start + max_bar_w + bar_margin)
                    ]
                    if len(valid_f_cols) >= min_bar_pixels:
                        fw_seg, _, _ = self._longest_segment(valid_f_cols, gap_tol=3)
                        nearby_full_widths.append(fw_seg)

                has_depleted = len(nearby_full_widths) > 0 and max(nearby_full_widths) > h_width + 3
                if nearby_full_widths:
                    full_width = max(nearby_full_widths)
                else:
                    full_width = h_width  # Assume 100% HP

                full_width = max(full_width, h_width, 12)
                
                # REMOVED AUTO-CLAMP: Always calculate real HP ratio
                # The old logic assumed 100% HP too often, causing wrong readings
                # when player had low HP but bar was wide.
                hp_pct = int((h_width / full_width) * 100)
                hp_pct = max(1, min(100, hp_pct))

                row_estimates.append(hp_pct)
                bar_widths.append(full_width)
                bar_rows.append((r, hp_pct, has_depleted, full_width))

            if not row_estimates:
                self._debug_log(f"[EMPTY] {'P' if is_player else 'E'} NO bar rows found! "
                               f"crop={crop_h}x{crop_w} health_px={health_px} "
                               f"min_bar_px={min_bar_pixels} max_bar_w={max_bar_w}")
                return -1, 0.0, 0

            # vertical band clustering
            # The HP bar spans at most ~8 consecutive rows. If we found
            # valid rows scattered over a wide vertical range, keep only
            # the densest cluster and discard outlier rows (glow artifacts).
            if len(bar_rows) > 2:
                best_cluster = self._best_row_cluster(bar_rows, max_span=8)
                if best_cluster and len(best_cluster) >= 2:
                    row_estimates = [hp for (_, hp, _, _) in best_cluster]
                    bar_widths = [bw for (_, _, _, bw) in best_cluster]

            # multi-row consensus (median - safe to name text noise)
            row_estimates.sort()
            n = len(row_estimates)
            median_hp = row_estimates[n // 2]

            best_bar_width = max(bar_widths) if bar_widths else 0

            # confidence scoring
            agreement_count = sum(1 for h in row_estimates if abs(h - median_hp) < 8)
            coverage_conf = min(1.0, agreement_count / 3.0)  # 3+ rows = full
            spread = max(row_estimates) - min(row_estimates)
            spread_conf = max(0, 1.0 - spread / 40.0)
            confidence = coverage_conf * 0.6 + spread_conf * 0.4

            # If we saw 100% but no depleted background, reduce confidence
            # (green glow often fakes 100% by filling the whole mask width)
            any_depleted = any(d for (_, _, d, *_) in bar_rows)
            if median_hp >= 98 and not any_depleted and n <= 2:
                confidence *= 0.55

            # Store debug info for external logging
            self._last_raw_debug["player" if is_player else "enemy"] = {
                "search_above": search_above,
                "search_below": search_below,
                "crop_h": crop_h,
                "rows_found": len(row_estimates),
                "any_depleted": any_depleted,
                "median_hp": median_hp,
                "conf": round(confidence, 2),
            }

            self._debug_log(f"[RESULT] {'P' if is_player else 'E'} median_hp={median_hp} "
                           f"conf={confidence:.2f} rows={n} depleted={any_depleted} "
                           f"bar_w={best_bar_width} all_hp={row_estimates}")

            return median_hp, confidence, best_bar_width

        except Exception:
            traceback.print_exc()
            return -1, 0.0, 0

    # --- TEMPORARY DEBUG LOGGER (DISABLED for performance) ---
    _debug_log_counter = 0
    @classmethod
    def _debug_log(cls, msg: str):
        # DISABLED: File I/O kills performance (was causing 7 IPS instead of 60+)
        return

    # --- hEALTH MASK BUILDERS (expanded HSV ranges) ---

    @staticmethod
    def _build_health_mask(crop_hsv: np.ndarray, is_player: bool) -> np.ndarray:
        """Build a binary mask of the health-colored portion of the bar.

        Covers ALL Brawl Stars HP bar color variants:
        - Player/teammate: green (full HP) -> yellow -> orange (low HP)
        - Enemy: red/pink/magenta (all HP levels)
        """
        if is_player:
            # Green bar (full HP): H=20-100, S≥12, V≥30 (broadened for dark/washed-out)
            g_mask = cv2.inRange(crop_hsv,
                                  np.array([20, 12, 30]),
                                  np.array([100, 255, 255]))
            # Yellow bar (medium HP): H=8-40, relaxed thresholds
            y_mask = cv2.inRange(crop_hsv,
                                  np.array([8, 12, 30]),
                                  np.array([40, 255, 255]))
            # Orange bar (low HP): H=0-20
            o_mask = cv2.inRange(crop_hsv,
                                  np.array([0, 15, 30]),
                                  np.array([20, 255, 255]))
            health_mask = cv2.bitwise_or(g_mask, y_mask)
            health_mask = cv2.bitwise_or(health_mask, o_mask)
        else:
            # ENEMY HP BAR COLORS - EXPANDED RANGES
            # Red bar range 1: H=0-20 (main red/orange range)
            r1 = cv2.inRange(crop_hsv,
                              np.array([0, 25, 40]),
                              np.array([20, 255, 255]))
            # Red bar range 2: H=150-180 (wrap-around for pinkish red)
            r2 = cv2.inRange(crop_hsv,
                              np.array([150, 25, 40]),
                              np.array([180, 255, 255]))
            # Pink/Magenta bar: H=130-160 (common enemy HP color!)
            pink = cv2.inRange(crop_hsv,
                                np.array([130, 30, 50]),
                                np.array([170, 255, 255]))
            # Deep red / dark red on certain map backgrounds: lower V
            r3 = cv2.inRange(crop_hsv,
                              np.array([0, 40, 25]),
                              np.array([15, 255, 90]))
            # Orange/Yellow enemy HP (some skins/modes): H=8-30
            orange = cv2.inRange(crop_hsv,
                                  np.array([8, 40, 50]),
                                  np.array([30, 255, 255]))
            health_mask = cv2.bitwise_or(r1, r2)
            health_mask = cv2.bitwise_or(health_mask, pink)
            health_mask = cv2.bitwise_or(health_mask, r3)
            health_mask = cv2.bitwise_or(health_mask, orange)

        return health_mask

    # --- vERTICAL BAND CLUSTERING ---

    @staticmethod
    def _best_row_cluster(bar_rows: list, max_span: int = 8) -> list:
        """Find the densest vertical cluster of bar rows within *max_span* pixels.

        bar_rows: list of (row_idx, hp_pct, has_depleted)
        Returns the subset of bar_rows in the best cluster.
        """
        if not bar_rows:
            return bar_rows
        # Sort by row index
        sorted_rows = sorted(bar_rows, key=lambda x: x[0])
        best_cluster = []
        for i in range(len(sorted_rows)):
            ri = sorted_rows[i][0]
            cluster = []
            for j in range(i, len(sorted_rows)):
                rj = sorted_rows[j][0]
                if rj - ri <= max_span:
                    cluster.append(sorted_rows[j])
                else:
                    break
            if len(cluster) > len(best_cluster):
                best_cluster = cluster
        return best_cluster

    # --- sEGMENT DETECTION ---

    @staticmethod
    def _longest_segment(cols: np.ndarray, gap_tol: int = 2) -> Tuple[int, int, int]:
        """Find the longest contiguous segment in sorted column indices.

        gap_tol: max gap allowed between consecutive pixels within one segment.
        Returns (width, start_col, end_col).
        """
        if len(cols) == 0:
            return 0, 0, 0

        gaps = np.diff(cols) > gap_tol
        seg_ids = np.concatenate([[0], np.cumsum(gaps)])

        best_width = 0
        best_start = cols[0]
        best_end = cols[0]

        for sid in range(seg_ids[-1] + 1):
            seg_cols = cols[seg_ids == sid]
            w = seg_cols[-1] - seg_cols[0] + 1
            if w > best_width:
                best_width = w
                best_start = seg_cols[0]
                best_end = seg_cols[-1]

        return best_width, best_start, best_end

    # --- tEMPORAL SMOOTHING ---

    def _smooth(self, entity_key: str, raw_hp: int, raw_conf: float) -> Tuple[int, float]:
        """Robust temporal smoothing using MEDIAN + fast damage detection.

        8-frame median naturally rejects outliers (noisy RAW readings).
        Fast-damage: 2+ of last 3 readings below threshold → respond NOW.
        Recovery cap: smoothed HP can only rise +3 per call to prevent
                     noise spikes from instantly restoring HP.
        """
        history = self._history.get(entity_key)
        if not history:
            return raw_hp, raw_conf

        if len(history) == 1:
            return history[0][0], history[0][1]

        last_smoothed = self._last_smoothed.get(entity_key, 100)
        hps = [h for h, c, t in history]
        confs = [c for h, c, t in history]

        # === FAST DAMAGE DETECTION ===
        # If 2+ of the last 3 readings are significantly below last smoothed,
        # this is real damage — respond immediately with their median.
        n_recent = min(3, len(hps))
        recent = hps[-n_recent:]
        recent_conf = confs[-n_recent:]
        low_count = sum(1 for h in recent if h < last_smoothed - 12)
        if low_count >= 2:
            low_readings = [h for h in recent if h < last_smoothed - 5]
            if low_readings:
                new_hp = int(np.median(low_readings))
                avg_c = float(np.mean([c for h, c in zip(recent, recent_conf)
                                       if h < last_smoothed - 5]))
                return max(1, min(100, new_hp)), max(0.3, avg_c)

        # === MEDIAN SMOOTHING ===
        # Median of the full 8-frame buffer — naturally rejects outliers.
        median_hp = int(np.median(hps))
        avg_conf = float(np.mean(confs))

        # === RECOVERY CAP ===
        # HP can only rise +3 per call to prevent noisy spikes.
        # At ~30-60 IPS this allows 90-180 %/s recovery — more than
        # enough for real Brawl Stars regen (~13%/s after 3s delay).
        if median_hp > last_smoothed + 3:
            median_hp = last_smoothed + 3

        return max(1, min(100, median_hp)), avg_conf

    # --- sPECIAL EFFECT DETECTION ---

    def _get_bar_crop(self, frame, bbox) -> Optional[np.ndarray]:
        """Get the HSV crop of the HP bar region above a bbox.
        Resolution-independent offsets.
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            char_height = y2 - y1
            char_width = x2 - x1
            if char_width < 10 or char_height < 10:
                return None

            arr = np.asarray(frame)
            fh, fw = arr.shape[:2]

            search_above = max(8, int(char_height * 0.40))
            search_below = max(6, int(char_height * 0.10))
            pad_x = max(5, int(char_width * 0.08))

            sy1 = max(0, y1 - search_above)
            sy2 = min(fh, y1 + search_below)
            sx1 = max(0, x1 - pad_x)
            sx2 = min(fw, x2 + pad_x)

            if sy2 - sy1 < 3 or sx2 - sx1 < 10:
                return None

            crop = arr[sy1:sy2, sx1:sx2]
            if crop.size == 0:
                return None

            return cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        except Exception:
            return None

    def has_shield(self, frame, bbox) -> bool:
        """Detect blue/cyan shield overlay (Rosa super, Darryl super, etc.)."""
        crop_hsv = self._get_bar_crop(frame, bbox)
        if crop_hsv is None:
            return False
        shield_mask = cv2.inRange(crop_hsv,
                                   np.array([90, 50, 80]),
                                   np.array([130, 255, 255]))
        shield_ratio = np.count_nonzero(shield_mask) / max(1, shield_mask.size)
        return shield_ratio > 0.08

    def is_being_healed(self, frame, bbox, is_player: bool = False) -> bool:
        """Detect healing glow (brighter green bar - Poco, Byron, etc.)."""
        if not is_player:
            return False
        crop_hsv = self._get_bar_crop(frame, bbox)
        if crop_hsv is None:
            return False
        heal_mask = cv2.inRange(crop_hsv,
                                 np.array([40, 100, 200]),
                                 np.array([80, 255, 255]))
        heal_ratio = np.count_nonzero(heal_mask) / max(1, heal_mask.size)
        return heal_ratio > 0.12

    def is_poisoned(self, frame, bbox) -> bool:
        """Detect poison effect (purple/magenta bar tint)."""
        crop_hsv = self._get_bar_crop(frame, bbox)
        if crop_hsv is None:
            return False
        poison_mask = cv2.inRange(crop_hsv,
                                   np.array([130, 40, 60]),
                                   np.array([170, 255, 255]))
        poison_ratio = np.count_nonzero(poison_mask) / max(1, poison_mask.size)
        return poison_ratio > 0.06

    # --- uTILITIES ---

    def get_stable_bar_width(self, entity_key: str) -> float:
        """Return the stable 100% bar width reference for an entity.
        Uses the 90th percentile of recently observed widths.
        """
        widths = self._max_bar_widths.get(entity_key)
        if not widths or len(widths) < 2:
            return 0.0
        sorted_w = sorted(widths)
        idx = int(len(sorted_w) * 0.9)
        return sorted_w[min(idx, len(sorted_w) - 1)]

    def reset(self, entity_key: str = None):
        """Reset smoothing history.

        """
        if entity_key:
            self._history.pop(entity_key, None)
            self._max_bar_widths.pop(entity_key, None)
            self._last_smoothed.pop(entity_key, None)
            self._last_confident.pop(entity_key, None)
            self._last_len_hp.pop(entity_key, None)
            self._len_miss_count.pop(entity_key, None)
            self._model_hp.pop(entity_key, None)
            self._model_last_hit.pop(entity_key, None)
            self._model_last_update.pop(entity_key, None)
            self._consecutive_low.pop(entity_key, None)
            self._consecutive_high.pop(entity_key, None)
        else:
            self._history.clear()
            self._max_bar_widths.clear()
            self._last_smoothed.clear()
            self._last_confident.clear()
            self._last_len_hp.clear()
            self._len_miss_count.clear()
            self._model_hp.clear()
            self._model_last_hit.clear()
            self._model_last_update.clear()
            self._consecutive_low.clear()
            self._consecutive_high.clear()

    def clear_entity(self, entity_key: str):
        """Clear history for a single entity (e.g. when the tracked enemy changes).

        Unlike reset(entity_key), this is a lightweight call designed for
        frequent use when the closest enemy bbox jumps to a different character.
        """
        self._history.pop(entity_key, None)
        self._last_smoothed.pop(entity_key, None)
        # Keep _last_confident and _max_bar_widths - they are still useful
        # as a fallback reference until fresh data comes in.

    def clear_stale_entities(self, keep_keys: set):
        """Remove history for entity keys NOT in *keep_keys*.

        Call once per frame with the set of currently-visible entity keys
        so that old enemy histories don't leak smoothing data into new enemies.
        """
        for key in list(self._history.keys()):
            if key not in keep_keys:
                self._history.pop(key, None)
                self._last_smoothed.pop(key, None)
                self._last_confident.pop(key, None)
                self._max_bar_widths.pop(key, None)
                self._last_len_hp.pop(key, None)
                self._len_miss_count.pop(key, None)
                self._model_hp.pop(key, None)
                self._model_last_hit.pop(key, None)
                self._model_last_update.pop(key, None)
                self._consecutive_low.pop(key, None)
                self._consecutive_high.pop(key, None)
