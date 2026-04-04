# manual aim via touch swipe: drag from attack/super button toward target

from __future__ import annotations

import math
import time
import threading
from typing import Tuple, Optional


# base coordinates (1920×1080)
ATTACK_BUTTON = (1725, 800)     # "M" key
SUPER_BUTTON = (1510, 880)      # "E" key
GADGET_BUTTON = (1640, 990)     # "G" key

# Max drag distance in pixels (scaled coords). Higher = more precise aim direction.
MAX_DRAG_DISTANCE = 180
# For throwers: they need much longer swipes for farther throws
MAX_DRAG_DISTANCE_THROWER = 420


class ManualAimer:
    """Implements aimed attacks via touch swipe on attack/super buttons."""

    def __init__(self, window_controller):
        self.wc = window_controller
        self._wr = 1.0
        self._hr = 1.0
        self._lock = threading.Lock()

        # Timing
        self._last_aimed_shot_time = 0.0
        self._aim_hold_duration = 0.04  # Seconds to hold aim before release (fast!)

        # Statistics
        self.total_aimed_shots = 0
        self.total_auto_shots = 0

    def _update_ratios(self):
        """Pull current scaling ratios from WindowController."""
        if self.wc:
            self._wr = getattr(self.wc, 'width_ratio', 1.0) or 1.0
            self._hr = getattr(self.wc, 'height_ratio', 1.0) or 1.0

    def _scale(self, base_x: float, base_y: float) -> Tuple[int, int]:
        """Scale base 1920×1080 coordinates to actual resolution."""
        return int(base_x * self._wr), int(base_y * self._hr)

    @staticmethod
    def _compute_aim_direction(player_screen_pos: Tuple[float, float],
                                target_screen_pos: Tuple[float, float],
                                lead_offset: Tuple[float, float] = (0, 0)
                                ) -> Tuple[float, float]:
        """Compute normalized aim direction from player to target+lead.

        Returns (dx, dy) normalized direction vector.
        """
        tx = target_screen_pos[0] + lead_offset[0]
        ty = target_screen_pos[1] + lead_offset[1]

        dx = tx - player_screen_pos[0]
        dy = ty - player_screen_pos[1]

        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1:
            return (0.0, -1.0)  # Default: aim up

        return (dx / mag, dy / mag)

    def aimed_attack(self, player_pos: Tuple[float, float],
                      target_pos: Tuple[float, float],
                      lead_offset: Tuple[float, float] = (0, 0),
                      hold_time: float = None,
                      playstyle: str = "fighter",
                      attack_range: float = 300):
        """Fire an aimed normal attack toward target_pos.
        
        For throwers: swipe length determines throw distance!
        """
        self._fire_aimed(ATTACK_BUTTON, player_pos, target_pos,
                          lead_offset, hold_time or self._aim_hold_duration,
                          playstyle, attack_range)
        self.total_aimed_shots += 1

    def aimed_super(self, player_pos: Tuple[float, float],
                     target_pos: Tuple[float, float],
                     lead_offset: Tuple[float, float] = (0, 0),
                     hold_time: float = None,
                     playstyle: str = "fighter",
                     attack_range: float = 300):
        """Fire an aimed super attack toward target_pos."""
        self._fire_aimed(SUPER_BUTTON, player_pos, target_pos,
                          lead_offset, hold_time or self._aim_hold_duration * 1.5,
                          playstyle, attack_range)

    def aimed_gadget(self, player_pos: Tuple[float, float],
                      target_pos: Tuple[float, float],
                      lead_offset: Tuple[float, float] = (0, 0),
                      playstyle: str = "fighter",
                      attack_range: float = 300):
        """Fire an aimed gadget (for gadgets that require aiming)."""
        self._fire_aimed(GADGET_BUTTON, player_pos, target_pos,
                          lead_offset, self._aim_hold_duration,
                          playstyle, attack_range)

    def auto_attack(self):
        """Quick tap on attack button -- auto-aim (fallback)."""
        self._update_ratios()
        bx, by = self._scale(*ATTACK_BUTTON)
        with self._lock:
            self.wc.touch_down(bx, by, pointer_id=self.wc.PID_ATTACK)
            time.sleep(0.03)
            self.wc.touch_up(bx, by, pointer_id=self.wc.PID_ATTACK)
        self.total_auto_shots += 1

    def _fire_aimed(self, button_base: Tuple[int, int],
                     player_pos: Tuple[float, float],
                     target_pos: Tuple[float, float],
                     lead_offset: Tuple[float, float],
                     hold_time: float,
                     playstyle: str = "fighter",
                     attack_range: float = 300):
        """Core aimed-fire implementation using touch swipe.

        Steps:
          1. Touch down on button
          2. Compute aim direction from player->target (with lead)
          3. Touch move to button + (direction × drag_distance)
          4. Wait hold_time for stability
          5. Touch up to release
          
        For THROWERS: swipe length determines throw distance!
        - Short swipe = short throw
        - Long swipe = long throw
        """
        self._update_ratios()
        btn_x, btn_y = self._scale(*button_base)

        # Aim direction
        aim_dx, aim_dy = self._compute_aim_direction(
            player_pos, target_pos, lead_offset
        )

        dist_to_target = math.hypot(
            (target_pos[0] + lead_offset[0]) - player_pos[0],
            (target_pos[1] + lead_offset[1]) - player_pos[1]
        )
        
        # THROWER-SPECIFIC: swipe length = throw distance
        if playstyle == "thrower":
            # For throwers, swipe length directly controls throw distance
            # Map distance to attack_range -> proportional swipe length
            # dist_to_target / attack_range gives us how far we need to throw (0.0 to 1.5+)
            dist_ratio = dist_to_target / max(attack_range, 1)
            # Throwers were under-throwing at long range; use wider clamp and boost.
            # 0.45 = avoid tiny lobs, 1.75 = allow full-range / slight overshoot throws.
            dist_ratio = min(1.75, max(0.45, dist_ratio))
            drag_dist = (MAX_DRAG_DISTANCE_THROWER * 1.35) * dist_ratio * self._wr
        else:
            # Non-throwers: drag for direction only, moderate scaling
            drag_ratio = min(1.0, max(0.55, dist_to_target / 350))
            drag_dist = MAX_DRAG_DISTANCE * drag_ratio * self._wr

        end_x = int(btn_x + aim_dx * drag_dist)
        end_y = int(btn_y + aim_dy * drag_dist)

        # Clamp to screen bounds
        if self.wc.width and self.wc.height:
            end_x = max(0, min(self.wc.width - 1, end_x))
            end_y = max(0, min(self.wc.height - 1, end_y))

        with self._lock:
            try:
                self.wc.touch_down(btn_x, btn_y, pointer_id=self.wc.PID_ATTACK)
                time.sleep(0.015)  # Brief pause before drag
                self.wc.touch_move(end_x, end_y, pointer_id=self.wc.PID_ATTACK)
                time.sleep(hold_time)
                self.wc.touch_up(end_x, end_y, pointer_id=self.wc.PID_ATTACK)
            except Exception as e:
                # Release finger on error to avoid stuck touches
                try:
                    self.wc.touch_up(end_x, end_y, pointer_id=self.wc.PID_ATTACK)
                except Exception:
                    pass
                print(f"[ManualAimer] ERROR: {e}")

        self._last_aimed_shot_time = time.time()

    def should_use_manual_aim(self, distance: float, enemy_speed: float,
                                playstyle: str, confidence: float = 1.0
                                ) -> bool:
        """Decide whether to use manual aim vs auto-aim.

        AGGRESSIVE manual aim policy -- auto-aim misses too often because
        enemies move between the tap and projectile arrival. Manual aim
        with lead prediction dramatically improves hit rate.

        Auto-aim only for:
          - Extremely close range (<80px) -- physically can't miss
          - Tanks at close range (<150px) -- wide attacks compensate
        """
        # Only auto-aim at point-blank range
        if distance < 80:
            return False

        # Tanks have wide attacks, auto-aim OK at close range
        if playstyle == "tank" and distance < 150:
            return False

        # ANY moving enemy at ANY meaningful distance = manual aim
        if enemy_speed > 40 and distance > 100:
            return True

        # Snipers/Throwers: ALWAYS manual aim (except point-blank above)
        if playstyle in ("sniper", "thrower"):
            return True

        # Everyone else: manual aim at medium+ range
        if distance > 150:
            return True

        # Close range non-tank: still use manual for better accuracy
        if distance > 80:
            return True

        return False

    def get_stats(self) -> dict:
        """Return aim statistics for debug overlay."""
        total = self.total_aimed_shots + self.total_auto_shots
        return {
            "aimed_shots": self.total_aimed_shots,
            "auto_shots": self.total_auto_shots,
            "manual_ratio": self.total_aimed_shots / max(1, total),
        }

    def reset(self):
        """Reset for new match."""
        self.total_aimed_shots = 0
        self.total_auto_shots = 0
        self._last_aimed_shot_time = 0.0
