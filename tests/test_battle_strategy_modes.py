import unittest
import time
from unittest.mock import patch

from play import Play


class DummyWindow:
    width = 1920
    height = 1080
    scale_factor = 1.0


def make_play():
    play = object.__new__(Play)
    play.window_controller = DummyWindow()
    play.selected_gamemode = "brawlball"
    play.game_mode = 3
    play._battle_runtime = {
        "match_started_at": 100.0,
        "active_strategy": "",
    }
    play._brawl_ball_opening_seconds = 6.0
    play._brawl_ball_opening_lock_seconds = 4.5
    play._brawl_ball_opening_hold_seconds = 1.4
    play._brawl_ball_opening_angle = None
    play._brawl_ball_opening_angle_until = 0.0
    play._last_brawl_ball_opening_log_at = 0.0
    play._last_brawl_ball_opening_override_log_at = 0.0
    play._last_angle_smoothing_log_at = 0.0
    play._showdown_roam_spin_angle = 270.0
    play._showdown_roam_angle_until = 0.0
    play._showdown_roam_hold_seconds = 1.25
    return play


class BattleStrategyModeTests(unittest.TestCase):
    def test_brawl_ball_opening_locks_vertical_up(self):
        play = make_play()

        angle = play._get_brawl_ball_opening_angle(player_pos=(1600, 900), current_time=101.0)

        self.assertEqual(angle, 270.0)
        self.assertEqual(play._battle_runtime["active_strategy"], "brawlball_opening")

    def test_brawl_ball_opening_finishes_after_window(self):
        play = make_play()

        self.assertIsNone(play._get_brawl_ball_opening_angle(player_pos=(1600, 900), current_time=107.0))

    def test_role_desired_range_labels_are_distinct(self):
        self.assertEqual(Play._role_desired_range_label("tank"), "close")
        self.assertEqual(Play._role_desired_range_label("sniper"), "long")
        self.assertEqual(Play._role_desired_range_label("thrower"), "wall_safe")

    def test_showdown_roam_direction_is_stable_inside_hold_window(self):
        play = make_play()
        play.selected_gamemode = "showdown"
        play.is_showdown_mode = True
        play._find_best_angle = lambda _player, angle, _wall: angle

        with patch("play.time.time", return_value=200.0):
            first = play._get_showdown_roam_move((960, 540), {})
        with patch("play.time.time", return_value=200.4):
            second = play._get_showdown_roam_move((960, 540), {})

        self.assertEqual(first, second)

    def test_movement_watchdog_initializes_missing_timestamp(self):
        play = object.__new__(Play)
        play._last_movement_refresh_at = 0.0
        play._last_movement_watchdog_log_at = 0.0

        silence = play._movement_silence_seconds()

        self.assertEqual(silence, 0.0)
        self.assertGreater(play._last_movement_refresh_at, 0.0)

    def test_movement_watchdog_uses_recent_refresh(self):
        play = object.__new__(Play)
        play._last_movement_refresh_at = time.monotonic()
        play._last_movement_watchdog_log_at = 0.0

        silence = play._movement_silence_seconds()

        self.assertLess(silence, 0.1)


if __name__ == "__main__":
    unittest.main()
