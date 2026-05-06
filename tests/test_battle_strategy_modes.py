import unittest
import time
from unittest.mock import patch

from play import Play


class DummyWindow:
    width = 1920
    height = 1080
    scale_factor = 1.0

    def __init__(self):
        self.moves = []

    def move_joystick_angle(self, angle, radius=None):
        self.moves.append((float(angle), radius))
        return {"ok": True, "attempted": True, "detail": "sent"}


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
    play._brawl_ball_spawn_escape_seconds = 8.0
    play._brawl_ball_spawn_escape_min_seconds = 5.8
    play._brawl_ball_spawn_escape_nudge_after = 6.2
    play._brawl_ball_spawn_escape_nudge_interval = 1.8
    play._brawl_ball_spawn_escape_uncertain_seconds = 10.0
    play._brawl_ball_spawn_escape_extended_seconds = 14.0
    play._brawl_ball_opening_angle = None
    play._brawl_ball_opening_angle_until = 0.0
    play._brawl_ball_spawn_escape_active = False
    play._brawl_ball_spawn_escape_complete = False
    play._brawl_ball_spawn_escape_started_at = 0.0
    play._brawl_ball_spawn_escape_last_nudge_at = 0.0
    play._last_brawl_ball_spawn_escape_log_at = 0.0
    play._last_brawl_ball_opening_log_at = 0.0
    play._last_brawl_ball_opening_override_log_at = 0.0
    play._last_angle_smoothing_log_at = 0.0
    play._last_brawl_ball_lane_log_at = 0.0
    play._last_corner_escape_log_at = 0.0
    play._last_no_action_tick_log_at = 0.0
    play._showdown_roam_spin_angle = 270.0
    play._showdown_roam_angle_until = 0.0
    play._showdown_roam_hold_seconds = 1.25
    play._brawl_ball_lane_angle = 270.0
    play._brawl_ball_lane_angle_until = 0.0
    play._corner_escape_active = False
    play._corner_escape_until = 0.0
    play._authoritative_movement_angle = None
    play._authoritative_movement_source = ""
    play._authoritative_movement_at = 0.0
    play._wall_blocked_threshold_seconds = 0.7
    play._wall_blocked_tick_threshold = 2
    play._wall_escape_nudge_seconds = 0.45
    play._wall_escape_return_seconds = 0.65
    play._wall_escape_state = {
        "active": False,
        "source": "",
        "angle": None,
        "started_at": 0.0,
        "last_seen_at": 0.0,
        "count": 0,
        "nudge_until": 0.0,
        "return_until": 0.0,
        "nudge_angle": None,
        "return_angle": None,
        "side": 1,
        "return_logged": False,
    }
    play._last_watchdog_skip_log_at = 0.0
    play._analog_goal_hold_times = {
        "brawlball_lane_push": 1.15,
        "spawn_escape_no_vision": 1.4,
        "corner_escape": 0.85,
    }
    play._analog_goal_priorities = {
        "brawlball_lane_push": 3,
        "corner_escape": 7,
        "spawn_escape_no_vision": 8,
    }
    play._committed_analog_reason = ""
    play._committed_analog_until = 0.0
    play._planned_analog_reason = None
    play._analog_movement_radius = 145.0
    play.last_movement = None
    play.last_movement_time = 0.0
    play._last_movement_refresh_at = time.monotonic()
    play._last_movement_watchdog_log_at = 0.0
    play._runtime_state = "match"
    play._action_budget = {
        "tick_started_at": 0.0,
        "tick_id": 0,
        "movement_sent": False,
        "attack_sent": False,
        "ability_sent": False,
        "watchdog_sent": False,
    }
    play._last_battle_input_log_at = 0.0
    play._last_battle_skip_log_at = 0.0
    return play


class BattleStrategyModeTests(unittest.TestCase):
    def test_brawl_ball_opening_locks_vertical_up(self):
        play = make_play()

        angle = play._get_brawl_ball_opening_angle(player_pos=(1600, 900), current_time=101.0)

        self.assertEqual(angle, 270.0)
        self.assertEqual(play._battle_runtime["active_strategy"], "brawlball_spawn_escape")

    def test_brawl_ball_opening_finishes_after_window(self):
        play = make_play()

        self.assertIsNone(play._get_brawl_ball_opening_angle(player_pos=(1600, 900), current_time=107.0))

    def test_spawn_escape_inside_own_spawn_returns_vertical(self):
        play = make_play()

        angle = play._get_brawl_ball_spawn_escape_angle(player_pos=(960, 820), current_time=103.0)

        self.assertEqual(angle, 270.0)
        self.assertEqual(play._battle_runtime["active_strategy"], "brawlball_spawn_escape")

    def test_spawn_escape_rejects_noisy_side_candidate(self):
        play = make_play()

        angle = play._get_brawl_ball_spawn_escape_angle(
            player_pos=(960, 820),
            current_time=103.0,
            candidate_angle=150,
        )

        self.assertEqual(angle, 270.0)

    def test_spawn_escape_uses_deterministic_nudge_then_returns_vertical(self):
        play = make_play()

        nudge = play._get_brawl_ball_spawn_escape_angle(player_pos=(760, 820), current_time=106.3)
        vertical = play._get_brawl_ball_spawn_escape_angle(player_pos=(760, 820), current_time=106.7)

        self.assertEqual(nudge, 285.0)
        self.assertEqual(vertical, 270.0)

    def test_spawn_escape_completes_after_leaving_spawn(self):
        play = make_play()

        self.assertIsNone(play._get_brawl_ball_spawn_escape_angle(player_pos=(960, 520), current_time=109.0))
        self.assertTrue(play._brawl_ball_spawn_escape_complete)

    def test_no_vision_spawn_gate_dominates_side_candidates(self):
        play = make_play()
        play._battle_runtime["match_started_at"] = 100.0
        play.last_movement = 186.0
        play._brawl_ball_lane_angle = 356.0
        data = {
            "enemy": [],
            "teammate": [],
            "player": [],
            "_player_source": "missing",
        }

        status = play._brawlball_no_vision_spawn_status(data, current_time=107.5)
        self.assertIsNotNone(status)
        self.assertEqual(round(status["angle"]), 270)

        sent = play._force_brawlball_no_vision_spawn_escape(data, current_time=107.5)

        self.assertTrue(sent)
        self.assertEqual(play.window_controller.moves[-1][0], 270.0)
        self.assertEqual(play.last_movement, 270.0)
        self.assertEqual(play._brawl_ball_lane_angle, 270.0)

    def test_timer_expired_but_self_unknown_keeps_spawn_escape_active(self):
        play = make_play()
        play._battle_runtime["match_started_at"] = 100.0

        status = play._brawlball_no_vision_spawn_status(
            {"enemy": [], "teammate": [], "player": [], "_player_source": "missing"},
            current_time=112.0,
        )

        self.assertIsNotNone(status)
        self.assertEqual(round(status["angle"]), 270)
        self.assertTrue(play._brawl_ball_spawn_escape_active)
        self.assertFalse(play._brawl_ball_spawn_escape_complete)

    def test_extended_unknown_spawn_escape_falls_back_to_lane_push(self):
        play = make_play()
        play._battle_runtime["match_started_at"] = 100.0

        status = play._brawlball_no_vision_spawn_status(
            {"enemy": [], "teammate": [], "player": [], "_player_source": "missing"},
            current_time=116.0,
        )

        self.assertIsNone(status)
        self.assertTrue(play._brawl_ball_spawn_escape_complete)
        self.assertEqual(play._authoritative_movement_source, "brawlball_lane_push")
        self.assertEqual(play._authoritative_movement_angle, 270.0)

    def test_watchdog_uses_spawn_escape_angle_not_stale_side_angle(self):
        play = make_play()
        play._battle_runtime["match_started_at"] = 100.0
        play._begin_action_tick(108.0)
        play.last_movement = 155.0
        data = {"enemy": [], "teammate": [], "player": [], "_player_source": "missing"}

        self.assertTrue(play._ensure_active_tick_has_action(108.0, data, reason="test"))

        self.assertEqual(play.window_controller.moves[-1][0], 270.0)
        self.assertTrue(play._action_budget["movement_sent"])
        self.assertTrue(play._action_budget["watchdog_sent"])

    def test_brawlball_lane_push_rejects_random_side_loop(self):
        play = make_play()
        play._brawl_ball_lane_angle = 270.0
        play._brawl_ball_lane_angle_until = 205.0

        angle = play._get_brawl_ball_lane_push_angle((50.0, 700.0), current_time=204.0)

        self.assertEqual(round(angle), 270)
        self.assertEqual(play._battle_runtime["active_strategy"], "brawlball_objective")

    def test_wall_blocked_lane_push_triggers_deterministic_nudge(self):
        play = make_play()
        play._is_path_blocked_angle = lambda _player, _angle, _wall: True
        wall_context = {"rectangles": [[900, 300, 1000, 500]], "line_cache": {}}

        self.assertIsNone(play._wall_blocked_escape_angle(270.0, "brawlball_lane_push", (960, 820), wall_context, 200.0))
        nudge = play._wall_blocked_escape_angle(270.0, "brawlball_lane_push", (960, 820), wall_context, 200.8)

        self.assertIsNotNone(nudge)
        self.assertNotEqual(round(nudge), 270)
        self.assertEqual(play._authoritative_movement_source, "wall_escape")

    def test_wall_escape_returns_to_lane_after_nudge(self):
        play = make_play()
        play._is_path_blocked_angle = lambda _player, _angle, _wall: True
        wall_context = {"rectangles": [[900, 300, 1000, 500]], "line_cache": {}}
        play._wall_blocked_escape_angle(270.0, "spawn_escape_no_vision", (960, 820), wall_context, 200.0)
        play._wall_blocked_escape_angle(270.0, "spawn_escape_no_vision", (960, 820), wall_context, 200.8)

        returned = play._wall_blocked_escape_angle(270.0, "spawn_escape_no_vision", (960, 820), wall_context, 201.3)

        self.assertEqual(round(returned), 270)

    def test_watchdog_uses_lane_angle_not_stale_random_angle(self):
        play = make_play()
        play.last_movement = 297.0
        play._set_authoritative_movement_angle(270.0, "brawlball_lane_push", 200.0)

        angle, source = play._authoritative_watchdog_angle({"enemy": [], "teammate": [], "player": []}, 200.4)

        self.assertEqual(round(angle), 270)
        self.assertEqual(source, "brawlball_lane_push")

    def test_bottom_right_corner_escapes_toward_center_upfield(self):
        play = make_play()

        angle = play._get_corner_escape_angle((1850.0, 1010.0), current_time=200.0, no_target=True)
        dx, dy = play.angle_to_vector(angle)

        self.assertLess(dx, 0.0)
        self.assertLess(dy, 0.0)
        self.assertEqual(play._battle_runtime["active_strategy"], "corner_escape")

    def test_bottom_left_corner_escapes_toward_center_upfield(self):
        play = make_play()

        angle = play._get_corner_escape_angle((40.0, 1010.0), current_time=200.0, no_target=True)
        dx, dy = play.angle_to_vector(angle)

        self.assertGreater(dx, 0.0)
        self.assertLess(dy, 0.0)
        self.assertEqual(play._battle_runtime["active_strategy"], "corner_escape")

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
