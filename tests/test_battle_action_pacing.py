import time
import unittest
from unittest.mock import patch

from play import Play


class DummyWindow:
    def __init__(self):
        self.keys = []
        self.moves = []
        self.releases = []

    def press_key(self, key):
        self.keys.append(key)
        return True

    def move_joystick_angle(self, angle, radius=None):
        self.moves.append((float(angle), radius))
        return {"ok": True, "attempted": True, "detail": "sent"}

    def release_all_inputs(self, reason=""):
        self.releases.append(reason)
        return True


def make_play():
    play = object.__new__(Play)
    play.window_controller = DummyWindow()
    play._runtime_state = "match"
    play._allow_skill_inputs = True
    play._battle_runtime = {
        "last_input_at": 0.0,
        "last_action_at": 0.0,
        "last_attack_decision": "",
        "last_movement": "",
    }
    play._last_confirmed_match_time = 100.0
    play._last_match_evidence_time = 100.0
    play._last_battle_input_log_at = 0.0
    play._last_battle_skip_log_at = 0.0
    play._last_battle_perf_log_at = 0.0
    play._last_movement_watchdog_log_at = 0.0
    play._last_movement_refresh_at = time.monotonic()
    play._analog_movement_radius = 145.0
    play._ability_skip_log_at = {}
    play._target_gate = {
        "allow_attack": True,
        "allow_ability": True,
        "target_confirmed": True,
        "target_confidence": 1.0,
        "target_age": 0.0,
        "target_source": "real_detection",
        "updated_at": time.time(),
    }
    play._target_confirmation_max_age = 0.55
    play._ability_target_confirmation_max_age = 0.35
    play._brawl_ball_no_fire_spawn_seconds = 2.2
    play.selected_gamemode = "knockout"
    play.game_mode = 3
    play._battle_tick_id = 0
    play._action_budget = {
        "tick_started_at": 0.0,
        "movement_sent": False,
        "attack_sent": False,
        "ability_sent": False,
        "ability_name": "",
    }
    play.super_cooldown = 1.0
    play.gadget_cooldown = 1.0
    play.hypercharge_cooldown = 2.5
    play.last_super_time = 0.0
    play.last_gadget_time = 0.0
    play.last_hypercharge_time = 0.0
    return play


def confirm_target(play, current_time):
    play._target_gate = {
        "allow_attack": True,
        "allow_ability": True,
        "target_confirmed": True,
        "target_confidence": 1.0,
        "target_age": 0.0,
        "target_source": "real_detection",
        "updated_at": current_time,
    }


class BattleActionPacingTests(unittest.TestCase):
    def test_one_ability_is_allowed_per_tick(self):
        play = make_play()
        play._begin_action_tick(20.0)
        confirm_target(play, 20.0)

        with patch("play.time.time", return_value=20.0):
            self.assertTrue(play.use_hypercharge(reason="ready_and_engaged"))
            self.assertFalse(play.use_super(reason="close_range_confirmed"))

        self.assertEqual(play.window_controller.keys, ["H"])
        self.assertTrue(play._action_budget["ability_sent"])
        self.assertEqual(play._action_budget["ability_name"], "hypercharge")

    def test_hypercharge_cooldown_suppresses_spam(self):
        play = make_play()
        play._begin_action_tick(30.0)
        confirm_target(play, 30.0)

        with patch("play.time.time", return_value=30.0):
            self.assertTrue(play.use_hypercharge())

        play._begin_action_tick(30.5)
        confirm_target(play, 30.5)
        with patch("play.time.time", return_value=30.5):
            self.assertFalse(play.use_hypercharge())

        self.assertEqual(play.window_controller.keys, ["H"])

    def test_one_movement_dispatch_is_allowed_per_tick(self):
        play = make_play()
        play._begin_action_tick(40.0)

        self.assertTrue(play._dispatch_movement_angle(270, detail="first", current_time=40.0))
        self.assertFalse(play._dispatch_movement_angle(90, detail="second", current_time=40.1))

        self.assertEqual(len(play.window_controller.moves), 1)
        self.assertEqual(play.window_controller.moves[0][0], 270.0)

    def test_watchdog_does_not_warn_after_recent_movement(self):
        play = make_play()
        play._last_movement_refresh_at = time.monotonic()

        self.assertLess(play._movement_silence_seconds(), 0.1)

    def test_ability_dispatch_does_not_release_movement_hold(self):
        play = make_play()
        play._begin_action_tick(50.0)
        confirm_target(play, 50.0)

        with patch("play.time.time", return_value=50.0):
            self.assertTrue(play.use_super(reason="close_range_confirmed"))

        self.assertEqual(play.window_controller.releases, [])

    def test_target_gate_blocks_attack_and_ability_without_enemy(self):
        play = make_play()
        play._begin_action_tick(60.0)
        play._target_gate = play._default_target_gate(60.0)

        with patch("play.time.time", return_value=60.0):
            self.assertFalse(play.attack())
            self.assertFalse(play.use_hypercharge())

        self.assertEqual(play.window_controller.keys, [])
        self.assertFalse(play._action_budget["attack_sent"])
        self.assertFalse(play._action_budget["ability_sent"])

    def test_target_gate_rejects_stale_target(self):
        play = make_play()
        play._begin_action_tick(62.0)
        play._target_gate = {
            "allow_attack": True,
            "allow_ability": True,
            "target_confirmed": True,
            "target_confidence": 1.0,
            "target_age": 3.0,
            "target_source": "stale_detection",
            "updated_at": 60.0,
        }

        with patch("play.time.time", return_value=62.0):
            self.assertFalse(play.attack())
            self.assertFalse(play.use_super())

        self.assertEqual(play.window_controller.keys, [])

    def test_no_fire_spawn_window_suppresses_attack_and_ability(self):
        play = make_play()
        play.selected_gamemode = "brawlball"
        play._battle_runtime["match_started_at"] = 100.0
        play._begin_action_tick(101.0)

        play._update_target_gate(
            enemy_count=1,
            target_confirmed=True,
            target_confidence=1.0,
            target_age=0.0,
            target_source="real_detection",
            allow_attack=True,
            allow_ability=True,
            current_time=101.0,
        )

        with patch("play.time.time", return_value=101.0):
            self.assertFalse(play.attack())
            self.assertFalse(play.use_hypercharge())

        self.assertEqual(play.window_controller.keys, [])

    def test_no_action_active_tick_refreshes_movement_and_updates_budget(self):
        play = make_play()
        play._begin_action_tick(70.0)
        play._last_movement_refresh_at = time.monotonic() - 1.0
        play.last_movement = 270.0
        play.game_mode = 3
        play.selected_gamemode = "knockout"
        play._active_tick_no_action_refresh_seconds = 0.1

        self.assertTrue(play._ensure_active_tick_has_action(70.0, {"enemy": [], "teammate": [], "wall": []}))

        self.assertTrue(play._action_budget["movement_sent"])
        self.assertTrue(play._action_budget["watchdog_sent"])
        self.assertEqual(play.window_controller.moves[-1][0], 270.0)


if __name__ == "__main__":
    unittest.main()
