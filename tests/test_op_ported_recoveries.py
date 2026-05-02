import json
import os
import time
import unittest
from unittest.mock import patch

import numpy as np

from stage_manager import StageManager


class DummyWindow:
    width_ratio = 1.0
    height_ratio = 1.0
    scale_factor = 1.0

    def __init__(self):
        self.pressed = []
        self.released = []
        self.clicked = []

    def screenshot(self):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def keys_up(self, keys):
        self.released.extend(keys)

    def press_key(self, key):
        self.pressed.append(key)

    def click(self, x, y, *args, **kwargs):
        self.clicked.append((x, y, kwargs))


class DummyTrophyObserver:
    def __init__(self, trophies=0):
        self.current_trophies = trophies
        self.current_wins = 0
        self.win_streak = 0

    def change_trophies(self, value):
        self.current_trophies = value


class OPPortedRecoveryTests(unittest.TestCase):
    def make_start_manager(self):
        manager = object.__new__(StageManager)
        manager.window_controller = DummyWindow()
        manager.lobby_config = {"template_matching": {"go_back_arrow": [0, 0, 175, 110]}}
        manager._last_brawler_menu_recovery_at = 0.0
        manager._lobby_visible_since = time.time() - 2.0
        manager._last_start_press_at = 0.0
        manager._start_press_attempts = 0
        manager._start_wait_logged_at = 0.0
        manager._lobby_start_settle_delay = 0.0
        manager._lobby_start_retry_delay = 0.0
        manager._lobby_start_blocked_until = 0.0
        manager._lobby_start_block_reason = ""
        manager._consecutive_lobby_start_fails = 0
        return manager

    def test_lobby_start_requires_confirmed_play_button(self):
        manager = self.make_start_manager()
        with patch("stage_manager.get_state", return_value="lobby"), \
                patch("stage_manager.is_lobby_play_button_visible", return_value=False):
            self.assertFalse(manager._try_press_lobby_start())

        self.assertEqual(manager.window_controller.pressed, [])

    def test_lobby_start_presses_q_after_state_and_play_button_confirm(self):
        manager = self.make_start_manager()
        with patch("stage_manager.get_state", return_value="lobby"), \
                patch("stage_manager.is_lobby_play_button_visible", return_value=True):
            self.assertTrue(manager._try_press_lobby_start())

        self.assertEqual(manager.window_controller.pressed, ["Q"])

    def test_post_match_hold_keeps_match_probes_in_end_recovery_window(self):
        manager = object.__new__(StageManager)
        manager._awaiting_lobby_result_sync = True
        manager._end_transition_started_at = 0.0
        manager._end_transition_last_action_at = 0.0
        manager._end_transition_last_result = None
        manager._end_transition_hold_match_until = 0.0
        manager._end_transition_hold_seconds = 10.0
        manager._end_transition_timeout = 12.0

        manager._begin_end_transition("victory", now=100.0)

        self.assertEqual(manager.get_end_transition_state(), "end_victory")
        self.assertTrue(manager.should_hold_match_probe(now=109.5))
        self.assertTrue(manager.should_hold_match_probe(now=111.5))
        self.assertFalse(manager.should_hold_match_probe(now=112.5))

    @patch("stage_manager.save_brawler_data")
    @patch.object(StageManager, "fetch_push_all_player_data")
    def test_push_all_api_refresh_selects_next_lowest_when_current_hits_target(self, mock_fetch, _mock_save):
        manager = object.__new__(StageManager)
        manager.brawlers_pick_data = [
            {
                "brawler": "first",
                "trophies": 990,
                "wins": 0,
                "win_streak": 0,
                "push_until": 1000,
                "type": "trophies",
                "selection_method": "lowest_trophies",
            },
            {
                "brawler": "second",
                "trophies": 30,
                "wins": 0,
                "win_streak": 0,
                "push_until": 1000,
                "type": "trophies",
                "selection_method": "lowest_trophies",
            },
        ]
        manager.Trophy_observer = DummyTrophyObserver(1000)
        manager.push_all_needs_selection = False
        manager._last_push_all_refresh_at = 0.0
        manager._push_all_refresh_interval = 0.0
        mock_fetch.return_value = {
            "brawlers": [
                {"name": "FIRST", "trophies": 1000},
                {"name": "SECOND", "trophies": 25},
            ]
        }

        self.assertTrue(manager.refresh_push_all_trophies_from_api())

        self.assertTrue(manager.push_all_needs_selection)
        self.assertEqual(manager.brawlers_pick_data[0]["brawler"], "second")
        self.assertEqual(manager.Trophy_observer.current_trophies, 25)

    def test_new_op_brawler_icons_and_info_are_present(self):
        with open("cfg/brawlers_info.json", "r", encoding="utf-8") as handle:
            brawlers = json.load(handle)

        for name in ("bolt", "buzzlightyear", "damian", "starrnova"):
            with self.subTest(name=name):
                self.assertIn(name, brawlers)
                self.assertIn("playstyle", brawlers[name])
                self.assertTrue(os.path.exists(f"api/assets/brawler_icons/{name}.png"))


if __name__ == "__main__":
    unittest.main()
