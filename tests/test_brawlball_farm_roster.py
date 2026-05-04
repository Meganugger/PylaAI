import unittest

from dashboard import Dashboard
from play import Play
from qt_ui.bridge import QtBridge


class DummyWindow:
    width_ratio = 1.0
    height_ratio = 1.0


class BrawlBallAndFarmTests(unittest.TestCase):
    def test_brawlball_5v5_uses_brawlball_movement_rules(self):
        play = object.__new__(Play)
        play.game_mode_name = "brawlball_5v5"

        self.assertFalse(play._is_brawl_ball_mode())

    def test_brawlball_solo_search_moves_back_toward_midfield_from_corner(self):
        play = object.__new__(Play)
        play.window_controller = DummyWindow()
        play.game_mode_name = "brawlball"
        play._solo_search_target_idx = 0
        play._solo_search_last_switch = 0.0
        play._solo_search_target_hold_time = 2.8
        play._visited_zones = []
        play.last_decision_reason = ""
        play.is_path_blocked = lambda *_args, **_kwargs: False
        play._get_pathfinder_movement = lambda *_args, **_kwargs: None

        move = play._get_solo_search_movement((120.0, 940.0), [])

        self.assertIn("W", move)
        self.assertIn("D", move)

    def test_farm_candidates_show_roster_without_scan_data(self):
        dashboard = object.__new__(Dashboard)
        dashboard.all_brawlers = ["shelly", "colt", "darryl"]
        dashboard.brawlers_data = []
        dashboard._brawler_scan_data = {}

        self.assertEqual(dashboard._farm_candidate_brawlers(), ["colt", "darryl", "shelly"])

    def test_qt_farm_preview_uses_discovered_brawlers_without_scan_data(self):
        bridge = QtBridge.__new__(QtBridge)
        bridge.bot_config = {
            "trophy_farm_target": 500,
            "trophy_farm_strategy": "lowest_first",
            "trophy_farm_excluded": [],
        }
        bridge.general_config = {"auto_push_target_trophies": 1000}
        bridge.brawlers_info = {"shelly": {}, "colt": {}, "darryl": {}}
        bridge._all_brawlers = ["shelly", "colt", "darryl"]
        bridge.brawlers_data = []
        bridge._brawler_scan_data = lambda: {}
        bridge._match_history_map = lambda: {}
        bridge._as_int = QtBridge._as_int
        bridge._normalize_farm_strategy = QtBridge._normalize_farm_strategy
        bridge._normalize_brawler_set = QtBridge._normalize_brawler_set.__get__(bridge, QtBridge)
        bridge._canonical_brawler_name = QtBridge._canonical_brawler_name.__get__(bridge, QtBridge)
        bridge._scan_entry_for_brawler = QtBridge._scan_entry_for_brawler.__get__(bridge, QtBridge)

        roster, queue, _target, _strategy = bridge._build_trophy_farm_roster()

        self.assertEqual([item["brawler"] for item in queue], ["colt", "darryl", "shelly"])
        self.assertEqual([item["brawler"] for item in roster], ["colt", "darryl", "shelly"])

    def test_farm_candidates_keep_selected_brawler_with_scan_data(self):
        dashboard = object.__new__(Dashboard)
        dashboard.all_brawlers = ["shelly", "colt", "darryl"]
        dashboard.brawlers_data = [{"brawler": "darryl"}]
        dashboard._brawler_scan_data = {
            "shelly": {"unlocked": True},
            "colt": {"unlocked": False},
        }

        self.assertEqual(dashboard._farm_candidate_brawlers(), ["darryl", "shelly"])

    def test_brawler_name_resolution_is_case_and_punctuation_safe(self):
        movement = object.__new__(Play)
        movement.brawlers_info = {"darryl": {}, "8bit": {}}

        self.assertEqual(movement.resolve_brawler_name("DARRYL"), "darryl")
        self.assertEqual(movement.resolve_brawler_name("8-Bit"), "8bit")

    def test_unknown_brawler_range_uses_generic_fallback(self):
        movement = object.__new__(Play)
        movement.brawlers_info = {"darryl": {}}
        movement.brawler_ranges = None
        movement.window_controller = DummyWindow()
        movement._battle_runtime = movement._new_battle_runtime_state()

        self.assertEqual(movement.get_brawler_range("Not A Brawler"), [260, 440, 520])
        self.assertTrue(movement._battle_runtime["fallback_active"])


if __name__ == "__main__":
    unittest.main()
