import unittest

from dashboard import Dashboard
from play import Play


class DummyWindow:
    width_ratio = 1.0
    height_ratio = 1.0


class BrawlBallAndFarmTests(unittest.TestCase):
    def test_brawlball_5v5_uses_brawlball_movement_rules(self):
        play = object.__new__(Play)
        play.game_mode_name = "brawlball_5v5"

        self.assertTrue(play._is_brawl_ball_mode())

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

    def test_farm_candidates_keep_selected_brawler_with_scan_data(self):
        dashboard = object.__new__(Dashboard)
        dashboard.all_brawlers = ["shelly", "colt", "darryl"]
        dashboard.brawlers_data = [{"brawler": "darryl"}]
        dashboard._brawler_scan_data = {
            "shelly": {"unlocked": True},
            "colt": {"unlocked": False},
        }

        self.assertEqual(dashboard._farm_candidate_brawlers(), ["darryl", "shelly"])


if __name__ == "__main__":
    unittest.main()
