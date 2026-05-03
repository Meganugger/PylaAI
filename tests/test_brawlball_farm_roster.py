import unittest

from dashboard import Dashboard
from play import Movement, Play


class DummyWindow:
    width = 1920
    height = 1080
    scale_factor = 1.0


class BrawlBallAndFarmTests(unittest.TestCase):
    def test_brawlball_5v5_uses_brawlball_movement_rules(self):
        self.assertTrue(Movement._is_brawl_ball_mode("brawlball_5v5"))
        self.assertTrue(Movement._should_detect_walls_for_mode("brawlball_5v5"))

    def test_brawlball_solo_roam_moves_back_toward_midfield_from_corner(self):
        movement = object.__new__(Play)
        movement.window_controller = DummyWindow()
        movement.selected_gamemode = "brawlball"
        movement.game_mode = 3
        movement.is_showdown_mode = False
        movement.should_detect_walls = True
        movement.wall_path_probe_tiles = 1.5
        movement._brawl_ball_patrol_idx = 0
        movement._brawl_ball_patrol_switch_time = 0.0
        movement._brawl_ball_patrol_hold_time = 2.8
        movement._brawl_ball_patrol_arrival_radius = 95.0
        movement.wall_detour_state = {"angle": None, "goal_angle": None, "side": 0, "until": 0.0}

        angle = movement._get_brawl_ball_roam_movement(
            (120.0, 940.0),
            {"rectangles": [], "line_cache": {}, "enemy_hittable_cache": {}},
        )
        dx, dy = movement.angle_to_vector(angle)

        self.assertGreater(dx, 0.0)
        self.assertLess(dy, 0.0)

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
