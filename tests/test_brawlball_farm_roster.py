import unittest
import time
import numpy as np

from brawlstars_api import parse_player_profile_payload, normalize_player_tag
from dashboard import Dashboard
from farm_roster import build_farm_plan
from play import Movement, Play
from qt_ui.bridge import QtBridge


class DummyWindow:
    width = 1920
    height = 1080
    scale_factor = 1.0


class BrawlBallAndFarmTests(unittest.TestCase):
    def test_brawlball_is_not_treated_as_5v5(self):
        self.assertTrue(Movement._is_brawl_ball_mode("brawlball"))
        self.assertTrue(Movement._should_detect_walls_for_mode("brawlball"))
        self.assertFalse(Movement._is_brawl_ball_mode("brawlball_5v5"))

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

    def test_brawlball_opening_route_returns_stable_angle_across_ticks(self):
        movement = object.__new__(Play)
        movement.window_controller = DummyWindow()
        movement.selected_gamemode = "brawlball"
        movement.game_mode = 3
        movement._battle_runtime = Play._new_battle_runtime_state(movement)
        start = time.time()
        movement._battle_runtime["match_started_at"] = start
        movement._brawl_ball_opening_seconds = 6.0
        movement._brawl_ball_opening_hold_seconds = 2.0
        movement._brawl_ball_opening_angle = None
        movement._brawl_ball_opening_angle_until = 0.0
        movement._last_brawl_ball_opening_log_at = start
        movement._last_angle_smoothing_log_at = start

        first = movement._get_brawl_ball_opening_angle((960.0, 940.0), start + 0.2)
        second = movement._get_brawl_ball_opening_angle((960.0, 160.0), start + 0.6)

        self.assertEqual(round(first), round(second))

    def test_brawlball_no_detection_fallback_uses_opening_route(self):
        movement = object.__new__(Play)
        movement.window_controller = DummyWindow()
        movement.selected_gamemode = "brawlball"
        movement.game_mode = 3
        movement._runtime_state = "match"
        movement._last_confirmed_match_time = time.time()
        movement._last_match_evidence_time = time.time()
        movement._battle_runtime = Play._new_battle_runtime_state(movement)
        movement._battle_runtime["match_started_at"] = time.time()
        movement._brawl_ball_opening_seconds = 6.0
        movement._brawl_ball_opening_hold_seconds = 1.0
        movement._brawl_ball_opening_angle = None
        movement._brawl_ball_opening_angle_until = 0.0
        movement._last_brawl_ball_opening_log_at = time.time()
        movement._last_angle_smoothing_log_at = time.time()

        angle = movement._get_brawl_ball_opening_angle(None, time.time() + 0.1)

        self.assertEqual(round(angle), 270)
        self.assertEqual(movement._battle_runtime["active_strategy"], "brawlball_spawn_escape")

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

    def test_qt_trophy_farm_uses_all_local_brawlers_without_scan_or_roster(self):
        bridge = QtBridge.__new__(QtBridge)
        bridge._all_brawlers = ["shelly", "colt"]
        bridge.brawlers_data = []
        bridge._farm_state = {}
        bridge.bot_config = {
            "trophy_farm_mode": "manual",
            "trophy_farm_target": 500,
            "trophy_farm_strategy": "lowest_first",
            "trophy_farm_excluded": [],
        }
        bridge.general_config = {"auto_push_target_trophies": 1000}
        bridge.brawlers_info = {}
        bridge._brawler_scan_data = lambda: {}
        bridge._match_history_map = lambda: {}

        roster, queue, target, _strategy = bridge._build_trophy_farm_roster()

        self.assertEqual(target, 500)
        self.assertEqual([item["brawler"] for item in queue], ["colt", "shelly"])
        self.assertEqual([item["brawler"] for item in roster], ["colt", "shelly"])

    def test_manual_farm_mode_builds_queue_from_owned_brawlers_below_target(self):
        plan = build_farm_plan(
            all_brawlers=["shelly", "colt", "darryl"],
            farm_state={
                "manual_roster": {
                    "shelly": {"owned": True, "included": True, "trophies": 40},
                    "colt": {"owned": False, "included": True, "trophies": 10},
                    "darryl": {"owned": True, "included": False, "trophies": 0},
                }
            },
            target=100,
            strategy="lowest_first",
            mode="manual",
        )

        self.assertEqual([row["brawler"] for row in plan["queue"]], ["shelly"])
        reasons = {row["brawler"]: row["reason"] for row in plan["rows"]}
        self.assertEqual(reasons["colt"], "Not marked owned/unlocked")

    def test_api_farm_mode_uses_only_unlocked_api_roster(self):
        payload = {
            "name": "Tester",
            "tag": "#P123",
            "brawlers": [
                {"name": "SHELLY", "trophies": 50, "highestTrophies": 80, "power": 9, "rank": 12},
                {"name": "DARRYL", "trophies": 140, "power": 7},
            ],
        }
        parsed = parse_player_profile_payload(payload, "#P123")
        plan = build_farm_plan(
            all_brawlers=["shelly", "colt", "darryl"],
            farm_state={"api_roster": parsed["brawlers"]},
            target=100,
            strategy="lowest_first",
            mode="api",
        )

        self.assertEqual([row["brawler"] for row in plan["queue"]], ["shelly"])
        self.assertNotIn("colt", [row["brawler"] for row in plan["rows"]])

    def test_invalid_api_tag_is_rejected_before_network_use(self):
        with self.assertRaises(ValueError):
            normalize_player_tag("#BADTAG!")

    def test_brawlball_combat_strafe_is_disabled(self):
        movement = object.__new__(Play)
        movement.selected_gamemode = "brawlball"
        movement.is_showdown_mode = False
        movement.current_ammo = 3

        self.assertFalse(
            movement._should_enable_combat_strafe(
                target_hittable=True,
                should_retreat_for_ammo=False,
                enemy_distance=220,
                effective_safe_range=180,
                attack_range=300,
            )
        )

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

    def test_role_mapping_uses_brawler_playstyle_with_fighter_fallback(self):
        movement = object.__new__(Play)
        movement.brawlers_info = {"darryl": {"playstyle": "tank"}, "mystery": {"playstyle": "unknown"}}

        self.assertEqual(movement._playstyle_name("DARRYL"), "tank")
        self.assertEqual(movement._playstyle_name("mystery"), "fighter")

    def test_missing_player_can_use_estimated_match_position(self):
        movement = object.__new__(Play)
        movement.window_controller = DummyWindow()
        movement._runtime_state = "match"
        movement._last_confirmed_match_time = time.time()
        movement._last_match_evidence_time = time.time()
        movement.time_since_player_last_found = time.time() - 2.0

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        data = movement._estimate_player_detection(frame, {"enemy": [[100, 100, 150, 150]]}, time.time())

        self.assertEqual(data["_player_source"], "estimated")
        self.assertTrue(data["player"])


if __name__ == "__main__":
    unittest.main()
