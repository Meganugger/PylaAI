import math
import random
import time

import cv2
import numpy as np
from state_finder.main import get_state, find_game_result
from detect import Detect
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info

brawl_stars_width, brawl_stars_height = 1920, 1080
debug = load_toml_as_dict("cfg/general_config.toml").get("super_debug", "no") == "yes"

class Movement:

    def __init__(self, window_controller):
        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        self.fix_movement_keys = {
            "delay_to_trigger": bot_config["unstuck_movement_delay"],
            "duration": bot_config["unstuck_movement_hold_time"],
            "toggled": False,
            "started_at": time.time(),
            "fixed": ""
        }
        self.game_mode = bot_config["gamemode_type"]
        self.selected_gamemode = self._normalize_gamemode_name(bot_config.get("gamemode", "knockout"))
        self.game_mode_name = self.selected_gamemode
        self.is_showdown_mode = self._is_showdown_behavior_enabled(self.selected_gamemode, self.game_mode)
        gadget_value = bot_config["bot_uses_gadgets"]
        self.should_use_gadget = str(gadget_value).lower() in ("yes", "true", "1")
        self.super_treshold = time_config["super"]
        self.gadget_treshold = time_config["gadget"]
        self.hypercharge_treshold = time_config["hypercharge"]
        self.walls_treshold = time_config["wall_detection"]
        self.keep_walls_in_memory = self.walls_treshold <= 1
        self.last_walls_data = []
        self.keys_hold = []
        self.time_since_different_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.is_gadget_ready = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready = False
        self.window_controller = window_controller
        self.TILE_SIZE = 60
        self._showdown_regroup_distance = float(bot_config.get("showdown_regroup_distance", 165))
        self._showdown_team_pull_distance = float(bot_config.get("showdown_team_pull_distance", 130))
        self._showdown_enemy_chase_timeout = float(bot_config.get("showdown_enemy_chase_timeout", 2.0))
        self._search_target_hold_time = float(bot_config.get("search_target_hold_time", 3.5))
        self._unstuck_progress_distance = float(bot_config.get("unstuck_progress_distance", 42))
        self._showdown_regroup_release_distance = float(
            bot_config.get("showdown_regroup_release_distance", max(96.0, self._showdown_team_pull_distance - 28.0))
        )
        self._showdown_fog_pixels_minimum = 500
        self._showdown_fog_low = np.array((50, 95, 215), dtype=np.uint8)
        self._showdown_fog_high = np.array((60, 125, 245), dtype=np.uint8)
        self._showdown_fog_escape_distance = 220.0
        self._showdown_fog_flee_distance = float(bot_config.get("showdown_fog_flee_distance", 130))
        self._showdown_fog_min_blob_pixels = int(bot_config.get("showdown_fog_min_blob_pixels", 300))
        self._showdown_fog_min_pixels_in_radius = int(bot_config.get("showdown_fog_min_pixels_in_radius", 50))
        self._showdown_fog_check_every_n_frames = max(
            1,
            int(bot_config.get("showdown_fog_check_every_n_frames", 3)),
        )
        self._showdown_fog_check_counter = 0
        self._showdown_fog_cached_angle = None
        self._showdown_fog_mask_cache_key = None
        self._showdown_fog_mask_cache_value = None
        self._showdown_teammate_hysteresis = float(bot_config.get("showdown_teammate_hysteresis", 0.20))
        self._showdown_locked_teammate = None
        self._showdown_locked_teammate_distance = float("inf")
        self._showdown_regroup_active = False
        self._movement_anchor_pos = None
        self._movement_anchor_command = ""
        self._movement_anchor_angle = None
        self._movement_anchor_angle_pos = None
        self._analog_movement_radius = float(bot_config.get("analog_movement_radius", 145.0))
        self._analog_turn_threshold = float(bot_config.get("analog_turn_threshold", 12.0))
        self._analog_turn_emergency_threshold = float(bot_config.get("analog_turn_emergency_threshold", 82.0))
        self._analog_strafe_offset = float(bot_config.get("analog_strafe_offset", 22.0))
        self._analog_strafe_interval = float(bot_config.get("analog_strafe_interval", 0.28))
        self._analog_strafe_direction = 1
        self._analog_strafe_switch_time = time.time()
        self._analog_goal_hold_times = {
            "fog_escape": float(bot_config.get("analog_fog_hold_time", 0.06)),
            "retreat": float(bot_config.get("analog_retreat_hold_time", 0.12)),
            "engage": float(bot_config.get("analog_engage_hold_time", 0.16)),
            "memory_chase": float(bot_config.get("analog_memory_hold_time", 0.22)),
            "team_regroup": float(bot_config.get("analog_team_hold_time", 0.28)),
            "search": float(bot_config.get("analog_search_hold_time", 0.32)),
            "roam": float(bot_config.get("analog_roam_hold_time", 0.36)),
        }
        self._analog_goal_priorities = {
            "roam": 0,
            "search": 1,
            "team_regroup": 2,
            "memory_chase": 3,
            "engage": 4,
            "retreat": 5,
            "fog_escape": 6,
        }
        self._planned_analog_reason = None
        self._committed_analog_reason = None
        self._committed_analog_until = 0.0
        self.fix_angle_state = {
            "delay_to_trigger": bot_config["unstuck_movement_delay"],
            "duration": bot_config["unstuck_movement_hold_time"],
            "toggled": False,
            "started_at": time.time(),
            "fixed_angle": 0.0,
        }

    @staticmethod
    def _normalize_gamemode_name(gamemode):
        return str(gamemode or "").strip().lower().replace("_", " ")

    @classmethod
    def _is_showdown_behavior_enabled(cls, gamemode, gamemode_type):
        normalized = cls._normalize_gamemode_name(gamemode)
        return "showdown" in normalized and int(gamemode_type or 0) == 3

    @classmethod
    def _should_detect_walls_for_mode(cls, gamemode):
        normalized = cls._normalize_gamemode_name(gamemode)
        return "showdown" in normalized or normalized in {"brawlball", "brawl ball", "brawll ball"}

    @classmethod
    def _is_brawl_ball_mode(cls, gamemode):
        normalized = cls._normalize_gamemode_name(gamemode)
        return normalized in {"brawlball", "brawl ball", "brawll ball"}

    def _uses_analog_movement(self):
        return self.is_showdown_mode or self._is_brawl_ball_mode(self.selected_gamemode)

    @staticmethod
    def _opposite_key(key):
        return {"W": "S", "S": "W", "A": "D", "D": "A"}.get(key, "")

    @staticmethod
    def angle_from_direction(dx, dy):
        return math.degrees(math.atan2(dy, dx)) % 360

    @staticmethod
    def angle_opposite(angle_degrees):
        return (angle_degrees + 180.0) % 360.0

    @staticmethod
    def _angle_difference(first_angle, second_angle):
        return abs((float(first_angle) - float(second_angle) + 180.0) % 360.0 - 180.0)

    @staticmethod
    def get_enemy_pos(enemy):
        return (enemy[0] + enemy[2]) / 2, (enemy[1] + enemy[3]) / 2

    @staticmethod
    def get_player_pos(player_data):
        return (player_data[0] + player_data[2]) / 2, (player_data[1] + player_data[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

    def _build_showdown_fog_mask(self, frame, player_position):
        if frame is None or not self.is_showdown_mode:
            return None

        roi_radius = int(max(48.0, self._showdown_fog_flee_distance))
        cache_key = (
            id(frame),
            int(player_position[0]),
            int(player_position[1]),
            roi_radius,
        )
        if self._showdown_fog_mask_cache_key == cache_key:
            return self._showdown_fog_mask_cache_value

        frame_height, frame_width = frame.shape[:2]
        px, py = int(player_position[0]), int(player_position[1])
        x0 = max(0, px - roi_radius)
        y0 = max(0, py - roi_radius)
        x1 = min(frame_width, px + roi_radius + 1)
        y1 = min(frame_height, py + roi_radius + 1)
        if x0 >= x1 or y0 >= y1:
            self._showdown_fog_mask_cache_key = cache_key
            self._showdown_fog_mask_cache_value = None
            return None

        try:
            hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        except Exception:
            self._showdown_fog_mask_cache_key = cache_key
            self._showdown_fog_mask_cache_value = None
            return None

        mask = cv2.inRange(hsv, self._showdown_fog_low, self._showdown_fog_high)
        if int(cv2.countNonZero(mask)) < self._showdown_fog_pixels_minimum:
            self._showdown_fog_mask_cache_key = cache_key
            self._showdown_fog_mask_cache_value = None
            return None

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            self._showdown_fog_mask_cache_key = cache_key
            self._showdown_fog_mask_cache_value = None
            return None

        trusted_mask = np.zeros_like(mask)
        kept_any = False
        for label_idx in range(1, component_count):
            if stats[label_idx, cv2.CC_STAT_AREA] >= self._showdown_fog_min_blob_pixels:
                trusted_mask[labels == label_idx] = 255
                kept_any = True

        result = (trusted_mask, (x0, y0)) if kept_any and cv2.countNonZero(trusted_mask) > 0 else None
        self._showdown_fog_mask_cache_key = cache_key
        self._showdown_fog_mask_cache_value = result
        return result

    def _detect_showdown_fog_escape_angle(self, frame, player_position):
        built = self._build_showdown_fog_mask(frame, player_position)
        if built is None:
            return None

        mask, (origin_x, origin_y) = built
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None

        px, py = int(player_position[0]), int(player_position[1])
        dx_all = (xs + origin_x) - px
        dy_all = (ys + origin_y) - py
        dist_sq = dx_all * dx_all + dy_all * dy_all
        inside = dist_sq <= (self._showdown_fog_flee_distance * self._showdown_fog_flee_distance)
        if int(inside.sum()) < self._showdown_fog_min_pixels_in_radius:
            return None

        centroid_dx = float(dx_all[inside].mean())
        centroid_dy = float(dy_all[inside].mean())
        if math.hypot(centroid_dx, centroid_dy) < 1.0:
            return None
        return self.angle_opposite(self.angle_from_direction(centroid_dx, centroid_dy))

    def _get_showdown_fog_escape_move(self, player_data, wall_context):
        if not self.is_showdown_mode:
            self._showdown_fog_cached_angle = None
            return None

        current_frame = getattr(self, "current_frame", None)
        if current_frame is None:
            return None

        player_position = self.get_player_pos(player_data)
        self._showdown_fog_check_counter += 1
        if (
            self._showdown_fog_cached_angle is None
            or self._showdown_fog_check_counter >= self._showdown_fog_check_every_n_frames
        ):
            self._showdown_fog_cached_angle = self._detect_showdown_fog_escape_angle(current_frame, player_position)
            self._showdown_fog_check_counter = 0

        escape_angle = self._showdown_fog_cached_angle
        if escape_angle is None:
            return None

        escape_radians = math.radians(escape_angle)
        distance = max(
            150.0,
            self._showdown_fog_escape_distance * max(float(self.window_controller.scale_factor or 1.0), 0.75),
        )
        width = max(1, int(getattr(self.window_controller, "width", 1920) or 1920))
        height = max(1, int(getattr(self.window_controller, "height", 1080) or 1080))
        margin = 24
        target_pos = (
            min(width - margin, max(margin, player_position[0] + math.cos(escape_radians) * distance)),
            min(height - margin, max(margin, player_position[1] + math.sin(escape_radians) * distance)),
        )
        return self._get_move_toward(player_position, target_pos, wall_context, allow_detour=True)

    @staticmethod
    def get_horizontal_move_key(direction_x, opposite=False):
        if opposite:
            return "A" if direction_x > 0 else "D"
        return "D" if direction_x > 0 else "A"

    @staticmethod
    def get_vertical_move_key(direction_y, opposite=False):
        if opposite:
            return "W" if direction_y > 0 else "S"
        return "S" if direction_y > 0 else "W"

    def attack(self):
        self.window_controller.press_key("M")

    def use_hypercharge(self):
        print("Using hypercharge")
        self.window_controller.press_key("H")

    def use_gadget(self):
        print("Using gadget")
        self.window_controller.press_key("G")

    def use_super(self):
        print("Using super")
        self.window_controller.press_key("E")

    @staticmethod
    def get_random_attack_key():
        random_movement = random.choice(["A", "W", "S", "D"])
        random_movement += random.choice(["A", "W", "S", "D"])
        return random_movement

    @staticmethod
    def reverse_movement(movement):
        # Create a translation table
        movement = movement.lower()
        translation_table = str.maketrans("wasd", "sdwa")
        return movement.translate(translation_table)

    def unstuck_movement_if_needed(self, movement, current_time=None, player_pos=None):
        if current_time is None:
            current_time = time.time()
        movement = movement.lower()
        if self.fix_movement_keys['toggled']:
            if current_time - self.fix_movement_keys['started_at'] > self.fix_movement_keys['duration']:
                self.fix_movement_keys['toggled'] = False

            return self.fix_movement_keys['fixed']

        if "".join(self.keys_hold) != movement and movement[::-1] != "".join(self.keys_hold):
            self.time_since_different_movement = current_time
            self._movement_anchor_command = movement
            self._movement_anchor_pos = player_pos
        elif player_pos is not None:
            if self._movement_anchor_command != movement or self._movement_anchor_pos is None:
                self._movement_anchor_command = movement
                self._movement_anchor_pos = player_pos
                self.time_since_different_movement = current_time
            elif self.get_distance(self._movement_anchor_pos, player_pos) >= self._unstuck_progress_distance:
                self._movement_anchor_pos = player_pos
                self.time_since_different_movement = current_time

        # print(f"Last change: {self.time_since_different_movement}", f" self.hold: {self.keys_hold}",f" c movement: {movement}")
        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            reversed_movement = self.reverse_movement(movement)

            if reversed_movement == "s":
                reversed_movement = random.choice(['aw', 'dw'])
            elif reversed_movement == "w":
                reversed_movement = random.choice(['as', 'ds'])

            """
            If reverse movement is either "w" or "s" it means the bot is stuck
            going forward or backward. This happens when it doesn't detect a wall in front
            so to go around it it could either go to the left diagonal or right
            """

            self.fix_movement_keys['fixed'] = reversed_movement
            self.fix_movement_keys['toggled'] = True
            self.fix_movement_keys['started_at'] = current_time
            #print(f"REVERSED! from {movement} to {reversed_movement}!")
            return reversed_movement

        return movement


class Play(Movement):

    def __init__(self, main_info_model, tile_detector_model, window_controller):
        super().__init__(window_controller)

        bot_config = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")

        self.Detect_main_info = Detect(main_info_model, classes=['enemy', 'teammate', 'player'])
        self.tile_detector_model_classes = bot_config["wall_model_classes"]
        self.Detect_tile_detector = Detect(
            tile_detector_model,
            classes=self.tile_detector_model_classes
        )

        self.time_since_movement = time.time()
        self.time_since_gadget_checked = time.time()
        self.time_since_hypercharge_checked = time.time()
        self.time_since_super_checked = time.time()
        self.time_since_walls_checked = 0
        self.time_since_movement_change = time.time()
        self.time_since_player_last_found = time.time()
        self.current_brawler = None
        self.is_hypercharge_ready = False
        self.is_gadget_ready = False
        self.is_super_ready = False
        self.brawlers_info = load_brawlers_info()
        self.brawler_ranges = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy": time.time(),
        }
        self.time_since_last_proceeding = time.time()

        self.last_movement = ''
        self.last_movement_time = time.time()
        self.wall_history = []
        self.wall_history_length = 3  # Number of frames to keep walls
        self.wall_context = {
            "signature": None,
            "rectangles": [],
            "line_cache": {},
            "enemy_hittable_cache": {},
        }
        self.hud_regions = {
            "hud_bounds": (1350, 830, 1700, 1050),
            "hypercharge": (1350, 940, 1450, 1050),
            "gadget": (1580, 930, 1700, 1050),
            "super": (1460, 830, 1560, 930),
        }
        self.hud_hsv_ranges = {
            "hypercharge": (
                np.array((137, 158, 159), dtype=np.uint8),
                np.array((179, 255, 255), dtype=np.uint8),
            ),
            "gadget": (
                np.array((57, 219, 165), dtype=np.uint8),
                np.array((62, 255, 255), dtype=np.uint8),
            ),
            "super": (
                np.array((17, 170, 200), dtype=np.uint8),
                np.array((27, 255, 255), dtype=np.uint8),
            ),
        }
        self._scaled_hud_cache = None
        self.scene_data = []
        self.should_detect_walls = self.is_showdown_mode or self._should_detect_walls_for_mode(self.selected_gamemode)
        self.minimum_movement_delay = bot_config["minimum_movement_delay"]
        self.no_detection_proceed_delay = time_config["no_detection_proceed"]
        self.gadget_pixels_minimum = bot_config["gadget_pixels_minimum"]
        self.hypercharge_pixels_minimum = bot_config["hypercharge_pixels_minimum"]
        self.super_pixels_minimum = bot_config["super_pixels_minimum"]
        self.wall_detection_confidence = bot_config["wall_detection_confidence"]
        self.entity_detection_confidence = bot_config["entity_detection_confidence"]
        self._runtime_state = "starting"
        self._last_state_probe_runtime = ""
        self._last_state_probe_result = ""
        self._last_state_probe_time = 0.0
        self._match_state_grace_until = 0.0
        self._last_state_guard_log_time = 0.0
        self._state_guard_detected_frames = 0
        self._last_state_guard_detection_time = 0.0
        self._last_no_player_log_time = 0.0
        self._last_end_result_probe_time = 0.0
        self._pending_end_result = None
        self._last_confirmed_match_time = 0.0
        self._last_match_evidence_time = 0.0
        self._last_valid_player_bbox = None
        self._last_valid_player_seen_at = 0.0
        self._teammate_positions = []
        self._last_known_enemies = []
        self._enemy_memory_duration = float(bot_config.get("enemy_memory_duration", 2.0))
        self._last_enemy_seen_at = time.time()
        self._search_target_idx = 0
        self._search_target_switch_time = 0.0

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        screen_size_ratio = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            attack_range = info['attack_range']
            safe_range = info['safe_range']
            super_range = info['super_range']
            v = [safe_range, attack_range, super_range]
            ranges[brawler] = [int(v[0] * screen_size_ratio), int(v[1] * screen_size_ratio), int(v[2] * screen_size_ratio)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        if skill_type == "attack":
            return brawlers_info[brawler]['ignore_walls_for_attacks']
        elif skill_type == "super":
            return brawlers_info[brawler]['ignore_walls_for_supers']
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def _entity_count(data, key):
        if not data:
            return 0
        value = data.get(key)
        if not value:
            return 0
        try:
            return len(value)
        except TypeError:
            return 1

    def _reset_match_state_guard(self):
        self._state_guard_detected_frames = 0
        self._last_state_guard_detection_time = 0.0

    def note_confirmed_match_state(self, current_time=None):
        self._last_confirmed_match_time = current_time if current_time is not None else time.time()

    def has_recent_match_context(self, current_time=None):
        now = current_time if current_time is not None else time.time()
        if now - self._last_confirmed_match_time > 1.0:
            return False
        if now - self._last_match_evidence_time > 0.45:
            return False
        return True

    def _is_plausible_player_detection(self, data):
        players = (data or {}).get("player") or []
        if not players:
            return False
        try:
            x1, y1, x2, y2 = [float(value) for value in players[0][:4]]
        except Exception:
            return False
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        area = width * height
        scale = max(float(self.window_controller.scale_factor or 1.0), 0.5)
        return (
            width >= (55.0 * scale)
            and height >= (78.0 * scale)
            and area >= (6500.0 * scale)
        )

    def _should_promote_detected_match(self, data, runtime_state, checked_state, current_time):
        player_count = self._entity_count(data, "player")
        if player_count <= 0 or not self._is_plausible_player_detection(data):
            self._reset_match_state_guard()
            return False

        enemy_count = self._entity_count(data, "enemy")
        teammate_count = self._entity_count(data, "teammate")
        supporting_entities = enemy_count + teammate_count

        if checked_state == "match":
            self._reset_match_state_guard()
            return True

        if runtime_state not in {"starting", "lobby"}:
            self._reset_match_state_guard()
            return False

        if current_time - self._last_state_guard_detection_time > 1.0:
            self._state_guard_detected_frames = 0

        self._last_state_guard_detection_time = current_time
        self._state_guard_detected_frames += 1

        if supporting_entities > 0:
            if debug:
                print(
                    "[MATCH] promoting to match from gameplay detections "
                    f"(player={player_count}, enemy={enemy_count}, teammate={teammate_count}, state={checked_state})"
                )
            return True

        required_frames = 4 if checked_state == "lobby" else 3
        if checked_state in {"", "lobby", "starting"} and self._state_guard_detected_frames >= required_frames:
            if debug:
                print(
                    "[MATCH] promoting after repeated player detections "
                    f"(frames={self._state_guard_detected_frames}, state={checked_state})"
                )
            return True

        return False

    def _remember_player_detection(self, data, current_time):
        players = (data or {}).get("player") or []
        if not players or not self._is_plausible_player_detection(data):
            return
        try:
            self._last_valid_player_bbox = [float(value) for value in players[0][:4]]
            self._last_valid_player_seen_at = current_time
        except Exception:
            pass

    def _restore_recent_player_detection(self, data, current_time, runtime_state):
        if not isinstance(data, dict):
            return data
        if data.get("player"):
            return data
        if not self._last_valid_player_bbox:
            return data

        player_age = current_time - self._last_valid_player_seen_at
        if player_age > 0.8:
            return data

        supporting_entities = self._entity_count(data, "enemy") + self._entity_count(data, "teammate")
        has_recent_match = (
            runtime_state == "match"
            and (current_time - self._last_confirmed_match_time) <= 1.0
        )
        allow_restore = False
        if supporting_entities > 0 and has_recent_match:
            allow_restore = True
        elif has_recent_match and player_age <= 0.35:
            allow_restore = True

        if not allow_restore:
            return data

        restored = dict(data)
        restored["player"] = [list(self._last_valid_player_bbox)]
        if debug and (current_time - self._last_no_player_log_time >= 1.5):
            print(
                "[MATCH] reusing recent player detection "
                f"(age={player_age:.2f}s, enemy={self._entity_count(data, 'enemy')}, teammate={self._entity_count(data, 'teammate')})"
            )
            self._last_no_player_log_time = current_time
        return restored


    def get_wall_context(self, walls):
        signature = tuple(tuple(wall) for wall in walls)
        if self.wall_context["signature"] != signature:
            self.wall_context = {
                "signature": signature,
                "rectangles": signature,
                "line_cache": {},
                "enemy_hittable_cache": {},
            }
        else:
            self.wall_context["line_cache"].clear()
            self.wall_context["enemy_hittable_cache"].clear()
        return self.wall_context

    @staticmethod
    def _line_cache_key(start_pos, end_pos):
        return (start_pos, end_pos)

    @staticmethod
    def _segment_intersects_rect(start_pos, end_pos, rect):
        x0, y0 = start_pos
        x1, y1 = end_pos
        rx1, ry1, rx2, ry2 = rect
        min_x, max_x = sorted((rx1, rx2))
        min_y, max_y = sorted((ry1, ry2))

        if (min_x <= x0 <= max_x and min_y <= y0 <= max_y) or (min_x <= x1 <= max_x and min_y <= y1 <= max_y):
            return True

        dx = x1 - x0
        dy = y1 - y0
        p = (-dx, dx, -dy, dy)
        q = (x0 - min_x, max_x - x0, y0 - min_y, max_y - y0)

        u1, u2 = 0.0, 1.0
        for pi, qi in zip(p, q):
            if pi == 0:
                if qi < 0:
                    return False
                continue

            t = qi / pi
            if pi < 0:
                u1 = max(u1, t)
            else:
                u2 = min(u2, t)
            if u1 > u2:
                return False

        return True

    def walls_are_in_line_of_sight(self, start_pos, end_pos, wall_context):
        if not wall_context["rectangles"]:
            return False

        cache_key = self._line_cache_key(start_pos, end_pos)
        reverse_key = self._line_cache_key(end_pos, start_pos)
        cached = wall_context["line_cache"].get(cache_key)
        if cached is None:
            cached = wall_context["line_cache"].get(reverse_key)
        if cached is not None:
            return cached

        intersects = any(
            self._segment_intersects_rect(start_pos, end_pos, wall_rect)
            for wall_rect in wall_context["rectangles"]
        )
        wall_context["line_cache"][cache_key] = intersects
        return intersects

    def no_enemy_movement(self, player_data, wall_context):
        player_position = self.get_player_pos(player_data)
        if self.is_showdown_mode:
            fog_escape_move = self._get_showdown_fog_escape_move(player_data, wall_context)
            if fog_escape_move:
                return self._plan_analog_reason(fog_escape_move, "fog_escape")

            regroup_target, regroup_distance = self._get_showdown_regroup_target(player_position)
            should_regroup = regroup_target and self._should_hold_showdown_regroup(regroup_distance)
            if should_regroup:
                regroup_move = self._get_move_toward(player_position, regroup_target, wall_context, allow_detour=True)
                if regroup_move:
                    return self._plan_analog_reason(regroup_move, "team_regroup")

            teammate_target = None if regroup_target else self._get_teammate_centroid()
            if teammate_target and self.get_distance(teammate_target, player_position) > self._showdown_regroup_distance:
                regroup_move = self._get_move_toward(player_position, teammate_target, wall_context, allow_detour=True)
                if regroup_move:
                    return self._plan_analog_reason(regroup_move, "team_regroup")

            last_enemy_pos = self._get_last_known_enemy_pos()
            if last_enemy_pos is not None and (time.time() - self._last_enemy_seen_at) <= self._showdown_enemy_chase_timeout:
                chase_move = self._get_move_toward(player_position, last_enemy_pos, wall_context, allow_detour=True)
                if chase_move:
                    return self._plan_analog_reason(chase_move, "memory_chase")

            search_move = self._get_search_movement(player_position, wall_context)
            if search_move:
                return self._plan_analog_reason(search_move, "search")

        if self._uses_analog_movement():
            desired_angle = 270.0 if self.game_mode == 3 else 0.0
            return self._plan_analog_reason(
                self._find_best_angle(player_position, desired_angle, wall_context),
                "roam",
            )

        preferred_movement = 'W' if self.game_mode == 3 else 'D'  # Adjust based on game mode

        if not self.is_path_blocked(player_position, preferred_movement, wall_context):
            return preferred_movement
        else:
            # Try alternative movements
            alternative_moves = ['W', 'A', 'S', 'D']
            alternative_moves.remove(preferred_movement)
            random.shuffle(alternative_moves)
            for move in alternative_moves:
                if not self.is_path_blocked(player_position, move, wall_context):
                    return move
            print("no movement possible ?")
            # If no movement is possible, return empty string
            return preferred_movement

    def _build_target_move_candidates(self, direction_x, direction_y, allow_detour=False):
        horizontal = "D" if direction_x > 20 else "A" if direction_x < -20 else ""
        vertical = "S" if direction_y > 20 else "W" if direction_y < -20 else ""
        candidates = [
            vertical + horizontal,
            horizontal + vertical,
            vertical,
            horizontal,
        ]

        if allow_detour:
            opposite_horizontal = self._opposite_key(horizontal)
            opposite_vertical = self._opposite_key(vertical)
            dominant_horizontal = abs(direction_x) >= abs(direction_y)

            if dominant_horizontal:
                candidates.extend([
                    horizontal + opposite_vertical,
                    opposite_vertical + horizontal,
                    opposite_horizontal,
                ])
            else:
                candidates.extend([
                    vertical + opposite_horizontal,
                    opposite_horizontal + vertical,
                    opposite_vertical,
                ])

            candidates.extend([
                opposite_horizontal + opposite_vertical,
                opposite_vertical + opposite_horizontal,
                opposite_horizontal,
                opposite_vertical,
                "W",
                "A",
                "S",
                "D",
            ])

        unique_candidates = []
        for move in candidates:
            normalized = "".join(ch for ch in str(move or "").upper() if ch in "WASD")
            if not normalized or normalized in unique_candidates:
                continue
            unique_candidates.append(normalized)
        return unique_candidates

    def _get_move_toward(self, player_pos, target_pos, wall_context, allow_detour=False):
        direction_x = target_pos[0] - player_pos[0]
        direction_y = target_pos[1] - player_pos[1]
        if self._uses_analog_movement():
            desired_angle = self.angle_from_direction(direction_x, direction_y)
            if allow_detour:
                return self._find_best_angle(player_pos, desired_angle, wall_context)
            return desired_angle
        for move in self._build_target_move_candidates(direction_x, direction_y, allow_detour=allow_detour):
            if move and not self.is_path_blocked(player_pos, move, wall_context):
                return move
        return None

    def _is_path_blocked_angle(self, player_pos, angle_degrees, wall_context, distance=None):
        if distance is None:
            distance = self.TILE_SIZE * self.window_controller.scale_factor
        angle_rad = math.radians(float(angle_degrees) % 360.0)
        for probe_distance in (distance * 0.5, distance):
            new_pos = (
                player_pos[0] + math.cos(angle_rad) * probe_distance,
                player_pos[1] + math.sin(angle_rad) * probe_distance,
            )
            if self.walls_are_in_line_of_sight(player_pos, new_pos, wall_context):
                return True
        return False

    def _find_best_angle(self, player_pos, desired_angle, wall_context, sweep_range=165, step=10):
        if not self.should_detect_walls or not wall_context.get("rectangles"):
            return float(desired_angle) % 360.0

        if not self._is_path_blocked_angle(player_pos, desired_angle, wall_context):
            return float(desired_angle) % 360.0

        for offset in range(step, sweep_range + 1, step):
            for sign in (1, -1):
                candidate = (float(desired_angle) + sign * offset) % 360.0
                if not self._is_path_blocked_angle(player_pos, candidate, wall_context):
                    return candidate

        return float(desired_angle) % 360.0

    def is_enemy_hittable(self, player_pos, enemy_pos, wall_context, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        cache_key = (skill_type, player_pos, enemy_pos)
        cache = wall_context["enemy_hittable_cache"]
        if cache_key in cache:
            return cache[cache_key]

        hittable = not self.walls_are_in_line_of_sight(player_pos, enemy_pos, wall_context)
        cache[cache_key] = hittable
        return hittable

    def find_closest_enemy(self, enemy_data, player_coords, wall_context, skill_type):
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, wall_context, skill_type):
                if distance < closest_hittable_distance:
                    closest_hittable_distance = distance
                    closest_hittable = [enemy_pos, distance]
            else:
                if distance < closest_unhittable_distance:
                    closest_unhittable_distance = distance
                    closest_unhittable = [enemy_pos, distance]
        if closest_hittable:
            return closest_hittable
        elif closest_unhittable:
            return closest_unhittable

        return None, None

    def get_main_data(self, frame):
        data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        return data

    def is_path_blocked(self, player_pos, move_direction, wall_context, distance=None):  # Increased distance
        if distance is None:
            distance = self.TILE_SIZE*self.window_controller.scale_factor
        dx, dy = 0, 0
        if 'w' in move_direction.lower():
            dy -= distance
        if 's' in move_direction.lower():
            dy += distance
        if 'a' in move_direction.lower():
            dx -= distance
        if 'd' in move_direction.lower():
            dx += distance
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        return self.walls_are_in_line_of_sight(player_pos, new_pos, wall_context)

    @staticmethod
    def validate_game_data(data):
        incomplete = False
        if "player" not in data.keys():
            incomplete = True  # This is required so track_no_detections can also keep track if enemy is missing

        if "enemy" not in data.keys():
            data['enemy'] = None

        if 'wall' not in data.keys() or not data['wall']:
            data['wall'] = []

        return False if incomplete else data

    @staticmethod
    def _is_non_blocking_tile_class(class_name):
        return "bush" in class_name.lower()

    def track_no_detections(self, data):
        if not data:
            data = {
                "enemy": None,
                "player": None
            }
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = time.time()

    def do_movement(self, movement):
        if isinstance(movement, (float, int)):
            self.window_controller.move_joystick_angle(float(movement), radius=self._analog_movement_radius)
            self.keys_hold = []
            return

        movement = movement.lower()
        keys_to_keyDown = []
        keys_to_keyUp = []
        for key in ['w', 'a', 's', 'd']:
            if key in movement:
                keys_to_keyDown.append(key)
            else:
                keys_to_keyUp.append(key)

        if keys_to_keyDown:
            self.window_controller.keys_down(keys_to_keyDown)

        self.window_controller.keys_up(keys_to_keyUp)

        self.keys_hold = keys_to_keyDown

    def _debounce_angle(self, angle):
        if self.last_movement is None or not isinstance(self.last_movement, (float, int)):
            self.last_movement = float(angle)
            self.last_movement_time = time.time()
            return float(angle)

        if self._angle_difference(angle, self.last_movement) > self._analog_turn_threshold:
            self.last_movement = float(angle)
            self.last_movement_time = time.time()

        return float(self.last_movement)

    def unstuck_angle_if_needed(self, angle, current_time=None, player_pos=None):
        if current_time is None:
            current_time = time.time()
        angle = float(angle) % 360.0
        state = self.fix_angle_state

        if state["toggled"]:
            if current_time - state["started_at"] > state["duration"]:
                state["toggled"] = False
                self._movement_anchor_angle = angle
                self._movement_anchor_angle_pos = player_pos
                self.time_since_different_movement = current_time
            else:
                return float(state["fixed_angle"])

        if (
            self._movement_anchor_angle is None
            or self._angle_difference(angle, self._movement_anchor_angle) > 18.0
        ):
            self._movement_anchor_angle = angle
            self._movement_anchor_angle_pos = player_pos
            self.time_since_different_movement = current_time
            return angle

        if (
            player_pos is not None
            and self._movement_anchor_angle_pos is not None
            and self.get_distance(self._movement_anchor_angle_pos, player_pos) >= self._unstuck_progress_distance
        ):
            self._movement_anchor_angle_pos = player_pos
            self.time_since_different_movement = current_time
            return angle

        if current_time - self.time_since_different_movement > state["delay_to_trigger"]:
            offset = random.choice((125.0, -125.0))
            fixed_angle = (angle + offset) % 360.0
            state["fixed_angle"] = fixed_angle
            state["toggled"] = True
            state["started_at"] = current_time
            return fixed_angle

        return angle

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        return self.brawler_ranges[brawler]

    def loop(self, brawler, data, current_time, wall_context):
        player_pos = self.get_player_pos(data['player'][0])
        movement = self.get_movement(
            player_data=data['player'][0],
            enemy_data=data['enemy'],
            wall_context=wall_context,
            brawler=brawler
        )
        current_time = time.time()
        if current_time - self.time_since_movement > self.minimum_movement_delay:
            if isinstance(movement, (float, int)):
                movement = self._stabilize_analog_angle(movement, current_time)
                movement = self._debounce_angle(movement)
                movement = self.unstuck_angle_if_needed(movement, current_time, player_pos)
            else:
                self._planned_analog_reason = None
                movement = self.unstuck_movement_if_needed(movement, current_time, player_pos)
            self.do_movement(movement)
            self.time_since_movement = time.time()
        return movement

    def _add_strafe_angle(self, angle, current_time):
        if current_time - self._analog_strafe_switch_time >= self._analog_strafe_interval:
            self._analog_strafe_direction *= -1
            self._analog_strafe_switch_time = current_time
        return (float(angle) + (self._analog_strafe_direction * self._analog_strafe_offset)) % 360.0

    def _get_analog_engagement_move(self, player_pos, direction_x, direction_y, wall_context, retreat, current_time, should_strafe=False):
        desired_angle = self.angle_from_direction(direction_x, direction_y)
        if retreat:
            desired_angle = self.angle_opposite(desired_angle)

        movement = self._find_best_angle(player_pos, desired_angle, wall_context)
        if should_strafe:
            movement = self._add_strafe_angle(movement, current_time)
            movement = self._find_best_angle(player_pos, movement, wall_context)
        return movement

    def scale_region(self, region):
        x1, y1, x2, y2 = region
        return (
            int(x1 * self.window_controller.width_ratio),
            int(y1 * self.window_controller.height_ratio),
            int(x2 * self.window_controller.width_ratio),
            int(y2 * self.window_controller.height_ratio),
        )

    def _get_scaled_hud_cache(self):
        cache_key = (
            self.window_controller.width_ratio,
            self.window_controller.height_ratio,
        )
        if self._scaled_hud_cache is None or self._scaled_hud_cache["key"] != cache_key:
            scaled_regions = {
                name: self.scale_region(region)
                for name, region in self.hud_regions.items()
            }
            self._scaled_hud_cache = {
                "key": cache_key,
                "regions": scaled_regions,
            }
        return self._scaled_hud_cache["regions"]

    def count_hud_hsv_pixels(self, hsv_frame, scaled_region, low_hsv, high_hsv, origin=(0, 0)):
        sx1, sy1, sx2, sy2 = scaled_region
        ox, oy = origin
        region_hsv = hsv_frame[sy1 - oy:sy2 - oy, sx1 - ox:sx2 - ox]
        mask = cv2.inRange(
            region_hsv,
            low_hsv,
            high_hsv
        )
        return int(np.count_nonzero(mask))

    def get_hud_hsv(self, frame):
        scaled_regions = self._get_scaled_hud_cache()
        sx1, sy1, sx2, sy2 = scaled_regions["hud_bounds"]
        hud_frame = frame[sy1:sy2, sx1:sx2]
        return cv2.cvtColor(hud_frame, cv2.COLOR_BGR2HSV), (sx1, sy1)

    def check_if_hypercharge_ready(self, hsv_frame, hsv_origin):
        scaled_regions = self._get_scaled_hud_cache()
        low_hsv, high_hsv = self.hud_hsv_ranges["hypercharge"]
        purple_pixels = self.count_hud_hsv_pixels(
            hsv_frame,
            scaled_regions["hypercharge"],
            low_hsv,
            high_hsv,
            origin=hsv_origin
        )
        return purple_pixels > self.hypercharge_pixels_minimum

    def check_if_gadget_ready(self, hsv_frame, hsv_origin):
        scaled_regions = self._get_scaled_hud_cache()
        low_hsv, high_hsv = self.hud_hsv_ranges["gadget"]
        green_pixels = self.count_hud_hsv_pixels(
            hsv_frame,
            scaled_regions["gadget"],
            low_hsv,
            high_hsv,
            origin=hsv_origin
        )
        return green_pixels > self.gadget_pixels_minimum

    def check_if_super_ready(self, hsv_frame, hsv_origin):
        scaled_regions = self._get_scaled_hud_cache()
        low_hsv, high_hsv = self.hud_hsv_ranges["super"]
        yellow_pixels = self.count_hud_hsv_pixels(
            hsv_frame,
            scaled_regions["super"],
            low_hsv,
            high_hsv,
            origin=hsv_origin
        )
        return yellow_pixels > self.super_pixels_minimum

    def get_tile_data(self, frame):
        tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self.wall_detection_confidence)
        return tile_data

    def process_tile_data(self, tile_data):
        walls = []
        for class_name, boxes in tile_data.items():
            if not self._is_non_blocking_tile_class(class_name):
                walls.extend(boxes)

        # Add walls to history
        self.wall_history.append(walls)
        if len(self.wall_history) > self.wall_history_length:
            self.wall_history.pop(0)
        # Combine walls from history
        combined_walls = self.combine_walls_from_history()

        return combined_walls

    def combine_walls_from_history(self):
        wall_counts = {}
        for walls in self.wall_history:
            for wall in walls:
                wall_key = tuple(wall)
                wall_counts[wall_key] = wall_counts.get(wall_key, 0) + 1

        threshold = 1

        combined_walls = [list(wall) for wall, count in wall_counts.items() if count >= threshold]
        # print(f"Combined walls: {combined_walls}")

        return combined_walls

    def _update_enemy_memory(self, enemies):
        now = time.time()
        for enemy in enemies:
            enemy_pos = self.get_enemy_pos(enemy)
            self._last_known_enemies.append((enemy_pos[0], enemy_pos[1], now))

        self._last_known_enemies = [
            (x, y, seen_at)
            for x, y, seen_at in self._last_known_enemies
            if now - seen_at < self._enemy_memory_duration
        ]

    def _get_last_known_enemy_pos(self):
        if not self._last_known_enemies:
            return None
        newest = max(self._last_known_enemies, key=lambda entry: entry[2])
        return newest[0], newest[1]

    def _get_teammate_centroid(self):
        if not self._teammate_positions:
            return None
        centroid_x = sum(position[0] for position in self._teammate_positions) / len(self._teammate_positions)
        centroid_y = sum(position[1] for position in self._teammate_positions) / len(self._teammate_positions)
        return centroid_x, centroid_y

    def _get_nearest_teammate_target(self, player_pos):
        if not self._teammate_positions:
            return None, float("inf")
        nearest_target = min(
            self._teammate_positions,
            key=lambda teammate_pos: self.get_distance(teammate_pos, player_pos),
        )
        return nearest_target, self.get_distance(nearest_target, player_pos)

    def _reset_showdown_teammate_lock(self):
        self._showdown_locked_teammate = None
        self._showdown_locked_teammate_distance = float("inf")
        self._showdown_regroup_active = False

    def _get_showdown_regroup_target(self, player_pos):
        if not self._teammate_positions:
            self._reset_showdown_teammate_lock()
            return None, float("inf")

        nearest_target, nearest_distance = self._get_nearest_teammate_target(player_pos)
        selected_target = nearest_target
        selected_distance = nearest_distance

        if self._showdown_locked_teammate is not None:
            locked_target = min(
                self._teammate_positions,
                key=lambda teammate_pos: self.get_distance(teammate_pos, self._showdown_locked_teammate),
            )
            locked_distance = self.get_distance(locked_target, player_pos)
            if nearest_distance >= locked_distance * (1.0 - self._showdown_teammate_hysteresis):
                selected_target = locked_target
                selected_distance = locked_distance

        self._showdown_locked_teammate = selected_target
        self._showdown_locked_teammate_distance = selected_distance
        return selected_target, selected_distance

    def _should_hold_showdown_regroup(self, teammate_distance):
        if teammate_distance == float("inf"):
            self._showdown_regroup_active = False
            return False

        start_distance = max(self._showdown_team_pull_distance, self._showdown_regroup_release_distance + 10.0)
        if self._showdown_regroup_active:
            self._showdown_regroup_active = teammate_distance > self._showdown_regroup_release_distance
        else:
            self._showdown_regroup_active = teammate_distance > start_distance
        return self._showdown_regroup_active

    def _plan_analog_reason(self, movement, reason):
        if isinstance(movement, (float, int)):
            self._planned_analog_reason = reason
        return movement

    def _stabilize_analog_angle(self, angle, current_time):
        angle = float(angle) % 360.0
        reason = self._planned_analog_reason or "engage"
        self._planned_analog_reason = None

        if self.last_movement is None or not isinstance(self.last_movement, (float, int)):
            self._committed_analog_reason = reason
            self._committed_analog_until = current_time + self._analog_goal_hold_times.get(reason, 0.16)
            return angle

        angle_delta = self._angle_difference(angle, self.last_movement)
        current_reason = self._committed_analog_reason or reason
        current_priority = self._analog_goal_priorities.get(current_reason, 0)
        next_priority = self._analog_goal_priorities.get(reason, 0)

        if current_time < self._committed_analog_until:
            if next_priority < current_priority and angle_delta < self._analog_turn_emergency_threshold:
                return float(self.last_movement)
            if next_priority == current_priority and angle_delta < self._analog_turn_emergency_threshold:
                return float(self.last_movement)

        self._committed_analog_reason = reason
        self._committed_analog_until = current_time + self._analog_goal_hold_times.get(reason, 0.16)
        return angle

    def _get_search_targets(self):
        width = self.window_controller.width
        height = self.window_controller.height
        if self.is_showdown_mode:
            return [
                (width * 0.50, height * 0.28),
                (width * 0.32, height * 0.38),
                (width * 0.68, height * 0.38),
                (width * 0.40, height * 0.56),
                (width * 0.60, height * 0.56),
                (width * 0.50, height * 0.68),
            ]
        if self.game_mode == 3:
            return [
                (width * 0.5, height * 0.22),
                (width * 0.3, height * 0.28),
                (width * 0.7, height * 0.28),
            ]
        return [
            (width * 0.78, height * 0.5),
            (width * 0.72, height * 0.3),
            (width * 0.72, height * 0.7),
        ]

    def _get_search_movement(self, player_pos, wall_context):
        targets = self._get_search_targets()
        if not targets:
            return None

        now = time.time()
        if self._search_target_switch_time == 0.0:
            self._search_target_idx = min(
                range(len(targets)),
                key=lambda idx: self.get_distance(targets[idx], player_pos),
            )
            self._search_target_switch_time = now
        idx = self._search_target_idx % len(targets)
        target = targets[idx]
        target_distance = self.get_distance(target, player_pos)
        stale_target = (now - self._search_target_switch_time) > self._search_target_hold_time

        if target_distance < 90 or stale_target:
            self._search_target_idx = (self._search_target_idx + 1) % len(targets)
            self._search_target_switch_time = now
            target = targets[self._search_target_idx]

        return self._get_move_toward(player_pos, target, wall_context, allow_detour=self.is_showdown_mode)

    def get_movement(self, player_data, enemy_data, wall_context, brawler):
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")
        safe_range, attack_range, super_range = self.get_brawler_range(brawler)

        player_pos = self.get_player_pos(player_data)
        if not self.is_there_enemy(enemy_data):
            if self.is_showdown_mode:
                team_regroup_target, team_regroup_distance = self._get_showdown_regroup_target(player_pos)
                should_team_regroup = team_regroup_target and self._should_hold_showdown_regroup(team_regroup_distance)
            else:
                team_regroup_target = self._get_teammate_centroid()
                team_regroup_distance = (
                    self.get_distance(team_regroup_target, player_pos) if team_regroup_target else float("inf")
                )
                should_team_regroup = team_regroup_target and team_regroup_distance > 160
            if should_team_regroup:
                team_move = self._get_move_toward(
                    player_pos,
                    team_regroup_target,
                    wall_context,
                    allow_detour=self.is_showdown_mode,
                )
                if team_move:
                    return self._plan_analog_reason(team_move, "team_regroup")

            last_enemy_pos = self._get_last_known_enemy_pos()
            if last_enemy_pos is not None:
                memory_move = self._get_move_toward(
                    player_pos,
                    last_enemy_pos,
                    wall_context,
                    allow_detour=self.is_showdown_mode,
                )
                if memory_move:
                    return self._plan_analog_reason(memory_move, "memory_chase")

            search_move = self._get_search_movement(player_pos, wall_context)
            if search_move:
                return self._plan_analog_reason(search_move, "search")
            return self.no_enemy_movement(player_data, wall_context)
        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, wall_context, "attack")
        if enemy_coords is None:
            return self.no_enemy_movement(player_data, wall_context)
        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]

        if self.is_showdown_mode:
            cohesion_target, _ = self._get_showdown_regroup_target(player_pos)
        else:
            cohesion_target = self._get_teammate_centroid()
        if cohesion_target:
            teammate_distance = self.get_distance(cohesion_target, player_pos)
            if teammate_distance > 100:
                cohesion_angle_gap = self._angle_difference(
                    self.angle_from_direction(direction_x, direction_y),
                    self.angle_from_direction(
                        cohesion_target[0] - player_pos[0],
                        cohesion_target[1] - player_pos[1],
                    ),
                )
                dist_factor = min(1.0, (teammate_distance - 100) / 300.0)
                cohesion_strength = 0.15 * dist_factor
                if self.is_showdown_mode:
                    cohesion_strength = max(cohesion_strength, min(0.4, 0.16 + (dist_factor * 0.2)))
                    if enemy_distance <= attack_range * 1.15:
                        cohesion_strength *= 0.30
                    elif cohesion_angle_gap > 70.0:
                        cohesion_strength *= 0.45
                    if teammate_distance <= max(180.0, self._showdown_team_pull_distance * 1.4):
                        cohesion_strength *= 0.55
                direction_x = direction_x * (1.0 - cohesion_strength) + (cohesion_target[0] - player_pos[0]) * cohesion_strength
                direction_y = direction_y * (1.0 - cohesion_strength) + (cohesion_target[1] - player_pos[1]) * cohesion_strength

        current_time = time.time()
        if self._uses_analog_movement():
            movement = self._get_analog_engagement_move(
                player_pos,
                direction_x,
                direction_y,
                wall_context,
                retreat=enemy_distance <= safe_range,
                current_time=current_time,
                should_strafe=(
                    enemy_distance > safe_range * 0.8
                    and enemy_distance <= attack_range * 1.05
                ),
            )
            movement = self._plan_analog_reason(
                movement,
                "retreat" if enemy_distance <= safe_range else "engage",
            )
        else:
            # Determine initial movement direction
            if enemy_distance > safe_range:  # Move towards the enemy
                move_horizontal = self.get_horizontal_move_key(direction_x)
                move_vertical = self.get_vertical_move_key(direction_y)
            else:  # Move away from the enemy
                move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
                move_vertical = self.get_vertical_move_key(direction_y, opposite=True)

            movement_options = self._build_target_move_candidates(
                direction_x,
                direction_y,
                allow_detour=self.should_detect_walls,
            )

            # Check for walls and adjust movement
            for move in movement_options:
                if not self.is_path_blocked(player_pos, move, wall_context):
                    movement = move
                    break
            else:
                print("default paths are blocked")
                # If all preferred directions are blocked, try other directions
                alternative_moves = ['W', 'A', 'S', 'D']
                random.shuffle(alternative_moves)
                for move in alternative_moves:
                    if not self.is_path_blocked(player_pos, move, wall_context):
                        movement = move
                        break
                else:
                    # if no movement is available, we still try to go in the best direction
                    # because it's better than doing nothing
                    movement = move_horizontal + move_vertical

            if movement != self.last_movement:
                if current_time - self.last_movement_time >= self.minimum_movement_delay:
                    self.last_movement = movement
                    self.last_movement_time = current_time
                else:
                    movement = self.last_movement  # Continue previous movement
            else:
                self.last_movement_time = current_time  # Reset timer if movement didn't change

        # Attack if enemy is within attack range and hittable
        if enemy_distance <= attack_range:
            if self.should_use_gadget == True and self.is_gadget_ready:
                self.use_gadget()
                self.time_since_gadget_checked = time.time()
                self.is_gadget_ready = False
            if self.is_hypercharge_ready and self.is_super_ready:
                self.use_hypercharge()
                self.time_since_hypercharge_checked = time.time()
                self.is_hypercharge_ready = False
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, wall_context, "attack")
            # print("enemy hittable", enemy_hittable, "enemy_distance", enemy_distance)
            if enemy_hittable:
                self.attack()
        if self.is_super_ready:
            super_type = brawler_info['super_type']
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, wall_context, "super")

            if (enemy_hittable and
                    (enemy_distance <= super_range
                     or super_type in ["spawnable", "other"]
                     or (brawler in ["stu", "surge"] and super_type == "charge" and enemy_distance <= super_range + attack_range)
                    )):
                self.use_super()
                self.time_since_super_checked = time.time()
                self.is_super_ready = False
        return movement

    def main(self, frame, brawler):
        current_time = time.time()
        runtime_state = str(getattr(self, "_runtime_state", "") or "")
        if runtime_state == "match" and (current_time - self._last_end_result_probe_time) >= 0.8:
            self._last_end_result_probe_time = current_time
            game_result = find_game_result(frame)
            if game_result:
                print(f"[RESULT] play fast probe detected {game_result}")
                self._pending_end_result = game_result
                self._runtime_state = f"end_{game_result}"
                self.window_controller.keys_up(list("wasd"))
                self.time_since_last_proceeding = current_time
                return
        data = self.get_main_data(frame)
        raw_supporting_entities = (
            self._entity_count(data, "enemy")
            + self._entity_count(data, "teammate")
        )
        if self.should_detect_walls and current_time - self.time_since_walls_checked > self.walls_treshold:

            tile_data = self.get_tile_data(frame)

            walls = self.process_tile_data(tile_data)

            self.time_since_walls_checked = current_time
            self.last_walls_data = walls
            data['wall'] = walls
        elif self.keep_walls_in_memory:
            data['wall'] = self.last_walls_data

        data = self._restore_recent_player_detection(data, current_time, runtime_state)

        data = self.validate_game_data(data)
        self.track_no_detections(data)
        if data:
            self._remember_player_detection(data, current_time)
            if (
                self._is_plausible_player_detection(data)
                and (
                    self._entity_count(data, "enemy") > 0
                    or self._entity_count(data, "teammate") > 0
                    or (
                        runtime_state == "match"
                        and (current_time - self._last_confirmed_match_time) <= 1.0
                    )
                )
            ):
                self._last_match_evidence_time = current_time
            self.time_since_player_last_found = time.time()
            if runtime_state != "match":
                should_recheck_state = (
                    current_time >= self._match_state_grace_until
                    and (
                        current_time - self._last_state_probe_time >= 0.35
                        or runtime_state != self._last_state_probe_runtime
                    )
                )
                if should_recheck_state:
                    checked_state = get_state(frame)
                    self._last_state_probe_time = current_time
                    self._last_state_probe_runtime = runtime_state
                    self._last_state_probe_result = checked_state
                    if checked_state == "match":
                        self.note_confirmed_match_state(current_time)
                        self._match_state_grace_until = current_time + 2.0
                    else:
                        self._match_state_grace_until = 0.0
                else:
                    checked_state = self._last_state_probe_result or runtime_state
                if self._should_promote_detected_match(data, runtime_state, checked_state, current_time):
                    self._runtime_state = "match"
                    self.note_confirmed_match_state(current_time)
                    self._match_state_grace_until = current_time + 2.0
                elif checked_state != "match":
                    if debug and (current_time - self._last_state_guard_log_time >= 2.0):
                        print(f"Player detected while state was '{runtime_state or 'unknown'}', rechecked as '{checked_state}'")
                        self._last_state_guard_log_time = current_time
                    data = None
                else:
                    self._reset_match_state_guard()
                    self._runtime_state = "match"
        if not data:
            self._reset_match_state_guard()
            player_missing_for = current_time - self.time_since_player_last_found
            if (
                runtime_state == "match"
                and player_missing_for >= 0.25
                and (current_time - self._last_end_result_probe_time) >= 0.5
            ):
                self._last_end_result_probe_time = current_time
                game_result = find_game_result(frame)
                if game_result:
                    print(f"[RESULT] play missing-player probe detected {game_result}")
                    self._pending_end_result = game_result
                    self._runtime_state = f"end_{game_result}"
                    self.window_controller.keys_up(list("wasd"))
                    self.time_since_last_proceeding = current_time
                    return
            if current_time - self.time_since_player_last_found > 1.0:
                self.window_controller.keys_up(list("wasd"))
            self.time_since_different_movement = time.time()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                current_state = get_state(frame)
                likely_post_match_reward = (
                    player_missing_for >= 2.5
                    and raw_supporting_entities <= 0
                    and (current_time - self._last_match_evidence_time) >= 1.5
                )
                allow_reward_ocr = (
                    current_state == "match"
                    and (
                        runtime_state == "reward_claim"
                        or str(runtime_state).startswith("end_")
                        or likely_post_match_reward
                    )
                )
                if allow_reward_ocr:
                    current_state = get_state(frame, allow_reward_ocr=True)
                if isinstance(current_state, str) and current_state.startswith("end_"):
                    print(f"[RESULT] play state probe detected {current_state}")
                    self._pending_end_result = current_state.split("_", 1)[1]
                    self._runtime_state = current_state
                    self.window_controller.keys_up(list("wasd"))
                    self.time_since_last_proceeding = current_time
                    return
                if current_state == "reward_claim":
                    print("[RESULT] play state probe detected reward_claim")
                    self._runtime_state = current_state
                    self.window_controller.keys_up(list("wasd"))
                    self.time_since_last_proceeding = current_time
                    return
                if current_state != "match":
                    self.time_since_last_proceeding = current_time
                else:
                    if debug and (current_time - self._last_no_player_log_time >= 2.0):
                        print("haven't detected the player in a while proceeding")
                        self._last_no_player_log_time = current_time
                    self.window_controller.press_continue()
                    self.time_since_last_proceeding = time.time()
            return
        self.time_since_last_proceeding = time.time()
        wall_context = self.get_wall_context(data['wall'])
        teammates = data.get('teammate') or []
        self._teammate_positions = [self.get_enemy_pos(teammate) for teammate in teammates]
        if not self._teammate_positions:
            self._reset_showdown_teammate_lock()
        enemies = data.get('enemy') or []
        if enemies:
            self._update_enemy_memory(enemies)
            self._last_enemy_seen_at = current_time
            self._search_target_switch_time = 0.0
        should_check_hypercharge = current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold
        should_check_gadget = current_time - self.time_since_gadget_checked > self.gadget_treshold
        should_check_super = current_time - self.time_since_super_checked > self.super_treshold
        hud_hsv = None
        hud_origin = (0, 0)
        if should_check_hypercharge or should_check_gadget or should_check_super:
            hud_hsv, hud_origin = self.get_hud_hsv(frame)

        if should_check_hypercharge:
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(hud_hsv, hud_origin)
            self.time_since_hypercharge_checked = current_time
        if should_check_gadget:
            self.is_gadget_ready = self.check_if_gadget_ready(hud_hsv, hud_origin)
            self.time_since_gadget_checked = current_time
        if should_check_super:
            self.is_super_ready = self.check_if_super_ready(hud_hsv, hud_origin)
            self.time_since_super_checked = current_time

        self.current_frame = frame
        movement = self.loop(brawler, data, current_time, wall_context)

        # if data:
        #     # Record scene data
        #     self.scene_data.append({
        #         'frame_number': len(self.scene_data),
        #         'player': data.get('player', []),
        #         'enemy': data.get('enemy', []),
        #         'wall': data.get('wall', []),
        #         'movement': movement,
        #     })

    def generate_visualization(self, output_filename='visualization.mp4'):
        import cv2
        import numpy as np

        frame_size = (1920, 1080)  # Adjust as needed
        fps = 10

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, frame_size)

        for frame_data in self.scene_data:
            # Create a blank image
            img = np.zeros((frame_size[1], frame_size[0], 3), np.uint8)

            # Scale factors if needed
            scale_x = frame_size[0] / 1920
            scale_y = frame_size[1] / 1080

            if frame_data['wall']:
                # Draw walls
                for wall in frame_data['wall']:
                    x1, y1, x2, y2 = map(int, wall)
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (128, 128, 128), -1)  # Gray walls

            if frame_data['enemy']:
                # Draw enemies
                for enemy in frame_data['enemy']:
                    x1, y1, x2, y2 = map(int, enemy)
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)  # Red enemies

            if frame_data['player']:
                # Draw player
                for player in frame_data['player']:
                    x1, y1, x2, y2 = map(int, player)
                    x1 = int(x1 * scale_x)
                    y1 = int(y1 * scale_y)
                    x2 = int(x2 * scale_x)
                    y2 = int(y2 * scale_y)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), -1)  # Green player

            # Draw movement decision
            movement = frame_data['movement']
            direction = self.movement_to_direction(movement)
            cv2.putText(img, f'Movement: {direction}', (10, frame_size[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # Write frame to video
            out.write(img)

        out.release()

    @staticmethod
    def movement_to_direction(movement):
        mapping = {
            'w': 'up',
            'a': 'left',
            's': 'down',
            'd': 'right',
            'wa': 'up-left',
            'aw': 'up-left',
            'wd': 'up-right',
            'dw': 'up-right',
            'sa': 'down-left',
            'as': 'down-left',
            'sd': 'down-right',
            'ds': 'down-right',
        }
        movement = movement.lower()
        movement = ''.join(sorted(movement))
        return mapping.get(movement, 'idle' if movement == '' else movement)

