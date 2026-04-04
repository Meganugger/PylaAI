import math
import random
import time

import cv2
import numpy as np
from shapely import LineString
from shapely.geometry import Polygon
from state_finder.main import get_state
from detect import Detect
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info
from visual_overlay import VisualOverlay
from hp_estimator import HPEstimator
from pathfinder import PathPlanner

brawl_stars_width, brawl_stars_height = 1920, 1080

# === BRAWLER ARCHETYPE SYSTEM ===
# Each playstyle defines how the bot moves, attacks, and retreats.
# Values are looked up at runtime based on brawler's "playstyle" field in brawlers_info.json.
PLAYSTYLE_CONFIG = {
    "fighter": {
        "range_mult": 1.0,
        "hp_retreat": 22,            # BALANCED - retreat when needed (not too late)
        "approach_factor": 1.1,      # Push but not reckless
        "keep_max_range": False,
        "rush_low_enemy": True,      # CHASE low HP enemies
        "prefer_wall_cover": False,
        "aggressive_no_enemy": True, # HUNT when no enemies visible
        "dodge_chance": 0.45,        # Good juking for survival
        "attack_interval": 0.12,     # Fast shooting
        "kite_interval": 0.10,       # Fast kiting
        "team_cohesion": 0.30,       # Stay with team
        "predict_shots": True,       # Lead targets (NEW)
        "retreat_on_low_ammo": True, # Smart ammo management (NEW)
    },
    "sniper": {
        "range_mult": 1.0,
        "hp_retreat": 35,            # Retreat earlier - snipers are fragile
        "approach_factor": 0.3,      # Stay at range but not camping
        "keep_max_range": False,     # Don't camp max range
        "rush_low_enemy": True,      # Finish low HP enemies
        "prefer_wall_cover": True,   # Use cover for survival
        "aggressive_no_enemy": True, # HUNT
        "dodge_chance": 0.60,        # Excellent juking
        "attack_interval": 0.15,     # Fast
        "kite_interval": 0.12,
        "team_cohesion": 0.20,
        "predict_shots": True,       # Lead targets - critical for snipers
        "retreat_on_low_ammo": True,
    },
    "tank": {
        "range_mult": 1.0,
        "hp_retreat": 12,            # Tanks can stay longer
        "approach_factor": 1.5,      # RUSH hard
        "keep_max_range": False,
        "rush_low_enemy": True,
        "prefer_wall_cover": False,
        "aggressive_no_enemy": True,
        "dodge_chance": 0.15,        # Tanks face-tank more
        "attack_interval": 0.10,     # SPAM
        "kite_interval": 0.10,
        "team_cohesion": 0.35,       # Stay with team for support
        "predict_shots": False,      # Close range - no need
        "retreat_on_low_ammo": False,
    },
    "assassin": {
        "range_mult": 1.0,
        "hp_retreat": 28,            # Retreat when low - assassins need HP to dive
        "approach_factor": 1.3,      # RUSH
        "keep_max_range": False,
        "rush_low_enemy": True,
        "prefer_wall_cover": False,
        "aggressive_no_enemy": True,
        "dodge_chance": 0.70,        # BEST juking - survival critical
        "attack_interval": 0.10,     # FASTEST
        "kite_interval": 0.08,       # Very fast
        "team_cohesion": 0.12,       # Flank independently
        "predict_shots": True,       # Predict for burst damage
        "retreat_on_low_ammo": True, # Need ammo for burst
    },
    "thrower": {
        "range_mult": 1.0,
        "hp_retreat": 30,            # Fragile - retreat earlier
        "approach_factor": 0.6,      # Mid-range
        "keep_max_range": False,
        "rush_low_enemy": True,
        "prefer_wall_cover": False,
        "aggressive_no_enemy": True,
        "dodge_chance": 0.55,        # Good juking
        "attack_interval": 0.10,     # SPAM
        "kite_interval": 0.08,
        "team_cohesion": 0.35,       # Stay with team
        "predict_shots": True,       # Lead throws
        "retreat_on_low_ammo": True,
    },
    "support": {
        "range_mult": 1.0,
        "hp_retreat": 38,            # Retreat earlier - stay alive to heal team
        "approach_factor": 0.6,      # Mid-range
        "keep_max_range": False,
        "rush_low_enemy": True,      # Help finish kills
        "prefer_wall_cover": True,   # Use cover
        "aggressive_no_enemy": True, # HUNT
        "prefer_teammates": True,
        "dodge_chance": 0.40,        # Decent juking
        "attack_interval": 0.14,     # Fast
        "kite_interval": 0.10,
        "team_cohesion": 0.50,       # Stay CLOSE to teammates
        "predict_shots": True,
        "retreat_on_low_ammo": True,
    },
}

class Movement:

    # Forward declarations for methods defined in Play subclass
    # (Pylance needs these to resolve cross-references within Movement methods)
    def is_path_blocked(self, *args, **kwargs) -> bool: ...  # type: ignore[empty-body]
    def is_enemy_hittable(self, *args, **kwargs) -> bool: ...  # type: ignore[empty-body]
    time_since_player_last_found: float

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
        self.game_mode_name = bot_config.get("gamemode", "knockout")  # Human-readable name
        self.is_showdown = False  # Updated from stage_manager detection
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
        self.TILE_SIZE = 120  # Wall check distance (reduced from 180 for tighter detection)
        # Usage cooldowns (separate from detection timers)
        self.time_since_gadget_used = 0
        self.time_since_super_used = 0
        self.time_since_hypercharge_used = 0
        self.GADGET_USE_COOLDOWN = 1.0        # seconds between gadget uses (fast!)
        self.SUPER_USE_COOLDOWN = 0.8         # seconds between super uses (fast!)
        self.HYPERCHARGE_USE_COOLDOWN = 3.0   # seconds between hypercharge uses
        # All 8 directions for movement fallback
        self.ALL_DIRECTIONS = ['W', 'A', 'S', 'D', 'WA', 'WD', 'SA', 'SD']
        # Debug window state
        self._debug_window_created = False
        self._last_debug_time = 0.0  # Throttle debug overlay
        # Health tracking & regen
        self.player_hp_percent = 100
        self.enemy_hp_percent = -1    # -1 = unknown until first measurement
        self._player_hp_fail_count = 0   # Track consecutive HP detection failures
        self._enemy_hp_fail_count = 0
        self._last_valid_player_hp = 100  # Last successfully detected HP
        self._last_valid_enemy_hp = -1
        self._hp_debug_info = {}  # Debug info for overlay
        self.last_attack_time = time.time()  # Initialize to now so regen doesn't trigger at start
        self.REGEN_DELAY = 3.0        # seconds without attacking before HP starts regenerating
        self.LOW_HP_THRESHOLD = 25    # below this %, retreat (but still attack if safe)
        self.is_regenerating = False
        self.last_enemy_hp_update = time.time()  # Track staleness of enemy HP
        # Per-enemy HP map: bbox_key -> (hp_percent, confidence, timestamp)
        self._per_enemy_hp = {}  # {bbox_key: (hp, conf, time)}
        self._last_closest_enemy_center = None  # Track target switch for HPEstimator reset
        self._last_enemy_raw_hp = None  # Raw HP number from OCR (e.g., 1760, 4000)
        # Enhanced HP estimator v2 with reduced smoothing for faster reaction
        self._hp_estimator = HPEstimator(smoothing_frames=2)  # Was 3 - now faster
        self._last_damage_taken_time = time.time()  # Track when we last took damage (for regen rate guard)
        self._last_enemy_damage_time = time.time()   # Track when enemy last took damage
        self._hp_confidence_player = 1.0  # Confidence of last player HP reading
        self._hp_confidence_enemy = 1.0   # Confidence of last enemy HP reading
        self._last_valid_player_hp_time = time.time()
        self._hp_data_age_player = 0.0
        self._hp_state_player = "healthy"  # healthy|warning|critical|unknown

        # HP safety / confidence policy (can be overridden via general_config.toml)
        self._hp_check_interval = 0.10
        self._hp_conf_low_threshold = 0.45
        self._hp_stale_timeout = 0.55
        self._hp_warning_enter = 45
        self._hp_warning_exit = 55
        self._hp_critical_enter = 20
        self._hp_critical_exit = 26
        # Rolling HP windows for high-percentile filtering
        # HP bar detection mostly UNDERestimates (character model occludes green pixels)
        # so the highest recent readings are closest to the true HP value.
        # Window of 4 = ~0.27s at 15fps - fast damage response.
        from collections import deque as _deque
        self._player_hp_window = _deque(maxlen=4)
        self._enemy_hp_window = _deque(maxlen=4)
        self._showdown_detected_in_match = False  # In-match detection flag
        self._last_showdown_check = 0.0  # Throttle in-match OCR
        # === TARGET INFO (for overlay + stats) ===
        self.target_info = {
            'name': None,          # OCR'd name of targeted enemy
            'distance': 0,         # px distance to target
            'hp': -1,              # HP% of target
            'hittable': False,     # Is line of sight clear?
            'bbox': None,          # Target bounding box
            'n_enemies': 0,        # Total enemies visible
            'n_teammates': 0,      # Total teammates visible
        }
        self._last_name_ocr_time = 0.0  # Throttle name OCR (expensive)
        self._last_bush_check_time = 0.0  # Cooldown for bush-check attacks
        self._player_name = None           # Own player name (to filter from enemy OCR)
        self._player_name_read = False     # Whether we've attempted to read own name

        # === COMMITTED STRAFE SYSTEM ===
        # Instead of random direction every frame, commit to a strafe for 0.3-0.8s
        self._strafe_direction = None       # Current strafe: 'A' or 'D'
        self._strafe_start_time = 0.0
        self._strafe_duration = 0.6         # How long to hold current strafe
        # === MOVEMENT MOMENTUM (prevents erratic zig-zagging) ===
        self._last_move_dir = ''            # Last movement direction string
        self._move_dir_start = 0.0          # When we committed to current direction
        self._move_momentum_min = 0.15      # Minimum seconds to hold a direction

        # === AMMO TRACKING (estimated) ===
        self._ammo = 3                      # Current ammo count (estimated)
        self._max_ammo = 3
        self._last_ammo_spend_time = 0.0    # When we last fired
        self._reload_speed = 1.4            # Seconds per ammo (default, overridden per brawler)
        self._ammo_conserve_threshold = 1   # Don't fire below this unless finishing

        # === LAST KNOWN ENEMY POSITIONS ===
        self._last_known_enemies = []       # [(x, y, timestamp), ...]
        self._enemy_memory_duration = 2.5   # Remember enemies for 2.5 seconds (short - avoid chasing ghosts)

        # === ENEMY DEATH TRACKING ===
        self._enemy_death_positions = []    # [(x, y, timestamp)] where enemies likely died
        self._enemy_death_cooldown = 5.0    # Ignore death positions for this long (respawn time)
        self._enemy_hp_before_disappear = 100  # Last known HP% before enemy vanished
        self._enemy_was_low_hp_when_lost = False  # Was enemy <30% when they disappeared?
        self._last_enemy_kill_time = 0.0    # When we last scored a kill
        self._enemies_killed_this_match = 0 # Kill count this match
        self._enemy_death_radius = 120      # px radius around death pos to ignore

        # === ACTIVE ENEMY SEARCH / PATROL ===
        self._patrol_phase = 'idle'             # 'idle'/'advance'/'sweep_left'/'sweep_right'/'check_bush'
        self._patrol_start_time = 0.0
        self._patrol_sweep_dir = 1              # 1 = right, -1 = left
        self._patrol_advance_timer = 0.0        # Time spent advancing before next sweep
        self._last_patrol_change = 0.0          # Prevent rapid patrol mode changes
        self._visited_zones = []                # [(x, y, timestamp)] - recently visited areas
        self._no_enemy_duration = 0.0           # How long since we last saw any enemy
        self._solo_search_target_idx = 0        # Index for rotating solo-search waypoints
        self._solo_search_last_switch = 0.0     # Last time solo-search target changed

        # === DECISION REASON LOGGING ===
        self.last_decision_reason = ""      # Why the bot chose its current action

        # === ADAPTIVE AGGRESSION ===
        self.aggression_modifier = 1.0      # 0.7 = defensive, 1.3 = aggressive
        self._matches_for_adaptation = 0    # Track matches for win-rate based tuning

        # === IPS OPTIMIZATION: FRAME SKIPPING ===
        self._frame_skip_counter = 0         # Counts frames since last full detection
        self._cached_detection_data = None   # Cached ONNX detection result
        self._last_hp_check_time = 0.0       # Throttle HP detection
        self._hp_check_interval = self._hp_check_interval  # Configurable throttle (default 100ms)
        self._brawler_detected = False       # Whether we auto-detected brawler via HP
        self._match_start_hp_sample = None   # First HP reading at match start

        # === BUSH TILE AWARENESS ===
        self.last_bush_data = []             # Detected bush bounding boxes from tile detector

        # === ENEMY VELOCITY TRACKING (predictive aim) ===
        self._enemy_pos_history = []         # [(x, y, timestamp), ...] for velocity calc
        self._enemy_velocity = (0, 0)        # Estimated px/sec (vx, vy)
        self._enemy_velocity_smooth = (0, 0)  # EMA-smoothed velocity (less jittery)
        self._enemy_move_direction = 'none'   # 'left'/'right'/'up'/'down'/'none'
        self._enemy_direction_since = 0.0     # When enemy started moving in this direction
        self._enemy_speed_magnitude = 0.0     # px/sec speed for quick checks
        self._last_tracked_enemy_pos = None   # For detecting target switches
        # Enhanced aim prediction state
        self._enemy_accel = (0.0, 0.0)        # Estimated acceleration (px/sec²)
        self._enemy_prev_velocity = (0.0, 0.0) # Previous frame velocity for accel calc
        self._enemy_prev_vel_time = 0.0        # Timestamp of previous velocity
        self._velocity_confidence = 0.0        # 0-1: how reliable is current velocity estimate
        self._enemy_dir_changes = 0            # Direction changes in last 1s (stutter detection)
        self._enemy_dir_change_times = []      # Timestamps of direction changes
        self._last_aim_lead = (0, 0)           # Last computed lead for debug overlay

        # === GHOST WALL TRACKING ===
        self._ghost_wall = None               # (wall_coords, expire_time)
        self._ghost_wall_expire = 0.0         # When the ghost wall expires

        # === BURST FIRE MODE ===
        # When fully loaded (3/3 ammo) and enemy in range, dump ALL ammo ASAP.
        # After burst, play defensively until ammo recovers.
        self._burst_mode = False              # Currently dumping ammo
        self._burst_start_time = 0.0
        self._last_burst_end_time = 0.0       # When we finished dumping
        self._burst_defensive_duration = 3.5  # Seconds of cautious play after burst
        self._burst_interval = 0.06           # 60ms between shots during burst

        # === COMBO CHAIN SYSTEM ===
        self._combo_queued = False
        self._combo_queue_time = 0.0
        self._combo_type = None              # 'super_then_attack'

        # === TEAMMATE HP TRACKING ===
        self._teammate_hp_data = []          # [(center_x, center_y, hp_percent), ...]
        self._lowest_teammate_hp = 100
        self._teammate_positions = []        # [(cx, cy), ...] cached each frame for team targeting

        # === GAME-MODE OBJECTIVE ===
        self.objective_pos = None            # (x, y) mode-specific waypoint

        # === SPAWN-SIDE AWARENESS ===
        self._spawn_side = None              # 'left'/'right'/'top'/'bottom' - detected from first frame
        self._spawn_detected = False
        self._spawn_detect_frames = 0        # Count frames used for detection

        # === RESPAWN INVINCIBILITY RUSH ===
        self._time_player_disappeared = 0.0
        self._respawn_shield_active = False
        self._respawn_shield_until = 0.0
        self._player_was_visible = False       # Was player detected last frame?
        self._is_dead = False                  # True while player is dead/respawning

        # === DEATH / LIFE ECONOMY TRACKING ===
        self._death_count = 0                  # Number of times we died this match
        self._alive_teammates_last = 0         # Teammates visible last check
        self._no_teammate_since = 0.0          # When teammates last disappeared
        self._last_life_mode = False            # True when on final life in Knockout

        # === SCORE AWARENESS ===
        self._our_score = 0
        self._their_score = 0
        self._last_score_check = 0.0
        self._score_diff = 0                   # Positive = winning, negative = losing

        # === BRAWLER COMBAT STATS (loaded from brawlers_info.json) ===
        self._my_health = 3200                  # Our brawler's base HP
        self._my_attack_damage = 1200           # Our total attack damage
        self._my_projectile_count = 1           # Number of projectiles per attack
        self._my_movement_speed = 720           # Our movement speed (580-820)
        self._my_super_damage = 0               # Our super damage
        self._my_dps = 0.0                      # Damage per second (attack_damage / reload_speed)
        self._shots_to_kill_default = 5         # How many attacks to kill avg enemy (3200 HP)

        # === ANTI-STUCK ESCALATION ===
        self._stuck_escalation = 0             # 0=normal, 1=reverse, 2=try-all, 3=spiral
        self._stuck_trigger_times = []         # Timestamps of recent anti-stuck triggers
        self._spiral_index = 0                 # Current step in spiral escape pattern
        self._stuck_check_pos = None           # Player position when stuck timer started
        self._stuck_pos_threshold = 8          # px - must move at least this far to count as "not stuck" (was 12)
        self._has_enemy_target = False          # Whether enemies are visible
        self._spiral_step_time = 0.0
        self._position_history = []            # Last N (pos, time) for oscillation detection
        self._position_history_max = 30        # Track last 30 positions (~3 seconds)
        self._oscillation_bbox_threshold = 90  # If bounding box of last N positions < this, we're oscillating

        # === STUTTER-STEP TIMING ===
        self._stutter_step_active = False
        self._stutter_step_until = 0.0
        self._stutter_step_keys = ''

        # === RETREAT PATH SAFETY (dead-end avoidance) ===
        self._retreat_safety_cache = {}         # direction -> score, refreshed per frame
        self._retreat_cache_time = 0.0

        # === NUMBER ADVANTAGE PUSH ===
        self._expected_enemy_count = 3          # 3v3 default, updated per mode
        self._last_full_enemy_count = 0
        self._enemy_count_drop_time = 0.0
        self._number_advantage_active = False
        self._number_advantage_until = 0.0      # Auto-expire after 6s

        # === SHOWDOWN STORM / KNOCKOUT POISON GAS ZONE ===
        self._storm_center = (960, 540)         # Estimated safe zone center
        self._storm_radius = 9999               # Estimated safe zone radius (px)
        self._last_storm_check = 0.0
        self._last_gas_check = 0.0              # Separate timer for gas detection
        self._in_storm = False
        self._gas_active = False                 # Whether any poison gas/storm is detected
        self._gas_density_map = None             # Per-region gas density for avoidance

        # === ENEMY RELOAD WINDOW EXPLOITATION ===
        self._enemy_last_attack_time = {}       # pos_bucket -> timestamp
        self._prev_player_hp = 100              # HP last frame for delta detection
        self._enemy_in_reload_window = False

        # === JUKE PATTERN ENTROPY ===
        self._juke_patterns = [
            [('A', 0.5), ('D', 0.5)],             # Smooth L-R arcs
            [('A', 0.7), ('D', 0.35)],             # Long left, short right
            [('D', 0.6), ('A', 0.4), ('D', 0.5)],  # Flowing right-left-right
            [('A', 0.4), ('D', 0.8)],               # Short left into long right
            [('D', 0.8), ('A', 0.4)],               # Long right into short left
            [('A', 0.3), ('D', 0.6), ('A', 0.6)],   # Flowing three-step
            [('D', 0.5), ('A', 0.5), ('D', 0.4)],   # Smooth three-step
            [('A', 0.9)],  # Long commit left
            [('D', 0.9)],  # Long commit right
        ]
        self._juke_current_pattern = None
        self._juke_step = 0
        self._juke_step_start = 0.0
        self._juke_engaged = False              # True when enemy visible

        # === LANE ASSIGNMENT (3v3) ===
        self._assigned_lane = None              # 'left'/'center'/'right' or 'top'/'mid'/'bottom'
        self._lane_center = None                # (x, y) center of our lane
        self._last_lane_check = 0.0

        # === WALL DESTRUCTION MEMORY ===
        self._force_wall_refresh = False
        self._destroyed_wall_zones = []         # [(x1,y1,x2,y2), ...] where walls were destroyed
        self._pre_super_walls = None            # Snapshot of walls before super

        # === PEEK-SHOOT WALL CYCLING ===
        self._peek_phase = 'idle'               # 'idle'/'expose'/'fire'/'hide'
        self._peek_wall_anchor = None            # (x, y) wall position to peek around
        self._peek_timer = 0.0                   # Time current phase started
        self._peek_expose_dir = ''               # Direction to step out

        # === REACTIVE PERPENDICULAR DODGE ===
        self._reactive_dodge_until = 0.0         # Override movement until this time
        self._reactive_dodge_keys = ''           # Forced dodge direction
        self._last_damage_pos = None             # Attacker pos when we took damage

        # === DISENGAGE-TO-HEAL PROTOCOL ===
        self._disengage_until = 0.0              # Hold fire until this time
        self._disengage_active = False

        # === MATCH PHASE ADAPTATION ===
        self._match_start_time = 0.0             # When player first appeared
        self._match_phase = 'early'              # 'early'/'mid'/'late'
        self._match_phase_set = False            # True once first detection sets start time
        self._last_episode_reset_token = None    # Dedup token: (match_counter, last_game_result)
        self._last_episode_reset_time = 0.0      # Last reset timestamp
        self._min_episode_interval_sec = 30.0    # Hard minimum duration between episodes
        self._storm_flee_delay_sec = 30.0        # Do not flee storm/gas before this many seconds

        # === CHOKE POINT DETECTION ===
        self._choke_points = []                  # [(cx, cy, width, angle), ...]
        self._last_choke_scan = 0.0

        # === SUPER VALUE HOLD ===
        self._hold_super = False                 # True = save super for value moment

        # === TEAMMATE DEATH GAP COVER ===
        self._prev_teammate_count = 0
        self._teammate_death_pos = None          # (x, y) last position of dead teammate
        self._teammate_death_time = 0.0
        self._original_lane_center = None        # Saved lane before gap cover

    def reset_match_state(self, match_won: bool = False):
        """Reset all per-match state variables for a fresh match.
        Called when a new match is detected (spawn detection).

        """
        self._spawn_detected = False
        self._spawn_detect_frames = 0
        self._spawn_side = None
        self._death_count = 0
        self._score_diff = 0
        self._our_score = 0
        self._their_score = 0
        self._match_phase = 'early'
        self._match_phase_set = False
        self._match_start_time = 0.0
        self._showdown_detected_in_match = False
        self._last_showdown_check = 0.0
        self._number_advantage_active = False
        self._number_advantage_until = 0.0
        self._last_known_enemies = []
        self._enemy_pos_history = []
        self._enemy_velocity = (0, 0)
        self._enemy_velocity_smooth = (0, 0)
        self._enemy_move_direction = 'none'
        self._enemy_direction_since = 0.0
        self._enemy_speed_magnitude = 0.0
        self._last_tracked_enemy_pos = None
        self._ghost_wall = None
        self._ghost_wall_expire = 0.0
        self._visited_zones = []
        self._no_enemy_duration = 0.0
        self._solo_search_target_idx = 0
        self._solo_search_last_switch = 0.0
        self._enemy_death_positions = []
        self._enemy_hp_before_disappear = 100
        self._enemy_was_low_hp_when_lost = False
        self._last_enemy_kill_time = 0.0
        self._enemies_killed_this_match = 0
        self._patrol_phase = 'idle'
        self._patrol_start_time = 0.0
        self._patrol_sweep_dir = 1
        self._last_patrol_change = 0.0
        self._teammate_death_pos = None
        self._teammate_death_time = 0.0
        self._original_lane_center = None
        self._prev_teammate_count = 0
        self._teammate_hp_data = []
        self._lowest_teammate_hp = 100
        self._teammate_positions = []
        self._assigned_lane = None
        self._lane_center = None
        self._hold_super = False
        self._is_dead = False
        self._respawn_shield_active = False
        self._in_storm = False
        self._gas_active = False
        self._last_gas_check = 0.0
        self._storm_radius = 9999
        self._storm_center = (960, 540)
        self._ammo = 3
        self.player_hp_percent = 100
        self.enemy_hp_percent = -1    # -1 = unknown until first measurement
        self._last_valid_enemy_hp = -1
        self._per_enemy_hp = {}      # Reset per-enemy HP map
        self._last_closest_enemy_center = None
        # Clear HP rolling windows for the new match
        if hasattr(self, '_player_hp_window'):
            self._player_hp_window.clear()
        if hasattr(self, '_enemy_hp_window'):
            self._enemy_hp_window.clear()
        self.aggression_modifier = 1.0
        self._disengage_active = False
        self._disengage_until = 0.0
        self._choke_points = []
        self._destroyed_wall_zones = []
        self._combo_queued = False
        self._burst_mode = False
        self._last_burst_end_time = 0.0
        self.last_bush_data = []
        self.last_decision_reason = ""
        self._stuck_escalation = 0
        self._stuck_trigger_times = []
        self._last_move_dir = ''
        self._move_dir_start = 0.0
        self.time_since_holding_attack = None  # Reset hold-attack state
        # Reset brawler auto-detection for new match
        self._brawler_detected = False
        self._match_start_hp_sample = None
        # Reset HP estimator for new match
        if hasattr(self, '_hp_estimator') and self._hp_estimator is not None:
            self._hp_estimator.reset()
        # Reset pathfinder + spatial memory for new match
        if hasattr(self, '_path_planner') and self._path_planner is not None:
            self._path_planner.reset()
        if hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
            self._spatial_memory.reset()
        # Reset BT combat system if active
        if hasattr(self, '_bt_combat') and self._bt_combat is not None:
            try:
                self._bt_combat.reset_match(match_won=match_won)
            except Exception as e:
                print(f"[BT] reset_match error: {e}")
        print("[RESET] Match state cleared for new match.")

    def _lookup_per_enemy_hp(self, ecx, ecy):
        """Look up HP for a specific enemy from the per-enemy HP map by position.

        Returns hp_percent (0-100) or -1 if no match found.
        """
        per_hp = getattr(self, '_per_enemy_hp', {})
        if not per_hp:
            return -1
        best_hp = -1
        best_dist = 999999
        for key, (hp, conf, ts) in per_hp.items():
            try:
                kx, ky = [float(v) for v in key.split(",")]
            except (ValueError, AttributeError):
                continue
            d = abs(kx - ecx) + abs(ky - ecy)
            if d < best_dist:
                best_dist = d
                best_hp = hp
        return best_hp if best_dist < 80 else -1

    @staticmethod
    def estimate_hp_from_bar(frame, bbox, is_player=False, debug_info=None):
        """
        Estimate HP percentage by reading the health bar above a character.
        Player/teammate bars are GREEN; enemy bars are RED.
        Scans rows both above and inside the top of the bbox to find the HP bar,
        then uses the FULL bar width detected across ALL rows (including the grey/dark
        background portion) as the 100% reference, instead of unreliable YOLO bbox width.
        Returns HP 1-100 on success, -1 on failure.
        If debug_info dict is provided, fills in debug data for visualization.
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            char_width = x2 - x1
            char_height = y2 - y1
            if char_width < 10:
                return -1

            frame_arr = np.asarray(frame)
            fh, fw = frame_arr.shape[:2]

            # Resolution-independent search area: scale with bbox height
            search_above = max(8, int(char_height * 0.40))
            search_below = max(3, int(char_height * 0.04))
            pad_x = max(5, int(char_width * 0.08))
            search_y1 = max(0, y1 - search_above)
            search_y2 = min(fh, y1 + search_below)
            search_x1 = max(0, x1 - pad_x)
            search_x2 = min(fw, x2 + pad_x)

            if search_y1 >= search_y2 or search_x1 >= search_x2:
                return -1

            crop = frame_arr[search_y1:search_y2, search_x1:search_x2]
            if crop.size == 0:
                return -1

            crop_hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            crop_h, crop_w = crop_hsv.shape[:2]

            # --- Detect HEALTH (colored) portion of bar ---
            if is_player:
                # Player/teammate bars: GREEN + YELLOW + ORANGE (low HP)
                g_mask = cv2.inRange(crop_hsv, np.array([30, 45, 60]), np.array([90, 255, 255]))
                y_mask = cv2.inRange(crop_hsv, np.array([15, 45, 60]), np.array([35, 255, 255]))
                o_mask = cv2.inRange(crop_hsv, np.array([8, 50, 60]), np.array([18, 255, 255]))
                health_mask = cv2.bitwise_or(g_mask, y_mask)
                health_mask = cv2.bitwise_or(health_mask, o_mask)
            else:
                # Enemy bars: RED (expanded S/V for faded bars)
                r1 = cv2.inRange(crop_hsv, np.array([0, 40, 50]), np.array([15, 255, 255]))
                r2 = cv2.inRange(crop_hsv, np.array([155, 40, 50]), np.array([180, 255, 255]))
                r3 = cv2.inRange(crop_hsv, np.array([0, 60, 35]), np.array([10, 255, 70]))
                health_mask = cv2.bitwise_or(r1, r2)
                health_mask = cv2.bitwise_or(health_mask, r3)

            # Morphological cleanup: close tiny gaps, remove noise
            _kern_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            _kern_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            health_mask = cv2.morphologyEx(health_mask, cv2.MORPH_CLOSE, _kern_close)
            health_mask = cv2.morphologyEx(health_mask, cv2.MORPH_OPEN, _kern_open)

            # --- Detect FULL BAR (health + depleted background) ---
            # Widened V range (20-120) to cover bright & dark maps
            bar_bg_mask = cv2.inRange(crop_hsv, np.array([0, 0, 20]), np.array([180, 60, 120]))
            bar_bg_mask = cv2.morphologyEx(bar_bg_mask, cv2.MORPH_CLOSE, _kern_close)
            full_bar_mask = cv2.bitwise_or(health_mask, bar_bg_mask)

            # === SCAN EVERY ROW to find bar rows ===
            min_bar_pixels = max(5, int(char_width * 0.10))

            # vectorized row scanning (replaces Python per-row loop)
            # For each row, find the longest contiguous segment of lit pixels.
            # Uses numpy diff-based segmentation - ~10× faster than Python loops.

            def _longest_segment_per_row(mask_2d, gap_tol, min_px):
                """Return (row, seg_width, seg_start, seg_end) for each qualifying row.
                gap_tol: max gap between consecutive lit pixels within one segment."""
                rows, widths, starts, ends = [], [], [], []
                binary = (mask_2d > 0).astype(np.uint8)
                row_sums = binary.sum(axis=1)
                candidate_rows = np.where(row_sums >= min_px)[0]
                for r in candidate_rows:
                    cols = np.where(binary[r])[0]
                    # Split into contiguous segments where gap > gap_tol
                    gaps = np.diff(cols) > gap_tol
                    seg_ids = np.concatenate([[0], np.cumsum(gaps)])
                    # Find longest segment
                    best_len = 0
                    best_s, best_e = cols[0], cols[0]
                    for sid in range(seg_ids[-1] + 1):
                        seg_cols = cols[seg_ids == sid]
                        sw = seg_cols[-1] - seg_cols[0] + 1
                        if sw > best_len:
                            best_len = sw
                            best_s, best_e = seg_cols[0], seg_cols[-1]
                    rows.append(r)
                    widths.append(best_len)
                    starts.append(best_s)
                    ends.append(best_e)
                return rows, widths, starts, ends

            h_rows, h_widths, h_starts, h_ends = _longest_segment_per_row(
                health_mask, 2, min_bar_pixels)
            fb_rows, fb_widths, _, _ = _longest_segment_per_row(
                full_bar_mask, 3, min_bar_pixels)

            # Build result lists with size filter
            row_health_widths = [
                (r, w, s, e) for r, w, s, e in zip(h_rows, h_widths, h_starts, h_ends)
                if w >= min_bar_pixels and w >= crop_w * 0.06
            ]
            row_fullbar_widths = [(r, w) for r, w in zip(fb_rows, fb_widths)
                                  if w >= min_bar_pixels]

            if not row_health_widths:
                if debug_info is not None:
                    debug_info['search_rect'] = (search_x1, search_y1, search_x2, search_y2)
                    debug_info['status'] = 'NO_BAR_FOUND'
                    debug_info['mask_pixels'] = int(np.sum(health_mask > 0))
                return -1

            # Find the row with the widest health segment
            best_entry = max(row_health_widths, key=lambda e: e[1])
            best_row, best_filled, best_seg_start, best_seg_end = best_entry

            # --- Determine 100% reference width ---
            # Strategy: look at full-bar rows near the best health row (±3 rows)
            # and use the widest one as the "full bar" reference.
            nearby_fb = [fb_w for (fb_r, fb_w) in row_fullbar_widths
                         if abs(fb_r - best_row) <= 3]
            if nearby_fb:
                expected_width = max(nearby_fb)
            else:
                # Fallback: use the widest health segment across all rows
                # (assumes at least one frame had full HP)
                expected_width = max(e[1] for e in row_health_widths)

            # Sanity: expected_width must be >= filled width
            expected_width = max(expected_width, best_filled, 15)

            hp_percent = int((best_filled / expected_width) * 100)
            hp_percent = max(1, min(100, hp_percent))

            if debug_info is not None:
                debug_info['search_rect'] = (search_x1, search_y1, search_x2, search_y2)
                debug_info['bar_row'] = search_y1 + best_row
                debug_info['bar_width'] = best_filled
                debug_info['expected_width'] = expected_width
                debug_info['status'] = f'OK:{hp_percent}%'

            return hp_percent
        except Exception:
            return -1

    # === COMMITTED STRAFE SYSTEM ===
    def get_strafe_key(self):
        """Return a committed strafe direction ('A' or 'D') that holds for a random duration.
        This replaces random.choice(['A','D']) everywhere for smoother, less twitchy movement."""
        now = time.time()
        if (self._strafe_direction is None
                or (now - self._strafe_start_time) >= self._strafe_duration):
            self._strafe_direction = 'A' if self._strafe_direction == 'D' else 'D'
            self._strafe_start_time = now
            self._strafe_duration = random.uniform(0.4, 1.0)  # Longer commits = smoother arcs
        return self._strafe_direction

    def get_vstrafe_key(self):
        """Vertical strafe counterpart - alternates W/S in sync with horizontal strafe."""
        # Use opposite phase from horizontal strafe for diagonal juking
        h = self._get_juke_direction()
        return 'W' if h == 'D' else 'S'

    # === AMMO MANAGEMENT ===
    def _update_ammo(self, now):
        """Update estimated ammo count based on time since last shot + reload speed."""
        if self._ammo < self._max_ammo:
            elapsed = now - self._last_ammo_spend_time
            recovered = int(elapsed / self._reload_speed)
            if recovered > 0:
                self._ammo = min(self._max_ammo, self._ammo + recovered)
                self._last_ammo_spend_time = now - (elapsed % self._reload_speed)

    def _should_fire(self, enemy_hp_low=False):
        """Decide whether to fire based on ammo. Conserve last ammo unless finishing."""
        if self._ammo > self._ammo_conserve_threshold:
            return True
        if self._ammo == self._ammo_conserve_threshold and enemy_hp_low:
            return True  # Finishing blow
        return False

    def _spend_ammo(self):
        """Track that we fired a shot."""
        self._ammo = max(0, self._ammo - 1)
        self._last_ammo_spend_time = time.time()

    # === LAST KNOWN ENEMY POSITIONS ===
    def _update_enemy_memory(self, enemies, player_pos):
        """Store enemy positions so we know where they disappeared.
        Filters out positions near recent kill locations to avoid chasing dead enemies."""
        now = time.time()
        # Prune expired death positions first
        self._enemy_death_positions = [
            (x, y, t) for x, y, t in self._enemy_death_positions
            if now - t < self._enemy_death_cooldown
        ]
        # Add current enemy positions (skip if near a death position)
        for en in enemies:
            ex = (en[0] + en[2]) / 2
            ey = (en[1] + en[3]) / 2
            if not self._is_near_death_position(ex, ey):
                self._last_known_enemies.append((ex, ey, now))
            else:
                # Enemy reappeared near a death pos - they respawned, clear that death marker
                self._clear_death_position_near(ex, ey)
                self._last_known_enemies.append((ex, ey, now))
        # Prune old entries AND entries near death positions
        self._last_known_enemies = [
            (x, y, t) for x, y, t in self._last_known_enemies
            if now - t < self._enemy_memory_duration
            and not self._is_near_death_position(x, y)
        ]

    def _get_last_known_enemy_pos(self):
        """Return the most recent remembered enemy position, or None.
        Excludes positions near recent kill locations."""
        if not self._last_known_enemies:
            return None
        # Filter out positions near death locations
        valid = [
            (x, y, t) for x, y, t in self._last_known_enemies
            if not self._is_near_death_position(x, y)
        ]
        if not valid:
            return None
        newest = max(valid, key=lambda e: e[2])
        return (newest[0], newest[1])

    # === ENEMY DEATH DETECTION ===
    def _is_near_death_position(self, x, y):
        """Check if (x, y) is close to any recent enemy death position."""
        for dx, dy, dt in self._enemy_death_positions:
            if math.hypot(x - dx, y - dy) < self._enemy_death_radius:
                return True
        return False

    def _clear_death_position_near(self, x, y):
        """Remove death markers near (x, y) - enemy respawned there."""
        self._enemy_death_positions = [
            (dx, dy, dt) for dx, dy, dt in self._enemy_death_positions
            if math.hypot(x - dx, y - dy) >= self._enemy_death_radius
        ]

    def _register_enemy_kill(self, pos_x, pos_y):
        """Mark that an enemy likely died at this position."""
        now = time.time()
        self._enemy_death_positions.append((pos_x, pos_y, now))
        self._last_enemy_kill_time = now
        self._enemies_killed_this_match += 1
        # In Knockout, enemies don't respawn within a round -> long cooldown
        if self.game_mode_name in ('knockout', 'bounty', 'wipeout', 'duels'):
            self._enemy_death_cooldown = 60.0  # Effectively permanent for the round
        else:
            self._enemy_death_cooldown = 5.0   # Normal 3v3 (Gem Grab etc.): ~5s respawn
        # Purge last_known_enemies near the death position (stop chasing ghost)
        self._last_known_enemies = [
            (x, y, t) for x, y, t in self._last_known_enemies
            if math.hypot(x - pos_x, y - pos_y) >= self._enemy_death_radius
        ]
        print(f"[KILL] Enemy likely killed at ({int(pos_x)}, {int(pos_y)}) - "
              f"total kills: {self._enemies_killed_this_match} | "
              f"cooldown: {self._enemy_death_cooldown}s")

    # === ENEMY VELOCITY TRACKING (for predictive aim) ===
    def _update_enemy_velocity(self, enemy_pos):
        """Track enemy movement across frames to estimate velocity vector.
        Uses exponential moving average (EMA) for smooth, responsive tracking.
        Detects target switches and resets history to avoid contamination.
        Also computes acceleration and velocity confidence for aim prediction."""
        now = time.time()
        # Detect target switch: if new position jumps >200px from last tracked, reset
        if self._last_tracked_enemy_pos is not None:
            jump_dist = math.hypot(
                enemy_pos[0] - self._last_tracked_enemy_pos[0],
                enemy_pos[1] - self._last_tracked_enemy_pos[1]
            )
            if jump_dist > 200:
                # Target switched - clear stale velocity data
                self._enemy_pos_history = []
                self._enemy_velocity = (0, 0)
                self._enemy_velocity_smooth = (0, 0)
                self._enemy_speed_magnitude = 0.0
                self._enemy_move_direction = 'none'
                self._enemy_accel = (0.0, 0.0)
                self._enemy_prev_velocity = (0.0, 0.0)
                self._velocity_confidence = 0.0
                self._enemy_dir_changes = 0
                self._enemy_dir_change_times = []
        self._last_tracked_enemy_pos = (enemy_pos[0], enemy_pos[1])
        self._enemy_pos_history.append((enemy_pos[0], enemy_pos[1], now))
        # Keep last 12 data points within 1.0 seconds (more data for acceleration)
        self._enemy_pos_history = [
            p for p in self._enemy_pos_history if now - p[2] < 1.0
        ][-12:]
        if len(self._enemy_pos_history) >= 2:
            # Use weighted velocity: recent frames matter more
            total_vx, total_vy, total_weight = 0.0, 0.0, 0.0
            frame_velocities = []  # For consistency check
            for i in range(1, len(self._enemy_pos_history)):
                prev = self._enemy_pos_history[i - 1]
                curr = self._enemy_pos_history[i]
                dt = curr[2] - prev[2]
                if dt > 0.01:
                    vx = (curr[0] - prev[0]) / dt
                    vy = ((curr[1] - prev[1]) * 1.25) / dt  # Isometric Y correction
                    # Weight increases for more recent samples
                    weight = i  # 1, 2, 3, ... (latest = heaviest)
                    total_vx += vx * weight
                    total_vy += vy * weight
                    total_weight += weight
                    frame_velocities.append((vx, vy))
            if total_weight > 0:
                raw_vx = total_vx / total_weight
                raw_vy = total_vy / total_weight
                self._enemy_velocity = (raw_vx, raw_vy)
                # EMA smooth: 70% new, 30% old -> responsive but not jittery
                alpha = 0.7
                old_sx, old_sy = self._enemy_velocity_smooth
                self._enemy_velocity_smooth = (
                    alpha * raw_vx + (1 - alpha) * old_sx,
                    alpha * raw_vy + (1 - alpha) * old_sy
                )
                self._enemy_speed_magnitude = math.hypot(
                    self._enemy_velocity_smooth[0],
                    self._enemy_velocity_smooth[1]
                )

                # --- ACCELERATION ESTIMATION ---
                # Compare current velocity to previous velocity
                prev_vx, prev_vy = self._enemy_prev_velocity
                dt_vel = now - self._enemy_prev_vel_time if self._enemy_prev_vel_time > 0 else 0.1
                if dt_vel > 0.03:  # Need meaningful time gap
                    ax = (self._enemy_velocity_smooth[0] - prev_vx) / dt_vel
                    ay = (self._enemy_velocity_smooth[1] - prev_vy) / dt_vel
                    # EMA smooth acceleration (very aggressive - acceleration is noisy)
                    a_alpha = 0.4
                    self._enemy_accel = (
                        a_alpha * ax + (1 - a_alpha) * self._enemy_accel[0],
                        a_alpha * ay + (1 - a_alpha) * self._enemy_accel[1],
                    )
                    self._enemy_prev_velocity = self._enemy_velocity_smooth
                    self._enemy_prev_vel_time = now

                # --- VELOCITY CONFIDENCE ---
                # High confidence = consistent velocity across frames
                # Low confidence = erratic/noisy movement (strafing, lag)
                if len(frame_velocities) >= 3:
                    # Check how consistent recent velocities are
                    recent_vels = frame_velocities[-4:]  # Last 4 frames
                    avg_vx = sum(v[0] for v in recent_vels) / len(recent_vels)
                    avg_vy = sum(v[1] for v in recent_vels) / len(recent_vels)
                    # Variance from average
                    variance = sum(
                        math.hypot(v[0] - avg_vx, v[1] - avg_vy)
                        for v in recent_vels
                    ) / len(recent_vels)
                    # Normalize: 0 variance = 1.0 confidence, >300px/s variance = 0.0
                    self._velocity_confidence = max(0.0, min(1.0, 1.0 - variance / 300.0))
                else:
                    self._velocity_confidence = 0.3  # Not enough data

                # --- DIRECTION CHANGE TRACKING ---
                svx, svy = self._enemy_velocity_smooth
                if self._enemy_speed_magnitude > 50:  # Moving meaningfully
                    if abs(svx) > abs(svy):
                        new_dir = 'right' if svx > 0 else 'left'
                    else:
                        new_dir = 'down' if svy > 0 else 'up'
                    if new_dir != self._enemy_move_direction:
                        self._enemy_move_direction = new_dir
                        self._enemy_direction_since = now
                        self._enemy_dir_change_times.append(now)
                else:
                    self._enemy_move_direction = 'none'
                # Prune old direction changes (keep last 1.5s)
                self._enemy_dir_change_times = [
                    t for t in self._enemy_dir_change_times if now - t < 1.5
                ]
                self._enemy_dir_changes = len(self._enemy_dir_change_times)
        return self._enemy_velocity_smooth

    def _get_lead_offset(self, distance, playstyle):
        """Return predicted enemy movement during projectile travel time.

        Enhanced prediction v2:
          1. Per-brawler projectile speed from brawlers_info.json
          2. Acceleration-aware prediction (not just linear velocity)
          3. Velocity confidence weighting (reduce lead when data is noisy)
          4. Stutter-step / juke detection (frequent dir changes = less lead)
          5. Direction-change dampening (recent dir change = reduce lead)
          6. Distance-scaled max lead cap
          7. Iterative intercept refinement
        """
        # --- 1. Per-brawler projectile speed (px/sec) ---
        brawler_info = self.brawlers_info.get(self.current_brawler, {})
        speed = brawler_info.get('projectile_speed', 0)
        if speed <= 0:
            proj_speed_fallback = {"sniper": 1000, "thrower": 600, "fighter": 800,
                                   "tank": 0, "assassin": 900, "support": 750}
            speed = proj_speed_fallback.get(playstyle, 800)
        if speed <= 0 or distance <= 0:
            return (0, 0)  # Melee brawler - no lead needed

        # --- 2. Initial travel time + velocity-based lead ---
        travel_time = distance / speed
        vx, vy = self._enemy_velocity_smooth

        # --- 3. Acceleration-aware prediction ---
        # s = v*t + 0.5*a*t²  (predict with acceleration for curving movement)
        ax, ay = self._enemy_accel
        accel_mag = math.hypot(ax, ay)
        # Only apply acceleration term if it's significant and consistent
        if accel_mag > 50 and self._velocity_confidence > 0.4:
            lead_x = vx * travel_time + 0.5 * ax * travel_time * travel_time
            lead_y = vy * travel_time + 0.5 * ay * travel_time * travel_time
        else:
            lead_x = vx * travel_time
            lead_y = vy * travel_time

        # --- 4. Iterative intercept refinement ---
        # The lead changes the aim point distance -> recalculate travel time
        refined_dist = math.hypot(distance + lead_x, lead_y)
        if refined_dist > 0 and speed > 0:
            travel_time_2 = refined_dist / speed
            if accel_mag > 50 and self._velocity_confidence > 0.4:
                lead_x = vx * travel_time_2 + 0.5 * ax * travel_time_2 * travel_time_2
                lead_y = vy * travel_time_2 + 0.5 * ay * travel_time_2 * travel_time_2
            else:
                lead_x = vx * travel_time_2
                lead_y = vy * travel_time_2

        # --- 5. Base lead boost ---
        # Compensate for systematic under-prediction: enemy often moves
        # further than simple v*t estimates, especially during combat strafing.
        lead_x *= 1.35
        lead_y *= 1.35

        # --- 6. Multi-projectile spread reduction (mild) ---
        if self._my_projectile_count >= 5:
            lead_x *= 0.6
            lead_y *= 0.6
        elif self._my_projectile_count >= 3:
            lead_x *= 0.8
            lead_y *= 0.8

        # --- 7. Velocity confidence weighting ---
        # Low confidence = blend lead toward zero (aim more at current pos)
        # High confidence = trust the prediction fully
        conf = self._velocity_confidence
        # Minimum confidence floor: always apply at least 50% of computed lead
        conf_factor = 0.5 + 0.5 * conf
        lead_x *= conf_factor
        lead_y *= conf_factor

        # --- 8. Stutter-step / juke detection ---
        # If enemy changed direction many times = erratic movement
        # Reduce lead - but not as aggressively (still better to lead some)
        if self._enemy_dir_changes >= 5:
            # Heavy juking - partial lead, aim closer to center
            lead_x *= 0.4
            lead_y *= 0.4
        elif self._enemy_dir_changes >= 3:
            # Moderate strafing - mild reduction
            lead_x *= 0.7
            lead_y *= 0.7

        # --- 9. Recent direction change dampening ---
        now = time.time()
        dir_hold_time = now - self._enemy_direction_since
        if dir_hold_time < 0.15 and self._enemy_speed_magnitude > 60:
            # Direction JUST changed (<150ms) - mild dampen, recovers fast
            dampen = 0.5 + (dir_hold_time / 0.15) * 0.4  # 0.5->0.9 over 0.15s
            lead_x *= dampen
            lead_y *= dampen

        # --- 10. Distance-scaled max lead ---
        max_lead = min(350, max(80, distance * 0.45))
        lead_mag = math.hypot(lead_x, lead_y)
        if lead_mag > max_lead:
            scale = max_lead / lead_mag
            lead_x *= scale
            lead_y *= scale

        self._last_aim_lead = (lead_x, lead_y)
        return (lead_x, lead_y)

    def _get_intercept_position(self, player_pos, enemy_pos, enemy_distance):
        """Calculate where to move to intercept a fleeing/strafing enemy.
        Returns (target_x, target_y) - the predicted future position to move toward."""
        vx, vy = self._enemy_velocity_smooth
        if self._enemy_speed_magnitude < 30:
            return enemy_pos  # Enemy is stationary or nearly so
        # Estimate time to reach enemy based on our movement speed
        our_speed_px = self._my_movement_speed * 0.5  # Rough px/sec conversion
        if our_speed_px <= 0:
            return enemy_pos
        time_to_reach = enemy_distance / our_speed_px
        # Cap prediction time (don't predict too far into future)
        time_to_reach = min(time_to_reach, 1.0)
        # Predicted future position (vy already isometric-corrected from velocity tracking)
        pred_x = enemy_pos[0] + vx * time_to_reach * 0.6  # 60% weight - don't overshoot
        pred_y = enemy_pos[1] + vy * time_to_reach * 0.6  # vy includes 1.25× isometric
        return (pred_x, pred_y)

    # === BUSH DANGER ASSESSMENT ===
    def _is_bush_dangerous(self, bush):
        """Check if a bush is very close to where we RECENTLY saw an enemy (likely hiding)."""
        now = time.time()
        bush_cx = (bush[0] + bush[2]) / 2
        bush_cy = (bush[1] + bush[3]) / 2
        for ex, ey, t in self._last_known_enemies:
            # Only consider very recent sightings (3s, not full memory)
            if now - t < 3.0:
                # Tight radius: enemy must have been practically inside this bush
                if math.hypot(bush_cx - ex, bush_cy - ey) < 80:
                    return True
        return False

    def _is_path_through_dangerous_bush(self, player_pos, move_direction, distance=None):
        """Check if a movement direction would take us through a bush where an enemy recently was."""
        if not self.last_bush_data or not self._last_known_enemies:
            return False
        if distance is None:
            distance = self.TILE_SIZE * self.window_controller.scale_factor
        dx, dy = 0, 0
        if 'w' in move_direction.lower(): dy -= 1
        if 's' in move_direction.lower(): dy += 1
        if 'a' in move_direction.lower(): dx -= 1
        if 'd' in move_direction.lower(): dx += 1
        length = math.hypot(dx, dy)
        if length == 0:
            return False
        dx /= length
        dy /= length
        end_pos = (player_pos[0] + dx * distance, player_pos[1] + dy * distance)
        path_line = LineString([player_pos, end_pos])
        for bush in self.last_bush_data:
            if self._is_bush_dangerous(bush):
                bx1, by1, bx2, by2 = bush
                bush_poly = Polygon([(bx1, by1), (bx2, by1), (bx2, by2), (bx1, by2)])
                if path_line.intersects(bush_poly):
                    return True
        return False

    def _find_nearest_bush_in_range(self, player_pos, attack_range):
        """Find the closest bush within attack range that could hide an enemy."""
        best_bush = None
        best_dist = float('inf')
        for bush in self.last_bush_data:
            bush_cx = (bush[0] + bush[2]) / 2
            bush_cy = (bush[1] + bush[3]) / 2
            dist = self.get_distance((bush_cx, bush_cy), player_pos)
            if dist <= attack_range and dist < best_dist:
                best_dist = dist
                best_bush = (bush_cx, bush_cy)
        return best_bush, best_dist

    # === RETREAT PATH SAFETY SCORING (dead-end avoidance) ===
    def _score_retreat_path(self, player_pos, direction, walls):
        """Score a retreat direction by how many escape routes the destination has."""
        dx, dy = 0, 0
        if 'w' in direction.lower(): dy -= 1
        if 's' in direction.lower(): dy += 1
        if 'a' in direction.lower(): dx -= 1
        if 'd' in direction.lower(): dx += 1
        length = math.hypot(dx, dy)
        if length == 0:
            return 0.0
        dx /= length
        dy /= length
        step = self.TILE_SIZE * self.window_controller.scale_factor
        dest = (player_pos[0] + dx * step, player_pos[1] + dy * step)
        open_routes = 0
        for d in self.ALL_DIRECTIONS:
            if not self.is_path_blocked(dest, d, walls, distance=step * 0.6):
                open_routes += 1
        if self.is_path_blocked(player_pos, direction, walls, distance=step * 0.5):
            return -10.0
        return open_routes * 10.0

    def _get_safe_retreat_direction(self, player_pos, enemy_dir_x, enemy_dir_y, walls, spawn_side=None):
        """Pick the best retreat direction that avoids dead-ends."""
        now = time.time()
        if now - self._retreat_cache_time > 0.1:
            self._retreat_safety_cache = {}
            self._retreat_cache_time = now
        primary_h = 'A' if enemy_dir_x > 0 else 'D'
        primary_v = 'W' if enemy_dir_y > 0 else 'S'
        spawn_h, spawn_v = primary_h, primary_v
        if spawn_side == 'left': spawn_h = 'A'
        elif spawn_side == 'right': spawn_h = 'D'
        elif spawn_side == 'top': spawn_v = 'W'
        elif spawn_side == 'bottom': spawn_v = 'S'
        candidates = [
            primary_v + primary_h,
            primary_v,
            primary_h,
            spawn_v + spawn_h,
            primary_v + ('D' if primary_h == 'A' else 'A'),
        ]
        seen = set()
        unique = []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                unique.append(c)
        best_dir = unique[0] if unique else primary_v + primary_h
        best_score = float('-inf')
        for d in unique:
            if d not in self._retreat_safety_cache:
                self._retreat_safety_cache[d] = self._score_retreat_path(player_pos, d, walls)
            score = self._retreat_safety_cache[d]
            if score > best_score:
                best_score = score
                best_dir = d
        return best_dir

    # === JUKE PATTERN ENTROPY (anti-prediction dodging) ===
    def _get_juke_direction(self):
        """Return a strafe key from the current juke pattern sequence."""
        now = time.time()
        if (self._juke_current_pattern is None
                or self._juke_step >= len(self._juke_current_pattern)):
            self._juke_current_pattern = random.choice(self._juke_patterns)
            self._juke_step = 0
            self._juke_step_start = now
        current_key, current_dur = self._juke_current_pattern[self._juke_step]
        if now - self._juke_step_start >= current_dur:
            self._juke_step += 1
            self._juke_step_start = now
            if self._juke_step >= len(self._juke_current_pattern):
                self._juke_current_pattern = random.choice(self._juke_patterns)
                self._juke_step = 0
                self._juke_step_start = now
            current_key, current_dur = self._juke_current_pattern[self._juke_step]
        return current_key

    # === LANE ASSIGNMENT (3v3 modes) ===
    def _assign_lane(self, player_pos, teammates):
        """Assign bot to the least-covered lane based on teammate positions."""
        w = brawl_stars_width * self.window_controller.width_ratio
        h = brawl_stars_height * self.window_controller.height_ratio
        if self.game_mode == 5:
            lane_names = ['top', 'mid', 'bottom']
            axis = 1
            strip_sz = h / 3
        else:
            lane_names = ['left', 'center', 'right']
            axis = 0
            strip_sz = w / 3
        counts = [0, 0, 0]
        for tm in teammates:
            tx = (tm[0] + tm[2]) / 2
            ty = (tm[1] + tm[3]) / 2
            pos = tx if axis == 0 else ty
            lane_idx = min(2, int(pos / strip_sz))
            counts[lane_idx] += 1
        p_pos = player_pos[0] if axis == 0 else player_pos[1]
        p_lane = min(2, int(p_pos / strip_sz))
        min_count = min(counts)
        best_lanes = [i for i, c in enumerate(counts) if c == min_count]
        chosen = p_lane if p_lane in best_lanes else best_lanes[0]
        self._assigned_lane = lane_names[chosen]
        thirds = [strip_sz / 2, strip_sz * 1.5, strip_sz * 2.5]
        if axis == 0:
            self._lane_center = (thirds[chosen], h / 2)
        else:
            self._lane_center = (w / 2, thirds[chosen])

    # === ENEMY RELOAD WINDOW DETECTION ===
    def _detect_enemy_fired(self, enemy_data, player_pos, player_hp_dropped):
        """If our HP dropped while an enemy is in range, that enemy just fired."""
        now = time.time()
        if not player_hp_dropped or not enemy_data:
            return
        for enemy in enemy_data:
            epos = self.get_enemy_pos(enemy)
            dist = self.get_distance(epos, player_pos)
            if dist < 500:
                bucket = (int(epos[0] / 50), int(epos[1] / 50))
                self._enemy_last_attack_time[bucket] = now
                break
        self._enemy_last_attack_time = {
            k: v for k, v in self._enemy_last_attack_time.items()
            if now - v < 5.0
        }

    def _is_target_reloading(self, enemy_pos):
        """Check if the target enemy is likely in a reload window."""
        now = time.time()
        bucket = (int(enemy_pos[0] / 50), int(enemy_pos[1] / 50))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                key = (bucket[0] + dx, bucket[1] + dy)
                last_attack = self._enemy_last_attack_time.get(key, 0)
                if 0 < (now - last_attack) < 1.6:
                    return True
        return False

    def _is_storm_flee_delay_over(self, current_time=None):
        """Return True once storm/gas flee behavior is allowed for this round."""
        if current_time is None:
            current_time = time.time()
        if self._match_start_time <= 0:
            return False
        return (current_time - self._match_start_time) >= self._storm_flee_delay_sec

    # === SHOWDOWN STORM ZONE DETECTION ===
    def _detect_storm_zone(self, frame):
        """Detect the Showdown poison/storm zone by scanning for blue/purple edges."""
        try:
            if not self._is_storm_flee_delay_over():
                self._storm_radius = 9999
                self._in_storm = False
                return

            arr = np.asarray(frame)
            fh, fw = arr.shape[:2]
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array([110, 50, 30]), np.array([155, 255, 180]))
            edge_w = max(10, fw // 20)
            edge_h = max(10, fh // 20)
            left_storm = np.mean(mask[:, :edge_w]) / 255
            right_storm = np.mean(mask[:, -edge_w:]) / 255
            top_storm = np.mean(mask[:edge_h, :]) / 255
            bottom_storm = np.mean(mask[-edge_h:, :]) / 255
            if max(left_storm, right_storm, top_storm, bottom_storm) < 0.15:
                self._storm_radius = 9999
                self._in_storm = False
                return
            cx_shift = (left_storm - right_storm) * fw * 0.3
            cy_shift = (top_storm - bottom_storm) * fh * 0.3
            self._storm_center = (fw / 2 + cx_shift, fh / 2 + cy_shift)
            total_storm = (left_storm + right_storm + top_storm + bottom_storm) / 4
            self._storm_radius = max(100, fw * 0.5 * (1.0 - total_storm))
            center_storm = np.mean(mask[fh//3:2*fh//3, fw//3:2*fw//3]) / 255
            self._in_storm = center_storm > 0.3
        except Exception:
            pass

    def _detect_poison_gas(self, frame):
        """Detect poison gas (Knockout overtime, etc.) and compute
        a directional avoidance vector.  Works for ANY game mode.
        
        Detects hazard types:
        - GREEN gas (poison clouds) - the main hazard
        - BLUE/PURPLE storm edge (Showdown)
        
        NOTE: We do NOT detect red/orange as those are often just map decorations
        (lava at edges) which causes massive false positives.
        """
        try:
            arr = np.asarray(frame)
            fh, fw = arr.shape[:2]
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)

            # GREEN gas HSV range - bright green poison clouds
            # Hue 35-85 (green), Sat 40-255, Val 60-255
            green_mask = cv2.inRange(hsv, np.array([35, 40, 60]), np.array([85, 255, 255]))
            
            # Use ONLY green mask for gas detection
            # Red/orange decorations at map edges cause too many false positives
            gas_mask = green_mask

            # Divide frame into a 3×3 grid and measure gas density each region
            rh, rw = fh // 3, fw // 3
            densities = {}
            for gy in range(3):
                for gx in range(3):
                    region = gas_mask[gy*rh:(gy+1)*rh, gx*rw:(gx+1)*rw]
                    densities[(gx, gy)] = np.mean(region) / 255

            self._gas_density_map = densities

            # Strict timing gate: never activate gas/storm flee states in the first 30s.
            if not self._is_storm_flee_delay_over():
                self._gas_active = False
                self._in_storm = False
                if not self.is_showdown and self._storm_radius < 5000:
                    self._storm_radius = 9999
                return

            # Gas is 'active' if any outer region has >30% gas pixels (strict threshold)
            outer_max = max(
                densities.get((0,0), 0), densities.get((1,0), 0), densities.get((2,0), 0),
                densities.get((0,1), 0),                          densities.get((2,1), 0),
                densities.get((0,2), 0), densities.get((1,2), 0), densities.get((2,2), 0),
            )
            self._gas_active = outer_max > 0.30  # Strict threshold to avoid false positives

            # Are we IN the gas? Center region has significant gas
            center_gas = densities.get((1,1), 0)
            if center_gas > 0.25:  # Stricter threshold
                self._in_storm = True
            elif not self.is_showdown:
                # Only clear _in_storm if we're not also detecting blue storm
                if self._storm_radius > 5000:
                    self._in_storm = center_gas > 0.15

            # Compute safe zone center = opposite of gas direction
            if self._gas_active:
                # Weighted centroid of gas-free regions
                safe_x, safe_y, safe_w = 0, 0, 0.001
                for (gx, gy), d in densities.items():
                    freedom = max(0, 1.0 - d * 3)  # Higher = less gas
                    cx = (gx + 0.5) * rw
                    cy = (gy + 0.5) * rh
                    safe_x += cx * freedom
                    safe_y += cy * freedom
                    safe_w += freedom
                safe_x /= safe_w
                safe_y /= safe_w

                # Update storm center/radius for the flee logic to use
                if not self.is_showdown or self._storm_radius > 5000:
                    self._storm_center = (safe_x, safe_y)
                    # Estimate radius from gas coverage
                    total_gas = sum(densities.values()) / 9
                    self._storm_radius = max(80, fw * 0.5 * (1.0 - total_gas * 2))
                    if self._gas_active:
                        print(f"[GAS] Poison detected! center=({int(safe_x)},{int(safe_y)}) "
                              f"density={total_gas:.2f} in_gas={self._in_storm}")
            elif not self.is_showdown:
                # No gas detected - clear storm state
                if self._storm_radius < 5000:
                    self._storm_radius = 9999
                    self._in_storm = False
                    self._gas_active = False
        except Exception:
            pass

    # === PEEK-SHOOT WALL CYCLING ===
    def _update_peek_cycle(self, player_pos, enemy_pos, walls, attack_range):
        """Manage peek-shoot state machine: idle->expose->fire->hide->idle.
        Returns a movement override string, or None to use normal movement."""
        now = time.time()
        enemy_dist = self.get_distance(enemy_pos, player_pos)
        enemy_hittable = self.is_enemy_hittable(player_pos, enemy_pos, walls, "attack")

        # Find nearby walls to use as cover anchor
        if self._peek_phase == 'idle':
            if enemy_dist > attack_range * 1.3 or enemy_dist < 80:
                return None  # Too far or too close for peek
            # Look for a wall within 120px that blocks enemy LOS
            best_wall = None
            best_dist = float('inf')
            for wall in walls:
                wcx = (wall[0] + wall[2]) / 2
                wcy = (wall[1] + wall[3]) / 2
                wd = math.hypot(wcx - player_pos[0], wcy - player_pos[1])
                if 30 < wd < 150:
                    # Wall should be roughly between us and enemy
                    wall_line = LineString([player_pos, enemy_pos])
                    wall_poly = Polygon([(wall[0], wall[1]), (wall[2], wall[1]),
                                         (wall[2], wall[3]), (wall[0], wall[3])])
                    if wall_poly.distance(LineString([player_pos, enemy_pos])) < 80:
                        if wd < best_dist:
                            best_dist = wd
                            best_wall = (wcx, wcy)
            if best_wall:
                self._peek_wall_anchor = best_wall
                self._peek_phase = 'expose'
                self._peek_timer = now
                # Step out perpendicular to wall-enemy line
                to_enemy_x = enemy_pos[0] - best_wall[0]
                to_enemy_y = enemy_pos[1] - best_wall[1]
                # Step perpendicular to get around the wall
                perp_x = -to_enemy_y
                perp_y = to_enemy_x
                self._peek_expose_dir = ('D' if perp_x > 0 else 'A') + ('S' if perp_y > 0 else 'W')
            return None

        elif self._peek_phase == 'expose':
            # Step out from cover (0.25s)
            if now - self._peek_timer > 0.25:
                self._peek_phase = 'fire'
                self._peek_timer = now
            return self._peek_expose_dir

        elif self._peek_phase == 'fire':
            # Fire (done in attack logic - just hold position briefly)
            if now - self._peek_timer > 0.15:
                self._peek_phase = 'hide'
                self._peek_timer = now
                # Reverse the expose direction to go back behind wall
            return ''  # Hold still during fire frame

        elif self._peek_phase == 'hide':
            # Retreat back behind wall (0.3s)
            if now - self._peek_timer > 0.3:
                self._peek_phase = 'idle'
                self._peek_timer = now
                return None
            # Reverse expose direction
            rev = self._peek_expose_dir.translate(str.maketrans('WASD', 'SDWA'))
            return rev

        return None

    # === REACTIVE PERPENDICULAR DODGE ===
    def _trigger_reactive_dodge(self, player_pos, attacker_pos):
        """Compute perpendicular dodge direction when we take damage."""
        dx = attacker_pos[0] - player_pos[0]
        dy = attacker_pos[1] - player_pos[1]
        # Perpendicular: rotate 90° (randomly left or right)
        if random.random() < 0.5:
            perp_x, perp_y = -dy, dx
        else:
            perp_x, perp_y = dy, -dx
        h = 'D' if perp_x > 0 else 'A'
        v = 'S' if perp_y > 0 else 'W'
        self._reactive_dodge_keys = (v + h).lower()
        self._reactive_dodge_until = time.time() + random.uniform(0.2, 0.4)
        self._last_damage_pos = attacker_pos

    # === CHOKE POINT DETECTION ===
    def _detect_choke_points(self, walls):
        """Find narrow passages between walls - areas where movement is funneled."""
        chokes = []
        if len(walls) < 2:
            self._choke_points = []
            return
        # Check pairs of nearby walls for narrow gaps
        for i in range(len(walls)):
            w1 = walls[i]
            w1_cx, w1_cy = (w1[0] + w1[2]) / 2, (w1[1] + w1[3]) / 2
            for j in range(i + 1, min(len(walls), i + 20)):  # Limit pairs for performance
                w2 = walls[j]
                w2_cx, w2_cy = (w2[0] + w2[2]) / 2, (w2[1] + w2[3]) / 2
                gap_dist = math.hypot(w2_cx - w1_cx, w2_cy - w1_cy)
                # A choke point is where two walls are 60-180px apart
                if 60 < gap_dist < 180:
                    # Verify there's open space between them (no wall blocking)
                    mid_x = (w1_cx + w2_cx) / 2
                    mid_y = (w1_cy + w2_cy) / 2
                    gap_blocked = False
                    for k, w3 in enumerate(walls):
                        if k == i or k == j:
                            continue
                        w3_cx = (w3[0] + w3[2]) / 2
                        w3_cy = (w3[1] + w3[3]) / 2
                        if math.hypot(w3_cx - mid_x, w3_cy - mid_y) < gap_dist * 0.4:
                            gap_blocked = True
                            break
                    if not gap_blocked:
                        angle = math.atan2(w2_cy - w1_cy, w2_cx - w1_cx)
                        chokes.append((mid_x, mid_y, gap_dist, angle))
        self._choke_points = chokes

    # === MATCH PHASE COMPUTATION ===
    def _update_match_phase(self, current_time):
        """Determine whether we're in early/mid/late game based on elapsed time."""
        if not self._match_phase_set:
            return
        elapsed = current_time - self._match_start_time
        if elapsed < 30:
            self._match_phase = 'early'
        elif elapsed < 90:
            self._match_phase = 'mid'
        else:
            self._match_phase = 'late'

    # === SUPER VALUE HOLD CHECK ===
    def _should_hold_super(self, super_type, n_enemies, enemy_is_low, enemy_distance, super_range):
        """Decide if we should save super for a higher-value moment."""
        # Never hold utility/self-buff supers
        if super_type in ["spawnable", "other", "other_target", "charge"]:
            return False
        # Always use if: finishing blow, multiple enemies, or losing
        if enemy_is_low:
            return False
        if n_enemies >= 2:
            return False
        if self._score_diff <= -1:
            return False
        if self._match_phase == 'late':
            return False  # Late game = use everything
        # Hold when winning comfortably and only 1 enemy
        if self._score_diff >= 2 and n_enemies <= 1:
            return True
        return False

    # === DISENGAGE-TO-HEAL PROTOCOL ===
    def _should_disengage_for_heal(self, player_hp, enemy_distance, safe_range=300):
        """Check if bot should retreat to heal rather than keep fighting.
        Returns True when moderately wounded and enemy isn't too close."""
        now = time.time()
        # If already disengaging, continue until timer ends
        if self._disengage_active and now < self._disengage_until:
            return True
        if self._disengage_active and now >= self._disengage_until:
            self._disengage_active = False
            return False
        # Trigger disengage: moderate HP + enemy at mid range
        if 15 < player_hp < 55 and enemy_distance > safe_range * 0.6:
            # Don't disengage in late game - too risky
            if self._match_phase == 'late':
                return False
            # More likely to disengage at lower HP
            threshold = 0.7 if player_hp < 35 else 0.35
            if random.random() < threshold:
                self._disengage_active = True
                self._disengage_until = now + random.uniform(1.5, 3.0)
                return True
        return False

    # === TEAMMATE DEATH GAP COVER ===
    def _handle_teammate_death_gap(self, teammate_coords_list, player_pos, game_mode):
        """When a teammate dies, temporarily shift lane to cover the gap."""
        now = time.time()
        current_count = len(teammate_coords_list)

        # If a teammate was lost since last check
        if current_count < self._prev_teammate_count:
            if self._prev_teammate_count > 0 and game_mode in ['gemGrab', 'hotZone', 'brawlBall', 'knockout', 'bounty']:
                # Estimate where the gap is - midpoint between us and last teammate
                if teammate_coords_list:
                    t = teammate_coords_list[0]
                    gap_x = (player_pos[0] + t[0]) / 2
                    gap_y = (player_pos[1] + t[1]) / 2
                else:
                    # No teammates left - move toward center
                    gap_x = 640 / 2
                    gap_y = 360 / 2
                self._teammate_death_pos = (gap_x, gap_y)
                self._teammate_death_time = now
                # Save our current lane so we can return later
                if self._original_lane_center is None:
                    self._original_lane_center = self._lane_center

        self._prev_teammate_count = current_count

        # While gap-covering (for up to 8 seconds after teammate death)
        if self._teammate_death_pos and now - self._teammate_death_time < 8.0:
            return self._teammate_death_pos  # Movement target to cover gap
        elif self._teammate_death_pos and now - self._teammate_death_time >= 8.0:
            # Teammate should have respawned - return to original lane
            self._teammate_death_pos = None
            if self._original_lane_center is not None:
                self._lane_center = self._original_lane_center
                self._original_lane_center = None
        return None

    # === THREAT-SCORED TARGETING ===
    def find_best_target(self, enemy_data, player_coords, walls, skill_type, attack_range):

        """Score all enemies and pick the best target - not just the closest.
        Considers: distance, estimated HP, hittability, kill-potential."""
        best_target = None
        best_score = float('-inf')
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            hittable = self.is_enemy_hittable(player_coords, enemy_pos, walls, skill_type)

            # Use bbox size as rough HP proxy (smaller bbox = possibly lower HP)
            bbox_w = enemy[2] - enemy[0]
            bbox_h = enemy[3] - enemy[1]
            size_factor = bbox_w * bbox_h  # Larger = tankier

            # Scoring formula
            score = 0.0
            score -= distance * 0.3                       # Closer = better
            score += 200 if hittable else -500            # STRONG penalty for unhittable (behind wall)
            score += 100 if distance <= attack_range else 0  # In-range bonus
            score -= size_factor * 0.001                   # Smaller targets = easier kills
            # If we've been hitting this enemy, prefer to focus-fire
            if self.target_info.get('bbox') is not None:
                prev = self.target_info['bbox']
                prev_pos = self.get_enemy_pos(prev)
                if self.get_distance(enemy_pos, prev_pos) < 80:
                    score += 120  # Focus-fire bonus for same target

            # === TEAM FOCUS FIRE: prefer enemies that our teammates are already engaging ===
            # Enemies close to teammates = teammates are fighting them = we should help
            if hasattr(self, '_teammate_positions') and self._teammate_positions:
                for tm_pos in self._teammate_positions:
                    tm_to_enemy = math.hypot(enemy_pos[0] - tm_pos[0], enemy_pos[1] - tm_pos[1])
                    if tm_to_enemy < attack_range * 0.8:
                        score += 100  # Big bonus: teammate is engaging this enemy
                    elif tm_to_enemy < attack_range * 1.2:
                        score += 50   # Moderate: enemy is near a teammate

            # Lane bonus: prefer enemies in our assigned lane
            if self._lane_center and self._assigned_lane:
                lx, ly = self._lane_center
                enemy_lane_dist = abs(enemy_pos[0] - lx) if self.game_mode != 5 else abs(enemy_pos[1] - ly)
                if enemy_lane_dist < 200:
                    score += 60  # Bonus for enemies in our lane

            # === DEATH POSITION PENALTY: avoid targeting ghost detections near kill sites ===
            if self._is_near_death_position(enemy_pos[0], enemy_pos[1]):
                score -= 400  # Heavy penalty - this might be a ghost detection of a dead enemy

            if score > best_score:
                best_score = score
                best_target = [enemy_pos, distance]

        return best_target if best_target else (None, None)

    @staticmethod
    def read_name_above_bbox(frame, bbox):
        """
        Read the name text displayed above a character's bounding box.
        Returns the name string or None.
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            char_width = x2 - x1
            if char_width < 15:
                return None

            frame_arr = np.asarray(frame)
            fh, fw = frame_arr.shape[:2]

            # Name text sits above the HP bar: ~25-50px above bbox top
            name_y1 = max(0, y1 - 55)
            name_y2 = max(0, y1 - 10)
            name_x1 = max(0, x1 - 30)
            name_x2 = min(fw, x2 + 30)

            if name_y1 >= name_y2 or name_x1 >= name_x2:
                return None

            name_crop = frame_arr[name_y1:name_y2, name_x1:name_x2]
            if name_crop.size == 0:
                return None

            from utils import extract_text_and_positions
            texts = extract_text_and_positions(name_crop)
            if texts:
                # Return the longest text found (likely the name)
                best = max(texts.keys(), key=len)
                # Filter out obvious non-names (numbers only, tiny text)
                clean = best.strip()
                if len(clean) >= 2:
                    return clean
            return None
        except Exception:
            return None

    @staticmethod
    def read_enemy_hp_number_fast(frame, bbox, _ocr_cache={}):
        """
        FAST HP number reading using cached OCR and optimized crop.
        Uses color filtering to isolate HP text (green/yellow/red numbers).
        Returns raw HP value as integer, or None if not readable.
        """
        try:
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            char_width = x2 - x1
            if char_width < 20:
                return None

            frame_arr = np.asarray(frame)
            fh, fw = frame_arr.shape[:2]

            # HP number is above the character, below name text
            # Narrow crop for speed: only 20px height, centered on expected location
            hp_y1 = max(0, y1 - 32)
            hp_y2 = max(0, y1 - 12)
            hp_x1 = max(0, x1 - 15)
            hp_x2 = min(fw, x2 + 15)

            if hp_y1 >= hp_y2 or hp_x1 >= hp_x2:
                return None

            hp_crop = frame_arr[hp_y1:hp_y2, hp_x1:hp_x2]
            if hp_crop.size == 0 or hp_crop.shape[0] < 8:
                return None

            # Color filter: HP numbers are bright (green/yellow/red text)
            # Convert to HSV and filter for bright colored text
            hsv = cv2.cvtColor(hp_crop, cv2.COLOR_RGB2HSV)
            
            # Bright text mask (V > 180 for white/bright colored numbers)
            bright_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 100, 255]))
            # Green HP text
            green_mask = cv2.inRange(hsv, np.array([35, 80, 150]), np.array([85, 255, 255]))
            # Yellow HP text  
            yellow_mask = cv2.inRange(hsv, np.array([15, 100, 150]), np.array([35, 255, 255]))
            # Red HP text (low HP)
            red_mask = cv2.inRange(hsv, np.array([0, 100, 150]), np.array([15, 255, 255]))
            
            combined_mask = cv2.bitwise_or(bright_mask, green_mask)
            combined_mask = cv2.bitwise_or(combined_mask, yellow_mask)
            combined_mask = cv2.bitwise_or(combined_mask, red_mask)
            
            # Count colored pixels - if too few, no HP number visible
            if np.count_nonzero(combined_mask) < 15:
                return None

            # Apply mask to crop for OCR
            masked_crop = cv2.bitwise_and(hp_crop, hp_crop, mask=combined_mask)
            
            # Use cached OCR reader for speed
            if 'reader' not in _ocr_cache:
                import easyocr
                _ocr_cache['reader'] = easyocr.Reader(['en'], gpu=True, verbose=False)
            
            reader = _ocr_cache['reader']
            # Fast OCR with limited settings
            results = reader.readtext(masked_crop, detail=0, paragraph=False, 
                                     allowlist='0123456789', batch_size=1)
            
            if results:
                for text in results:
                    digits = ''.join(c for c in text if c.isdigit())
                    if digits and 1 <= len(digits) <= 5:
                        hp_val = int(digits)
                        if 1 <= hp_val <= 12000:
                            return hp_val
            return None
        except Exception:
            return None

    @staticmethod
    def read_enemy_hp_number(frame, bbox):
        """
        Read the HP NUMBER displayed below enemy name (e.g., '40', '1760', '4000').
        The HP shows as a number that pulses between green/yellow/red based on HP level.
        Returns the raw HP value as integer, or None if not readable.
        """
        # Use fast version with caching
        return Play.read_enemy_hp_number_fast(frame, bbox)
        
    @staticmethod
    def get_enemy_pos(enemy):
        return (enemy[0] + enemy[2]) / 2, (enemy[1] + enemy[3]) / 2

    @staticmethod
    def get_player_pos(player_data):
        return (player_data[0] + player_data[2]) / 2, (player_data[1] + player_data[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        dx = enemy_coords[0] - player_coords[0]
        dy = (enemy_coords[1] - player_coords[1]) * 1.25  # Isometric camera correction
        return math.hypot(dx, dy)

    @staticmethod
    def is_there_enemy(enemy_data):
        if not enemy_data:
            return False
        return True

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

    def attack(self, touch_up=True, touch_down=True):
        self.window_controller.press_key("M", touch_up=touch_up, touch_down=touch_down)
        self.last_attack_time = time.time()
        self.is_regenerating = False

    def use_hypercharge(self):
        now = time.time()
        if now - self.time_since_hypercharge_used < self.HYPERCHARGE_USE_COOLDOWN:
            return False
        print("Using hypercharge")
        self.window_controller.press_key("H")
        self.time_since_hypercharge_used = now
        return True

    def use_gadget(self):
        now = time.time()
        if now - self.time_since_gadget_used < self.GADGET_USE_COOLDOWN:
            return False  # still on cooldown
        print("Using gadget")
        self.window_controller.press_key("G")
        self.time_since_gadget_used = now
        return True

    def use_super(self):
        now = time.time()
        if now - self.time_since_super_used < self.SUPER_USE_COOLDOWN:
            return False
        print("Using super")
        # Snapshot walls before super (for wall destruction detection)
        self._pre_super_walls = list(self.last_walls_data) if self.last_walls_data else None
        self._force_wall_refresh = True  # Force fresh tile scan after super
        self.window_controller.press_key("E")
        self.time_since_super_used = now
        return True

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

    def unstuck_movement_if_needed(self, movement, current_time=None):
        if current_time is None:
            current_time = time.time()
        movement = movement.lower()
        if self.fix_movement_keys['toggled']:
            if current_time - self.fix_movement_keys['started_at'] > self.fix_movement_keys['duration']:
                self.fix_movement_keys['toggled'] = False
                # Reset escalation if we unstuck naturally
                if self._stuck_escalation > 0:
                    self._stuck_escalation = max(0, self._stuck_escalation - 1)
                # Reset stuck timers so we don't immediately re-trigger
                self.time_since_different_movement = current_time
                self._stuck_check_pos = None
                self._position_history.clear()
                return movement  # Return normal movement, override done

            return self.fix_movement_keys['fixed']

        if "".join(self.keys_hold) != movement and movement[::-1] != "".join(self.keys_hold):
            # Movement keys changed - but DON'T reset stuck timer!
            # In BT mode, keys change every tick (WA->WD->W->WA) even when
            # stuck in a U-trap.  Position-based check below is authoritative.
            pass

        # Get current player position for real stuck detection
        cur_player_pos = None
        try:
            if hasattr(self, 'time_since_player_last_found'):
                # Only check position if we recently saw the player
                if current_time - self.time_since_player_last_found < 1.0:
                    # Use stored position from last get_movement call
                    cur_player_pos = getattr(self, '_last_known_player_pos', None)
        except Exception:
            pass

        # Position-based stuck check: if our position has actually changed, we're NOT stuck
        if cur_player_pos and self._stuck_check_pos:
            moved_dist = math.hypot(
                cur_player_pos[0] - self._stuck_check_pos[0],
                cur_player_pos[1] - self._stuck_check_pos[1])
            if moved_dist > self._stuck_pos_threshold:
                # We've actually moved - not stuck, reset timer
                self.time_since_different_movement = current_time
                self._stuck_check_pos = cur_player_pos
                self._stuck_escalation = 0
                return movement
        elif cur_player_pos and not self._stuck_check_pos:
            self._stuck_check_pos = cur_player_pos

        # --- EARLY CONCAVITY ESCAPE: before even waiting for stuck timer ---
        # If we're trapped in a U-shape, override direction immediately.
        # COOLDOWN: only re-check every 5s to avoid constant false-positive
        # overrides that prevent the BT from moving normally.
        if not hasattr(self, '_last_utrap_time'):
            self._last_utrap_time = 0.0
        utrap_cooldown = 5.0  # seconds between U-trap overrides (increased from 2s)
        if (cur_player_pos and hasattr(self, '_spatial_memory') and self._spatial_memory is not None
                and current_time - self._last_utrap_time >= utrap_cooldown):
            from bt_combat import _detect_concavity
            _onnx_walls = getattr(self, 'last_walls_data', None) or []
            escape_dir, blocked_n = _detect_concavity(cur_player_pos, walls=_onnx_walls)
            if escape_dir and blocked_n >= 4:
                # Only trigger on FULL concavity (4/4), require position
                # didn't change much from last check to confirm genuinely stuck
                prev_trap_pos = getattr(self, '_last_utrap_pos', None)
                if prev_trap_pos:
                    trap_moved = math.hypot(cur_player_pos[0] - prev_trap_pos[0],
                                           cur_player_pos[1] - prev_trap_pos[1])
                else:
                    trap_moved = 0
                if trap_moved < 35:  # Barely moved since last check -> genuinely stuck (reduced from 60)
                    self.fix_movement_keys['fixed'] = escape_dir.lower()
                    self.fix_movement_keys['toggled'] = True
                    self.fix_movement_keys['started_at'] = current_time
                    self.fix_movement_keys['duration'] = 1.5  # Long enough to actually leave U-trap
                    self.last_decision_reason = f"U-TRAP ESCAPE ({blocked_n}/4 blocked) -> {escape_dir}"
                    print(f"[TRAP] U-trap detected at ({int(cur_player_pos[0])},{int(cur_player_pos[1])}) "
                          f"- {blocked_n}/4 blocked -> escape {escape_dir}")
                    self._last_utrap_time = current_time
                    self._last_utrap_pos = cur_player_pos
                    # Reset BT patrol state so PUSH FORWARD picks new direction after escape
                    if hasattr(self, '_bt_combat') and self._bt_combat is not None:
                        _bb = self._bt_combat.blackboard
                        _bb.set("_patrol_tick", 0)
                        _bb.set("_patrol_target", None)
                    return escape_dir.lower()
                self._last_utrap_pos = cur_player_pos
                self._last_utrap_time = current_time
            else:
                self._last_utrap_pos = cur_player_pos

        # --- OSCILLATION DETECTION: track position history ---
        # Even if the bot moves > threshold each tick, it might be bouncing
        # back and forth in a small area (U-trap). Track bounding box of
        # recent positions to detect this.
        is_oscillating = False
        if cur_player_pos:
            self._position_history.append((cur_player_pos, current_time))
            # Keep only recent positions (last ~3 seconds)
            cutoff = current_time - 3.0
            self._position_history = [(p, t) for p, t in self._position_history if t > cutoff]
            # Trim to max length
            if len(self._position_history) > self._position_history_max:
                self._position_history = self._position_history[-self._position_history_max:]
            # Check bounding box of positions
            if len(self._position_history) >= 8:  # Need enough history (was 15 - too slow)
                xs = [p[0] for p, t in self._position_history]
                ys = [p[1] for p, t in self._position_history]
                bbox_w = max(xs) - min(xs)
                bbox_h = max(ys) - min(ys)
                bbox_diag = math.hypot(bbox_w, bbox_h)
                if bbox_diag < self._oscillation_bbox_threshold:
                    is_oscillating = True

        # When oscillating (stuck in U-trap), force anti-stuck even without enemies
        if is_oscillating:
            # Don't skip anti-stuck - override the no-enemy bypass
            if current_time - self.time_since_different_movement > 0.8:  # Faster trigger when oscillating
                print(f"[STUCK] Oscillation detected! bbox={bbox_diag:.0f}px < {self._oscillation_bbox_threshold}px over {len(self._position_history)} positions")
                # Force stuck trigger even without enemies
                pass  # Fall through to normal anti-stuck logic below
            elif not self._has_enemy_target:
                # Not yet long enough - but still track the no-enemy case
                return movement
        elif not self._has_enemy_target:
            # No enemies — still trigger anti-stuck if stuck for >4s
            # (prevents getting permanently stuck in spawn with no enemy in sight)
            idle_dur = current_time - self.time_since_different_movement
            if idle_dur < 4.0:
                self.time_since_different_movement = current_time
                return movement
            else:
                print(f"[STUCK] No enemies, idle {idle_dur:.1f}s — forcing unstuck")
                # Fall through to anti-stuck logic

        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            # === PATHFINDER-BASED UNSTUCK (replaces ghost wall injection) ===
            # Notify pathfinder that we're stuck at this position so it adds
            # penalty costs around here and replans.
            if cur_player_pos and hasattr(self, '_path_planner') and self._path_planner is not None:
                self._path_planner.report_stuck(cur_player_pos)
                print(f"[STUCK] Pathfinder notified - will replan with penalty at ({int(cur_player_pos[0])}, {int(cur_player_pos[1])})")

            # Also inject ghost walls into spatial memory grid (temporary, auto-expires)
            # Inject a 3-wide wall perpendicular to movement direction for better blocking
            if cur_player_pos and movement:
                gdx, gdy = 0, 0
                if 'w' in movement: gdy = -1
                if 's' in movement: gdy = 1
                if 'a' in movement: gdx = -1
                if 'd' in movement: gdx = 1
                if gdx != 0 or gdy != 0:
                    sf = getattr(self.window_controller, 'scale_factor', 1.0)
                    cell_sz = 40 * sf
                    ghost_dist = 50 * sf
                    gx = cur_player_pos[0] + gdx * ghost_dist
                    gy = cur_player_pos[1] + gdy * ghost_dist
                    if hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
                        # Inject center + perpendicular neighbours for a wider wall
                        self._spatial_memory.inject_ghost_wall(gx, gy, duration=8.0)
                        # Perpendicular offsets (swap dx/dy)
                        perp_dx, perp_dy = -gdy, gdx
                        self._spatial_memory.inject_ghost_wall(
                            gx + perp_dx * cell_sz, gy + perp_dy * cell_sz, duration=8.0)
                        self._spatial_memory.inject_ghost_wall(
                            gx - perp_dx * cell_sz, gy - perp_dy * cell_sz, duration=8.0)

            # Track how often we trigger anti-stuck
            self._stuck_trigger_times = [t for t in self._stuck_trigger_times if current_time - t < 8.0]
            self._stuck_trigger_times.append(current_time)

            # Escalate if we've triggered anti-stuck multiple times recently
            if len(self._stuck_trigger_times) >= 4:
                self._stuck_escalation = 3  # Spiral escape
            elif len(self._stuck_trigger_times) >= 3:
                self._stuck_escalation = 2  # Try all 8 directions
            elif len(self._stuck_trigger_times) >= 2:
                self._stuck_escalation = 1  # Reverse + diagonal
            else:
                self._stuck_escalation = 0

            # --- ESCALATION LEVEL 0: Try concavity escape first, then A* pathfinder ---
            # Check if we're in a U-trap and need to go backwards.
            # Require ALL 4 cardinals blocked to avoid false positives from
            # ghost walls we just injected above.
            if cur_player_pos and hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
                from bt_combat import _detect_concavity
                import bt_combat
                # Pass storm center to concavity detection for smarter escape direction
                if getattr(self, '_gas_active', False) or getattr(self, '_in_storm', False):
                    bt_combat._active_storm_center = getattr(self, '_storm_center', None)
                else:
                    bt_combat._active_storm_center = None
                _onnx_walls = getattr(self, 'last_walls_data', None) or []
                escape_dir, blocked_n = _detect_concavity(cur_player_pos, walls=_onnx_walls)
                if escape_dir and blocked_n >= 4:
                    self.fix_movement_keys['fixed'] = escape_dir.lower()
                    self.fix_movement_keys['toggled'] = True
                    self.fix_movement_keys['started_at'] = current_time
                    self.fix_movement_keys['duration'] = 1.5  # Long enough to actually leave U-trap
                    self.last_decision_reason = f"ANTI-STUCK TRAP: escape {escape_dir} ({blocked_n}/4 blocked)"
                    print(f"[STUCK] U-trap escape: {escape_dir} ({blocked_n}/4 blocked)")
                    # Reset BT patrol state so PUSH FORWARD picks new direction after escape
                    if hasattr(self, '_bt_combat') and self._bt_combat is not None:
                        _bb = self._bt_combat.blackboard
                        _bb.set("_patrol_tick", 0)
                        _bb.set("_patrol_target", None)
                    return escape_dir.lower()

            # Try A* pathfinder to find way around walls
            # The pathfinder already has stuck-position penalties applied above.
            if self._stuck_escalation <= 1 and cur_player_pos:
                # Get a reasonable target: storm center when in gas, else enemy or screen center
                target = None
                # PRIORITY: When in storm/gas, always flee toward safe zone!
                if getattr(self, '_gas_active', False) or getattr(self, '_in_storm', False):
                    target = getattr(self, '_storm_center', None)
                    if target:
                        print(f"[STUCK] Using storm center as escape target: {target}")
                if target is None:
                    try:
                        if hasattr(self, '_last_known_enemy_pos') and self._last_known_enemy_pos:
                            target = self._last_known_enemy_pos
                    except Exception:
                        pass
                if target is None:
                    target = (960, 540)  # Map center
                pf_move = self._get_pathfinder_movement(cur_player_pos, target)
                if pf_move:
                    self.fix_movement_keys['fixed'] = pf_move.lower()
                    self.fix_movement_keys['toggled'] = True
                    self.fix_movement_keys['started_at'] = current_time
                    self.fix_movement_keys['duration'] = 0.4
                    self.last_decision_reason = f"ANTI-STUCK A*: pathfind around wall -> {pf_move}"
                    print(f"[STUCK] A* pathfinder found escape route: {pf_move}")
                    return pf_move.lower()

            # --- ESCALATION LEVEL 3: Spiral escape pattern ---
            if self._stuck_escalation >= 3:
                spiral_dirs = ['W', 'WD', 'D', 'DS', 'S', 'SA', 'A', 'AW']
                self._spiral_index = (self._spiral_index + 1) % len(spiral_dirs)
                escaped = spiral_dirs[self._spiral_index]
                self.fix_movement_keys['fixed'] = escaped.lower()
                self.fix_movement_keys['toggled'] = True
                self.fix_movement_keys['started_at'] = current_time
                self.fix_movement_keys['duration'] = 0.3  # Quick steps
                self.last_decision_reason = f"ANTI-STUCK L3: spiral -> {escaped}"
                print(f"[STUCK] Escalation L3 - spiral step {self._spiral_index}: {escaped}")
                return escaped.lower()

            # --- ESCALATION LEVEL 2: Try all 8 directions, pick first unblocked ---
            elif self._stuck_escalation >= 2:
                player_pos = None
                if hasattr(self, 'last_walls_data'):
                    walls = self.last_walls_data
                else:
                    walls = []
                # Try to get player position from keys_hold context
                all_dirs = list(self.ALL_DIRECTIONS)
                random.shuffle(all_dirs)
                # Just pick the first direction that differs from current
                for d in all_dirs:
                    if d.lower() != movement:
                        self.fix_movement_keys['fixed'] = d.lower()
                        self.fix_movement_keys['toggled'] = True
                        self.fix_movement_keys['started_at'] = current_time
                        self.last_decision_reason = f"ANTI-STUCK L2: try {d}"
                        print(f"[STUCK] Escalation L2 - trying: {d}")
                        return d.lower()

            # --- ESCALATION LEVEL 0-1: Original reverse logic (improved) ---
            reversed_movement = self.reverse_movement(movement)

            if reversed_movement == "s":
                reversed_movement = random.choice(['aw', 'dw'])
            elif reversed_movement == "w":
                reversed_movement = random.choice(['as', 'ds'])

            self.fix_movement_keys['fixed'] = reversed_movement
            self.fix_movement_keys['toggled'] = True
            self.fix_movement_keys['started_at'] = current_time
            self.last_decision_reason = f"ANTI-STUCK L{self._stuck_escalation}: reverse -> {reversed_movement}"
            return reversed_movement

        return movement


class Play(Movement):

    # wall polygon cache (avoids recreating Shapely Polygons 100s of times/frame)
    _wall_poly_cache_key = None
    _wall_poly_cache = None

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
        self.wall_history_length = 5  # Shorter history reduces stale/ghost walls
        self.scene_data = []
        self.should_detect_walls = bot_config["gamemode"] in ["brawlball", "brawl_ball", "brawll ball", "knockout", "gemgrab", "heist", "hotzone", "bounty", "wipeout", "duels", "other"]
        self.minimum_movement_delay = bot_config["minimum_movement_delay"]
        self.no_detection_proceed_delay = time_config["no_detection_proceed"]
        self.gadget_pixels_minimum = bot_config["gadget_pixels_minimum"]
        self.hypercharge_pixels_minimum = bot_config["hypercharge_pixels_minimum"]
        self.super_pixels_minimum = bot_config["super_pixels_minimum"]
        self.wall_detection_confidence = float(bot_config["wall_detection_confidence"])
        self._wall_conf_base = max(0.15, min(0.85, self.wall_detection_confidence))
        self._wall_conf_active = self._wall_conf_base
        self.entity_detection_confidence = bot_config["entity_detection_confidence"]
        self.seconds_to_hold_attack_after_reaching_max = bot_config.get("seconds_to_hold_attack_after_reaching_max", 1.5)
        self.time_since_holding_attack = None  # None = not holding; time.time() = when hold started

        # --- Visual tracking overlay (transparent window over game) ---
        try:
            gc = load_toml_as_dict("cfg/general_config.toml")
            overlay_enabled = str(gc.get("visual_overlay_enabled", "no")).lower() in ("yes", "true", "1")
            if overlay_enabled:
                self.visual_overlay = VisualOverlay()
                print("[OVERLAY] Visual tracking overlay started.")
            else:
                self.visual_overlay = None
        except Exception as e:
            print(f"[OVERLAY] Could not start visual overlay: {e}")
            self.visual_overlay = None

        # --- A* Pathfinder + SpatialMemory (used by both BT and legacy) ---
        self._path_planner = PathPlanner(cell_size=40)
        try:
            from spatial_memory import SpatialMemory
            w = getattr(self.window_controller, 'width', 1920) or 1920
            h = getattr(self.window_controller, 'height', 1080) or 1080
            self._spatial_memory = SpatialMemory(w, h)
            print("[PATHFINDER] A* pathfinder + SpatialMemory initialized")
        except Exception as e:
            self._spatial_memory = None
            print(f"[PATHFINDER] SpatialMemory init failed: {e}")

        # --- Water / hazard tile detector (HSV color-based) ---
        try:
            from water_detector import WaterDetector
            self._water_detector = WaterDetector()
            self._water_bboxes: list = []       # Current water bounding boxes
            self._last_water_scan = 0.0         # Timestamp of last water scan
            self._water_scan_interval = 3.0     # Seconds between scans (water doesn't move)
            print("[WATER] Water tile detector initialized")
        except Exception as e:
            self._water_detector = None
            self._water_bboxes = []
            self._last_water_scan = 0.0
            self._water_scan_interval = 3.0
            print(f"[WATER] Could not initialize water detector: {e}")

        # --- Wall detection precision tuning ---
        self._wall_min_size_px = 18
        self._wall_max_aspect = 6.0
        _hr = getattr(self.window_controller, 'height_ratio', None)
        if _hr is None:
            _hr = 1.0
        self._wall_ui_top_margin = int(95 * _hr)
        self._wall_ui_bottom_margin = int(25 * _hr)
        self._wall_small_area_px2 = 1200

        # --- Behavior Tree combat system (optional, controlled by config) ---
        self._bt_combat = None
        try:
            gc = load_toml_as_dict("cfg/general_config.toml")
            # HP safety policy overrides
            self._hp_check_interval = float(gc.get("hp_check_interval_s", self._hp_check_interval) or self._hp_check_interval)
            self._hp_conf_low_threshold = float(gc.get("hp_conf_low_threshold", self._hp_conf_low_threshold) or self._hp_conf_low_threshold)
            self._hp_stale_timeout = float(gc.get("hp_stale_timeout_s", self._hp_stale_timeout) or self._hp_stale_timeout)
            self._hp_warning_enter = int(gc.get("hp_warning_enter_pct", self._hp_warning_enter) or self._hp_warning_enter)
            self._hp_warning_exit = int(gc.get("hp_warning_exit_pct", self._hp_warning_exit) or self._hp_warning_exit)
            self._hp_critical_enter = int(gc.get("hp_critical_enter_pct", self._hp_critical_enter) or self._hp_critical_enter)
            self._hp_critical_exit = int(gc.get("hp_critical_exit_pct", self._hp_critical_exit) or self._hp_critical_exit)

            # Safety clamps
            self._hp_check_interval = max(0.05, min(0.30, self._hp_check_interval))
            self._hp_conf_low_threshold = max(0.05, min(0.95, self._hp_conf_low_threshold))
            self._hp_stale_timeout = max(0.20, min(2.00, self._hp_stale_timeout))
            self._hp_warning_enter = max(5, min(95, self._hp_warning_enter))
            self._hp_warning_exit = max(self._hp_warning_enter + 1, min(100, self._hp_warning_exit))
            self._hp_critical_enter = max(1, min(self._hp_warning_enter - 1, self._hp_critical_enter))
            self._hp_critical_exit = max(self._hp_critical_enter + 1, min(self._hp_warning_exit, self._hp_critical_exit))

            ai_mode = str(gc.get("ai_mode", "rules")).lower()
            if ai_mode in ("bt", "hybrid", "behavior_tree"):
                from bt_combat import BTCombat
                self._bt_combat = BTCombat(self)
                print(f"[BT] Behavior Tree combat system enabled (mode={ai_mode})")
            else:
                print(f"[BT] Using legacy rules AI (ai_mode={ai_mode})")
        except Exception as e:
            print(f"[BT] Could not initialize Behavior Tree: {e}")
            self._bt_combat = None

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
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        """Check if this brawler uses hold-to-charge attacks (e.g. Hank, Angelo)."""
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        brawler_data = brawlers_info.get(brawler)
        if not brawler_data:
            return False
        return brawler_data.get('hold_attack', 0) > 0

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info: brawlers_info = load_brawlers_info()
        brawler_data = brawlers_info.get(brawler)
        if not brawler_data:
            return False
        if skill_type == "attack":
            return brawler_data.get('ignore_walls_for_attacks', False)
        elif skill_type == "super":
            return brawler_data.get('ignore_walls_for_supers', False)
        raise ValueError("skill_type must be either 'attack' or 'super'")


    @staticmethod
    def _get_wall_polygons(walls):
        """Get cached Shapely Polygons for the current wall set.
        Walls rarely change between frames, so caching saves 100s of
        Polygon constructions per frame (was the #1 CPU bottleneck)."""
        key = id(walls)  # Fast identity check (same list object -> same polygons)
        if Play._wall_poly_cache_key == key and Play._wall_poly_cache is not None:
            return Play._wall_poly_cache
        padding = 4  # Reduced from 8 to avoid false wall blocks at close range
        polys = []
        for wall in walls:
            x1, y1, x2, y2 = wall
            polys.append(Polygon([
                (x1 - padding, y1 - padding), (x2 + padding, y1 - padding),
                (x2 + padding, y2 + padding), (x1 - padding, y2 + padding)
            ]))
        Play._wall_poly_cache_key = key
        Play._wall_poly_cache = polys
        return polys

    @staticmethod
    def walls_are_in_line_of_sight(line_of_sight, walls):
        """Check if a line intersects any wall. Uses inflated wall hitboxes
        (padded by 15px each side) for safer collision margins.
        Wall polygons are cached per frame to avoid re-creating 30-50 Polygons
        on every one of the 5-20 LOS checks per frame."""
        for poly in Play._get_wall_polygons(walls):
            if line_of_sight.intersects(poly):
                return True
        return False

    def _get_pathfinder_movement(self, player_pos, target_pos):
        """Use A* pathfinder to navigate around walls toward a target.
        
        Returns a WASD string, or None if pathfinding is unavailable/fails.
        """
        if (not hasattr(self, '_path_planner') or self._path_planner is None
                or not hasattr(self, '_spatial_memory') or self._spatial_memory is None):
            return None
        try:
            move = self._path_planner.get_movement_toward(
                player_pos, target_pos, self._spatial_memory
            )
            if move:
                self.last_decision_reason = f"A* PATH: {move} (plan {self._path_planner.last_plan_ms:.0f}ms)"
                return move
        except Exception:
            pass
        return None

    def _get_pathfinder_movement_away(self, player_pos, threat_pos):
        """Use A* pathfinder to flee from a threat around walls.
        
        Returns a WASD string, or None if pathfinding is unavailable/fails.
        """
        if (not hasattr(self, '_path_planner') or self._path_planner is None
                or not hasattr(self, '_spatial_memory') or self._spatial_memory is None):
            return None
        try:
            move = self._path_planner.get_movement_away(
                player_pos, threat_pos, self._spatial_memory
            )
            if move:
                self.last_decision_reason = f"A* FLEE: {move}"
                return move
        except Exception:
            pass
        return None

    def no_enemy_movement(self, player_data, walls, playstyle="fighter", teammates=None):
        player_position = self.get_player_pos(player_data)
        style = PLAYSTYLE_CONFIG.get(playstyle, PLAYSTYLE_CONFIG["fighter"])
        now = time.time()

        # Track how long since we last saw an enemy
        last_enemy_seen = self.time_since_detections.get('enemy', 0)
        self._no_enemy_duration = now - last_enemy_seen

        # --- RECORD VISITED ZONE (for intelligent patrol) ---
        self._visited_zones.append((player_position[0], player_position[1], now))
        self._visited_zones = [(x, y, t) for x, y, t in self._visited_zones if now - t < 10.0]

        # --- STORM / GAS FLEE (HIGHEST priority - never chase enemies into gas!) ---
        if self._is_storm_flee_delay_over() and (self._storm_radius < 5000 or self._gas_active):
            dist_to_center = math.hypot(
                player_position[0] - self._storm_center[0],
                player_position[1] - self._storm_center[1])
            if self._in_storm or dist_to_center > self._storm_radius * 0.85:
                sdx = self._storm_center[0] - player_position[0]
                sdy = self._storm_center[1] - player_position[1]
                sh = 'D' if sdx > 0 else 'A'
                sv = 'S' if sdy > 0 else 'W'
                for m in [sv + sh, sv, sh]:
                    if m and not self.is_path_blocked(player_position, m, walls):
                        self.last_decision_reason = "GAS/STORM: flee to safety"
                        return m

        # === POST-BURST DEFENSIVE: no enemies -> retreat to teammates, don't hunt ===
        _post_burst_no_enemy = (
            self._last_burst_end_time > 0
            and (now - self._last_burst_end_time) < self._burst_defensive_duration
            and not self._burst_mode
        )
        if _post_burst_no_enemy and teammates:
            # Stay close to teammates - don't hunt or advance aggressively
            nearest_tm_pos = None
            nearest_tm_dist = float('inf')
            tm_cx_sum, tm_cy_sum = 0.0, 0.0
            for tm in teammates:
                tcx = (tm[0] + tm[2]) / 2
                tcy = (tm[1] + tm[3]) / 2
                tm_cx_sum += tcx
                tm_cy_sum += tcy
                td = math.hypot(tcx - player_position[0], tcy - player_position[1])
                if td < nearest_tm_dist:
                    nearest_tm_dist = td
                    nearest_tm_pos = (tcx, tcy)
            if nearest_tm_pos and nearest_tm_dist > 80:
                # Move toward team centroid
                tm_cx_avg = tm_cx_sum / len(teammates)
                tm_cy_avg = tm_cy_sum / len(teammates)
                dx = tm_cx_avg - player_position[0]
                dy = tm_cy_avg - player_position[1]
                h_key = 'D' if dx > 0 else 'A'
                v_key = 'S' if dy > 0 else 'W'
                for move in [v_key + h_key, v_key, h_key]:
                    if move and not self.is_path_blocked(player_position, move, walls):
                        remaining = self._burst_defensive_duration - (now - self._last_burst_end_time)
                        self.last_decision_reason = f"DEFENSIVE: regroup ({remaining:.1f}s, ammo={self._ammo})"
                        return move
            else:
                # Already near teammate - hold position (slight juke)
                remaining = self._burst_defensive_duration - (now - self._last_burst_end_time)
                self.last_decision_reason = f"DEFENSIVE: holding ({remaining:.1f}s, ammo={self._ammo})"
                juke = random.choice(['A', 'D', 'W', 'S'])
                if not self.is_path_blocked(player_position, juke, walls):
                    return juke
                return 'W'

        # --- TEAMMATE FOLLOWING (PRIORITY #2 - stay grouped before hunting!) ---
        if teammates:
            nearest_tm_pos = None
            nearest_tm_dist = float('inf')
            tm_centroid_x, tm_centroid_y = 0.0, 0.0
            for tm in teammates:
                tm_cx = (tm[0] + tm[2]) / 2
                tm_cy = (tm[1] + tm[3]) / 2
                tm_centroid_x += tm_cx
                tm_centroid_y += tm_cy
                td = math.hypot(tm_cx - player_position[0], tm_cy - player_position[1])
                if td < nearest_tm_dist:
                    nearest_tm_dist = td
                    nearest_tm_pos = (tm_cx, tm_cy)
            tm_centroid_x /= len(teammates)
            tm_centroid_y /= len(teammates)

            if nearest_tm_pos:
                # Tighter follow distances - stay grouped!
                if self.is_showdown or self.game_mode == 3:
                    follow_dist = 120  # Knockout: very tight grouping
                elif style.get("prefer_teammates", False):
                    follow_dist = 160  # Support: stick close
                else:
                    follow_dist = 200  # Everyone else: still stay near

                if nearest_tm_dist > follow_dist:
                    # Move toward team centroid (not just nearest) for better grouping
                    target_centroid = (tm_centroid_x, tm_centroid_y)
                    dx = tm_centroid_x - player_position[0]
                    dy = tm_centroid_y - player_position[1]
                    h_key = 'D' if dx > 0 else 'A'
                    v_key = 'S' if dy > 0 else 'W'
                    found_direct = False
                    for move in [v_key + h_key, v_key, h_key]:
                        if move and not self.is_path_blocked(player_position, move, walls):
                            self.last_decision_reason = f"FOLLOW: teammate {int(nearest_tm_dist)}px"
                            return move
                    # Direct path blocked -> use A* pathfinder
                    pf_move = self._get_pathfinder_movement(player_position, target_centroid)
                    if pf_move:
                        self.last_decision_reason = f"A* FOLLOW: teammate {int(nearest_tm_dist)}px"
                        return pf_move
                elif nearest_tm_dist > 80:
                    # Close to teammate - advance TOGETHER toward enemy side
                    advance_move = self._get_advance_movement(player_position, walls, style)
                    if advance_move:
                        self.last_decision_reason = f"ADVANCE TOGETHER: team grouped ({int(nearest_tm_dist)}px)"
                        return advance_move

        # --- HUNT LAST KNOWN ENEMY (only after confirming we're near teammates) ---
        # Skip hunting if: recently killed, OR enemy position is stale (>2s)
        recently_killed = (now - self._last_enemy_kill_time) < self._enemy_death_cooldown
        last_enemy = self._get_last_known_enemy_pos()
        # Check freshness: only hunt if we saw the enemy very recently (<2s)
        enemy_freshness = self._no_enemy_duration  # How long since we last saw ANY enemy
        hunt_is_stale = enemy_freshness > 2.0
        if last_enemy and not recently_killed and not hunt_is_stale:
            dx = last_enemy[0] - player_position[0]
            dy = last_enemy[1] - player_position[1]
            dist = math.hypot(dx, dy)
            if dist > 50:
                h_key = 'D' if dx > 0 else 'A'
                v_key = 'S' if dy > 0 else 'W'
                # Add slight strafe to avoid running in predictable straight line
                if random.random() < 0.2:
                    h_key = random.choice(['A', 'D'])
                for move in [v_key + h_key, v_key, h_key]:
                    if move and not self.is_path_blocked(player_position, move, walls):
                        self.last_decision_reason = f"HUNT: last seen {int(dist)}px ({enemy_freshness:.1f}s ago)"
                        return move
                # Direct path blocked -> A* pathfinder
                pf_move = self._get_pathfinder_movement(player_position, last_enemy)
                if pf_move:
                    self.last_decision_reason = f"A* HUNT: last seen {int(dist)}px"
                    return pf_move
        # If hunt was skipped, fall through to patrol/advance (no lingering HUNT display)

        # --- OBJECTIVE MOVEMENT ---
        if self.objective_pos is not None and not self.is_showdown:
            obj_pos = self.objective_pos
            ox, oy = obj_pos[0], obj_pos[1]
            obj_dx = ox - player_position[0]
            obj_dy = oy - player_position[1]
            obj_dist = math.hypot(obj_dx, obj_dy)
            if obj_dist > 80:
                h_key = 'D' if obj_dx > 0 else 'A'
                v_key = 'S' if obj_dy > 0 else 'W'
                for move in [v_key + h_key, v_key, h_key]:
                    if move and not self.is_path_blocked(player_position, move, walls):
                        self.last_decision_reason = f"OBJECTIVE: {int(obj_dist)}px to target"
                        return move
                # Direct path blocked -> A* pathfinder to objective
                pf_move = self._get_pathfinder_movement(player_position, (ox, oy))
                if pf_move:
                    self.last_decision_reason = f"A* OBJECTIVE: {int(obj_dist)}px"
                    return pf_move

        # =====================================================
        # === ACTIVE ENEMY SEARCH / PATROL SYSTEM ===
        # When no enemies visible for >2s, actively search the map
        # instead of standing still or wandering aimlessly.
        # =====================================================
        if self._no_enemy_duration > 1.0 and (not teammates or len(teammates) == 0):
            solo_move = self._get_solo_search_movement(player_position, walls)
            if solo_move:
                return solo_move

        if self._no_enemy_duration > 2.0:
            patrol_move = self._get_patrol_movement(player_position, walls, playstyle)
            if patrol_move:
                return patrol_move

        # --- LANE DRIFT (if assigned) ---
        if self._lane_center and not self.is_showdown:
            lx, ly = self._lane_center
            lane_dx = lx - player_position[0]
            lane_dy = ly - player_position[1]
            lane_dist = math.hypot(lane_dx, lane_dy)
            if lane_dist > 120:
                lh = 'D' if lane_dx > 30 else ('A' if lane_dx < -30 else '')
                lv = 'S' if lane_dy > 30 else ('W' if lane_dy < -30 else '')
                lane_move = lv + lh
                if lane_move and not self.is_path_blocked(player_position, lane_move, walls):
                    self.last_decision_reason = f"LANE: drifting to {self._assigned_lane}"
                    return lane_move

        # --- FALLBACK: advance toward enemy side ---
        return self._get_advance_movement(player_position, walls, style)

    def _get_solo_search_movement(self, player_pos, walls):
        """Aggressive full-map search when no teammates are nearby."""
        now = time.time()
        width = brawl_stars_width * self.window_controller.width_ratio
        height = brawl_stars_height * self.window_controller.height_ratio

        if self._spawn_side == 'left':
            ratios = [(0.58, 0.50), (0.70, 0.30), (0.70, 0.70), (0.85, 0.50), (0.55, 0.20), (0.55, 0.80)]
        elif self._spawn_side == 'right':
            ratios = [(0.42, 0.50), (0.30, 0.30), (0.30, 0.70), (0.15, 0.50), (0.45, 0.20), (0.45, 0.80)]
        elif self._spawn_side == 'bottom':
            ratios = [(0.50, 0.42), (0.30, 0.30), (0.70, 0.30), (0.50, 0.15), (0.20, 0.45), (0.80, 0.45)]
        elif self._spawn_side == 'top':
            ratios = [(0.50, 0.58), (0.30, 0.70), (0.70, 0.70), (0.50, 0.85), (0.20, 0.55), (0.80, 0.55)]
        else:
            ratios = [(0.50, 0.50), (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65)]

        targets = [(width * rx, height * ry) for rx, ry in ratios]
        idx = self._solo_search_target_idx % len(targets)
        target = targets[idx]

        dist = math.hypot(target[0] - player_pos[0], target[1] - player_pos[1])
        reached_target = dist < 95
        stale_target = (now - self._solo_search_last_switch) > 3.6

        recent = [(x, y) for x, y, t in self._visited_zones if now - t < 2.8]
        stuck_here = False
        if len(recent) >= 5:
            spread = max(
                max(p[0] for p in recent) - min(p[0] for p in recent),
                max(p[1] for p in recent) - min(p[1] for p in recent)
            )
            stuck_here = spread < 50

        if reached_target or stale_target or stuck_here:
            self._solo_search_target_idx = (self._solo_search_target_idx + 1) % len(targets)
            self._solo_search_last_switch = now
            idx = self._solo_search_target_idx
            target = targets[idx]

        dx = target[0] - player_pos[0]
        dy = target[1] - player_pos[1]
        h_key = 'D' if dx > 20 else ('A' if dx < -20 else '')
        v_key = 'S' if dy > 20 else ('W' if dy < -20 else '')

        for move in [v_key + h_key, v_key, h_key]:
            if move and not self.is_path_blocked(player_pos, move, walls):
                self.last_decision_reason = f"SEARCH SOLO: zone {idx + 1}/{len(targets)}"
                return move

        pf_move = self._get_pathfinder_movement(player_pos, target)
        if pf_move:
            self.last_decision_reason = f"A* SEARCH SOLO: zone {idx + 1}/{len(targets)}"
            return pf_move

        return None

    def _get_patrol_movement(self, player_pos, walls, playstyle):
        """Active enemy search: sweep-advance pattern toward enemy side."""
        now = time.time()

        # Determine primary advance direction based on game mode and spawn
        if self.game_mode == 3:  # Knockout
            advance_dir = 'W'  # Forward = up (toward enemy)
            sweep_dirs = ['A', 'D']
            adv_diagonals = ['WA', 'WD']
        else:
            advance_dir = 'D'  # Right side usually
            sweep_dirs = ['W', 'S']
            adv_diagonals = ['DW', 'DS']

        # Override advance direction based on spawn side
        if self._spawn_side == 'left':
            advance_dir = 'D'
            sweep_dirs = ['W', 'S']
            adv_diagonals = ['DW', 'DS']
        elif self._spawn_side == 'right':
            advance_dir = 'A'
            sweep_dirs = ['W', 'S']
            adv_diagonals = ['AW', 'AS']
        elif self._spawn_side == 'bottom':
            advance_dir = 'W'
            sweep_dirs = ['A', 'D']
            adv_diagonals = ['WA', 'WD']
        elif self._spawn_side == 'top':
            advance_dir = 'S'
            sweep_dirs = ['A', 'D']
            adv_diagonals = ['SA', 'SD']

        # Check if we've been in the same area too long (need to move elsewhere)
        recent_positions = [(x, y) for x, y, t in self._visited_zones if now - t < 3.0]
        staying_put = False
        if len(recent_positions) >= 5:
            avg_x = sum(p[0] for p in recent_positions) / len(recent_positions)
            avg_y = sum(p[1] for p in recent_positions) / len(recent_positions)
            spread = max(
                max(p[0] for p in recent_positions) - min(p[0] for p in recent_positions),
                max(p[1] for p in recent_positions) - min(p[1] for p in recent_positions))
            if spread < 40:
                staying_put = True  # We're stuck in a small area

        # Patrol phase state machine: ADVANCE -> SWEEP_LEFT -> ADVANCE -> SWEEP_RIGHT -> ...
        # Scale phase durations with urgency: the longer we haven't seen enemies, the faster we search
        base_duration = 1.2
        if staying_put:
            base_duration = 0.6  # Much faster when stuck
        elif self._no_enemy_duration > 8.0:
            base_duration = max(0.5, 1.2 - self._no_enemy_duration * 0.04)  # Ramp up urgency
        phase_duration = base_duration
        time_in_phase = now - self._last_patrol_change

        if time_in_phase > phase_duration:
            self._last_patrol_change = now
            if self._patrol_phase == 'advance':
                # Switch to sweep
                self._patrol_phase = 'sweep_left' if self._patrol_sweep_dir > 0 else 'sweep_right'
                self._patrol_sweep_dir *= -1  # Alternate sweep direction
            elif self._patrol_phase in ('sweep_left', 'sweep_right'):
                self._patrol_phase = 'advance'
            else:
                self._patrol_phase = 'advance'

        # --- Generate movement based on patrol phase ---
        if self._patrol_phase == 'advance':
            # Advance toward enemy side (diagonal for better coverage)
            move_options = adv_diagonals + [advance_dir]
            if staying_put:
                # Mix in a random direction to escape stuck area
                random.shuffle(move_options)
            for move in move_options:
                if not self.is_path_blocked(player_pos, move, walls):
                    self.last_decision_reason = f"SEARCH: advance ({move})"
                    # Fire at bushes while advancing (area denial / bush check)
                    if (self.last_bush_data and self._ammo >= 2
                            and random.random() < 0.20):
                        self.attack()
                        self._spend_ammo()
                        self.last_decision_reason += " +BUSH"
                    return move

        elif self._patrol_phase in ('sweep_left', 'sweep_right'):
            # Sweep perpendicular to advance direction (scout for enemies)
            sweep_idx = 0 if self._patrol_phase == 'sweep_left' else 1
            primary_sweep = sweep_dirs[sweep_idx % len(sweep_dirs)]
            # Diagonal sweep: sweep + slight advance (don't lose ground)
            diag_sweep = primary_sweep + advance_dir
            for move in [diag_sweep, primary_sweep]:
                if move and not self.is_path_blocked(player_pos, move, walls):
                    self.last_decision_reason = f"SEARCH: sweep ({move})"
                    return move

        # Fallback: try center or forward advance
        center_x = brawl_stars_width * self.window_controller.width_ratio / 2
        center_y = brawl_stars_height * self.window_controller.height_ratio / 2
        cdx = center_x - player_pos[0]
        cdy = center_y - player_pos[1]
        if math.hypot(cdx, cdy) > 100:
            ch = 'D' if cdx > 30 else ('A' if cdx < -30 else '')
            cv = 'S' if cdy > 30 else ('W' if cdy < -30 else '')
            cmove = cv + ch
            if cmove and not self.is_path_blocked(player_pos, cmove, walls):
                self.last_decision_reason = f"SEARCH: toward center"
                return cmove

        return None  # Let caller handle fallback

    def _get_advance_movement(self, player_pos, walls, style):
        """Fallback movement: advance toward enemy side with wall avoidance."""
        center_x = brawl_stars_width * self.window_controller.width_ratio / 2
        center_y = brawl_stars_height * self.window_controller.height_ratio / 2
        dx = center_x - player_pos[0]
        dy = center_y - player_pos[1]
        center_h = 'D' if dx > 50 else ('A' if dx < -50 else '')
        center_v = 'S' if dy > 50 else ('W' if dy < -50 else '')
        center_move = center_v + center_h

        if self._spawn_side == 'left':
            forward_moves = ['D', 'DW', 'DS']
        elif self._spawn_side == 'right':
            forward_moves = ['A', 'AW', 'AS']
        elif self._spawn_side == 'bottom':
            forward_moves = ['W', 'WA', 'WD']
        elif self._spawn_side == 'top':
            forward_moves = ['S', 'SA', 'SD']
        elif self.game_mode == 3:
            forward_moves = ['W', 'WA', 'WD']
        else:
            forward_moves = ['D', 'DW', 'DS']

        if style["aggressive_no_enemy"]:
            priority = list(forward_moves)
            if center_move and center_move not in priority:
                priority.append(center_move)
        else:
            priority = []
            if center_move:
                priority.append(center_move)
            priority.extend(forward_moves)

        remaining = [m for m in self.ALL_DIRECTIONS if m not in priority]
        if self.fix_movement_keys.get('toggled'):
            random.shuffle(remaining)
        priority.extend(remaining)

        for move in priority:
            if not self.is_path_blocked(player_pos, move, walls):
                if self._is_path_through_dangerous_bush(player_pos, move):
                    continue
                self.last_decision_reason = f"ADVANCE: {move}"
                return move
        for move in priority:
            if not self.is_path_blocked(player_pos, move, walls):
                self.last_decision_reason = "NAVIGATE: risky path"
                return move
        return center_move if center_move else 'W'

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        # Filter out expired ghost walls before line-of-sight check
        active_walls = walls
        if self._ghost_wall is not None and time.time() > self._ghost_wall_expire:
            # Ghost wall expired - exclude it from wall list
            ghost_tuple = tuple(self._ghost_wall)
            active_walls = [w for w in walls if tuple(w) != ghost_tuple]
            self._ghost_wall = None
        if self.walls_are_in_line_of_sight(LineString([player_pos, enemy_pos]), active_walls):
            return False
        return True

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        player_pos_x, player_pos_y = player_coords
        closest_hittable_distance = float('inf')
        closest_unhittable_distance = float('inf')
        closest_hittable = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable((player_pos_x, player_pos_y), enemy_pos, walls, skill_type):
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
            # Fallback to unhittable - bot should NOT attack, only reposition
            print(f"[TARGET] No hittable enemy! Using unhittable at {int(closest_unhittable_distance)}px for positioning only")
            return closest_unhittable

        return None, None

    def _filter_ui_detections(self, detections, frame_width=1280, frame_height=720, frame=None):
        """Filter out false positive detections from UI regions and invalid enemy shapes.
        
        Filters:
        - UI corners (player/enemy icons) - but allow enemies further in
        - Too small bboxes (likely map objects)
        - Extreme aspect ratios (barrels, boxes, etc.)
        - Bottom UI area (action buttons)
        - GRAY OBJECTS (skulls, gravestones) - real brawlers are colorful!
        """
        if not detections:
            return detections
        
        # Size thresholds (scaled for 1600x900 resolution)
        scale = frame_width / 1280
        min_enemy_size = int(35 * scale)   # Enemies > 35px (was 40)
        max_enemy_size = int(220 * scale)  # Enemies < 220px (was 200)
        min_aspect = 0.35   # Not too narrow (was 0.4)
        max_aspect = 2.8    # Not too wide (was 2.5)
        
        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            
            # --- Full-width HUD portrait strip (top ~60px) ---
            # Brawler portraits / score / icons span the entire top bar.
            # Any small detection there is UI, not a real enemy.
            in_hud_strip = (cy < 65 * scale and max(w, h) < 120 * scale)

            # Corner-based exclusion for slightly larger UI elements
            in_top_left = cx < 350 * scale and cy < 180 * scale
            in_top_right = cx > (frame_width - 180 * scale) and cy < 180 * scale
            in_bottom_ui = cy > (frame_height - 120 * scale)
            
            if in_hud_strip or in_top_left or in_top_right or in_bottom_ui:
                continue
            
            # Size filter: enemies have expected size range
            if w < min_enemy_size or h < min_enemy_size:
                continue  # Too small - likely map object
            if w > max_enemy_size or h > max_enemy_size:
                continue  # Too large - likely false positive
            
            # Aspect ratio filter: enemies are roughly human-shaped
            aspect = w / max(h, 1)
            if aspect < min_aspect or aspect > max_aspect:
                continue  # Extreme ratio - likely barrel/box
            
            # RED CIRCLE vs LAVA FILTER: Distinguish enemy indicator from danger zones
            # - Small red circle under brawler = ENEMY (keep)
            # - Large red area extending beyond brawler = LAVA/DANGER (reject)
            if frame is not None:
                try:
                    import cv2
                    import numpy as np
                    
                    # Check area UNDER the brawler for red indicator
                    circle_y1 = max(0, int(y2 - h * 0.3))  # Lower 30% of bbox
                    circle_y2 = min(frame.shape[0], int(y2 + h * 0.2))  # Extend slightly below
                    circle_x1 = max(0, int(cx - w * 0.7))  # Check wider area
                    circle_x2 = min(frame.shape[1], int(cx + w * 0.7))
                    
                    circle_area = frame[circle_y1:circle_y2, circle_x1:circle_x2]
                    
                    if circle_area.size > 0:
                        # Convert to HSV and look for RED color
                        hsv = cv2.cvtColor(circle_area, cv2.COLOR_RGB2HSV)
                        
                        # Red in HSV: H = 0-15 OR H = 165-180, S > 70, V > 50
                        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
                        
                        red_mask_low = (h_ch <= 15) & (s_ch > 70) & (v_ch > 50)
                        red_mask_high = (h_ch >= 165) & (s_ch > 70) & (v_ch > 50)
                        red_mask = red_mask_low | red_mask_high
                        
                        red_pixels = np.sum(red_mask)
                        total_pixels = red_mask.size
                        red_ratio = red_pixels / max(total_pixels, 1)
                        
                        # No red at all = not an enemy
                        if red_ratio < 0.02:
                            continue
                        
                        # TOO MUCH red = LAVA / danger zone, not an enemy circle
                        # Enemy circles are small (~10-30% of check area)
                        # Lava zones fill most of the area (>50%)
                        if red_ratio > 0.50:
                            continue  # This is lava, not an enemy
                        
                        # Check if red is LOCALIZED (circle) vs SPREAD OUT (lava)
                        # Find contours of red area
                        red_binary = red_mask.astype(np.uint8) * 255
                        contours, _ = cv2.findContours(red_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            # Get largest red contour
                            largest = max(contours, key=cv2.contourArea)
                            contour_area = cv2.contourArea(largest)
                            
                            # Enemy circle: compact, roughly circular
                            # Lava: irregular, large, extends in multiple directions
                            perimeter = cv2.arcLength(largest, True)
                            if perimeter > 0:
                                circularity = 4 * np.pi * contour_area / (perimeter ** 2)
                                # Circle has circularity ~1.0, irregular shapes < 0.3
                                # Enemy indicator circles: circularity > 0.3
                                # Lava patches: circularity < 0.2 (irregular)
                                if circularity < 0.15 and contour_area > 500:
                                    continue  # Irregular large red = lava
                                    
                except Exception:
                    pass  # If check fails, keep the detection
            
            filtered.append(det)
        
        return filtered

    def get_main_data(self, frame):
        # === FRAME SKIPPING FOR IPS OPTIMIZATION ===
        # Process every 2nd frame for higher GPU utilization
        self._frame_skip_counter += 1
        
        # Always run ONNX inference for consistent GPU usage (no frame skipping)
        # This keeps GPU at steady utilization instead of fluctuating
        data = self.Detect_main_info.detect_objects(frame, conf_tresh=self.entity_detection_confidence)
        
        # --- HUD portrait strip: filter ALL classes in the top ~7% of frame ---
        # The ONNX model sometimes classifies HUD brawler portraits as
        # enemies or teammates.  Remove any small detection whose centre_y
        # is in the top HUD bar, regardless of class.
        _fh, _fw = frame.shape[:2]
        _hud_y_max = int(_fh * 0.07)          # ~63px at 900h
        _hud_max_bbox = int(_fw * 0.08)       # ~128px at 1600w — real chars are bigger
        for _cls_key in ('enemy', 'teammate', 'player'):
            if _cls_key in data and data[_cls_key]:
                data[_cls_key] = [
                    d for d in data[_cls_key]
                    if not (
                        (d[1] + d[3]) / 2 < _hud_y_max           # centre in HUD strip
                        and max(d[2] - d[0], d[3] - d[1]) < _hud_max_bbox  # small = icon
                    )
                ]

        # Filter out UI false positives, map objects, and gray decorations (skulls)
        if 'enemy' in data and data['enemy']:
            data['enemy'] = self._filter_ui_detections(data['enemy'], _fw, _fh, frame)

        # --- Teammate-enemy overlap dedup ---
        # When teammates are near the top edge of the play area the model
        # sometimes outputs BOTH an "enemy" and a "teammate" bbox for the
        # same character.  Remove the "enemy" duplicate if it overlaps a
        # teammate by ≥40% IoU.
        if data.get('enemy') and data.get('teammate'):
            tm_boxes = data['teammate']
            cleaned_enemies = []
            for e in data['enemy']:
                ex1, ey1, ex2, ey2 = e[:4]
                e_area = max(1, (ex2 - ex1) * (ey2 - ey1))
                is_dup = False
                for t in tm_boxes:
                    tx1, ty1, tx2, ty2 = t[:4]
                    ix1 = max(ex1, tx1); iy1 = max(ey1, ty1)
                    ix2 = min(ex2, tx2); iy2 = min(ey2, ty2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    t_area = max(1, (tx2 - tx1) * (ty2 - ty1))
                    iou = inter / (e_area + t_area - inter) if (e_area + t_area - inter) > 0 else 0
                    if iou > 0.30:
                        is_dup = True
                        break
                if not is_dup:
                    cleaned_enemies.append(e)
            data['enemy'] = cleaned_enemies
        
        self._cached_detection_data = data
        self._frame_skip_counter = 0
        
        return data

    def _direction_enters_gas(self, move_direction):
        """Return True if the given move direction leads into a gas-heavy region."""
        if not self._is_storm_flee_delay_over():
            return False
        if not self._gas_active or not self._gas_density_map:
            return False
        # Map direction to grid region (player is roughly at center = 1,1)
        dx, dy = 0, 0
        if 'w' in move_direction.lower():
            dy -= 1
        if 's' in move_direction.lower():
            dy += 1
        if 'a' in move_direction.lower():
            dx -= 1
        if 'd' in move_direction.lower():
            dx += 1
        target_gx = 1 + dx  # 0, 1, or 2
        target_gy = 1 + dy  # 0, 1, or 2
        target_gx = max(0, min(2, target_gx))
        target_gy = max(0, min(2, target_gy))
        gas = self._gas_density_map.get((target_gx, target_gy), 0)
        return gas > 0.20  # >20% gas pixels = dangerous

    def is_path_blocked(self, player_pos, move_direction, walls, distance=None):
        """Check if path is blocked at multiple distances ahead, or leads into poison gas."""
        # Gas avoidance: treat gas-heavy directions as blocked
        if self._direction_enters_gas(move_direction):
            return True

        if distance is None:
            distance = self.TILE_SIZE * self.window_controller.scale_factor
        dx, dy = 0, 0
        if 'w' in move_direction.lower():
            dy -= 1
        if 's' in move_direction.lower():
            dy += 1
        if 'a' in move_direction.lower():
            dx -= 1
        if 'd' in move_direction.lower():
            dx += 1
        # Normalize diagonal movement
        length = math.hypot(dx, dy)
        if length == 0:
            return False
        dx /= length
        dy /= length
        # Check at multiple distances: 25%, 50%, 75%, 100% of full distance
        for ratio in [0.25, 0.5, 0.75, 1.0]:
            check_dist = distance * ratio
            new_pos = (player_pos[0] + dx * check_dist, player_pos[1] + dy * check_dist)
            path_line = LineString([player_pos, new_pos])
            if self.walls_are_in_line_of_sight(path_line, walls):
                return True

        # Also check spatial memory grid for water/wall cells that may not be in bbox list
        sm = getattr(self, '_spatial_memory', None)
        if sm is not None:
            for ratio in [0.4, 0.8]:
                check_dist = distance * ratio
                cx = player_pos[0] + dx * check_dist
                cy = player_pos[1] + dy * check_dist
                if not sm.is_walkable(cx, cy):
                    return True

        return False

    @staticmethod
    def validate_game_data(data):
        incomplete = False
        if "player" not in data.keys():
            incomplete = True  # This is required so track_no_detections can also keep track if enemy is missing

        if "enemy" not in data.keys() or data['enemy'] is None:
            data['enemy'] = []

        if 'wall' not in data.keys() or not data['wall']:
            data['wall'] = []

        return False if incomplete else data

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

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        return self.brawler_ranges[brawler]

    def loop(self, brawler, data, current_time):
        teammates = data.get('teammate', []) or []

        # behavior Tree mode: use BT for decisions instead of get_movement
        if self._bt_combat is not None:
            try:
                # Get the frame for subsystem updates (stored by main())
                bt_frame = getattr(self, '_current_frame', None)
                movement = self._bt_combat.tick(data, bt_frame, brawler)
                # --- Bridge BT blackboard -> play.py attributes ---
                # Without this, unstuck_movement_if_needed() has no position
                # data and ALL anti-stuck / U-trap detection is dead.
                _bb = self._bt_combat.blackboard
                _pp = _bb.get("player.pos", None)
                if _pp and _pp != (0, 0):
                    self._last_known_player_pos = _pp
                    self.time_since_player_last_found = time.time()
                _ec = _bb.get("enemies_count", 0)
                self._has_enemy_target = (_ec > 0)
            except Exception as e:
                # Fallback to legacy if BT fails
                print(f"[BT] Tick failed, falling back to rules: {e}")
                movement = self.get_movement(
                    player_data=data['player'][0],
                    enemy_data=data['enemy'],
                    walls=data['wall'],
                    brawler=brawler,
                    teammates=teammates,
                )
        else:
            movement = self.get_movement(
                player_data=data['player'][0],
                enemy_data=data['enemy'],
                walls=data['wall'],
                brawler=brawler,
                teammates=teammates,
            )

        current_time = time.time()  # Use fresh timestamp for movement timing
        if current_time - self.time_since_movement > self.minimum_movement_delay:
            # Always run unstuck detection (including BT mode).
            # The position-based stuck check prevents idle disconnects when
            # BT navigation gets stuck against undetected walls.
            movement = self.unstuck_movement_if_needed(movement, current_time)
            self.do_movement(movement)
            self.time_since_movement = time.time()
        else:
            # Even when delay hasn't passed, keep pressing current keys (prevents key release gaps)
            if self.keys_hold:
                self.window_controller.keys_down(self.keys_hold)
        return movement

    def check_if_hypercharge_ready(self, frame):
        screenshot = frame.crop((int(1350 * self.window_controller.width_ratio), int(940 * self.window_controller.height_ratio), int(1450 * self.window_controller.width_ratio), int(1050 * self.window_controller.height_ratio)))
        purple_pixels = count_hsv_pixels(screenshot, (120, 100, 100), (179, 255, 255))
        if purple_pixels > self.hypercharge_pixels_minimum:
            return True
        return False

    def check_if_gadget_ready(self, frame):
        screenshot = frame.crop((int(1580 * self.window_controller.width_ratio), int(930 * self.window_controller.height_ratio), int(1700 * self.window_controller.width_ratio), int(1050 * self.window_controller.height_ratio)))
        green_pixels = count_hsv_pixels(screenshot, (40, 150, 100), (80, 255, 255))
        if green_pixels > self.gadget_pixels_minimum:
            return True
        return False

    def check_if_super_ready(self, frame):
        screenshot = frame.crop((int(1460 * self.window_controller.width_ratio), int(830 * self.window_controller.height_ratio), int(1560 * self.window_controller.width_ratio), int(930 * self.window_controller.height_ratio)))
        yellow_pixels = count_hsv_pixels(screenshot, (15, 120, 150), (35, 255, 255))
        if yellow_pixels > self.super_pixels_minimum:
            return True
        return False

    def get_tile_data(self, frame):
        tile_data = self.Detect_tile_detector.detect_objects(frame, conf_tresh=self._wall_conf_active)
        return tile_data

    def _filter_wall_boxes(self, boxes):
        """Filter noisy wall detections by geometry and HUD exclusion."""
        if not boxes:
            return []

        filtered = []
        frame_h = int(getattr(self.window_controller, 'height', 1080) or 1080)
        y_top_limit = self._wall_ui_top_margin
        y_bottom_limit = frame_h - self._wall_ui_bottom_margin

        for box in boxes:
            if len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            w = x2 - x1
            h = y2 - y1
            if w < self._wall_min_size_px or h < self._wall_min_size_px:
                continue
            if y2 <= y_top_limit or y1 >= y_bottom_limit:
                continue
            shorter = max(1.0, float(min(w, h)))
            aspect = float(max(w, h)) / shorter
            if aspect > self._wall_max_aspect:
                continue
            filtered.append([x1, y1, x2, y2])

        return filtered

    def _filter_walls_overlapping_entities(self, walls, data):
        """Remove wall boxes whose centers are inside dynamic entity boxes.

        This suppresses common tile-model false positives drawn over players/enemies.
        """
        if not walls or not data:
            return walls

        entities = []
        for key in ('player', 'enemy', 'teammate'):
            boxes = data.get(key, []) or []
            for box in boxes:
                if len(box) >= 4:
                    entities.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        if not entities:
            return walls

        margin = 12
        filtered = []
        for wall in walls:
            wx = (wall[0] + wall[2]) * 0.5
            wy = (wall[1] + wall[3]) * 0.5
            overlaps_entity_center = False
            for ex1, ey1, ex2, ey2 in entities:
                if (ex1 - margin) <= wx <= (ex2 + margin) and (ey1 - margin) <= wy <= (ey2 + margin):
                    overlaps_entity_center = True
                    break
            if not overlaps_entity_center:
                filtered.append(wall)

        return filtered

    def _color_based_wall_detection(self, frame):
        """HSV color fallback: detect wall-like regions when ONNX model finds too few.

        Targets teal/cyan stone walls (H≈75-110) that the ONNX tile detector
        fails to recognise on certain Knockout maps.  Uses connected-component
        analysis with morphological cleanup to produce bounding boxes.

        Returns a list of [x1, y1, x2, y2] bounding boxes for suspected walls.
        """
        result_boxes: list = []
        try:
            if frame is None:
                return result_boxes

            # Convert to numpy RGB -> HSV
            img = np.asarray(frame)
            if img.ndim == 2:
                return result_boxes
            # PIL images are RGB; OpenCV cvtColor expects BGR for BGR2HSV
            # but we can go RGB->HSV directly
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h_img, w_img = img.shape[:2]

            # --- HUD exclusion zones (scale with resolution) ---
            y_top = int(h_img * 0.08)       # top HUD / icons
            y_bot = int(h_img * 0.92)       # bottom bar / controls
            x_right = int(w_img * 0.67)     # right-side stats overlay

            # --- Wall colour masks ---
            # Teal/cyan stone: H=75-110, S>=50, V>=50
            mask_teal = cv2.inRange(hsv, (75, 50, 50), (110, 255, 220))
            # Slightly darker teal accents: H=70-85
            mask_dark_teal = cv2.inRange(hsv, (70, 60, 40), (85, 255, 180))
            wall_mask = cv2.bitwise_or(mask_teal, mask_dark_teal)

            # --- Exclude orange bush regions (H=5-25, high S&V) ---
            bush_mask = cv2.inRange(hsv, (5, 90, 90), (25, 255, 255))
            bush_mask = cv2.dilate(bush_mask,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
            wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(bush_mask))

            # --- Exclude dark blue ground (H=110-125, low V) ---
            ground_mask = cv2.inRange(hsv, (108, 30, 20), (130, 200, 100))
            ground_mask = cv2.dilate(ground_mask,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(ground_mask))

            # Zero out HUD / right overlay / edges
            wall_mask[:y_top, :] = 0
            wall_mask[y_bot:, :] = 0
            wall_mask[:, x_right:] = 0
            edge = max(10, int(w_img * 0.02))
            wall_mask[:, :edge] = 0

            # --- Morphological cleanup ---
            kern5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kern5, iterations=2)
            wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_OPEN, kern5, iterations=1)

            # --- Connected components ---
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                wall_mask, connectivity=8)

            MIN_AREA = 800
            MAX_AREA = 80000
            MAX_ASPECT = 7.0
            # Target wall-tile size for splitting large regions
            TILE_SIZE = int(60 * (w_img / 1600))  # ~60px at 1600w

            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area < MIN_AREA or area > MAX_AREA:
                    continue
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                bw = stats[i, cv2.CC_STAT_WIDTH]
                bh = stats[i, cv2.CC_STAT_HEIGHT]
                if y + bh <= y_top or y >= y_bot:
                    continue
                shorter = max(1, min(bw, bh))
                if max(bw, bh) / shorter > MAX_ASPECT:
                    continue

                # Split large regions into tile-sized sub-boxes for finer
                # pathfinding grid coverage.
                if bw > TILE_SIZE * 2 or bh > TILE_SIZE * 2:
                    for ty in range(y, y + bh, TILE_SIZE):
                        for tx in range(x, x + bw, TILE_SIZE):
                            tx2 = min(tx + TILE_SIZE, x + bw)
                            ty2 = min(ty + TILE_SIZE, y + bh)
                            if (tx2 - tx) >= 20 and (ty2 - ty) >= 20:
                                result_boxes.append([tx, ty, tx2, ty2])
                else:
                    result_boxes.append([x, y, x + bw, y + bh])

            # --- Player exclusion zone: remove wall boxes overlapping the
            # player position so we don't wall-off our own spawn area. ---
            player_pos = getattr(self, '_last_known_player_pos', None)
            if player_pos and result_boxes:
                EXCL_RADIUS = 150  # pixels around player centre
                px, py = player_pos
                result_boxes = [
                    b for b in result_boxes
                    if not (b[0] - EXCL_RADIUS <= px <= b[2] + EXCL_RADIUS
                            and b[1] - EXCL_RADIUS <= py <= b[3] + EXCL_RADIUS)
                ]

            # Cap at 80 boxes
            if len(result_boxes) > 80:
                result_boxes.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                result_boxes = result_boxes[:80]

            # Inject into spatial memory as ghost walls
            if result_boxes and hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
                injected = 0
                for box in result_boxes:
                    cx = (box[0] + box[2]) / 2
                    cy = (box[1] + box[3]) / 2
                    # Skip ghost walls within 120px of player
                    if player_pos:
                        if math.hypot(cx - player_pos[0], cy - player_pos[1]) < 120:
                            continue
                    r, c = self._spatial_memory.pixel_to_grid(cx, cy)
                    if 0 <= r < self._spatial_memory.rows and 0 <= c < self._spatial_memory.cols:
                        cell = self._spatial_memory.grid[r, c]
                        if cell in (0, 1):  # UNKNOWN or EMPTY
                            self._spatial_memory.inject_ghost_wall(cx, cy, duration=3.0)
                            injected += 1

        except Exception as e:
            print(f"[COLOR-WALLS] Error: {e}")

        return result_boxes

    def process_tile_data(self, tile_data):
        walls = []
        bushes = []
        class_counts = {}
        for class_name, boxes in tile_data.items():
            class_counts[class_name] = len(boxes)
            if 'bush' in class_name.lower():
                bushes.extend(boxes)
            else:
                walls.extend(boxes)
        raw_wall_count = len(walls)
        walls = self._filter_wall_boxes(walls)
        kept_wall_count = len(walls)

        # Debug logging (every 10th call)
        self._wall_debug_ctr = getattr(self, '_wall_debug_ctr', 0) + 1
        if self._wall_debug_ctr % 10 == 1:
            cls_str = ' '.join(f'{k}={v}' for k, v in class_counts.items())
            print(f"[WALLS] raw={raw_wall_count} kept={kept_wall_count} "
                  f"conf={self._wall_conf_active:.2f} classes: {cls_str}")

        # Adaptive confidence: adjust based on detection quality.
        # Lower bound must respect the user-configured base (0.28), NOT a hardcoded 0.35.
        conf_floor = max(0.15, self._wall_conf_base - 0.10)
        if raw_wall_count >= 12 and kept_wall_count <= max(2, int(raw_wall_count * 0.25)):
            self._wall_conf_active = min(0.80, self._wall_conf_active + 0.03)
        elif raw_wall_count <= 4 and kept_wall_count <= 2:
            self._wall_conf_active = max(conf_floor, self._wall_conf_active - 0.02)
        else:
            # Gentle return toward user-configured baseline
            if self._wall_conf_active > self._wall_conf_base:
                self._wall_conf_active = max(self._wall_conf_base, self._wall_conf_active - 0.01)
            elif self._wall_conf_active < self._wall_conf_base:
                self._wall_conf_active = min(self._wall_conf_base, self._wall_conf_active + 0.01)

        self.last_bush_data = bushes  # Store for bush-checking & danger pathing

        # HSV color fallback: when ONNX finds very few walls, use targeted
        # teal/cyan colour detection to find stone walls on maps the model
        # wasn't trained on.  Throttled to once per 5 frames (~0.3s) to
        # limit CPU overhead.
        self._color_wall_tick = getattr(self, '_color_wall_tick', 0) + 1
        if kept_wall_count < 4 and self._color_wall_tick % 5 == 0:
            _frame_for_color = getattr(self, '_last_frame_ref', None)
            if _frame_for_color is not None:
                color_walls = self._color_based_wall_detection(_frame_for_color)
                if color_walls:
                    walls = walls + color_walls
                    if self._wall_debug_ctr % 10 == 1:
                        print(f"[COLOR-WALLS] +{len(color_walls)} color walls "
                              f"(total {len(walls)})")

        # Add walls to history
        self.wall_history.append(walls)
        if len(self.wall_history) > self.wall_history_length:
            self.wall_history.pop(0)
        # Combine walls from history
        combined_walls = self.combine_walls_from_history()

        # Feed SpatialMemory grid with detected walls/bushes (for A* pathfinding)
        if hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
            try:
                player_pos = getattr(self, '_last_known_player_pos', (960, 540))
                destroyed = getattr(self, '_destroyed_wall_zones', None)
                gas_active = getattr(self, '_gas_active', False)
                storm_center = getattr(self, '_storm_center', None)
                storm_radius = getattr(self, '_storm_radius', None)
                self._spatial_memory.update(
                    combined_walls, bushes, player_pos,
                    destroyed_zones=destroyed,
                    gas_active=gas_active,
                    storm_center=storm_center,
                    storm_radius=storm_radius,
                )
            except Exception as e:
                pass  # Don't crash the bot if grid update fails

        return combined_walls

    @staticmethod
    def _wall_iou(a, b):
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (aa + ab - inter) if (aa + ab - inter) > 0 else 0.0

    def combine_walls_from_history(self):
        """Merge wall detections from recent frames using IoU-based clustering.

        YOLO outputs fluctuate by several pixels each frame, so exact-match
        deduplication fails. Instead we:
          1. Collect all raw detections from the history ring buffer.
          2. Cluster overlapping boxes (IoU ≥ 0.3) together.
          3. Average the coordinates within each cluster for a stable box.
          4. Require a wall to be seen in ≥2 frames to suppress hallucinations.
        """
        if not self.wall_history:
            return []

        # Tag each detection with its frame index so we can count unique frames
        tagged = []  # (box, frame_idx)
        for fi, walls in enumerate(self.wall_history):
            for wall in walls:
                tagged.append((wall, fi))

        if not tagged:
            return []

        # Greedy IoU clustering
        used = [False] * len(tagged)
        clusters = []  # list of [(box, frame_idx), ...]

        for i in range(len(tagged)):
            if used[i]:
                continue
            cluster = [tagged[i]]
            used[i] = True
            ref = tagged[i][0]
            for j in range(i + 1, len(tagged)):
                if used[j]:
                    continue
                if self._wall_iou(ref, tagged[j][0]) >= 0.3:
                    cluster.append(tagged[j])
                    used[j] = True
            clusters.append(cluster)

        # Build stable walls: average coords, require temporal consistency + recency
        combined_walls = []
        min_frames = 1  # Accept walls seen in just 1 frame (was min(2,...) — too strict)
        newest_frame_idx = len(self.wall_history) - 1
        for cluster in clusters:
            frame_ids = set(fi for _, fi in cluster)
            unique_frames = len(frame_ids)
            n = len(cluster)
            ax1 = sum(b[0] for b, _ in cluster) / n
            ay1 = sum(b[1] for b, _ in cluster) / n
            ax2 = sum(b[2] for b, _ in cluster) / n
            ay2 = sum(b[3] for b, _ in cluster) / n
            avg_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))

            required_frames = min_frames
            if avg_area < self._wall_small_area_px2:
                required_frames = max(required_frames, min(2, len(self.wall_history)))

            if unique_frames < required_frames:
                continue
            # Require at least one hit in latest 3 scans to avoid stale ghosts (was 2 — too strict).
            if newest_frame_idx >= 2 and not any(fi >= (newest_frame_idx - 2) for fi in frame_ids):
                continue
            combined_walls.append([int(ax1), int(ay1), int(ax2), int(ay2)])

        # Filter out destroyed wall zones
        if self._destroyed_wall_zones:
            filtered = []
            for wall in combined_walls:
                wx = (wall[0] + wall[2]) / 2
                wy = (wall[1] + wall[3]) / 2
                destroyed = False
                for dz in self._destroyed_wall_zones:
                    if dz[0] <= wx <= dz[2] and dz[1] <= wy <= dz[3]:
                        destroyed = True
                        break
                if not destroyed:
                    filtered.append(wall)
            combined_walls = filtered

        return combined_walls

    def get_movement(self, player_data, enemy_data, walls, brawler, teammates=None):
        if teammates is None:
            teammates = []
        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            raise ValueError(f"Brawler '{brawler}' not found in brawlers info.")
        safe_range, attack_range, super_range = self.get_brawler_range(brawler)

        # --- LOAD PLAYSTYLE CONFIG ---
        playstyle = brawler_info.get('playstyle', 'fighter')
        style = PLAYSTYLE_CONFIG.get(playstyle, PLAYSTYLE_CONFIG["fighter"])
        range_mult = style["range_mult"]
        hp_retreat_threshold = style["hp_retreat"]
        approach_factor = style["approach_factor"]
        keep_max_range = style["keep_max_range"]
        rush_low_enemy = style["rush_low_enemy"]
        prefer_wall_cover = style["prefer_wall_cover"]

        # Update per-brawler reload speed (accurate ammo tracking)
        self._reload_speed = brawler_info.get('reload_speed', 1.4)

        # --- LOAD DETAILED COMBAT STATS ---
        self._my_health = brawler_info.get('health', 3200)
        self._my_attack_damage = brawler_info.get('attack_damage', 1200)
        self._my_projectile_count = brawler_info.get('projectile_count', 1)
        self._my_movement_speed = brawler_info.get('movement_speed', 720)
        self._my_super_damage = brawler_info.get('super_damage', 0)
        self._my_dps = self._my_attack_damage / max(0.1, self._reload_speed)
        # How many attacks to kill an average enemy (~3200 HP)
        self._shots_to_kill_default = max(1, math.ceil(3200 / max(1, self._my_attack_damage)))

        player_pos = self.get_player_pos(player_data)
        self._last_known_player_pos = player_pos  # For position-based stuck detection

        # === STORM / POISON GAS ZONE OVERRIDE (highest priority movement) ===
        if self._is_storm_flee_delay_over() and (self._storm_radius < 5000 or self._gas_active):
            dist_to_center = math.hypot(
                player_pos[0] - self._storm_center[0],
                player_pos[1] - self._storm_center[1])
            if self._in_storm or dist_to_center > self._storm_radius * 0.85:
                # In or near the gas/storm - flee toward safe zone center
                sdx = self._storm_center[0] - player_pos[0]
                sdy = self._storm_center[1] - player_pos[1]
                sh = 'D' if sdx > 0 else 'A'
                sv = 'S' if sdy > 0 else 'W'
                storm_move = sv + sh
                if not self.is_path_blocked(player_pos, storm_move, walls):
                    gas_type = 'STORM' if self.is_showdown else 'POISON GAS'
                    self.last_decision_reason = f"{gas_type} FLEE: {'IN GAS!' if self._in_storm else 'near edge'}"
                    return storm_move
                # Try components
                for m in [sv, sh]:
                    if m and not self.is_path_blocked(player_pos, m, walls):
                        self.last_decision_reason = f"GAS FLEE: {m}"
                        return m
            # Even if not in gas, avoid moving TOWARD gas-heavy regions
            elif self._gas_density_map:
                # Check if current movement direction leads into gas
                pass  # Gas avoidance in movement direction handled below

        # === REACTIVE PERPENDICULAR DODGE (highest combat priority) ===
        if time.time() < self._reactive_dodge_until:
            dodge_mv = self._reactive_dodge_keys.upper()
            if dodge_mv and not self.is_path_blocked(player_pos, dodge_mv, walls):
                self.last_decision_reason = f"DODGE: reactive ({dodge_mv})"
                return dodge_mv

        # === MATCH PHASE UPDATE ===
        self._update_match_phase(time.time())

        if not self.is_there_enemy(enemy_data):
            self._has_enemy_target = False
            # TARGETED bush-check: fire TOWARD the nearest bush within attack range
            # Only do this if we RECENTLY saw an enemy (within 2s), have ammo, and didn't just kill them
            now = time.time()
            time_since_bush_check = now - self._last_bush_check_time
            recently_saw_enemy = (now - self.time_since_detections.get('enemy', 0)) < 2.0
            recently_killed_enemy = (now - self._last_enemy_kill_time) < self._enemy_death_cooldown
            if (recently_saw_enemy
                    and not recently_killed_enemy
                    and time_since_bush_check > 2.0
                    and self.last_bush_data
                    and self._ammo >= 2):  # Don't waste last ammo on bush check
                bush_pos, bush_dist = self._find_nearest_bush_in_range(player_pos, attack_range)
                if bush_pos:
                    self._last_bush_check_time = now
                    # Aim toward the bush: briefly press movement toward bush, then fire
                    bush_dx = bush_pos[0] - player_pos[0]
                    bush_dy = bush_pos[1] - player_pos[1]
                    aim_h = 'D' if bush_dx > 0 else 'A'
                    aim_v = 'S' if bush_dy > 0 else 'W'
                    # Set movement toward bush before auto-aim fires in that direction
                    aim_keys = list((aim_v + aim_h).lower())
                    self.window_controller.keys_down(aim_keys)
                    time.sleep(0.02)
                    self.attack()
                    self._spend_ammo()
                    self.last_decision_reason = f"BUSH CHECK: {int(bush_dist)}px"
            return self.no_enemy_movement(player_data, walls, playstyle, teammates=teammates)

        # --- THREAT-SCORED TARGETING (picks best target, not just closest) ---
        self._has_enemy_target = True
        target_result = self.find_best_target(enemy_data, player_pos, walls, "attack", attack_range)
        enemy_coords, enemy_distance = target_result
        if enemy_coords is None:
            # Fallback to closest
            enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        if enemy_coords is None or enemy_distance is None:
            return self.no_enemy_movement(player_data, walls, playstyle, teammates=teammates)

        # Track enemy velocity for predictive aim
        self._update_enemy_velocity(enemy_coords)

        # === RANGED PRIORITY ATTACK (throwers/snipers fire BEFORE movement decisions) ===
        # Throwers lob projectiles for area denial - they must attack constantly.
        # This runs before any early-return movement paths to guarantee attacks happen.
        # NOTE: attack() uses auto-aim (taps "M" button) so aim direction keys are unnecessary.
        # We just fire directly - no aim-walk that nudges us toward the enemy.
        if keep_max_range and enemy_distance <= attack_range:
            now_atk = time.time()
            self._update_ammo(now_atk)
            area_deny_interval = style.get("attack_interval", 0.30)
            if (now_atk - self.last_attack_time >= area_deny_interval
                    and self._ammo >= 1):
                if self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack"):
                    # Auto-aim fires at nearest enemy - no aim-walk needed
                    self.attack()
                    self._spend_ammo()
                    self.last_decision_reason = f"AREA DENY: {int(enemy_distance)}px ammo={self._ammo}"

        # If the closest enemy is behind a wall, try to move perpendicular to flank
        enemy_behind_wall = not self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
        if enemy_behind_wall and enemy_distance <= attack_range:
            # We're close but blocked by a wall - flank by going perpendicular
            perp_x = -(enemy_coords[1] - player_pos[1])  # perpendicular direction
            perp_y = enemy_coords[0] - player_pos[0]
            flank_h = 'D' if perp_x > 0 else 'A'
            flank_v = 'S' if perp_y > 0 else 'W'
            flank_move = flank_v + flank_h
            if not self.is_path_blocked(player_pos, flank_move, walls):
                return flank_move

        # --- TEAMMATE TRACKING (positions for cooperative logic) ---
        nearest_teammate_dist = float('inf')
        nearest_teammate_pos = None
        self._teammate_positions = []  # Cache for target scoring
        for tm in teammates:
            tm_cx = (tm[0] + tm[2]) / 2
            tm_cy = (tm[1] + tm[3]) / 2
            self._teammate_positions.append((tm_cx, tm_cy))
            tm_dist = math.sqrt((tm_cx - player_pos[0])**2 + (tm_cy - player_pos[1])**2)
            if tm_dist < nearest_teammate_dist:
                nearest_teammate_dist = tm_dist
                nearest_teammate_pos = (tm_cx, tm_cy)

        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]

        # --- MULTI-ENEMY DANGER VECTOR ---
        # When 2+ enemies visible, blend retreat direction toward enemy group centroid
        # so we don't flee from one enemy into another
        if len(enemy_data) >= 2:
            all_ex = sum((e[0]+e[2])/2 for e in enemy_data) / len(enemy_data)
            all_ey = sum((e[1]+e[3])/2 for e in enemy_data) / len(enemy_data)
            danger_dx = all_ex - player_pos[0]
            danger_dy = all_ey - player_pos[1]
            # Only apply when retreat direction differs significantly from danger direction
            # This prevents running toward the group when retreating from the target
            direction_x = direction_x * 0.65 + danger_dx * 0.35
            direction_y = direction_y * 0.65 + danger_dy * 0.35

        # === TEAM COHESION: gravitate toward nearest teammate during combat ===
        # ALL playstyles get pulled toward teammates - strength varies by team_cohesion param
        # This prevents solo play and ensures coordinated attacks
        team_cohesion = style.get('team_cohesion', 0.25)
        if nearest_teammate_pos and nearest_teammate_dist > 100:
            tm_dx = nearest_teammate_pos[0] - player_pos[0]
            tm_dy = nearest_teammate_pos[1] - player_pos[1]
            # Scale cohesion by distance (stronger pull when further from teammate)
            # Max pull at 400px+, no pull below 100px
            dist_factor = min(1.0, (nearest_teammate_dist - 100) / 300)
            cohesion_strength = team_cohesion * dist_factor
            # Blend: pull direction toward teammate without overriding enemy direction
            direction_x = direction_x * (1.0 - cohesion_strength) + tm_dx * cohesion_strength
            direction_y = direction_y * (1.0 - cohesion_strength) + tm_dy * cohesion_strength

        # === DISENGAGE-TO-HEAL CHECK ===
        if self._should_disengage_for_heal(self.player_hp_percent, enemy_distance, safe_range):
            # Moderately wounded - retreat to heal instead of fighting
            rdx = -(enemy_coords[0] - player_pos[0])
            rdy = -(enemy_coords[1] - player_pos[1])
            dis_h = 'D' if rdx > 0 else 'A'
            dis_v = 'S' if rdy > 0 else 'W'
            # Add perpendicular strafe to retreat (harder to chase)
            if random.random() < 0.35:
                if abs(rdx) > abs(rdy):
                    dis_v = self.get_vstrafe_key()
                else:
                    dis_h = self._get_juke_direction()
            dis_move = dis_v + dis_h
            if not self.is_path_blocked(player_pos, dis_move, walls):
                self.last_decision_reason = f"DISENGAGE: heal (HP {self.player_hp_percent}%)"
                # Throwers/snipers: still fire while retreating (long range advantage)
                if keep_max_range and enemy_distance <= attack_range:
                    now_dis = time.time()
                    self._update_ammo(now_dis)
                    kite_int = style.get("kite_interval", 0.25)
                    if (now_dis - self.last_attack_time >= kite_int and self._ammo >= 1):
                        if self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack"):
                            self.attack()
                            self._spend_ammo()
                            self.last_decision_reason += " +FIRE"
                # ALL brawlers: fire while retreating if enemy is close
                elif enemy_distance <= safe_range and not keep_max_range:
                    now_dis = time.time()
                    self._update_ammo(now_dis)
                    if (now_dis - self.last_attack_time >= 0.12 and self._ammo >= 1):
                        if self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack"):
                            self.attack()
                            self._spend_ammo()
                            self.last_decision_reason += " +CLOSE FIRE"
                return dis_move
            else:
                self._disengage_active = False  # Can't disengage - wall blocked

        # === PEEK-SHOOT WALL CYCLING ===
        peek_move = self._update_peek_cycle(player_pos, enemy_coords, walls, attack_range)
        if peek_move is not None and peek_move != '':
            if not self.is_path_blocked(player_pos, peek_move.upper(), walls):
                self.last_decision_reason = f"PEEK: {self._peek_phase} ({peek_move})"
                # During 'fire' phase, also shoot
                if self._peek_phase == 'fire':
                    enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
                    if enemy_hittable and self._should_fire(enemy_hp_low=(self.enemy_hp_percent < 40)):
                        self.attack()
                        self._spend_ammo()
                return peek_move.upper() if peek_move else 'W'
        elif peek_move == '':
            # Hold still during fire phase
            pass

        # --- HEALTH-BASED BEHAVIOR (threshold varies by playstyle + adaptive aggression) ---
        # === STAT-AWARE RETREAT THRESHOLD ===
        # Low-HP brawlers (≤2600) retreat earlier, tanky brawlers (≥4500) retreat later
        hp_retreat_adj = 1.0
        if self._my_health <= 2600:
            hp_retreat_adj = 1.25  # Fragile: retreat 25% earlier
        elif self._my_health >= 4500:
            hp_retreat_adj = 0.7   # Tanky: retreat 30% later
        adjusted_retreat = int(hp_retreat_threshold * hp_retreat_adj / self.aggression_modifier)
        low_hp = (self.player_hp_percent < adjusted_retreat
                  and (time.time() - self.last_attack_time) < 10)

        # === POST-BURST DEFENSIVE MODE ===
        # After dumping all ammo in a burst, play cautiously until ammo recovers.
        # Increase retreat threshold and reduce approach factor.
        _post_burst_defensive = False
        _time_since_burst = time.time() - self._last_burst_end_time
        if (self._last_burst_end_time > 0
                and _time_since_burst < self._burst_defensive_duration
                and not self._burst_mode
                and not self._respawn_shield_active):
            _post_burst_defensive = True
            # Raise retreat threshold: more likely to retreat while ammo recovers
            adjusted_retreat = int(adjusted_retreat * 1.5)
            approach_factor *= 0.4  # Stay back - don't rush in without ammo
            low_hp = low_hp or self.player_hp_percent < adjusted_retreat
            if not self.last_decision_reason:
                self.last_decision_reason = f"DEFENSIVE: post-burst ({self._burst_defensive_duration - _time_since_burst:.1f}s)"

        # --- RESPAWN SHIELD OVERRIDE: ignore retreat, rush enemies! ---
        if self._respawn_shield_active:
            low_hp = False  # Don't retreat when invincible
            self.last_decision_reason = f"RESPAWN RUSH: shield active ({max(0, self._respawn_shield_until - time.time()):.1f}s left)"

        # --- LAST LIFE CAUTION (Knockout) ---
        if self._last_life_mode and not self._respawn_shield_active:
            adjusted_retreat = max(adjusted_retreat, 50)  # Retreat earlier on last life
            low_hp = (self.player_hp_percent < adjusted_retreat
                      and (time.time() - self.last_attack_time) < 10)

        # --- MATCH PHASE MODIFIERS ---
        if self._match_phase == 'early' and not self._respawn_shield_active:
            # Early game: be cautious, don't overcommit
            approach_factor *= 0.8
            adjusted_retreat = int(adjusted_retreat * 1.2)
        elif self._match_phase == 'late':
            # Late game: be more aggressive (time running out)
            approach_factor = max(approach_factor, 1.1)
            adjusted_retreat = max(5, int(adjusted_retreat * 0.7))

        # --- SCORE AWARENESS: adjust behavior based on score differential ---
        if self._score_diff >= 2 and not self._respawn_shield_active:
            # Winning big - play safe
            adjusted_retreat = int(adjusted_retreat * 1.4)
            low_hp = low_hp or self.player_hp_percent < adjusted_retreat
        elif self._score_diff <= -2:
            # Losing big - fight harder
            adjusted_retreat = max(5, int(adjusted_retreat * 0.6))

        # --- NUMBER ADVANTAGE PUSH (fewer enemies = push harder) ---
        if self._number_advantage_active and not low_hp:
            # We outnumber them! Lower retreat threshold and push forward
            adjusted_retreat = max(5, int(adjusted_retreat * 0.5))
            approach_factor = max(approach_factor, 1.2)
            if not self.last_decision_reason:
                self.last_decision_reason = f"ADVANTAGE: {self.target_info.get('n_enemies', 0)}/{self._expected_enemy_count} enemies"

        # --- ENEMY RELOAD WINDOW: push aggressively when target just fired ---
        self._enemy_in_reload_window = self._is_target_reloading(enemy_coords)
        if self._enemy_in_reload_window and not low_hp:
            approach_factor = max(approach_factor, 1.3)
            # Reduce dodge while they're reloading (approach faster)
            if not self.last_decision_reason:
                self.last_decision_reason = "PUNISH: enemy reloading"

        # --- TEAMMATE DEATH GAP COVER (movement bias toward undefended lane) ---
        if (self._teammate_death_pos and not low_hp
                and time.time() - self._teammate_death_time < 8.0):
            gap_dx = self._teammate_death_pos[0] - player_pos[0]
            gap_dy = self._teammate_death_pos[1] - player_pos[1]
            gap_dist = math.hypot(gap_dx, gap_dy)
            if gap_dist > 80:
                # Blend direction toward gap (30% gap + 70% normal combat direction)
                direction_x = direction_x * 0.7 + gap_dx * 0.3
                direction_y = direction_y * 0.7 + gap_dy * 0.3
                if not self.last_decision_reason:
                    self.last_decision_reason = f"GAP COVER: teammate down ({int(gap_dist)}px)"

        # --- CHOKE POINT AVOIDANCE (avoid retreating into narrow passages) ---
        # Bias direction away from nearby choke points when retreating
        if low_hp and self._choke_points:
            for cp_x, cp_y, cp_gap, cp_angle in self._choke_points:
                cp_dist = math.hypot(cp_x - player_pos[0], cp_y - player_pos[1])
                if cp_dist < 200:
                    # Repel away from choke point
                    repel_x = player_pos[0] - cp_x
                    repel_y = player_pos[1] - cp_y
                    repel_strength = (200 - cp_dist) / 200 * 0.4
                    direction_x += repel_x * repel_strength
                    direction_y += repel_y * repel_strength

        # --- MOVEMENT DECISION (varies by playstyle) ---
        should_group = True  # ALL modes: stay near teammates for coordinated play

        # Default movement (toward enemy) - overridden by branches below
        move_horizontal = self.get_horizontal_move_key(direction_x)
        move_vertical = self.get_vertical_move_key(direction_y)

        if low_hp:
            self.last_decision_reason = f"RETREAT: HP {self.player_hp_percent}% < {adjusted_retreat}%"
            # ALL modes: retreat TOWARD teammate for safety (strength in numbers)
            if teammates and nearest_teammate_pos:
                best_tm = nearest_teammate_pos
                best_td = nearest_teammate_dist
                if best_td > 80:
                    # Blend: retreat away from enemy (60%) + toward teammate (40%)
                    retreat_dx = -(enemy_coords[0] - player_pos[0])
                    retreat_dy = -(enemy_coords[1] - player_pos[1])
                    tm_dx = best_tm[0] - player_pos[0]
                    tm_dy = best_tm[1] - player_pos[1]
                    blend_x = retreat_dx * 0.6 + tm_dx * 0.4
                    blend_y = retreat_dy * 0.6 + tm_dy * 0.4
                    move_horizontal = 'D' if blend_x > 0 else 'A'
                    move_vertical = 'S' if blend_y > 0 else 'W'
                else:
                    move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
                    move_vertical = self.get_vertical_move_key(direction_y, opposite=True)
            else:
                # No teammates - use SAFE retreat (avoids dead-ends)
                safe_retreat = self._get_safe_retreat_direction(
                    player_pos, direction_x, direction_y, walls, self._spawn_side)
                # Parse the direction string into h/v components
                move_horizontal = ''
                move_vertical = ''
                for ch in safe_retreat.upper():
                    if ch in ('A', 'D'):
                        move_horizontal = ch
                    elif ch in ('W', 'S'):
                        move_vertical = ch
            # Add strafe to retreat for organic feel (dodge while retreating)
            if random.random() < 0.25:
                if abs(direction_x) > abs(direction_y):
                    move_vertical = self.get_vstrafe_key()
                else:
                    move_horizontal = self._get_juke_direction()
            # Throwers/snipers: keep firing while retreating (kiting)
            if keep_max_range and enemy_distance <= attack_range:
                now_ret = time.time()
                self._update_ammo(now_ret)
                kite_int = style.get("kite_interval", 0.25)
                if now_ret - self.last_attack_time >= kite_int and self._ammo >= 1:
                    if self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack"):
                        self.attack()
                        self._spend_ammo()
                        self.last_decision_reason += " +KITE FIRE"
            # ALL brawlers: fire while retreating if enemy is close
            elif enemy_distance <= safe_range and not keep_max_range:
                now_ret = time.time()
                self._update_ammo(now_ret)
                if now_ret - self.last_attack_time >= 0.12 and self._ammo >= 1:
                    if self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack"):
                        self.attack()
                        self._spend_ammo()
                        self.last_decision_reason += " +CLOSE KITE"

        elif should_group and teammates and nearest_teammate_dist > 160:
            self.last_decision_reason = f"REGROUP: teammate {int(nearest_teammate_dist)}px away"
            # Teammate is too far - regroup before fighting (strength in numbers!)
            if nearest_teammate_pos:
                # Blend: 60% toward teammate, 40% toward enemy (don't fully abandon fight)
                tm_dx = nearest_teammate_pos[0] - player_pos[0]
                tm_dy = nearest_teammate_pos[1] - player_pos[1]
                blend_x = tm_dx * 0.6 + direction_x * 0.4
                blend_y = tm_dy * 0.6 + direction_y * 0.4
                move_horizontal = 'D' if blend_x > 0 else 'A'
                move_vertical = 'S' if blend_y > 0 else 'W'
            else:
                move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
                move_vertical = self.get_vertical_move_key(direction_y, opposite=True)

        elif rush_low_enemy and self.enemy_hp_percent < 50 and self.enemy_hp_percent > 0:
            # Tanks & Assassins: RUSH low-HP enemies (ignore safe_range)
            # === STAT-AWARE: calculate if we can finish them ===
            estimated_enemy_hp = 3200 * (self.enemy_hp_percent / 100.0)
            can_finish_in = max(1, math.ceil(estimated_enemy_hp / max(1, self._my_attack_damage)))
            # Rush more aggressively if we can kill in ≤2 attacks
            if can_finish_in <= 2 and self._ammo >= can_finish_in:
                self.last_decision_reason = f"FINISH: {can_finish_in} shot kill! (enemy {self.enemy_hp_percent}%)"
            else:
                self.last_decision_reason = f"RUSH: enemy HP {self.enemy_hp_percent}% ({can_finish_in} shots)"
            # Intercept: move toward predicted position instead of current position
            intercept = self._get_intercept_position(player_pos, enemy_coords, enemy_distance)
            intercept_dx = intercept[0] - player_pos[0]
            intercept_dy = intercept[1] - player_pos[1]
            move_horizontal = 'D' if intercept_dx > 0 else 'A'
            move_vertical = 'S' if intercept_dy > 0 else 'W'

        elif (not rush_low_enemy and not low_hp
              and self.enemy_hp_percent < 35 and self.enemy_hp_percent > 0
              and self.target_info.get('n_enemies', 1) <= 1):
            # ALL brawlers: chase LOW-HP enemies to finish them (if alone and safe)
            estimated_enemy_hp = 3200 * (self.enemy_hp_percent / 100.0)
            can_finish_in = max(1, math.ceil(estimated_enemy_hp / max(1, self._my_attack_damage)))
            if can_finish_in <= 3 and self._ammo >= 1 and enemy_distance <= attack_range * 1.5:
                self.last_decision_reason = f"CHASE FINISH: {can_finish_in} shots (enemy {self.enemy_hp_percent}%)"
                # Intercept: cut off their escape route
                intercept = self._get_intercept_position(player_pos, enemy_coords, enemy_distance)
                intercept_dx = intercept[0] - player_pos[0]
                intercept_dy = intercept[1] - player_pos[1]
                move_horizontal = 'D' if intercept_dx > 0 else 'A'
                move_vertical = 'S' if intercept_dy > 0 else 'W'

        elif keep_max_range and enemy_distance < attack_range * (0.85 if playstyle == 'thrower' else 0.65):
            # Snipers & Throwers: enemy is too close! Back off to ideal range
            # Throwers kite at 85% range (they need MORE space for safety buffer), snipers at 65%
            self.last_decision_reason = f"KITE BACK: too close ({int(enemy_distance)}px)"
            # Retreat with a perpendicular strafe component for organic movement
            retreat_h = self.get_horizontal_move_key(direction_x, opposite=True)
            retreat_v = self.get_vertical_move_key(direction_y, opposite=True)
            # Add diagonal dodge: blend retreat with slight perpendicular movement
            if random.random() < 0.4:
                # Swap one component for a strafe (makes retreat path unpredictable)
                if abs(direction_x) > abs(direction_y):
                    retreat_v = self.get_vstrafe_key()
                else:
                    retreat_h = self._get_juke_direction()
            move_horizontal = retreat_h
            move_vertical = retreat_v

        elif prefer_wall_cover and enemy_distance <= attack_range:
            # Throwers: in range, try to position behind nearby walls
            # Check if any wall is between us and the enemy (good cover)
            has_cover = False
            if walls:
                player_to_enemy = LineString([player_pos, enemy_coords])
                for wall in walls:
                    wx1, wy1, wx2, wy2 = wall
                    wall_poly = Polygon([(wx1, wy1), (wx2, wy1), (wx2, wy2), (wx1, wy2)])
                    if player_to_enemy.intersects(wall_poly):
                        has_cover = True
                        break
            if has_cover:
                # Good: we have wall cover - diagonal strafe to maintain cover naturally
                self.last_decision_reason = "COVER: thrower behind wall - holding"
                # Use diagonal strafe with slight retreat bias for organic movement
                juke_h = self._get_juke_direction()
                juke_v = self.get_vstrafe_key()
                retreat_h = self.get_horizontal_move_key(direction_x, opposite=True)
                retreat_v = self.get_vertical_move_key(direction_y, opposite=True)
                # Mostly strafe perpendicular, occasionally add retreat component
                if random.random() < 0.25:
                    move_horizontal = juke_h
                    move_vertical = retreat_v  # Diagonal: strafe + slight back
                else:
                    move_horizontal = juke_h
                    move_vertical = juke_v  # Diagonal strafe (organic feel)
            else:
                # No cover: find the BEST cover wall (perpendicular to enemy LOS)
                # Score walls by how well they block the enemy's line of sight
                best_wall_score = float('-inf')
                best_wall_dir_x = 0
                best_wall_dir_y = 0
                enemy_angle = math.atan2(direction_y, direction_x)
                for wall in (walls or []):
                    wcx = (wall[0] + wall[2]) / 2
                    wcy = (wall[1] + wall[3]) / 2
                    wdist = math.hypot(wcx - player_pos[0], wcy - player_pos[1])
                    if wdist < 20 or wdist > 400:
                        continue  # Too close (on top of) or too far
                    # Angle from player to this wall
                    wall_angle = math.atan2(wcy - player_pos[1], wcx - player_pos[0])
                    # How perpendicular is this wall relative to enemy direction?
                    # Best cover: wall is between us and the enemy (angle diff ~0)
                    # or slightly off to the side (60-120 degrees = good flanking cover)
                    angle_diff = abs(math.atan2(
                        math.sin(wall_angle - enemy_angle),
                        math.cos(wall_angle - enemy_angle)
                    ))
                    # Prefer walls that are roughly between us and enemy (±90°)
                    if angle_diff < math.pi / 2:
                        cover_score = (math.pi / 2 - angle_diff) * 200  # Higher = more aligned
                    else:
                        cover_score = -50  # Wall is behind us, not useful
                    # Also prefer closer walls
                    cover_score -= wdist * 0.3
                    # Prefer larger walls (more cover)
                    wall_w = abs(wall[2] - wall[0])
                    wall_h = abs(wall[3] - wall[1])
                    cover_score += (wall_w + wall_h) * 0.1
                    if cover_score > best_wall_score:
                        best_wall_score = cover_score
                        best_wall_dir_x = wcx - player_pos[0]
                        best_wall_dir_y = wcy - player_pos[1]
                if best_wall_score > float('-inf'):
                    self.last_decision_reason = f"COVER: seeking wall (score={best_wall_score:.0f})"
                    move_horizontal = 'D' if best_wall_dir_x > 0 else 'A'
                    move_vertical = 'S' if best_wall_dir_y > 0 else 'W'
                else:
                    # Fallback: committed strafe
                    if abs(direction_y) > abs(direction_x):
                        move_horizontal = self._get_juke_direction()
                        move_vertical = self.get_vertical_move_key(direction_y, opposite=True)
                    else:
                        move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
                        move_vertical = self.get_vstrafe_key()

        elif enemy_distance <= safe_range:
            # Within safe range - behavior depends on playstyle
            if approach_factor >= 1.5:
                # Tanks: DON'T back off, keep rushing
                move_horizontal = self.get_horizontal_move_key(direction_x)
                move_vertical = self.get_vertical_move_key(direction_y)
            else:
                # Everyone else: back off
                move_horizontal = self.get_horizontal_move_key(direction_x, opposite=True)
                move_vertical = self.get_vertical_move_key(direction_y, opposite=True)

        elif approach_factor < 0.5:
            # Snipers/Throwers (approach_factor < 0.5): HOLD POSITION (don't approach)
            if enemy_distance <= attack_range:
                # In range - organic strafe with slight retreat bias to maintain distance
                self.last_decision_reason = f"HOLD RANGE: {int(enemy_distance)}px (strafing)"
                # Diagonal strafe: perpendicular + slight retreat for natural movement
                juke_h = self._get_juke_direction()
                juke_v = self.get_vstrafe_key()
                # Blend: 70% strafe perpendicular, 30% retreat bias (keeps distance)
                retreat_h = self.get_horizontal_move_key(direction_x, opposite=True)
                retreat_v = self.get_vertical_move_key(direction_y, opposite=True)
                if random.random() < 0.3:
                    # Retreat-strafe: diagonal retreat with perpendicular component
                    if abs(direction_y) > abs(direction_x):
                        move_horizontal = juke_h
                        move_vertical = retreat_v
                    else:
                        move_horizontal = retreat_h
                        move_vertical = juke_v
                else:
                    # Pure perpendicular strafe (use diagonal for organic feel)
                    move_horizontal = juke_h
                    move_vertical = juke_v
            elif enemy_distance <= attack_range * 1.3:
                # Slightly out of range - cautious approach with strafe
                approach_h = self.get_horizontal_move_key(direction_x)
                approach_v = self.get_vertical_move_key(direction_y)
                juke_h = self._get_juke_direction()
                # Approach diagonally (approach + perpendicular strafe)
                if abs(direction_y) > abs(direction_x):
                    move_horizontal = juke_h
                    move_vertical = approach_v
                else:
                    move_horizontal = approach_h
                    move_vertical = self.get_vstrafe_key()
            else:
                # Far out of range - move toward enemy directly
                move_horizontal = self.get_horizontal_move_key(direction_x)
                move_vertical = self.get_vertical_move_key(direction_y)

        elif style.get("prefer_teammates", False) and teammates:
            # Support: bias movement toward nearest teammate
            # PRIORITY: rush to low-HP teammate if they need healing
            heal_target_pos = None
            if self._teammate_hp_data:
                injured = [(x, y, hp) for x, y, hp in self._teammate_hp_data if hp < 60]
                if injured:
                    # Find nearest injured teammate
                    best_injured = min(injured,
                        key=lambda t: math.hypot(t[0] - player_pos[0], t[1] - player_pos[1]))
                    heal_target_pos = (best_injured[0], best_injured[1])
                    heal_dist = math.hypot(heal_target_pos[0] - player_pos[0],
                                           heal_target_pos[1] - player_pos[1])
                    if heal_dist > 80:
                        self.last_decision_reason = f"HEAL: teammate HP {best_injured[2]}% ({int(heal_dist)}px)"
                        move_horizontal = 'D' if heal_target_pos[0] > player_pos[0] else 'A'
                        move_vertical = 'S' if heal_target_pos[1] > player_pos[1] else 'W'
                    else:
                        heal_target_pos = None  # Already close enough

            if heal_target_pos is None:
                # Normal support behavior: stay near nearest teammate
                nearest_tm_pos = None
                nearest_tm_dist = float('inf')
                for tm in teammates:
                    tm_cx = (tm[0] + tm[2]) / 2
                    tm_cy = (tm[1] + tm[3]) / 2
                    td = math.hypot(tm_cx - player_pos[0], tm_cy - player_pos[1])
                    if td < nearest_tm_dist:
                        nearest_tm_dist = td
                        nearest_tm_pos = (tm_cx, tm_cy)
                if nearest_tm_pos and nearest_tm_dist > 200:
                    # Move toward teammate (blend with enemy direction)
                    tm_dx = nearest_tm_pos[0] - player_pos[0]
                    tm_dy = nearest_tm_pos[1] - player_pos[1]
                    blend_x = direction_x * approach_factor + tm_dx * (1 - approach_factor)
                    blend_y = direction_y * approach_factor + tm_dy * (1 - approach_factor)
                    move_horizontal = 'D' if blend_x > 0 else 'A'
                    move_vertical = 'S' if blend_y > 0 else 'W'
                else:
                    # Close to teammate or no teammate, use scaled approach
                    if approach_factor < 1.0 and enemy_distance <= attack_range:
                        # In range: strafe instead of approaching further
                        if abs(direction_y) > abs(direction_x):
                            move_horizontal = self._get_juke_direction()
                            move_vertical = ''
                        else:
                            move_horizontal = ''
                            move_vertical = self.get_vstrafe_key()
                    else:
                        move_horizontal = self.get_horizontal_move_key(direction_x)
                        move_vertical = self.get_vertical_move_key(direction_y)

        elif approach_factor < 1.0 and enemy_distance <= attack_range:
            # Ranged brawlers in attack range: diagonal strafe to dodge while maintaining range
            # Key: use BOTH axes for organic diagonal movement, not single-axis
            self.last_decision_reason = f"STRAFE: in range ({int(enemy_distance)}px)"
            juke_h = self._get_juke_direction()
            juke_v = self.get_vstrafe_key()
            retreat_h = self.get_horizontal_move_key(direction_x, opposite=True)
            retreat_v = self.get_vertical_move_key(direction_y, opposite=True)
            # Mix strafe patterns: pure strafe, retreat-strafe, approach-strafe
            roll = random.random()
            if roll < 0.5:
                # Diagonal strafe (most common - unpredictable)
                move_horizontal = juke_h
                move_vertical = juke_v
            elif roll < 0.75:
                # Retreat-strafe: back off slightly while strafing
                if abs(direction_y) > abs(direction_x):
                    move_horizontal = juke_h
                    move_vertical = retreat_v
                else:
                    move_horizontal = retreat_h
                    move_vertical = juke_v
            else:
                # Approach-strafe: close in slightly while strafing  
                approach_h = self.get_horizontal_move_key(direction_x)
                approach_v = self.get_vertical_move_key(direction_y)
                if abs(direction_y) > abs(direction_x):
                    move_horizontal = juke_h
                    move_vertical = approach_v
                else:
                    move_horizontal = approach_h
                    move_vertical = juke_v

        else:
            # Standard approach (fighter/assassin with factor >= 1.0, or out of range)
            if enemy_distance <= attack_range and approach_factor >= 1.0:
                # In attack range - committed strafe to dodge (playstyle-based probability)
                dodge_chance = style.get("dodge_chance", 0.35) * self.aggression_modifier
                if random.random() < dodge_chance:
                    if abs(direction_y) > abs(direction_x):
                        move_horizontal = self._get_juke_direction()
                        move_vertical = self.get_vertical_move_key(direction_y)
                    else:
                        move_horizontal = self.get_horizontal_move_key(direction_x)
                        move_vertical = self.get_vstrafe_key()
                else:
                    move_horizontal = self.get_horizontal_move_key(direction_x)
                    move_vertical = self.get_vertical_move_key(direction_y)
            else:
                move_horizontal = self.get_horizontal_move_key(direction_x)
                move_vertical = self.get_vertical_move_key(direction_y)

        movement_options = [move_horizontal + move_vertical]
        if self.game_mode == 3:
            movement_options += [move_vertical, move_horizontal]
        elif self.game_mode == 5:
            movement_options += [move_horizontal, move_vertical]
        else:
            # Unknown game mode type - fallback to vertical priority
            print(f"[WARN] Unknown gamemode type {self.game_mode}, defaulting to type 3 behavior")
            movement_options += [move_vertical, move_horizontal]

        # Filter empty strings
        movement_options = [m for m in movement_options if m]
        if not movement_options:
            movement_options = [self.no_enemy_movement(player_data, walls, playstyle)]

        # Check for walls and adjust movement - use A* pathfinding FIRST
        movement = None
        
        # PRIMARY: Use A* pathfinder proactively (avoids walls before hitting them)
        pf_move = self._get_pathfinder_movement(player_pos, enemy_coords)
        if pf_move:
            movement = pf_move
        else:
            # FALLBACK: Direct WASD with wall collision check
            for move in movement_options:
                if not self.is_path_blocked(player_pos, move, walls):
                    movement = move
                    break

        if movement is None:
            # All direct options blocked too -> try all 8 directions randomly
            all_dirs = list(self.ALL_DIRECTIONS)
            random.shuffle(all_dirs)
            for move in all_dirs:
                if not self.is_path_blocked(player_pos, move, walls):
                    movement = move
                    break
            if movement is None:
                movement = movement_options[0] if movement_options else 'W'

        # --- MOVEMENT MOMENTUM: resist rapid direction flipping for organic paths ---
        # If we just committed to a direction, keep it for at least _move_momentum_min seconds
        # UNLESS: retreating (low_hp), wall-blocked, or gas-fleeing
        current_time = time.time()
        if (movement != self._last_move_dir
                and self._last_move_dir
                and not low_hp
                and (current_time - self._move_dir_start) < self._move_momentum_min
                and not self.is_path_blocked(player_pos, self._last_move_dir, walls)):
            movement = self._last_move_dir  # Keep current direction (momentum)
        else:
            if movement != self._last_move_dir:
                self._last_move_dir = movement
                self._move_dir_start = current_time

        if movement != self.last_movement:
            self.last_movement = movement
            self.last_movement_time = current_time
        else:
            self.last_movement_time = current_time

        # --- ATTACK LOGIC (ammo-aware, with kiting support) ---
        now = time.time()
        self._update_ammo(now)

        # --- COMBO CHAIN: fire follow-up attack shortly after super ---
        if self._combo_queued and (now - self._combo_queue_time) >= 0.12:
            if self._combo_type == 'super_then_attack':
                enemy_hittable_combo = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")
                if enemy_hittable_combo and enemy_distance <= attack_range:
                    self.attack()
                    self._spend_ammo()
                    self.last_decision_reason = "COMBO: attack after super"
            self._combo_queued = False
            self._combo_type = None

        enemy_hittable_attack = self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack")

        # SAFETY: Skip ALL attack logic if enemy is behind a wall
        if not enemy_hittable_attack:
            self.last_decision_reason = f"NO SHOT: wall blocked ({int(enemy_distance)}px)"
            return movement

        # === BURST FIRE MODE: dump all ammo when fully loaded + enemy in range ===
        # Trigger: ammo is FULL and enemy is hittable within range
        # Effect: near-zero attack interval until ammo is empty
        # After burst: play defensively (handled in retreat threshold section)
        burst_active = False
        if self._ammo >= self._max_ammo and enemy_distance <= attack_range:
            # Full ammo - START or CONTINUE burst
            if not self._burst_mode:
                self._burst_mode = True
                self._burst_start_time = now
            burst_active = True
        elif self._burst_mode and self._ammo > 0:
            # Mid-burst - keep dumping until empty
            burst_active = True
        elif self._burst_mode and self._ammo <= 0:
            # Burst finished - switch to defensive
            self._burst_mode = False
            self._last_burst_end_time = now

        # Use per-playstyle attack interval; fire faster while kiting (retreating)
        base_interval = style.get("attack_interval", 0.35)
        kite_interval = style.get("kite_interval", 0.25)
        attack_interval = kite_interval if low_hp else base_interval

        if burst_active:
            # BURST: override interval to near-instant (60ms between shots)
            attack_interval = self._burst_interval
        else:
            # Normal pacing modifiers (only when NOT in burst)
            # Fire faster during enemy reload window and number advantage
            if self._enemy_in_reload_window:
                attack_interval *= 0.7  # 30% faster when they can't shoot back
            if self._number_advantage_active:
                attack_interval *= 0.85  # 15% faster when we outnumber them
            # === STAT-AWARE ATTACK PACING ===
            # High-damage brawlers (≥2000 dmg) space shots out to aim carefully
            # Low-damage brawlers (<1000 dmg) spam faster for DPS
            # Multi-projectile brawlers (≥3 projectiles) fire slightly faster (wider spread = forgiving)
            if self._my_attack_damage >= 2000 and self._my_projectile_count == 1:
                attack_interval *= 1.1  # Heavy hitters: wait 10% longer for accuracy
            elif self._my_attack_damage < 1000:
                attack_interval *= 0.8  # Low damage: spam 20% faster to compensate
            if self._my_projectile_count >= 3:
                attack_interval *= 0.9  # Multi-projectile: 10% faster (wider spread is forgiving)
            # Low-HP brawler with enemy close: panic fire faster
            if self._my_health <= 2600 and enemy_distance < safe_range:
                attack_interval *= 0.7  # Desperate: unload ammo fast before dying

            # === CLOSE-RANGE SPAM: massively increase fire rate when enemy is nearby ===
            # Point-blank (< 50% safe_range): MAXIMUM spam - nearly instant firing
            if enemy_distance < safe_range * 0.5:
                attack_interval = min(attack_interval, 0.08)  # Cap at 80ms - dump everything
            # Close range (< safe_range): heavy spam
            elif enemy_distance < safe_range:
                attack_interval = min(attack_interval, 0.12)  # Cap at 120ms
            # Mid-close (< 70% attack_range): moderately faster
            elif enemy_distance < attack_range * 0.7:
                attack_interval *= 0.6  # 40% faster at mid-close range

        # Attack when in REAL range and hittable - with ammo conservation
        # Use predictive aim: if enemy is moving fast, require them to be closer (higher hit chance)
        enemy_is_low = (self.enemy_hp_percent < 40 and self.enemy_hp_percent > 0)
        lead_offset = self._get_lead_offset(enemy_distance, playstyle)
        lead_magnitude = math.hypot(lead_offset[0], lead_offset[1])
        # Tighten effective range for fast movers - but NOT for throwers (area denial, no precision needed)
        effective_range = attack_range
        if playstyle != 'thrower':
            if lead_magnitude > 80:
                effective_range = attack_range * 0.70  # Very fast movers - get closer
            elif lead_magnitude > 40:
                effective_range = attack_range * 0.85  # Moderate strafing - slight tightening

        if enemy_hittable_attack and enemy_distance <= effective_range:
            if (now - self.last_attack_time) >= attack_interval:
                # === HOLD-ATTACK BRAWLER CHECK ===
                # Brawlers like Hank & Angelo charge their attack by holding, then release.
                _is_hold_brawler = self.must_brawler_hold_attack(self.current_brawler, self.brawlers_info)
                if _is_hold_brawler:
                    hold_duration = self.brawlers_info.get(self.current_brawler, {}).get('hold_attack', 3)
                    if self.time_since_holding_attack is None:
                        # Start holding attack (press down only)
                        self.attack(touch_up=False, touch_down=True)
                        self.time_since_holding_attack = time.time()
                        self.last_decision_reason = f"HOLD-ATTACK START: charging ({hold_duration}s)"
                    elif now - self.time_since_holding_attack >= hold_duration + self.seconds_to_hold_attack_after_reaching_max:
                        # Fully charged + buffer - release attack
                        self.attack(touch_up=True, touch_down=False)
                        self.time_since_holding_attack = None
                        self._spend_ammo()
                        self.last_decision_reason = f"HOLD-ATTACK RELEASE: {int(enemy_distance)}px"
                    else:
                        self.last_decision_reason = f"HOLD-ATTACK CHARGING: {now - self.time_since_holding_attack:.1f}/{hold_duration}s"
                else:
                    # === Normal tap-fire brawlers ===
                    # === AUTO-AIM TARGET CONSISTENCY CHECK ===
                    # Auto-aim fires at the NEAREST enemy, not our chosen target.
                    # If a closer unhittable enemy exists, auto-aim would waste ammo on them.
                    # Only skip if multiple enemies visible and our target isn't the closest.
                    autoaim_would_miss = False
                    closest_dist = enemy_distance  # Default: our target is the closest
                    if len(enemy_data) >= 2:
                        closest_dist = float('inf')
                        for en in enemy_data:
                            en_pos = self.get_enemy_pos(en)
                            en_dist = self.get_distance(en_pos, player_pos)
                            if en_dist < closest_dist:
                                closest_dist = en_dist
                        # Our target is NOT the closest - auto-aim will fire at wrong enemy
                        if enemy_distance > closest_dist + 30:  # 30px tolerance
                            autoaim_would_miss = True
                    # Close range: fire at ANY ammo (no conservation - dump everything)
                    # Throwers also fire at 1 ammo. Only conserve at longer ranges.
                    # BURST MODE: always fire - no conservation whatsoever
                    if burst_active:
                        should_fire = (self._ammo >= 1)  # Burst = dump everything
                    elif enemy_distance < safe_range:
                        should_fire = (self._ammo >= 1)  # Close range = no ammo conservation
                    elif playstyle == 'thrower':
                        should_fire = (self._ammo >= 1)  # Throwers always spam
                    else:
                        should_fire = self._should_fire(enemy_hp_low=enemy_is_low)
                    if should_fire and not autoaim_would_miss:
                        # Auto-aim targets nearest enemy - verified our target IS the closest
                        self.attack()
                        self._spend_ammo()
                        # Restore previous movement direction
                        if movement:
                            self.do_movement(movement)
                        burst_tag = " [BURST]" if burst_active else ""
                        self.last_decision_reason = f"ATTACK: {int(enemy_distance)}px, ammo={self._ammo}{burst_tag}"
                        # POST-ATTACK DODGE: force a fresh strafe direction right after shooting
                        dodge_chance = style.get("dodge_chance", 0.35)
                        if random.random() < dodge_chance:
                            self._strafe_direction = random.choice(['A', 'D'])
                            self._strafe_start_time = now
                            self._strafe_duration = random.uniform(0.30, 0.60)  # Longer dodge arcs
                    elif autoaim_would_miss:
                        self.last_decision_reason = f"SKIP: auto-aim wrong target ({int(enemy_distance)}px vs {int(closest_dist)}px)"
                    else:
                        self.last_decision_reason = f"CONSERVE AMMO: {self._ammo}/{self._max_ammo}"
            else:
                # Enemy out of range or on cooldown - cancel held attack if any
                if self.time_since_holding_attack is not None:
                    self.attack(touch_up=True, touch_down=False)
                    self.time_since_holding_attack = None
                    self.last_decision_reason = "HOLD-ATTACK CANCELLED: out of range"

        # Use hypercharge proactively in combat (not gated behind attack range)
        if self.is_hypercharge_ready and enemy_distance <= attack_range * 1.5:
            if self.use_hypercharge():
                self.time_since_hypercharge_checked = now
                self.is_hypercharge_ready = False
                self.last_decision_reason = "HYPERCHARGE USED"

        # Use gadget AGGRESSIVELY - fire as soon as enemy is in attack range
        if self.should_use_gadget and self.is_gadget_ready:
            # Close range: ALWAYS use gadget immediately
            if enemy_distance <= safe_range:
                if self.use_gadget():
                    self.time_since_gadget_checked = now
                    self.is_gadget_ready = False
                    self.last_decision_reason = "GADGET USED (close range!)"
            # In attack range: use gadget proactively
            elif enemy_distance <= attack_range:
                if self.use_gadget():
                    self.time_since_gadget_checked = now
                    self.is_gadget_ready = False
                    self.last_decision_reason = "GADGET USED (in range)"
            # Defensive: use when low HP and enemy nearby
            elif low_hp and enemy_distance <= attack_range * 1.5:
                if self.use_gadget():
                    self.time_since_gadget_checked = now
                    self.is_gadget_ready = False
                    self.last_decision_reason = "GADGET USED (defensive)"

        # Use super - SMART conditions based on super_type and game state
        if self.is_super_ready:
            super_type = brawler_info['super_type']
            enemy_hittable_super = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")
            n_enemies = self.target_info.get('n_enemies', 1)
            enemy_is_low_super = (self.enemy_hp_percent < 40 and self.enemy_hp_percent > 0)

            # === SUPER VALUE HOLD: skip if we should save it ===
            if self._should_hold_super(super_type, n_enemies, enemy_is_low_super, enemy_distance, super_range):
                self._hold_super = True
                self.last_decision_reason = f"HOLD SUPER: saving (score +{self._score_diff})"
            else:
                self._hold_super = False

            # Self-buff supers (other/spawnable) can be used with enemies nearby
            if not self._hold_super and super_type in ["spawnable", "other", "other_target"]:
                if enemy_distance <= attack_range * 2:
                    if self.use_super():
                        self.time_since_super_checked = now
                        self.is_super_ready = False
                        self.last_decision_reason = f"SUPER: {super_type} (utility)"

            # Charge supers (Bull, Darryl, Stu) - save for escape when low HP
            elif not self._hold_super and super_type == "charge":
                if low_hp and self.player_hp_percent < 15:
                    # Emergency escape! Use charge super to flee
                    if self.use_super():
                        self.time_since_super_checked = now
                        self.is_super_ready = False
                        self.last_decision_reason = "SUPER: charge (ESCAPE!)"
                elif enemy_hittable_super and enemy_distance <= super_range + attack_range:
                    if self.use_super():
                        self.time_since_super_checked = now
                        self.is_super_ready = False
                        self.last_decision_reason = f"SUPER: charge (gap-close {int(enemy_distance)}px)"
                        # Queue follow-up attack after gap-closing
                        self._combo_queued = True
                        self._combo_queue_time = now
                        self._combo_type = 'super_then_attack'

            # Damage supers - USE whenever enemy is in range and hittable
            # === AGGRESSIVE: fire super as soon as conditions are met ===
            elif not self._hold_super and super_type == "damage":
                should_use = False
                estimated_enemy_hp = 3200 * (self.enemy_hp_percent / 100.0) if self.enemy_hp_percent > 0 else 3200
                super_can_kill = (self._my_super_damage >= estimated_enemy_hp) if self._my_super_damage > 0 else False
                if n_enemies >= 2 and enemy_hittable_super and enemy_distance <= super_range:
                    should_use = True  # Multi-kill opportunity!
                    self.last_decision_reason = f"SUPER: damage ({n_enemies} enemies!)"
                elif super_can_kill and enemy_hittable_super and enemy_distance <= super_range:
                    should_use = True  # Super alone can finish them
                    self.last_decision_reason = f"SUPER: lethal! ({self._my_super_damage} dmg vs {int(estimated_enemy_hp)} hp)"
                elif enemy_hittable_super and enemy_distance <= super_range and enemy_is_low:
                    should_use = True  # Finishing blow
                    self.last_decision_reason = "SUPER: damage (finish)"
                elif enemy_hittable_super and enemy_distance <= super_range:
                    should_use = True  # In range and hittable - JUST USE IT
                    self.last_decision_reason = f"SUPER: damage (in range: {int(enemy_distance)}px)"
                if should_use:
                    if self.use_super():
                        self.time_since_super_checked = now
                        self.is_super_ready = False
                        # Queue follow-up attack to burst after damage super
                        self._combo_queued = True
                        self._combo_queue_time = now
                        self._combo_type = 'super_then_attack'

            # Normal offensive supers - need hittable + in range
            elif not self._hold_super and enemy_hittable_super and enemy_distance <= super_range:
                if self.use_super():
                    self.time_since_super_checked = now
                    self.is_super_ready = False
                    self.last_decision_reason = f"SUPER: {super_type}"
                    self._combo_queued = True
                    self._combo_queue_time = now
                    self._combo_type = 'super_then_attack'

        return movement

    def main(self, frame, brawler):
        current_time = time.time()
        self._current_frame = frame  # Store for BT subsystems
        data = self.get_main_data(frame)

        # --- IN-MATCH SHOWDOWN DETECTION ---
        # Check for "Teams left" text at top of screen (only once, throttled)
        if not self._showdown_detected_in_match and not self.is_showdown:
            if current_time - self._last_showdown_check > 10.0:
                self._last_showdown_check = current_time
                try:
                    wr = self.window_controller.width_ratio
                    hr = self.window_controller.height_ratio
                    # "Teams left: X" appears at top center during Showdown
                    top_crop = frame.crop((
                        int(600 * wr), int(0 * hr),
                        int(1300 * wr), int(80 * hr)
                    ))
                    from utils import extract_text_and_positions
                    texts = extract_text_and_positions(top_crop)
                    for text_key in texts:
                        if 'team' in text_key.lower() and 'left' in text_key.lower():
                            print(f"[IN-MATCH] Detected Showdown from '{text_key}' text!")
                            self.is_showdown = True
                            self._showdown_detected_in_match = True
                            self.game_mode_name = "showdown"
                            break
                except Exception:
                    pass

        # --- SHOWDOWN STORM ZONE DETECTION (every 3s) ---
        if self.is_showdown and current_time - self._last_storm_check > 3.0:
            self._last_storm_check = current_time
            self._detect_storm_zone(frame)

        # --- POISON GAS DETECTION (only for modes that have it) ---
        # Knockout, Showdown, Duels, Wipeout have poison gas.
        # Gem Grab, Brawl Ball, Hot Zone, Heist, Bounty do NOT.
        _gas_modes = ('knockout', 'showdown', 'duels', 'wipeout',
                      'soloshowdown', 'duoshowdown', 'solo showdown', 'duo showdown')
        if self.game_mode_name in _gas_modes and current_time - self._last_gas_check > 1.0:
            self._last_gas_check = current_time
            self._detect_poison_gas(frame)

        # --- SCORE DETECTION (throttled OCR every 12s) ---
        if not self.is_showdown and current_time - self._last_score_check > 12.0:
            self._last_score_check = current_time
            try:
                wr = self.window_controller.width_ratio
                hr = self.window_controller.height_ratio
                # Score is displayed top-center: "X - Y" format
                score_crop = frame.crop((
                    int(860 * wr), int(0 * hr),
                    int(1060 * wr), int(60 * hr)
                ))
                from utils import extract_text_and_positions
                score_texts = extract_text_and_positions(score_crop)
                for text_key in score_texts:
                    import re
                    # Match patterns like "2-1", "2 - 1", "0-3", etc.
                    match = re.search(r'(\d)\s*[-:]\s*(\d)', text_key)
                    if match:
                        s1, s2 = int(match.group(1)), int(match.group(2))
                        # In Knockout/Bounty: left score is blue (our) team
                        # Determine which is ours based on spawn side
                        if self._spawn_side in ('bottom', 'left'):
                            self._our_score, self._their_score = s1, s2
                        else:
                            self._our_score, self._their_score = s2, s1
                        self._score_diff = self._our_score - self._their_score
                        # Detect last life in Knockout (best of 3 or 5)
                        if self.game_mode_name in ('knockout', 'wipeout') and self._their_score >= 2:
                            self._last_life_mode = True
                        print(f"[SCORE] {self._our_score}-{self._their_score} (diff={self._score_diff:+d})")
                        break
            except Exception:
                pass

        # --- WALL DETECTION (with forced refresh after super for destroyed walls) ---
        should_scan_walls = (
            self.should_detect_walls
            and (current_time - self.time_since_walls_checked > self.walls_treshold
                 or self._force_wall_refresh)
        )
        if should_scan_walls:
            self._last_frame_ref = frame  # Store for color-based wall fallback
            tile_data = self.get_tile_data(frame)
            walls = self.process_tile_data(tile_data)
            walls = self._filter_walls_overlapping_entities(walls, data)

            # Detect destroyed walls (compare with pre-super snapshot)
            if self._force_wall_refresh and self._pre_super_walls is not None:
                pre_set = set(tuple(w) for w in self._pre_super_walls)  # type: ignore[union-attr]
                post_set = set(tuple(w) for w in walls)
                destroyed = pre_set - post_set
                if destroyed:
                    for dw in list(destroyed):
                        self._destroyed_wall_zones.append(list(dw))
                    print(f"[WALLS] Detected {len(destroyed)} destroyed walls after super!")
                    # Clear wall history that contained these walls
                    self.wall_history = [walls]
                self._pre_super_walls = None
            self._force_wall_refresh = False

            self.time_since_walls_checked = current_time
            self.last_walls_data = walls
            # Clear expired ghost wall when fresh wall data arrives
            self._ghost_wall = None
            self._ghost_wall_expire = 0.0
            data['wall'] = walls

            # --- CHOKE POINT DETECTION (scan every wall refresh) ---
            if current_time - self._last_choke_scan > 5.0:
                self._last_choke_scan = current_time
                self._detect_choke_points(walls)
        elif self.keep_walls_in_memory:
            data['wall'] = self.last_walls_data

        # --- WATER / HAZARD DETECTION (HSV color-based, runs less often) ---
        if (self._water_detector is not None
                and current_time - self._last_water_scan > self._water_scan_interval):
            self._last_water_scan = current_time
            try:
                player_bbox = data.get('player', [[]])[0] if data.get('player') else None
                water_bboxes = self._water_detector.detect(frame, player_bbox=player_bbox)
                self._water_bboxes = water_bboxes

                # Feed water tiles into spatial memory as WATER cells
                if water_bboxes and hasattr(self, '_spatial_memory') and self._spatial_memory is not None:
                    from spatial_memory import WATER as _SM_WATER
                    for wb in water_bboxes:
                        x1, y1, x2, y2 = wb[:4]
                        for r, c in self._spatial_memory.bbox_to_grid_cells(x1, y1, x2, y2):
                            if self._spatial_memory.grid[r, c] not in (2,):  # Don't overwrite WALL
                                self._spatial_memory.grid[r, c] = _SM_WATER
                                self._spatial_memory.observed[r, c] = True
            except Exception:
                pass

        # Merge water bboxes into wall data so is_path_blocked() avoids them
        if self._water_bboxes and 'wall' in data and data['wall'] is not None:
            data['wall'] = data['wall'] + self._water_bboxes

        data = self.validate_game_data(data)
        self.track_no_detections(data)
        if data:
            # --- RESPAWN DETECTION: player reappeared after dying ---
            time_gone = current_time - self.time_since_player_last_found
            # Wipeout/duels have FAST respawns (~1.5s); knockout/bounty ~3s; others ~2s+
            # Use shorter threshold for fast-respawn modes
            fast_respawn_modes = ('wipeout', 'duels', 'knockout', 'bounty')
            respawn_threshold = 1.0 if self.game_mode_name in fast_respawn_modes else 2.0
            if not self._player_was_visible and time_gone > respawn_threshold and self._spawn_detected:
                # Player was gone and just came back = we respawned
                self._death_count += 1
                self._respawn_shield_active = True
                self._respawn_shield_until = current_time + 3.0  # 3s invincibility
                self.player_hp_percent = 100
                # Force HP damage model to 100% (prevents false low-HP after respawn)
                if hasattr(self, '_hp_estimator') and self._hp_estimator:
                    self._hp_estimator.register_respawn("player")
                print(f"[RESPAWN] Death #{self._death_count} - shield active for 3s!")
                # Reset spawn detection for new respawn position
                self._spawn_detect_frames = 0
                self._spawn_detected = False
            self._player_was_visible = True
            self._is_dead = False
            self.time_since_player_last_found = time.time()

            # Expire respawn shield
            if self._respawn_shield_active and current_time > self._respawn_shield_until:
                self._respawn_shield_active = False
                print("[RESPAWN] Shield expired.")

            # --- SPAWN-SIDE DETECTION (first few frames of a match) ---
            if not self._spawn_detected and self._spawn_detect_frames < 10:
                self._spawn_detect_frames += 1
                if 'player' in data and data['player']:
                    pp = self.get_player_pos(data['player'][0])
                    mid_x = brawl_stars_width * self.window_controller.width_ratio / 2
                    mid_y = brawl_stars_height * self.window_controller.height_ratio / 2
                    if self.game_mode == 5:  # Horizontal modes (Brawl Ball, Heist)
                        self._spawn_side = 'left' if pp[0] < mid_x else 'right'
                    else:  # Vertical modes (Knockout, Gem Grab, etc.)
                        self._spawn_side = 'top' if pp[1] < mid_y else 'bottom'
                    self._spawn_detected = True
                    print(f"[SPAWN] Detected spawn side: {self._spawn_side} (pos={int(pp[0])},{int(pp[1])})")
                    # Initialize match start time on first spawn detection
                    if not self._match_phase_set:
                        # New match! Determine if previous match was won for RL training
                        prev_match_won = False
                        reset_token = None
                        try:
                            # Check trophy observer for last game result via module-level reference
                            import sys
                            main_mod = sys.modules.get('__main__')
                            sm = getattr(main_mod, '_active_stage_manager', None) if main_mod else None
                            if sm and hasattr(sm, 'Trophy_observer'):
                                tobs = sm.Trophy_observer
                                last_result = getattr(tobs, '_last_game_result', None)
                                match_counter = int(getattr(tobs, 'match_counter', 0) or 0)
                                reset_token = (match_counter, last_result)
                                prev_match_won = (last_result == "victory")
                        except Exception:
                            pass

                        # Duplicate guard: avoid 2-3 resets around respawn/state flicker.
                        # If TrophyObserver signature didn't change recently, skip reset.
                        should_reset = True
                        min_interval = float(getattr(self, '_min_episode_interval_sec', 30.0) or 30.0)
                        since_last_reset = current_time - float(self._last_episode_reset_time or 0.0)
                        if self._last_episode_reset_time > 0 and since_last_reset < min_interval:
                            should_reset = False
                        if reset_token is not None and self._last_episode_reset_token is not None:
                            if (reset_token == self._last_episode_reset_token
                                    and (current_time - self._last_episode_reset_time) < 60.0):
                                should_reset = False
                        elif (current_time - self._last_episode_reset_time) < 8.0:
                            # Fallback when no token is available
                            should_reset = False

                        if should_reset:
                            # Reset all per-match state from previous match
                            self.reset_match_state(match_won=prev_match_won)
                            self._last_episode_reset_token = reset_token
                            self._last_episode_reset_time = current_time
                        else:
                            print(f"[RESET] Episode reset prevented (token={reset_token}, dt={since_last_reset:.1f}s)")

                        self._spawn_detected = True  # Re-set after reset cleared it
                        self._spawn_side = 'left' if (self.game_mode == 5 and pp[0] < mid_x) else \
                                           'right' if self.game_mode == 5 else \
                                           'top' if pp[1] < mid_y else 'bottom'
                        self._match_start_time = current_time
                        self._match_phase_set = True
                        self._match_phase = 'early'
                        print(f"[PHASE] Match started - phase: early")

        if not data:
            self._player_was_visible = False
            # Mark as dead if player not seen for threshold time and match has started
            # Wipeout/duels have FAST respawns - use shorter threshold
            fast_respawn_modes = ('wipeout', 'duels', 'knockout', 'bounty')
            dead_threshold = 1.0 if self.game_mode_name in fast_respawn_modes else 2.0
            if current_time - self.time_since_player_last_found > dead_threshold and self._spawn_detected:
                self._is_dead = True
            # Release all movement keys immediately when player is not detected
            # (round ended, dead, or transition screen)
            time_since_player = current_time - self.time_since_player_last_found
            if time_since_player > 0.5:
                self.window_controller.keys_up(list("wasd"))
            self.time_since_different_movement = time.time()
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                current_state = get_state(frame)
                if current_state != "match":
                    self.time_since_last_proceeding = current_time
                else:
                    print("haven't detected the player in a while proceeding")
                    self.window_controller.press_key("Q")
                    self.time_since_last_proceeding = time.time()
            # Show debug overlay even when no player detected
            self._show_debug_overlay(frame, {}, "NO PLAYER", brawler,
                                     stats_info=getattr(self, '_stats_info', None))
            # Update visual overlay with dead state so it auto-hides
            if self.visual_overlay is not None:
                self.visual_overlay.update(
                    is_dead=self._is_dead,
                    game_state=getattr(self, '_stats_info', {}).get('state', 'match'),
                    match_phase=getattr(self, '_match_phase', 'early'),
                    our_score=getattr(self, '_our_score', 0),
                    their_score=getattr(self, '_their_score', 0),
                    death_count=getattr(self, '_death_count', 0),
                    kills=getattr(self, '_enemies_killed_this_match', 0),
                    player_hp=0 if self._is_dead else getattr(self, 'player_hp_percent', 100),
                    enemy_hp=getattr(self, 'enemy_hp_percent', -1),
                )
            return
        self.time_since_last_proceeding = time.time()
        if current_time - self.time_since_hypercharge_checked > self.hypercharge_treshold:
            self.is_hypercharge_ready = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time
        if current_time - self.time_since_gadget_checked > self.gadget_treshold:
            self.is_gadget_ready = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time
        if current_time - self.time_since_super_checked > self.super_treshold:
            self.is_super_ready = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time

        # Read health bars + target info BEFORE combat decisions (so data is fresh)
        # === HP DETECTION THROTTLING (IPS optimization) ===
        should_check_hp = current_time - self._last_hp_check_time > self._hp_check_interval
        try:
            frame_for_hp = frame
            # Player HP - uses HPEstimator v2 (with built-in smoothing + confidence)
            if 'player' in data and data['player'] and should_check_hp:
                self._last_hp_check_time = current_time
                hp_debug = {}

                # Primary: HPEstimator v2 with temporal smoothing
                player_hp, player_conf = self._hp_estimator.estimate(
                    frame_for_hp, data['player'][0],
                    is_player=True, entity_key="player")

                # Fallback: legacy method if HPEstimator fails
                if player_hp < 0:
                    player_hp = self.estimate_hp_from_bar(
                        frame_for_hp, data['player'][0],
                        is_player=True, debug_info=hp_debug)
                    player_conf = 0.3 if player_hp >= 0 else 0.0

                self._hp_debug_info = hp_debug
                self._hp_confidence_player = player_conf

                # Debug: print HP detection info every ~60 checks (~4s)
                self._hp_debug_counter = getattr(self, '_hp_debug_counter', 0) + 1
                if self._hp_debug_counter % 60 == 0:  # Log every 60 frames for debugging
                    bbox = data['player'][0]
                    cw = int(bbox[2]) - int(bbox[0])
                    ch = int(bbox[3]) - int(bbox[1])
                    raw_dbg = getattr(self._hp_estimator, '_last_raw_debug', {}).get('player', {})
                    rows_found = raw_dbg.get('rows_found', '?')
                    s_above = raw_dbg.get('search_above', '?')
                    s_below = raw_dbg.get('search_below', '?')
                    depleted = raw_dbg.get('any_depleted', '?')
                    print(f"[HP-DBG] hp={player_hp} conf={player_conf:.2f} "
                          f"cur={self.player_hp_percent} "
                          f"rows={rows_found} above={s_above} below={s_below} "
                          f"depleted={depleted} bbox={cw}x{ch}")

                if player_hp >= 0:
                    # HPEstimator already handles smoothing + fast-descent
                    # No double-smoothing needed - just accept the value
                    old_hp = self.player_hp_percent

                    # Detect damage taken (for reactive dodge + regen tracking)
                    if player_hp < old_hp - 5:
                        self._last_damage_taken_time = current_time

                    # === HP-BASED DEATH DETECTION (Wipeout fix) ===
                    # If HP was very low and suddenly jumps to 100% = respawned (fast modes)
                    # This catches deaths even when ONNX doesn't lose player bbox
                    if old_hp <= 15 and player_hp >= 95 and self._spawn_detected:
                        self._death_count += 1
                        self._respawn_shield_active = True
                        self._respawn_shield_until = current_time + 3.0
                        print(f"[RESPAWN-HP] Death #{self._death_count} detected via HP jump ({old_hp}%->{player_hp}%)")
                        self._hp_estimator.register_respawn("player")

                    self.player_hp_percent = max(1, min(100, player_hp))
                    self._last_valid_player_hp = self.player_hp_percent
                    self._last_valid_player_hp_time = current_time
                    self._player_hp_fail_count = 0
                    
                    # === BRAWLER AUTO-DETECTION BY HP ===
                    # At match start (first few seconds), try to detect brawler from max HP
                    if not self._brawler_detected and self._spawn_detected:
                        if self._match_start_hp_sample is None and self.player_hp_percent >= 95:
                            # First full HP reading - likely max HP
                            self._match_start_hp_sample = self.player_hp_percent
                            # Try to detect brawler from this HP value
                            try:
                                from utils import find_brawler_by_hp
                                # Get brawler info to look up HP values
                                brawler_info = self.brawlers_info.get(self.current_brawler, {})
                                expected_hp = brawler_info.get('health', 3200)
                                # Only auto-detect if HP differs significantly from configured brawler
                                actual_hp_approx = int(expected_hp * (self.player_hp_percent / 100))
                                detected = find_brawler_by_hp(expected_hp, self.brawlers_info, tolerance=400)
                                if detected and detected != self.current_brawler:
                                    # If the detected brawler has the SAME base HP as configured,
                                    # it's just an ambiguity (e.g. mortis/nita/poco all 3800,
                                    # pam/carl/bibi all 4200).  Trust the config.
                                    detected_hp = self.brawlers_info.get(detected, {}).get('health', 0)
                                    if detected_hp == expected_hp:
                                        # Same HP -> ambiguous, keep configured brawler
                                        pass
                                    else:
                                        print(f"[AUTO-DETECT] Brawler mismatch! Configured={self.current_brawler}, "
                                              f"HP suggests={detected} (base HP: {detected_hp})")
                                        self.current_brawler = detected
                                        self.brawler_ranges = None  # Force reload
                                        print(f"[AUTO-DETECT] Switched to brawler: {detected}")
                                self._brawler_detected = True
                            except Exception as e:
                                print(f"[AUTO-DETECT] Error: {e}")
                                self._brawler_detected = True
                else:
                    # Detection failed - track failures
                    self._player_hp_fail_count += 1
                    # After ~2 seconds of consecutive failures, lower confidence
                    if self._player_hp_fail_count >= 13:
                        self._hp_confidence_player = max(0.1, self._hp_confidence_player * 0.9)

            # Derive HP state from value + confidence + data age (used by BT safety gating)
            self._hp_data_age_player = max(0.0, current_time - self._last_valid_player_hp_time)
            conf = float(getattr(self, '_hp_confidence_player', 0.0) or 0.0)
            hp_val = int(max(1, min(100, getattr(self, 'player_hp_percent', 100) or 100)))

            if self._hp_data_age_player > self._hp_stale_timeout or conf < self._hp_conf_low_threshold:
                self._hp_state_player = "unknown"
            else:
                prev_state = getattr(self, '_hp_state_player', 'healthy')
                if prev_state == "critical":
                    self._hp_state_player = "critical" if hp_val < self._hp_critical_exit else (
                        "warning" if hp_val < self._hp_warning_exit else "healthy"
                    )
                elif prev_state == "warning":
                    if hp_val <= self._hp_critical_enter:
                        self._hp_state_player = "critical"
                    elif hp_val < self._hp_warning_exit:
                        self._hp_state_player = "warning"
                    else:
                        self._hp_state_player = "healthy"
                else:
                    if hp_val <= self._hp_critical_enter:
                        self._hp_state_player = "critical"
                    elif hp_val <= self._hp_warning_enter:
                        self._hp_state_player = "warning"
                    else:
                        self._hp_state_player = "healthy"

            # Track target info + enemy memory
            enemies = data.get('enemy') or []
            teammates = data.get('teammate') or []

            # === FILTER GHOST DETECTIONS: remove enemies near recent kill positions ===
            # YOLO may briefly detect the death animation or loot as "enemy"
            if self._enemy_death_positions and enemies:
                now_death_check = time.time()
                # Prune expired death positions
                self._enemy_death_positions = [
                    (x, y, t) for x, y, t in self._enemy_death_positions
                    if now_death_check - t < self._enemy_death_cooldown
                ]
                filtered = []
                for en in enemies:
                    en_cx = (en[0] + en[2]) / 2
                    en_cy = (en[1] + en[3]) / 2
                    if not self._is_near_death_position(en_cx, en_cy):
                        filtered.append(en)
                    else:
                        print(f"[GHOST] Ignoring detection at ({int(en_cx)}, {int(en_cy)}) - near kill site")
                enemies = filtered

            # Write filtered enemies back so BT receives the cleaned list
            data['enemy'] = enemies

            self.target_info['n_enemies'] = len(enemies)
            self.target_info['n_teammates'] = len(teammates)

            # --- NUMBER ADVANTAGE DETECTION ---
            n_enemies = len(enemies)
            if n_enemies > 0 and n_enemies < self._expected_enemy_count:
                if not self._number_advantage_active:
                    self._enemy_count_drop_time = current_time
                    self._number_advantage_active = True
                    self._number_advantage_until = current_time + 6.0  # 6s window
                    print(f"[ADVANTAGE] {n_enemies}/{self._expected_enemy_count} enemies - PUSHING!")
            elif n_enemies >= self._expected_enemy_count:
                if self._number_advantage_active:
                    self._number_advantage_active = False
                    print("[ADVANTAGE] All enemies back - normal mode.")
            # Auto-expire number advantage
            if self._number_advantage_active and current_time > self._number_advantage_until:
                self._number_advantage_active = False

            # --- LANE ASSIGNMENT (3v3 modes, every 5s) ---
            if (not self.is_showdown and teammates and len(teammates) >= 1
                    and current_time - self._last_lane_check > 5.0):
                self._last_lane_check = current_time
                if 'player' in data and data['player']:
                    pp = self.get_player_pos(data['player'][0])
                    self._assign_lane(pp, teammates)

            # --- ENEMY RELOAD WINDOW DETECTION ---
            if enemies and 'player' in data and data['player']:
                pp = self.get_player_pos(data['player'][0])
                hp_dropped = self.player_hp_percent < self._prev_player_hp - 3
                self._detect_enemy_fired(enemies, pp, hp_dropped)
                self._prev_player_hp = self.player_hp_percent

                # --- REACTIVE DODGE TRIGGER (on HP drop = we got hit) ---
                if hp_dropped and enemies:
                    closest_en = min(enemies,
                        key=lambda e: self.get_distance(self.get_enemy_pos(e), pp))
                    self._trigger_reactive_dodge(pp, self.get_enemy_pos(closest_en))

            # --- TEAMMATE HP TRACKING (for support brawler heal priority) ---
            if teammates and len(teammates) <= 3:
                tm_hp_list = []
                for idx_tm, tm in enumerate(teammates):
                    tm_hp, tm_conf = self._hp_estimator.estimate(
                        frame_for_hp, tm, is_player=True,
                        entity_key=f"teammate_{idx_tm}")
                    if tm_hp < 0:
                        # Fallback to legacy
                        tm_hp = self.estimate_hp_from_bar(frame_for_hp, tm, is_player=True)
                    tm_cx = (tm[0] + tm[2]) / 2
                    tm_cy = (tm[1] + tm[3]) / 2
                    hp = tm_hp if tm_hp >= 0 else 100
                    tm_hp_list.append((tm_cx, tm_cy, hp))
                self._teammate_hp_data = tm_hp_list
                self._lowest_teammate_hp = min((h for _, _, h in tm_hp_list), default=100)
            else:
                self._teammate_hp_data = []
                self._lowest_teammate_hp = 100

            # --- TEAMMATE DEATH GAP COVER (3v3 modes) ---
            if not self.is_showdown and 'player' in data and data['player']:
                pp = self.get_player_pos(data['player'][0])
                tm_coords = []
                for tm in teammates:
                    tm_coords.append(((tm[0]+tm[2])/2, (tm[1]+tm[3])/2))
                gap_target = self._handle_teammate_death_gap(tm_coords, pp, self.game_mode_name)
                if gap_target is not None:
                    self._teammate_death_pos = gap_target
                    # The gap target will be used as a movement bias in get_movement

            if enemies and 'player' in data and data['player']:
                player_pos = self.get_player_pos(data['player'][0])
                walls = data.get('wall', [])

                # Update enemy position memory (so we know where they disappeared)
                self._update_enemy_memory(enemies, player_pos)

                # Enemy reappeared - they're NOT dead anymore
                self._enemy_was_low_hp_when_lost = False

                # Sort enemies by distance (closest first)
                enemies_sorted = sorted(enemies,
                    key=lambda e: self.get_distance(self.get_enemy_pos(e), player_pos))
                closest_enemy = enemies_sorted[0]
                target_pos = self.get_enemy_pos(closest_enemy)
                target_dist = self.get_distance(target_pos, player_pos)

                # --- Detect target switch: if closest enemy jumped far, reset HPEstimator ---
                new_center = ((closest_enemy[0]+closest_enemy[2])/2,
                              (closest_enemy[1]+closest_enemy[3])/2)
                if self._last_closest_enemy_center is not None:
                    dx = abs(new_center[0] - self._last_closest_enemy_center[0])
                    dy = abs(new_center[1] - self._last_closest_enemy_center[1])
                    if dx > 120 or dy > 120:
                        # Target switched to a different enemy - clear smoothing
                        self._hp_estimator.clear_entity("enemy_0")
                self._last_closest_enemy_center = new_center

                # === PER-ENEMY HP DETECTION (all visible enemies) ===
                # Only run when HP check is due (throttled like main HP)
                if should_check_hp:
                    active_keys = {"player"}  # keys used this frame
                    # Include teammate keys so they aren't purged
                    for _ti in range(len(teammates) if teammates else 0):
                        active_keys.add(f"teammate_{_ti}")
                    new_per_enemy_hp = {}
                    
                    for ei, en_bbox in enumerate(enemies_sorted):
                        entity_key = f"enemy_{ei}"
                        active_keys.add(entity_key)

                        # === FAST bar-based detection ONLY (OCR disabled for speed) ===
                        en_hp, en_conf = self._hp_estimator.estimate(
                            frame_for_hp, en_bbox,
                            is_player=False, entity_key=entity_key)

                        # Fallback to legacy if HPEstimator fails
                        if en_hp < 0:
                            en_debug = {}
                            en_hp = self.estimate_hp_from_bar(
                                frame_for_hp, en_bbox,
                                is_player=False, debug_info=en_debug)
                            en_conf = 0.3 if en_hp >= 0 else 0.0

                        ecx = (en_bbox[0] + en_bbox[2]) / 2
                        ecy = (en_bbox[1] + en_bbox[3]) / 2
                        hp_val = max(1, min(100, en_hp)) if en_hp >= 0 else -1
                        new_per_enemy_hp[f"{ecx:.0f},{ecy:.0f}"] = (hp_val, en_conf, current_time)

                        # First enemy = closest = our main target
                        if ei == 0:
                            self._hp_confidence_enemy = en_conf
                            if en_hp >= 0:
                                old_ehp = self.enemy_hp_percent
                                if en_hp < (old_ehp if old_ehp > 0 else 100) - 5:
                                    self._last_enemy_damage_time = current_time
                                self.enemy_hp_percent = max(1, min(100, en_hp))
                                self._last_valid_enemy_hp = self.enemy_hp_percent
                                self._enemy_hp_fail_count = 0
                                self.last_enemy_hp_update = current_time
                                self._enemy_hp_before_disappear = self.enemy_hp_percent if self.enemy_hp_percent > 0 else 100
                            else:
                                self._enemy_hp_fail_count += 1
                                # After 1s without valid reading -> mark as unknown
                                if current_time - self.last_enemy_hp_update > 1.0:
                                    self.enemy_hp_percent = -1

                    # Store per-enemy HP + purge stale HPEstimator entries
                    self._per_enemy_hp = new_per_enemy_hp
                    self._hp_estimator.clear_stale_entities(active_keys)

                # Hittable check
                hittable = self.is_enemy_hittable(player_pos, target_pos, walls, "attack")

                # Update target info
                self.target_info['bbox'] = closest_enemy
                self.target_info['distance'] = int(target_dist)
                self.target_info['hp'] = self.enemy_hp_percent
                self.target_info['hittable'] = hittable

                # OCR enemy name (throttled - expensive operation, every 2s)
                if current_time - self._last_name_ocr_time > 2.0:
                    self._last_name_ocr_time = current_time

                    # Read own player name once (to filter from enemy OCR)
                    if not self._player_name_read and 'player' in data and data['player']:
                        player_bbox = data['player'][0]
                        pname = self.read_name_above_bbox(frame_for_hp, player_bbox)
                        if pname:
                            self._player_name = pname.lower().strip()
                            self._player_name_read = True

                    name = self.read_name_above_bbox(frame_for_hp, closest_enemy)
                    if name:
                        # Filter: reject if it matches our own player name
                        name_lower = name.lower().strip()
                        own = self._player_name or ""
                        if own and (name_lower == own
                                    or name_lower in own
                                    or own in name_lower):
                            pass  # Skip - this is our own name
                        else:
                            self.target_info['name'] = name

                # Also read names for ALL enemies (for overlay) - but only along with name OCR
                if current_time - self._last_name_ocr_time < 0.1:
                    # We just did OCR this frame, tag all enemies
                    enemy_names = {}
                    for i, en in enumerate(enemies):
                        if en == closest_enemy:
                            enemy_names[i] = self.target_info.get('name', None)
                        # Don't OCR non-target enemies (too expensive)
                    self.target_info['_all_enemy_names'] = enemy_names
            else:
                # No enemies visible - check if enemy DIED or just disappeared
                time_since_last_hp = current_time - self.last_enemy_hp_update

                # === ENEMY DEATH DETECTION ===
                # In fast-paced modes (Knockout, etc.), enemies can go from
                # 80%+ to 0 between frames - burst damage exceeds what HP
                # estimation can track.  Use generous thresholds so kills
                # aren't missed.  False-positive risk is low because the
                # time window (0.3–3 s) already filters random de-spawns.
                recently_attacked = (current_time - self.last_attack_time) < 2.5
                
                # knockout/elimination modes: disappearance = probable kill
                # Enemies don't respawn within a round.  HP estimation can't
                # keep up with burst damage (last read often 90-100%).  If we
                # attacked recently and the enemy vanished, count it as a kill
                # regardless of last-read HP.
                _elimination_mode = self.game_mode_name in (
                    'knockout', 'wipeout', 'duels', 'bounty',
                    'soloshowdown', 'duoshowdown', 'showdown')
                # Distance guard: only count elimination-mode kill if enemy
                # was within plausible firing range on last visible frame
                _last_dist = self.target_info.get('distance', 0)
                try:
                    _atk_range = self.get_brawler_range(self.current_brawler)[1]
                except Exception:
                    _atk_range = 442
                _in_fire_range = _last_dist > 0 and _last_dist < _atk_range * 1.5
                if _elimination_mode and recently_attacked and _in_fire_range:
                    kill_hp_threshold = 101  # any HP: vanish after attack = kill
                else:
                    kill_hp_threshold = 80 if recently_attacked else 60
                if (not self._enemy_was_low_hp_when_lost
                        and self._enemy_hp_before_disappear < kill_hp_threshold
                        and time_since_last_hp > 0.3
                        and time_since_last_hp < 3.0):
                    self._enemy_was_low_hp_when_lost = True
                    # Register kill at last known enemy position
                    last_pos = self._get_last_known_enemy_pos()
                    if last_pos:
                        self._register_enemy_kill(last_pos[0], last_pos[1])
                    else:
                        # No position data, but still mark as killed
                        print("[KILL] Enemy likely killed (no position data)")
                        self._last_enemy_kill_time = current_time
                        self._enemies_killed_this_match += 1
                    # Set HP to 0 (dead), don't reset to 100
                    self.enemy_hp_percent = 0
                    self._enemy_hp_before_disappear = 0
                elif self._enemy_was_low_hp_when_lost:
                    # Already marked as killed - keep HP at 0 until new enemy appears
                    self.enemy_hp_percent = 0
                elif time_since_last_hp > 1.5:
                    # Enemy vanished with HIGH hp - they retreated or we lost sight.
                    # Set to unknown (-1): don't hold stale data, don't assume full.
                    self.enemy_hp_percent = -1

                # Clear per-enemy HP (no enemies visible)
                self._per_enemy_hp = {}
                self._last_closest_enemy_center = None

                # Reset target info to avoid stale data
                self.target_info['bbox'] = None
                self.target_info['distance'] = 0
                self.target_info['hittable'] = False
                self.target_info['name'] = None
                self.target_info['hp'] = self.enemy_hp_percent if self._enemy_was_low_hp_when_lost else -1
                self.target_info['n_enemies'] = 0
                self.target_info['n_teammates'] = len(teammates) if teammates else 0

            # Regen status - PROPER check: no shooting AND no damage taken for REGEN_DELAY
            time_since_attack = current_time - self.last_attack_time
            time_since_damage = current_time - self._last_damage_taken_time
            # HP regen only starts if BOTH conditions are met
            can_regen = (time_since_attack >= self.REGEN_DELAY and 
                         time_since_damage >= self.REGEN_DELAY and 
                         self.player_hp_percent < 95)
            self.is_regenerating = can_regen
        except Exception:
            pass

        movement = self.loop(brawler, data, current_time)

        # Show debug overlay with detections + stats
        self._show_debug_overlay(frame, data, movement, brawler,
                                 stats_info=getattr(self, '_stats_info', None))

        # --- Update transparent visual tracking overlay ---
        self._update_visual_overlay(data, movement, brawler)

    def _update_visual_overlay(self, data, movement, brawler):
        """Push latest game data to the transparent visual overlay."""
        if self.visual_overlay is None:
            return
        try:
            player_pos = None
            enemies = data.get('enemy') or []
            walls = data.get('wall') or []
            teammates = data.get('teammate') or []
            target_bbox = self.target_info.get('bbox')
            frame_w = self.window_controller.width
            frame_h = self.window_controller.height

            player_bbox = None
            if 'player' in data and data['player']:
                player_bbox = data['player'][0]
                player_pos = self.get_player_pos(player_bbox)

            # Brawler ranges (already scaled to frame resolution)
            brawler_ranges = None
            try:
                brawler_ranges = self.get_brawler_range(brawler)
            except Exception:
                pass

            # Build hittable map for all enemies
            hittable_map = {}
            if player_pos and enemies:
                for i, enemy in enumerate(enemies):
                    enemy_pos = self.get_enemy_pos(enemy)
                    hittable_map[i] = self.is_enemy_hittable(
                        player_pos, enemy_pos, walls, "attack"
                    )

            # Gather brawler stats for the HUD panel
            brawler_info = self.brawlers_info.get(brawler, {}) if brawler else {}

            self.visual_overlay.update(
                player_pos=player_pos,
                player_bbox=player_bbox,
                enemies=enemies,
                walls=walls,
                brawler_ranges=brawler_ranges,
                movement=movement,
                is_super_ready=getattr(self, 'is_super_ready', False),
                teammates=teammates,
                hittable_map=hittable_map,
                target_bbox=target_bbox,
                frame_w=frame_w,
                frame_h=frame_h,
                is_dead=getattr(self, '_is_dead', False),
                game_state=getattr(self, '_stats_info', {}).get('state', 'match'),
                brawler_name=brawler or "",
                brawler_info=brawler_info,
                storm_center=self._storm_center,
                storm_radius=self._storm_radius,
                gas_active=self._gas_active,
                gas_density_map=getattr(self, '_gas_density_map', None),
                in_storm=self._in_storm,
                choke_points=self._choke_points,
                bushes=getattr(self, 'last_bush_data', []),
                # nEW overlay data
                player_hp=getattr(self, 'player_hp_percent', 100),
                enemy_hp=getattr(self, 'enemy_hp_percent', -1),
                per_enemy_hp=dict(getattr(self, '_per_enemy_hp', {})),
                hp_confidence_player=getattr(self, '_hp_confidence_player', 1.0),
                hp_confidence_enemy=getattr(self, '_hp_confidence_enemy', 1.0),
                ammo=getattr(self, '_ammo', 3),
                max_ammo=getattr(self, '_max_ammo', 3),
                decision_reason=getattr(self, 'last_decision_reason', ''),
                match_phase=getattr(self, '_match_phase', 'early'),
                our_score=getattr(self, '_our_score', 0),
                their_score=getattr(self, '_their_score', 0),
                death_count=getattr(self, '_death_count', 0),
                kills=getattr(self, '_enemies_killed_this_match', 0),
                target_info=dict(self.target_info) if hasattr(self, 'target_info') else {},
                last_known_enemies=list(getattr(self, '_last_known_enemies', [])),
                behavior_flags={
                    'disengage': getattr(self, '_disengage_active', False),
                    'number_advantage': getattr(self, '_number_advantage_active', False),
                    'stutter_step': getattr(self, '_stutter_step_active', False),
                    'peek': getattr(self, '_peek_phase', 'idle') != 'idle',
                },
                # extended overlay data
                enemy_velocity=getattr(self, '_enemy_velocity_smooth', (0, 0)),
                enemy_move_dir=getattr(self, '_enemy_move_direction', 'none'),
                teammate_hp_data=list(getattr(self, '_teammate_hp_data', [])),
                is_gadget_ready=getattr(self, 'is_gadget_ready', False),
                is_hypercharge_ready=getattr(self, 'is_hypercharge_ready', False),
                hold_super=getattr(self, '_hold_super', False),
                respawn_shield_active=getattr(self, '_respawn_shield_active', False),
                respawn_shield_until=getattr(self, '_respawn_shield_until', 0.0),
                is_regenerating=getattr(self, 'is_regenerating', False),
                match_start_time=getattr(self, '_match_start_time', 0.0),
                game_mode_name=getattr(self, 'game_mode_name', ''),
                assigned_lane=getattr(self, '_assigned_lane', ''),
                lane_center=getattr(self, '_lane_center', None),
                objective_pos=getattr(self, 'objective_pos', None),
                spawn_side=getattr(self, '_spawn_side', ''),
                aggression_modifier=getattr(self, 'aggression_modifier', 1.0),
                destroyed_wall_zones=list(getattr(self, '_destroyed_wall_zones', [])),
                patrol_phase=getattr(self, '_patrol_phase', 'idle'),
                no_enemy_duration=getattr(self, '_no_enemy_duration', 0.0),
                enemy_reload_window=getattr(self, '_enemy_in_reload_window', False),
                stuck_level=getattr(self, '_stuck_escalation', 0),
                bot_ips=getattr(self, '_stats_info', {}).get('ips', 0),
                pathfinder_path=(self._path_planner.current_path_px
                                 if getattr(self, '_path_planner', None) else None),
            )
            
            # Pass reward/adaptive data from BT combat to overlay
            bt = getattr(self, '_bt_combat', None)
            if bt and self.visual_overlay:
                self.visual_overlay._last_reward_score = bt._get_reward_score()
                self.visual_overlay._last_aggr_mod = bt._get_adaptive_aggression()
        except Exception:
            pass

    def _show_debug_overlay(self, frame, data, movement, brawler, stats_info=None):
        """Show a live debug window with bounding boxes, target info, and stats."""
        try:
            now = time.time()
            if now - self._last_debug_time < 0.5:
                return
            self._last_debug_time = now

            debug_img = np.array(frame)  # Need a mutable copy for drawing
            debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
            h, w = debug_img.shape[:2]
            scale = 720 / w
            disp_w = 720
            disp_h = int(h * scale)
            debug_img = cv2.resize(debug_img, (disp_w, disp_h))

            ti = self.target_info  # shorthand
            target_bbox = ti.get('bbox')

            if data:
                # --- Draw player (bright green, thick) ---
                for player in data.get('player', []):
                    x1, y1, x2, y2 = [int(v * scale) for v in player]
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    hp_text = f"YOU {self.player_hp_percent}%"
                    cv2.putText(debug_img, hp_text, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

                    # --- Draw HP bar search area (yellow dashed rect for debugging) ---
                    hp_dbg = getattr(self, '_hp_debug_info', {})
                    sr = hp_dbg.get('search_rect')
                    if sr:
                        sx1, sy1, sx2, sy2 = [int(v * scale) for v in sr]
                        cv2.rectangle(debug_img, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
                        # Draw detected bar row if found
                        bar_row = hp_dbg.get('bar_row')
                        bar_w = hp_dbg.get('bar_width', 0)
                        if bar_row is not None:
                            br_y = int(bar_row * scale)
                            cv2.line(debug_img, (sx1, br_y), (sx1 + int(bar_w * scale), br_y),
                                     (0, 255, 0), 2)
                        # Status text
                        status = hp_dbg.get('status', '?')
                        cv2.putText(debug_img, f"HP:{status}", (x2 + 4, y1 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1, cv2.LINE_AA)

                # --- Draw enemies (red = normal, magenta = TARGET) ---
                enemies = data.get('enemy') or []
                for i, enemy in enumerate(enemies):
                    x1, y1, x2, y2 = [int(v * scale) for v in enemy]
                    is_target = (target_bbox is not None and
                                 abs(enemy[0] - target_bbox[0]) < 5 and
                                 abs(enemy[1] - target_bbox[1]) < 5)

                    if is_target:
                        # TARGET enemy - thick magenta box + crosshair
                        color = (255, 0, 255)
                        cv2.rectangle(debug_img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, 3)
                        # Crosshair lines
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.line(debug_img, (cx - 15, cy), (cx + 15, cy), color, 1)
                        cv2.line(debug_img, (cx, cy - 15), (cx, cy + 15), color, 1)
                        # Target label with name
                        name = ti.get('name', None)
                        dist = ti.get('distance', 0)
                        hp = ti.get('hp', -1)
                        hittable = ti.get('hittable', False)
                        label = "TARGET"
                        if name:
                            label = f">> {name.upper()}"
                        hp_str = f" {hp}%" if hp >= 0 else ""
                        dist_str = f" {dist}px"
                        hit_str = " HIT" if hittable else " BLOCKED"
                        # Name above box
                        cv2.putText(debug_img, label, (x1, y1 - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
                        # HP + distance below name
                        info_text = f"HP:{hp_str} D:{dist_str}{hit_str}"
                        cv2.putText(debug_img, info_text, (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 200, 255), 1, cv2.LINE_AA)
                        # Draw line from player to target
                        if data.get('player'):
                            px1, py1, px2, py2 = [int(v * scale) for v in data['player'][0]]
                            pcx, pcy = (px1 + px2) // 2, (py1 + py2) // 2
                            line_color = (0, 255, 0) if hittable else (0, 0, 255)
                            cv2.line(debug_img, (pcx, pcy), (cx, cy), line_color, 1, cv2.LINE_AA)
                    else:
                        # Non-target enemy - thin red box + per-enemy HP
                        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        # Look up per-enemy HP
                        _ecx = (enemy[0] + enemy[2]) / 2
                        _ecy = (enemy[1] + enemy[3]) / 2
                        _ehp = self._lookup_per_enemy_hp(_ecx, _ecy)
                        if _ehp >= 0:
                            _lbl = f"ENEMY {_ehp}%"
                            _clr = (0, 220, 100) if _ehp > 50 else (0, 220, 220) if _ehp > 25 else (80, 80, 255)
                        else:
                            _lbl = "ENEMY"
                            _clr = (0, 0, 255)
                        cv2.putText(debug_img, _lbl, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, _clr, 1, cv2.LINE_AA)

                # --- Draw teammates (cyan) ---
                for teammate in data.get('teammate', []):
                    x1, y1, x2, y2 = [int(v * scale) for v in teammate]
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.putText(debug_img, "ALLY", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

                # --- Draw walls (subtle gray) ---
                for wall in data.get('wall', []):
                    x1, y1, x2, y2 = [int(v * scale) for v in wall]
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (80, 80, 80), 1)

            # === STATS PANEL (right side) ===
            panel_w = 270
            combined_w = disp_w + panel_w
            combined = np.full((disp_h, combined_w, 3), (25, 25, 25), dtype=np.uint8)
            combined[:, :disp_w] = debug_img

            x0 = disp_w + 8
            y_off = 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = 0.38
            white = (220, 220, 220)
            gray = (120, 120, 120)
            green = (0, 220, 100)
            red = (80, 80, 255)
            yellow = (0, 220, 220)
            orange = (0, 140, 255)
            magenta = (255, 80, 255)
            cyan = (255, 200, 0)

            def put(text, color=white, size=fs):
                nonlocal y_off
                cv2.putText(combined, text, (x0, y_off), font, size, color, 1, cv2.LINE_AA)
                y_off += 16

            def header(text):
                nonlocal y_off
                y_off += 2
                cv2.rectangle(combined, (disp_w, y_off - 11), (combined_w, y_off + 5), (45, 45, 45), -1)
                cv2.putText(combined, text, (x0, y_off + 1), font, 0.42, orange, 1, cv2.LINE_AA)
                y_off += 18

            def hp_bar(label, hp_val, healthy_color, size=10):
                nonlocal y_off
                filled = max(0, min(size, int(hp_val / 100 * size)))
                bar = chr(9608) * filled + chr(9617) * (size - filled)
                hp_col = healthy_color if hp_val > 50 else yellow if hp_val > 25 else red
                text = f" {label} [{bar}] {hp_val:>3}%"
                cv2.putText(combined, text, (x0, y_off), font, fs, hp_col, 1, cv2.LINE_AA)
                y_off += 16

            # --- BRAWLER HEADER ---
            header(f"{brawler.upper()} | {self.game_mode_name.upper()}")
            put(f" Mv: {movement}", yellow)

            # --- TARGET INFO (prominent section) ---
            header("TARGET")
            t_name = ti.get('name')
            t_dist = ti.get('distance', 0)
            t_hp = ti.get('hp', -1)
            t_hit = ti.get('hittable', False)
            n_enemies = ti.get('n_enemies', 0)
            n_teammates = ti.get('n_teammates', 0)

            if n_enemies > 0 and ti.get('bbox') is not None:
                # Target name (big + colored)
                if t_name:
                    put(f" >> {t_name.upper()}", magenta, 0.45)
                else:
                    put(f" >> UNKNOWN ENEMY", magenta, 0.42)
                # Target details
                hit_text = "CLEAR" if t_hit else "BLOCKED"
                hit_color = green if t_hit else red
                put(f" Dist: {t_dist}px  |  {hit_text}", hit_color)
                if t_hp >= 0:
                    hp_bar("EN", t_hp, red)
                    _econf = getattr(self, '_hp_confidence_enemy', 0)
                    put(f"   conf={_econf:.2f}", gray)
                else:
                    put(f" EN HP: scanning...", gray)
                if n_enemies > 1:
                    put(f" +{n_enemies - 1} more enemies", gray)
            else:
                put(f" No enemies visible", gray)

            put(f" Allies: {n_teammates}", cyan if n_teammates > 0 else gray)

            # --- PLAYER HP ---
            _php = max(0, self.player_hp_percent)
            hp_bar("HP", _php, green)
            _pconf = getattr(self, '_hp_confidence_player', 0)
            put(f"   conf={_pconf:.2f}", gray)
            if self.is_regenerating:
                put(f" REGENERATING...", green)

            # --- ABILITIES ---
            header("ABILITIES")
            g_cd = max(0, self.GADGET_USE_COOLDOWN - (now - self.time_since_gadget_used))
            s_cd = max(0, self.SUPER_USE_COOLDOWN - (now - self.time_since_super_used))
            h_cd = max(0, self.HYPERCHARGE_USE_COOLDOWN - (now - self.time_since_hypercharge_used))
            put(f" Gadget:  {'READY' if self.is_gadget_ready and g_cd <= 0 else f'cd {g_cd:.1f}s' if g_cd > 0 else 'off'}",
                green if self.is_gadget_ready and g_cd <= 0 else gray)
            put(f" Super:   {'READY' if self.is_super_ready and s_cd <= 0 else f'cd {s_cd:.1f}s' if s_cd > 0 else 'off'}",
                green if self.is_super_ready and s_cd <= 0 else gray)
            put(f" Hyper:   {'READY' if self.is_hypercharge_ready and h_cd <= 0 else f'cd {h_cd:.1f}s' if h_cd > 0 else 'off'}",
                green if self.is_hypercharge_ready and h_cd <= 0 else gray)

            # --- SESSION + TROPHIES ---
            if stats_info:
                header("SESSION")
                put(f" IPS: {stats_info.get('ips', 0):.1f}", yellow)
                elapsed = now - stats_info.get('start_time', now)
                m, s = divmod(int(elapsed), 60)
                h_t, m = divmod(m, 60)
                put(f" Time: {h_t:02d}:{m:02d}:{s:02d}")
                put(f" State: {stats_info.get('state', '?').upper()}",
                    green if stats_info.get('state') == 'match' else yellow)

                tobs = stats_info.get('trophy_observer')
                if tobs:
                    header("TROPHIES")
                    tr = tobs.current_trophies
                    target = stats_info.get('target', '?')
                    put(f" {tr} -> {target}", green if tr else white)
                    put(f" W:{tobs.current_wins} Str:{tobs.win_streak} Games:{tobs.match_counter}")
                    bname = brawler.lower()
                    hist = tobs.match_history.get(bname, {})
                    v = hist.get('victory', 0)
                    d_val = hist.get('defeat', 0)
                    dr = hist.get('draw', 0)
                    total = v + d_val + dr
                    if total > 0:
                        wr = v / total * 100
                        put(f" V:{v} D:{d_val} Dr:{dr} (WR:{wr:.0f}%)",
                            green if wr >= 50 else red)

            # Anti-stuck
            if self.fix_movement_keys.get('toggled'):
                put(" !! ANTI-STUCK", red)

            if not self._debug_window_created:
                cv2.namedWindow("PylaAI Debug", cv2.WINDOW_AUTOSIZE)
                cv2.moveWindow("PylaAI Debug", 50, 50)
                self._debug_window_created = True
                try:
                    import ctypes
                    user32 = ctypes.windll.user32
                    hwnd = user32.FindWindowW(None, "PylaAI Debug")
                    if hwnd:
                        user32.SetForegroundWindow(hwnd)
                        user32.BringWindowToTop(hwnd)
                        user32.ShowWindow(hwnd, 9)
                except Exception:
                    pass
            cv2.imshow("PylaAI Debug", combined)
            cv2.waitKey(1)
        except Exception:
            pass

    def generate_visualization(self, output_filename='visualization.mp4'):
        import cv2
        import numpy as np

        frame_size = (1920, 1080)  # Adjust as needed
        fps = 10

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
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
