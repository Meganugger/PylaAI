from utils import load_toml_as_dict, save_dict_as_toml


CONFIG_SPECS = {
    "bot": {
        "path": "cfg/bot_config.toml",
        "defaults": {
            "gamemode_type": 3,
            "gamemode": "brawlball",
            "bot_uses_gadgets": "yes",
            "smart_trophy_farm": "no",
            "trophy_farm_mode": "manual",
            "trophy_farm_target": 500,
            "trophy_farm_strategy": "lowest_first",
            "trophy_farm_excluded": [],
            "minimum_movement_delay": 0.4,
            "wall_detection_confidence": 0.9,
            "entity_detection_confidence": 0.6,
            "unstuck_movement_delay": 3.0,
            "unstuck_movement_hold_time": 1.5,
            "post_play_again_match_hold_seconds": 2.0,
            "brawl_ball_patrol_hold_time": 2.8,
            "brawl_ball_patrol_arrival_radius": 95.0,
            "brawl_ball_opening_seconds": 6.0,
            "brawl_ball_opening_lock_seconds": 4.5,
            "brawl_ball_opening_hold_seconds": 1.4,
            "analog_brawlball_opening_hold_time": 0.9,
            "brawl_ball_spawn_escape_seconds": 8.0,
            "brawl_ball_spawn_escape_min_seconds": 5.8,
            "brawl_ball_spawn_escape_nudge_after": 6.2,
            "brawl_ball_spawn_escape_nudge_interval": 1.8,
            "brawl_ball_spawn_escape_uncertain_seconds": 10.0,
            "brawl_ball_spawn_escape_extended_seconds": 14.0,
            "movement_watchdog_seconds": 0.65,
            "active_tick_no_action_refresh_seconds": 0.32,
            "target_confirmation_max_age": 0.55,
            "ability_target_confirmation_max_age": 0.35,
            "brawl_ball_no_fire_spawn_seconds": 2.2,
            "wall_blocked_escape_seconds": 0.7,
            "wall_blocked_escape_ticks": 2,
            "wall_escape_nudge_seconds": 0.45,
            "wall_escape_return_seconds": 0.65,
            "analog_brawlball_lane_hold_time": 1.15,
            "analog_spawn_escape_hold_time": 1.4,
            "analog_corner_escape_hold_time": 0.85,
            "hypercharge_cooldown": 2.5,
            "showdown_team_behavior": "follow",
            "showdown_roam_hold_seconds": 2.2,
            "showdown_border_hold_seconds": 2.4,
            "analog_showdown_roam_hold_time": 0.65,
            "analog_showdown_border_hold_time": 0.75,
            "analog_showdown_wall_escape_hold_time": 0.50,
            "battle_debug_verbose": "no",
        },
        "legacy_keys": set(),
    },
    "time": {
        "path": "cfg/time_tresholds.toml",
        "defaults": {
            "state_check": 3,
            "no_detections": 10,
            "idle": 10,
            "super": 0.1,
            "gadget": 0.5,
            "hypercharge": 2,
            "wall_detection": 0.2,
            "no_detection_proceed": 6.5,
            "check_if_brawl_stars_crashed": 10,
        },
        "legacy_keys": {"game_start"},
    },
    "general": {
        "path": "cfg/general_config.toml",
        "defaults": {
            "max_ips": 60,
            "target_ips": 60,
            "super_debug": "yes",
            "input_debug": "yes",
            "preferred_backend": "auto",
            "cpu_or_gpu": "auto",
            "onnx_allow_cuda_with_missing_nvrtc": True,
            "long_press_star_drop": "no",
            "trophies_multiplier": 1,
            "run_for_minutes": 600,
            "emulator_port": 5037,
            "brawlstars_api_key": "",
            "brawlstars_player_tag": "",
            "auto_push_target_trophies": 1000,
            "personal_webhook": "",
            "discord_id": "",
            "brawlstars_package": "com.supercell.brawlstars",
            "api_base_url": "localhost",
            "current_emulator": "LDPlayer",
            "scrcpy_max_fps": 60,
            "scrcpy_max_width": 960,
            "scrcpy_bitrate": 3000000,
            "joystick_refresh_seconds": 0.35,
            "joystick_repress_seconds": 1.8,
            "joystick_down_move_delay": 0.012,
            "map_orientation": "vertical",
            "performance_profile": "balanced",
            "auto_update_checks": "no",
            "auto_update_ignored_sha": "",
        },
        "legacy_keys": set(),
    },
    "match_history": {
        "path": "cfg/match_history.toml",
        "defaults": {},
        "legacy_keys": set(),
    },
}


def sanitize_config(config_name, config_data):
    spec = CONFIG_SPECS[config_name]
    sanitized = dict(config_data)
    for key, default_value in spec["defaults"].items():
        sanitized.setdefault(key, default_value)
    for key in spec["legacy_keys"]:
        sanitized.pop(key, None)
    return sanitized


def load_config(config_name):
    spec = CONFIG_SPECS[config_name]
    loaded = load_toml_as_dict(spec["path"])
    return sanitize_config(config_name, loaded)


def save_config(config_name, config_data):
    spec = CONFIG_SPECS[config_name]
    sanitized = sanitize_config(config_name, config_data)
    save_dict_as_toml(sanitized, spec["path"])
    return sanitized


def update_config_value(config_name, config_data, key, value):
    config_data[key] = value
    return save_config(config_name, config_data)
