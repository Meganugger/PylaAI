from utils import load_toml_as_dict, save_dict_as_toml


CONFIG_SPECS = {
    "bot": {
        "path": "cfg/bot_config.toml",
        "defaults": {
            "gamemode_type": 3,
            "gamemode": "knockout",
            "bot_uses_gadgets": "yes",
            "minimum_movement_delay": 0.08,
            "wall_detection_confidence": 0.9,
            "entity_detection_confidence": 0.6,
            "unstuck_movement_delay": 1.5,
            "unstuck_movement_hold_time": 0.8,
            "seconds_to_hold_attack_after_reaching_max": 1.5,
            "play_again_on_win": "no",
            "smart_trophy_farm": "yes",
            "trophy_farm_target": 500,
            "trophy_farm_strategy": "lowest_first",
            "trophy_farm_excluded": [],
            "quest_farm_enabled": "no",
            "quest_farm_mode": "games",
            "quest_farm_excluded": [],
            "dynamic_rotation_enabled": "no",
            "dynamic_rotation_every": 20,
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
        },
        "legacy_keys": {"game_start"},
    },
    "general": {
        "path": "cfg/general_config.toml",
        "defaults": {
            "max_ips": "auto",
            "super_debug": "yes",
            "preferred_backend": "auto",
            "cpu_or_gpu": "auto",
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
            "visual_overlay_enabled": "no",
            "visual_overlay_player_dot": "yes",
            "visual_overlay_attack_range": "yes",
            "visual_overlay_safe_range": "yes",
            "visual_overlay_super_range": "yes",
            "visual_overlay_movement_arrow": "yes",
            "visual_overlay_los_all_enemies": "yes",
            "visual_overlay_enemies": "yes",
            "visual_overlay_teammates": "yes",
            "visual_overlay_walls": "yes",
            "visual_overlay_hide_when_dead": "yes",
            "visual_overlay_brawler_hud": "yes",
            "visual_overlay_gas_zone": "yes",
            "visual_overlay_danger_zones": "yes",
            "visual_overlay_decision_banner": "yes",
            "visual_overlay_hp_bars": "yes",
            "visual_overlay_target_info": "yes",
            "visual_overlay_ghost_dots": "yes",
            "visual_overlay_opacity": 194,
            "visual_overlay_hud_position": "top-left",
            "map_orientation": "vertical",
            "ai_mode": "hybrid",
            "manual_aim_enabled": "yes",
            "projectile_detection_enabled": "yes",
            "rl_training_enabled": "yes",
            "rl_model_dir": "rl_models",
            "hp_check_interval_s": 0.1,
            "hp_conf_low_threshold": 0.35,
            "hp_stale_timeout_s": 0.7,
            "hp_warning_enter_pct": 45,
            "hp_warning_exit_pct": 55,
            "hp_critical_enter_pct": 20,
            "hp_critical_exit_pct": 26,
        },
        "legacy_keys": {"current_emulator"},
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
