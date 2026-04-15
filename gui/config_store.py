from utils import load_toml_as_dict, save_dict_as_toml


CONFIG_SPECS = {
    "bot": {
        "path": "cfg/bot_config.toml",
        "defaults": {
            "gamemode_type": 3,
            "gamemode": "brawlball",
            "bot_uses_gadgets": "yes",
            "minimum_movement_delay": 0.4,
            "wall_detection_confidence": 0.9,
            "entity_detection_confidence": 0.6,
            "unstuck_movement_delay": 3.0,
            "unstuck_movement_hold_time": 1.5,
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
            "api_base_url": "localhost",
            "current_emulator": "LDPlayer",
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
