import asyncio
import hashlib
import io
import os
import re
from io import BytesIO
import ctypes
import json
import threading

from runtime_threads import apply_process_thread_limits, configure_torch_threads, configure_opencv_threads
from instance_support import current_config_dir, current_runtime_root, get_brawler_data_path

apply_process_thread_limits()

import aiohttp
import google_play_scraper
import requests
import toml
from PIL import Image
from discord import Webhook
import discord
import cv2
import numpy as np
from packaging import version
import bettercam
import time
import onnxruntime as ort

configure_opencv_threads(cv2)

DEVELOPER_API_BASE_URL = "https://developer.brawlstars.com/api/"
_brawl_stars_api_refresh_done = False
_brawl_stars_api_refresh_signature = None

def to_bgr_array(image):
    if isinstance(image, Image.Image):
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if isinstance(image, np.ndarray):
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")

def extract_text_and_positions(image_input):
    if isinstance(image_input, Image.Image):
        image_input = np.asarray(image_input)
    results = reader.readtext(image_input)
    text_details = {}
    for (bbox, text, prob) in results:
        top_left, top_right, bottom_right, bottom_left = bbox
        cx = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4
        cy = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4
        center = (cx, cy)
        formatted_bbox = {
            'top_left': top_left,
            'top_right': top_right,
            'bottom_right': bottom_right,
            'bottom_left': bottom_left,
            'center': center
        }

        text_details[text.lower()] = formatted_bbox

    return text_details

class DefaultEasyOCR:
    def __init__(self):
        self.reader = None
        self.easyocr_module = None
        self._reader_lock = threading.Lock()

    @staticmethod
    def _torch_gpu_available():
        try:
            import torch
        except Exception:
            return False

        try:
            if torch.cuda.is_available():
                return True
        except Exception:
            pass

        try:
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            if mps_backend and mps_backend.is_available():
                return True
        except Exception:
            pass

        return False

    @staticmethod
    def _should_use_gpu():
        configured_value = str(load_toml_as_dict("cfg/general_config.toml").get("easyocr_gpu", "auto")).lower()
        has_torch_gpu = DefaultEasyOCR._torch_gpu_available()

        if configured_value in ("yes", "true", "1", "gpu"):
            return has_torch_gpu
        if configured_value in ("no", "false", "0", "cpu"):
            return False

        preferred_device = str(load_toml_as_dict("cfg/general_config.toml").get("cpu_or_gpu", "auto")).lower()
        if preferred_device not in ("gpu", "auto"):
            return False

        return has_torch_gpu

    def _get_reader(self):
        if self.reader is not None:
            return self.reader

        with self._reader_lock:
            if self.reader is not None:
                return self.reader
            if self.easyocr_module is None:
                import easyocr
                self.easyocr_module = easyocr
                try:
                    import torch
                    configure_torch_threads(torch)
                except Exception:
                    pass
            use_gpu = self._should_use_gpu()
            print(f"Initializing EasyOCR (gpu={use_gpu})")
            self.reader = self.easyocr_module.Reader(['en'], gpu=use_gpu)
        return self.reader

    def readtext(self, image_input, *args, **kwargs):
        return self._get_reader().readtext(image_input, *args, **kwargs)

    def is_ready(self):
        return self.reader is not None

    def warm_up(self):
        try:
            self._get_reader()
            return True
        except Exception:
            return False

_CONFIG_DEFAULTS = {
    "cfg/general_config.toml": {
        "personal_webhook": "",
        "discord_id": "",
        "super_debug": "yes",
        "cpu_or_gpu": "auto",
        "preferred_backend": "auto",
        "max_ips": "auto",
        "pyla_version": "1.0.0+performance",
        "long_press_star_drop": "yes",
        "trophies_multiplier": 1,
        "run_for_minutes": 600,
        "emulator_port": 5037,
        "instance_index": 1,
        "instance_count": 1,
        "scrcpy_max_fps": "auto",
        "brawlstars_package": "com.supercell.brawlstars",
        "brawlstars_api_key": "",
        "brawlstars_player_tag": "",
        "auto_push_target_trophies": 1000,
        "process_threads": "auto",
        "opencv_threads": "auto",
        "onnx_intra_threads": "auto",
        "onnx_inter_threads": "auto",
        "torch_threads": "auto",
        "torch_interop_threads": "auto",
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
        "rl_kpi_adj_profile": "balanced",
        "visual_overlay_bt_path": "yes",
        "visual_overlay_projectiles": "yes",
        "visual_overlay_spatial_grid": "yes",
        "visual_overlay_combo_queue": "yes",
        "visual_overlay_aim_stats": "yes",
        "rl_kpi_adj_bonus_base": 0.05,
        "rl_kpi_adj_bonus_threat_scale": 0.05,
        "rl_kpi_adj_attack_block_base_penalty": 0.02,
        "rl_kpi_adj_attack_block_threat_penalty": 0.04,
        "rl_kpi_adj_pattern_block_penalty": 0.012,
        "rl_kpi_adj_clip_abs": 2.5,
        "hp_check_interval_s": 0.1,
        "hp_conf_low_threshold": 0.35,
        "hp_stale_timeout_s": 0.7,
        "hp_warning_enter_pct": 45,
        "hp_warning_exit_pct": 55,
        "hp_critical_enter_pct": 20,
        "hp_critical_exit_pct": 26,
    },
    "cfg/bot_config.toml": {
        "gamemode_type": 3,
        "bot_uses_gadgets": "yes",
        "minimum_movement_delay": 0.08,
        "gamemode": "knockout",
        "unstuck_movement_delay": 1.5,
        "unstuck_movement_hold_time": 0.8,
        "wall_model_classes": ["wall", "bush", "close_bush"],
        "gadget_pixels_minimum": 500.0,
        "hypercharge_pixels_minimum": 500.0,
        "super_pixels_minimum": 800.0,
        "wall_detection_confidence": 0.9,
        "entity_detection_confidence": 0.6,
        "seconds_to_hold_attack_after_reaching_max": 1.5,
        "play_again_on_win": "no",
        "post_match_dismiss_hold_seconds": 10.0,
        "push_all_api_refresh_interval": 8.0,
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
    "cfg/brawl_stars_api.toml": {
        "player_tag": "#YOURTAG",
        "timeout_seconds": 15,
        "auto_refresh_token": False,
        "developer_email": "",
        "developer_password": "",
        "public_ip_service": "https://api.ipify.org",
        "key_name_prefix": "PylaAI Auto",
        "key_description": "Auto-generated by PylaAI for the current public IP.",
        "delete_old_auto_tokens": True,
        "delete_all_tokens": False,
        "last_public_ip": "",
        "api_token": "",
    },
}


def _config_defaults_for_path(file_path):
    normalized = file_path.replace("\\", "/").lstrip("./")
    return _CONFIG_DEFAULTS.get(normalized, {})


def resolve_cfg_path(file_path):
    normalized = str(file_path or "").replace("\\", "/")
    config_dir = current_config_dir()
    if normalized.startswith("./cfg/"):
        return os.path.join(config_dir, normalized[6:])
    if normalized.startswith("cfg/"):
        return os.path.join(config_dir, normalized[4:])
    return file_path


def resolve_runtime_path(file_path):
    normalized = str(file_path or "").replace("\\", "/").lstrip("./")
    if normalized == "latest_brawler_data.json":
        return get_brawler_data_path()
    if normalized.startswith("cfg/"):
        return resolve_cfg_path(normalized)
    return file_path


def get_runtime_root():
    return current_runtime_root()


def load_toml_as_dict(file_path):
    defaults = _config_defaults_for_path(file_path)
    file_path = resolve_runtime_path(file_path)
    loaded = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            text = f.read()
        toml_decode_error = getattr(toml, "TomlDecodeError", ValueError)
        try:
            loaded = toml.loads(text.lstrip("\ufeffï»¿"))
        except toml_decode_error as exc:
            sanitized = text.encode("utf-8", errors="ignore").decode("utf-8-sig", errors="ignore")
            sanitized = sanitized.lstrip("\ufeffï»¿\x00\r\n\t ")
            try:
                loaded = toml.loads(sanitized)
            except toml_decode_error:
                if hasattr(toml, "TomlDecodeError"):
                    raise toml.TomlDecodeError(
                        f"Could not parse {file_path}. The file may contain invalid leading bytes or malformed TOML: {exc}",
                        getattr(exc, "doc", text),
                        getattr(exc, "pos", 0),
                    )
                raise ValueError(
                    f"Could not parse {file_path}. The file may contain invalid leading bytes or malformed TOML: {exc}"
                )

    if not defaults:
        return loaded

    merged = dict(defaults)
    merged.update(loaded)
    return merged


reader = DefaultEasyOCR()
cfg_api_base_url = str(load_toml_as_dict("cfg/general_config.toml").get("api_base_url", "localhost")).strip()
api_base_url = cfg_api_base_url if cfg_api_base_url and cfg_api_base_url != "default" else "localhost"
brawlers_info_file_path = resolve_cfg_path("cfg/brawlers_info.json")
_timing_stats = {}
_timing_enabled = None
DEVELOPER_API_BASE_URL = "https://developer.brawlstars.com/api/"
_brawl_stars_api_refresh_done = False
_brawl_stars_api_refresh_signature = None
STATE_ICON_PATHS = (
    "./state_finder/images_to_detect/brawl_stars_icon_big.png",
    "./state_finder/images_to_detect/brawl_stars_icon.png",
)

def timing_enabled():
    global _timing_enabled
    if _timing_enabled is None:
        _timing_enabled = str(load_toml_as_dict("cfg/general_config.toml").get("timing_debug", "no")).lower() in ("yes", "true", "1")
    return _timing_enabled

def record_timing(name, duration_seconds, print_every=120):
    if not timing_enabled():
        return

    stats = _timing_stats.setdefault(name, {"count": 0, "total": 0.0, "max": 0.0})
    stats["count"] += 1
    stats["total"] += duration_seconds
    stats["max"] = max(stats["max"], duration_seconds)

    if stats["count"] % print_every == 0:
        avg_ms = (stats["total"] / stats["count"]) * 1000
        max_ms = stats["max"] * 1000
        print(f"[timing] {name}: avg={avg_ms:.2f}ms max={max_ms:.2f}ms samples={stats['count']}")

def count_hsv_pixels(pil_image, low_hsv, high_hsv):
    opencv_image = to_bgr_array(pil_image)
    hsv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, np.array(low_hsv), np.array(high_hsv))
    pixel_count = np.count_nonzero(mask)
    return pixel_count

def save_brawler_data(data):
    """
    Save the given data to a json file. As a list of dictionaries.
    """
    with open(get_brawler_data_path(), 'w', encoding="utf-8") as f:
        json.dump(data, f, indent=4)



def find_template_center(main_img, template, threshold=0.8):
    main_image_arr = to_bgr_array(main_img)
    if len(main_image_arr.shape) == 3 and main_image_arr.shape[2] == 3:
        main_image_cv = cv2.cvtColor(main_image_arr, cv2.COLOR_BGR2GRAY)
    else:
        main_image_cv = main_image_arr

    template_arr = np.asarray(template)
    if len(template_arr.shape) == 3 and template_arr.shape[2] == 3:
        template_cv = cv2.cvtColor(template_arr, cv2.COLOR_BGR2GRAY)
    else:
        template_cv = template_arr
    w, h = template_cv.shape[::-1]

    # Perform template matching
    result = cv2.matchTemplate(main_image_cv, template_cv, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Check if the match is found based on a threshold value
    if max_val >= threshold:
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2

        return center_x, center_y
    else:
        return False





def save_dict_as_toml(data, file_path):
    file_path = resolve_runtime_path(file_path)
    with open(file_path, 'w', encoding="utf-8") as f:
        toml.dump(data, f)


def _config_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _extract_api_token(value):
    if isinstance(value, dict):
        return str(value.get("key") or value.get("token") or "").strip()
    return str(value or "").strip()


def _developer_api_post(session, endpoint, payload, timeout):
    response = session.post(
        DEVELOPER_API_BASE_URL + endpoint,
        json=payload,
        timeout=timeout,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://developer.brawlstars.com",
            "Referer": "https://developer.brawlstars.com/",
        },
    )
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            return {}
    try:
        error_payload = response.json()
        reason = error_payload.get("description") or error_payload.get("message") or response.text
    except ValueError:
        reason = response.text
    raise RuntimeError(f"Developer portal error {response.status_code} at {endpoint}: {reason}")


def get_public_ip(service_url="https://api.ipify.org"):
    response = requests.get(service_url, timeout=15)
    response.raise_for_status()
    ip_address = response.text.strip()
    if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip_address):
        raise RuntimeError(f"Public IP service returned an invalid IPv4 address: {ip_address}")
    return ip_address


def get_config_player_tag(config):
    tag = str(config.get("player_tag", "") or "").strip()
    if tag and tag.upper() != "#YOURTAG":
        return tag
    general_config = load_toml_as_dict("cfg/general_config.toml")
    general_tag = str(general_config.get("brawlstars_player_tag", "") or "").strip()
    if general_tag and general_tag.upper() != "#YOURTAG":
        return general_tag
    legacy_tag = str(general_config.get("player_tag", "") or "").strip()
    if legacy_tag and legacy_tag.upper() != "#YOURTAG":
        return legacy_tag
    return tag


def _general_brawlstars_api_config():
    general_config = load_toml_as_dict("cfg/general_config.toml")
    return {
        "api_token": str(general_config.get("brawlstars_api_key", "") or "").strip(),
        "player_tag": get_config_player_tag({}),
        "timeout_seconds": 15,
        "auto_refresh_token": "no",
    }


def refresh_brawl_stars_api_token_if_enabled(config, file_path="cfg/brawl_stars_api.toml", force=False):
    global _brawl_stars_api_refresh_done, _brawl_stars_api_refresh_signature
    if not _config_bool(config.get("auto_refresh_token"), False):
        return config
    email = str(config.get("developer_email", "") or "").strip()
    password = str(config.get("developer_password", "") or "").strip()
    player_tag = str(config.get("player_tag", "") or "").strip()
    existing_token = _extract_api_token(config.get("api_token", ""))
    refresh_signature = (str(file_path), email, password, player_tag)
    if not force and _brawl_stars_api_refresh_done and _brawl_stars_api_refresh_signature == refresh_signature and existing_token:
        return config
    if not email or not password:
        _brawl_stars_api_refresh_done = False
        _brawl_stars_api_refresh_signature = None
        print("Brawl Stars API auto-refresh is enabled but developer_email/developer_password are missing in cfg/brawl_stars_api.toml; using the saved API token if present.")
        return config
    timeout = int(config.get("timeout_seconds", 15) or 15)
    key_name_prefix = str(config.get("key_name_prefix", "PylaAI Auto") or "PylaAI Auto").strip()
    delete_old_auto_tokens = _config_bool(config.get("delete_old_auto_tokens"), True)
    delete_all_tokens = _config_bool(config.get("delete_all_tokens"), False)
    public_ip_service = str(config.get("public_ip_service", "https://api.ipify.org") or "").strip()
    session = requests.Session()
    public_ip = get_public_ip(public_ip_service or "https://api.ipify.org")
    _developer_api_post(session, "login", {"email": email, "password": password}, timeout)
    account = _developer_api_post(session, "account/load", {}, timeout)
    scopes = account.get("developer", {}).get("allowedScopes") or ["brawlstars"]
    existing_keys = _developer_api_post(session, "apikey/list", {}, timeout).get("keys", [])
    if delete_all_tokens or delete_old_auto_tokens:
        for api_key in existing_keys:
            key_name = str(api_key.get("name", ""))
            if delete_all_tokens or key_name.startswith(key_name_prefix):
                key_id = api_key.get("id")
                if key_id:
                    _developer_api_post(session, "apikey/revoke", {"id": key_id}, timeout)
    key_name = f"{key_name_prefix} {time.strftime('%Y-%m-%d %H:%M:%S')}"
    description = str(config.get("key_description", "Auto-generated by PylaAI for the current public IP."))
    created = _developer_api_post(session, "apikey/create", {"name": key_name, "description": description, "cidrRanges": [public_ip], "scopes": scopes}, timeout)
    new_token = _extract_api_token(created.get("key") or created.get("token"))
    if not new_token:
        refreshed_keys = _developer_api_post(session, "apikey/list", {}, timeout).get("keys", [])
        matching_keys = [api_key for api_key in refreshed_keys if api_key.get("name") == key_name]
        if matching_keys:
            new_token = _extract_api_token(matching_keys[0].get("key") or matching_keys[0].get("token"))
    if not new_token:
        raise RuntimeError("Created a Brawl Stars developer key, but could not read the returned API token.")
    config["api_token"] = new_token
    config["player_tag"] = player_tag
    config["last_public_ip"] = public_ip
    save_dict_as_toml(config, file_path)
    _brawl_stars_api_refresh_done = True
    _brawl_stars_api_refresh_signature = refresh_signature
    print(f"Refreshed Brawl Stars API token for public IP {public_ip}.")
    return config


def load_brawl_stars_api_config(file_path="cfg/brawl_stars_api.toml", force_refresh=False):
    fallback = _general_brawlstars_api_config()
    resolved_file_path = globals().get("resolve_runtime_path", lambda value: value)(file_path)
    if not os.path.exists(resolved_file_path):
        return fallback
    try:
        config = dict(load_toml_as_dict(file_path) or {})
    except toml.TomlDecodeError:
        with open(resolved_file_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
        config = {}
        token_match = re.search(r'api_token\s*=\s*"(.*?)"', text, re.DOTALL)
        if token_match:
            config["api_token"] = "".join(token_match.group(1).split())
        tag_match = re.search(r'player_tag\s*=\s*"([^"]*)"', text)
        if tag_match:
            config["player_tag"] = tag_match.group(1).strip()
        timeout_match = re.search(r"timeout_seconds\s*=\s*(\d+)", text)
        if timeout_match:
            config["timeout_seconds"] = int(timeout_match.group(1))
    config["api_token"] = _extract_api_token(config.get("api_token")) or fallback.get("api_token", "")
    config["player_tag"] = get_config_player_tag(config)
    config["timeout_seconds"] = int(config.get("timeout_seconds") or fallback.get("timeout_seconds", 15) or 15)
    try:
        return refresh_brawl_stars_api_token_if_enabled(config, file_path, force=force_refresh)
    except Exception as exc:
        print(f"Brawl Stars API token refresh failed; using saved token if available. {exc}")
        return config


def fetch_brawl_stars_player(api_token, player_tag, timeout=15):
    api_token = _extract_api_token(api_token)
    cleaned_tag = str(player_tag or "").strip().upper()
    if not api_token:
        raise ValueError("Missing Brawl Stars API token.")
    if not cleaned_tag or cleaned_tag == "#YOURTAG":
        raise ValueError("Missing Brawl Stars player tag.")
    if not cleaned_tag.startswith("#"):
        cleaned_tag = f"#{cleaned_tag}"
    encoded_tag = cleaned_tag.replace("#", "%23")
    response = requests.get(f"https://api.brawlstars.com/v1/players/{encoded_tag}", headers={"Authorization": f"Bearer {api_token}"}, timeout=timeout)
    if response.status_code == 200:
        return response.json()
    try:
        error_payload = response.json()
        reason = error_payload.get("reason") or error_payload.get("message") or response.text
    except ValueError:
        reason = response.text
    if response.status_code == 403:
        raise RuntimeError("Brawl Stars API accessDenied. The token is not valid for the current public IP.")
    raise RuntimeError(f"Brawl Stars API error {response.status_code}: {reason}")


def update_toml_file(path, new_data):
    path = resolve_runtime_path(path)
    with open(path, 'w', encoding="utf-8") as file:
        toml.dump(new_data, file)


def _config_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _developer_api_post(session, endpoint, payload, timeout):
    response = session.post(
        DEVELOPER_API_BASE_URL + endpoint,
        json=payload,
        timeout=timeout,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://developer.brawlstars.com",
            "Referer": "https://developer.brawlstars.com/",
        },
    )
    if response.status_code == 200:
        try:
            return response.json()
        except ValueError:
            return {}
    try:
        error_payload = response.json()
        reason = error_payload.get("description") or error_payload.get("message") or response.text
    except ValueError:
        reason = response.text
    raise RuntimeError(f"Developer portal error {response.status_code} at {endpoint}: {reason}")


def _extract_api_token(value):
    if isinstance(value, dict):
        return str(value.get("key") or value.get("token") or "").strip()
    return str(value or "").strip()


def get_public_ip(service_url="https://api.ipify.org"):
    response = requests.get(service_url, timeout=15)
    response.raise_for_status()
    ip_address = response.text.strip()
    if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip_address):
        raise RuntimeError(f"Public IP service returned an invalid IPv4 address: {ip_address}")
    return ip_address


def normalize_brawler_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def get_config_player_tag(config):
    tag = str(config.get("player_tag", "")).strip()
    if tag and tag.upper() != "#YOURTAG":
        return tag
    general_tag = str(load_toml_as_dict("cfg/general_config.toml").get("player_tag", "")).strip()
    if general_tag and general_tag.upper() != "#YOURTAG":
        return general_tag
    legacy_tag = str(load_toml_as_dict("cfg/general_config.toml").get("brawlstars_player_tag", "")).strip()
    if legacy_tag and legacy_tag.upper() != "#YOURTAG":
        return legacy_tag
    return tag


def refresh_brawl_stars_api_token_if_enabled(config, file_path="cfg/brawl_stars_api.toml", force=False):
    global _brawl_stars_api_refresh_done, _brawl_stars_api_refresh_signature
    if not _config_bool(config.get("auto_refresh_token"), False):
        _brawl_stars_api_refresh_done = True
        return config

    email = str(config.get("developer_email", "")).strip()
    password = str(config.get("developer_password", "")).strip()
    player_tag = str(config.get("player_tag", "")).strip()
    existing_token = _extract_api_token(config.get("api_token", ""))
    refresh_signature = (str(file_path), email, password, player_tag)

    if (
        not force
        and _brawl_stars_api_refresh_done
        and _brawl_stars_api_refresh_signature == refresh_signature
        and existing_token
    ):
        return config

    if not email or not password:
        _brawl_stars_api_refresh_done = False
        _brawl_stars_api_refresh_signature = None
        raise ValueError(
            "auto_refresh_token is enabled, but developer_email/developer_password are missing. "
            "Open cfg/brawl_stars_api.toml and fill developer_email, developer_password, and player_tag."
        )

    timeout = int(config.get("timeout_seconds", 15))
    key_name_prefix = str(config.get("key_name_prefix", "PylaAI Auto")).strip() or "PylaAI Auto"
    delete_old_auto_tokens = _config_bool(config.get("delete_old_auto_tokens"), True)
    delete_all_tokens = _config_bool(config.get("delete_all_tokens"), False)
    public_ip_service = str(config.get("public_ip_service", "https://api.ipify.org")).strip()
    public_ip = get_public_ip(public_ip_service)

    session = requests.Session()
    _developer_api_post(session, "login", {"email": email, "password": password}, timeout)
    account = _developer_api_post(session, "account/load", {}, timeout)
    developer = account.get("developer", {})
    scopes = developer.get("allowedScopes") or ["brawlstars"]

    existing_keys = _developer_api_post(session, "apikey/list", {}, timeout).get("keys", [])
    if delete_all_tokens or delete_old_auto_tokens:
        for api_key in existing_keys:
            key_name = str(api_key.get("name", ""))
            if delete_all_tokens or key_name.startswith(key_name_prefix):
                key_id = api_key.get("id")
                if key_id:
                    _developer_api_post(session, "apikey/revoke", {"id": key_id}, timeout)

    key_name = f"{key_name_prefix} {time.strftime('%Y-%m-%d %H:%M:%S')}"
    description = str(config.get("key_description", "Auto-generated by PylaAI for the current public IP."))
    created = _developer_api_post(
        session,
        "apikey/create",
        {
            "name": key_name,
            "description": description,
            "cidrRanges": [public_ip],
            "scopes": scopes,
        },
        timeout,
    )

    new_token = _extract_api_token(created.get("key") or created.get("token"))
    if not new_token:
        refreshed_keys = _developer_api_post(session, "apikey/list", {}, timeout).get("keys", [])
        matching_keys = [api_key for api_key in refreshed_keys if api_key.get("name") == key_name]
        if matching_keys:
            new_token = _extract_api_token(matching_keys[0].get("key") or matching_keys[0].get("token"))

    if not new_token:
        raise RuntimeError("Created a developer key, but could not find the returned API token.")

    config["api_token"] = new_token
    config["last_public_ip"] = public_ip
    save_dict_as_toml(config, file_path)
    _brawl_stars_api_refresh_done = True
    _brawl_stars_api_refresh_signature = refresh_signature
    print(f"Refreshed Brawl Stars API token for public IP {public_ip}.")
    return config


def fetch_brawl_stars_player(api_token, player_tag, timeout=15):
    api_token = _extract_api_token(api_token)
    cleaned_tag = str(player_tag).strip().upper()
    if not api_token:
        raise ValueError(
            "Missing Brawl Stars API token. Fill cfg/brawl_stars_api.toml or enable auto_refresh_token."
        )
    if not cleaned_tag or cleaned_tag == "#YOURTAG":
        raise ValueError("Missing player_tag in cfg/brawl_stars_api.toml.")
    if not cleaned_tag.startswith("#"):
        cleaned_tag = f"#{cleaned_tag}"

    encoded_tag = cleaned_tag.replace("#", "%23")
    response = requests.get(
        f"https://api.brawlstars.com/v1/players/{encoded_tag}",
        headers={"Authorization": f"Bearer {api_token}"},
        timeout=timeout,
    )
    if response.status_code == 200:
        return response.json()
    try:
        error_payload = response.json()
        reason = error_payload.get("reason") or error_payload.get("message") or response.text
    except ValueError:
        reason = response.text
    if response.status_code == 403:
        raise RuntimeError(
            "Brawl Stars API accessDenied. The API token is not valid for the current public IP."
        )
    raise RuntimeError(f"Brawl Stars API error {response.status_code}: {reason}")


def load_brawl_stars_api_config(file_path="cfg/brawl_stars_api.toml", force_refresh=False):
    try:
        config = dict(load_toml_as_dict(file_path))
        if "api_token" in config:
            config["api_token"] = "".join(str(config["api_token"]).split())
    except toml.TomlDecodeError:
        with open(file_path, "r", encoding="utf-8-sig") as f:
            text = f.read()
        config = {}
        token_match = re.search(r'api_token\s*=\s*"(.*?)"', text, re.DOTALL)
        if token_match:
            config["api_token"] = "".join(token_match.group(1).split())
        tag_match = re.search(r'player_tag\s*=\s*"([^"]*)"', text)
        if tag_match:
            config["player_tag"] = tag_match.group(1).strip()
        timeout_match = re.search(r"timeout_seconds\s*=\s*(\d+)", text)
        config["timeout_seconds"] = int(timeout_match.group(1)) if timeout_match else 15
        auto_refresh_match = re.search(r"auto_refresh_token\s*=\s*(true|false)", text, re.IGNORECASE)
        config["auto_refresh_token"] = (
            auto_refresh_match.group(1).lower() == "true" if auto_refresh_match else False
        )
        for key in (
            "developer_email",
            "developer_password",
            "public_ip_service",
            "key_name_prefix",
            "key_description",
            "last_public_ip",
        ):
            match = re.search(rf'{key}\s*=\s*"([^"]*)"', text)
            if match:
                config[key] = match.group(1)
        for key in ("delete_old_auto_tokens", "delete_all_tokens"):
            match = re.search(rf"{key}\s*=\s*(true|false)", text, re.IGNORECASE)
            if match:
                config[key] = match.group(1).lower() == "true"

    config["player_tag"] = get_config_player_tag(config)
    if force_refresh:
        return refresh_brawl_stars_api_token_if_enabled(config, file_path, force=True)
    return refresh_brawl_stars_api_token_if_enabled(config, file_path)

def load_brawlers_info():
    if os.path.exists(brawlers_info_file_path):
        with open(brawlers_info_file_path, 'r') as f:
            return json.load(f)
    else:
        return {}

def update_brawlers_info(brawlers_info):
    with open(brawlers_info_file_path, 'w') as f:
        json.dump(brawlers_info, f, indent=4)


def get_brawler_list():
    if api_base_url == "localhost":
        brawler_list = list(load_brawlers_info().keys())
        return brawler_list
    url = f'https://{api_base_url}/get_brawler_list'
    response = requests.post(url)
    if response.status_code == 201:
        data = response.json()
        return data.get('brawlers', [])
    else:
        return []


def update_missing_brawlers_info(brawlers):
    brawlers_info = load_brawlers_info()
    for brawler in brawlers:
        if brawler not in brawlers_info:
            brawler_info = get_brawler_info(brawler)
            if brawler_info:
                brawlers_info[brawler] = brawler_info
                update_brawlers_info(brawlers_info)
                print(f"Added info for brawler '{brawler}': {brawler_info}")
                # Download the brawler icon
                save_brawler_icon(brawler)
            else:
                print(f"Could not find info for brawler '{brawler}'")
        if not os.path.exists(f"./api/assets/brawler_icons/{brawler}.png"):
            save_brawler_icon(brawler)


def get_brawler_info(brawler_name):
    url = f'https://{api_base_url}/get_brawler_info'  # Adjust the URL if necessary
    response = requests.post(url, json={'brawler_name': brawler_name})
    if response.status_code == 200:
        data = response.json()
        return data.get('info', [])
    else:
        print(f"Error fetching range for '{brawler_name}': {response.status_code} - {response.text}")
        return None


def save_brawler_icon(brawler_name):
    # Clean the brawler name for filename
    brawler_name_clean = brawler_name.lower().replace(' ', '').replace('-', '').replace('.', '').replace('&',
                                                                                                         '')
    brawlers_url = "https://api.brawlapi.com/v1/brawlers"
    response = requests.get(brawlers_url)
    if response.status_code != 200:
        print(f"Failed to fetch brawlers from API: {response.status_code}")
        return
    brawlers_data = response.json()['list']

    # Find the brawler in the API data
    for brawler_obj in brawlers_data:
        api_brawler_name = brawler_obj['name'].lower().replace(' ', '').replace('-', '').replace('.',
                                                                                                 '').replace(
            '&', '')
        if api_brawler_name == brawler_name_clean:
            icon_url = brawler_obj['imageUrl2']
            img_response = requests.get(icon_url)
            if img_response.status_code == 200:
                image = Image.open(BytesIO(img_response.content))
                image.save(f"api/assets/brawler_icons/{brawler_name_clean}.png")
                print(f"Saved icon for brawler '{brawler_name}'")
            else:
                print(f"Failed to download icon for '{brawler_name}'")
            return
    print(f"Icon not found for brawler '{brawler_name}'")


def update_icons():
    try:
        icon_link = google_play_scraper.app("com.supercell.brawlstars")["icon"]
    except:
        time.sleep(1)
        try:
            icon_link = google_play_scraper.app("com.supercell.brawlstars")["icon"]
        except Exception as e:
            print(f"Failed to get latest icon link from Google Play Store: {e}")
            return

    response = requests.get(icon_link)
    big_icon = 'brawl_stars_icon_big.png'
    small_icon = 'brawl_stars_icon.png'
    if response.status_code == 200:
        icon_image = Image.open(BytesIO(response.content))

        # big icon
        big_icon_image = icon_image.resize((69, 69))
        width, height = big_icon_image.size
        left = (width - 50) / 2
        top = (height - 50) / 2
        right = (width + 50) / 2
        bottom = (height + 50) / 2
        big_icon_image = big_icon_image.crop((left, top, right, bottom))
        big_icon_image.save(f'./state_finder/images_to_detect/{big_icon}')

        # small icon resize to 16x16
        small_icon_image = icon_image.resize((16, 16))
        width, height = small_icon_image.size
        left = (width - 12) / 2
        top = (height - 12) / 2
        right = (width + 12) / 2
        bottom = (height + 12) / 2
        small_icon_image = small_icon_image.crop((left, top, right, bottom))
        small_icon_image.save(f'./state_finder/images_to_detect/{small_icon}')
        print(f"Updated to the latest icon !")
    else:
        print(f"Failed to download latest icon. Status code: {response.status_code}")


def state_icons_present():
    return all(os.path.exists(path) for path in STATE_ICON_PATHS)


def ensure_state_icons_present():
    if state_icons_present():
        return True
    update_icons()
    return state_icons_present()


def get_latest_version():
    url = f'https://{api_base_url}/check_version'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('version', '')
    else:
        return None

def check_version():
    if api_base_url != "localhost":
        latest_version = get_latest_version()
        if latest_version:
            current_version = load_toml_as_dict("cfg/general_config.toml").get('pyla_version', '')
            if version.parse(current_version) < version.parse(latest_version):
                print(f"Warning: (ignore if you're using early access) You are not using the latest public version of Pyla. \nCheck the discord for the latest download link.")
        else:
            print("Error, couldn't get the version, please check your internet connection or go ask for help in the discord.")


def has_notification_webhook() -> bool:
    webhook_url = str(load_toml_as_dict("cfg/general_config.toml").get("personal_webhook", "") or "").strip()
    return bool(webhook_url)


def _safe_notification_int(value, default=0):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _format_signed_value(value):
    number = _safe_notification_int(value, 0)
    return f"+{number}" if number > 0 else str(number)


def _normalize_discord_ping(user_id):
    raw = str(user_id or "").strip()
    if not raw:
        return None
    if raw.startswith("<@") and raw.endswith(">"):
        return raw
    raw = raw.strip("<>@! ")
    return f"<@{raw}>" if raw else None


def _normalize_notification_event(message_type, subject=None):
    raw_event = str(message_type or "").strip()
    event = raw_event.lower()
    aliases = {
        "completed": "all_targets_completed",
        "bot_is_stuck": "bot_stuck",
    }
    event = aliases.get(event, event)
    known_events = {
        "all_targets_completed",
        "farm_completed",
        "quests_completed",
        "bot_stuck",
        "brawler_completed",
        "milestone_reached",
        "status_update",
    }
    if event not in known_events:
        if subject is None and raw_event:
            subject = raw_event
        if subject:
            return "brawler_completed", subject
        return "status_update", subject
    return event, subject


def _notification_heading(event_type, subject=None, live_summary=None):
    summary = live_summary or {}
    brawler_name = str(subject or summary.get("brawler") or "current brawler").replace("_", " ").title()
    if event_type == "all_targets_completed":
        return "All Targets Completed", "Pyla has completed all configured targets."
    if event_type == "farm_completed":
        return "Trophy Farm Completed", "Pyla has completed the current Trophy Farm queue."
    if event_type == "quests_completed":
        return "Quest Queue Completed", "Pyla has completed all queued quest targets."
    if event_type == "bot_stuck":
        return "Bot Needs Attention", "Pyla could not recover cleanly and may need manual help."
    if event_type == "milestone_reached":
        milestone_start = _safe_notification_int(summary.get("milestone_start"), 0)
        milestone_end = _safe_notification_int(summary.get("milestone_end"), milestone_start + 249)
        return "Trophy Milestone Reached", f"{brawler_name} reached the {milestone_start}-{milestone_end} trophy range."
    if event_type == "brawler_completed":
        return "Brawler Goal Completed", f"Pyla completed {brawler_name}'s target and moved on."
    return "Pyla Update", f"Pyla sent a new update for {brawler_name}."


def _build_notification_embed(event_type, subject=None, live_summary=None):
    summary = live_summary or {}
    title, description = _notification_heading(event_type, subject=subject, live_summary=summary)
    embed = discord.Embed(title=title, description=description)

    if not summary:
        return embed

    brawler_name = str(summary.get("brawler") or subject or "").replace("_", " ").title()
    if brawler_name:
        embed.add_field(name="Brawler", value=brawler_name, inline=True)

    trophies = _safe_notification_int(summary.get("trophies"), None)
    if trophies is not None:
        trophy_text = str(trophies)
        if "session_trophy_delta" in summary:
            trophy_text += f" ({_format_signed_value(summary.get('session_trophy_delta'))} session)"
        embed.add_field(name="Trophies", value=trophy_text, inline=True)

    session_matches = _safe_notification_int(summary.get("session_matches"), 0)
    session_victories = _safe_notification_int(summary.get("session_victories"), 0)
    session_defeats = _safe_notification_int(summary.get("session_defeats"), 0)
    session_draws = _safe_notification_int(summary.get("session_draws"), 0)
    session_winrate = float(summary.get("session_winrate") or 0.0)
    if session_matches > 0:
        embed.add_field(
            name="Session",
            value=(
                f"{session_victories}W / {session_defeats}L / {session_draws}D\n"
                f"{session_matches} matches ({session_winrate:.1f}% WR)"
            ),
            inline=True,
        )

    current_wins = _safe_notification_int(summary.get("current_wins"), 0)
    win_streak = _safe_notification_int(summary.get("win_streak"), 0)
    embed.add_field(
        name="Progress",
        value=f"{current_wins} wins\n{win_streak} streak",
        inline=True,
    )

    last_match_result = str(summary.get("last_match_result") or "").strip().lower()
    if last_match_result:
        verification = "verified" if summary.get("last_match_verified") else "estimated"
        embed.add_field(
            name="Last Match",
            value=f"{last_match_result.title()} ({_format_signed_value(summary.get('last_match_trophy_delta'))}, {verification})",
            inline=True,
        )

    if event_type == "milestone_reached":
        milestone_start = _safe_notification_int(summary.get("milestone_start"), 0)
        milestone_end = _safe_notification_int(summary.get("milestone_end"), milestone_start + 249)
        embed.add_field(
            name="Milestone",
            value=f"{milestone_start}-{milestone_end} trophies",
            inline=True,
        )

    return embed


def _prepare_webhook_image(screenshot):
    image_to_send = None
    if isinstance(screenshot, Image.Image):
        image_to_send = screenshot
    elif isinstance(screenshot, np.ndarray):
        try:
            if screenshot.ndim == 2:
                image_to_send = Image.fromarray(screenshot)
            elif screenshot.ndim == 3 and screenshot.shape[2] == 4:
                image_to_send = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGBA))
            elif screenshot.ndim == 3:
                image_to_send = Image.fromarray(cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB))
        except Exception:
            image_to_send = None

    if image_to_send is None:
        return None

    buffer = io.BytesIO()
    image_to_send.save(buffer, format="PNG")
    buffer.seek(0)
    return discord.File(buffer, filename="screenshot.png")


async def async_notify_user(
    message_type: str | None = None,
    screenshot: Image = None,
    subject: str | None = None,
    live_summary: dict | None = None,
) -> bool:
    general_config = load_toml_as_dict("cfg/general_config.toml")
    webhook_url = str(general_config.get("personal_webhook", "") or "").strip()
    if not webhook_url:
        print("Couldn't notify: no webhook configured.")
        return False

    event_type, subject = _normalize_notification_event(message_type, subject=subject)
    ping = _normalize_discord_ping(general_config.get("discord_id", ""))
    embed = _build_notification_embed(event_type, subject=subject, live_summary=live_summary)
    file = _prepare_webhook_image(screenshot)
    if file is not None:
        embed.set_image(url="attachment://screenshot.png")

    send_kwargs = {
        "embed": embed,
        "username": "Pyla notifier",
    }
    if file is not None:
        send_kwargs["file"] = file
    if ping:
        send_kwargs["content"] = ping

    async with aiohttp.ClientSession() as session:
        webhook = Webhook.from_url(webhook_url, session=session)
        print(f"sending webhook ({event_type})")
        await webhook.send(**send_kwargs)
    return True


def notify_user(
    message_type: str | None = None,
    screenshot: Image = None,
    subject: str | None = None,
    live_summary: dict | None = None,
    timeout: float = 15.0,
) -> bool:
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return bool(loop.run_until_complete(
            asyncio.wait_for(
                async_notify_user(
                    message_type,
                    screenshot=screenshot,
                    subject=subject,
                    live_summary=live_summary,
                ),
                timeout=timeout,
            )
        ))
    except Exception as exc:
        print(f"Couldn't notify via webhook: {exc}")
        return False
    finally:
        asyncio.set_event_loop(None)
        loop.close()
        
def get_discord_link():
    if api_base_url == "localhost":
        return "https://discord.gg/xUusk3fw4A"
    url = f'https://{api_base_url}/get_discord_link'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('link', '')
    else:
        return None

def get_online_wall_model_hash():
    url = f'https://{api_base_url}/get_wall_model_hash'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('hash', '')
    else:
        return None

def calculate_sha256(file_path):
    """
    Calculate the SHA-256 hash of a file.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as file:
        # Read the file in chunks to handle large files
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def current_wall_model_is_latest() -> bool:
    """
    Check if the current wall model is the latest version.
    """
    local_hash = calculate_sha256("models/tileDetector.onnx")
    online_hash = get_online_wall_model_hash()
    return local_hash == online_hash

def get_latest_wall_model_file():
    #download the new model to replace the current file and also updates the tile list
    url = f'https://{api_base_url}/get_wall_model_file'
    response = requests.get(url)
    if response.status_code == 200:
        with open("./models/tileDetector.onnx", "wb") as file:
            file.write(response.content)
        print("Downloaded the latest wall model.")
    else:
        print(f"Failed to download the latest wall model. Status code: {response.status_code}")

def get_latest_wall_model_classes():
    url = f'https://{api_base_url}/get_wall_model_classes'
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get('classes', [])
    else:
        return None

def update_wall_model_classes():
    classes = get_latest_wall_model_classes()
    current_classes = load_toml_as_dict("cfg/bot_config.toml")["wall_model_classes"]
    if classes:
        if classes != current_classes:
            print("New wall model classes found. Updating...")
            full_config = load_toml_as_dict("cfg/bot_config.toml")
            full_config["wall_model_classes"] = classes
            update_toml_file("cfg/bot_config.toml", full_config)
            print("Updated the wall model classes.")
    else:
        print("Failed to update the wall model classes, please report this error.")


def cprint(text: str, hex_color: str): #omg color!!!
    try:
        hex_color = hex_color.lstrip("#")
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        print(f"\033[38;2;{r};{g};{b}m{text}\033[0m")
    except Exception:
        print(text)

def get_dpi_scale():
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    return int(user32.GetDpiForSystem())
