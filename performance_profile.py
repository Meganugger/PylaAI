from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import tomllib

try:
    import toml
except Exception:  # pragma: no cover - depends on local environment
    toml = None


PERFORMANCE_PROFILES = {
    "balanced": {
        "description": "Stable default capture and detection settings for most PCs.",
        "general_config": {
            "cpu_or_gpu": "auto",
            "max_ips": 24,
            "scrcpy_max_fps": 30,
            "scrcpy_max_width": 960,
            "scrcpy_bitrate": 3000000,
            "onnx_cpu_threads": 4,
            "used_threads": 4,
        },
        "bot_config": {
            "entity_detection_confidence": 0.55,
            "entity_detection_retry_confidence": 0.35,
        },
    },
    "low_end": {
        "description": "Lower heat and CPU use for older laptops or thermal throttling.",
        "general_config": {
            "cpu_or_gpu": "auto",
            "max_ips": 20,
            "scrcpy_max_fps": 24,
            "scrcpy_max_width": 854,
            "scrcpy_bitrate": 2000000,
            "onnx_cpu_threads": 2,
            "used_threads": 2,
        },
        "bot_config": {
            "entity_detection_confidence": 0.55,
            "entity_detection_retry_confidence": 0.35,
        },
    },
    "quality": {
        "description": "Sharper capture for stronger PCs; use when IPS remains stable.",
        "general_config": {
            "cpu_or_gpu": "auto",
            "max_ips": 24,
            "scrcpy_max_fps": 30,
            "scrcpy_max_width": 1280,
            "scrcpy_bitrate": 5000000,
            "onnx_cpu_threads": 4,
            "used_threads": 4,
        },
        "bot_config": {
            "entity_detection_confidence": 0.55,
            "entity_detection_retry_confidence": 0.35,
        },
    },
}


def _read_toml(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    with file_path.open("rb") as handle:
        return dict(tomllib.load(handle))


def _format_toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_format_toml_value(item) for item in value) + "]"
    if value is None:
        return '""'
    return json.dumps(str(value))


def _write_toml(data: dict, path: str) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if toml is not None:
        with file_path.open("w", encoding="utf-8") as handle:
            toml.dump(data, handle)
        return
    lines = [f"{key} = {_format_toml_value(value)}" for key, value in data.items()]
    file_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _profile_key(profile_name: str) -> str:
    return str(profile_name or "balanced").strip().lower().replace("-", "_")


def apply_performance_profile(
    profile_name: str = "balanced",
    general_config_path: str = "cfg/general_config.toml",
    bot_config_path: str = "cfg/bot_config.toml",
    save: bool = True,
) -> dict:
    profile_key = _profile_key(profile_name)
    if profile_key not in PERFORMANCE_PROFILES:
        available = ", ".join(sorted(PERFORMANCE_PROFILES))
        raise ValueError(f"Unknown performance profile '{profile_name}'. Available profiles: {available}")

    profile = PERFORMANCE_PROFILES[profile_key]
    general_config = deepcopy(_read_toml(general_config_path))
    bot_config = deepcopy(_read_toml(bot_config_path))

    general_config.update(profile["general_config"])
    bot_config.update(profile["bot_config"])

    if save:
        _write_toml(general_config, general_config_path)
        _write_toml(bot_config, bot_config_path)

    return {
        "profile": profile_key,
        "description": profile["description"],
        "general_config": general_config,
        "bot_config": bot_config,
        "changed_general_keys": sorted(profile["general_config"]),
        "changed_bot_keys": sorted(profile["bot_config"]),
    }


def get_performance_profile_summary(profile_name: str = "balanced") -> str:
    profile_key = _profile_key(profile_name)
    if profile_key not in PERFORMANCE_PROFILES:
        available = ", ".join(sorted(PERFORMANCE_PROFILES))
        raise ValueError(f"Unknown performance profile '{profile_name}'. Available profiles: {available}")
    profile = PERFORMANCE_PROFILES[profile_key]
    settings = []
    for key, value in profile["general_config"].items():
        settings.append(f"{key}={value}")
    for key, value in profile["bot_config"].items():
        settings.append(f"{key}={value}")
    return f"{profile_key}: {profile['description']} ({', '.join(settings)})"
