import argparse
import os
import re
import shutil
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_DIR = os.path.join(PROJECT_ROOT, "cfg")
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT
INSTANCE_ENV_KEY = "PYLA_INSTANCE_ID"
CONFIG_DIR_ENV_KEY = "PYLA_CONFIG_DIR"
RUNTIME_ROOT_ENV_KEY = "PYLA_RUNTIME_ROOT"
AUTO_START_ENV_KEY = "PYLA_AUTO_START"


def parse_bootstrap_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instance", type=int)
    parser.add_argument("--autostart", action="store_true")
    parser.add_argument("--setup-instances", nargs="?", const="interactive")
    args, _ = parser.parse_known_args(argv if argv is not None else sys.argv[1:])
    return args


def instance_root(instance_id):
    return os.path.join(PROJECT_ROOT, "instances", str(int(instance_id)))


def instance_config_dir(instance_id):
    return os.path.join(instance_root(instance_id), "cfg")


def launcher_path(instance_id):
    return os.path.join(PROJECT_ROOT, f"start_{int(instance_id)}.bat")


def current_instance_id(default=1):
    raw_value = os.environ.get(INSTANCE_ENV_KEY)
    if raw_value not in (None, ""):
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            pass
    try:
        return max(1, int(default))
    except (TypeError, ValueError):
        return 1


def current_runtime_root():
    return os.environ.get(RUNTIME_ROOT_ENV_KEY, DEFAULT_RUNTIME_ROOT)


def current_config_dir():
    return os.environ.get(CONFIG_DIR_ENV_KEY, DEFAULT_CONFIG_DIR)


def get_brawler_data_path():
    return os.path.join(current_runtime_root(), "latest_brawler_data.json")


def apply_bootstrap_environment(args):
    if not getattr(args, "instance", None):
        if getattr(args, "autostart", False):
            os.environ[AUTO_START_ENV_KEY] = "1"
        return

    runtime_root = instance_root(args.instance)
    config_dir = instance_config_dir(args.instance)
    os.environ[INSTANCE_ENV_KEY] = str(int(args.instance))
    os.environ[RUNTIME_ROOT_ENV_KEY] = runtime_root
    os.environ[CONFIG_DIR_ENV_KEY] = config_dir
    if getattr(args, "autostart", False):
        os.environ[AUTO_START_ENV_KEY] = "1"


def _upsert_toml_scalar(path, key, value_literal):
    content = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()

    pattern = re.compile(rf"(?m)^{re.escape(key)}\s*=.*$")
    replacement = f"{key} = {value_literal}"
    if pattern.search(content):
        updated = pattern.sub(replacement, content)
    else:
        trimmed = content.rstrip("\r\n")
        updated = f"{trimmed}\n{replacement}\n" if trimmed else f"{replacement}\n"

    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(updated)


def _copy_default_configs(target_cfg_dir):
    os.makedirs(target_cfg_dir, exist_ok=True)
    template_cfg_dir = current_config_dir()
    source_cfg_dir = template_cfg_dir if os.path.isdir(template_cfg_dir) else DEFAULT_CONFIG_DIR
    for name in os.listdir(DEFAULT_CONFIG_DIR):
        source = os.path.join(source_cfg_dir, name)
        destination = os.path.join(target_cfg_dir, name)
        if not os.path.isfile(source):
            continue
        if name in {"match_history.toml", "brawler_scan.json"}:
            continue
        if not os.path.exists(destination):
            shutil.copy2(source, destination)


def _seed_roster(instance_root_path):
    source = get_brawler_data_path()
    destination = os.path.join(instance_root_path, "latest_brawler_data.json")
    if os.path.exists(source) and not os.path.exists(destination):
        shutil.copy2(source, destination)


def _python_launcher():
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return r".venv\Scripts\python.exe"
    return "python"


def _generate_launcher(instance_id):
    launcher_file = launcher_path(instance_id)
    python_cmd = _python_launcher()
    content = (
        "@echo off\n"
        "cd /d \"%~dp0\"\n"
        ":restart_loop\n"
        f"echo [%date% %time%] Starting performance instance {instance_id}...\n"
        f"{python_cmd} main.py --instance {instance_id} --autostart\n"
        "echo [%date% %time%] Instance exited (code %errorlevel%). Restarting in 5s...\n"
        "timeout /t 5 /nobreak >nul\n"
        "goto restart_loop\n"
    )
    with open(launcher_file, "w", encoding="ascii", newline="\r\n") as handle:
        handle.write(content)


def _normalize_count(requested_count):
    try:
        count = int(requested_count)
    except (TypeError, ValueError):
        count = 1
    return max(1, count)


def _normalize_ports(count, ports=None):
    normalized = []
    if isinstance(ports, str):
        ports = [part.strip() for part in ports.split(",")]
    ports = list(ports or [])

    for index in range(1, count + 1):
        provided = ports[index - 1] if index - 1 < len(ports) else None
        if provided in (None, ""):
            normalized.append(None)
            continue
        try:
            normalized.append(int(provided))
        except (TypeError, ValueError):
            normalized.append(None)
    return normalized


def _normalize_scrcpy_max_fps(value):
    raw_value = str(value or "").strip().lower()
    if raw_value in ("", "auto", "none"):
        return "auto"
    try:
        return str(max(0, int(float(raw_value))))
    except (TypeError, ValueError):
        return "auto"


def _scrcpy_fps_literal(value):
    normalized = _normalize_scrcpy_max_fps(value)
    return f'"{normalized}"' if normalized == "auto" else normalized


def _read_instance_general_settings(config_path):
    settings = {
        "emulator_port": 5037,
        "instance_index": 1,
        "instance_count": 1,
        "scrcpy_max_fps": "auto",
    }
    if not os.path.exists(config_path):
        return settings

    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            content = handle.read()
    except Exception:
        return settings

    for key in ("emulator_port", "instance_index", "instance_count"):
        match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*(\d+)\s*$", content)
        if match:
            settings[key] = int(match.group(1))

    match = re.search(r'(?m)^scrcpy_max_fps\s*=\s*"([^"]*)"\s*$', content)
    if match:
        settings["scrcpy_max_fps"] = _normalize_scrcpy_max_fps(match.group(1))
        return settings

    match = re.search(r"(?m)^scrcpy_max_fps\s*=\s*([^\r\n#]+)\s*$", content)
    if match:
        settings["scrcpy_max_fps"] = _normalize_scrcpy_max_fps(match.group(1))

    return settings


def autostart_command(instance_id):
    return f"{_python_launcher()} main.py --instance {int(instance_id)} --autostart"


def configure_instances(requested_count, ports=None, scrcpy_max_fps="auto"):
    count = _normalize_count(requested_count)
    normalized_ports = _normalize_ports(count, ports)
    normalized_fps = _normalize_scrcpy_max_fps(scrcpy_max_fps)
    os.makedirs(os.path.join(PROJECT_ROOT, "instances"), exist_ok=True)
    configured = []

    for index in range(1, count + 1):
        root = instance_root(index)
        cfg_dir = instance_config_dir(index)
        os.makedirs(root, exist_ok=True)
        _copy_default_configs(cfg_dir)
        _seed_roster(root)

        config_path = os.path.join(cfg_dir, "general_config.toml")
        current_settings = _read_instance_general_settings(config_path)
        current_port = current_settings["emulator_port"]

        if normalized_ports[index - 1] is not None:
            current_port = normalized_ports[index - 1]

        _upsert_toml_scalar(config_path, "emulator_port", str(current_port))
        _upsert_toml_scalar(config_path, "instance_index", str(index))
        _upsert_toml_scalar(config_path, "instance_count", str(count))
        _upsert_toml_scalar(config_path, "scrcpy_max_fps", _scrcpy_fps_literal(normalized_fps))
        _generate_launcher(index)
        configured.append({
            "instance": index,
            "port": current_port,
            "instance_count": count,
            "scrcpy_max_fps": normalized_fps,
            "config_dir": cfg_dir,
            "runtime_root": root,
            "launcher_path": launcher_path(index),
            "launcher_exists": os.path.exists(launcher_path(index)),
            "autostart_command": autostart_command(index),
        })

    return configured


def list_instances():
    instances_dir = os.path.join(PROJECT_ROOT, "instances")
    rows = []
    if not os.path.isdir(instances_dir):
        return rows

    for name in sorted(os.listdir(instances_dir), key=lambda value: int(value) if str(value).isdigit() else 10**9):
        if not str(name).isdigit():
            continue
        instance_id = int(name)
        cfg_dir = instance_config_dir(instance_id)
        general_path = os.path.join(cfg_dir, "general_config.toml")
        settings = _read_instance_general_settings(general_path)
        rows.append({
            "instance": instance_id,
            "port": settings["emulator_port"],
            "instance_count": settings["instance_count"],
            "scrcpy_max_fps": settings["scrcpy_max_fps"],
            "config_dir": cfg_dir,
            "runtime_root": instance_root(instance_id),
            "launcher_path": launcher_path(instance_id),
            "launcher_exists": os.path.exists(launcher_path(instance_id)),
            "autostart_command": autostart_command(instance_id),
        })
    return rows


def setup_instances(requested_count=None):
    if requested_count in (None, "", "interactive"):
        raw = input("How many instances do you want to configure? ").strip()
        try:
            requested_count = int(raw)
        except ValueError:
            requested_count = 1

    count = _normalize_count(requested_count)
    prompted_ports = []
    for index in range(1, count + 1):
        current_port = 5037
        general_path = os.path.join(instance_config_dir(index), "general_config.toml")
        if os.path.exists(general_path):
            try:
                with open(general_path, "r", encoding="utf-8") as handle:
                    content = handle.read()
                match = re.search(r"(?m)^emulator_port\s*=\s*(\d+)\s*$", content)
                if match:
                    current_port = int(match.group(1))
            except Exception:
                pass
        prompt = f"Instance {index} emulator port [{current_port}]: "
        entered = input(prompt).strip()
        if entered:
            try:
                current_port = int(entered)
            except ValueError:
                print(f"Invalid port '{entered}', keeping {current_port}.")
        prompted_ports.append(current_port)

    return configure_instances(count, prompted_ports)
