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
    for name in os.listdir(DEFAULT_CONFIG_DIR):
        source = os.path.join(DEFAULT_CONFIG_DIR, name)
        destination = os.path.join(target_cfg_dir, name)
        if not os.path.isfile(source):
            continue
        if name in {"match_history.toml", "brawler_scan.json"}:
            continue
        if not os.path.exists(destination):
            shutil.copy2(source, destination)


def _seed_roster(instance_root_path):
    source = os.path.join(PROJECT_ROOT, "latest_brawler_data.json")
    destination = os.path.join(instance_root_path, "latest_brawler_data.json")
    if os.path.exists(source) and not os.path.exists(destination):
        shutil.copy2(source, destination)


def _python_launcher():
    venv_python = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv_python):
        return r".venv\Scripts\python.exe"
    return "python"


def _generate_launcher(instance_id):
    launcher_path = os.path.join(PROJECT_ROOT, f"start_{instance_id}.bat")
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
    with open(launcher_path, "w", encoding="ascii", newline="\r\n") as handle:
        handle.write(content)


def setup_instances(requested_count=None):
    if requested_count in (None, "", "interactive"):
        raw = input("How many instances do you want to configure? ").strip()
        try:
            requested_count = int(raw)
        except ValueError:
            requested_count = 1

    try:
        count = int(requested_count)
    except (TypeError, ValueError):
        count = 1
    count = max(1, count)

    os.makedirs(os.path.join(PROJECT_ROOT, "instances"), exist_ok=True)
    configured = []

    for index in range(1, count + 1):
        root = instance_root(index)
        cfg_dir = instance_config_dir(index)
        os.makedirs(root, exist_ok=True)
        _copy_default_configs(cfg_dir)
        _seed_roster(root)

        config_path = os.path.join(cfg_dir, "general_config.toml")
        current_port = 5037
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as handle:
                content = handle.read()
            match = re.search(r"(?m)^emulator_port\s*=\s*(\d+)\s*$", content)
            if match:
                current_port = int(match.group(1))

        prompt = f"Instance {index} emulator port [{current_port}]: "
        entered = input(prompt).strip()
        if entered:
            try:
                current_port = int(entered)
            except ValueError:
                print(f"Invalid port '{entered}', keeping {current_port}.")

        _upsert_toml_scalar(config_path, "emulator_port", str(current_port))
        _upsert_toml_scalar(config_path, "instance_index", str(index))
        _upsert_toml_scalar(config_path, "instance_count", str(count))
        _upsert_toml_scalar(config_path, 'scrcpy_max_fps', '"auto"')
        _generate_launcher(index)
        configured.append({"instance": index, "port": current_port, "config_dir": cfg_dir})

    return configured
