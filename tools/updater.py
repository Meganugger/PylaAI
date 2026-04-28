from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_REPO_OWNER = "Meganugger"
DEFAULT_REPO_NAME = "PylaAI"
DEFAULT_BRANCH = "main"
UPDATE_INFO_PATH = Path("cfg") / "update_info.json"

SKIPPED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    "logs",
    "build",
    "dist",
}

SKIPPED_FILES = {
    "adb.exe",
    "adbwinapi.dll",
    "adbwinusbapi.dll",
    "updater.exe",
}


def wait_for_enter(prompt="Press Enter to close...") -> None:
    try:
        input(prompt)
    except EOFError:
        pass


def print_green(message: str) -> None:
    if os.name != "nt":
        print(f"\033[92m{message}\033[0m")
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)
        kernel32.SetConsoleTextAttribute(handle, 0x0A)
        print(message)
        kernel32.SetConsoleTextAttribute(handle, 0x07)
    except Exception:
        print(message)


def repo_owner() -> str:
    return os.environ.get("PYLA_UPDATE_OWNER", DEFAULT_REPO_OWNER).strip() or DEFAULT_REPO_OWNER


def repo_name() -> str:
    return os.environ.get("PYLA_UPDATE_REPO", DEFAULT_REPO_NAME).strip() or DEFAULT_REPO_NAME


def repo_branch() -> str:
    return os.environ.get("PYLA_UPDATE_BRANCH", DEFAULT_BRANCH).strip() or DEFAULT_BRANCH


def repo_slug() -> str:
    return f"{repo_owner()}/{repo_name()}"


def latest_release_api() -> str:
    return f"https://api.github.com/repos/{repo_slug()}/releases/latest"


def branch_api() -> str:
    return f"https://api.github.com/repos/{repo_slug()}/commits/{repo_branch()}"


def branch_zip_url() -> str:
    return f"https://github.com/{repo_slug()}/archive/refs/heads/{repo_branch()}.zip"


def app_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "PylaAI-Updater",
    })
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.loads(response.read().decode("utf-8"))


def choose_release_download(release: dict) -> tuple[str, str]:
    assets = release.get("assets") or []
    zip_assets = [
        asset for asset in assets
        if str(asset.get("browser_download_url", "")).lower().endswith(".zip")
    ]
    if zip_assets:
        asset = zip_assets[0]
        return asset["browser_download_url"], asset.get("name") or "release asset"
    if release.get("zipball_url"):
        return release["zipball_url"], "GitHub source zip"
    return branch_zip_url(), f"{repo_branch()} branch zip"


def latest_download_url() -> tuple[str, str]:
    try:
        release = request_json(latest_release_api())
        return choose_release_download(release)
    except Exception as exc:
        if "404" in str(exc):
            print("No GitHub release update was found yet.")
        else:
            print("Could not check GitHub releases right now.")
        print(f"Checking the latest {repo_branch()} version instead.")
        return branch_zip_url(), f"{repo_branch()} branch zip"


def latest_branch_sha() -> str | None:
    try:
        data = request_json(branch_api())
        sha = str(data.get("sha") or "").strip()
        return sha or None
    except Exception:
        return None


def read_local_update_sha(project_dir: Path) -> str | None:
    info_path = project_dir / UPDATE_INFO_PATH
    if not info_path.exists():
        return None
    try:
        data = json.loads(info_path.read_text(encoding="utf-8-sig"))
        sha = str(data.get("branch_sha") or data.get("main_sha") or "").strip()
        return sha or None
    except Exception:
        return None


def write_local_update_info(project_dir: Path, sha: str | None) -> None:
    if not sha:
        return
    info_path = project_dir / UPDATE_INFO_PATH
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        json.dumps({
            "branch_sha": sha,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "repo": repo_slug(),
            "branch": repo_branch(),
        }, indent=4),
        encoding="utf-8",
    )


def download_file(url: str, destination: Path, label: str) -> Path:
    print(f"Downloading latest PylaAI update ({label})...")
    request = urllib.request.Request(url, headers={"User-Agent": "PylaAI-Updater"})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def parse_simple_toml(text: str) -> dict:
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if key:
            values[key] = raw_value
    return values


def merge_toml_text(new_text: str, old_text: str) -> str:
    old_values = parse_simple_toml(old_text)
    new_values = parse_simple_toml(new_text)
    merged_lines = []
    used_keys = set()
    key_pattern = re.compile(r"^(\s*)([A-Za-z0-9_\-]+)(\s*=\s*)(.*?)(\s*(?:#.*)?)$")

    for line in new_text.splitlines():
        match = key_pattern.match(line)
        if not match:
            merged_lines.append(line)
            continue
        prefix, key, equals, new_value, suffix = match.groups()
        if key in old_values:
            merged_lines.append(f"{prefix}{key}{equals}{old_values[key]}{suffix}")
            used_keys.add(key)
        else:
            merged_lines.append(line)

    missing_user_keys = [key for key in old_values if key not in used_keys and key not in new_values]
    if missing_user_keys:
        if merged_lines and merged_lines[-1].strip():
            merged_lines.append("")
        merged_lines.append("# Kept from your previous config")
        for key in missing_user_keys:
            merged_lines.append(f"{key} = {old_values[key]}")

    return "\n".join(merged_lines).rstrip() + "\n"


def merge_json_data(new_data, old_data):
    if isinstance(new_data, dict) and isinstance(old_data, dict):
        merged = dict(new_data)
        for key, old_value in old_data.items():
            if key in merged:
                merged[key] = merge_json_data(merged[key], old_value)
            else:
                merged[key] = old_value
        return merged
    return old_data


def find_project_root(extracted_dir: Path) -> Path:
    if (extracted_dir / "main.py").exists() and (extracted_dir / "cfg").exists():
        return extracted_dir
    candidates = [
        path for path in extracted_dir.rglob("main.py")
        if (path.parent / "cfg").exists()
    ]
    if not candidates:
        raise FileNotFoundError("Downloaded update does not look like a PylaAI project.")
    candidates.sort(key=lambda path: len(path.parts))
    return candidates[0].parent


def backup_preserved_files(project_dir: Path, backup_dir: Path) -> None:
    cfg_dir = project_dir / "cfg"
    if not cfg_dir.exists():
        return
    for source in cfg_dir.iterdir():
        if source.suffix.lower() not in (".toml", ".json") or not source.is_file():
            continue
        relative_path = source.relative_to(project_dir)
        destination = backup_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        print(f"Preserved {relative_path}")


def restore_preserved_files(project_dir: Path, backup_dir: Path) -> None:
    cfg_backup = backup_dir / "cfg"
    if not cfg_backup.exists():
        return
    for old_config in cfg_backup.iterdir():
        if old_config.suffix.lower() not in (".toml", ".json") or not old_config.is_file():
            continue
        relative_path = old_config.relative_to(backup_dir)
        destination = project_dir / relative_path
        if destination.exists() and old_config.suffix.lower() == ".toml":
            merged = merge_toml_text(
                destination.read_text(encoding="utf-8-sig"),
                old_config.read_text(encoding="utf-8-sig"),
            )
            destination.write_text(merged, encoding="utf-8")
            print(f"Merged {relative_path}")
        elif destination.exists() and old_config.suffix.lower() == ".json":
            try:
                new_data = json.loads(destination.read_text(encoding="utf-8-sig"))
                old_data = json.loads(old_config.read_text(encoding="utf-8-sig"))
                merged = merge_json_data(new_data, old_data)
                destination.write_text(json.dumps(merged, indent=4), encoding="utf-8")
                print(f"Merged {relative_path}")
            except Exception:
                shutil.copy2(old_config, destination)
                print(f"Restored {relative_path}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old_config, destination)
            print(f"Restored {relative_path}")


def should_skip(relative_path: Path, source: Path) -> bool:
    if set(relative_path.parts) & SKIPPED_DIRS:
        return True
    if source.is_file() and relative_path.name.lower() in SKIPPED_FILES:
        return True
    return False


def copy_update_files(source_root: Path, project_dir: Path) -> None:
    for source in source_root.rglob("*"):
        relative_path = source.relative_to(source_root)
        if should_skip(relative_path, source):
            continue
        destination = project_dir / relative_path
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, destination)
        except PermissionError:
            print(f"Skipped locked file: {relative_path}")


def main() -> int:
    if "--help" in sys.argv or "-h" in sys.argv:
        print("PylaAI updater")
        print("Downloads the latest GitHub update and keeps your cfg settings.")
        print("Use --force to reinstall even when this folder is already current.")
        print("Use --smoke-test to verify that the updater can see this project.")
        return 0

    project_dir = app_dir()
    print("=" * 50)
    print("PylaAI Updater")
    print("=" * 50)
    print(f"Project folder: {project_dir}")
    print(f"Update source: {repo_slug()} [{repo_branch()}]")

    if not (project_dir / "main.py").exists():
        print("The updater must be inside the PylaAI project folder next to main.py.")
        wait_for_enter()
        return 1

    if "--smoke-test" in sys.argv:
        print("Smoke test passed. Updater can see the PylaAI project folder.")
        return 0

    latest_sha = latest_branch_sha()
    local_sha = read_local_update_sha(project_dir)
    if latest_sha and local_sha == latest_sha and "--force" not in sys.argv:
        print_green("You're on the latest version.")
        wait_for_enter()
        return 0

    temp_dir = Path(tempfile.mkdtemp(prefix="pyla_update_"))
    backup_dir = temp_dir / "preserved_user_files"
    zip_path = temp_dir / "latest_pylaai.zip"

    try:
        backup_preserved_files(project_dir, backup_dir)
        url, label = latest_download_url()
        download_file(url, zip_path, label)
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting update...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)
        source_root = find_project_root(extract_dir)
        print(f"Installing update from: {source_root}")
        copy_update_files(source_root, project_dir)
        restore_preserved_files(project_dir, backup_dir)
        write_local_update_info(project_dir, latest_sha)
        print("")
        print("Update completed.")
        print("Your cfg settings were kept, with new config keys added.")
        print("Run setup.exe if the update added new dependencies.")
        wait_for_enter()
        return 0
    except Exception as exc:
        print("")
        print(f"Update failed: {exc}")
        if backup_dir.exists():
            try:
                restore_preserved_files(project_dir, backup_dir)
            except Exception:
                pass
        wait_for_enter()
        return 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
