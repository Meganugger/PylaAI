import inspect
import json
import os
import re
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from brawlstars_api import fetch_player_profile, normalize_brawler_name, normalize_player_tag
from farm_roster import build_farm_plan, load_farm_state, mask_api_key, normalize_farm_mode, normalize_farm_strategy, now_iso, save_farm_state
from gui.api import check_if_exists
from gui.config_store import load_config
from instance_support import (
    autostart_command,
    configure_instances,
    current_config_dir,
    current_instance_id,
    current_runtime_root,
    launcher_path,
    list_instances,
)
from lobby_automation import LobbyAutomation
from stage_manager import StageManager
from utils import (
    get_brawler_data_path,
    load_brawlers_info,
    load_toml_as_dict,
    resolve_cfg_path,
    save_brawler_data,
    save_dict_as_toml,
)


GAMEMODES = [
    {"value": "knockout", "label": "Knockout"},
    {"value": "brawlball", "label": "Brawl Ball"},
    {"value": "gemgrab", "label": "Gem Grab"},
    {"value": "showdown", "label": "Showdown"},
    {"value": "basketbrawl", "label": "Basket Brawl"},
    {"value": "wipeout", "label": "Wipeout"},
    {"value": "bounty", "label": "Bounty"},
    {"value": "hotzone", "label": "Hot Zone"},
    {"value": "heist", "label": "Heist"},
    {"value": "duels", "label": "Duels"},
    {"value": "5v5", "label": "5v5"},
    {"value": "other", "label": "Other"},
]

EMULATORS = ["LDPlayer", "BlueStacks", "MEmu", "MuMu", "Others"]


class QtBridge(QObject):
    stateChanged = Signal("QVariantMap")
    rosterChanged = Signal("QVariantList")
    liveDataChanged = Signal("QVariantMap")
    historyChanged = Signal("QVariantList")
    logsChanged = Signal("QVariantList")
    notificationRaised = Signal(str, str)
    sessionSummaryReady = Signal("QVariantMap")
    updateStatusChanged = Signal("QVariantMap")

    def __init__(self, version_str, brawlers, pyla_main_fn, login_fn=None, saved_brawler_data=None):
        super().__init__()
        self._version_str = str(version_str).strip()
        self._pyla_main = pyla_main_fn
        self._login_fn = login_fn
        self._input_brawlers = list(brawlers or [])
        self._bot_thread = None
        self._bot_stop_event = None
        self._bot_pause_event = None
        self._bot_stop_requested = False
        self._bot_paused = False
        self._bot_control_state = "stopped"
        self._live_data = {}
        self._session_summary = None
        self._event_log = []
        self._logged_in = False
        self._live_lock = threading.Lock()

        self.bot_config = self._load_bot_config()
        self.general_config = self._load_general_config()
        self._farm_state = load_farm_state()
        self.time_config = load_config("time")
        self.login_config = load_toml_as_dict("cfg/login.toml")
        self._brawler_load_error = ""
        self._brawler_load_state = {"state": "loading", "message": "Loading brawlers...", "count": 0}
        try:
            self.brawlers_info = load_brawlers_info()
        except Exception as exc:
            self.brawlers_info = {}
            self._brawler_load_error = str(exc)
        self._all_brawlers = self._discover_brawler_names(self._input_brawlers)
        self._refresh_brawler_load_state()
        self._update_status = self._default_update_status("idle")
        self.brawlers_data = self._normalize_roster(saved_brawler_data or self._load_saved_roster())
        self.capabilities = {
            "visual_overlay": os.path.exists("visual_overlay.py"),
            "advanced_live": all(os.path.exists(path) for path in ("behavior_tree.py", "bt_combat.py")),
            "brawler_scan": hasattr(LobbyAutomation, "scan_all_brawlers"),
            "quest_farm": hasattr(StageManager, "_handle_quest_rotation"),
            "quest_scan": hasattr(LobbyAutomation, "scan_quest_brawlers"),
            "updater": os.path.exists(os.path.join("tools", "updater.py")) or os.path.exists("updater.exe"),
            "runtime_preflight": os.path.exists(os.path.join("tools", "runtime_preflight.py")),
            "performance_profiles": os.path.exists("performance_profile.py"),
        }

        self._validate_existing_login()

    @staticmethod
    def _as_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _as_float(value, default=0.0):
        try:
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _coerce_int(value, fallback, minimum=None):
        try:
            parsed = int(str(value).strip())
        except Exception:
            parsed = fallback
        if minimum is not None:
            parsed = max(minimum, parsed)
        return parsed

    @staticmethod
    def _normalize_scrcpy_max_fps(value):
        raw_value = str(value or "").strip().lower()
        if raw_value in ("", "auto", "none"):
            return "auto"
        try:
            return str(max(0, int(float(raw_value))))
        except Exception:
            return "auto"

    @staticmethod
    def _gamemode_type_for(gamemode):
        normalized = str(gamemode or "").strip().lower()
        if normalized in {"basketbrawl", "5v5", "brawlball_5v5"}:
            return 5
        return 3

    def _load_bot_config(self):
        config = load_config("bot")
        config.setdefault("gamemode", "knockout")
        config.setdefault("gamemode_type", 3)
        config.setdefault("smart_trophy_farm", "no")
        config.setdefault("trophy_farm_mode", "manual")
        config.setdefault("trophy_farm_target", 500)
        config.setdefault("trophy_farm_strategy", "lowest_first")
        config.setdefault("trophy_farm_excluded", [])
        config.setdefault("quest_farm_enabled", "no")
        config.setdefault("quest_farm_mode", "games")
        config.setdefault("quest_farm_excluded", [])
        return config

    def _load_general_config(self):
        config = load_config("general")
        config.setdefault("current_emulator", "LDPlayer")
        config.setdefault("map_orientation", "vertical")
        config.setdefault("performance_profile", "balanced")
        config.setdefault("auto_update_checks", "no")
        config.setdefault("auto_update_ignored_sha", "")
        config.setdefault("brawlstars_api_key", "")
        config.setdefault("brawlstars_player_tag", "")
        config.setdefault("target_ips", 60)
        config.setdefault("scrcpy_max_fps", 60)
        return config

    def _discover_brawler_names(self, names):
        discovered = set()
        for name in names or []:
            value = str(name or "").strip().lower()
            if value:
                discovered.add(value)
        for name in (self.brawlers_info or {}).keys():
            value = str(name or "").strip().lower()
            if value:
                discovered.add(value)
        icon_dir = os.path.join("api", "assets", "brawler_icons")
        if os.path.isdir(icon_dir):
            for filename in os.listdir(icon_dir):
                stem, ext = os.path.splitext(filename)
                if ext.lower() == ".png":
                    value = stem.strip().lower()
                    if value:
                        discovered.add(value)
        return sorted(discovered)

    def _refresh_brawler_load_state(self):
        count = len(self._all_brawlers or [])
        if self._brawler_load_error:
            self._brawler_load_state = {
                "state": "error",
                "message": f"Could not load brawler data: {self._brawler_load_error}",
                "count": count,
            }
        elif count <= 0:
            self._brawler_load_state = {
                "state": "empty",
                "message": "No brawlers were found in cfg/brawlers_info.json or the icon folders.",
                "count": 0,
            }
        else:
            self._brawler_load_state = {
                "state": "ready",
                "message": f"{count} brawlers loaded.",
                "count": count,
            }

    @staticmethod
    def _is_enabled(value):
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_farm_strategy(value):
        return normalize_farm_strategy(value)

    def _validate_existing_login(self):
        api_base_url = str(self.general_config.get("api_base_url", "localhost")).strip()
        if api_base_url == "localhost":
            self._logged_in = True
            return
        auth_key = str(self.login_config.get("key", "")).strip()
        if auth_key:
            try:
                self._logged_in = bool(check_if_exists(auth_key))
            except Exception:
                self._logged_in = False

    @staticmethod
    def _format_version_tag(version_str):
        raw = str(version_str).strip()
        if not raw:
            return "PylaAI"
        local_labels = {
            "main": "main",
            "performance": "performance",
            "strongestbot": "strongest-bot",
            "strongestbotfull": "strongest-bot-full",
            "strongestbotrl": "strongest-bot-rl",
        }
        if "+" in raw:
            base, local = raw.split("+", 1)
            pretty_local = local_labels.get(local.lower(), local.replace("_", "-"))
            return f"PylaAI {pretty_local}  v{base}"
        if raw.lower().startswith("v"):
            return f"PylaAI {raw}"
        return f"PylaAI v{raw}"

    @staticmethod
    def _load_saved_roster():
        roster_path = get_brawler_data_path()
        if not os.path.exists(roster_path):
            return []
        try:
            with open(roster_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _normalize_roster(self, roster):
        normalized = []
        for entry in roster or []:
            if not isinstance(entry, dict):
                continue
            brawler = str(entry.get("brawler", "")).strip().lower()
            if not brawler:
                continue
            normalized.append({
                "brawler": brawler,
                "push_until": self._as_int(entry.get("push_until", self.general_config.get("auto_push_target_trophies", 1000)), 0),
                "trophies": self._as_int(entry.get("trophies", 0), 0),
                "wins": self._as_int(entry.get("wins", 0), 0),
                "type": str(entry.get("type", "trophies") or "trophies"),
                "automatically_pick": bool(entry.get("automatically_pick", True)),
                "win_streak": self._as_int(entry.get("win_streak", 0), 0),
                "manual_trophies": bool(entry.get("manual_trophies", False)),
            })
        return normalized

    def _emit_state(self):
        self.stateChanged.emit(self.initialState())

    def _emit_roster(self):
        self.rosterChanged.emit(self.getRoster())

    def _emit_history(self):
        self.historyChanged.emit(self.getHistory())

    def _emit_logs(self):
        self.logsChanged.emit(self.getLogs())

    def _push_log(self, level, message):
        text = str(message or "").strip()
        if not text:
            return
        self._event_log.append({
            "level": str(level or "info"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "message": text,
        })
        self._event_log = self._event_log[-120:]
        self._emit_logs()

    def sync_runtime_roster(self, roster, emit_history=False):
        normalized = self._normalize_roster(roster)
        roster_changed = normalized != self.brawlers_data
        if roster_changed:
            self.brawlers_data = normalized
            self._emit_roster()
            self._emit_state()
        if emit_history:
            self._emit_history()

    def _icon_url_for(self, brawler):
        icon_path = os.path.abspath(os.path.join("api", "assets", "brawler_icons", f"{brawler}.png"))
        return f"file:///{icon_path.replace(os.sep, '/')}" if os.path.exists(icon_path) else ""

    def _brawler_scan_data(self):
        scan_path = resolve_cfg_path("cfg/brawler_scan.json")
        if not os.path.exists(scan_path):
            return {}
        try:
            with open(scan_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data.get("brawlers", {}) if isinstance(data, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _save_brawler_scan_data(data):
        scan_path = resolve_cfg_path("cfg/brawler_scan.json")
        payload = {"brawlers": data if isinstance(data, dict) else {}}
        with open(scan_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)

    @staticmethod
    def _slug_name(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    def _canonical_brawler_name(self, value, scan_data=None):
        target = self._slug_name(value)
        if not target:
            return ""
        sources = list(self._all_brawlers)
        sources.extend(
            entry.get("brawler", "") for entry in self.brawlers_data
            if isinstance(entry, dict)
        )
        if isinstance(scan_data, dict):
            sources.extend(scan_data.keys())
        for name in sources:
            if self._slug_name(name) == target:
                return str(name).strip().lower()
        return str(value or "").strip().lower()

    def _scan_entry_for_brawler(self, brawler, scan_data=None):
        data = scan_data if isinstance(scan_data, dict) else self._brawler_scan_data()
        direct = data.get(brawler)
        if isinstance(direct, dict):
            return direct
        target = self._slug_name(brawler)
        for scan_name, scan_entry in data.items():
            if self._slug_name(scan_name) == target and isinstance(scan_entry, dict):
                return scan_entry
        return {}

    def _normalize_brawler_set(self, values, scan_data=None):
        return {
            canonical for canonical in (
                self._canonical_brawler_name(value, scan_data=scan_data) for value in values
            )
            if canonical
        }

    def _resolve_internal_brawler_name(self, external_name):
        target = self._slug_name(external_name)
        if not target:
            return None
        for name in self._all_brawlers:
            if self._slug_name(name) == target or self._slug_name(name.title()) == target:
                return name
        return None

    def _fetch_player_brawlers_from_api(self):
        api_key = str(self.general_config.get("brawlstars_api_key", "")).strip()
        player_tag = str(self.general_config.get("brawlstars_player_tag", "")).strip()
        profile = fetch_player_profile(api_key, player_tag, timeout=15)
        rows = []
        for brawler, entry in profile.get("brawlers", {}).items():
            row = dict(entry)
            row["name"] = row.get("name") or brawler
            rows.append(row)
        return rows

    def _build_brawler_payload(self):
        scan_data = self._brawler_scan_data()
        roster_lookup = {
            self._canonical_brawler_name(entry["brawler"], scan_data=scan_data): entry
            for entry in self.brawlers_data
            if isinstance(entry, dict) and entry.get("brawler")
        }
        items = []
        for name in self._all_brawlers:
            canonical = self._canonical_brawler_name(name, scan_data=scan_data)
            scan_entry = self._scan_entry_for_brawler(canonical, scan_data)
            selected = roster_lookup.get(canonical, {})
            trophies = selected.get("trophies", scan_entry.get("trophies", 0))
            items.append({
                "name": canonical,
                "displayName": canonical.title(),
                "icon": self._icon_url_for(canonical),
                "selected": bool(selected),
                "trophies": int(trophies or 0),
                "pushUntil": int(selected.get("push_until", self.general_config.get("auto_push_target_trophies", 1000)) or 0),
                "wins": int(selected.get("wins", 0) or 0),
                "winStreak": int(selected.get("win_streak", 0) or 0),
                "type": str(selected.get("type", "trophies") or "trophies"),
                "autoPick": bool(selected.get("automatically_pick", True)),
                "manualTrophies": bool(selected.get("manual_trophies", False)),
                "holdAttack": float(self.brawlers_info.get(canonical, {}).get("hold_attack", 0) or 0),
            })
        return items

    def _history_rows(self):
        rows = []
        history = load_toml_as_dict("cfg/match_history.toml")
        if not isinstance(history, dict):
            return rows
        for brawler, data in history.items():
            if not isinstance(data, dict):
                continue
            if str(brawler).lower() == "total":
                continue
            wins = int(data.get("victory", 0) or 0)
            defeats = int(data.get("defeat", 0) or 0)
            draws = int(data.get("draw", 0) or 0)
            total = wins + defeats + draws
            winrate = round((wins / total) * 100, 1) if total else 0.0
            rows.append({
                "brawler": brawler,
                "displayName": brawler.title(),
                "wins": wins,
                "defeats": defeats,
                "draws": draws,
                "matches": total,
                "winrate": winrate,
                "icon": self._icon_url_for(brawler),
            })
        rows.sort(key=lambda item: (-item["matches"], item["displayName"]))
        return rows

    def _match_history_map(self):
        history = load_toml_as_dict("cfg/match_history.toml")
        return history if isinstance(history, dict) else {}

    def _farm_settings(self, overrides=None):
        source = dict(self.bot_config)
        if isinstance(overrides, dict):
            source.update(overrides)
        target = self._coerce_int(source.get("trophy_farm_target", 500), 500, minimum=0)
        strategy = self._normalize_farm_strategy(source.get("trophy_farm_strategy", "lowest_first"))
        mode = normalize_farm_mode(source.get("trophy_farm_mode", "manual"))
        excluded = self._normalize_brawler_set(
            source.get("trophy_farm_excluded", []),
            scan_data=self._brawler_scan_data(),
        )
        return {
            "mode": mode,
            "target": target,
            "strategy": strategy,
            "excluded": excluded,
        }

    def _farm_plan(self, overrides=None):
        settings = self._farm_settings(overrides)
        return build_farm_plan(
            all_brawlers=self._all_brawlers,
            selected_roster=self.brawlers_data,
            scan_data=self._brawler_scan_data(),
            farm_state=self._farm_state,
            target=settings["target"],
            strategy=settings["strategy"],
            excluded=settings["excluded"],
            mode=settings["mode"],
            history=self._match_history_map(),
        )

    def _farm_status_payload(self, overrides=None):
        plan = self._farm_plan(overrides)
        api_roster = self._farm_state.get("api_roster", {}) if isinstance(self._farm_state, dict) else {}
        return {
            "mode": plan["mode"],
            "target": plan["target"],
            "strategy": plan["strategy"],
            "queueCount": len(plan["queue"]),
            "rosterCount": len(plan["rows"]),
            "emptyReason": plan["empty_reason"],
            "apiLoadedCount": len(api_roster) if isinstance(api_roster, dict) else 0,
            "apiPlayerName": str(self._farm_state.get("api_player_name", "") or ""),
            "apiPlayerTag": str(self._farm_state.get("api_player_tag", "") or ""),
            "apiLastRefresh": str(self._farm_state.get("api_last_refresh", "") or ""),
            "apiKeyMasked": mask_api_key(self.general_config.get("brawlstars_api_key", "")),
        }

    def _build_trophy_farm_roster(self, overrides=None):
        plan = self._farm_plan(overrides)
        target = plan["target"]
        strategy = plan["strategy"]
        roster = []
        for item in plan["queue"]:
            roster.append({
                "brawler": item["brawler"],
                "push_until": target,
                "trophies": item["trophies"],
                "wins": 0,
                "type": "trophies",
                "automatically_pick": True,
                "win_streak": 0,
                "manual_trophies": True,
                "selection_method": f"farm_{plan['mode']}",
            })

        return roster, plan["queue"], target, strategy

    def _build_trophy_farm_preview(self, overrides=None):
        plan = self._farm_plan(overrides)
        queue = plan["queue"]
        target = plan["target"]
        strategy = plan["strategy"]
        preview = []
        for index, item in enumerate(queue, start=1):
            preview.append({
                "order": index,
                "brawler": item["brawler"],
                "displayName": item["brawler"].title(),
                "icon": self._icon_url_for(item["brawler"]),
                "trophies": self._as_int(item.get("trophies", 0), 0),
                "winrate": self._as_int(item.get("winrate", 50), 50),
                "matches": self._as_int(item.get("total_games", 0), 0),
                "target": target,
                "strategy": strategy,
                "mode": plan["mode"],
                "source": item.get("source", plan["mode"]),
                "owned": bool(item.get("owned", True)),
                "included": bool(item.get("included", True)),
                "reason": item.get("reason", ""),
            })
        return preview

    def _detect_adb_ports(self):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        adb_candidates = [
            os.path.join(project_root, "adb.exe"),
            "adb.exe",
            "adb",
        ]
        adb_command = next((candidate for candidate in adb_candidates if candidate == "adb" or candidate == "adb.exe" or os.path.exists(candidate)), None)
        if not adb_command:
            return []

        try:
            result = subprocess.run(
                [adb_command, "devices"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            return []

        ports = []
        seen = set()
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line or line.lower().startswith("list of devices"):
                continue
            parts = line.split()
            if len(parts) < 2 or parts[1].lower() != "device":
                continue
            serial = parts[0]
            match = re.fullmatch(r"(?:127\.0\.0\.1|localhost):(\d+)", serial)
            if not match:
                match = re.fullmatch(r"emulator-(\d+)", serial)
            if not match:
                continue
            port = self._coerce_int(match.group(1), 0, minimum=1)
            if port and port not in seen:
                seen.add(port)
                ports.append(port)
        return sorted(ports)

    def _suggest_multi_instance_ports(self, count, current_port, detected_ports=None):
        total = self._coerce_int(count, 1, minimum=1)
        detected = list(detected_ports if detected_ports is not None else self._detect_adb_ports())
        base_port = self._coerce_int(current_port, detected[0] if detected else 5037, minimum=1)
        ports = []

        def add_port(value):
            port = self._coerce_int(value, 0, minimum=1)
            if port and port not in ports:
                ports.append(port)

        if not detected or base_port != 5037 or base_port in detected:
            add_port(base_port)
        for port in detected:
            add_port(port)

        candidate = ports[0] if ports else base_port
        while len(ports) < total:
            add_port(candidate)
            candidate += 1

        return ports[:total]

    def _multi_instance_state(self):
        configured_count = self._coerce_int(self.general_config.get("instance_count", 1), 1, minimum=1)
        rows = list_instances(max_instance_count=configured_count)
        current_id = current_instance_id(self._coerce_int(self.general_config.get("instance_index", 1), 1, minimum=1))
        current_port = self._coerce_int(self.general_config.get("emulator_port", 5037), 5037, minimum=1)
        current_fps = self._normalize_scrcpy_max_fps(self.general_config.get("scrcpy_max_fps", "auto"))
        detected_ports = self._detect_adb_ports()
        suggested_ports = self._suggest_multi_instance_ports(configured_count, current_port, detected_ports=detected_ports)
        ports_csv = ", ".join(str(row.get("port", 5037)) for row in rows) if rows else str(current_port)
        current_launcher = launcher_path(current_id)

        summary_lines = [
            "Quick start:",
            f"1. Set the total instances to {configured_count}.",
            f"2. Make sure this window is using port {current_port}.",
            "3. Use Auto Fill Ports to prefer live ADB ports first.",
            "4. Click Create / Update Launchers.",
            "5. Start one bot with each generated start_n.bat file.",
            "",
            f"Active instance: {current_id}",
            f"Configured instances: {configured_count}",
            f"This instance port: {current_port}",
            f"scrcpy max FPS: {current_fps}",
            f"ADB detected ports: {', '.join(str(port) for port in detected_ports) if detected_ports else 'none detected'}",
            f"Launcher for this window: {os.path.basename(current_launcher)}",
        ]
        if rows:
            summary_lines.append("")
            summary_lines.append("Generated launchers:")
            for row in rows:
                status = "ready" if row.get("launcher_exists") else "missing launcher"
                summary_lines.append(
                    f"{os.path.basename(str(row.get('launcher_path', '') or ''))} -> port {row.get('port', 5037)} | "
                    f"scrcpy {row.get('scrcpy_max_fps', 'auto')} | {status}"
                )
        else:
            summary_lines.append("")
            summary_lines.append("No generated launchers yet. Use Create / Update Launchers to make them.")

        summary_lines.extend([
            "",
            f"Config dir: {current_config_dir()}",
            f"Runtime root: {current_runtime_root()}",
            f"Autostart command: {autostart_command(current_id)}",
        ])

        return {
            "currentInstance": current_id,
            "instanceCount": configured_count,
            "currentPort": current_port,
            "scrcpyMaxFps": current_fps,
            "portsCsv": ports_csv,
            "detectedAdbPorts": detected_ports,
            "detectedPortsCsv": ", ".join(str(port) for port in detected_ports),
            "suggestedPortsCsv": ", ".join(str(port) for port in suggested_ports),
            "configDir": current_config_dir(),
            "runtimeRoot": current_runtime_root(),
            "launcherPath": current_launcher,
            "launcherExists": os.path.exists(current_launcher),
            "autostartCommand": autostart_command(current_id),
            "instances": rows,
            "summary": "\n".join(summary_lines),
        }

    def _notification(self, level, message):
        self._push_log(level, message)
        self.notificationRaised.emit(level, message)

    @staticmethod
    def _tool_path(*parts):
        return Path(os.getcwd(), *parts)

    def _performance_profiles_payload(self):
        try:
            from performance_profile import PERFORMANCE_PROFILES

            rows = []
            labels = {
                "balanced": "Balanced",
                "low_end": "Low Quality",
                "quality": "High Quality",
            }
            for key in ("balanced", "low_end", "quality"):
                profile = PERFORMANCE_PROFILES.get(key, {})
                rows.append({
                    "value": key,
                    "label": profile.get("label", labels.get(key, key)),
                    "description": profile.get("description", ""),
                })
            return rows
        except Exception:
            return [
                {
                    "value": "balanced",
                    "label": "Balanced",
                    "description": "Recommended mode. Balances speed, accuracy, and resource usage for most users.",
                },
                {
                    "value": "low_end",
                    "label": "Low Quality",
                    "description": "Fastest and lightest mode for weaker PCs, with less visual detail for detection.",
                },
                {
                    "value": "quality",
                    "label": "High Quality",
                    "description": "Most detailed mode for stronger PCs when runtime speed remains stable.",
                },
            ]

    def _default_update_status(self, state="idle", error=""):
        return {
            "ok": False,
            "state": state,
            "currentVersion": str(self.general_config.get("pyla_version", self._version_str) or self._version_str),
            "localSha": "",
            "latestSha": "",
            "availableVersion": "",
            "updateAvailable": False,
            "source": "Meganugger/PylaAI [main]",
            "summary": "",
            "changelog": "",
            "url": "",
            "error": error,
            "ignored": False,
            "autoUpdateEnabled": self._is_enabled(self.general_config.get("auto_update_checks", "no")),
        }

    def _emit_update_status(self):
        self.updateStatusChanged.emit(dict(self._update_status))

    def _tool_status_payload(self):
        updater_script = self._tool_path("tools", "updater.py")
        updater_exe = self._tool_path("updater.exe")
        preflight_script = self._tool_path("tools", "runtime_preflight.py")
        profile_module = self._tool_path("performance_profile.py")
        local_sha = ""
        update_source = "Meganugger/PylaAI [main]"
        try:
            from tools import updater

            local_sha = updater.read_installed_update_sha(Path(os.getcwd())) or ""
            update_source = f"{updater.repo_slug()} [{updater.repo_branch(Path(os.getcwd()))}]"
        except Exception:
            pass
        return {
            "updaterAvailable": updater_script.exists() or updater_exe.exists(),
            "updaterScript": str(updater_script),
            "updaterExe": str(updater_exe),
            "runtimePreflightAvailable": preflight_script.exists(),
            "runtimePreflightScript": str(preflight_script),
            "performanceProfilesAvailable": profile_module.exists(),
            "localUpdateSha": local_sha,
            "updateSource": update_source,
            "autoUpdateEnabled": self._is_enabled(self.general_config.get("auto_update_checks", "no")),
            "ignoredUpdateSha": str(self.general_config.get("auto_update_ignored_sha", "") or ""),
        }

    @Slot(result="QVariantMap")
    def getToolStatus(self):
        return self._tool_status_payload()

    @Slot(result="QVariantMap")
    def runRuntimePreflight(self):
        script = self._tool_path("tools", "runtime_preflight.py")
        if not script.exists():
            self._notification("error", "Runtime preflight is not available on this branch.")
            return {"ok": False, "output": "runtime_preflight.py missing"}
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=os.getcwd(),
                text=True,
                capture_output=True,
                timeout=90,
            )
            output = (result.stdout or "") + (result.stderr or "")
            ok = result.returncode == 0
            self._notification(
                "success" if ok else "error",
                "Runtime preflight passed." if ok else "Runtime preflight found a problem. See Logs.",
            )
            if output.strip():
                self._push_log("info" if ok else "error", output.strip()[-900:])
            return {"ok": ok, "code": result.returncode, "output": output}
        except Exception as exc:
            self._notification("error", f"Runtime preflight failed to run: {exc}")
            return {"ok": False, "output": str(exc)}

    @Slot(str, result="QVariantMap")
    def applyPerformanceProfile(self, profile):
        try:
            from performance_profile import apply_performance_profile

            result = apply_performance_profile(profile or "balanced", save=True)
            self.general_config = self._load_general_config()
            self.bot_config = self._load_bot_config()
            self._emit_state()
            self._notification(
                "success",
                f"Applied {result['profile']} performance profile. Restart the bot before playing.",
            )
            return {
                "ok": True,
                "profile": result["profile"],
                "description": result["description"],
            }
        except Exception as exc:
            self._notification("error", f"Could not apply performance profile: {exc}")
            return {"ok": False, "error": str(exc)}

    @Slot(bool, result="QVariantMap")
    def launchUpdater(self, force=False):
        status = self._tool_status_payload()
        command = None
        updater_exe = Path(status["updaterExe"])
        updater_script = Path(status["updaterScript"])
        if updater_exe.exists():
            command = [str(updater_exe)]
        elif updater_script.exists():
            command = [sys.executable, str(updater_script)]
        if command is None:
            self._notification("error", "Updater is not available on this branch.")
            return {"ok": False, "error": "updater missing"}
        if force:
            command.append("--force")
        try:
            self._update_status.update({"state": "installing", "error": ""})
            self._emit_update_status()
            creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
            subprocess.Popen(command, cwd=os.getcwd(), creationflags=creationflags)
            self._update_status.update({"state": "restart required"})
            self._emit_update_status()
            self._notification(
                "info",
                "Updater launched in a separate console. Close PylaAI before installing over this running copy.",
            )
            return {"ok": True, "command": " ".join(command)}
        except Exception as exc:
            self._notification("error", f"Could not launch updater: {exc}")
            return {"ok": False, "error": str(exc)}

    @Slot(result="QVariantMap")
    def checkForUpdates(self):
        self._update_status = self._default_update_status("checking")
        self._emit_update_status()
        try:
            from tools import updater

            status = updater.build_update_status(Path(os.getcwd()))
            ignored_sha = str(self.general_config.get("auto_update_ignored_sha", "") or "")
            status["ignored"] = bool(status.get("latestSha") and status.get("latestSha") == ignored_sha)
            status["autoUpdateEnabled"] = self._is_enabled(self.general_config.get("auto_update_checks", "no"))
            self._update_status = status
            self._emit_update_status()
            if status.get("updateAvailable") and not status.get("ignored"):
                self._notification("info", "A newer GitHub branch revision is available.")
            elif status.get("updateAvailable") and status.get("ignored"):
                self._notification("info", "A newer GitHub branch revision is available, but this version is ignored.")
            elif status.get("ok"):
                self._notification("success", "This folder is already marked as the latest GitHub branch revision.")
            else:
                self._notification("warning", "Could not read the latest GitHub revision right now.")
            return dict(self._update_status)
        except Exception as exc:
            self._update_status = self._default_update_status("failed", str(exc))
            self._emit_update_status()
            self._notification("error", f"Update check failed: {exc}")
            return dict(self._update_status)

    @Slot(bool, result="QVariantMap")
    def setAutoUpdateChecks(self, enabled):
        self.general_config["auto_update_checks"] = "yes" if enabled else "no"
        save_dict_as_toml(self.general_config, "cfg/general_config.toml")
        self._update_status["autoUpdateEnabled"] = bool(enabled)
        self._emit_state()
        self._emit_update_status()
        self._notification(
            "success",
            "Automatic update checks enabled." if enabled else "Automatic update checks disabled.",
        )
        return {"ok": True, "autoUpdateEnabled": bool(enabled)}

    @Slot(str, result="QVariantMap")
    def ignoreUpdateVersion(self, sha):
        ignored_sha = str(sha or self._update_status.get("latestSha", "") or "").strip()
        if not ignored_sha:
            return {"ok": False, "error": "missing update version"}
        self.general_config["auto_update_ignored_sha"] = ignored_sha
        save_dict_as_toml(self.general_config, "cfg/general_config.toml")
        self._update_status["ignored"] = True
        self._emit_state()
        self._emit_update_status()
        self._notification("info", "This update version will not prompt again.")
        return {"ok": True, "ignoredSha": ignored_sha}

    def _prepare_bot_control_events(self):
        self._bot_stop_requested = False
        self._bot_paused = False
        self._bot_control_state = "running"
        self._bot_stop_event = threading.Event()
        self._bot_pause_event = threading.Event()
        self._live_data = {}
        self._event_log = []
        self._live_data.update({
            "bot_control_state": self._bot_control_state,
            "bot_paused": False,
        })
        self.liveDataChanged.emit(self._live_data.copy())
        self._emit_logs()

    def _is_bot_thread_alive(self):
        return bool(self._bot_thread and self._bot_thread.is_alive())

    def _set_bot_control_state(self, control_state, paused=None):
        self._bot_control_state = str(control_state or "stopped")
        if paused is not None:
            self._bot_paused = bool(paused)
        with self._live_lock:
            self._live_data.update({
                "bot_control_state": self._bot_control_state,
                "bot_paused": self._bot_paused,
            })
            payload = dict(self._live_data)
        self.liveDataChanged.emit(payload)
        return payload

    def _release_active_bot_inputs(self, reason):
        import sys

        managers = []
        for module in (sys.modules.get("__main__"), inspect.getmodule(self._pyla_main)):
            if module is None:
                continue
            manager = getattr(module, "_active_stage_manager", None)
            if manager is not None and manager not in managers:
                managers.append(manager)
        released = False
        for manager in managers:
            controller = getattr(manager, "window_controller", None)
            if controller is None:
                continue
            try:
                if hasattr(controller, "release_all_inputs"):
                    controller.release_all_inputs(reason)
                else:
                    controller.keys_up(list("wasd"))
                released = True
            except Exception as exc:
                self._push_log("warning", f"[BOT][WARN] input release failed: {exc}")
        if not released:
            self._push_log("info", "[INPUT] releasing active touches skipped; no active controller yet")

    def _set_runtime_binding(self, name, value):
        import sys

        target_modules = []
        main_module = sys.modules.get("__main__")
        if main_module is not None:
            target_modules.append(main_module)

        pyla_module = inspect.getmodule(self._pyla_main)
        if pyla_module is not None and pyla_module not in target_modules:
            target_modules.append(pyla_module)

        for module in target_modules:
            try:
                setattr(module, name, value)
            except Exception:
                pass

    def _run_bot(self):
        try:
            if self._bot_stop_requested:
                return
            self._set_runtime_binding("_active_dashboard", self)
            try:
                sig = inspect.signature(self._pyla_main)
                if "external_stop_event" in sig.parameters:
                    self._pyla_main(
                        self.brawlers_data,
                        external_stop_event=self._bot_stop_event,
                        external_pause_event=self._bot_pause_event,
                    )
                else:
                    self._pyla_main(self.brawlers_data)
            except (TypeError, ValueError):
                self._pyla_main(self.brawlers_data)
        except Exception as exc:
            traceback.print_exc()
            self._notification("error", f"Bot thread error: {exc}")
        finally:
            self._set_runtime_binding("_active_dashboard", None)
            self._set_runtime_binding("_active_stage_manager", None)
            self._bot_paused = False
            self._bot_stop_requested = False
            self._set_bot_control_state("stopped", paused=False)
            self._push_log("info", "[BOT] stopped")

    def after(self, ms, callback):
        QTimer.singleShot(int(ms), callback)

    def update_live(self, **kw):
        previous_state = str(self._live_data.get("state", "") or "").lower()
        previous_brawler = str(self._live_data.get("brawler", "") or "")
        with self._live_lock:
            self._live_data.update(kw)
            payload = dict(self._live_data)
        current_state = str(payload.get("state", "") or "").lower()
        current_brawler = str(payload.get("brawler", "") or "")
        if current_state and current_state != previous_state:
            self._push_log("info", f"State -> {current_state}")
        if current_brawler and current_brawler != previous_brawler:
            self._push_log("info", f"Brawler -> {current_brawler}")
        self.liveDataChanged.emit(payload)

    def _show_session_summary(self):
        summary = getattr(self, "_session_summary", None)
        if summary:
            self.sessionSummaryReady.emit(summary)
        self._live_data.clear()
        self.liveDataChanged.emit({})
        self._emit_history()

    @Slot(result="QVariantMap")
    def initialState(self):
        return {
            "versionTag": self._format_version_tag(self._version_str),
            "version": self._version_str,
            "branchLabel": self._format_version_tag(self._version_str).replace("PylaAI ", ""),
            "loggedIn": self._logged_in,
            "capabilities": dict(self.capabilities),
            "general": dict(self.general_config),
            "bot": dict(self.bot_config),
            "time": dict(self.time_config),
            "login": {"key": str(self.login_config.get("key", ""))},
            "roster": self.getRoster(),
            "brawlers": self._build_brawler_payload(),
            "history": self.getHistory(),
            "farmPreview": self.getFarmPreview(),
            "farmRoster": self.getFarmRoster(),
            "farmStatus": self.getFarmStatus(),
            "logs": self.getLogs(),
            "gamemodes": list(GAMEMODES),
            "emulators": list(EMULATORS),
            "toolStatus": self._tool_status_payload(),
            "updateStatus": dict(self._update_status),
            "performanceProfiles": self._performance_profiles_payload(),
            "brawlerLoadState": dict(self._brawler_load_state),
            "live": dict(self._live_data),
            "multiInstance": self._multi_instance_state(),
        }

    @Slot(result="QVariantList")
    def getRoster(self):
        roster = []
        for entry in self.brawlers_data:
            row = dict(entry)
            row["displayName"] = row["brawler"].title()
            row["icon"] = self._icon_url_for(row["brawler"])
            roster.append(row)
        return roster

    @Slot(result="QVariantList")
    def getHistory(self):
        return self._history_rows()

    @Slot(result="QVariantList")
    def getLogs(self):
        return list(self._event_log)

    @Slot(result="QVariantList")
    def getBrawlers(self):
        return self._build_brawler_payload()

    @Slot(result="QVariantMap")
    def refreshBrawlers(self):
        self._brawler_load_state = {"state": "loading", "message": "Loading brawlers...", "count": 0}
        self._emit_state()
        self._brawler_load_error = ""
        try:
            self.brawlers_info = load_brawlers_info()
        except Exception as exc:
            self.brawlers_info = {}
            self._brawler_load_error = str(exc)
        self._all_brawlers = self._discover_brawler_names(self._input_brawlers)
        self._refresh_brawler_load_state()
        self._emit_state()
        if self._brawler_load_state["state"] == "ready":
            self._notification("success", self._brawler_load_state["message"])
        else:
            self._notification("warning" if self._brawler_load_state["state"] == "empty" else "error", self._brawler_load_state["message"])
        return dict(self._brawler_load_state)

    @Slot(result="QVariantList")
    def getFarmPreview(self):
        return self._build_trophy_farm_preview()

    @Slot(result="QVariantList")
    def getFarmRoster(self):
        plan = self._farm_plan()
        rows = []
        for row in sorted(plan["rows"], key=lambda item: (not item.get("qualifies", False), item.get("trophies", 0), item.get("brawler", ""))):
            brawler = row.get("brawler", "")
            rows.append({
                "brawler": brawler,
                "displayName": str(row.get("displayName") or brawler).title(),
                "icon": self._icon_url_for(brawler),
                "trophies": self._as_int(row.get("trophies", 0), 0),
                "highestTrophies": self._as_int(row.get("highestTrophies", row.get("trophies", 0)), 0),
                "power": self._as_int(row.get("power", 0), 0),
                "rank": self._as_int(row.get("rank", 0), 0),
                "owned": bool(row.get("owned", False)),
                "included": bool(row.get("included", True)),
                "excluded": bool(row.get("excluded", False)),
                "qualifies": bool(row.get("qualifies", False)),
                "source": str(row.get("source", plan["mode"])),
                "reason": str(row.get("reason", "")),
                "target": plan["target"],
                "order": self._as_int(row.get("order", 0), 0),
                "winrate": self._as_int(row.get("winrate", 50), 50),
            })
        return rows

    @Slot(result="QVariantMap")
    def getFarmStatus(self):
        return self._farm_status_payload()

    @Slot("QVariantMap", result="QVariantList")
    def previewFarmSettings(self, payload):
        return self._build_trophy_farm_preview(payload or {})

    @Slot("QVariantMap", result="QVariantMap")
    def updateManualFarmBrawler(self, payload):
        brawler = self._canonical_brawler_name(payload.get("brawler", ""))
        if not brawler:
            return {"ok": False, "error": "Choose a brawler first."}
        manual_roster = self._farm_state.setdefault("manual_roster", {})
        if not isinstance(manual_roster, dict):
            manual_roster = {}
            self._farm_state["manual_roster"] = manual_roster
        row = dict(manual_roster.get(brawler, {}))
        row["owned"] = bool(payload.get("owned", row.get("owned", True)))
        row["included"] = bool(payload.get("included", row.get("included", True)))
        row["trophies"] = self._as_int(payload.get("trophies", row.get("trophies", 0)), 0)
        row["priority"] = self._as_int(payload.get("priority", row.get("priority", 0)), 0)
        row["source"] = "manual"
        manual_roster[brawler] = row
        save_farm_state(self._farm_state)
        self._emit_state()
        return {"ok": True, "brawler": brawler}

    @Slot(str, str, result="QVariantMap")
    def fetchFarmRosterFromApi(self, apiKey, playerTag):
        api_key = str(apiKey or self.general_config.get("brawlstars_api_key", "") or "").strip()
        player_tag = str(playerTag or self.general_config.get("brawlstars_player_tag", "") or "").strip()
        try:
            normalized_tag = f"#{normalize_player_tag(player_tag)}"
            profile = fetch_player_profile(api_key, normalized_tag, timeout=15)
        except Exception as exc:
            self._notification("error", f"Could not fetch Brawl Stars roster: {exc}")
            return {"ok": False, "error": str(exc)}

        api_roster = {}
        for api_name, row in profile.get("brawlers", {}).items():
            internal = self._resolve_internal_brawler_name(api_name) or normalize_brawler_name(api_name)
            item = dict(row)
            item["brawler"] = internal
            item["displayName"] = str(row.get("name") or internal).title()
            item["owned"] = True
            item["included"] = True
            item["source"] = "api"
            api_roster[internal] = item

        self.general_config["brawlstars_api_key"] = api_key
        self.general_config["brawlstars_player_tag"] = profile.get("player_tag") or normalized_tag
        self.bot_config["trophy_farm_mode"] = "api"
        self._farm_state["api_roster"] = api_roster
        self._farm_state["api_player_name"] = profile.get("player_name", "")
        self._farm_state["api_player_tag"] = profile.get("player_tag", normalized_tag)
        self._farm_state["api_last_refresh"] = now_iso()
        save_dict_as_toml(self.general_config, "cfg/general_config.toml")
        save_dict_as_toml(self.bot_config, "cfg/bot_config.toml")
        save_farm_state(self._farm_state)
        self._emit_state()
        self._notification("success", f"Fetched {len(api_roster)} unlocked brawler(s) from the Brawl Stars API.")
        return {"ok": True, "count": len(api_roster), "playerTag": self._farm_state["api_player_tag"]}

    @Slot(result="QVariantMap")
    def clearFarmApiCredentials(self):
        self.general_config["brawlstars_api_key"] = ""
        self.general_config["brawlstars_player_tag"] = ""
        self._farm_state["api_roster"] = {}
        self._farm_state["api_player_name"] = ""
        self._farm_state["api_player_tag"] = ""
        self._farm_state["api_last_refresh"] = ""
        save_dict_as_toml(self.general_config, "cfg/general_config.toml")
        save_farm_state(self._farm_state)
        self._emit_state()
        self._notification("info", "Cleared saved Brawl Stars API credentials and API roster.")
        return {"ok": True}

    @Slot()
    def importAllBrawlersFromBrawlStarsApi(self):
        try:
            api_brawlers = self._fetch_player_brawlers_from_api()
        except Exception as exc:
            self._notification("error", f"Could not import from the Brawl Stars API: {exc}")
            return

        trophy_map = {}
        for entry in api_brawlers:
            internal_name = self._resolve_internal_brawler_name(entry.get("name", ""))
            if internal_name:
                trophy_map[internal_name] = self._as_int(entry.get("trophies", 0), 0)

        if not trophy_map:
            self._notification("warning", "No compatible brawler data was returned by the Brawl Stars API.")
            return

        scan_data = self._brawler_scan_data()
        for brawler_name, trophies in trophy_map.items():
            existing = scan_data.get(brawler_name, {})
            if not isinstance(existing, dict):
                existing = {}
            existing["trophies"] = trophies
            scan_data[brawler_name] = existing
        self._save_brawler_scan_data(scan_data)

        queued_updated_count = 0
        for row in self.brawlers_data:
            brawler = row.get("brawler")
            if brawler in trophy_map:
                row["trophies"] = trophy_map[brawler]
                queued_updated_count += 1

        save_brawler_data(self.brawlers_data)
        self._emit_roster()
        self._emit_state()
        self._notification("success", f"Imported trophies for {len(trophy_map)} brawler(s) from the Brawl Stars API and synced {queued_updated_count} queued entry(ies).")

    @Slot("QVariantMap")
    def saveControlSettings(self, payload):
        self.general_config["map_orientation"] = str(payload.get("map_orientation", self.general_config.get("map_orientation", "vertical"))).lower()
        self.general_config["current_emulator"] = str(payload.get("current_emulator", self.general_config.get("current_emulator", "LDPlayer")))
        run_for_minutes = payload.get("run_for_minutes", self.general_config.get("run_for_minutes", 600))
        self.general_config["run_for_minutes"] = self._as_int(run_for_minutes, 600)

        gamemode = str(payload.get("gamemode", self.bot_config.get("gamemode", "knockout"))).lower()
        self.bot_config["gamemode"] = gamemode
        matching = next((mode for mode in GAMEMODES if mode["value"] == gamemode), None)
        if matching:
            self.bot_config["gamemode_type"] = self._gamemode_type_for(gamemode)

        save_dict_as_toml(self.general_config, "cfg/general_config.toml")
        save_dict_as_toml(self.bot_config, "cfg/bot_config.toml")
        self._emit_state()
        self._notification("success", "Control Center settings saved.")

    @Slot("QVariantMap")
    def saveFarmSettings(self, payload):
        farm_enabled = self._is_enabled(payload.get("smart_trophy_farm", self.bot_config.get("smart_trophy_farm", "no")))
        self.bot_config["smart_trophy_farm"] = "yes" if farm_enabled else "no"
        self.bot_config["trophy_farm_mode"] = normalize_farm_mode(
            payload.get("trophy_farm_mode", self.bot_config.get("trophy_farm_mode", "manual"))
        )
        self.bot_config["trophy_farm_strategy"] = self._normalize_farm_strategy(
            payload.get("trophy_farm_strategy", self.bot_config.get("trophy_farm_strategy", "lowest_first"))
        )
        self.bot_config["trophy_farm_target"] = self._as_int(payload.get("trophy_farm_target", self.bot_config.get("trophy_farm_target", 500)), 500)
        excluded = payload.get("trophy_farm_excluded", self.bot_config.get("trophy_farm_excluded", []))
        if isinstance(excluded, list):
            self.bot_config["trophy_farm_excluded"] = sorted(self._normalize_brawler_set(excluded))

        if self.capabilities.get("quest_farm"):
            self.bot_config["quest_farm_enabled"] = "yes" if self._is_enabled(payload.get("quest_farm_enabled", self.bot_config.get("quest_farm_enabled", "no"))) else "no"
            self.bot_config["quest_farm_mode"] = str(payload.get("quest_farm_mode", self.bot_config.get("quest_farm_mode", "games")))
            quest_excluded = payload.get("quest_farm_excluded", self.bot_config.get("quest_farm_excluded", []))
            if isinstance(quest_excluded, list):
                self.bot_config["quest_farm_excluded"] = sorted(self._normalize_brawler_set(quest_excluded))

        save_dict_as_toml(self.bot_config, "cfg/bot_config.toml")
        self._emit_state()
        if farm_enabled:
            self._notification("success", f"Farm settings saved in {self.bot_config['trophy_farm_mode'].upper()} mode. Trophy Farm will build its queue the next time you start the bot.")
        else:
            self._notification("success", "Farm settings saved. Start Bot will keep using your normal roster until Trophy Farm is enabled.")

    @Slot("QVariantMap")
    def saveSettings(self, payload):
        general = payload.get("general", {})
        bot = payload.get("bot", {})
        time_cfg = payload.get("time", {})
        login = payload.get("login", {})

        for key in (
            "max_ips",
            "target_ips",
            "cpu_or_gpu",
            "super_debug",
            "input_debug",
            "personal_webhook",
            "discord_id",
            "brawlstars_api_key",
            "brawlstars_player_tag",
            "api_base_url",
            "brawlstars_package",
            "emulator_port",
            "run_for_minutes",
            "auto_push_target_trophies",
            "current_emulator",
            "map_orientation",
            "instance_count",
            "performance_profile",
            "auto_update_checks",
            "auto_update_ignored_sha",
            "scrcpy_max_fps",
            "scrcpy_max_width",
            "scrcpy_bitrate",
            "joystick_refresh_seconds",
            "joystick_repress_seconds",
            "joystick_down_move_delay",
        ):
            if key in general:
                self.general_config[key] = general[key]

        self.general_config["emulator_port"] = self._coerce_int(self.general_config.get("emulator_port", 5037), 5037, minimum=1)
        self.general_config["run_for_minutes"] = self._coerce_int(self.general_config.get("run_for_minutes", 600), 600, minimum=1)
        self.general_config["auto_push_target_trophies"] = self._coerce_int(self.general_config.get("auto_push_target_trophies", 1000), 1000, minimum=0)
        previous_count = self._coerce_int(load_config("general").get("instance_count", 1), 1, minimum=1)
        self.general_config["instance_count"] = self._coerce_int(self.general_config.get("instance_count", previous_count), previous_count, minimum=1)
        self.general_config["target_ips"] = self._as_int(self.general_config.get("target_ips", self.general_config.get("max_ips", 60)), 60)
        self.general_config["scrcpy_max_fps"] = self._as_int(self.general_config.get("scrcpy_max_fps", self.general_config["target_ips"]), self.general_config["target_ips"])
        self.general_config["scrcpy_max_width"] = self._as_int(self.general_config.get("scrcpy_max_width", 960), 960)
        self.general_config["scrcpy_bitrate"] = self._as_int(self.general_config.get("scrcpy_bitrate", 3000000), 3000000)
        self.general_config["joystick_refresh_seconds"] = self._as_float(self.general_config.get("joystick_refresh_seconds", 0.35), 0.35)
        self.general_config["joystick_repress_seconds"] = self._as_float(self.general_config.get("joystick_repress_seconds", 1.8), 1.8)
        self.general_config["joystick_down_move_delay"] = self._as_float(self.general_config.get("joystick_down_move_delay", 0.012), 0.012)

        for key in (
            "minimum_movement_delay",
            "unstuck_movement_delay",
            "unstuck_movement_hold_time",
            "wall_detection_confidence",
            "entity_detection_confidence",
            "seconds_to_hold_attack_after_reaching_max",
            "play_again_on_win",
            "bot_uses_gadgets",
        ):
            if key in bot:
                self.bot_config[key] = bot[key]

        for key, default in (
            ("minimum_movement_delay", 0.08),
            ("unstuck_movement_delay", 1.5),
            ("unstuck_movement_hold_time", 0.8),
            ("wall_detection_confidence", 0.9),
            ("entity_detection_confidence", 0.6),
            ("seconds_to_hold_attack_after_reaching_max", 1.5),
        ):
            self.bot_config[key] = self._as_float(self.bot_config.get(key, default), default)

        for key in (
            "state_check",
            "no_detections",
            "idle",
            "gadget",
            "hypercharge",
            "super",
            "wall_detection",
            "no_detection_proceed",
            "check_if_brawl_stars_crashed",
        ):
            if key in time_cfg:
                self.time_config[key] = time_cfg[key]

        for key, default in (
            ("state_check", 5),
            ("no_detections", 10),
            ("idle", 5),
            ("gadget", 0.5),
            ("hypercharge", 1.0),
            ("super", 0.1),
            ("wall_detection", 0.2),
            ("no_detection_proceed", 6.5),
            ("check_if_brawl_stars_crashed", 10),
        ):
            caster = self._as_int if key in {"state_check", "no_detections", "idle", "check_if_brawl_stars_crashed"} else self._as_float
            self.time_config[key] = caster(self.time_config.get(key, default), default)

        if "key" in login:
            self.login_config["key"] = str(login.get("key", ""))
            save_dict_as_toml(self.login_config, "cfg/login.toml")

        save_dict_as_toml(self.general_config, "cfg/general_config.toml")
        save_dict_as_toml(self.bot_config, "cfg/bot_config.toml")
        save_dict_as_toml(self.time_config, "cfg/time_tresholds.toml")
        self._validate_existing_login()
        self._emit_state()
        self._notification("success", "Settings saved.")

    @Slot("QVariantMap")
    def configureMultiInstance(self, payload):
        try:
            existing_count = self._coerce_int(self.general_config.get("instance_count", 1), 1, minimum=1)
            count = self._coerce_int(payload.get("instance_count", existing_count), existing_count, minimum=1)
            scrcpy_max_fps = self._normalize_scrcpy_max_fps(payload.get("scrcpy_max_fps", self.general_config.get("scrcpy_max_fps", "auto")))
            ports = payload.get("ports", "")
            current_id = current_instance_id(self._coerce_int(self.general_config.get("instance_index", 1), 1, minimum=1))
            current_port = self._coerce_int(payload.get("current_port", self.general_config.get("emulator_port", 5037)), self._coerce_int(self.general_config.get("emulator_port", 5037), 5037, minimum=1), minimum=1)
            configured = configure_instances(
                count,
                ports=ports,
                scrcpy_max_fps=scrcpy_max_fps,
                current_instance=current_id,
                current_port=current_port,
            )

            self.general_config["instance_count"] = count
            self.general_config["scrcpy_max_fps"] = scrcpy_max_fps
            current_row = next((row for row in configured if int(row.get("instance", 0) or 0) == current_id), None)
            if current_row and current_row.get("port") is not None:
                self.general_config["emulator_port"] = self._coerce_int(current_row.get("port"), current_port, minimum=1)
            else:
                self.general_config["emulator_port"] = current_port

            save_dict_as_toml(self.general_config, "cfg/general_config.toml")
            self.general_config = self._load_general_config()
            self._emit_state()
            self._notification("success", f"Configured {count} instance(s) and refreshed launcher files.")
        except Exception as exc:
            self._notification("error", f"Could not configure multi-instance mode: {exc}")

    @Slot(result="QVariantMap")
    def getMultiInstanceState(self):
        return self._multi_instance_state()

    @Slot("QVariantMap")
    def addOrUpdateRosterEntry(self, payload):
        brawler = str(payload.get("brawler", "")).strip().lower()
        if not brawler:
            self._notification("warning", "Choose a brawler first.")
            return

        entry = {
            "brawler": brawler,
            "push_until": self._as_int(payload.get("push_until", self.general_config.get("auto_push_target_trophies", 1000)), 0),
            "trophies": self._as_int(payload.get("trophies", 0), 0),
            "wins": self._as_int(payload.get("wins", 0), 0),
            "type": str(payload.get("type", "trophies") or "trophies"),
            "automatically_pick": bool(payload.get("automatically_pick", True)),
            "win_streak": self._as_int(payload.get("win_streak", 0), 0),
            "manual_trophies": bool(payload.get("manual_trophies", False)),
        }

        self.brawlers_data = [row for row in self.brawlers_data if row.get("brawler") != brawler]
        self.brawlers_data.append(entry)
        save_brawler_data(self.brawlers_data)
        self._emit_roster()
        self._emit_state()
        self._notification("success", f"{brawler.title()} added to roster.")

    @Slot(str)
    def removeRosterEntry(self, brawler):
        target = str(brawler).strip().lower()
        before = len(self.brawlers_data)
        self.brawlers_data = [row for row in self.brawlers_data if row.get("brawler") != target]
        if len(self.brawlers_data) != before:
            save_brawler_data(self.brawlers_data)
            self._emit_roster()
            self._emit_state()
            self._notification("info", f"{target.title()} removed from roster.")

    @Slot()
    def clearRoster(self):
        self.brawlers_data = []
        save_brawler_data(self.brawlers_data)
        self._emit_roster()
        self._emit_state()
        self._notification("info", "Roster cleared.")

    @Slot()
    def loadRosterFile(self):
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Load Brawler Config",
            "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            self.brawlers_data = self._normalize_roster(data)
            save_brawler_data(self.brawlers_data)
            self._emit_roster()
            self._emit_state()
            self._notification("success", f"Loaded roster from {os.path.basename(path)}.")
        except Exception as exc:
            self._notification("error", f"Could not load roster: {exc}")

    @Slot()
    def exportRosterFile(self):
        path, _ = QFileDialog.getSaveFileName(
            None,
            "Export Brawler Config",
            "pyla-roster.json",
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.brawlers_data, handle, indent=4)
            self._notification("success", f"Exported roster to {os.path.basename(path)}.")
        except Exception as exc:
            self._notification("error", f"Could not export roster: {exc}")

    @Slot()
    def startBot(self):
        if self._bot_thread and self._bot_thread.is_alive():
            self._notification("warning", "The bot is already running.")
            return

        runtime_roster = list(self.brawlers_data)
        if self._is_enabled(self.bot_config.get("smart_trophy_farm", "no")):
            runtime_roster, queue, target, _strategy = self._build_trophy_farm_roster()
            if not runtime_roster:
                reason = self._farm_status_payload().get("emptyReason") or f"No eligible brawlers are below {target} trophies."
                self._notification("warning", f"Trophy Farm is enabled, but the queue is empty: {reason}")
                return
            self.brawlers_data = self._normalize_roster(runtime_roster)
            self._emit_roster()
            self._emit_state()
            self._notification("info", f"Trophy Farm queue ready: {len(queue)} brawler(s) below {target} trophies.")
        elif not runtime_roster:
            self._notification("warning", "Select at least one brawler first.")
            return

        api_base_url = str(self.general_config.get("api_base_url", "localhost")).strip()
        if api_base_url != "localhost":
            auth_key = str(self.login_config.get("key", "")).strip()
            if not auth_key:
                self._notification("warning", "Add your Pyla API key in Settings before starting.")
                return
            try:
                self._logged_in = bool(check_if_exists(auth_key))
            except Exception as exc:
                self._notification("error", f"Could not validate API key: {exc}")
                return
            if not self._logged_in:
                self._notification("warning", "The current API key was not accepted.")
                return

        save_dict_as_toml(self.general_config, "cfg/general_config.toml")
        save_dict_as_toml(self.bot_config, "cfg/bot_config.toml")
        save_dict_as_toml(self.time_config, "cfg/time_tresholds.toml")
        save_dict_as_toml(self.login_config, "cfg/login.toml")
        save_brawler_data(self.brawlers_data)
        self._prepare_bot_control_events()

        self._bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self._bot_thread.start()
        self._set_bot_control_state("running", paused=False)
        self._notification("success", "Bot started.")
        self._emit_state()

    @Slot()
    def stopBot(self):
        if not self._is_bot_thread_alive():
            self._bot_stop_requested = False
            if self._bot_pause_event:
                self._bot_pause_event.clear()
            self._set_bot_control_state("stopped", paused=False)
            self._notification("warning", "[BOT][WARN] stop requested but bot was not running")
            return
        self._push_log("info", "[BOT] stop requested")
        self._bot_stop_requested = True
        if self._bot_stop_event:
            self._bot_stop_event.set()
        if self._bot_pause_event:
            self._bot_pause_event.clear()
        self._bot_paused = False
        self._set_bot_control_state("stopping", paused=False)
        self._release_active_bot_inputs("stop requested")
        self._notification("info", "Stop signal sent to the bot.")

    @Slot()
    def pauseBot(self):
        if not self._is_bot_thread_alive():
            self._notification("warning", "[BOT][WARN] pause requested but bot is not running")
            return
        if self._bot_paused:
            self._notification("warning", "[BOT][WARN] pause requested but bot is already paused")
            return
        if self._bot_pause_event:
            self._bot_pause_event.set()
        self._bot_paused = True
        self._set_bot_control_state("paused", paused=True)
        self._push_log("info", "[BOT] pause requested")
        self._release_active_bot_inputs("pause requested")
        self._notification("info", "Bot paused.")

    @Slot()
    def resumeBot(self):
        if not self._is_bot_thread_alive():
            self._notification("warning", "[BOT][WARN] resume requested but bot is not running")
            return
        if not self._bot_paused:
            self._notification("warning", "[BOT][WARN] resume requested but bot is not paused")
            return
        if self._bot_pause_event:
            self._bot_pause_event.clear()
        self._bot_paused = False
        self._set_bot_control_state("running", paused=False)
        self._notification("info", "[BOT] resumed")

    @Slot()
    def on_app_about_to_quit(self):
        if self._bot_stop_event:
            self._bot_stop_event.set()
        self._release_active_bot_inputs("app closing")
