import inspect
import json
import os
import re
import threading
import traceback
from datetime import datetime

import requests
from PySide6.QtCore import QObject, QTimer, Signal, Slot
from PySide6.QtWidgets import QFileDialog

from gui.api import check_if_exists
from gui.config_store import load_config
from lobby_automation import LobbyAutomation
from stage_manager import StageManager
from utils import load_brawlers_info, load_toml_as_dict, save_brawler_data, save_dict_as_toml


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

EMULATORS = ["LDPlayer", "BlueStacks", "MEmu", "Others"]


class QtBridge(QObject):
    stateChanged = Signal("QVariantMap")
    rosterChanged = Signal("QVariantList")
    liveDataChanged = Signal("QVariantMap")
    historyChanged = Signal("QVariantList")
    logsChanged = Signal("QVariantList")
    notificationRaised = Signal(str, str)
    sessionSummaryReady = Signal("QVariantMap")

    def __init__(self, version_str, brawlers, pyla_main_fn, login_fn=None, saved_brawler_data=None):
        super().__init__()
        self._version_str = str(version_str).strip()
        self._pyla_main = pyla_main_fn
        self._login_fn = login_fn
        self._all_brawlers = list(brawlers or [])
        self._bot_thread = None
        self._bot_stop_event = None
        self._bot_pause_event = None
        self._bot_stop_requested = False
        self._live_data = {}
        self._session_summary = None
        self._event_log = []
        self._logged_in = False
        self._live_lock = threading.Lock()

        self.bot_config = self._load_bot_config()
        self.general_config = self._load_general_config()
        self.time_config = load_config("time")
        self.login_config = load_toml_as_dict("cfg/login.toml")
        self.brawlers_info = load_brawlers_info()
        self.brawlers_data = self._normalize_roster(saved_brawler_data or self._load_saved_roster())
        self.capabilities = {
            "visual_overlay": os.path.exists("visual_overlay.py"),
            "advanced_live": all(os.path.exists(path) for path in ("behavior_tree.py", "bt_combat.py")),
            "brawler_scan": hasattr(LobbyAutomation, "scan_all_brawlers"),
            "quest_farm": hasattr(StageManager, "_handle_quest_rotation"),
            "quest_scan": hasattr(LobbyAutomation, "scan_quest_brawlers"),
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
        return config

    @staticmethod
    def _is_enabled(value):
        return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_farm_strategy(value):
        normalized = str(value or "").strip().lower()
        aliases = {
            "lowest_first": "lowest_first",
            "highest_first": "highest_winrate",
            "highest_winrate": "highest_winrate",
            "in_order": "sequential",
            "sequential": "sequential",
        }
        return aliases.get(normalized, "lowest_first")

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
        if not os.path.exists("latest_brawler_data.json"):
            return []
        try:
            with open("latest_brawler_data.json", "r", encoding="utf-8") as handle:
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
        scan_path = os.path.join("cfg", "brawler_scan.json")
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
        scan_path = os.path.join("cfg", "brawler_scan.json")
        payload = {"brawlers": data if isinstance(data, dict) else {}}
        with open(scan_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)

    @staticmethod
    def _slug_name(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

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
        player_tag = str(self.general_config.get("brawlstars_player_tag", "")).strip().upper().replace("#", "")
        if not api_key:
            raise ValueError("Add a Brawl Stars API key in Settings first.")
        if not player_tag:
            raise ValueError("Add a player tag in Settings first.")

        response = requests.get(
            f"https://api.brawlstars.com/v1/players/%23{player_tag}",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        if response.status_code == 403:
            raise ValueError("The Brawl Stars API key was rejected.")
        if response.status_code == 404:
            raise ValueError("The player tag was not found.")
        response.raise_for_status()
        payload = response.json()
        brawlers = payload.get("brawlers", [])
        if not isinstance(brawlers, list):
            raise ValueError("The API response did not contain a valid brawler list.")
        return brawlers

    def _build_brawler_payload(self):
        scan_data = self._brawler_scan_data()
        roster_lookup = {entry["brawler"]: entry for entry in self.brawlers_data}
        items = []
        for name in self._all_brawlers:
            scan_entry = scan_data.get(name, {})
            selected = roster_lookup.get(name, {})
            trophies = selected.get("trophies", scan_entry.get("trophies", 0))
            items.append({
                "name": name,
                "displayName": name.title(),
                "icon": self._icon_url_for(name),
                "selected": bool(selected),
                "trophies": int(trophies or 0),
                "pushUntil": int(selected.get("push_until", self.general_config.get("auto_push_target_trophies", 1000)) or 0),
                "wins": int(selected.get("wins", 0) or 0),
                "winStreak": int(selected.get("win_streak", 0) or 0),
                "type": str(selected.get("type", "trophies") or "trophies"),
                "autoPick": bool(selected.get("automatically_pick", True)),
                "manualTrophies": bool(selected.get("manual_trophies", False)),
                "holdAttack": float(self.brawlers_info.get(name, {}).get("hold_attack", 0) or 0),
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
        target = max(0, self._as_int(source.get("trophy_farm_target", 500), 500))
        strategy = self._normalize_farm_strategy(source.get("trophy_farm_strategy", "lowest_first"))
        excluded = {
            str(item or "").strip().lower()
            for item in source.get("trophy_farm_excluded", [])
            if str(item or "").strip()
        }
        return {
            "target": target,
            "strategy": strategy,
            "excluded": excluded,
        }

    def _build_trophy_farm_roster(self, overrides=None):
        farm_settings = self._farm_settings(overrides)
        target = farm_settings["target"]
        strategy = farm_settings["strategy"]
        excluded = farm_settings["excluded"]
        scan_data = self._brawler_scan_data()
        roster_lookup = {
            str(entry.get("brawler", "")).strip().lower(): entry
            for entry in self.brawlers_data
            if isinstance(entry, dict) and str(entry.get("brawler", "")).strip()
        }
        history = self._match_history_map()
        queue = []

        for brawler in self._all_brawlers:
            if brawler in excluded:
                continue

            scan_entry = scan_data.get(brawler, {})
            if not isinstance(scan_entry, dict):
                scan_entry = {}
            if scan_entry and scan_entry.get("unlocked") is False:
                continue

            selected_entry = roster_lookup.get(brawler, {})
            trophies = self._as_int(selected_entry.get("trophies", scan_entry.get("trophies", 0)), 0)
            if trophies >= target:
                continue

            history_entry = history.get(brawler, {})
            if not isinstance(history_entry, dict):
                history_entry = {}
            wins = self._as_int(history_entry.get("victory", 0), 0)
            defeats = self._as_int(history_entry.get("defeat", 0), 0)
            total_games = wins + defeats
            winrate = round((wins / total_games) * 100) if total_games else 50
            queue.append({
                "brawler": brawler,
                "trophies": trophies,
                "winrate": winrate,
                "total_games": total_games,
                "source": selected_entry,
            })

        if strategy == "highest_winrate":
            queue.sort(key=lambda item: (-item["winrate"], item["trophies"], item["brawler"]))
        elif strategy == "sequential":
            queue.sort(key=lambda item: item["brawler"])
        else:
            queue.sort(key=lambda item: (item["trophies"], item["brawler"]))

        roster = []
        for item in queue:
            source = item.get("source", {})
            roster.append({
                "brawler": item["brawler"],
                "push_until": target,
                "trophies": item["trophies"],
                "wins": self._as_int(source.get("wins", 0), 0),
                "type": "trophies",
                "automatically_pick": bool(source.get("automatically_pick", True)),
                "win_streak": self._as_int(source.get("win_streak", 0), 0),
                "manual_trophies": bool(source.get("manual_trophies", False)),
            })

        return roster, queue, target, strategy

    def _build_trophy_farm_preview(self, overrides=None):
        _roster, queue, target, strategy = self._build_trophy_farm_roster(overrides)
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
            })
        return preview

    def _notification(self, level, message):
        self._push_log(level, message)
        self.notificationRaised.emit(level, message)

    def _prepare_bot_control_events(self):
        self._bot_stop_requested = False
        self._bot_stop_event = threading.Event()
        self._bot_pause_event = threading.Event()
        self._live_data = {}
        self._event_log = []
        self.liveDataChanged.emit(self._live_data.copy())
        self._emit_logs()

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
            "logs": self.getLogs(),
            "gamemodes": list(GAMEMODES),
            "emulators": list(EMULATORS),
            "live": dict(self._live_data),
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

    @Slot(result="QVariantList")
    def getFarmPreview(self):
        return self._build_trophy_farm_preview()

    @Slot("QVariantMap", result="QVariantList")
    def previewFarmSettings(self, payload):
        return self._build_trophy_farm_preview(payload or {})

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
        self.bot_config["trophy_farm_strategy"] = self._normalize_farm_strategy(
            payload.get("trophy_farm_strategy", self.bot_config.get("trophy_farm_strategy", "lowest_first"))
        )
        self.bot_config["trophy_farm_target"] = self._as_int(payload.get("trophy_farm_target", self.bot_config.get("trophy_farm_target", 500)), 500)
        excluded = payload.get("trophy_farm_excluded", self.bot_config.get("trophy_farm_excluded", []))
        if isinstance(excluded, list):
            self.bot_config["trophy_farm_excluded"] = sorted({str(item).lower() for item in excluded if item})

        if self.capabilities.get("quest_farm"):
            self.bot_config["quest_farm_enabled"] = "yes" if self._is_enabled(payload.get("quest_farm_enabled", self.bot_config.get("quest_farm_enabled", "no"))) else "no"
            self.bot_config["quest_farm_mode"] = str(payload.get("quest_farm_mode", self.bot_config.get("quest_farm_mode", "games")))
            quest_excluded = payload.get("quest_farm_excluded", self.bot_config.get("quest_farm_excluded", []))
            if isinstance(quest_excluded, list):
                self.bot_config["quest_farm_excluded"] = sorted({str(item).lower() for item in quest_excluded if item})

        save_dict_as_toml(self.bot_config, "cfg/bot_config.toml")
        self._emit_state()
        if farm_enabled:
            self._notification("success", "Farm settings saved. Trophy Farm will build its queue the next time you start the bot.")
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
            "cpu_or_gpu",
            "super_debug",
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
        ):
            if key in general:
                self.general_config[key] = general[key]

        self.general_config["emulator_port"] = self._as_int(self.general_config.get("emulator_port", 5037), 5037)
        self.general_config["run_for_minutes"] = self._as_int(self.general_config.get("run_for_minutes", 600), 600)
        self.general_config["auto_push_target_trophies"] = self._as_int(self.general_config.get("auto_push_target_trophies", 1000), 1000)

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
                self._notification("warning", f"Trophy Farm is enabled, but no eligible brawlers are below {target} trophies.")
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
        self._notification("success", "Bot started.")
        self._emit_state()

    @Slot()
    def stopBot(self):
        self._bot_stop_requested = True
        if self._bot_stop_event:
            self._bot_stop_event.set()
        if self._bot_pause_event:
            self._bot_pause_event.clear()
        self._notification("info", "Stop signal sent to the bot.")

    @Slot()
    def on_app_about_to_quit(self):
        if self._bot_stop_event:
            self._bot_stop_event.set()
