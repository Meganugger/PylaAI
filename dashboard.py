"""
PylaAI Dashboard - Unified Bot Interface
==========================================
Single-window application with sidebar navigation.
Pages: Home, Settings, Brawlers, Live Stats.
Bot runs in a background thread while the dashboard stays open.
"""

import customtkinter as ctk
import tkinter as tk
import threading
import time
import os
import json
import re
import subprocess
import webbrowser
import inspect

from PIL import Image
from customtkinter import CTkImage
from tkinter import filedialog

from gui.theme import COLORS, FONT_FAMILY, FONT_FAMILY_ALT, S, apply_appearance, font, ui_font
from lobby_automation import LobbyAutomation
from stage_manager import StageManager
from utils import (
    load_toml_as_dict, save_dict_as_toml, update_toml_file,
    load_brawlers_info, save_brawler_icon,
    api_base_url, save_brawler_data, get_discord_link
)


# Unified control-center palette
BG       = COLORS["bg"]
BG_ALT   = COLORS["bg_alt"]
SIDEBAR  = COLORS["sidebar"]
PANEL    = COLORS["surface"]
SECTION  = COLORS["surface_alt"]
CARD     = COLORS["surface_alt_2"]
ACCENT   = COLORS["accent"]
ACCENT_H = COLORS["accent_hover"]
ACCENT_D = COLORS["accent_soft"]
GOLD     = COLORS["gold"]
GREEN    = COLORS["success"]
RED      = COLORS["danger"]
BLUE     = COLORS["info"]
CYAN     = COLORS["accent_muted"]
PURPLE   = COLORS["accent_dim"]
TXT      = COLORS["text"]
DIM      = COLORS["muted"]
DIM_ALT  = COLORS["muted_alt"]
BRIGHT   = COLORS["text_bright"]
SEP      = COLORS["border"]
SEP_STRONG = COLORS["border_strong"]
HP_G     = COLORS["success"]
HP_Y     = COLORS["warning"]
HP_R     = COLORS["danger"]

PS_COLORS = {
    "fighter": GOLD, "tank": RED, "sniper": CYAN,
    "assassin": ACCENT, "thrower": PURPLE, "support": GREEN,
}

# gamemode map (value -> display)
GAMEMODES = {
    "brawlball": ("Brawl Ball", 3),
    "knockout": ("Knockout", 3),
    "gemgrab": ("Gem Grab", 3),
    "showdown": ("Showdown", 1),
    "basketbrawl": ("Basket Brawl", 3),
    "wipeout": ("Wipeout", 3),
    "bounty": ("Bounty", 3),
    "hotzone": ("Hot Zone", 3),
    "heist": ("Heist", 3),
    "duels": ("Duels", 1),
    "5v5": ("5v5", 5),
    "other": ("Other", 3),
}

EMULATORS = ["BlueStacks", "LDPlayer", "MEmu", "Others"]


class Dashboard(ctk.CTk):
    """Unified PylaAI dashboard - single window, sidebar navigation."""

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

    # --- iNIT ---

    def __init__(self, version_str, brawlers, pyla_main_fn,
                 login_fn=None, latest_version_fn=None):
        super().__init__()

        self.version_str = str(version_str).strip()
        self.version_tag = self._format_version_tag(self.version_str)
        self.all_brawlers = brawlers
        self._pyla_main = pyla_main_fn
        self._login_fn = login_fn
        self._logged_in = False
        self._latest_version_fn = latest_version_fn
        self._full_branch_live = all(os.path.exists(path) for path in ("behavior_tree.py", "bt_combat.py"))
        self._capabilities = {
            "visual_overlay": os.path.exists("visual_overlay.py"),
            "advanced_live": self._full_branch_live,
            "brawler_scan": hasattr(LobbyAutomation, "scan_all_brawlers"),
            "quest_farm": hasattr(StageManager, "_handle_quest_rotation"),
            "quest_scan": hasattr(LobbyAutomation, "scan_quest_brawlers"),
        }
        self._page_nav_owner = {
            "home": "home",
            "brawler": "brawler",
            "farm": "farm",
            "quest": "farm",
            "live": "live",
            "history": "history",
            "settings": "settings",
        }

        # state
        self.bot_running = False
        self.bot_thread = None
        self._bot_stop_event = None
        self._bot_pause_event = None
        self._bot_stop_requested = False
        self.brawlers_data = []
        self.farm_type = ""
        self._live_data = {}
        self._prev_live = {}          # previous widget values for change detection
        self._trophy_farm_active = False
        self._trophy_farm_target = 500
        self._trophy_farm_strategy = "lowest_first"
        self._trophy_farm_excluded = set()
        self._brawler_scan_data = {}  # Loaded from cfg/brawler_scan.json
        self._brawler_grid_refresh_after_id = None
        self._brawler_sorted_cache = None
        self._brawler_render_job_id = None
        self._brawler_render_rows = []
        self._brawler_render_cursor = 0
        self._scan_in_progress = False
        self._live_running = True
        self._live_lock = threading.Lock()
        self._last_live_update_ts = 0.0
        self._live_session_start_ts = None
        self._wins_per_hour_ema = None
        self._training_stats_cache = {}
        self._training_stats_mtime = 0.0
        self._last_history_refresh_ts = 0.0
        self._quest_farm_active = False
        self._quest_brawlers = []          # brawler names with active quests
        self._quest_farm_excluded = set()
        self._quest_scan_in_progress = False

        # load configs
        self.bot_config = load_toml_as_dict("cfg/bot_config.toml")
        self.general_config = load_toml_as_dict("cfg/general_config.toml")
        self.login_config = load_toml_as_dict("cfg/login.toml")
        self.time_tresholds = load_toml_as_dict("cfg/time_tresholds.toml")
        self.match_history = load_toml_as_dict("cfg/match_history.toml")
        self.brawlers_info = load_brawlers_info()
        self._apply_defaults()
        self._load_scan_data()
        self._load_excluded_brawlers()

        # window
        self.withdraw()
        self.title(f"Pyla Control Center - {self.version_tag}")
        w, h = S(1420), S(860)
        self.geometry(f"{w}x{h}")
        self.minsize(S(1140), S(700))
        self.configure(fg_color=BG)
        apply_appearance()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # build UI
        self._build_sidebar()
        self._content = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        self._content.pack(side="right", fill="both", expand=True)

        self._pages = {}
        self._page_builders = {
            "settings": self._build_settings_page,
            "home": self._build_home_page,
            "brawler": self._build_brawler_page,
            "farm": self._build_farm_page,
            "quest": self._build_quest_page,
            "live": self._build_live_page,
            "history": self._build_history_page,
        }
        self._current_page = None
        self._build_settings_page()
        self._build_home_page()
        self._build_live_page()
        self.show_page("home")
        self.after_idle(self.deiconify)

    def _load_scan_data(self):
        """Load brawler scan data from cfg/brawler_scan.json if it exists."""
        scan_path = "cfg/brawler_scan.json"
        if os.path.exists(scan_path):
            try:
                with open(scan_path, 'r') as f:
                    data = json.load(f)
                self._brawler_scan_data = data.get("brawlers", {})
                ts = data.get("timestamp", "")
                print(f"[SCAN] Loaded scan data for {len(self._brawler_scan_data)} brawlers (scanned: {ts})")
            except Exception as e:
                print(f"[SCAN] Error loading scan data: {e}")
                self._brawler_scan_data = {}
        else:
            self._brawler_scan_data = {}

    def _save_scan_data(self, scan_results):
        """Save brawler scan results to cfg/brawler_scan.json."""
        import datetime
        data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "brawlers": scan_results
        }
        scan_path = "cfg/brawler_scan.json"
        try:
            with open(scan_path, 'w') as f:
                json.dump(data, f, indent=4)
            self._brawler_scan_data = scan_results
            self._brawler_sorted_cache = None
            for entry in self.brawlers_data:
                if entry.get("brawler") in scan_results:
                    entry["manual_trophies"] = False
            print(f"[SCAN] Saved scan data for {len(scan_results)} brawlers")
        except Exception as e:
            print(f"[SCAN] Error saving scan data: {e}")

    def _schedule_brawler_grid_refresh(self):
        if self._brawler_grid_refresh_after_id is not None:
            try:
                self.after_cancel(self._brawler_grid_refresh_after_id)
            except Exception:
                pass
        if self._brawler_render_job_id is not None:
            try:
                self.after_cancel(self._brawler_render_job_id)
            except Exception:
                pass
            self._brawler_render_job_id = None
        self._brawler_grid_refresh_after_id = self.after(150, self._refresh_brawler_grid)

    def _get_selected_brawler_entry(self, brawler):
        for entry in self.brawlers_data:
            if entry.get("brawler") == brawler:
                return entry
        return None

    def _get_effective_trophies(self, brawler, selected_entry=None, scan_entry=None):
        selected = selected_entry if selected_entry is not None else self._get_selected_brawler_entry(brawler)
        if selected is not None and selected.get("manual_trophies", False):
            manual_value = selected.get("trophies")
            if isinstance(manual_value, int):
                return manual_value, "manual"

        if scan_entry is None:
            scan_entry = self._brawler_scan_data.get(brawler, {})
        scan_value = scan_entry.get("trophies") if isinstance(scan_entry, dict) else None
        if isinstance(scan_value, int):
            return scan_value, "scan"

        if selected is not None:
            selected_value = selected.get("trophies")
            if isinstance(selected_value, int):
                return selected_value, "session"

        return 0, "default"

    def _build_sorted_brawler_cache(self):
        has_scan = bool(self._brawler_scan_data)
        cache = list(self._brawler_images)
        if has_scan:
            cache.sort(
                key=lambda x: (0 if self._brawler_scan_data.get(x[0], {}).get("unlocked", True) else 1, x[0])
            )
        self._brawler_sorted_cache = cache

    def _collect_visible_brawlers(self):
        filt = self._brawler_filter.get().lower()
        role_filter = self._role_filter_var.get().strip().lower() if hasattr(self, "_role_filter_var") else "all roles"
        hide_locked = getattr(self, '_hide_locked_var', None) and self._hide_locked_var.get()
        has_scan = bool(self._brawler_scan_data)
        selected_entries = {
            d.get("brawler"): d for d in self.brawlers_data
            if isinstance(d, dict) and d.get("brawler")
        }
        if self._brawler_sorted_cache is None:
            self._build_sorted_brawler_cache()
        sorted_imgs = self._brawler_sorted_cache or self._brawler_images

        rows = []
        for b, img_tk in sorted_imgs:
            if filt and filt not in b.lower():
                continue
            scan_entry = self._brawler_scan_data.get(b, {})
            b_unlocked = scan_entry.get("unlocked", True)
            if hide_locked and has_scan and not b_unlocked:
                continue

            info = self.brawlers_info.get(b, {})
            ps = info.get("playstyle", "")
            if role_filter not in ("", "all roles") and ps.lower() != role_filter:
                continue

            selected_entry = selected_entries.get(b)
            hist = self.match_history.get(b, {})
            wins = hist.get("victory", 0) if isinstance(hist, dict) else 0
            losses = hist.get("defeat", 0) if isinstance(hist, dict) else 0
            total = wins + losses
            wr = round(100 * wins / total) if total > 0 else -1
            trophy_value, trophy_source = self._get_effective_trophies(
                b, selected_entry=selected_entry, scan_entry=scan_entry
            )

            rows.append({
                "name": b,
                "img": img_tk,
                "scan": scan_entry,
                "unlocked": b_unlocked,
                "selected": selected_entry,
                "playstyle": ps,
                "wins": wins,
                "losses": losses,
                "total": total,
                "wr": wr,
                "trophies": trophy_value if isinstance(trophy_value, int) else 0,
                "trophy_source": trophy_source,
            })

        sort_mode = self._sort_mode_var.get() if hasattr(self, "_sort_mode_var") else "Unlocked"
        if sort_mode == "Name A-Z":
            rows.sort(key=lambda r: r["name"])
        elif sort_mode == "Trophies Low->High":
            rows.sort(key=lambda r: (r["trophies"], r["name"]))
        elif sort_mode == "Trophies High->Low":
            rows.sort(key=lambda r: (-r["trophies"], r["name"]))
        elif sort_mode == "Winrate High->Low":
            rows.sort(key=lambda r: ((-1 if r["wr"] < 0 else 0), -r["wr"], r["name"]))
        else:
            rows.sort(key=lambda r: (0 if r["unlocked"] else 1, r["name"]))

        return rows

    def _select_visible_brawlers(self):
        rows = self._collect_visible_brawlers()
        visible_unlocked = [r for r in rows if r.get("unlocked", True)]
        if not visible_unlocked:
            self._status_label.configure(text="⚠ No visible unlocked brawlers", text_color=GOLD)
            return

        mode = self._selection_mode_var.get() if hasattr(self, "_selection_mode_var") else "Multi"
        if mode == "Single":
            visible_unlocked = visible_unlocked[:1]

        selected_map = {
            d.get("brawler"): d for d in self.brawlers_data
            if isinstance(d, dict) and d.get("brawler")
        }

        new_entries = []
        for row in visible_unlocked:
            brawler = row["name"]
            existing = selected_map.get(brawler)
            if existing is not None:
                new_entries.append(existing)
                continue
            new_entries.append({
                "brawler": brawler,
                "push_until": "",
                "trophies": row.get("trophies", 0),
                "wins": "",
                "type": "trophies",
                "automatically_pick": True,
                "win_streak": 0,
                "manual_trophies": False,
            })

        if mode == "Single":
            self.brawlers_data = new_entries[:1]
        else:
            self.brawlers_data = [
                d for d in self.brawlers_data
                if d.get("brawler") not in {r["name"] for r in visible_unlocked}
            ] + new_entries

        self._refresh_brawler_grid()
        self._update_sidebar_brawler()
        self._status_label.configure(text=f"✔ Selected {len(new_entries)} visible brawler(s)", text_color=GREEN)

    def _clear_brawler_selection(self):
        self.brawlers_data = []
        self._refresh_brawler_grid()
        self._update_sidebar_brawler()
        self._status_label.configure(text="✔ Brawler selection cleared", text_color=GREEN)

    def _load_excluded_brawlers(self):
        """Load excluded brawlers from bot_config."""
        excluded = self.bot_config.get("trophy_farm_excluded", [])
        if isinstance(excluded, list):
            self._trophy_farm_excluded = set(excluded)
        else:
            self._trophy_farm_excluded = set()

        quest_excluded = self.bot_config.get("quest_farm_excluded", [])
        if isinstance(quest_excluded, list):
            self._quest_farm_excluded = set(quest_excluded)
        else:
            self._quest_farm_excluded = set()

        # handle login
        self._handle_login()

    # --- cONFIG DEFAULTS ---

    def _apply_defaults(self):
        bc = self.bot_config
        bc.setdefault("gamemode_type", 3)
        bc.setdefault("gamemode", "brawlball")
        bc.setdefault("bot_uses_gadgets", "yes")
        bc.setdefault("minimum_movement_delay", 0.4)
        bc.setdefault("wall_detection_confidence", 0.9)
        bc.setdefault("entity_detection_confidence", 0.6)
        bc.setdefault("unstuck_movement_delay", 3.0)
        bc.setdefault("unstuck_movement_hold_time", 1.5)
        bc.setdefault("seconds_to_hold_attack_after_reaching_max", 1.5)
        bc.setdefault("play_again_on_win", "no")
        bc.setdefault("smart_trophy_farm", "no")
        bc.setdefault("trophy_farm_target", 500)
        bc.setdefault("trophy_farm_strategy", "lowest_first")
        bc.setdefault("trophy_farm_excluded", [])
        bc.setdefault("quest_farm_enabled", "no")
        bc.setdefault("quest_farm_mode", "games")
        bc.setdefault("quest_farm_excluded", [])

        gc = self.general_config
        gc.setdefault("max_ips", "auto")
        gc.setdefault("super_debug", "yes")
        gc.setdefault("cpu_or_gpu", "auto")
        gc.setdefault("current_emulator", "LDPlayer")
        gc.setdefault("visual_overlay_enabled", "no")
        gc.setdefault("visual_overlay_opacity", 180)
        gc.setdefault("run_for_minutes", 0)
        gc.setdefault("map_orientation", "vertical")
        gc.setdefault("long_press_star_drop", "no")
        gc.setdefault("trophies_multiplier", 1.0)

        tt = self.time_tresholds
        tt.setdefault("state_check", 3)
        tt.setdefault("no_detections", 10)
        tt.setdefault("idle", 10)
        tt.setdefault("super", 0.1)
        tt.setdefault("gadget", 0.5)
        tt.setdefault("hypercharge", 2)

    # --- lOGIN ---

    def _handle_login(self):
        """Check API login. If localhost, auto-pass. Otherwise verify key."""
        if api_base_url == "localhost":
            self._logged_in = True
            return

        # Check saved key
        try:
            from gui.api import check_if_exists
            login_data = load_toml_as_dict('./cfg/login.toml')
            key = login_data.get('key', '')
            if key and check_if_exists(key):
                self._logged_in = True
                return
        except Exception:
            pass

        # Show login dialog
        self._show_login_dialog()

    def _show_login_dialog(self):
        dialog = ctk.CTkToplevel(self)
        dialog.title("PylaAI - Login")
        dialog.geometry(f"{S(460)}x{S(220)}")
        dialog.configure(fg_color=BG)
        dialog.transient(self)
        dialog.grab_set()
        dialog.attributes("-topmost", True)

        ctk.CTkLabel(dialog, text="\u2605 PylaAI Login",
                     font=("Segoe UI", S(20), "bold"),
                     text_color=ACCENT).pack(pady=(S(16), S(8)))

        ctk.CTkLabel(dialog, text="Enter your API Key:",
                     font=("Segoe UI", S(14)), text_color=TXT).pack()

        entry = ctk.CTkEntry(dialog, placeholder_text="Paste API Key...",
                             font=("Segoe UI", S(14)), width=S(360),
                             fg_color=PANEL, border_color=ACCENT,
                             text_color=BRIGHT, corner_radius=S(6))
        entry.pack(pady=S(8))
        existing_key = str(self.login_config.get("key", "")).strip()
        if existing_key:
            entry.insert(0, existing_key)

        status = ctk.CTkLabel(dialog, text="", font=("Segoe UI", S(12)))
        status.pack()

        def try_login():
            try:
                from gui.api import check_if_exists
                key = entry.get().strip()
                if check_if_exists(key):
                    self._logged_in = True
                    update_toml_file("./cfg/login.toml", {"key": key})
                    self.login_config["key"] = key
                    dialog.destroy()
                else:
                    status.configure(text="Invalid API Key", text_color=RED)
            except Exception as e:
                status.configure(text=f"Error: {e}", text_color=RED)

        ctk.CTkButton(dialog, text="Login", command=try_login,
                      fg_color=ACCENT, hover_color=ACCENT_H,
                      font=("Segoe UI", S(14), "bold"),
                      width=S(160), height=S(36),
                      corner_radius=S(8)).pack(pady=S(4))

    # --- sIDEBAR ---

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=S(252), fg_color=SIDEBAR, corner_radius=0)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)
        self._sidebar = sb

        logo = ctk.CTkFrame(sb, fg_color="transparent", height=S(112))
        logo.pack(fill="x", padx=S(18), pady=(S(14), S(8)))
        logo.pack_propagate(False)
        ctk.CTkLabel(
            logo,
            text="PYLA",
            font=(FONT_FAMILY, S(23), "bold"),
            text_color=BRIGHT,
            anchor="w",
        ).pack(anchor="w", pady=(S(10), 0))
        ctk.CTkFrame(logo, fg_color=ACCENT, width=S(34), height=S(3), corner_radius=S(6)).pack(anchor="w", pady=(S(6), S(10)))
        ctk.CTkLabel(
            logo,
            text="CONTROL CENTER",
            font=(FONT_FAMILY_ALT, S(10), "bold"),
            text_color=DIM,
            anchor="w",
        ).pack(anchor="w")
        ctk.CTkLabel(
            logo,
            text=f"Unified shell  {self.version_tag}",
            font=(FONT_FAMILY_ALT, S(11)),
            text_color=DIM_ALT,
            anchor="w",
        ).pack(anchor="w", pady=(S(6), 0))

        ctk.CTkFrame(sb, fg_color=SEP, height=1).pack(fill="x", padx=S(18), pady=(0, S(12)))

        # navigation
        self._nav_btns = {}
        nav_items = [
            ("home", "Control Center"),
            ("brawler", "Brawlers"),
            ("farm", "Farm"),
            ("live", "Live"),
            ("history", "History"),
        ]
        ctk.CTkLabel(
            sb,
            text="NAVIGATION",
            font=(FONT_FAMILY_ALT, S(10), "bold"),
            text_color=DIM,
        ).pack(anchor="w", padx=S(18), pady=(0, S(8)))
        for key, label in nav_items:
            btn = ctk.CTkButton(
                sb,
                text=label,
                anchor="w",
                font=(FONT_FAMILY_ALT, S(14), "bold"),
                fg_color="transparent",
                text_color=TXT,
                hover_color=SECTION,
                height=S(48),
                corner_radius=S(12),
                border_width=1,
                border_color=SIDEBAR,
                command=lambda k=key: self.show_page(k)
            )
            btn.pack(fill="x", padx=S(14), pady=S(3))
            self._nav_btns[key] = btn

        # spacer
        ctk.CTkFrame(sb, fg_color="transparent").pack(fill="both", expand=True)

        self._sb_brawler_frame = ctk.CTkFrame(sb, fg_color=PANEL, corner_radius=S(14), border_width=1, border_color=SEP)
        self._sb_brawler_frame.pack(fill="x", padx=S(14), pady=(S(4), S(8)))
        ctk.CTkLabel(self._sb_brawler_frame, text="SELECTED ROSTER",
                     font=(FONT_FAMILY_ALT, S(10), "bold"), text_color=DIM).pack(
                         anchor="w", padx=S(12), pady=(S(10), 0))
        self._sb_brawler_label = ctk.CTkLabel(
            self._sb_brawler_frame, text="None",
            font=(FONT_FAMILY_ALT, S(16), "bold"), text_color=BRIGHT)
        self._sb_brawler_label.pack(anchor="w", padx=S(12), pady=(S(2), 0))
        self._sb_brawler_meta = ctk.CTkLabel(
            self._sb_brawler_frame,
            text="0 ready",
            font=(FONT_FAMILY_ALT, S(11)),
            text_color=DIM_ALT,
        )
        self._sb_brawler_meta.pack(anchor="w", padx=S(12), pady=(0, S(10)))

        self._status_frame = ctk.CTkFrame(sb, fg_color=PANEL,
                                          corner_radius=S(12), height=S(48), border_width=1, border_color=SEP)
        self._status_frame.pack(fill="x", padx=S(14), pady=S(4))
        self._status_frame.pack_propagate(False)
        self._status_label = ctk.CTkLabel(
            self._status_frame, text="\u25CF READY",
            font=(FONT_FAMILY_ALT, S(12), "bold"), text_color=DIM)
        self._status_label.pack(expand=True)

        self._start_btn = ctk.CTkButton(
            sb, text="\u25B6  START BOT",
            font=(FONT_FAMILY_ALT, S(16), "bold"),
            fg_color=ACCENT, hover_color=ACCENT_H, text_color=BRIGHT,
            height=S(52), corner_radius=S(12),
            command=self._toggle_bot
        )
        self._start_btn.pack(fill="x", padx=S(14), pady=(S(6), S(10)))

        bottom = ctk.CTkFrame(sb, fg_color="transparent")
        bottom.pack(fill="x", padx=S(14), pady=(0, S(16)))
        self._settings_btn = ctk.CTkButton(
            bottom,
            text="Settings",
            anchor="w",
            font=(FONT_FAMILY_ALT, S(13), "bold"),
            fg_color=SECTION,
            hover_color=CARD,
            text_color=TXT,
            height=S(44),
            corner_radius=S(12),
            border_width=1,
            border_color=SEP,
            command=lambda: self.show_page("settings"),
        )
        self._settings_btn.pack(fill="x")
        self._sidebar_footer = ctk.CTkLabel(
            bottom,
            text="Device: disconnected",
            font=(FONT_FAMILY_ALT, S(10)),
            text_color=DIM_ALT,
        )
        self._sidebar_footer.pack(anchor="w", pady=(S(8), 0))

    # --- pAGE SWITCHING ---

    def show_page(self, name):
        if name not in self._pages and name in self._page_builders:
            self._page_builders[name]()
        if name not in self._pages:
            name = "home"
        for frame in self._pages.values():
            frame.pack_forget()
        if name in self._pages:
            self._pages[name].pack(fill="both", expand=True)
        self._current_page = name
        active_key = self._page_nav_owner.get(name, name)
        for key, btn in self._nav_btns.items():
            if key == active_key:
                btn.configure(fg_color=SECTION, text_color=BRIGHT, border_color=SEP_STRONG)
            else:
                btn.configure(fg_color="transparent", text_color=TXT, border_color=SIDEBAR)
        if hasattr(self, "_settings_btn"):
            if active_key == "settings":
                self._settings_btn.configure(fg_color=SECTION, border_color=SEP_STRONG, text_color=BRIGHT)
            else:
                self._settings_btn.configure(fg_color=SECTION, border_color=SEP, text_color=TXT)

    # --- hOME PAGE ---

    def _build_home_page(self):
        page = ctk.CTkScrollableFrame(self._content, fg_color=BG,
                                      corner_radius=0,
                                      scrollbar_button_color=ACCENT)
        self._pages["home"] = page

        hdr = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(18), height=S(92), border_width=1, border_color=SEP)
        hdr.pack(fill="x", padx=S(20), pady=(S(16), S(12)))
        hdr.pack_propagate(False)
        left = ctk.CTkFrame(hdr, fg_color="transparent")
        left.pack(side="left", fill="both", expand=True, padx=S(20), pady=S(16))
        ctk.CTkLabel(left, text="PYLA CONTROL CENTER",
                     font=(FONT_FAMILY, S(28), "bold"),
                     text_color=BRIGHT).pack(anchor="w")
        ctk.CTkLabel(left, text="One place for launch setup, roster management, farm routing and live control.",
                     font=(FONT_FAMILY_ALT, S(12)), text_color=DIM_ALT).pack(anchor="w", pady=(S(4), 0))
        right = ctk.CTkFrame(hdr, fg_color="transparent")
        right.pack(side="right", padx=S(18), pady=S(16))
        self._home_status_pill = ctk.CTkLabel(
            right,
            text="READY",
            fg_color=SECTION,
            corner_radius=S(999),
            padx=S(16),
            pady=S(8),
            font=(FONT_FAMILY_ALT, S(11), "bold"),
            text_color=BRIGHT,
        )
        self._home_status_pill.pack(anchor="e")
        ctk.CTkLabel(right, text=self.version_tag,
                     font=(FONT_FAMILY_ALT, S(11)), text_color=DIM).pack(anchor="e", pady=(S(8), 0))

        cards = ctk.CTkFrame(page, fg_color="transparent")
        cards.pack(fill="x", padx=S(20), pady=S(4))
        cards.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._home_brawler_lbl = self._info_card(
            cards, "Primary Brawler", "None", ACCENT, 0, 0)
        gm = self.bot_config.get("gamemode", "?").title()
        self._home_gm_lbl = self._info_card(
            cards, "Gamemode", gm, GOLD, 0, 1)
        emu = self.general_config.get("current_emulator", "?")
        self._home_emu_lbl = self._info_card(
            cards, "Emulator", emu, CYAN, 0, 2)
        t = int(self.general_config.get("run_for_minutes", 0))
        self._home_timer_lbl = self._info_card(
            cards, "Runtime", f"{t} min" if t > 0 else "\u221E", PURPLE, 0, 3)

        control_grid = ctk.CTkFrame(page, fg_color="transparent")
        control_grid.pack(fill="x", padx=S(20), pady=(S(6), S(4)))
        control_grid.grid_columnconfigure((0, 1), weight=1)

        launch_card = ctk.CTkFrame(control_grid, fg_color=PANEL, corner_radius=S(18), border_width=1, border_color=SEP)
        launch_card.grid(row=0, column=0, sticky="nsew", padx=(0, S(8)), pady=0)
        ctk.CTkLabel(launch_card, text="Launch Configuration",
                     font=(FONT_FAMILY_ALT, S(17), "bold"), text_color=BRIGHT).pack(anchor="w", padx=S(18), pady=(S(16), S(4)))
        ctk.CTkLabel(launch_card, text="Set the essentials once, then launch from here.",
                     font=(FONT_FAMILY_ALT, S(12)), text_color=DIM_ALT).pack(anchor="w", padx=S(18), pady=(0, S(10)))

        gm_values = [v[0] for v in GAMEMODES.values()]
        self._cc_gm_primary_values = ["Brawl Ball", "Knockout", "Gem Grab", "Other"]
        current_gm = self.bot_config.get("gamemode", "brawlball")
        current_display_gm = GAMEMODES.get(current_gm, ("Brawl Ball", 3))[0]
        current_primary = current_display_gm if current_display_gm in self._cc_gm_primary_values[:-1] else "Other"
        self._cc_gm_var = ctk.StringVar(value=current_primary)
        self._control_segment_row(launch_card, "Gamemode", self._cc_gm_primary_values, self._cc_gm_var, self._on_control_center_gamemode_change)

        self._cc_other_gm_var = ctk.StringVar(value=current_display_gm)
        other_row = ctk.CTkFrame(launch_card, fg_color="transparent")
        other_row.pack(fill="x", padx=S(18), pady=(0, S(10)))
        ctk.CTkLabel(other_row, text="Extended List", font=(FONT_FAMILY_ALT, S(12), "bold"), text_color=TXT).pack(side="left")
        ctk.CTkOptionMenu(
            other_row,
            variable=self._cc_other_gm_var,
            values=gm_values,
            font=(FONT_FAMILY_ALT, S(12)),
            fg_color=SECTION,
            button_color=ACCENT,
            button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL,
            dropdown_hover_color=CARD,
            width=S(180),
            command=self._on_control_center_gamemode_change,
        ).pack(side="right")

        self._cc_orient_var = ctk.StringVar(value=self.general_config.get("map_orientation", "vertical").title())
        self._control_segment_row(launch_card, "Map Orientation", ["Vertical", "Horizontal"], self._cc_orient_var, self._on_control_center_orientation_change)

        self._cc_emu_var = ctk.StringVar(value=self.general_config.get("current_emulator", "LDPlayer"))
        self._control_segment_row(launch_card, "Emulator Target", EMULATORS, self._cc_emu_var, self._on_control_center_emulator_change)

        timer_row = ctk.CTkFrame(launch_card, fg_color="transparent")
        timer_row.pack(fill="x", padx=S(18), pady=(S(2), S(16)))
        ctk.CTkLabel(timer_row, text="Run Timer (minutes, 0 = infinite)",
                     font=(FONT_FAMILY_ALT, S(12), "bold"), text_color=TXT).pack(side="left")
        self._cc_timer_var = tk.StringVar(value=str(self.general_config.get("run_for_minutes", 0)))
        ctk.CTkEntry(
            timer_row,
            textvariable=self._cc_timer_var,
            font=(FONT_FAMILY_ALT, S(12)),
            width=S(92),
            height=S(34),
            fg_color=SECTION,
            border_color=SEP_STRONG,
            text_color=BRIGHT,
            corner_radius=S(10),
            justify="center",
        ).pack(side="right")

        action_card = ctk.CTkFrame(control_grid, fg_color=PANEL, corner_radius=S(18), border_width=1, border_color=SEP)
        action_card.grid(row=0, column=1, sticky="nsew", padx=(S(8), 0), pady=0)
        ctk.CTkLabel(action_card, text="Ready Queue",
                     font=(FONT_FAMILY_ALT, S(17), "bold"), text_color=BRIGHT).pack(anchor="w", padx=S(18), pady=(S(16), S(4)))
        self._home_roster_summary = ctk.CTkLabel(
            action_card,
            text="No brawlers selected yet. Build your roster in the Brawlers page.",
            font=(FONT_FAMILY_ALT, S(12)),
            text_color=DIM_ALT,
            justify="left",
            wraplength=S(420),
        )
        self._home_roster_summary.pack(anchor="w", padx=S(18), pady=(0, S(12)))

        cta_row = ctk.CTkFrame(action_card, fg_color="transparent")
        cta_row.pack(fill="x", padx=S(18), pady=(0, S(10)))
        self._home_action_btn = ctk.CTkButton(
            cta_row,
            text="START BOT",
            font=(FONT_FAMILY_ALT, S(18), "bold"),
            fg_color=ACCENT,
            hover_color=ACCENT_H,
            height=S(54),
            corner_radius=S(14),
            command=self._toggle_bot,
        )
        self._home_action_btn.pack(fill="x")

        quick_row = ctk.CTkFrame(action_card, fg_color="transparent")
        quick_row.pack(fill="x", padx=S(18), pady=(0, S(8)))
        ctk.CTkButton(
            quick_row,
            text="Open Brawlers",
            font=(FONT_FAMILY_ALT, S(12), "bold"),
            fg_color=SECTION,
            hover_color=CARD,
            height=S(38),
            corner_radius=S(12),
            border_width=1,
            border_color=SEP,
            command=lambda: self.show_page("brawler"),
        ).pack(side="left", expand=True, fill="x", padx=(0, S(5)))
        ctk.CTkButton(
            quick_row,
            text="Farm Modes",
            font=(FONT_FAMILY_ALT, S(12), "bold"),
            fg_color=SECTION,
            hover_color=CARD,
            height=S(38),
            corner_radius=S(12),
            border_width=1,
            border_color=SEP,
            command=lambda: self.show_page("farm"),
        ).pack(side="left", expand=True, fill="x", padx=(S(5), 0))

        hint = ctk.CTkFrame(page, fg_color=SECTION, corner_radius=S(16), border_width=1, border_color=SEP)
        hint.pack(fill="x", padx=S(20), pady=(S(12), S(8)))
        ctk.CTkLabel(hint, text="\u2139  Login stays intact when needed, but the setup flow is now unified here. If you are running localhost mode, you can launch immediately once your roster is ready.",
                     font=(FONT_FAMILY_ALT, S(12)), text_color=TXT,
                     wraplength=S(980), justify="left").pack(anchor="w", padx=S(16), pady=S(14))

        ctk.CTkFrame(page, fg_color=SEP, height=1).pack(
            fill="x", padx=S(20), pady=S(10))
        ctk.CTkLabel(page, text="Recent Match History",
                     font=(FONT_FAMILY_ALT, S(18), "bold"),
                     text_color=BRIGHT).pack(anchor="w", padx=S(24), pady=(S(4), S(6)))
        self._home_hist_frame = ctk.CTkFrame(page, fg_color="transparent")
        self._home_hist_frame.pack(fill="x", padx=S(20), pady=S(4))
        self._refresh_home_history()

        info = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(16), border_width=1, border_color=SEP)
        info.pack(fill="x", padx=S(20), pady=(S(10), S(16)))
        ctk.CTkLabel(info,
                     text="F9 Stop  \u2502  F8 Pause/Resume  \u2502  F7 Toggle Overlay",
                     font=(FONT_FAMILY_ALT, S(12)), text_color=DIM).pack(pady=S(12))

    def _info_card(self, parent, title, value, color, row, col):
        card = ctk.CTkFrame(parent, fg_color=PANEL, corner_radius=S(16),
                            height=S(94), border_width=1, border_color=SEP)
        card.grid(row=row, column=col, padx=S(5), pady=S(4), sticky="nsew")
        card.pack_propagate(False)
        ctk.CTkLabel(card, text=title, font=(FONT_FAMILY_ALT, S(10), "bold"),
                     text_color=DIM).pack(anchor="w", padx=S(14), pady=(S(14), 0))
        lbl = ctk.CTkLabel(card, text=str(value),
                           font=(FONT_FAMILY_ALT, S(18), "bold"),
                           text_color=color)
        lbl.pack(anchor="w", padx=S(14))
        return lbl

    def _control_segment_row(self, parent, title, values, variable, command):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=S(18), pady=(0, S(12)))
        ctk.CTkLabel(row, text=title, font=(FONT_FAMILY_ALT, S(12), "bold"), text_color=TXT).pack(anchor="w", pady=(0, S(6)))
        ctk.CTkSegmentedButton(
            row,
            values=values,
            variable=variable,
            command=command,
            height=S(36),
            fg_color=SECTION,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_H,
            unselected_color=CARD,
            unselected_hover_color=SECTION,
            text_color=TXT,
            font=(FONT_FAMILY_ALT, S(11), "bold"),
        ).pack(fill="x")

    def _refresh_home_history(self):
        for w in self._home_hist_frame.winfo_children():
            w.destroy()

        hist = self.match_history
        rows = []
        for brawler, data in hist.items():
            if isinstance(data, dict):
                v = data.get("victory", 0)
                d = data.get("defeat", 0)
                dr = data.get("draw", 0)
                total = v + d + dr
                if total > 0:
                    rows.append((brawler, v, d, dr, total, v / total * 100))
        rows.sort(key=lambda x: x[4], reverse=True)

        if not rows:
            ctk.CTkLabel(self._home_hist_frame,
                         text="No match data yet - pick a brawler and start!",
                         font=(FONT_FAMILY_ALT, S(14)), text_color=DIM
                         ).pack(pady=S(16))
            return

        # Header
        hdr = ctk.CTkFrame(self._home_hist_frame, fg_color="transparent",
                           height=S(26))
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        for txt, w in [("Brawler", S(130)), ("Record", S(140)),
                       ("Games", S(70)), ("Win Rate", S(90))]:
            ctk.CTkLabel(hdr, text=txt, font=(FONT_FAMILY_ALT, S(10), "bold"),
                         text_color=DIM, width=w, anchor="w"
                         ).pack(side="left", padx=S(4))

        for brawler, v, d, dr, total, wr in rows[:12]:
            row = ctk.CTkFrame(self._home_hist_frame, fg_color=PANEL,
                               corner_radius=S(10), height=S(38), border_width=1, border_color=SEP)
            row.pack(fill="x", pady=S(2))
            row.pack_propagate(False)
            ctk.CTkLabel(row, text=brawler.title(),
                         font=(FONT_FAMILY_ALT, S(12), "bold"),
                         text_color=BRIGHT, width=S(130), anchor="w"
                         ).pack(side="left", padx=(S(10), S(4)))
            ctk.CTkLabel(row, text=f"{v}W / {d}L / {dr}D",
                         font=(FONT_FAMILY_ALT, S(11)), text_color=TXT,
                         width=S(140), anchor="w"
                         ).pack(side="left", padx=S(4))
            ctk.CTkLabel(row, text=str(total),
                         font=(FONT_FAMILY_ALT, S(11)), text_color=TXT,
                         width=S(70), anchor="w"
                         ).pack(side="left", padx=S(4))
            wr_c = GREEN if wr >= 50 else RED
            ctk.CTkLabel(row, text=f"{wr:.0f}%",
                         font=(FONT_FAMILY_ALT, S(12), "bold"),
                         text_color=wr_c, width=S(90), anchor="w"
                         ).pack(side="left", padx=S(4))

    def _refresh_history_page(self):
        if not hasattr(self, "_history_results_frame"):
            return
        for widget in self._history_results_frame.winfo_children():
            widget.destroy()

        hist = self.match_history
        rows = []
        for brawler, data in hist.items():
            if isinstance(data, dict):
                v = int(data.get("victory", 0))
                d = int(data.get("defeat", 0))
                dr = int(data.get("draw", 0))
                total = v + d + dr
                if total > 0:
                    rows.append((brawler, v, d, dr, total))
        rows.sort(key=lambda item: item[4], reverse=True)

        if not rows:
            ctk.CTkLabel(
                self._history_results_frame,
                text="No history yet.",
                font=(FONT_FAMILY_ALT, S(12)),
                text_color=DIM,
            ).pack(anchor="w", padx=S(6), pady=S(4))
            return

        for idx, (brawler, v, d, dr, total) in enumerate(rows[:12]):
            wr = (v / total) * 100 if total else 0
            row = ctk.CTkFrame(
                self._history_results_frame,
                fg_color=SECTION if idx % 2 == 0 else PANEL,
                corner_radius=S(10),
            )
            row.pack(fill="x", pady=S(2))
            ctk.CTkLabel(
                row,
                text=brawler.title(),
                font=(FONT_FAMILY_ALT, S(12), "bold"),
                text_color=BRIGHT,
                anchor="w",
            ).pack(side="left", padx=S(12), pady=S(8))
            ctk.CTkLabel(
                row,
                text=f"{v}W / {d}L / {dr}D  •  {total} matches",
                font=(FONT_FAMILY_ALT, S(11)),
                text_color=TXT,
            ).pack(side="left", padx=S(8))
            wr_color = GREEN if wr >= 55 else GOLD if wr >= 45 else RED
            ctk.CTkLabel(
                row,
                text=f"{wr:.0f}% WR",
                font=(FONT_FAMILY_ALT, S(11), "bold"),
                text_color=wr_color,
            ).pack(side="right", padx=S(12))

    def _load_credit_developers(self):
        """Load contributors from git history (name + commit count)."""
        fallback = [
            ("AngelFireLA", 0),
            ("Ivan", 0),
            ("bocchi-the-cat", 0),
            ("mzpoo", 0),
            ("Simon Rejzek", 0),
            ("awarzu", 0),
            ("ivanyordanovgt", 0),
            ("labycatuser", 0),
        ]
        try:
            root = os.path.dirname(os.path.abspath(__file__))
            out = subprocess.check_output(
                ["git", "shortlog", "-sne", "--all"],
                cwd=root,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            agg = {}
            for line in out.splitlines():
                m = re.match(r"\s*(\d+)\s+(.+?)\s+<.+>$", line)
                if not m:
                    continue
                commits = int(m.group(1))
                name = m.group(2).strip()
                key = name.lower()
                if key not in agg:
                    agg[key] = [name, 0]
                agg[key][1] += commits
            data = sorted(((v[0], v[1]) for v in agg.values()), key=lambda x: (-x[1], x[0].lower()))
            return data if data else fallback
        except Exception:
            return fallback

    def _build_history_page(self):
        page = ctk.CTkScrollableFrame(self._content, fg_color=BG,
                                      corner_radius=0,
                                      scrollbar_button_color=ACCENT)
        self._pages["history"] = page

        hdr = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(16), height=S(68), border_width=1, border_color=SEP)
        hdr.pack(fill="x", padx=S(20), pady=(S(16), S(10)))
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="History",
                     font=(FONT_FAMILY_ALT, S(22), "bold"),
                     text_color=BRIGHT).pack(side="left", padx=S(16))
        ctk.CTkLabel(hdr, text=self.version_tag,
                     font=(FONT_FAMILY_ALT, S(13)), text_color=DIM).pack(side="right", padx=S(16))

        hist_card = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(16), border_width=1, border_color=SEP)
        hist_card.pack(fill="x", padx=S(20), pady=(S(2), S(10)))
        ctk.CTkLabel(hist_card, text="Recent Results",
                     font=(FONT_FAMILY_ALT, S(15), "bold"),
                     text_color=BRIGHT).pack(anchor="w", padx=S(14), pady=(S(12), S(8)))
        self._history_results_frame = ctk.CTkFrame(hist_card, fg_color="transparent")
        self._history_results_frame.pack(fill="x", padx=S(10), pady=(0, S(10)))
        self._refresh_history_page()

        info = ctk.CTkFrame(page, fg_color=SECTION, corner_radius=S(10))
        info.pack(fill="x", padx=S(20), pady=S(6))
        ctk.CTkLabel(
            info,
            text="This page keeps the practical history view first, while still keeping project links and credits close by.",
            font=(FONT_FAMILY_ALT, S(12)), text_color=TXT, justify="left"
        ).pack(anchor="w", padx=S(12), pady=S(10))
        ctk.CTkLabel(
            info,
            text="Match history comes from cfg/match_history.toml and updates after each session.",
            font=(FONT_FAMILY_ALT, S(10)), text_color=DIM, justify="left"
        ).pack(anchor="w", padx=S(12), pady=(0, S(8)))

        project = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        project.pack(fill="x", padx=S(20), pady=S(4))
        ctk.CTkLabel(project, text="Project Info",
                     font=(FONT_FAMILY_ALT, S(14), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(12), pady=(S(8), S(4)))
        ctk.CTkLabel(project,
                     text=f"Name: PylaAI\nVersion: {self.version_tag}\n"
                          f"UI: Pyla Control Center\n"
                          "Main Areas: Control Center, Brawlers, Farm, Live, History, Settings",
                     font=(FONT_FAMILY_ALT, S(12)), text_color=TXT,
                     justify="left").pack(anchor="w", padx=S(12), pady=(0, S(10)))

        contrib = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        contrib.pack(fill="x", padx=S(20), pady=S(4))
        ctk.CTkLabel(contrib, text="Developers",
                     font=(FONT_FAMILY_ALT, S(14), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(12), pady=(S(8), S(6)))

        developers = self._load_credit_developers()
        for idx, (name, commits) in enumerate(developers, start=1):
            row = ctk.CTkFrame(contrib, fg_color=SECTION if idx % 2 else PANEL, corner_radius=S(6))
            row.pack(fill="x", padx=S(10), pady=S(1))
            ctk.CTkLabel(row, text=f"#{idx}",
                         font=("Segoe UI", S(11), "bold"), text_color=GOLD,
                         width=S(38), anchor="w").pack(side="left", padx=(S(8), S(2)), pady=S(6))
            ctk.CTkLabel(row, text=name,
                         font=("Segoe UI", S(12), "bold"), text_color=BRIGHT,
                         anchor="w").pack(side="left", padx=S(4), pady=S(6))
            commit_text = f"{commits} commits" if commits > 0 else "contributor"
            ctk.CTkLabel(row, text=commit_text,
                         font=("Segoe UI", S(11)), text_color=DIM,
                         anchor="e").pack(side="right", padx=S(10), pady=S(6))

        discord_card = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        discord_card.pack(fill="x", padx=S(20), pady=(S(6), S(12)))
        ctk.CTkLabel(discord_card, text="Discord",
                     font=(FONT_FAMILY_ALT, S(14), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(12), pady=(S(8), S(4)))

        discord_url = get_discord_link()
        self._credits_discord_url = discord_url
        ctk.CTkLabel(discord_card, text=discord_url,
                     font=("Segoe UI", S(12)), text_color=CYAN,
                     justify="left").pack(anchor="w", padx=S(12), pady=(0, S(6)))

        btn_row = ctk.CTkFrame(discord_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=S(10), pady=(0, S(10)))

        def open_discord():
            try:
                webbrowser.open(self._credits_discord_url)
                self._status_label.configure(text="✔ Opened Discord link", text_color=GREEN)
            except Exception:
                self._status_label.configure(text="⚠ Could not open Discord link", text_color=RED)

        def copy_discord():
            try:
                self.clipboard_clear()
                self.clipboard_append(self._credits_discord_url)
                self._status_label.configure(text="✔ Discord link copied", text_color=GREEN)
            except Exception:
                self._status_label.configure(text="⚠ Could not copy Discord link", text_color=RED)

        ctk.CTkButton(btn_row, text="Open Discord",
                      fg_color=BLUE, hover_color=CYAN, text_color=BG,
                      width=S(160), command=open_discord).pack(side="left", padx=S(4))
        ctk.CTkButton(btn_row, text="Copy Link",
                      fg_color=SEP, hover_color=SECTION, text_color=TXT,
                      width=S(140), command=copy_discord).pack(side="left", padx=S(4))

    # --- sETTINGS PAGE ---

    def _build_settings_page(self):
        page = ctk.CTkScrollableFrame(self._content, fg_color=BG,
                                      corner_radius=0,
                                      scrollbar_button_color=ACCENT)
        self._pages["settings"] = page

        # Header
        ctk.CTkLabel(page, text="Settings",
                     font=("Segoe UI", S(24), "bold"),
                     text_color=BRIGHT).pack(anchor="w", padx=S(24),
                                              pady=(S(16), S(8)))

        # aCCOUNT & API
        self._section_header(page, "Account & API")
        account_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        account_f.pack(fill="x", padx=S(20), pady=S(4))

        self._discord_id_var = tk.StringVar(value=str(self.general_config.get("discord_id", "")))
        self._pyla_key_var = tk.StringVar(value=str(self.login_config.get("key", "")))
        self._bs_api_key_var = tk.StringVar(value=str(self.general_config.get("brawlstars_api_key", "")))
        self._bs_player_tag_var = tk.StringVar(value=str(self.general_config.get("brawlstars_player_tag", "")))

        self._entry_row(account_f, "Discord ID", self._discord_id_var, "Optional user ID for webhook mentions")
        self._entry_row(account_f, "Pyla API Key", self._pyla_key_var, "Used for non-localhost Pyla auth")
        self._entry_row(account_f, "Brawl Stars API Key", self._bs_api_key_var, "Official Brawl Stars API key")
        self._entry_row(account_f, "Player Tag", self._bs_player_tag_var, "Example: #ABC123")

        # gAME SECTION
        self._section_header(page, "Game")
        game_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        game_f.pack(fill="x", padx=S(20), pady=S(4))

        # Gamemode
        row1 = ctk.CTkFrame(game_f, fg_color="transparent")
        row1.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row1, text="Gamemode", font=("Segoe UI", S(13)),
                     text_color=TXT).pack(side="left")
        gm_names = [v[0] for v in GAMEMODES.values()]
        current_gm = self.bot_config.get("gamemode", "brawlball")
        display_gm = GAMEMODES.get(current_gm, ("Brawl Ball", 3))[0]
        self._gm_var = ctk.StringVar(value=display_gm)
        gm_menu = ctk.CTkOptionMenu(
            row1, variable=self._gm_var, values=gm_names,
            font=("Segoe UI", S(13)), fg_color=SECTION,
            button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL, dropdown_hover_color=SEP,
            width=S(180), command=self._on_gamemode_change)
        gm_menu.pack(side="right")

        # Emulator
        row2 = ctk.CTkFrame(game_f, fg_color="transparent")
        row2.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row2, text="Emulator", font=("Segoe UI", S(13)),
                     text_color=TXT).pack(side="left")
        cur_emu = self.general_config.get("current_emulator", "LDPlayer")
        self._emu_var = ctk.StringVar(value=cur_emu)
        ctk.CTkOptionMenu(
            row2, variable=self._emu_var, values=EMULATORS,
            font=("Segoe UI", S(13)), fg_color=SECTION,
            button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL, dropdown_hover_color=SEP,
            width=S(180), command=self._on_emulator_change
        ).pack(side="right")

        # Map orientation
        row3 = ctk.CTkFrame(game_f, fg_color="transparent")
        row3.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row3, text="Map Orientation", font=("Segoe UI", S(13)),
                     text_color=TXT).pack(side="left")
        orient = self.general_config.get("map_orientation", "vertical")
        self._orient_var = ctk.StringVar(value=orient.title())
        ctk.CTkOptionMenu(
            row3, variable=self._orient_var, values=["Vertical", "Horizontal"],
            font=("Segoe UI", S(13)), fg_color=SECTION,
            button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL, dropdown_hover_color=SEP,
            width=S(180), command=self._on_orient_change
        ).pack(side="right")

        # Timer (run for N minutes)
        row4 = ctk.CTkFrame(game_f, fg_color="transparent")
        row4.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row4, text="Run Timer (minutes, 0=\u221E)",
                     font=("Segoe UI", S(13)), text_color=TXT).pack(side="left")
        self._timer_var = tk.StringVar(
            value=str(self.general_config.get("run_for_minutes", 0)))
        ctk.CTkEntry(row4, textvariable=self._timer_var,
                     font=("Segoe UI", S(13)), width=S(80), height=S(32),
                     fg_color=SECTION, border_color=ACCENT, text_color=BRIGHT,
                     corner_radius=S(6), justify="center"
                     ).pack(side="right")

        # bOT SECTION
        self._section_header(page, "Bot Behavior")
        bot_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        bot_f.pack(fill="x", padx=S(20), pady=S(4))

        # Movement delay
        self._mvd_var = tk.DoubleVar(
            value=float(self.bot_config.get("minimum_movement_delay", 0.4)))
        self._slider_row(bot_f, "Movement Delay", self._mvd_var, 0.1, 2.0, "s")

        # Unstuck delay
        self._usd_var = tk.DoubleVar(
            value=float(self.bot_config.get("unstuck_movement_delay", 3.0)))
        self._slider_row(bot_f, "Unstuck Delay", self._usd_var, 1.0, 10.0, "s")

        # Unstuck hold time
        self._ush_var = tk.DoubleVar(
            value=float(self.bot_config.get("unstuck_movement_hold_time", 1.5)))
        self._slider_row(bot_f, "Unstuck Hold Time", self._ush_var, 0.5, 5.0, "s")

        # Use gadgets
        self._gadget_var = ctk.StringVar(
            value=self.bot_config.get("bot_uses_gadgets", "yes"))
        self._toggle_row(bot_f, "Use Gadgets", self._gadget_var)

        # Play again on win
        self._play_again_var = ctk.StringVar(
            value=self.bot_config.get("play_again_on_win", "no"))
        self._toggle_row(bot_f, "Play Again on Win", self._play_again_var)

        # Debug mode
        self._debug_var = ctk.StringVar(
            value=self.general_config.get("super_debug", "yes"))
        self._toggle_row(bot_f, "Debug Mode", self._debug_var)

        # dETECTION SECTION
        self._section_header(page, "Detection")
        det_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        det_f.pack(fill="x", padx=S(20), pady=S(4))

        # Entity confidence
        self._ec_var = tk.DoubleVar(
            value=float(self.bot_config.get("entity_detection_confidence", 0.6)))
        self._slider_row(det_f, "Entity Confidence", self._ec_var, 0.1, 1.0)

        # Wall confidence
        self._wc_var = tk.DoubleVar(
            value=float(self.bot_config.get("wall_detection_confidence", 0.9)))
        self._slider_row(det_f, "Wall Confidence", self._wc_var, 0.1, 1.0)

        # CPU / GPU
        row_cg = ctk.CTkFrame(det_f, fg_color="transparent")
        row_cg.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row_cg, text="Inference Device",
                     font=("Segoe UI", S(13)), text_color=TXT).pack(side="left")
        self._device_var = ctk.StringVar(
            value=self.general_config.get("cpu_or_gpu", "auto"))
        ctk.CTkOptionMenu(
            row_cg, variable=self._device_var,
            values=["auto", "cpu", "gpu"],
            font=("Segoe UI", S(13)), fg_color=SECTION,
            button_color=ACCENT, button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL, dropdown_hover_color=SEP,
            width=S(140)
        ).pack(side="right")

        # Max IPS
        row_ips = ctk.CTkFrame(det_f, fg_color="transparent")
        row_ips.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row_ips, text="Max IPS",
                     font=("Segoe UI", S(13)), text_color=TXT).pack(side="left")
        self._ips_var = tk.StringVar(
            value=str(self.general_config.get("max_ips", "auto")))
        ctk.CTkEntry(row_ips, textvariable=self._ips_var,
                     font=("Segoe UI", S(13)), width=S(80), height=S(32),
                     fg_color=SECTION, border_color=ACCENT, text_color=BRIGHT,
                     corner_radius=S(6), justify="center"
                     ).pack(side="right")

        # tIMERS SECTION
        self._section_header(page, "Timers")
        tim_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        tim_f.pack(fill="x", padx=S(20), pady=S(4))

        self._t_state = tk.DoubleVar(
            value=float(self.time_tresholds.get("state_check", 3)))
        self._slider_row(tim_f, "State Check", self._t_state, 0.5, 10, "s")

        self._t_super = tk.DoubleVar(
            value=float(self.time_tresholds.get("super", 0.1)))
        self._slider_row(tim_f, "Super Delay", self._t_super, 0.05, 5, "s")

        self._t_gadget = tk.DoubleVar(
            value=float(self.time_tresholds.get("gadget", 0.5)))
        self._slider_row(tim_f, "Gadget Delay", self._t_gadget, 0.1, 10, "s")

        self._t_hyper = tk.DoubleVar(
            value=float(self.time_tresholds.get("hypercharge", 2)))
        self._slider_row(tim_f, "Hypercharge Delay", self._t_hyper, 0.5, 10, "s")

        self._t_nodet = tk.DoubleVar(
            value=float(self.time_tresholds.get("no_detections", 10)))
        self._slider_row(tim_f, "No Detection", self._t_nodet, 1, 30, "s")

        self._t_idle = tk.DoubleVar(
            value=float(self.time_tresholds.get("idle", 10)))
        self._slider_row(tim_f, "Idle Check", self._t_idle, 1, 30, "s")

        # oVERLAY SECTION
        self._ovl_toggles = {}
        self._ovl_var = ctk.StringVar(
            value=self.general_config.get("visual_overlay_enabled", "no"))
        self._ovl_opacity = tk.DoubleVar(
            value=float(self.general_config.get("visual_overlay_opacity", 180)))
        if self._capabilities.get("visual_overlay"):
            self._section_header(page, "Visual Overlay")
            ovl_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
            ovl_f.pack(fill="x", padx=S(20), pady=S(4))

            self._toggle_row(ovl_f, "Overlay Enabled", self._ovl_var)
            self._slider_row(ovl_f, "Opacity", self._ovl_opacity, 50, 255)

            overlay_toggles = [
                ("Player Dot", "visual_overlay_player_dot"),
                ("Attack Range", "visual_overlay_attack_range"),
                ("Safe Range", "visual_overlay_safe_range"),
                ("Super Range", "visual_overlay_super_range"),
                ("Movement Arrow", "visual_overlay_movement_arrow"),
                ("LoS Lines", "visual_overlay_los_all_enemies"),
                ("Enemies", "visual_overlay_enemies"),
                ("Teammates", "visual_overlay_teammates"),
                ("Walls", "visual_overlay_walls"),
                ("HP Bars", "visual_overlay_hp_bars"),
                ("HUD Panel", "visual_overlay_brawler_hud"),
                ("Gas Zone", "visual_overlay_gas_zone"),
                ("Danger Zones", "visual_overlay_danger_zones"),
                ("Decision Banner", "visual_overlay_decision_banner"),
                ("Target Info", "visual_overlay_target_info"),
                ("Ghost Dots", "visual_overlay_ghost_dots"),
                ("Hide When Dead", "visual_overlay_hide_when_dead"),
            ]
            grid_f = ctk.CTkFrame(ovl_f, fg_color="transparent")
            grid_f.pack(fill="x", padx=S(14), pady=S(4))
            grid_f.grid_columnconfigure((0, 1), weight=1)
            for i, (label, key) in enumerate(overlay_toggles):
                val = self.general_config.get(key, "yes")
                var = ctk.StringVar(value=val)
                self._ovl_toggles[key] = var
                r, c = divmod(i, 2)
                f = ctk.CTkFrame(grid_f, fg_color="transparent")
                f.grid(row=r, column=c, sticky="w", padx=S(4), pady=S(2))
                cb = ctk.CTkCheckBox(
                    f, text=label, font=("Segoe UI", S(12)),
                    variable=var, onvalue="yes", offvalue="no",
                    fg_color=ACCENT, hover_color=ACCENT_H,
                    text_color=TXT, checkbox_height=S(20), checkbox_width=S(20))
                cb.pack(anchor="w")

        # sAVE BUTTON
        ctk.CTkButton(page, text="\U0001F4BE  Save Settings",
                      font=("Segoe UI", S(15), "bold"),
                      fg_color=GREEN, hover_color="#00A050",
                      text_color=BG, height=S(44), corner_radius=S(10),
                      width=S(200), command=self._save_all_settings
                      ).pack(pady=(S(16), S(20)))

    def _section_header(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=("Segoe UI", S(16), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(24),
                                              pady=(S(12), S(2)))

    def _slider_row(self, parent, label, var, lo, hi, suffix=""):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=S(14), pady=S(6))
        ctk.CTkLabel(row, text=label, font=("Segoe UI", S(13)),
                     text_color=TXT).pack(side="left")
        val_lbl = ctk.CTkLabel(row, text=f"{var.get():.2f}{suffix}",
                               font=("Segoe UI", S(12), "bold"),
                               text_color=ACCENT, width=S(70))
        val_lbl.pack(side="right")
        slider = ctk.CTkSlider(
            row, from_=lo, to=hi, variable=var,
            width=S(200), height=S(18),
            fg_color=SECTION, progress_color=ACCENT,
            button_color=BRIGHT, button_hover_color=GOLD,
            command=lambda v, l=val_lbl, s=suffix: l.configure(
                text=f"{v:.2f}{s}")
        )
        slider.pack(side="right", padx=S(8))

    def _toggle_row(self, parent, label, var):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=S(14), pady=S(6))
        ctk.CTkLabel(row, text=label, font=("Segoe UI", S(13)),
                     text_color=TXT).pack(side="left")
        ctk.CTkSwitch(row, text="", variable=var,
                      onvalue="yes", offvalue="no",
                      fg_color=DIM, progress_color=GREEN,
                      button_color=BRIGHT,
                      button_hover_color=GOLD,
                      width=S(46), height=S(24)
                      ).pack(side="right")

    def _entry_row(self, parent, label, var, placeholder=""):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=S(14), pady=S(6))
        ctk.CTkLabel(row, text=label, font=("Segoe UI", S(13)),
                     text_color=TXT, width=S(180), anchor="w").pack(side="left")
        ctk.CTkEntry(
            row,
            textvariable=var,
            placeholder_text=placeholder,
            font=("Segoe UI", S(13)),
            height=S(34),
            fg_color=SECTION,
            border_color=SEP,
            text_color=BRIGHT,
            corner_radius=S(8),
        ).pack(side="right", fill="x", expand=True)

    @staticmethod
    def _mousewheel_units(event):
        delta = getattr(event, "delta", 0)
        if delta:
            step = int(-delta / 120)
            return step if step != 0 else (-1 if delta > 0 else 1)
        num = getattr(event, "num", None)
        if num == 4:
            return -1
        if num == 5:
            return 1
        return 0

    def _bind_scroll_target(self, widget, handler, seen=None):
        if seen is None:
            seen = set()
        widget_id = str(widget)
        if widget_id in seen:
            return
        seen.add(widget_id)
        try:
            widget.bind("<MouseWheel>", handler, add="+")
            widget.bind("<Button-4>", handler, add="+")
            widget.bind("<Button-5>", handler, add="+")
        except Exception:
            return
        for child in widget.winfo_children():
            self._bind_scroll_target(child, handler, seen)

    def _install_nested_scroll(self, scrollable):
        canvas = getattr(scrollable, "_parent_canvas", None)
        if canvas is None:
            return
        try:
            canvas.configure(yscrollincrement=max(1, S(18)))
        except Exception:
            pass

        def _on_wheel(event):
            units = self._mousewheel_units(event)
            if units == 0:
                return None
            canvas.yview_scroll(units, "units")
            return "break"

        seen = getattr(scrollable, "_nested_scroll_seen", set())
        self._bind_scroll_target(scrollable, _on_wheel, seen)
        scrollable._nested_scroll_seen = seen

    # settings callbacks
    def _on_gamemode_change(self, display_name):
        for key, (name, gtype) in GAMEMODES.items():
            if name == display_name:
                self.bot_config["gamemode"] = key
                self.bot_config["gamemode_type"] = gtype
                break
        if hasattr(self, "_home_gm_lbl"):
            self._home_gm_lbl.configure(text=display_name)
        if hasattr(self, "_gm_var") and self._gm_var.get() != display_name:
            self._gm_var.set(display_name)
        if hasattr(self, "_cc_gm_var"):
            primary_value = display_name if display_name in getattr(self, "_cc_gm_primary_values", [])[:-1] else "Other"
            if self._cc_gm_var.get() != primary_value:
                self._cc_gm_var.set(primary_value)
        if hasattr(self, "_cc_other_gm_var") and self._cc_other_gm_var.get() != display_name:
            self._cc_other_gm_var.set(display_name)

    def _on_emulator_change(self, emu):
        self.general_config["current_emulator"] = emu
        if hasattr(self, "_home_emu_lbl"):
            self._home_emu_lbl.configure(text=emu)
        if hasattr(self, "_emu_var") and self._emu_var.get() != emu:
            self._emu_var.set(emu)
        if hasattr(self, "_cc_emu_var") and self._cc_emu_var.get() != emu:
            self._cc_emu_var.set(emu)

    def _on_orient_change(self, val):
        self.general_config["map_orientation"] = val.lower()
        if hasattr(self, "_orient_var") and self._orient_var.get() != val:
            self._orient_var.set(val)
        if hasattr(self, "_cc_orient_var") and self._cc_orient_var.get() != val:
            self._cc_orient_var.set(val)

    def _on_control_center_gamemode_change(self, display_name):
        if display_name == "Other":
            display_name = self._cc_other_gm_var.get()
        self._on_gamemode_change(display_name)

    def _on_control_center_emulator_change(self, emu):
        self._on_emulator_change(emu)

    def _on_control_center_orientation_change(self, value):
        self._on_orient_change(value)

    def _save_all_settings(self):
        """Persist all settings to config files."""
        # Bot config
        bc = self.bot_config
        bc["minimum_movement_delay"] = round(self._mvd_var.get(), 2)
        bc["unstuck_movement_delay"] = round(self._usd_var.get(), 2)
        bc["unstuck_movement_hold_time"] = round(self._ush_var.get(), 2)
        bc["wall_detection_confidence"] = round(self._wc_var.get(), 2)
        bc["entity_detection_confidence"] = round(self._ec_var.get(), 2)
        bc["bot_uses_gadgets"] = self._gadget_var.get()
        bc["play_again_on_win"] = self._play_again_var.get()
        save_dict_as_toml(bc, "cfg/bot_config.toml")

        # General config
        gc = self.general_config
        gc["super_debug"] = self._debug_var.get()
        gc["cpu_or_gpu"] = self._device_var.get()
        gc["max_ips"] = self._ips_var.get()
        gc["discord_id"] = self._discord_id_var.get().strip()
        gc["brawlstars_api_key"] = self._bs_api_key_var.get().strip()
        tag_value = self._bs_player_tag_var.get().strip().upper().replace(" ", "")
        if tag_value and not tag_value.startswith("#"):
            tag_value = f"#{tag_value}"
        self._bs_player_tag_var.set(tag_value)
        gc["brawlstars_player_tag"] = tag_value
        gc["visual_overlay_enabled"] = self._ovl_var.get()
        gc["visual_overlay_opacity"] = int(self._ovl_opacity.get())
        try:
            timer_value = int(self._cc_timer_var.get() if hasattr(self, "_cc_timer_var") else self._timer_var.get())
            gc["run_for_minutes"] = timer_value
            if hasattr(self, "_timer_var") and self._timer_var.get() != str(timer_value):
                self._timer_var.set(str(timer_value))
        except ValueError:
            gc["run_for_minutes"] = 0
            if hasattr(self, "_timer_var"):
                self._timer_var.set("0")
        for key, var in self._ovl_toggles.items():
            gc[key] = var.get()
        save_dict_as_toml(gc, "cfg/general_config.toml")
        self.login_config["key"] = self._pyla_key_var.get().strip()
        save_dict_as_toml(self.login_config, "cfg/login.toml")

        # Time thresholds
        tt = self.time_tresholds
        tt["state_check"] = round(self._t_state.get(), 2)
        tt["super"] = round(self._t_super.get(), 2)
        tt["gadget"] = round(self._t_gadget.get(), 2)
        tt["hypercharge"] = round(self._t_hyper.get(), 2)
        tt["no_detections"] = round(self._t_nodet.get(), 2)
        tt["idle"] = round(self._t_idle.get(), 2)
        save_dict_as_toml(tt, "cfg/time_tresholds.toml")

        # Update home cards
        try:
            t = int(self._timer_var.get())
            self._home_timer_lbl.configure(
                text=f"{t} min" if t > 0 else "\u221E")
        except ValueError:
            pass

        # Flash save button
        self._flash_save_confirm()

    def _flash_save_confirm(self):
        """Brief visual confirmation that settings were saved."""
        # Find save button and briefly change it
        for page_frame in self._pages.values():
            for widget in page_frame.winfo_children():
                pass  # The button is inside the settings page
        # Simple approach: change status bar
        self._status_label.configure(text="\u2714 Settings Saved", text_color=GREEN)
        self.after(2000, lambda: self._update_status_display())

    # --- bRAWLER PAGE ---

    def _build_brawler_page(self):
        page = ctk.CTkFrame(self._content, fg_color=BG, corner_radius=0)
        self._pages["brawler"] = page

        # Header with search + actions (2 rows to avoid overflow)
        hdr = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(12),
                   height=S(112))
        hdr.pack(fill="x", padx=S(20), pady=(S(16), S(8)))
        hdr.pack_propagate(False)

        top_row = ctk.CTkFrame(hdr, fg_color="transparent")
        top_row.pack(fill="x", padx=S(10), pady=(S(8), S(4)))

        action_row = ctk.CTkFrame(hdr, fg_color="transparent")
        action_row.pack(fill="x", padx=S(10), pady=(0, S(8)))

        ctk.CTkLabel(top_row, text="Select Brawler",
                     font=("Segoe UI", S(22), "bold"),
                 text_color=BRIGHT).pack(side="left", padx=S(6))

        self._brawler_filter = tk.StringVar()
        ctk.CTkEntry(top_row, textvariable=self._brawler_filter,
                     placeholder_text="\U0001F50D  Search...",
                     font=("Segoe UI", S(14)), width=S(200), height=S(36),
                     fg_color=BG, border_color=ACCENT, text_color=BRIGHT,
                 corner_radius=S(6)).pack(side="left", padx=S(12))
        self._brawler_filter.trace_add("write", lambda *a: self._schedule_brawler_grid_refresh())

        self._role_filter_var = tk.StringVar(value="All Roles")
        role_values = sorted({
            (self.brawlers_info.get(name, {}) or {}).get("playstyle", "")
            for name in self.all_brawlers
        } - {""})
        self._role_filter_options = ["All Roles"] + [v.title() for v in role_values]
        ctk.CTkOptionMenu(
            top_row,
            variable=self._role_filter_var,
            values=self._role_filter_options,
            width=S(130),
            height=S(34),
            font=("Segoe UI", S(12)),
            fg_color=SEP,
            button_color=ACCENT,
            button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL,
            command=lambda _v: self._refresh_brawler_grid(),
        ).pack(side="left", padx=S(6))

        self._selection_mode_var = tk.StringVar(value="Multi")
        ctk.CTkSegmentedButton(
            top_row,
            values=["Multi", "Single"],
            variable=self._selection_mode_var,
            width=S(150),
            height=S(32),
            fg_color=SECTION,
            selected_color=ACCENT,
            selected_hover_color=ACCENT_H,
            unselected_color=SEP,
            unselected_hover_color=SECTION,
            command=lambda _v: self._refresh_brawler_grid(),
            font=("Segoe UI", S(12), "bold"),
        ).pack(side="left", padx=S(8))

        self._sort_mode_var = tk.StringVar(value="Unlocked")
        ctk.CTkOptionMenu(
            top_row,
            variable=self._sort_mode_var,
            values=[
                "Unlocked",
                "Name A-Z",
                "Trophies Low->High",
                "Trophies High->Low",
                "Winrate High->Low",
            ],
            width=S(165),
            height=S(34),
            font=("Segoe UI", S(11)),
            fg_color=SEP,
            button_color=ACCENT,
            button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL,
            command=lambda _v: self._refresh_brawler_grid(),
        ).pack(side="left", padx=S(6))

        self._sel_count_label = ctk.CTkLabel(
            top_row, text="0 selected", font=("Segoe UI", S(13)),
            text_color=DIM)
        self._sel_count_label.pack(side="right", padx=S(8))

        # Scan Brawlers button (explicitly in action row so it always stays visible)
        self._scan_btn = ctk.CTkButton(
            action_row, text="\U0001F50D Scan Brawlers", width=S(150), height=S(32),
            font=("Segoe UI", S(12), "bold"), fg_color=BLUE,
            hover_color=CYAN, text_color=BG, corner_radius=S(6),
            command=self._start_brawler_scan)
        if self._capabilities.get("brawler_scan"):
            self._scan_btn.pack(side="left", padx=S(4))

        # Scan status label (shows progress during scan)
        self._scan_status_label = ctk.CTkLabel(
            action_row, text="", font=("Segoe UI", S(11)),
            text_color=GOLD)
        if self._capabilities.get("brawler_scan"):
            self._scan_status_label.pack(side="left", padx=S(6))

        ctk.CTkButton(action_row, text="Select Visible", width=S(120), height=S(32),
                      font=("Segoe UI", S(12), "bold"), fg_color=ACCENT,
                      hover_color=ACCENT_H, text_color=BRIGHT,
                      corner_radius=S(6),
                      command=self._select_visible_brawlers
                      ).pack(side="left", padx=S(6))

        ctk.CTkButton(action_row, text="Clear Selection", width=S(120), height=S(32),
                      font=("Segoe UI", S(12)), fg_color=SEP,
                      hover_color=SECTION, text_color=TXT,
                      corner_radius=S(6),
                      command=self._clear_brawler_selection
                      ).pack(side="left", padx=S(4))

        # Hide locked toggle
        self._hide_locked_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            action_row, text="Hide Locked", variable=self._hide_locked_var,
            font=("Segoe UI", S(11)), text_color=TXT,
            fg_color=ACCENT, hover_color=ACCENT_H,
            border_color=SEP, width=S(24), height=S(24),
            command=self._refresh_brawler_grid
        ).pack(side="right", padx=S(6))

        ctk.CTkButton(action_row, text="Load Config", width=S(110), height=S(32),
                      font=("Segoe UI", S(12)), fg_color=SEP,
                      hover_color=SECTION, text_color=TXT,
                      corner_radius=S(6),
                      command=self._load_brawler_config
                      ).pack(side="right", padx=S(6))

        # Scrollable brawler grid
        self._brawler_scroll = ctk.CTkScrollableFrame(
            page, fg_color=BG, corner_radius=S(8),
            scrollbar_button_color=ACCENT,
            scrollbar_button_hover_color=ACCENT_H)
        self._brawler_scroll.pack(fill="both", expand=True,
                                  padx=S(16), pady=S(4))
        self._install_nested_scroll(self._brawler_scroll)

        # Load brawler images
        icon_sz = S(52)
        self._brawler_images = []
        for b in self.all_brawlers:
            path = f"./api/assets/brawler_icons/{b}.png"
            try:
                img = Image.open(path)
            except FileNotFoundError:
                try:
                    save_brawler_icon(b)
                    img = Image.open(path)
                except Exception:
                    img = Image.new("RGBA", (64, 64), (30, 30, 50, 255))
            self._brawler_images.append(
                (b, CTkImage(img, size=(icon_sz, icon_sz))))

        self._build_sorted_brawler_cache()
        self._refresh_brawler_grid()

    def _refresh_brawler_grid(self):
        self._brawler_grid_refresh_after_id = None
        if self._brawler_render_job_id is not None:
            try:
                self.after_cancel(self._brawler_render_job_id)
            except Exception:
                pass
            self._brawler_render_job_id = None

        if not hasattr(self, "_brawler_scroll"):
            return

        for w in self._brawler_scroll.winfo_children():
            w.destroy()

        cols = 6
        has_scan = bool(self._brawler_scan_data)
        self._brawler_render_rows = self._collect_visible_brawlers()
        self._brawler_render_cursor = 0

        for c in range(cols):
            self._brawler_scroll.grid_columnconfigure(c, weight=1)

        self._render_brawler_grid_chunk(cols=cols, has_scan=has_scan)

        n = len(self.brawlers_data)
        mode_txt = "Single" if getattr(self, "_selection_mode_var", None) and self._selection_mode_var.get() == "Single" else "Multi"
        if hasattr(self, "_sel_count_label"):
            self._sel_count_label.configure(
                text=(f"{n} selected" if n != 1 else "1 selected") + f"  •  {mode_txt}",
                text_color=ACCENT if n > 0 else DIM)

    def _render_brawler_grid_chunk(self, cols=6, has_scan=False):
        chunk_size = 8
        total = len(self._brawler_render_rows)
        start = self._brawler_render_cursor
        end = min(total, start + chunk_size)

        for idx in range(start, end):
            row_data = self._brawler_render_rows[idx]
            row = idx // cols
            col = idx % cols
            self._create_brawler_card(row_data, row=row, col=col, has_scan=has_scan)

        self._brawler_render_cursor = end
        if end < total:
            self._brawler_render_job_id = self.after(
                8, lambda: self._render_brawler_grid_chunk(cols=cols, has_scan=has_scan)
            )
        else:
            self._brawler_render_job_id = None

    def _create_brawler_card(self, row_data, row, col, has_scan=False):
        b = row_data["name"]
        img_tk = row_data["img"]
        b_unlocked = row_data["unlocked"]
        selected_entry = row_data["selected"]
        is_sel = selected_entry is not None
        ps = row_data["playstyle"]
        total = row_data["total"]
        wr = row_data["wr"]
        trophy_value = row_data["trophies"]
        trophy_source = row_data["trophy_source"]

        if has_scan and not b_unlocked:
            border = "#1a1a2e"
            card_color = "#08080e"
            name_alpha_color = "#3a3a3a"
            locked = True
        else:
            border = ACCENT if is_sel else SEP
            card_color = PANEL
            name_alpha_color = None
            locked = False

        card = ctk.CTkFrame(self._brawler_scroll, fg_color=card_color,
                            corner_radius=S(10), border_width=S(2),
                            border_color=border,
                            width=S(140), height=S(175))
        card.grid(row=row, column=col, padx=S(5), pady=S(5), sticky="nsew")
        card.grid_propagate(False)
        if not locked:
            card.bind("<Button-1>", lambda e, bx=b: self._open_brawler_config(bx))

        icon_l = ctk.CTkLabel(card, image=img_tk, text="", fg_color="transparent")
        icon_l.pack(pady=(S(8), S(2)))
        if not locked:
            icon_l.bind("<Button-1>", lambda e, bx=b: self._open_brawler_config(bx))

        if locked:
            lock_overlay = ctk.CTkLabel(
                card, text="\U0001F512", font=("Segoe UI", S(24)),
                fg_color="#08080e", text_color="#555555",
                width=S(50), height=S(28), corner_radius=S(6))
            lock_overlay.place(relx=0.5, rely=0.22, anchor="center")

        name_c = ACCENT if is_sel else (name_alpha_color if locked else BRIGHT)
        disp = b.title() if len(b) <= 10 else b[:9].title() + "."
        ctk.CTkLabel(card, text=disp,
                     font=("Segoe UI", S(11), "bold"),
                     text_color=name_c).pack(pady=(0, S(1)))

        if ps:
            ctk.CTkLabel(card, text=ps.upper(),
                         font=("Segoe UI", S(8)),
                         text_color=PS_COLORS.get(ps, DIM)).pack()

        if total > 0 and not locked:
            wr_c = GREEN if wr >= 50 else RED
            ctk.CTkLabel(card, text=f"{wr}% ({total})",
                         font=("Segoe UI", S(9)),
                         text_color=wr_c).pack(pady=(S(1), 0))
        elif locked:
            ctk.CTkLabel(card, text="Not unlocked",
                         font=("Segoe UI", S(9)),
                         text_color="#333333").pack(pady=(S(1), 0))
        else:
            ctk.CTkLabel(card, text="No games",
                         font=("Segoe UI", S(9)),
                         text_color=DIM).pack(pady=(S(1), 0))

        if not locked and isinstance(trophy_value, int) and trophy_value >= 0:
            t_color = GREEN if trophy_value >= 500 else GOLD if trophy_value >= 100 else TXT
            trophy_label = f"\U0001F3C6 {trophy_value}"
            if trophy_source == "manual":
                trophy_label += "  M"
            ctk.CTkLabel(card, text=trophy_label,
                         font=("Segoe UI", S(9), "bold"),
                         text_color=t_color).pack(pady=(0, S(2)))

    def _start_brawler_scan(self):
        """Start scanning all brawlers in-game to detect trophies and unlock status.
        Works both with a running bot (reuses connection) and standalone
        (creates its own WindowController + LobbyAutomation)."""
        if not self._capabilities.get("brawler_scan"):
            self._status_label.configure(
                text="\u26A0 Brawler scan is not available on this branch.", text_color=GOLD)
            return
        if self._scan_in_progress:
            self._status_label.configure(
                text="\u26A0 Scan already in progress!", text_color=GOLD)
            return

        self._scan_in_progress = True
        self._scan_btn.configure(state="disabled", text="\u23F3 Scanning...")
        self._scan_status_label.configure(text="Connecting to emulator...", text_color=GOLD)

        def run_scan():
            temp_wc = None  # Track if we created our own WindowController
            try:
                wc = None
                lobby_auto = None

                # 1) Try to reuse the running bot's connection
                if self.bot_running:
                    import sys
                    main_module = sys.modules.get('__main__')
                    if main_module:
                        sm = getattr(main_module, '_active_stage_manager', None)
                        if sm:
                            wc = sm.window_controller
                            lobby_auto = sm.Lobby_automation

                # 2) No running bot - create a temporary connection
                if wc is None:
                    self.after(0, lambda: self._scan_status_label.configure(
                        text="Connecting to emulator...", text_color=GOLD))
                    try:
                        from window_controller import WindowController
                        from lobby_automation import LobbyAutomation
                        temp_wc = WindowController()
                        wc = temp_wc
                        lobby_auto = LobbyAutomation(wc)
                        # Take a screenshot to initialise width_ratio / height_ratio
                        wc.screenshot()
                    except Exception as conn_err:
                        self.after(0, lambda e=str(conn_err): (
                            self._scan_status_label.configure(
                                text="Error: Could not connect to emulator",
                                text_color=RED),
                            self._status_label.configure(
                                text=f"\u26A0 {e[:50]}", text_color=RED)
                        ))
                        return

                self.after(0, lambda: self._scan_status_label.configure(
                    text="Starting scan...", text_color=GOLD))

                def progress_cb(scanned, total, brawler_name):
                    self.after(0, lambda s=scanned, t=total, b=brawler_name:
                        self._scan_status_label.configure(
                            text=f"Scanning {s}/{t}: {b.title()}", text_color=GOLD))

                results = lobby_auto.scan_all_brawlers(
                    self.all_brawlers, progress_callback=progress_cb)

                # Save results
                self._save_scan_data(results)

                unlocked = sum(1 for v in results.values() if v.get("unlocked"))
                self.after(0, lambda: (
                    self._scan_status_label.configure(
                        text=f"Done! {unlocked} unlocked", text_color=GREEN),
                    self._refresh_brawler_grid(),
                    self._status_label.configure(
                        text=f"\u2714 Scan complete: {unlocked} brawlers found",
                        text_color=GREEN)
                ))

            except Exception as e:
                print(f"[SCAN] Error during scan: {e}")
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._scan_status_label.configure(
                    text=f"Error: {str(e)[:40]}", text_color=RED))
            finally:
                # Clean up temporary connection if we created one
                if temp_wc is not None:
                    try:
                        temp_wc.close()
                    except Exception:
                        pass
                self._scan_in_progress = False
                self.after(0, lambda: self._scan_btn.configure(
                    state="normal", text="\U0001F50D Scan Brawlers"))

        threading.Thread(target=run_scan, daemon=True).start()

    def _open_brawler_config(self, brawler):
        """Open config dialog for a brawler."""
        top = ctk.CTkToplevel(self)
        top.configure(fg_color=BG)
        top.geometry(f"{S(420)}x{S(720)}")
        top.title(f"\u2605 {brawler.title()}")
        top.transient(self)
        top.attributes("-topmost", True)

        info = self.brawlers_info.get(brawler, {})
        ps = info.get("playstyle", "")
        hist = self.match_history.get(brawler, {})
        v = hist.get("victory", 0) if isinstance(hist, dict) else 0
        d_ = hist.get("defeat", 0) if isinstance(hist, dict) else 0
        total = v + d_
        existing = self._get_selected_brawler_entry(brawler)
        scan_entry = self._brawler_scan_data.get(brawler, {})
        effective_trophies, trophy_source = self._get_effective_trophies(
            brawler, selected_entry=existing, scan_entry=scan_entry
        )
        scan_trophies = scan_entry.get("trophies") if isinstance(scan_entry, dict) else None
        scan_unlocked = scan_entry.get("unlocked") if isinstance(scan_entry, dict) else None
        manual_locked = bool(existing and existing.get("manual_trophies", False))

        body = ctk.CTkScrollableFrame(top, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=S(6), pady=S(6))

        # Header
        hdr = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=S(8))
        hdr.pack(fill="x", padx=S(10), pady=(S(10), S(6)))
        ctk.CTkLabel(hdr, text=brawler.title(),
                     font=("Segoe UI", S(20), "bold"),
                     text_color=ACCENT).pack(pady=(S(8), S(2)))
        if ps:
            ctk.CTkLabel(hdr, text=ps.upper(),
                         font=("Segoe UI", S(10)),
                         text_color=PS_COLORS.get(ps, DIM)).pack()
        if total > 0:
            wr = round(100 * v / total)
            ctk.CTkLabel(hdr, text=f"{v}W / {d_}L  \u2022  {wr}% WR",
                         font=("Segoe UI", S(12)),
                         text_color=GREEN if wr >= 50 else RED
                         ).pack(pady=(S(2), S(8)))
        else:
            wr = 0
            ctk.CTkLabel(hdr, text="No match data",
                         font=("Segoe UI", S(11)),
                         text_color=DIM).pack(pady=(S(2), S(8)))

        detail = ctk.CTkFrame(body, fg_color=SECTION, corner_radius=S(8))
        detail.pack(fill="x", padx=S(10), pady=(S(2), S(4)))
        source_text = {
            "manual": "manual override",
            "scan": "scan",
            "session": "session",
            "default": "default",
        }.get(trophy_source, trophy_source)
        unlock_text = "Unknown"
        if scan_unlocked is True:
            unlock_text = "Unlocked"
        elif scan_unlocked is False:
            unlock_text = "Locked"
        summary_text = f"WR {wr}%  \u2022  {total} games  \u2022  Source: {source_text}"
        ctk.CTkLabel(detail,
                     text=summary_text,
                     font=("Segoe UI", S(11), "bold"),
                     text_color=TXT).pack(anchor="w", padx=S(8), pady=(S(6), S(1)))
        ctk.CTkLabel(detail,
                     text=f"Status: {unlock_text}",
                     font=("Segoe UI", S(11), "bold"),
                     text_color=TXT).pack(anchor="w", padx=S(8), pady=(S(2), S(1)))
        ctk.CTkLabel(detail,
                     text=f"Trophies: {effective_trophies} ({source_text})",
                     font=("Segoe UI", S(11)),
                     text_color=GOLD if manual_locked else TXT).pack(anchor="w", padx=S(8), pady=S(1))
        ctk.CTkLabel(detail,
                     text=f"Match stats: {v} wins / {d_} losses / {total} games",
                     font=("Segoe UI", S(11)),
                     text_color=TXT).pack(anchor="w", padx=S(8), pady=(S(1), S(6)))

        # Farm type
        ft_frame = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=S(8),
                                height=S(46))
        ft_frame.pack(fill="x", padx=S(10), pady=S(4))
        ft_frame.pack_propagate(False)

        local_farm = {"type": ""}

        def set_ft(t, wb, tb):
            local_farm["type"] = t
            wb.configure(fg_color=ACCENT if t == "wins" else SEP)
            tb.configure(fg_color=ACCENT if t == "trophies" else SEP)

        wb = ctk.CTkButton(ft_frame, text="Win Amount", width=S(130),
                           fg_color=SEP, hover_color=ACCENT,
                           font=("Segoe UI", S(13)),
                           border_color=ACCENT, border_width=S(2))
        wb.pack(side="left", padx=(S(10), S(4)), pady=S(5))
        tb = ctk.CTkButton(ft_frame, text="Trophies", width=S(130),
                           fg_color=SEP, hover_color=ACCENT,
                           font=("Segoe UI", S(13)),
                           border_color=ACCENT, border_width=S(2))
        tb.pack(side="left", padx=S(4), pady=S(5))
        wb.configure(command=lambda: set_ft("wins", wb, tb))
        tb.configure(command=lambda: set_ft("trophies", wb, tb))

        # Fields
        fields = ctk.CTkFrame(body, fg_color="transparent")
        fields.pack(fill="x", padx=S(10), pady=S(4))

        def field(parent, lbl, placeholder="", default=""):
            ctk.CTkLabel(parent, text=lbl, font=("Segoe UI", S(13)),
                         text_color=TXT).pack(anchor="w", padx=S(4),
                                               pady=(S(4), 0))
            var = tk.StringVar(value=default)
            ctk.CTkEntry(parent, textvariable=var,
                         fg_color=PANEL, text_color=BRIGHT,
                         placeholder_text=placeholder,
                         border_color=ACCENT, border_width=S(2),
                         height=S(30), corner_radius=S(6),
                         font=("Segoe UI", S(13))
                         ).pack(fill="x", padx=S(4), pady=(0, S(2)))
            return var

        push_default = ""
        wins_default = ""
        streak_default = "0"
        auto_default = True
        farm_default = ""
        if existing:
            pu = existing.get("push_until", "")
            push_default = str(pu) if pu != "" else ""
            wins_value = existing.get("wins", "")
            wins_default = str(wins_value) if wins_value != "" else ""
            streak_default = str(existing.get("win_streak", 0))
            auto_default = existing.get("automatically_pick", True)
            farm_default = existing.get("type", "")

        push_var = field(fields, "Target Amount", "e.g. 750", push_default)
        troph_var = field(fields, "Brawler Trophies", "e.g. 500", str(effective_trophies))
        wins_var = field(fields, "Current Wins", "e.g. 10", wins_default)
        streak_var = field(fields, "Win Streak", "", streak_default)

        quick_row = ctk.CTkFrame(fields, fg_color="transparent")
        quick_row.pack(fill="x", padx=S(4), pady=(S(2), S(6)))

        def adjust_trophies(delta):
            current = troph_var.get().strip()
            current_value = int(current) if current.isdigit() else effective_trophies
            troph_var.set(str(max(0, current_value + delta)))

        ctk.CTkButton(quick_row, text="-10", width=S(54), height=S(26),
                      fg_color=SEP, hover_color=SECTION, text_color=TXT,
                      command=lambda: adjust_trophies(-10)).pack(side="left", padx=(0, S(4)))
        ctk.CTkButton(quick_row, text="-1", width=S(46), height=S(26),
                      fg_color=SEP, hover_color=SECTION, text_color=TXT,
                      command=lambda: adjust_trophies(-1)).pack(side="left", padx=(0, S(4)))
        ctk.CTkButton(quick_row, text="+1", width=S(46), height=S(26),
                      fg_color=SEP, hover_color=SECTION, text_color=TXT,
                      command=lambda: adjust_trophies(1)).pack(side="left", padx=(0, S(4)))
        ctk.CTkButton(quick_row, text="+10", width=S(54), height=S(26),
                      fg_color=SEP, hover_color=SECTION, text_color=TXT,
                      command=lambda: adjust_trophies(10)).pack(side="left", padx=(0, S(4)))
        ctk.CTkButton(quick_row, text="Target = Trophy +50", height=S(26),
                      fg_color=SEP, hover_color=SECTION, text_color=TXT,
                      command=lambda: push_var.set(
                          str((int(troph_var.get()) if troph_var.get().isdigit() else effective_trophies) + 50)
                      )).pack(side="right")

        if isinstance(scan_trophies, int):
            ctk.CTkButton(fields,
                          text=f"Use scanned trophies ({scan_trophies})",
                          fg_color=SEP,
                          hover_color=SECTION,
                          text_color=TXT,
                          height=S(28),
                          command=lambda: troph_var.set(str(scan_trophies))
                          ).pack(fill="x", padx=S(4), pady=(S(2), S(4)))

        auto_var = tk.BooleanVar(value=auto_default)
        ctk.CTkCheckBox(fields, text="Auto-select brawler",
                        variable=auto_var, fg_color=ACCENT,
                        hover_color=ACCENT_H, text_color=TXT,
                        checkbox_height=S(22),
                        font=("Segoe UI", S(12))
                        ).pack(anchor="w", padx=S(4), pady=S(6))

        manual_trophy_var = tk.BooleanVar(value=manual_locked or existing is not None)
        ctk.CTkCheckBox(fields, text="Manual trophies (lock until rescan)",
                        variable=manual_trophy_var, fg_color=ACCENT,
                        hover_color=ACCENT_H, text_color=TXT,
                        checkbox_height=S(22),
                        font=("Segoe UI", S(12))
                        ).pack(anchor="w", padx=S(4), pady=(0, S(6)))
        ctk.CTkLabel(fields,
                 text="When enabled, OCR updates will not overwrite this trophy value until next scan.",
                 font=("Segoe UI", S(10)),
                 text_color=DIM,
                 wraplength=S(350),
                 justify="left").pack(anchor="w", padx=S(4), pady=(0, S(6)))

        if farm_default in ("wins", "trophies"):
            set_ft(farm_default, wb, tb)
        elif existing is None:
            set_ft("trophies", wb, tb)

        def submit():
            pu = push_var.get().strip()
            pu = int(pu) if pu.isdigit() else ""
            tr = troph_var.get().strip()
            tr = int(tr) if tr.isdigit() else 0
            wv = wins_var.get().strip()
            wv = int(wv) if wv.isdigit() else ""
            sv = streak_var.get().strip()
            ft = local_farm["type"]
            if ft == "trophies" and wv == "":
                wv = 0
            data = {
                "brawler": brawler,
                "push_until": pu,
                "trophies": tr,
                "wins": wv,
                "type": ft if ft else ("trophies" if tr <= (wv if isinstance(wv, int) else 0) else "wins"),
                "automatically_pick": auto_var.get(),
                "win_streak": int(sv) if str(sv).isdigit() else 0,
                "manual_trophies": manual_trophy_var.get(),
            }
            current_mode = self._selection_mode_var.get() if hasattr(self, "_selection_mode_var") else "Multi"
            if current_mode == "Single":
                self.brawlers_data = [data]
            else:
                self.brawlers_data = [
                    x for x in self.brawlers_data if x["brawler"] != brawler]
                self.brawlers_data.append(data)
            top.destroy()
            self._refresh_brawler_grid()
            self._update_sidebar_brawler()

        def remove_selected():
            self.brawlers_data = [x for x in self.brawlers_data if x.get("brawler") != brawler]
            top.destroy()
            self._refresh_brawler_grid()
            self._update_sidebar_brawler()

        if existing is not None:
            ctk.CTkButton(body, text="Remove from Selection", command=remove_selected,
                          fg_color=RED, hover_color="#cc3333",
                          text_color=BRIGHT, width=S(180), height=S(34),
                          corner_radius=S(8),
                          font=("Segoe UI", S(13), "bold")).pack(pady=(S(4), S(6)))

        ctk.CTkButton(body, text="\u2714  Confirm", command=submit,
                      fg_color=ACCENT, hover_color=ACCENT_H,
                      text_color=BRIGHT, width=S(160), height=S(38),
                      corner_radius=S(8),
                      font=("Segoe UI", S(14), "bold")).pack(pady=S(10))
        top.bind("<Return>", lambda _event: submit())

    def _load_brawler_config(self):
        path = filedialog.askopenfilename(
            title="Select Brawler Config",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                data = [d for d in data
                        if not (d.get("push_until", 0) <= d.get(d.get("type", ""), 0))]
                for entry in data:
                    entry.setdefault("manual_trophies", True)
                self.brawlers_data = data
                self._refresh_brawler_grid()
                self._update_sidebar_brawler()
            except Exception as e:
                print(f"Error loading config: {e}")

    def _update_sidebar_brawler(self):
        roster_count = len(self.brawlers_data)
        if self._quest_farm_active:
            self._sb_brawler_label.configure(text="Quest Farm")
            self._home_brawler_lbl.configure(text="Quest Farm")
        elif self._trophy_farm_active:
            self._sb_brawler_label.configure(text="Trophy Farm")
            self._home_brawler_lbl.configure(text="Trophy Farm")
        elif self.brawlers_data:
            name = self.brawlers_data[0]["brawler"].title()
            self._sb_brawler_label.configure(text=name)
            self._home_brawler_lbl.configure(text=name)
        else:
            self._sb_brawler_label.configure(text="None")
            self._home_brawler_lbl.configure(text="None")
        self._sb_brawler_meta.configure(text=f"{roster_count} ready" if roster_count else "0 ready")
        if hasattr(self, "_home_roster_summary"):
            if self.brawlers_data:
                preview = ", ".join(item["brawler"].title() for item in self.brawlers_data[:4])
                if len(self.brawlers_data) > 4:
                    preview += f" +{len(self.brawlers_data) - 4} more"
                self._home_roster_summary.configure(
                    text=f"{roster_count} brawler{'s' if roster_count != 1 else ''} ready.\n{preview}"
                )
            else:
                self._home_roster_summary.configure(
                    text="No brawlers selected yet. Build your roster in the Brawlers page."
                )

    # --- tROPHY FARM PAGE ---

    def _build_farm_page(self):
        page = ctk.CTkScrollableFrame(self._content, fg_color=BG,
                                      corner_radius=0,
                                      scrollbar_button_color=ACCENT)
        self._pages["farm"] = page

        hdr = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(16),
                           height=S(65))
        hdr.pack(fill="x", padx=S(20), pady=(S(16), S(10)))
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="Farm",
                     font=(FONT_FAMILY_ALT, S(22), "bold"),
                     text_color=BRIGHT).pack(side="left", padx=S(16))
        self._farm_status_lbl = ctk.CTkLabel(
            hdr, text="DISABLED", font=(FONT_FAMILY_ALT, S(13), "bold"),
            text_color=DIM)
        self._farm_status_lbl.pack(side="right", padx=S(16))

        mode_switch = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(14), border_width=1, border_color=SEP)
        mode_switch.pack(fill="x", padx=S(20), pady=(S(2), S(10)))
        mode_switch_text = (
            "Switch between trophy routing and quest routing without leaving the unified shell."
            if self._capabilities.get("quest_farm") and self._capabilities.get("quest_scan")
            else "Trophy routing is available on this branch. Quest routing is not exposed here."
        )
        ctk.CTkLabel(mode_switch,
                     text=mode_switch_text,
                     font=(FONT_FAMILY_ALT, S(12)), text_color=TXT).pack(side="left", padx=S(14), pady=S(12))
        if self._capabilities.get("quest_farm") and self._capabilities.get("quest_scan"):
            ctk.CTkButton(
                mode_switch,
                text="Quest Farm",
                font=(FONT_FAMILY_ALT, S(12), "bold"),
                fg_color=SECTION,
                hover_color=CARD,
                height=S(36),
                corner_radius=S(10),
                border_width=1,
                border_color=SEP,
                command=lambda: self.show_page("quest"),
            ).pack(side="right", padx=S(12), pady=S(10))

        desc = ctk.CTkFrame(page, fg_color=SECTION, corner_radius=S(10))
        desc.pack(fill="x", padx=S(20), pady=S(6))
        farm_desc_text = (
            "\u2139  Trophy Farm automatically rotates through eligible brawlers below your target count.\n"
            "Use Quest Farm when you want the bot to route by active quests instead."
            if self._capabilities.get("quest_farm") and self._capabilities.get("quest_scan")
            else "\u2139  Trophy Farm automatically rotates through eligible brawlers below your target count.\n"
                 "This branch keeps the farm flow focused on trophy routing."
        )
        ctk.CTkLabel(desc,
                     text=farm_desc_text,
                     font=(FONT_FAMILY_ALT, S(12)), text_color=TXT, justify="left",
                     wraplength=S(750)).pack(padx=S(14), pady=S(10))

        # aCTION BUTTONS (top, always visible)
        btn_row = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        btn_row.pack(fill="x", padx=S(20), pady=(S(8), S(10)))

        ctk.CTkButton(btn_row, text="\U0001F680  START FARM",
                      font=("Segoe UI", S(18), "bold"),
                      fg_color=GOLD, hover_color=ACCENT_H, text_color=BG,
                      height=S(52), corner_radius=S(10), width=S(240),
                      command=self._start_trophy_farm
                      ).pack(side="left", padx=(S(14), S(8)), pady=S(10))

        ctk.CTkButton(btn_row, text="\U0001F504  Refresh Queue",
                      font=("Segoe UI", S(13), "bold"),
                      fg_color=BLUE, hover_color=CYAN, text_color=BG,
                      height=S(42), corner_radius=S(8), width=S(170),
                      command=self._refresh_farm_queue
                      ).pack(side="left", padx=S(6), pady=S(10))

        ctk.CTkButton(btn_row, text="\U0001F4BE  Save Config",
                      font=("Segoe UI", S(13), "bold"),
                      fg_color=GREEN, hover_color="#00A050", text_color=BG,
                      height=S(42), corner_radius=S(8), width=S(170),
                      command=self._save_farm_config
                      ).pack(side="left", padx=S(6), pady=S(10))

        # enable toggle
        self._section_header(page, "Farm Settings")
        settings_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        settings_f.pack(fill="x", padx=S(20), pady=S(4))

        # Enable/Disable
        self._farm_enabled_var = ctk.StringVar(
            value=self.bot_config.get("smart_trophy_farm", "no"))
        self._toggle_row(settings_f, "Enable Trophy Farm", self._farm_enabled_var)

        # Target trophies
        row_target = ctk.CTkFrame(settings_f, fg_color="transparent")
        row_target.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row_target, text="Target Trophies",
                     font=("Segoe UI", S(14), "bold"),
                     text_color=TXT).pack(side="left")
        self._farm_target_var = tk.StringVar(
            value=str(self.bot_config.get("trophy_farm_target", 500)))
        ctk.CTkEntry(row_target, textvariable=self._farm_target_var,
                     font=("Segoe UI", S(14)), width=S(100), height=S(34),
                     fg_color=SECTION, border_color=GOLD, text_color=GOLD,
                     corner_radius=S(6), justify="center"
                     ).pack(side="right")

        # Strategy
        row_strat = ctk.CTkFrame(settings_f, fg_color="transparent")
        row_strat.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row_strat, text="Strategy",
                     font=("Segoe UI", S(13)), text_color=TXT).pack(side="left")
        strat = self.bot_config.get("trophy_farm_strategy", "lowest_first")
        self._farm_strat_var = ctk.StringVar(value=strat)
        ctk.CTkOptionMenu(
            row_strat, variable=self._farm_strat_var,
            values=["lowest_first", "highest_winrate", "sequential"],
            font=("Segoe UI", S(13)), fg_color=SECTION,
            button_color=GOLD, button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL, dropdown_hover_color=SEP,
            width=S(180)
        ).pack(side="right")

        # excluded Brawlers
        self._section_header(page, "Exclude Brawlers (click to toggle)")
        self._farm_exclude_scroll = ctk.CTkScrollableFrame(
            page, fg_color=PANEL, corner_radius=S(10), height=S(220),
            scrollbar_button_color=ACCENT)
        self._farm_exclude_scroll.pack(fill="x", padx=S(20), pady=S(4))
        self._install_nested_scroll(self._farm_exclude_scroll)
        self._farm_exclude_vars = {}
        self._build_farm_exclude_grid()

        # queue Preview
        self._section_header(page, "Farm Queue Preview")
        self._farm_queue_frame = ctk.CTkFrame(page, fg_color=PANEL,
                                              corner_radius=S(10))
        self._farm_queue_frame.pack(fill="x", padx=S(20), pady=S(4))
        self._refresh_farm_queue()

    def _build_farm_exclude_grid(self):
        """Build the brawler exclude grid in the farm page."""
        for w in self._farm_exclude_scroll.winfo_children():
            w.destroy()

        cols = 8
        col = row = 0
        for brawler in sorted(self.all_brawlers):
            excluded = brawler in self._trophy_farm_excluded
            var = tk.BooleanVar(value=excluded)
            self._farm_exclude_vars[brawler] = var

            card = ctk.CTkFrame(self._farm_exclude_scroll,
                                fg_color=RED if excluded else SECTION,
                                corner_radius=S(6), width=S(105), height=S(36))
            card.grid(row=row, column=col, padx=S(3), pady=S(3), sticky="nsew")
            card.grid_propagate(False)

            disp = brawler.title() if len(brawler) <= 9 else brawler[:8].title() + "."
            lbl = ctk.CTkLabel(card, text=disp,
                               font=("Segoe UI", S(10)),
                               text_color=BRIGHT if excluded else TXT)
            lbl.pack(expand=True)

            def toggle(b=brawler, c=card, l=lbl, v=var):
                if b in self._trophy_farm_excluded:
                    self._trophy_farm_excluded.discard(b)
                    v.set(False)
                    c.configure(fg_color=SECTION)
                    l.configure(text_color=TXT)
                else:
                    self._trophy_farm_excluded.add(b)
                    v.set(True)
                    c.configure(fg_color=RED)
                    l.configure(text_color=BRIGHT)
                self._refresh_farm_queue()

            card.bind("<Button-1>", lambda e, t=toggle: t())
            lbl.bind("<Button-1>", lambda e, t=toggle: t())

            col += 1
            if col >= cols:
                col = 0
                row += 1

        for c in range(cols):
            self._farm_exclude_scroll.grid_columnconfigure(c, weight=1)
        self._install_nested_scroll(self._farm_exclude_scroll)

    def _get_farm_queue(self):
        """Build the ordered list of brawlers to farm based on settings."""
        try:
            target = int(self._farm_target_var.get())
        except (ValueError, tk.TclError):
            target = 500
        strategy = self._farm_strat_var.get()
        has_scan_data = bool(self._brawler_scan_data)

        queue = []
        for brawler in self.all_brawlers:
            if brawler in self._trophy_farm_excluded:
                continue

            # Skip locked brawlers if we have scan data
            scan_info = self._brawler_scan_data.get(brawler, {})
            if has_scan_data and not scan_info.get("unlocked", False):
                continue

            # Priority for trophies: active session data > scan data > 0
            selected_entry = self._get_selected_brawler_entry(brawler)
            trophies, _ = self._get_effective_trophies(
                brawler,
                selected_entry=selected_entry,
                scan_entry=scan_info,
            )
            if not isinstance(trophies, int):
                trophies = 0
            hist = self.match_history.get(brawler, {})

            if trophies < target:
                wins = hist.get("victory", 0) if isinstance(hist, dict) else 0
                losses = hist.get("defeat", 0) if isinstance(hist, dict) else 0
                total = wins + losses
                wr = round(100 * wins / total) if total > 0 else 50
                queue.append({
                    "brawler": brawler,
                    "trophies": trophies,
                    "winrate": wr,
                    "total_games": total,
                })

        if strategy == "lowest_first":
            queue.sort(key=lambda x: x["trophies"])
        elif strategy == "highest_winrate":
            queue.sort(key=lambda x: (-x["winrate"], x["trophies"]))
        elif strategy == "sequential":
            queue.sort(key=lambda x: x["brawler"])

        return queue, target

    def _refresh_farm_queue(self):
        """Refresh the farm queue preview display."""
        for w in self._farm_queue_frame.winfo_children():
            w.destroy()

        queue, target = self._get_farm_queue()

        if not queue:
            ctk.CTkLabel(self._farm_queue_frame,
                         text="All brawlers are at or above the target! \U0001F389",
                         font=("Segoe UI", S(14)), text_color=GREEN
                         ).pack(padx=S(14), pady=S(16))
            self._farm_status_lbl.configure(text="ALL DONE", text_color=GREEN)
            return

        # Summary
        summary = ctk.CTkFrame(self._farm_queue_frame, fg_color="transparent")
        summary.pack(fill="x", padx=S(14), pady=(S(8), S(4)))
        ctk.CTkLabel(summary,
                     text=f"{len(queue)} brawlers below {target} trophies",
                     font=("Segoe UI", S(14), "bold"),
                     text_color=GOLD).pack(side="left")
        avg_trophies = sum(b["trophies"] for b in queue) / len(queue) if queue else 0
        ctk.CTkLabel(summary,
                     text=f"Avg: {avg_trophies:.0f} trophies",
                     font=("Segoe UI", S(12)), text_color=DIM
                     ).pack(side="right")

        self._farm_status_lbl.configure(
            text=f"{len(queue)} TO FARM", text_color=GOLD)

        # Header row
        hdr = ctk.CTkFrame(self._farm_queue_frame, fg_color="transparent",
                           height=S(26))
        hdr.pack(fill="x", padx=S(14))
        hdr.pack_propagate(False)
        for txt, w in [("#", S(30)), ("Brawler", S(120)), ("Trophies", S(80)),
                       ("Target", S(70)), ("WR", S(60)), ("Games", S(60))]:
            ctk.CTkLabel(hdr, text=txt, font=("Segoe UI", S(10)),
                         text_color=DIM, width=w, anchor="w"
                         ).pack(side="left", padx=S(2))

        # Brawler rows (show up to 20)
        for i, bdata in enumerate(queue[:20]):
            row_bg = SECTION if i % 2 == 0 else PANEL
            row = ctk.CTkFrame(self._farm_queue_frame, fg_color=row_bg,
                               corner_radius=S(4), height=S(32))
            row.pack(fill="x", padx=S(10), pady=S(1))
            row.pack_propagate(False)

            # Position
            pos_c = GOLD if i == 0 else ACCENT if i < 3 else TXT
            ctk.CTkLabel(row, text=f"{i+1}",
                         font=("Segoe UI", S(11), "bold"),
                         text_color=pos_c, width=S(30), anchor="w"
                         ).pack(side="left", padx=(S(6), S(2)))

            # Name
            ctk.CTkLabel(row, text=bdata["brawler"].title(),
                         font=("Segoe UI", S(12), "bold"),
                         text_color=BRIGHT, width=S(120), anchor="w"
                         ).pack(side="left", padx=S(2))

            # Trophies
            t = bdata["trophies"]
            pct = t / target if target > 0 else 0
            t_color = GREEN if pct > 0.8 else GOLD if pct > 0.5 else RED
            ctk.CTkLabel(row, text=str(t),
                         font=("Segoe UI", S(12), "bold"),
                         text_color=t_color, width=S(80), anchor="w"
                         ).pack(side="left", padx=S(2))

            # Target
            ctk.CTkLabel(row, text=str(target),
                         font=("Segoe UI", S(11)),
                         text_color=DIM, width=S(70), anchor="w"
                         ).pack(side="left", padx=S(2))

            # Win rate
            wr = bdata["winrate"]
            wr_c = GREEN if wr >= 50 else RED
            ctk.CTkLabel(row, text=f"{wr}%",
                         font=("Segoe UI", S(11)),
                         text_color=wr_c, width=S(60), anchor="w"
                         ).pack(side="left", padx=S(2))

            # Games
            ctk.CTkLabel(row, text=str(bdata["total_games"]),
                         font=("Segoe UI", S(11)),
                         text_color=TXT, width=S(60), anchor="w"
                         ).pack(side="left", padx=S(2))

        if len(queue) > 20:
            ctk.CTkLabel(self._farm_queue_frame,
                         text=f"... and {len(queue) - 20} more",
                         font=("Segoe UI", S(11)), text_color=DIM
                         ).pack(pady=S(4))

    def _save_farm_config(self):
        """Save trophy farm settings to bot_config.toml."""
        bc = self.bot_config
        bc["smart_trophy_farm"] = self._farm_enabled_var.get()
        try:
            bc["trophy_farm_target"] = int(self._farm_target_var.get())
        except ValueError:
            bc["trophy_farm_target"] = 500
        bc["trophy_farm_strategy"] = self._farm_strat_var.get()
        bc["trophy_farm_excluded"] = sorted(list(self._trophy_farm_excluded))
        save_dict_as_toml(bc, "cfg/bot_config.toml")

        self._trophy_farm_active = self._farm_enabled_var.get() == "yes"
        self._trophy_farm_target = bc["trophy_farm_target"]
        self._trophy_farm_strategy = bc["trophy_farm_strategy"]

        self._status_label.configure(text="\u2714 Farm Config Saved", text_color=GREEN)
        self.after(2000, lambda: self._update_status_display())

    def _start_trophy_farm(self):
        """Build brawler queue from farm settings and start the bot."""
        # Save farm config first
        self._save_farm_config()

        queue, target = self._get_farm_queue()
        if not queue:
            self._status_label.configure(
                text="\u26A0 No brawlers below target!", text_color=RED)
            return

        # Build brawlers_data list for the bot
        farm_data = []
        for bdata in queue:
            farm_data.append({
                "brawler": bdata["brawler"],
                "push_until": target,
                "trophies": bdata["trophies"],
                "wins": 0,
                "type": "trophies",
                "automatically_pick": True,
                "win_streak": 0,
            })

        self._trophy_farm_active = True
        self.brawlers_data = farm_data
        self._update_sidebar_brawler()

        # Save and start
        if not self._logged_in and api_base_url != "localhost":
            self._status_label.configure(
                text="\u26A0 Login required!", text_color=RED)
            self._show_login_dialog()
            return

        self._save_all_settings()
        save_brawler_data(self.brawlers_data)

        self._prepare_bot_control_events()

        self.bot_running = True
        self._update_status_display()
        self.show_page("live")

        self.bot_thread = threading.Thread(
            target=self._run_bot, daemon=True)
        self.bot_thread.start()

    # --- qUEST FARM PAGE ---

    def _build_quest_page(self):
        if not (self._capabilities.get("quest_farm") and self._capabilities.get("quest_scan")):
            return
        page = ctk.CTkScrollableFrame(self._content, fg_color=BG,
                                      corner_radius=0,
                                      scrollbar_button_color=ACCENT)
        self._pages["quest"] = page

        # Header
        hdr = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(12),
                           height=S(65))
        hdr.pack(fill="x", padx=S(20), pady=(S(16), S(10)))
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="\U0001F4DC  Quest Farm",
                     font=("Segoe UI", S(22), "bold"),
                     text_color=BRIGHT).pack(side="left", padx=S(16))
        self._quest_status_lbl = ctk.CTkLabel(
            hdr, text="NOT SCANNED", font=("Segoe UI", S(13), "bold"),
            text_color=DIM)
        self._quest_status_lbl.pack(side="right", padx=S(16))

        # Description
        desc = ctk.CTkFrame(page, fg_color=SECTION, corner_radius=S(10))
        desc.pack(fill="x", padx=S(20), pady=S(6))
        ctk.CTkLabel(desc,
                     text="\u2139  Automatically plays all brawlers that have an active quest.\n"
                          "The bot scans the brawler grid for the quest icon (\U0001F4CB), plays a match\n"
                          "with each quest brawler, then re-checks if the quest is done.\n"
                          "Switches to the next brawler once a quest is completed.",
                     font=("Segoe UI", S(12)), text_color=TXT, justify="left",
                     wraplength=S(750)).pack(padx=S(14), pady=S(10))

        # aCTION BUTTONS
        btn_row = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        btn_row.pack(fill="x", padx=S(20), pady=(S(8), S(10)))

        self._quest_scan_btn = ctk.CTkButton(
            btn_row, text="\U0001F50D  Scan Quests",
            font=("Segoe UI", S(15), "bold"),
            fg_color=BLUE, hover_color=CYAN, text_color=BG,
            height=S(48), corner_radius=S(10), width=S(200),
            command=self._scan_quests)
        self._quest_scan_btn.pack(side="left", padx=(S(14), S(8)), pady=S(10))

        ctk.CTkButton(btn_row, text="\U0001F680  START QUEST FARM",
                      font=("Segoe UI", S(18), "bold"),
                      fg_color=GOLD, hover_color=ACCENT_H, text_color=BG,
                      height=S(52), corner_radius=S(10), width=S(260),
                      command=self._start_quest_farm
                      ).pack(side="left", padx=S(8), pady=S(10))

        ctk.CTkButton(btn_row, text="\U0001F4BE  Save Config",
                      font=("Segoe UI", S(13), "bold"),
                      fg_color=GREEN, hover_color="#00A050", text_color=BG,
                      height=S(42), corner_radius=S(8), width=S(150),
                      command=self._save_quest_config
                      ).pack(side="left", padx=S(6), pady=S(10))

        # scan Status
        self._quest_scan_status = ctk.CTkLabel(
            page, text="", font=("Segoe UI", S(12)), text_color=DIM)
        self._quest_scan_status.pack(anchor="w", padx=S(24), pady=S(2))

        # settings
        self._section_header(page, "Quest Farm Settings")
        settings_f = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        settings_f.pack(fill="x", padx=S(20), pady=S(4))

        # Quest mode: games or wins
        row_mode = ctk.CTkFrame(settings_f, fg_color="transparent")
        row_mode.pack(fill="x", padx=S(14), pady=S(8))
        ctk.CTkLabel(row_mode, text="Quest Completion Mode",
                     font=("Segoe UI", S(14), "bold"),
                     text_color=TXT).pack(side="left")
        quest_mode = self.bot_config.get("quest_farm_mode", "games")
        self._quest_mode_var = ctk.StringVar(value=quest_mode)
        ctk.CTkOptionMenu(
            row_mode, variable=self._quest_mode_var,
            values=["games", "wins"],
            font=("Segoe UI", S(13)), fg_color=SECTION,
            button_color=GOLD, button_hover_color=ACCENT_H,
            dropdown_fg_color=PANEL, dropdown_hover_color=SEP,
            width=S(160)
        ).pack(side="right")

        mode_desc = ctk.CTkFrame(settings_f, fg_color="transparent")
        mode_desc.pack(fill="x", padx=S(14), pady=(0, S(8)))
        ctk.CTkLabel(mode_desc,
                     text="'games' = just play matches  |  'wins' = only count victories",
                     font=("Segoe UI", S(11)), text_color=DIM
                     ).pack(anchor="w")

        # exclude Brawlers
        self._section_header(page, "Exclude Brawlers from Quest Farm (click to toggle)")
        self._quest_exclude_scroll = ctk.CTkScrollableFrame(
            page, fg_color=PANEL, corner_radius=S(10), height=S(180),
            scrollbar_button_color=ACCENT)
        self._quest_exclude_scroll.pack(fill="x", padx=S(20), pady=S(4))
        self._install_nested_scroll(self._quest_exclude_scroll)
        self._quest_exclude_vars = {}
        self._build_quest_exclude_grid()

        # quest Queue Preview
        self._section_header(page, "Quest Brawlers (scan to populate)")
        self._quest_queue_frame = ctk.CTkFrame(page, fg_color=PANEL,
                                               corner_radius=S(10))
        self._quest_queue_frame.pack(fill="x", padx=S(20), pady=S(4))
        self._refresh_quest_queue()

    def _build_quest_exclude_grid(self):
        """Build the brawler exclude grid for quest farm."""
        for w in self._quest_exclude_scroll.winfo_children():
            w.destroy()

        cols = 8
        col = row = 0
        for brawler in sorted(self.all_brawlers):
            excluded = brawler in self._quest_farm_excluded
            var = tk.BooleanVar(value=excluded)
            self._quest_exclude_vars[brawler] = var

            card = ctk.CTkFrame(self._quest_exclude_scroll,
                                fg_color=RED if excluded else SECTION,
                                corner_radius=S(6), width=S(105), height=S(36))
            card.grid(row=row, column=col, padx=S(3), pady=S(3), sticky="nsew")
            card.grid_propagate(False)

            disp = brawler.title() if len(brawler) <= 9 else brawler[:8].title() + "."
            lbl = ctk.CTkLabel(card, text=disp,
                               font=("Segoe UI", S(10)),
                               text_color=BRIGHT if excluded else TXT)
            lbl.pack(expand=True)

            def toggle(b=brawler, c=card, l=lbl, v=var):
                if b in self._quest_farm_excluded:
                    self._quest_farm_excluded.discard(b)
                    v.set(False)
                    c.configure(fg_color=SECTION)
                    l.configure(text_color=TXT)
                else:
                    self._quest_farm_excluded.add(b)
                    v.set(True)
                    c.configure(fg_color=RED)
                    l.configure(text_color=BRIGHT)
                self._refresh_quest_queue()

            card.bind("<Button-1>", lambda e, t=toggle: t())
            lbl.bind("<Button-1>", lambda e, t=toggle: t())

            col += 1
            if col >= cols:
                col = 0
                row += 1

        for c in range(cols):
            self._quest_exclude_scroll.grid_columnconfigure(c, weight=1)
        self._install_nested_scroll(self._quest_exclude_scroll)

    def _refresh_quest_queue(self):
        """Refresh the quest brawler queue display."""
        for w in self._quest_queue_frame.winfo_children():
            w.destroy()

        # Filter out excluded brawlers
        queue = [b for b in self._quest_brawlers
                 if b not in self._quest_farm_excluded]

        if not self._quest_brawlers:
            ctk.CTkLabel(self._quest_queue_frame,
                         text="No scan data yet. Click '\U0001F50D Scan Quests' to detect quest brawlers.",
                         font=("Segoe UI", S(13)), text_color=DIM
                         ).pack(padx=S(14), pady=S(16))
            self._quest_status_lbl.configure(text="NOT SCANNED", text_color=DIM)
            return

        if not queue:
            ctk.CTkLabel(self._quest_queue_frame,
                         text="All quest brawlers are excluded or quests are done! \U0001F389",
                         font=("Segoe UI", S(14)), text_color=GREEN
                         ).pack(padx=S(14), pady=S(16))
            self._quest_status_lbl.configure(text="ALL DONE", text_color=GREEN)
            return

        # Summary
        summary = ctk.CTkFrame(self._quest_queue_frame, fg_color="transparent")
        summary.pack(fill="x", padx=S(14), pady=(S(8), S(4)))
        ctk.CTkLabel(summary,
                     text=f"{len(queue)} brawler(s) with active quests",
                     font=("Segoe UI", S(14), "bold"),
                     text_color=GOLD).pack(side="left")
        mode = self._quest_mode_var.get() if hasattr(self, '_quest_mode_var') else "games"
        ctk.CTkLabel(summary,
                     text=f"Mode: {mode}",
                     font=("Segoe UI", S(12)), text_color=DIM
                     ).pack(side="right")

        self._quest_status_lbl.configure(
            text=f"{len(queue)} QUESTS", text_color=GOLD)

        # Header row
        hdr = ctk.CTkFrame(self._quest_queue_frame, fg_color="transparent",
                           height=S(26))
        hdr.pack(fill="x", padx=S(14))
        hdr.pack_propagate(False)
        for txt, w in [("#", S(30)), ("Brawler", S(160)), ("Status", S(120))]:
            ctk.CTkLabel(hdr, text=txt, font=("Segoe UI", S(10)),
                         text_color=DIM, width=w, anchor="w"
                         ).pack(side="left", padx=S(2))

        # Brawler rows
        for i, brawler in enumerate(queue):
            row_bg = SECTION if i % 2 == 0 else PANEL
            row = ctk.CTkFrame(self._quest_queue_frame, fg_color=row_bg,
                               corner_radius=S(4), height=S(32))
            row.pack(fill="x", padx=S(10), pady=S(1))
            row.pack_propagate(False)

            # Position
            pos_c = GOLD if i == 0 else ACCENT if i < 3 else TXT
            ctk.CTkLabel(row, text=f"{i+1}",
                         font=("Segoe UI", S(11), "bold"),
                         text_color=pos_c, width=S(30), anchor="w"
                         ).pack(side="left", padx=(S(6), S(2)))

            # Name
            ctk.CTkLabel(row, text=brawler.title(),
                         font=("Segoe UI", S(12), "bold"),
                         text_color=BRIGHT, width=S(160), anchor="w"
                         ).pack(side="left", padx=S(2))

            # Status
            ctk.CTkLabel(row, text="\U0001F4DC Quest Active",
                         font=("Segoe UI", S(11)),
                         text_color=GOLD, width=S(120), anchor="w"
                         ).pack(side="left", padx=S(2))

    def _scan_quests(self):
        """Scan the brawler grid for quest icons in a background thread."""
        if not (self._capabilities.get("quest_farm") and self._capabilities.get("quest_scan")):
            self._status_label.configure(
                text="\u26A0 Quest farm scanning is not available on this branch.", text_color=GOLD)
            return
        if self._quest_scan_in_progress:
            self._status_label.configure(
                text="\u26A0 Quest scan already in progress!", text_color=GOLD)
            return

        self._quest_scan_in_progress = True
        self._quest_scan_btn.configure(state="disabled", text="\u23F3 Scanning...")
        self._quest_scan_status.configure(text="Connecting to emulator...", text_color=GOLD)

        def run_scan():
            temp_wc = None
            try:
                wc = None
                lobby_auto = None

                # 1) Try to reuse the running bot's connection
                if self.bot_running:
                    import sys
                    main_module = sys.modules.get('__main__')
                    if main_module:
                        sm = getattr(main_module, '_active_stage_manager', None)
                        if sm:
                            wc = sm.window_controller
                            lobby_auto = sm.Lobby_automation

                # 2) No running bot - create a temporary connection
                if wc is None:
                    self.after(0, lambda: self._quest_scan_status.configure(
                        text="Connecting to emulator...", text_color=GOLD))
                    try:
                        from window_controller import WindowController
                        from lobby_automation import LobbyAutomation
                        temp_wc = WindowController()
                        wc = temp_wc
                        lobby_auto = LobbyAutomation(wc)
                        wc.screenshot()
                    except Exception as conn_err:
                        self.after(0, lambda e=str(conn_err): (
                            self._quest_scan_status.configure(
                                text="Error: Could not connect to emulator",
                                text_color=RED),
                            self._status_label.configure(
                                text=f"\u26A0 {e[:50]}", text_color=RED)
                        ))
                        return

                self.after(0, lambda: self._quest_scan_status.configure(
                    text="Scanning for quest icons...", text_color=GOLD))

                def progress_cb(scanned, quest_count, brawler_name):
                    self.after(0, lambda s=scanned, q=quest_count, b=brawler_name:
                        self._quest_scan_status.configure(
                            text=f"Scanned {s} brawlers, {q} quests found - {b.title()}",
                            text_color=GOLD))

                quest_brawlers = lobby_auto.scan_quest_brawlers(
                    self.all_brawlers, progress_callback=progress_cb)

                self._quest_brawlers = quest_brawlers

                self.after(0, lambda: (
                    self._quest_scan_status.configure(
                        text=f"Done! {len(quest_brawlers)} brawler(s) with active quests",
                        text_color=GREEN),
                    self._refresh_quest_queue(),
                    self._status_label.configure(
                        text=f"\u2714 Quest scan: {len(quest_brawlers)} quests found",
                        text_color=GREEN)
                ))

            except Exception as e:
                print(f"[QUEST] Error during scan: {e}")
                import traceback
                traceback.print_exc()
                self.after(0, lambda: self._quest_scan_status.configure(
                    text=f"Error: {str(e)[:40]}", text_color=RED))
            finally:
                if temp_wc is not None:
                    try:
                        temp_wc.close()
                    except Exception:
                        pass
                self._quest_scan_in_progress = False
                self.after(0, lambda: self._quest_scan_btn.configure(
                    state="normal", text="\U0001F50D  Scan Quests"))

        threading.Thread(target=run_scan, daemon=True).start()

    def _save_quest_config(self):
        """Save quest farm settings to bot_config.toml."""
        bc = self.bot_config
        bc["quest_farm_enabled"] = "yes" if self._quest_farm_active else "no"
        bc["quest_farm_mode"] = self._quest_mode_var.get()
        bc["quest_farm_excluded"] = sorted(list(self._quest_farm_excluded))
        save_dict_as_toml(bc, "cfg/bot_config.toml")

        self._status_label.configure(text="\u2714 Quest Config Saved", text_color=GREEN)
        self.after(2000, lambda: self._update_status_display())

    def _start_quest_farm(self):
        """Build brawler queue from quest scan results and start the bot."""
        if not (self._capabilities.get("quest_farm") and self._capabilities.get("quest_scan")):
            self._status_label.configure(
                text="\u26A0 Quest farm is not available on this branch.", text_color=GOLD)
            return
        # Save config first
        self._save_quest_config()

        # Filter out excluded brawlers
        queue = [b for b in self._quest_brawlers
                 if b not in self._quest_farm_excluded]

        if not queue:
            self._status_label.configure(
                text="\u26A0 No quest brawlers! Scan first.", text_color=RED)
            return

        # Build brawlers_data list for the bot
        quest_data = []
        mode = self._quest_mode_var.get()
        for brawler in queue:
            # Get trophy data from scan data if available
            scan_info = self._brawler_scan_data.get(brawler, {})
            trophies = scan_info.get("trophies", 0)

            quest_data.append({
                "brawler": brawler,
                "push_until": 99999,  # Never reach this - quest check handles rotation
                "trophies": trophies,
                "wins": 0,
                "type": "quest",
                "quest_mode": mode,   # "games" or "wins"
                "automatically_pick": True,
                "win_streak": 0,
            })

        self._quest_farm_active = True
        self._trophy_farm_active = False
        self.brawlers_data = quest_data
        self._update_sidebar_brawler()

        # Login check
        if not self._logged_in and api_base_url != "localhost":
            self._status_label.configure(
                text="\u26A0 Login required!", text_color=RED)
            self._show_login_dialog()
            return

        self._save_all_settings()
        save_brawler_data(self.brawlers_data)

        self._prepare_bot_control_events()

        self.bot_running = True
        self._update_status_display()
        self.show_page("live")

        self.bot_thread = threading.Thread(
            target=self._run_bot, daemon=True)
        self.bot_thread.start()

    # --- lIVE STATS PAGE ---

    def _build_live_page(self):
        page = ctk.CTkScrollableFrame(self._content, fg_color=BG,
                                      corner_radius=0,
                                      scrollbar_button_color=ACCENT)
        self._pages["live"] = page

        # hEADER
        hdr = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(12),
                           height=S(50))
        hdr.pack(fill="x", padx=S(16), pady=(S(12), S(6)))
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="\U0001F4CA  Live Stats",
                     font=("Segoe UI", S(20), "bold"),
                     text_color=BRIGHT).pack(side="left", padx=S(14))
        self._live_status = ctk.CTkLabel(
            hdr, text="\u25CF NOT RUNNING",
            font=("Segoe UI", S(12), "bold"), text_color=DIM)
        self._live_status.pack(side="right", padx=S(14))

        # tOP ROW: Session + Brawler + Trophies (3 columns)
        top_row = ctk.CTkFrame(page, fg_color="transparent")
        top_row.pack(fill="x", padx=S(12), pady=S(3))
        top_row.grid_columnconfigure((0, 1, 2), weight=1)

        # session Card
        sess_card = ctk.CTkFrame(top_row, fg_color=PANEL, corner_radius=S(10))
        sess_card.grid(row=0, column=0, padx=S(3), pady=S(3), sticky="nsew")
        ctk.CTkLabel(sess_card, text="SESSION",
                     font=("Segoe UI", S(9), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(10), pady=(S(6), S(1)))

        s_row1 = ctk.CTkFrame(sess_card, fg_color="transparent")
        s_row1.pack(fill="x", padx=S(10), pady=S(1))
        self._live_uptime = ctk.CTkLabel(s_row1, text="00:00:00",
                                         font=("Consolas", S(20), "bold"),
                                         text_color=BLUE)
        self._live_uptime.pack(side="left")
        self._live_ips = ctk.CTkLabel(s_row1, text="0 IPS",
                                      font=("Segoe UI", S(12), "bold"),
                                      text_color=DIM)
        self._live_ips.pack(side="right")

        s_row2 = ctk.CTkFrame(sess_card, fg_color="transparent")
        s_row2.pack(fill="x", padx=S(10), pady=(0, S(2)))
        self._live_state = ctk.CTkLabel(s_row2, text="State: \u2014",
                                        font=("Segoe UI", S(11)),
                                        text_color=TXT)
        self._live_state.pack(side="left")
        self._live_gamemode = ctk.CTkLabel(s_row2, text="",
                                           font=("Segoe UI", S(10)),
                                           text_color=DIM)
        self._live_gamemode.pack(side="right")

        s_row3 = ctk.CTkFrame(sess_card, fg_color="transparent")
        s_row3.pack(fill="x", padx=S(10), pady=(0, S(6)))
        self._live_matches = ctk.CTkLabel(s_row3, text="Matches: 0",
                                          font=("Segoe UI", S(10)),
                                          text_color=TXT)
        self._live_matches.pack(side="left")
        self._live_farm_info = ctk.CTkLabel(s_row3, text="",
                                            font=("Segoe UI", S(10), "bold"),
                                            text_color=GOLD)
        self._live_farm_info.pack(side="right")

        # brawler Card
        brawl_card = ctk.CTkFrame(top_row, fg_color=PANEL, corner_radius=S(10))
        brawl_card.grid(row=0, column=1, padx=S(3), pady=S(3), sticky="nsew")
        ctk.CTkLabel(brawl_card, text="BRAWLER",
                     font=("Segoe UI", S(9), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(10), pady=(S(6), S(1)))

        self._live_brawler = ctk.CTkLabel(brawl_card, text="\u2014",
                                          font=("Segoe UI", S(20), "bold"),
                                          text_color=ACCENT)
        self._live_brawler.pack(anchor="w", padx=S(10))

        b_row2 = ctk.CTkFrame(brawl_card, fg_color="transparent")
        b_row2.pack(fill="x", padx=S(10), pady=S(1))
        self._live_playstyle = ctk.CTkLabel(b_row2, text="",
                                            font=("Segoe UI", S(10)),
                                            text_color=DIM)
        self._live_playstyle.pack(side="left")
        self._live_streak = ctk.CTkLabel(b_row2, text="",
                                         font=("Segoe UI", S(10), "bold"),
                                         text_color=GOLD)
        self._live_streak.pack(side="right")

        b_row3 = ctk.CTkFrame(brawl_card, fg_color="transparent")
        b_row3.pack(fill="x", padx=S(10), pady=(0, S(6)))
        self._live_wl = ctk.CTkLabel(b_row3, text="W:0  L:0  D:0",
                                     font=("Segoe UI", S(11)),
                                     text_color=TXT)
        self._live_wl.pack(side="left")
        self._live_wr_pct = ctk.CTkLabel(b_row3, text="",
                                         font=("Segoe UI", S(12), "bold"),
                                         text_color=DIM)
        self._live_wr_pct.pack(side="right")

        # trophy Card
        troph_card = ctk.CTkFrame(top_row, fg_color=PANEL, corner_radius=S(10))
        troph_card.grid(row=0, column=2, padx=S(3), pady=S(3), sticky="nsew")
        ctk.CTkLabel(troph_card, text="\U0001F3C6 TROPHIES",
                     font=("Segoe UI", S(9), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(10), pady=(S(6), S(1)))

        tr_top = ctk.CTkFrame(troph_card, fg_color="transparent")
        tr_top.pack(fill="x", padx=S(10), pady=S(1))
        self._live_trophy_current = ctk.CTkLabel(
            tr_top, text="0", font=("Segoe UI", S(20), "bold"),
            text_color=GOLD)
        self._live_trophy_current.pack(side="left")
        self._live_trophy_target = ctk.CTkLabel(
            tr_top, text="/ ?", font=("Segoe UI", S(12)),
            text_color=DIM)
        self._live_trophy_target.pack(side="left", padx=(S(4), 0))
        self._live_trophy_pct = ctk.CTkLabel(
            tr_top, text="", font=("Segoe UI", S(13), "bold"),
            text_color=DIM)
        self._live_trophy_pct.pack(side="right")

        self._live_trophy_bar = ctk.CTkProgressBar(
            troph_card, width=S(200), height=S(12),
            fg_color=SECTION, progress_color=GOLD,
            corner_radius=S(4))
        self._live_trophy_bar.pack(fill="x", padx=S(10), pady=(S(2), S(6)))
        self._live_trophy_bar.set(0)

        tr_meta = ctk.CTkFrame(troph_card, fg_color="transparent")
        tr_meta.pack(fill="x", padx=S(10), pady=(0, S(6)))
        self._live_trophy_delta = ctk.CTkLabel(
            tr_meta, text="", font=("Segoe UI", S(10), "bold"),
            text_color=DIM)
        self._live_trophy_delta.pack(side="left")
        self._live_trophy_detail = ctk.CTkLabel(
            tr_meta, text="", font=("Segoe UI", S(9)),
            text_color=DIM)
        self._live_trophy_detail.pack(side="right")

        # quick KPI strip
        kpi_row = ctk.CTkFrame(page, fg_color="transparent")
        kpi_row.pack(fill="x", padx=S(12), pady=(S(1), S(4)))
        kpi_row.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        def make_kpi(col, title, value="—", color=DIM):
            card = ctk.CTkFrame(kpi_row, fg_color=SECTION, corner_radius=S(8), height=S(54))
            card.grid(row=0, column=col, padx=S(3), pady=S(2), sticky="nsew")
            card.pack_propagate(False)
            ctk.CTkLabel(card, text=title,
                         font=("Segoe UI", S(9), "bold"),
                         text_color=ACCENT).pack(anchor="w", padx=S(8), pady=(S(5), 0))
            lbl = ctk.CTkLabel(card, text=value,
                               font=("Segoe UI", S(12), "bold"),
                               text_color=color)
            lbl.pack(anchor="w", padx=S(8), pady=(0, S(5)))
            return lbl

        self._live_kpi_kda = make_kpi(0, "KDA", "0.00", DIM)
        self._live_kpi_dpm = make_kpi(1, "DMG / MIN", "0", DIM)
        self._live_kpi_win_pace = make_kpi(2, "WINS / H", "0.0", DIM)
        self._live_kpi_target_gap = make_kpi(3, "TO TARGET", "—", DIM)
        self._live_kpi_feed = make_kpi(4, "LIVE FEED", "waiting", GOLD)

        # cOMBAT (full width)
        combat = ctk.CTkFrame(page, fg_color=PANEL, corner_radius=S(10))
        combat.pack(fill="x", padx=S(15), pady=S(3))

        # Title row with combat label + score + deaths
        combat_hdr = ctk.CTkFrame(combat, fg_color="transparent")
        combat_hdr.pack(fill="x", padx=S(10), pady=(S(6), S(3)))
        ctk.CTkLabel(combat_hdr, text="\u2694 COMBAT",
                     font=("Segoe UI", S(10), "bold"),
                     text_color=ACCENT).pack(side="left")
        self._live_score = ctk.CTkLabel(
            combat_hdr, text="", font=("Segoe UI", S(11), "bold"),
            text_color=BRIGHT)
        self._live_score.pack(side="left", padx=(S(14), 0))
        self._live_deaths = ctk.CTkLabel(
            combat_hdr, text="", font=("Segoe UI", S(10)),
            text_color=RED)
        self._live_deaths.pack(side="right")

        # HP bars side by side using grid
        hp_grid = ctk.CTkFrame(combat, fg_color="transparent")
        hp_grid.pack(fill="x", padx=S(10), pady=S(2))
        hp_grid.grid_columnconfigure((0, 1), weight=1)

        # Player HP
        player_hp_f = ctk.CTkFrame(hp_grid, fg_color=SECTION, corner_radius=S(6))
        player_hp_f.grid(row=0, column=0, padx=S(3), pady=S(2), sticky="nsew")
        php_inner = ctk.CTkFrame(player_hp_f, fg_color="transparent")
        php_inner.pack(fill="x", padx=S(8), pady=S(5))
        ctk.CTkLabel(php_inner, text="PLAYER",
                     font=("Segoe UI", S(9), "bold"),
                     text_color=TXT).pack(side="left")
        self._live_hp_text = ctk.CTkLabel(
            php_inner, text="100%", font=("Segoe UI", S(11), "bold"),
            text_color=HP_G)
        self._live_hp_text.pack(side="right")
        self._live_hp_bar = ctk.CTkProgressBar(
            player_hp_f, height=S(10),
            fg_color=BG, progress_color=HP_G, corner_radius=S(3))
        self._live_hp_bar.pack(fill="x", padx=S(8), pady=(0, S(5)))
        self._live_hp_bar.set(1.0)

        # Enemy HP
        enemy_hp_f = ctk.CTkFrame(hp_grid, fg_color=SECTION, corner_radius=S(6))
        enemy_hp_f.grid(row=0, column=1, padx=S(3), pady=S(2), sticky="nsew")
        ehp_inner = ctk.CTkFrame(enemy_hp_f, fg_color="transparent")
        ehp_inner.pack(fill="x", padx=S(8), pady=S(5))
        self._live_target = ctk.CTkLabel(
            ehp_inner, text="\u25B6 No target",
            font=("Segoe UI", S(9), "bold"), text_color=DIM)
        self._live_target.pack(side="left")
        self._live_ehp_text = ctk.CTkLabel(
            ehp_inner, text="\u2014", font=("Segoe UI", S(11), "bold"),
            text_color=DIM)
        self._live_ehp_text.pack(side="right")
        self._live_ehp_bar = ctk.CTkProgressBar(
            enemy_hp_f, height=S(10),
            fg_color=BG, progress_color=RED, corner_radius=S(3))
        self._live_ehp_bar.pack(fill="x", padx=S(8), pady=(0, S(5)))
        self._live_ehp_bar.set(0)

        # info strip: Decision | Ammo | Abilities - single row
        info_strip = ctk.CTkFrame(combat, fg_color=SECTION, corner_radius=S(6))
        info_strip.pack(fill="x", padx=S(10), pady=S(3))
        info_inner = ctk.CTkFrame(info_strip, fg_color="transparent")
        info_inner.pack(fill="x", padx=S(8), pady=S(5))

        self._live_decision = ctk.CTkLabel(
            info_inner, text="", font=("Segoe UI", S(11), "bold"),
            text_color=BRIGHT)
        self._live_decision.pack(side="left")

        # Abilities (right side)
        abil_fr = ctk.CTkFrame(info_inner, fg_color="transparent")
        abil_fr.pack(side="right")
        self._live_hyp = ctk.CTkLabel(abil_fr, text=" HYP ",
                                      font=("Segoe UI", S(9), "bold"),
                                      text_color=DIM, fg_color=PANEL,
                                      corner_radius=S(3))
        self._live_hyp.pack(side="right", padx=S(2))
        self._live_sup = ctk.CTkLabel(abil_fr, text=" SUP ",
                                      font=("Segoe UI", S(9), "bold"),
                                      text_color=DIM, fg_color=PANEL,
                                      corner_radius=S(3))
        self._live_sup.pack(side="right", padx=S(2))
        self._live_gad = ctk.CTkLabel(abil_fr, text=" GAD ",
                                      font=("Segoe UI", S(9), "bold"),
                                      text_color=DIM, fg_color=PANEL,
                                      corner_radius=S(3))
        self._live_gad.pack(side="right", padx=S(2))

        # Ammo dots (between decision and abilities)
        self._live_ammo_frame = ctk.CTkFrame(info_inner, fg_color="transparent")
        self._live_ammo_frame.pack(side="right", padx=(0, S(8)))
        self._live_ammo_dots = []
        for i in range(3):
            dot = ctk.CTkLabel(self._live_ammo_frame, text="\u25C9",
                               font=("Segoe UI", S(14)),
                               text_color=CYAN)
            dot.pack(side="left", padx=S(1))
            self._live_ammo_dots.append(dot)
        self._live_ammo_lbl = ctk.CTkLabel(
            self._live_ammo_frame, text="3/3",
            font=("Segoe UI", S(10)), text_color=CYAN)
        self._live_ammo_lbl.pack(side="left", padx=(S(4), 0))

        # status bar: Movement | Foes | Walls
        status_bar = ctk.CTkFrame(combat, fg_color="transparent")
        status_bar.pack(fill="x", padx=S(10), pady=(0, S(6)))
        self._live_movement = ctk.CTkLabel(
            status_bar, text="Move: \u2014  |  Foes: 0  |  Walls: 0",
            font=("Segoe UI", S(10)), text_color=DIM)
        self._live_movement.pack(side="left")

        # pERFORMANCE (3 compact cards)
        perf_label = ctk.CTkLabel(page, text="\U0001F4C8 PERFORMANCE",
                     font=("Segoe UI", S(10), "bold"),
                     text_color=ACCENT)
        perf_label.pack(anchor="w", padx=S(20), pady=(S(6), S(2)))

        perf = ctk.CTkFrame(page, fg_color="transparent")
        perf.pack(fill="x", padx=S(12), pady=(S(0), S(12)))
        perf.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self._live_perf_current = self._perf_card(perf, "THIS MATCH", 0, 0, show_deaths=True)
        self._live_perf_last = self._perf_card(perf, "LAST MATCH", 0, 1)
        self._live_perf_total = self._perf_card(perf, "TOTAL", 0, 2)
        self._live_perf_avg = self._perf_card(perf, "AVERAGE", 0, 3)

        if self._capabilities.get("advanced_live"):
            self._live_rl_info = ctk.CTkLabel(
                perf,
                text="Overview: waiting for live data...",
                font=("Segoe UI", S(10)),
                text_color=DIM,
                anchor="w",
                justify="left",
                fg_color=SECTION,
                corner_radius=S(8),
                padx=S(10),
                pady=S(6),
            )
            self._live_rl_info.grid(row=1, column=0, columnspan=4, padx=S(4), pady=(S(2), S(4)), sticky="ew")

    def _perf_card(self, parent, title, row, col, show_deaths=False):
        """Create a performance stats card and return dict of labels."""
        card = ctk.CTkFrame(parent, fg_color=SECTION, corner_radius=S(8))
        card.grid(row=row, column=col, padx=S(4), pady=S(4), sticky="nsew")

        ctk.CTkLabel(card, text=title,
                     font=("Segoe UI", S(9), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(10), pady=(S(6), S(2)))

        kills_lbl = ctk.CTkLabel(card, text="0 Kills",
                                 font=("Segoe UI", S(13), "bold"),
                                 text_color=RED)
        kills_lbl.pack(anchor="w", padx=S(10))

        assists_lbl = ctk.CTkLabel(card, text="0 Assists",
                                   font=("Segoe UI", S(11)),
                                   text_color=BLUE)
        assists_lbl.pack(anchor="w", padx=S(10))

        dmg_lbl = ctk.CTkLabel(card, text="0 Damage",
                                font=("Segoe UI", S(11)),
                                text_color=GOLD)
        dmg_lbl.pack(anchor="w", padx=S(10), pady=(0, S(2) if show_deaths else S(6)))

        result = {"kills": kills_lbl, "assists": assists_lbl, "damage": dmg_lbl}

        if show_deaths:
            deaths_lbl = ctk.CTkLabel(card, text="0 Deaths",
                                      font=("Segoe UI", S(11)),
                                      text_color=DIM)
            deaths_lbl.pack(anchor="w", padx=S(10), pady=(0, S(6)))
            result["deaths"] = deaths_lbl

        return result

    def _live_section(self, parent, text):
        ctk.CTkLabel(parent, text=text,
                     font=("Segoe UI", S(12), "bold"),
                     text_color=ACCENT).pack(anchor="w", padx=S(24),
                                              pady=(S(10), S(2)))

    def _load_training_stats_snapshot(self):
        """Load rl_models/training_stats.json with mtime cache for live fallback display."""
        path = os.path.join("rl_models", "training_stats.json")
        try:
            mtime = os.path.getmtime(path)
            if self._training_stats_cache and self._training_stats_mtime == mtime:
                return self._training_stats_cache
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._training_stats_cache = data if isinstance(data, dict) else {}
            self._training_stats_mtime = mtime
            return self._training_stats_cache
        except Exception:
            return self._training_stats_cache if isinstance(self._training_stats_cache, dict) else {}

    # --- lIVE DATA UPDATES (called from bot thread) ---

    def update_live(self, **kw):
        """Thread-safe live data update. Called from the bot's main loop."""
        with self._live_lock:
            self._live_data.update(kw)
            self._last_live_update_ts = time.time()

    # change-detection helpers to avoid redundant .configure() calls
    def _set(self, widget, key, **kwargs):
        """Only call widget.configure() when the value actually changed."""
        prev = self._prev_live.get(key)
        val = tuple(sorted(kwargs.items()))
        if prev == val:
            return
        self._prev_live[key] = val
        widget.configure(**kwargs)

    def _tick(self):
        """Periodic refresh of live page data."""
        if not self._live_running:
            return
        try:
            now = time.time()
            history_updated = False
            if now - self._last_history_refresh_ts > 2.0:
                self.match_history = load_toml_as_dict("cfg/match_history.toml")
                self._last_history_refresh_ts = now
                history_updated = True
            self._refresh_live()
            if history_updated and self._current_page == "home":
                self._refresh_home_history()
            elif history_updated and self._current_page == "history":
                self._refresh_history_page()
            if self._current_page == "live" or self.bot_running:
                self._update_status_display()
        except Exception:
            pass
        self.after(350, self._tick)

    def _update_status_display(self):
        if self.bot_running:
            state = self._live_data.get("state", "running").upper()
            self._status_label.configure(
                text=f"\u25CF {state}", text_color=GREEN)
            self._start_btn.configure(
                text="\u25A0  STOP BOT", fg_color=RED,
                hover_color="#CC2222")
            if hasattr(self, "_home_status_pill"):
                self._home_status_pill.configure(text=state, fg_color=ACCENT_D, text_color=BRIGHT)
            if hasattr(self, "_home_action_btn"):
                self._home_action_btn.configure(text="STOP BOT", fg_color=RED, hover_color="#CC2222")
            if hasattr(self, "_sidebar_footer"):
                self._sidebar_footer.configure(text="Device: bot active")
        else:
            self._status_label.configure(
                text="\u25CF READY", text_color=DIM)
            self._start_btn.configure(
                text="\u25B6  START BOT", fg_color=ACCENT,
                hover_color=ACCENT_H)
            if hasattr(self, "_home_status_pill"):
                self._home_status_pill.configure(text="READY", fg_color=SECTION, text_color=BRIGHT)
            if hasattr(self, "_home_action_btn"):
                self._home_action_btn.configure(text="START BOT", fg_color=ACCENT, hover_color=ACCENT_H)
            if hasattr(self, "_sidebar_footer"):
                self._sidebar_footer.configure(text="Device: ready")

    def _refresh_live(self):
        with self._live_lock:
            d = dict(self._live_data)
            last_update = self._last_live_update_ts
        if not d:
            if self.bot_running and self._current_page == "live":
                _s = self._set
                _s(self._live_status, "ls_t", text="● RUNNING (waiting data)", text_color=GOLD)
            return

        _s = self._set  # local alias for speed

        running = self.bot_running
        feed_age = (time.time() - last_update) if last_update else None
        if running:
            if feed_age is not None and feed_age > 2.5:
                _s(self._live_status, "ls_t", text="\u25CF RUNNING (slow feed)", text_color=GOLD)
            else:
                _s(self._live_status, "ls_t", text="\u25CF RUNNING", text_color=GREEN)
        else:
            _s(self._live_status, "ls_t", text="\u25CF NOT RUNNING", text_color=DIM)
            return

        if feed_age is None:
            _s(self._live_kpi_feed, "kpi_feed", text="waiting", text_color=GOLD)
        elif feed_age <= 0.5:
            _s(self._live_kpi_feed, "kpi_feed", text=f"{int(feed_age * 1000)} ms", text_color=GREEN)
        elif feed_age <= 2.5:
            _s(self._live_kpi_feed, "kpi_feed", text=f"{int(feed_age * 1000)} ms", text_color=GOLD)
        else:
            _s(self._live_kpi_feed, "kpi_feed", text=f"{feed_age:.1f}s lag", text_color=RED)

        # session card
        incoming_start = d.get("start_time")
        if isinstance(incoming_start, (int, float)) and incoming_start > 0:
            if self._live_session_start_ts is None:
                self._live_session_start_ts = incoming_start
            else:
                self._live_session_start_ts = min(self._live_session_start_ts, incoming_start)
        elif self._live_session_start_ts is None:
            self._live_session_start_ts = time.time()

        st = self._live_session_start_ts or time.time()
        elapsed = max(1.0, time.time() - st)
        h, rem = divmod(int(elapsed), 3600)
        m, s = divmod(rem, 60)
        _s(self._live_uptime, "up", text=f"{h:02d}:{m:02d}:{s:02d}")
        _s(self._live_ips, "ips", text=f"{d.get('ips', 0):.0f} IPS")

        state = d.get("state", "?").upper()
        state_colors = {"MATCH": GREEN, "LOBBY": GOLD, "PLAY_STORE": PURPLE}
        _s(self._live_state, "st", text=f"State: {state}",
           text_color=state_colors.get(state, DIM))

        gm = d.get("game_mode", "")
        _s(self._live_gamemode, "gm", text=gm.upper() if gm else "")

        v = d.get("victories", 0)
        de = d.get("defeats", 0)
        dr = d.get("draws", 0)
        session_victories = d.get("session_victories", v)
        session_defeats = d.get("session_defeats", de)
        session_draws = d.get("session_draws", dr)
        session_total_matches = d.get(
            "total_matches",
            d.get("session_matches", session_victories + session_defeats + session_draws)
        )
        _s(self._live_matches, "mt", text=f"Matches: {session_total_matches}")

        # Farm mode badge
        if self._quest_farm_active:
            remaining = len([b for b in self.brawlers_data if b.get("type") == "quest"])
            _s(self._live_farm_info, "fi", text=f"\U0001F4DC QUEST ({remaining} left)")
        elif d.get("farm_mode", False):
            remaining = d.get("farm_remaining", 0)
            _s(self._live_farm_info, "fi", text=f"\U0001F3C6 FARM ({remaining} left)")
        else:
            _s(self._live_farm_info, "fi", text="")

        # brawler card
        brawler_name = d.get("brawler", "?")
        _s(self._live_brawler, "bn", text=brawler_name.upper())

        playstyle = d.get("playstyle", "")
        _s(self._live_playstyle, "ps",
           text=playstyle.capitalize() if playstyle else "")

        # Win streak / loss streak
        streak = d.get("streak", 0)
        if streak > 0:
            _s(self._live_streak, "sk", text=f"\U0001F525 {streak}W Streak", text_color=GREEN)
        elif streak < 0:
            _s(self._live_streak, "sk", text=f"\u274C {abs(streak)}L Streak", text_color=RED)
        else:
            _s(self._live_streak, "sk", text="")

        # Win / Loss / Draw
        wl_v = session_victories if session_total_matches > 0 else v
        wl_d = session_defeats if session_total_matches > 0 else de
        wl_dr = session_draws if session_total_matches > 0 else dr
        total_games = wl_v + wl_d + wl_dr
        if total_games > 0:
            wr = wl_v / total_games * 100
            _s(self._live_wl, "wl", text=f"W:{wl_v}  L:{wl_d}  D:{wl_dr}", text_color=TXT)
            wr_color = GREEN if wr >= 55 else GOLD if wr >= 45 else RED
            _s(self._live_wr_pct, "wp", text=f"{wr:.0f}% WR", text_color=wr_color)
        else:
            _s(self._live_wl, "wl", text="W:0  L:0  D:0", text_color=DIM)
            _s(self._live_wr_pct, "wp", text="", text_color=DIM)

        # trophies
        trophies = d.get("trophies", "?")
        target = d.get("target", "?")
        _s(self._live_trophy_current, "tc",
           text=str(trophies) if trophies != "?" else "0")
        _s(self._live_trophy_target, "tt",
           text=f"/ {target}" if target != "?" else "/ ?")

        if isinstance(trophies, (int, float)) and isinstance(target, (int, float)) and target > 0:
            pct = min(1.0, max(0, trophies / target))
            self._live_trophy_bar.set(pct)
            pct_i = int(pct * 100)
            pct_color = GREEN if pct_i >= 90 else GOLD if pct_i >= 50 else DIM
            _s(self._live_trophy_pct, "tp", text=f"{pct_i}%", text_color=pct_color)
            _s(self._live_trophy_bar, "tb_pc", progress_color=GREEN if pct_i >= 100 else GOLD)

            remaining = int(target - trophies)
            if remaining > 0:
                _s(self._live_kpi_target_gap, "kpi_gap", text=f"{remaining} left", text_color=GOLD)
            elif remaining == 0:
                _s(self._live_kpi_target_gap, "kpi_gap", text="target reached", text_color=GREEN)
            else:
                _s(self._live_kpi_target_gap, "kpi_gap", text=f"+{abs(remaining)} over", text_color=GREEN)
        else:
            self._live_trophy_bar.set(0)
            _s(self._live_trophy_pct, "tp", text="", text_color=DIM)
            _s(self._live_trophy_bar, "tb_pc", progress_color=GOLD)
            _s(self._live_kpi_target_gap, "kpi_gap", text="—", text_color=DIM)

        last_result = str(d.get("last_result", "") or "").lower()
        last_delta = d.get("last_trophy_delta")
        last_verified = bool(d.get("last_trophy_delta_verified", False))
        streak_bonus = int(d.get("last_streak_bonus", 0) or 0)
        underdog_bonus = int(d.get("last_underdog_bonus", 0) or 0)
        trophy_adjustment = int(d.get("last_trophy_adjustment", 0) or 0)
        delta_text = ""
        delta_detail = ""
        delta_color = DIM

        if last_result in {"victory", "defeat", "draw"}:
            delta_value = int(last_delta or 0)
            delta_sign = "+" if delta_value > 0 else ""
            delta_text = f"{last_result.upper()} {delta_sign}{delta_value}"
            if not last_verified and last_result != "draw":
                delta_text += " est."
            if last_result == "victory":
                delta_color = GREEN
            elif last_result == "defeat":
                delta_color = RED
            else:
                delta_color = GOLD

            detail_parts = []
            if streak_bonus > 0:
                detail_parts.append(f"streak +{streak_bonus}")
            if underdog_bonus > 0:
                detail_parts.append(f"underdog +{underdog_bonus}")
            elif last_verified and trophy_adjustment != 0:
                adjustment_sign = "+" if trophy_adjustment > 0 else ""
                detail_parts.append(f"adj {adjustment_sign}{trophy_adjustment}")
            delta_detail = " | ".join(detail_parts)

        _s(self._live_trophy_delta, "td", text=delta_text, text_color=delta_color)
        _s(
            self._live_trophy_detail,
            "td2",
            text=delta_detail,
            text_color=DIM if delta_detail else delta_color,
        )

        # combat
        php = max(0, min(100, d.get("player_hp", 100)))
        is_dead = d.get("is_dead", False)
        hp_conf_p = d.get("hp_confidence_player", 1.0)

        if is_dead:
            _s(self._live_hp_bar, "hpc", progress_color=RED)
            self._live_hp_bar.set(0)
            _s(self._live_hp_text, "hpt", text="DEAD", text_color=RED)
        else:
            hc = HP_G if php > 60 else HP_Y if php > 30 else HP_R
            _s(self._live_hp_bar, "hpc", progress_color=hc)
            self._live_hp_bar.set(php / 100)
            hp_suffix = "?" if hp_conf_p < 0.4 else ""
            _s(self._live_hp_text, "hpt", text=f"{php}%{hp_suffix}", text_color=hc)

        # Target
        tn = d.get("target_name")
        ne = d.get("n_enemies", 0)
        has_enemy = d.get("enemies", ne) > 0 or ne > 0
        if has_enemy and tn:
            _s(self._live_target, "tg", text=f"\u25B6 {tn.upper()}", text_color=ACCENT)
        elif has_enemy:
            _s(self._live_target, "tg", text="\u25B6 ENEMY", text_color=RED)
        else:
            _s(self._live_target, "tg", text="\u25B6 No target", text_color=DIM)

        # Enemy HP
        ehp = d.get("target_hp", d.get("enemy_hp", -1))
        hp_conf_e = d.get("hp_confidence_enemy", 1.0)
        if ehp >= 0 and has_enemy:
            ec = HP_G if ehp > 60 else HP_Y if ehp > 30 else HP_R
            _s(self._live_ehp_bar, "ehc", progress_color=ec)
            self._live_ehp_bar.set(ehp / 100)
            ehp_suffix = "?" if hp_conf_e < 0.4 else ""
            _s(self._live_ehp_text, "eht", text=f"{ehp}%{ehp_suffix}", text_color=ec)
        elif has_enemy:
            # Enemy visible but HP unknown - show scanning indicator
            self._live_ehp_bar.set(0)
            _s(self._live_ehp_text, "eht", text="scanning...", text_color=DIM)
        else:
            self._live_ehp_bar.set(0)
            _s(self._live_ehp_text, "eht", text="\u2014", text_color=DIM)

        # Movement / Foes / Walls (merged into one label)
        mv = d.get("movement", "?")
        fc = d.get("enemies", ne)
        wc = d.get("walls", 0)
        _s(self._live_movement, "mvt",
           text=f"Move: {mv}  |  Foes: {fc}  |  Walls: {wc}")

        # Abilities (badge style)
        g = d.get("gadget_ready", False)
        su = d.get("super_ready", False)
        hy = d.get("hypercharge_ready", False)
        _s(self._live_gad, "ga", text=" GAD ",
           text_color=PANEL if g else DIM, fg_color=GREEN if g else PANEL)
        _s(self._live_sup, "sp", text=" SUP ",
           text_color=PANEL if su else DIM, fg_color=BLUE if su else PANEL)
        _s(self._live_hyp, "hp_", text=" HYP ",
           text_color=PANEL if hy else DIM, fg_color=PURPLE if hy else PANEL)

        # Decision
        dec = d.get("decision", "")
        if dec:
            dc = (GREEN if "ATTACK" in dec else RED if "RETREAT" in dec
                  else CYAN if "HUNT" in dec else GOLD if "REGROUP" in dec
                  else BRIGHT)
            _s(self._live_decision, "dc", text=f"\u25B6 {dec[:50]}", text_color=dc)
        else:
            _s(self._live_decision, "dc", text="")

        # Ammo dots
        ammo = d.get("ammo", 3)
        for i, dot in enumerate(self._live_ammo_dots):
            if i < ammo:
                _s(dot, f"ad{i}", text_color=CYAN)
            else:
                _s(dot, f"ad{i}", text_color=SECTION)
        ac = CYAN if ammo >= 2 else HP_Y if ammo == 1 else RED
        _s(self._live_ammo_lbl, "al", text=f"{ammo}/3", text_color=ac)

        # Score
        our = d.get("our_score", 0)
        their = d.get("their_score", 0)
        sd = d.get("score_diff", 0)
        if our or their:
            sc_c = GREEN if sd > 0 else RED if sd < 0 else DIM
            _s(self._live_score, "sc",
               text=f"Score  {our} \u2014 {their}  ({'+' if sd > 0 else ''}{sd})",
               text_color=sc_c)
        else:
            _s(self._live_score, "sc", text="")

        deaths = d.get("deaths", 0)
        if deaths:
            _s(self._live_deaths, "dth", text=f"\u2620 Deaths: {deaths}")
        else:
            _s(self._live_deaths, "dth", text="")

        # performance cards
        ck = d.get("current_kills", 0)
        cd = d.get("current_deaths", 0)
        ca = d.get("current_assists", 0)
        cdmg = d.get("current_damage", 0)
        match_active = bool(d.get("match_active", False))
        perf_source = str(d.get("perf_source", "INIT") or "INIT")
        tk_ = d.get("total_kills", 0)
        ta = d.get("total_assists", 0)
        tdmg = d.get("total_damage", 0)
        lk = d.get("last_kills", 0)
        la = d.get("last_assists", 0)
        ld_ = d.get("last_damage", 0)
        ak = d.get("avg_kills", 0)
        aa = d.get("avg_assists", 0)
        ad_ = d.get("avg_damage", 0)

        # Fallback for dashboard startup / delayed live counters:
        # use persisted training stats so bottom overview is informative
        # even before the first end-screen parse updates session stats.
        fallback_used = False
        if tk_ <= 0 and ta <= 0 and tdmg <= 0:
            stats_snapshot = self._load_training_stats_snapshot()
            if stats_snapshot:
                combat_s = stats_snapshot.get("combat", {}) if isinstance(stats_snapshot.get("combat", {}), dict) else {}
                brawler_s_all = stats_snapshot.get("brawler_stats", {}) if isinstance(stats_snapshot.get("brawler_stats", {}), dict) else {}
                bname_key = str(d.get("brawler", "") or "").strip().lower()
                brawler_s = brawler_s_all.get(bname_key, {}) if bname_key else {}

                tk_ = int(combat_s.get("total_kills", tk_) or tk_)
                tdmg = int(combat_s.get("total_damage_dealt", tdmg) or tdmg)
                # Assists are often unavailable in persisted stats; keep live value if present.
                ta = int(ta or 0)

                # Average values: prefer current brawler averages when available.
                ak = float(brawler_s.get("kills", ak) or ak) if isinstance(brawler_s, dict) and brawler_s.get("matches", 0) in (0, 1) else float(brawler_s.get("avg_kills", ak) or ak)
                ad_ = float(brawler_s.get("avg_damage", ad_) or ad_)
                aa = float(brawler_s.get("avg_assists", aa) or aa) if isinstance(brawler_s, dict) else aa

                if tk_ > 0 or tdmg > 0 or ak > 0 or ad_ > 0:
                    fallback_used = True

        rl_training_enabled = d.get("rl_training_enabled", False)
        rl_total_episodes = d.get("rl_total_episodes", 0)
        rl_total_updates = d.get("rl_total_updates", 0)
        rl_buffer_size = d.get("rl_buffer_size", 0)
        rl_buffer_capacity = d.get("rl_buffer_capacity", 0)
        rl_episode_reward = d.get("rl_episode_reward", 0.0)
        rl_kills = d.get("rl_kills", 0)
        rl_deaths = d.get("rl_deaths", 0)
        rl_damage_dealt = d.get("rl_damage_dealt", 0)
        rl_damage_taken = d.get("rl_damage_taken", 0)
        rl_hit_rate = d.get("rl_hit_rate", -1)

        # THIS MATCH card (always update - shows live in-match kills)
        ck_color = GREEN if ck > 0 else RED
        cdmg_color = GOLD if cdmg > 0 else DIM
        _s(self._live_perf_current["kills"], "ck_k", text=f"{ck} Kills", text_color=ck_color)
        _s(self._live_perf_current["assists"], "ck_a", text=f"{ca} Assists", text_color=BLUE)
        _s(self._live_perf_current["damage"], "ck_d", text=f"{cdmg:,} Damage", text_color=cdmg_color)
        if "deaths" in self._live_perf_current:
            _s(self._live_perf_current["deaths"], "ck_dt", text=f"{cd} Deaths", text_color=RED if cd > 0 else DIM)

        if match_active and ck == 0 and ca == 0 and cdmg == 0:
            _s(self._live_perf_current["assists"], "ck_a", text=f"Collecting... [{perf_source}]", text_color=DIM)

        kda = (ck + ca) / max(1, cd)
        kda_color = GREEN if kda >= 2.0 else GOLD if kda >= 1.0 else RED
        _s(self._live_kpi_kda, "kpi_kda", text=f"{kda:.2f}", text_color=kda_color)

        minutes = max(1e-6, elapsed / 60.0)
        dpm = int(cdmg / minutes)
        dpm_color = GREEN if dpm >= 15000 else GOLD if dpm >= 7000 else DIM
        _s(self._live_kpi_dpm, "kpi_dpm", text=f"{dpm:,}", text_color=dpm_color)

        wins_per_hour_raw = session_victories / max(1e-6, elapsed / 3600.0)
        if self._wins_per_hour_ema is None:
            self._wins_per_hour_ema = wins_per_hour_raw
        else:
            self._wins_per_hour_ema += 0.18 * (wins_per_hour_raw - self._wins_per_hour_ema)
        wins_per_hour = max(0.0, self._wins_per_hour_ema)
        wp_color = GREEN if wins_per_hour >= 8 else GOLD if wins_per_hour >= 4 else DIM
        _s(self._live_kpi_win_pace, "kpi_wp", text=self._fmt_compact_rate(wins_per_hour), text_color=wp_color)

        hit_rate_text = "--"
        if isinstance(rl_hit_rate, (int, float)) and rl_hit_rate >= 0:
            hit_rate_text = f"{rl_hit_rate * 100:.0f}%"
        buffer_text = f"{rl_buffer_size}/{rl_buffer_capacity}" if rl_buffer_capacity > 0 else str(rl_buffer_size)
        rl_mode = "ON" if rl_training_enabled else "OFF"
        summary_mode = "LIVE" if match_active else "SESSION"
        fallback_tag = " +TS" if fallback_used else ""
        if hasattr(self, "_live_rl_info"):
            _s(
                self._live_rl_info,
                "rl_info",
                text=(
                    f"Overview [{summary_mode}{fallback_tag}]  Source:{perf_source}  Match K/D/A: {ck}/{cd}/{ca}  Match Dmg:{cdmg:,}\n"
                    f"RL [{rl_mode}]  Ep:{rl_total_episodes}  Upd:{rl_total_updates}  Buf:{buffer_text}  Reward:{rl_episode_reward:.2f}"
                    f"  RL K/D:{rl_kills}/{rl_deaths}  RL Dmg:{rl_damage_dealt:,}/{rl_damage_taken:,}  Hit:{hit_rate_text}"
                ),
                text_color=ACCENT if rl_training_enabled else DIM,
            )

        if tk_ > 0 or tdmg > 0 or total_games > 0 or fallback_used:
            _s(self._live_perf_last["kills"], "lk_k", text=f"{lk} Kills")
            _s(self._live_perf_last["assists"], "lk_a", text=f"{la} Assists")
            _s(self._live_perf_last["damage"], "lk_d", text=f"{ld_:,} Damage")
            _s(self._live_perf_total["kills"], "tk_k", text=f"{tk_} Kills")
            _s(self._live_perf_total["assists"], "tk_a", text=f"{ta} Assists")
            _s(self._live_perf_total["damage"], "tk_d", text=f"{tdmg:,} Damage")
            _s(self._live_perf_avg["kills"], "ak_k",
               text=f"{ak:.1f} Kills" if isinstance(ak, float) else f"{ak} Kills")
            _s(self._live_perf_avg["assists"], "ak_a",
               text=f"{aa:.1f} Assists" if isinstance(aa, float) else f"{aa} Assists")
            _s(self._live_perf_avg["damage"], "ak_d",
               text=f"{ad_:,.0f} Damage" if isinstance(ad_, float) else f"{ad_:,} Damage")
        else:
            for pfx, card in [("lk", self._live_perf_last),
                               ("tk", self._live_perf_total),
                               ("ak", self._live_perf_avg)]:
                _s(card["kills"], f"{pfx}_k", text="0 Kills")
                _s(card["assists"], f"{pfx}_a", text="0 Assists")
                _s(card["damage"], f"{pfx}_d", text="0 Damage")

    # --- bOT CONTROL ---

    def _toggle_bot(self):
        if self.bot_running:
            self._stop_bot()
        else:
            self._start_bot()

    def _start_bot(self):
        if not self.brawlers_data:
            # Flash warning
            self._status_label.configure(
                text="\u26A0 Select a brawler first!", text_color=RED)
            self.show_page("brawler")
            return

        if not self._logged_in and api_base_url != "localhost":
            self._status_label.configure(
                text="\u26A0 Login required!", text_color=RED)
            self._show_login_dialog()
            return

        # Save settings first
        self._save_all_settings()

        # Save brawler data for the bot
        save_brawler_data(self.brawlers_data)

        self._prepare_bot_control_events()

        self.bot_running = True
        self._update_status_display()
        self.show_page("live")

        # Start bot in background thread
        self.bot_thread = threading.Thread(
            target=self._run_bot, daemon=True)
        self.bot_thread.start()

    def _run_bot(self):
        """Run pyla_main in a background thread."""
        try:
            if self._bot_stop_requested:
                return
            # Register this dashboard as the active one for live updates
            # Use __main__ since main.py runs as the entry point script
            import sys
            main_module = sys.modules.get('__main__')
            if main_module and hasattr(main_module, '_active_dashboard'):
                main_module._active_dashboard = self
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
        except Exception as e:
            print(f"[DASHBOARD] Bot thread error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.bot_running = False
            try:
                import sys
                main_module = sys.modules.get('__main__')
                if main_module:
                    if hasattr(main_module, '_active_dashboard'):
                        main_module._active_dashboard = None
                    if hasattr(main_module, '_active_stage_manager'):
                        main_module._active_stage_manager = None
            except Exception:
                pass

    def _prepare_bot_control_events(self):
        """Create fresh control events before launching the bot thread."""
        self._bot_stop_requested = False
        self._bot_stop_event = threading.Event()
        self._bot_pause_event = threading.Event()
        self._live_session_start_ts = None
        self._wins_per_hour_ema = None

    @staticmethod
    def _fmt_compact_rate(value):
        """Human-friendly compact format for KPI rates."""
        if value >= 1000:
            return f"{value / 1000:.1f}k"
        if value >= 100:
            return f"{value:.0f}"
        return f"{value:.1f}"

    def _stop_bot(self):
        """Signal the bot to stop via its _stop_event."""
        self._bot_stop_requested = True
        self.bot_running = False
        self._update_status_display()
        # Signal the bot's internal stop event
        if hasattr(self, '_bot_stop_event') and self._bot_stop_event:
            self._bot_stop_event.set()
        if hasattr(self, '_bot_pause_event') and self._bot_pause_event:
            self._bot_pause_event.clear()
        # Don't clear live data yet - summary popup will read it
        # self._live_data.clear() is done after summary is shown

    # --- sESSION SUMMARY POPUP ---

    def _show_session_summary(self):
        """Show a session summary popup window after bot stops."""
        s = getattr(self, '_session_summary', None)
        if not s:
            self._live_data.clear()
            return

        win = ctk.CTkToplevel(self)
        win.title("\u2605 Session Summary")
        w, h = S(650), S(580)
        win.geometry(f"{w}x{h}")
        win.configure(fg_color=BG)
        win.resizable(True, True)
        win.attributes("-topmost", True)
        win.after(200, lambda: win.attributes("-topmost", False))
        win.grab_set()

        # scrollable content
        scroll = ctk.CTkScrollableFrame(win, fg_color=BG,
                                        scrollbar_button_color=ACCENT)
        scroll.pack(fill="both", expand=True, padx=S(8), pady=S(8))

        # title
        ctk.CTkLabel(scroll, text="\u2605  SESSION SUMMARY  \u2605",
                     font=("Segoe UI", S(22), "bold"),
                     text_color=ACCENT).pack(pady=(S(10), S(16)))

        # overview card
        ov = ctk.CTkFrame(scroll, fg_color=PANEL, corner_radius=S(10))
        ov.pack(fill="x", padx=S(12), pady=S(4))
        row0 = ctk.CTkFrame(ov, fg_color="transparent")
        row0.pack(fill="x", padx=S(16), pady=S(10))

        for col, (label, value) in enumerate([
            ("Duration", s['duration']),
            ("Matches", str(s['total_matches'])),
            ("Win Rate", f"{s['winrate']:.0f}%"),
        ]):
            f = ctk.CTkFrame(row0, fg_color="transparent")
            f.pack(side="left", expand=True)
            ctk.CTkLabel(f, text=label, font=("Segoe UI", S(11)),
                         text_color=DIM).pack()
            vc = BRIGHT
            if label == "Win Rate":
                vc = GREEN if s['winrate'] >= 55 else GOLD if s['winrate'] >= 45 else RED
            ctk.CTkLabel(f, text=value, font=("Segoe UI", S(18), "bold"),
                         text_color=vc).pack()

        # W/L/D row
        row1 = ctk.CTkFrame(ov, fg_color="transparent")
        row1.pack(fill="x", padx=S(16), pady=(0, S(6)))
        for label, val, color in [
            ("Victories", s['victories'], GREEN),
            ("Defeats", s['defeats'], RED),
            ("Draws", s['draws'], DIM),
        ]:
            f = ctk.CTkFrame(row1, fg_color="transparent")
            f.pack(side="left", expand=True)
            ctk.CTkLabel(f, text=label, font=("Segoe UI", S(10)),
                         text_color=DIM).pack()
            ctk.CTkLabel(f, text=str(val), font=("Segoe UI", S(16), "bold"),
                         text_color=color).pack()

        # trophy total card
        tc = ctk.CTkFrame(scroll, fg_color=PANEL, corner_radius=S(10))
        tc.pack(fill="x", padx=S(12), pady=S(6))
        net = s['net_trophies']
        sign = "+" if net >= 0 else ""
        t_color = GREEN if net > 0 else RED if net < 0 else DIM
        trophy_row = ctk.CTkFrame(tc, fg_color="transparent")
        trophy_row.pack(pady=S(10))
        ctk.CTkLabel(trophy_row, text="\U0001F3C6",
                     font=("Segoe UI", S(28))).pack(side="left", padx=(S(12), S(6)))
        ctk.CTkLabel(trophy_row, text=f"{sign}{net} Trophies",
                     font=("Segoe UI", S(22), "bold"),
                     text_color=t_color).pack(side="left", padx=(0, S(12)))

        # performance row
        tk_ = s.get('total_kills', 0)
        ta = s.get('total_assists', 0)
        td = s.get('total_damage', 0)
        if tk_ or ta or td:
            pf = ctk.CTkFrame(scroll, fg_color=PANEL, corner_radius=S(10))
            pf.pack(fill="x", padx=S(12), pady=S(4))
            pr = ctk.CTkFrame(pf, fg_color="transparent")
            pr.pack(fill="x", padx=S(16), pady=S(10))
            for label, val, color in [
                ("Kills", str(tk_), ACCENT),
                ("Assists", str(ta), CYAN),
                ("Damage", f"{td:,}", BLUE),
            ]:
                f = ctk.CTkFrame(pr, fg_color="transparent")
                f.pack(side="left", expand=True)
                ctk.CTkLabel(f, text=label, font=("Segoe UI", S(10)),
                             text_color=DIM).pack()
                ctk.CTkLabel(f, text=val, font=("Segoe UI", S(16), "bold"),
                             text_color=color).pack()

        # per-brawler breakdown
        brawlers = s.get('brawlers', [])
        if brawlers:
            ctk.CTkLabel(scroll, text="BRAWLER BREAKDOWN",
                         font=("Segoe UI", S(14), "bold"),
                         text_color=ACCENT).pack(pady=(S(12), S(4)))

            # Header
            hdr = ctk.CTkFrame(scroll, fg_color=SECTION, corner_radius=S(6))
            hdr.pack(fill="x", padx=S(12), pady=S(2))
            cols = ctk.CTkFrame(hdr, fg_color="transparent")
            cols.pack(fill="x", padx=S(8), pady=S(6))
            for text, w_frac in [("Brawler", 0.25), ("W/L/D", 0.18),
                                  ("WR", 0.12), ("Trophies", 0.22), ("\u0394", 0.16)]:
                ctk.CTkLabel(cols, text=text, font=("Segoe UI", S(11), "bold"),
                             text_color=DIM, width=int(S(550) * w_frac),
                             anchor="w").pack(side="left")

            for b in brawlers:
                row = ctk.CTkFrame(scroll, fg_color=PANEL, corner_radius=S(6))
                row.pack(fill="x", padx=S(12), pady=S(1))
                rc = ctk.CTkFrame(row, fg_color="transparent")
                rc.pack(fill="x", padx=S(8), pady=S(5))

                delta = b['trophy_delta']
                d_sign = "+" if delta >= 0 else ""
                d_color = GREEN if delta > 0 else RED if delta < 0 else DIM
                wr_b = f"{b['winrate']:.0f}%"
                wr_color = GREEN if b['winrate'] >= 55 else GOLD if b['winrate'] >= 45 else RED

                for text, w_frac, color in [
                    (b['name'].title(), 0.25, BRIGHT),
                    (f"{b['victories']}/{b['defeats']}/{b['draws']}", 0.18, TXT),
                    (wr_b, 0.12, wr_color),
                    (f"{b['trophy_start']}\u2192{b['trophy_end']}", 0.22, TXT),
                    (f"{d_sign}{delta}", 0.16, d_color),
                ]:
                    ctk.CTkLabel(rc, text=text, font=("Segoe UI", S(12)),
                                 text_color=color, width=int(S(550) * w_frac),
                                 anchor="w").pack(side="left")

        # close button
        ctk.CTkButton(scroll, text="Close", font=("Segoe UI", S(14), "bold"),
                      fg_color=ACCENT, hover_color=ACCENT_H, text_color=BRIGHT,
                      height=S(40), corner_radius=S(8), width=S(160),
                      command=lambda: self._close_summary(win)).pack(pady=S(16))

    def _close_summary(self, win):
        """Close summary popup and clean up."""
        try:
            win.grab_release()
            win.destroy()
        except Exception:
            pass
        self._session_summary = None
        self._live_data.clear()

    # --- lIFECYCLE ---

    def _on_close(self):
        self._live_running = False
        self.bot_running = False
        # Signal bot thread to stop
        if self._bot_stop_event:
            self._bot_stop_event.set()
        try:
            self.destroy()
        except Exception:
            pass

    def run(self):
        """Start the dashboard main loop."""
        self.after(500, self._tick)
        self.mainloop()
