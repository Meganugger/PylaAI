import json
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
from PIL import Image
from customtkinter import CTkImage

from brawlstars_api import fetch_player_profile
from gui.config_store import load_config, save_config
from gui.theme import COLORS, S, apply_appearance, center_window, font
from utils import load_toml_as_dict, save_brawler_icon

debug = load_toml_as_dict("cfg/general_config.toml")["super_debug"] == "yes"
pyla_version = load_toml_as_dict("./cfg/general_config.toml")["pyla_version"]


class SelectBrawler:
    def __init__(self, data_setter=None, brawlers=None):
        apply_appearance()
        self.app = ctk.CTk()
        self.app.title(f"PylaAI Setup v{pyla_version}")
        app_width = S(1380)
        app_height = S(880)
        self.app.geometry(f"{app_width}x{app_height}")
        self.app.minsize(S(1180), S(760))
        self.app.configure(fg_color=COLORS["bg"])
        center_window(self.app, app_width, app_height)

        self.data_setter = data_setter
        self.brawlers = brawlers or []
        self.brawlers_data = []
        self.result_data = None
        self.images = []
        self.image_lookup = {}
        self.brawler_buttons = []
        self.roster_cards = {}
        self.search_update_after_id = None
        self.general_config = load_config("general")
        self.browser_scroll_handler = None
        self.roster_scroll_handler = None
        self._closing = False
        self.player_profile_name = ""
        self.player_brawler_stats = {}
        self.api_key_var = tk.StringVar(value=self.general_config.get("brawlstars_api_key", ""))
        self.player_tag_var = tk.StringVar(value=self.general_config.get("brawlstars_player_tag", ""))
        self.auto_push_target_var = tk.StringVar(value=str(self.general_config.get("auto_push_target_trophies", 1000)))
        self.api_status_var = tk.StringVar(value="Connect the official Brawl Stars API to sync trophies and build an auto-push roster.")

        self.filter_var = tk.StringVar()
        self.timer_var = tk.StringVar(value=str(self.general_config.get("run_for_minutes", 600)))
        self.filter_var.trace_add("write", lambda *_: self._schedule_image_refresh())
        self.app.protocol("WM_DELETE_WINDOW", self._request_close)

        self._load_images()
        self._build_layout()
        self.update_images("")
        self.refresh_roster_summary()

        try:
            self.app.mainloop()
        finally:
            self._finalize_window()

    def _load_images(self):
        icon_size = (S(82), S(82))
        roster_icon_size = (S(42), S(42))
        for brawler in self.brawlers:
            img_path = f"./api/assets/brawler_icons/{brawler}.png"
            try:
                img = Image.open(img_path)
            except FileNotFoundError:
                save_brawler_icon(brawler)
                img = Image.open(img_path)

            grid_image = CTkImage(light_image=img, dark_image=img, size=icon_size)
            roster_image = CTkImage(light_image=img, dark_image=img, size=roster_icon_size)
            self.images.append((brawler, grid_image))
            self.image_lookup[brawler] = roster_image

    def _build_layout(self):
        root = ctk.CTkFrame(self.app, fg_color="transparent")
        root.pack(fill="both", expand=True, padx=S(24), pady=S(24))
        root.grid_columnconfigure(0, weight=5)
        root.grid_columnconfigure(1, weight=4)
        root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(
            root,
            fg_color=COLORS["surface"],
            corner_radius=S(16),
            border_width=1,
            border_color=COLORS["border"]
        )
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, S(16)))
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            header,
            text="Brawler Setup",
            font=font(28, "bold"),
            text_color=COLORS["text"]
        )
        title.grid(row=0, column=0, sticky="w", padx=S(20), pady=(S(18), S(4)))

        subtitle = ctk.CTkLabel(
            header,
            text="Search a brawler, configure its goal, and review the full roster before starting the bot.",
            font=font(14),
            text_color=COLORS["muted"]
        )
        subtitle.grid(row=1, column=0, sticky="w", padx=S(20), pady=(0, S(18)))

        self.left_panel = ctk.CTkFrame(
            root,
            fg_color=COLORS["surface"],
            corner_radius=S(16),
            border_width=1,
            border_color=COLORS["border"]
        )
        self.left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, S(10)))
        self.left_panel.grid_columnconfigure(0, weight=1)
        self.left_panel.grid_rowconfigure(2, weight=1)

        self.right_panel = ctk.CTkFrame(
            root,
            fg_color=COLORS["surface"],
            corner_radius=S(16),
            border_width=1,
            border_color=COLORS["border"]
        )
        self.right_panel.grid(row=1, column=1, sticky="nsew", padx=(S(10), 0))
        self.right_panel.grid_columnconfigure(0, weight=1)
        self.right_panel.grid_rowconfigure(3, weight=1)

        self._build_brawler_browser()
        self._build_roster_panel()

    def _build_brawler_browser(self):
        browser_header = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        browser_header.grid(row=0, column=0, sticky="ew", padx=S(18), pady=(S(18), S(12)))
        browser_header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            browser_header,
            text="Choose a Brawler",
            font=font(20, "bold"),
            text_color=COLORS["text"]
        )
        title.grid(row=0, column=0, sticky="w")

        helper = ctk.CTkLabel(
            browser_header,
            text="Click any card to set its goal and current progress.",
            font=font(13),
            text_color=COLORS["muted"]
        )
        helper.grid(row=1, column=0, sticky="w", pady=(S(4), 0))

        self.filter_entry = ctk.CTkEntry(
            self.left_panel,
            textvariable=self.filter_var,
            placeholder_text="Search brawler names...",
            font=font(15),
            height=S(46),
            fg_color=COLORS["surface_alt_2"],
            border_color=COLORS["border"],
            text_color=COLORS["text"]
        )
        self.filter_entry.grid(row=1, column=0, sticky="ew", padx=S(18), pady=(0, S(12)))

        self.image_frame = ctk.CTkScrollableFrame(
            self.left_panel,
            fg_color="transparent",
            corner_radius=S(12)
        )
        self.image_frame.grid(row=2, column=0, sticky="nsew", padx=S(12), pady=(0, S(12)))
        self.browser_scroll_handler = self._make_scroll_handler(self.image_frame, lines_per_notch=20)
        self._bind_scroll_handler_recursive(self.image_frame, self.browser_scroll_handler)
        self._create_brawler_buttons()
        self.image_frame.bind("<Configure>", lambda _event: self.update_images(self.filter_var.get()), add="+")

    def _build_roster_panel(self):
        roster_header = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        roster_header.grid(row=0, column=0, sticky="ew", padx=S(18), pady=(S(18), S(10)))
        roster_header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(
            roster_header,
            text="Selected Roster",
            font=font(20, "bold"),
            text_color=COLORS["text"]
        )
        title.grid(row=0, column=0, sticky="w")

        self.roster_count_label = ctk.CTkLabel(
            roster_header,
            text="0 configured",
            font=font(13),
            text_color=COLORS["muted"]
        )
        self.roster_count_label.grid(row=1, column=0, sticky="w", pady=(S(4), 0))

        self._build_api_tools_card()

        self.roster_empty_label = ctk.CTkLabel(
            self.right_panel,
            text="No brawlers configured yet. Select one from the left to get started.",
            font=font(14),
            text_color=COLORS["muted"],
            wraplength=S(420),
            justify="left"
        )
        self.roster_empty_label.grid(row=2, column=0, sticky="ew", padx=S(18), pady=(0, S(10)))

        self.roster_frame = ctk.CTkScrollableFrame(
            self.right_panel,
            fg_color="transparent",
            corner_radius=S(12)
        )
        self.roster_frame.grid(row=3, column=0, sticky="nsew", padx=S(12), pady=(0, S(12)))
        self.roster_scroll_handler = self._make_scroll_handler(self.roster_frame, lines_per_notch=12)
        self._bind_scroll_handler_recursive(self.roster_frame, self.roster_scroll_handler)

        footer = ctk.CTkFrame(
            self.right_panel,
            fg_color=COLORS["surface_alt_2"],
            corner_radius=S(14),
            border_width=1,
            border_color=COLORS["border"]
        )
        footer.grid(row=4, column=0, sticky="ew", padx=S(12), pady=(0, S(12)))
        footer.grid_columnconfigure(0, weight=1)
        footer.grid_columnconfigure(1, weight=0)

        runtime_label = ctk.CTkLabel(
            footer,
            text="Run Time Limit (Minutes)",
            font=font(13, "bold"),
            text_color=COLORS["text"]
        )
        runtime_label.grid(row=0, column=0, sticky="w", padx=S(16), pady=(S(14), S(4)))

        runtime_helper = ctk.CTkLabel(
            footer,
            text="Use 0 to leave runtime uncapped.",
            font=font(12),
            text_color=COLORS["muted"]
        )
        runtime_helper.grid(row=1, column=0, sticky="w", padx=S(16), pady=(0, S(12)))

        self.timer_entry = ctk.CTkEntry(
            footer,
            textvariable=self.timer_var,
            width=S(110),
            height=S(42),
            font=font(15),
            fg_color=COLORS["surface"],
            border_color=COLORS["border"],
            text_color=COLORS["text"]
        )
        self.timer_entry.grid(row=0, column=1, rowspan=2, padx=S(16), pady=S(14), sticky="e")
        self.timer_entry.bind("<FocusOut>", self._save_runtime_limit)
        self.timer_entry.bind("<Return>", self._save_runtime_limit)

        actions = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        actions.grid(row=5, column=0, sticky="ew", padx=S(12), pady=(0, S(12)))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=2)

        self.load_button = ctk.CTkButton(
            actions,
            text="Load Saved Roster",
            command=self.load_brawler_config,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            corner_radius=S(10),
            font=font(15, "bold"),
            height=S(44)
        )
        self.load_button.grid(row=0, column=0, sticky="ew", padx=(0, S(8)))

        self.start_button = ctk.CTkButton(
            actions,
            text="Start Bot with This Roster",
            command=self.start_bot,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["text"],
            corner_radius=S(10),
            font=font(16, "bold"),
            height=S(50)
        )
        self.start_button.grid(row=0, column=1, sticky="ew", padx=(S(8), 0))

    def _build_api_tools_card(self):
        api_card = ctk.CTkFrame(
            self.right_panel,
            fg_color=COLORS["surface_alt_2"],
            corner_radius=S(14),
            border_width=1,
            border_color=COLORS["border"]
        )
        api_card.grid(row=1, column=0, sticky="ew", padx=S(12), pady=(0, S(12)))
        api_card.grid_columnconfigure(0, weight=1)
        api_card.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            api_card,
            text="Brawl Stars API Sync",
            font=font(16, "bold"),
            text_color=COLORS["text"]
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=S(16), pady=(S(14), S(4)))

        ctk.CTkLabel(
            api_card,
            text="Import your live trophy data and build a lowest-trophies-first push roster automatically.",
            font=font(12),
            text_color=COLORS["muted"],
            wraplength=S(420),
            justify="left"
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=S(16), pady=(0, S(12)))

        self.api_key_entry = ctk.CTkEntry(
            api_card,
            textvariable=self.api_key_var,
            placeholder_text="Official Brawl Stars API key",
            font=font(14),
            height=S(40),
            fg_color=COLORS["surface"],
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            show="*"
        )
        ctk.CTkLabel(
            api_card,
            text="API Key",
            font=font(12, "bold"),
            text_color=COLORS["text"]
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=S(16), pady=(0, S(6)))
        self.api_key_entry.grid(row=3, column=0, columnspan=2, sticky="ew", padx=S(16), pady=(0, S(10)))
        self.api_key_entry.bind("<FocusOut>", self._save_brawlstars_settings)

        self.player_tag_entry = ctk.CTkEntry(
            api_card,
            textvariable=self.player_tag_var,
            placeholder_text="Player tag, for example #ABC123",
            font=font(14),
            height=S(40),
            fg_color=COLORS["surface"],
            border_color=COLORS["border"],
            text_color=COLORS["text"]
        )
        ctk.CTkLabel(
            api_card,
            text="Player Tag",
            font=font(12, "bold"),
            text_color=COLORS["text"]
        ).grid(row=4, column=0, sticky="w", padx=(S(16), S(8)), pady=(0, S(6)))
        self.player_tag_entry.grid(row=5, column=0, sticky="ew", padx=(S(16), S(8)), pady=(0, S(10)))
        self.player_tag_entry.bind("<FocusOut>", self._save_brawlstars_settings)

        self.auto_push_target_entry = ctk.CTkEntry(
            api_card,
            textvariable=self.auto_push_target_var,
            placeholder_text="Target trophies, for example 1000",
            font=font(14),
            height=S(40),
            fg_color=COLORS["surface"],
            border_color=COLORS["border"],
            text_color=COLORS["text"]
        )
        ctk.CTkLabel(
            api_card,
            text="Target Trophies",
            font=font(12, "bold"),
            text_color=COLORS["text"]
        ).grid(row=4, column=1, sticky="w", padx=(S(8), S(16)), pady=(0, S(6)))
        self.auto_push_target_entry.grid(row=5, column=1, sticky="ew", padx=(S(8), S(16)), pady=(0, S(10)))
        self.auto_push_target_entry.bind("<FocusOut>", self._save_brawlstars_settings)

        import_button = ctk.CTkButton(
            api_card,
            text="Import Trophies",
            command=self.import_brawlstars_trophies,
            fg_color=COLORS["surface"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            corner_radius=S(10),
            font=font(14, "bold"),
            height=S(42)
        )
        import_button.grid(row=6, column=0, sticky="ew", padx=(S(16), S(8)), pady=(0, S(12)))

        auto_push_button = ctk.CTkButton(
            api_card,
            text="Build Auto Push",
            command=self.build_auto_push_roster,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["text"],
            corner_radius=S(10),
            font=font(14, "bold"),
            height=S(42)
        )
        auto_push_button.grid(row=6, column=1, sticky="ew", padx=(S(8), S(16)), pady=(0, S(12)))

        self.api_status_label = ctk.CTkLabel(
            api_card,
            textvariable=self.api_status_var,
            font=font(12),
            text_color=COLORS["muted"],
            wraplength=S(420),
            justify="left"
        )
        self.api_status_label.grid(row=7, column=0, columnspan=2, sticky="w", padx=S(16), pady=(0, S(14)))

    @staticmethod
    def _make_scroll_handler(scrollable_frame, lines_per_notch=4):
        canvas = scrollable_frame._parent_canvas

        def on_mousewheel(event):
            delta_steps = int(abs(event.delta) / 120) or 1
            scroll_units = delta_steps * lines_per_notch
            direction = -1 if event.delta > 0 else 1
            canvas.yview_scroll(direction * scroll_units, "units")
            return "break"

        return on_mousewheel

    @staticmethod
    def _bind_scroll_handler(widget, scroll_handler):
        widget.bind("<MouseWheel>", scroll_handler, add="+")

    def _bind_scroll_handler_recursive(self, widget, scroll_handler):
        self._bind_scroll_handler(widget, scroll_handler)
        for child in widget.winfo_children():
            self._bind_scroll_handler_recursive(child, scroll_handler)

    def _schedule_image_refresh(self):
        if self.search_update_after_id is not None:
            self.app.after_cancel(self.search_update_after_id)
        self.search_update_after_id = self.app.after(60, self._apply_image_filter)

    def _request_close(self):
        if self._closing:
            return
        self._closing = True
        if self.search_update_after_id is not None:
            try:
                self.app.after_cancel(self.search_update_after_id)
            except Exception:
                pass
            self.search_update_after_id = None
        try:
            self.app.quit()
        except Exception:
            pass

    def _finalize_window(self):
        if self.search_update_after_id is not None:
            try:
                self.app.after_cancel(self.search_update_after_id)
            except Exception:
                pass
            self.search_update_after_id = None
        if self.app.winfo_exists():
            try:
                self.app.destroy()
            except Exception:
                pass

    def _set_api_status(self, text, tone="muted"):
        color_map = {
            "muted": COLORS["muted"],
            "success": COLORS["success"],
            "danger": COLORS["danger"],
            "warning": COLORS["warning"],
        }
        self.api_status_var.set(text)
        if hasattr(self, "api_status_label"):
            self.api_status_label.configure(text_color=color_map.get(tone, COLORS["muted"]))

    def _save_brawlstars_settings(self, _event=None):
        self.general_config["brawlstars_api_key"] = self.api_key_var.get().strip()
        tag_value = self.player_tag_var.get().strip().upper().replace(" ", "")
        if tag_value and not tag_value.startswith("#"):
            tag_value = f"#{tag_value}"
        self.player_tag_var.set(tag_value)
        self.general_config["brawlstars_player_tag"] = tag_value

        try:
            target_value = int(self.auto_push_target_var.get().strip())
        except ValueError:
            target_value = int(self.general_config.get("auto_push_target_trophies", 1000))
        target_value = max(target_value, 1)
        self.general_config["auto_push_target_trophies"] = target_value
        self.auto_push_target_var.set(str(target_value))
        self.general_config = save_config("general", self.general_config)
        return target_value

    def _load_player_brawler_stats(self):
        self._save_brawlstars_settings()
        self._set_api_status("Contacting the official Brawl Stars API...", "muted")
        self.app.update_idletasks()

        profile = fetch_player_profile(
            self.api_key_var.get(),
            self.player_tag_var.get(),
        )
        self.player_profile_name = profile["player_name"]
        self.player_brawler_stats = profile["brawlers"]
        updated_count = self._sync_roster_trophies_from_api()
        self._refresh_brawler_button_labels()

        imported_count = len(self.player_brawler_stats)
        self._set_api_status(
            f"Imported {imported_count} owned brawlers for {self.player_profile_name or profile['player_tag']}. "
            f"Updated {updated_count} configured roster entries with live trophy counts.",
            "success",
        )
        return profile

    def _sync_roster_trophies_from_api(self):
        updated_count = 0
        for item in self.brawlers_data:
            imported = self.player_brawler_stats.get(item["brawler"])
            if imported is None:
                continue
            item["trophies"] = imported["trophies"]
            updated_count += 1
        if updated_count:
            self.refresh_roster_summary()
        return updated_count

    def import_brawlstars_trophies(self):
        try:
            self._load_player_brawler_stats()
        except ValueError as exc:
            self._set_api_status(str(exc), "danger")
        except Exception as exc:
            self._set_api_status(f"Import failed: {exc}", "danger")

    def build_auto_push_roster(self):
        try:
            target_trophies = self._save_brawlstars_settings()
            if not self.player_brawler_stats:
                self._load_player_brawler_stats()
        except ValueError as exc:
            self._set_api_status(str(exc), "danger")
            return
        except Exception as exc:
            self._set_api_status(f"Auto-push build failed: {exc}", "danger")
            return

        supported_brawlers = set(self.brawlers)
        auto_push_roster = []
        for brawler_name, stats in sorted(
            self.player_brawler_stats.items(),
            key=lambda item: (item[1]["trophies"], item[0]),
        ):
            if brawler_name not in supported_brawlers:
                continue
            if stats["trophies"] >= target_trophies:
                continue
            auto_push_roster.append({
                "brawler": brawler_name,
                "push_until": target_trophies,
                "trophies": stats["trophies"],
                "wins": 0,
                "type": "trophies",
                "automatically_pick": True,
                "win_streak": 0,
            })

        self.brawlers_data = auto_push_roster
        self.refresh_roster_summary()
        if auto_push_roster:
            self._set_api_status(
                f"Built an auto-push roster with {len(auto_push_roster)} brawlers below {target_trophies} trophies.",
                "success",
            )
        else:
            self._set_api_status(
                f"No supported owned brawlers are below {target_trophies} trophies.",
                "warning",
            )

    def _format_brawler_button_text(self, brawler):
        brawler_label = brawler.title()
        imported = self.player_brawler_stats.get(brawler)
        if imported is None:
            return brawler_label
        return f"{brawler_label}\n{imported['trophies']} trophies"

    def _refresh_brawler_button_labels(self):
        for brawler, button in self.brawler_buttons:
            button.configure(text=self._format_brawler_button_text(brawler))

    def _create_brawler_buttons(self):
        for brawler, image in self.images:
            button = ctk.CTkButton(
                self.image_frame,
                image=image,
                text=self._format_brawler_button_text(brawler),
                compound="top",
                command=lambda b=brawler: self.open_brawler_entry(b),
                width=S(128),
                height=S(146),
                corner_radius=S(14),
                fg_color=COLORS["surface_alt_2"],
                hover_color=COLORS["accent_soft"],
                text_color=COLORS["text"],
                font=font(13, "bold")
            )
            self._bind_scroll_handler_recursive(button, self.browser_scroll_handler)
            self.brawler_buttons.append((brawler, button))

    def _apply_image_filter(self):
        self.search_update_after_id = None
        self.update_images(self.filter_var.get())

    def start_bot(self):
        self.result_data = list(self.brawlers_data)
        if callable(self.data_setter):
            self.data_setter(self.result_data)
        self._request_close()

    def load_brawler_config(self):
        file_path = filedialog.askopenfilename(
            title="Select Saved Roster",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not file_path:
            return

        try:
            with open(file_path, "r") as file:
                brawlers_data = json.load(file)
            try:
                self.brawlers_data = [
                    bd for bd in brawlers_data
                    if not (bd["push_until"] <= bd[bd["type"]])
                ]
                print("Brawler data loaded successfully :", self.brawlers_data)
                self.refresh_roster_summary()
            except Exception as exc:
                print("Invalid data format. Expected a list of brawler data.", exc)
        except Exception as exc:
            print(f"Error loading brawler data: {exc}")

    @staticmethod
    def _parse_optional_int(raw_value):
        raw_value = raw_value.strip()
        return int(raw_value) if raw_value.isdigit() else ""

    @staticmethod
    def _parse_required_int(raw_value, default=0):
        raw_value = raw_value.strip()
        return int(raw_value) if raw_value.isdigit() else default

    @staticmethod
    def _infer_goal_type(goal_type, trophies_value, wins_value):
        if goal_type:
            return goal_type

        inferred_wins_value = wins_value if wins_value != "" else 0
        if trophies_value <= inferred_wins_value:
            return "trophies"
        return "wins"

    def _build_brawler_form_state(self, brawler, existing):
        imported = self.player_brawler_stats.get(brawler, {})
        imported_trophies = imported.get("trophies", 0)
        return {
            "push_until": "" if existing is None else str(existing["push_until"]),
            "trophies": str(imported_trophies if existing is None else existing["trophies"]),
            "wins": "" if existing is None else str(existing["wins"]),
            "win_streak": "0" if existing is None else str(existing["win_streak"]),
            "auto_pick": True if existing is None else bool(existing["automatically_pick"]),
            "goal_type": "trophies" if existing is None else str(existing["type"]),
        }

    def _build_brawler_payload(
        self,
        brawler,
        push_until_raw,
        trophies_raw,
        wins_raw,
        win_streak_raw,
        goal_type,
        automatically_pick,
    ):
        push_until_value = self._parse_optional_int(push_until_raw)
        trophies_value = self._parse_required_int(trophies_raw, default=0)
        wins_value = self._parse_optional_int(wins_raw)
        if goal_type == "trophies" and wins_value == "":
            wins_value = 0

        payload = {
            "brawler": brawler,
            "push_until": push_until_value,
            "trophies": trophies_value,
            "wins": wins_value,
            "type": self._infer_goal_type(goal_type, trophies_value, wins_value),
            "automatically_pick": automatically_pick,
            "win_streak": self._parse_required_int(win_streak_raw, default=0),
        }
        return payload

    def _upsert_brawler_data(self, payload):
        self.brawlers_data = [item for item in self.brawlers_data if item["brawler"] != payload["brawler"]]
        self.brawlers_data.append(payload)

    def open_brawler_entry(self, brawler):
        existing = next((item for item in self.brawlers_data if item["brawler"] == brawler), None)
        form_state = self._build_brawler_form_state(brawler, existing)
        self.app.update_idletasks()

        top = ctk.CTkToplevel(self.app)
        top.title("Configure Brawler")
        popup_width = S(460)
        popup_height = min(S(760), max(S(620), int(self.app.winfo_screenheight() * 0.82)))
        parent_x = self.app.winfo_rootx()
        parent_y = self.app.winfo_rooty()
        parent_width = self.app.winfo_width() or self.app.winfo_screenwidth()
        parent_height = self.app.winfo_height() or self.app.winfo_screenheight()
        popup_x = parent_x + max((parent_width - popup_width) // 2, S(20))
        popup_y = parent_y + max((parent_height - popup_height) // 2, S(20))
        top.geometry(f"{popup_width}x{popup_height}+{popup_x}+{popup_y}")
        top.minsize(S(440), S(620))
        top.resizable(True, True)
        top.attributes("-topmost", True)
        top.transient(self.app)
        top.grab_set()
        top.configure(fg_color=COLORS["bg"])
        top.grid_columnconfigure(0, weight=1)
        top.grid_rowconfigure(0, weight=1)

        card = ctk.CTkFrame(
            top,
            fg_color=COLORS["surface"],
            corner_radius=S(16),
            border_width=1,
            border_color=COLORS["border"]
        )
        card.grid(row=0, column=0, sticky="nsew", padx=S(16), pady=S(16))
        card.grid_columnconfigure(0, weight=1)
        card.grid_rowconfigure(0, weight=1)

        body = ctk.CTkScrollableFrame(
            card,
            fg_color=COLORS["surface"],
            corner_radius=S(14)
        )
        body.grid(row=0, column=0, sticky="nsew", padx=S(12), pady=(S(12), 0))
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        popup_scroll_handler = self._make_scroll_handler(body, lines_per_notch=9)

        ctk.CTkLabel(
            body,
            text=f"Configure {brawler.title()}",
            font=font(22, "bold"),
            text_color=COLORS["text"]
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=S(18), pady=(S(22), S(6)))

        ctk.CTkLabel(
            body,
            text="Choose what to push, then set the target and current progress.",
            font=font(13),
            text_color=COLORS["muted"],
            wraplength=S(360),
            justify="left"
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=S(18), pady=(0, S(16)))

        push_until_var = tk.StringVar(value=form_state["push_until"])
        trophies_var = tk.StringVar(value=form_state["trophies"])
        wins_var = tk.StringVar(value=form_state["wins"])
        current_win_streak_var = tk.StringVar(value=form_state["win_streak"])
        auto_pick_var = tk.BooleanVar(value=form_state["auto_pick"])
        goal_type_var = tk.StringVar(value=form_state["goal_type"])

        def create_field(row, label_text, variable):
            ctk.CTkLabel(
                body,
                text=label_text,
                font=font(13, "bold"),
                text_color=COLORS["text"]
            ).grid(row=row, column=0, columnspan=2, sticky="w", padx=S(18), pady=(0, S(6)))

            entry = ctk.CTkEntry(
                body,
                textvariable=variable,
                height=S(40),
                font=font(15),
                fg_color=COLORS["surface_alt_2"],
                border_color=COLORS["border"],
                text_color=COLORS["text"]
            )
            entry.grid(row=row + 1, column=0, columnspan=2, sticky="ew", padx=S(18), pady=(0, S(12)))
            return entry

        goal_buttons = ctk.CTkFrame(body, fg_color="transparent")
        goal_buttons.grid(row=2, column=0, columnspan=2, sticky="ew", padx=S(18), pady=(0, S(12)))
        goal_buttons.grid_columnconfigure(0, weight=1)
        goal_buttons.grid_columnconfigure(1, weight=1)

        def refresh_goal_buttons():
            selected = goal_type_var.get()
            if selected == "wins":
                wins_button.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"])
                trophies_button.configure(fg_color=COLORS["surface_alt"], hover_color=COLORS["border"])
            elif selected == "trophies":
                trophies_button.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"])
                wins_button.configure(fg_color=COLORS["surface_alt"], hover_color=COLORS["border"])
            else:
                wins_button.configure(fg_color=COLORS["surface_alt"], hover_color=COLORS["border"])
                trophies_button.configure(fg_color=COLORS["surface_alt"], hover_color=COLORS["border"])

        wins_button = ctk.CTkButton(
            goal_buttons,
            text="Wins",
            command=lambda: (goal_type_var.set("wins"), refresh_goal_buttons()),
            font=font(14, "bold"),
            height=S(42),
            corner_radius=S(10),
            fg_color=COLORS["surface_alt_2"],
            hover_color=COLORS["border"],
            text_color=COLORS["text"]
        )
        wins_button.grid(row=0, column=0, sticky="ew", padx=(0, S(6)))

        trophies_button = ctk.CTkButton(
            goal_buttons,
            text="Trophies",
            command=lambda: (goal_type_var.set("trophies"), refresh_goal_buttons()),
            font=font(14, "bold"),
            height=S(42),
            corner_radius=S(10),
            fg_color=COLORS["surface_alt_2"],
            hover_color=COLORS["border"],
            text_color=COLORS["text"]
        )
        trophies_button.grid(row=0, column=1, sticky="ew", padx=(S(6), 0))

        create_field(3, "Target Value", push_until_var)
        create_field(5, "Current Trophies", trophies_var)
        create_field(7, "Current Wins", wins_var)
        create_field(9, "Current Win Streak", current_win_streak_var)

        auto_pick_checkbox = ctk.CTkCheckBox(
            body,
            text="Auto-select this brawler in the lobby",
            variable=auto_pick_var,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["text"],
            font=font(13)
        )
        auto_pick_checkbox.grid(row=11, column=0, columnspan=2, sticky="w", padx=S(18), pady=(S(4), S(12)))

        action_panel = ctk.CTkFrame(
            card,
            fg_color=COLORS["surface_alt"],
            corner_radius=S(14),
            border_width=1,
            border_color=COLORS["border"]
        )
        action_panel.grid(row=1, column=0, sticky="ew", padx=S(12), pady=S(12))
        action_panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            action_panel,
            text="Save this configuration to add or update the brawler in your roster.",
            font=font(12, "bold"),
            text_color=COLORS["text"]
        ).grid(row=0, column=0, sticky="w", padx=S(16), pady=(S(14), S(8)))

        action_row = ctk.CTkFrame(action_panel, fg_color="transparent")
        action_row.grid(row=1, column=0, sticky="ew", padx=S(16), pady=(0, S(16)))
        action_row.grid_columnconfigure(0, weight=1)
        action_row.grid_columnconfigure(1, weight=2)

        def save_brawler_entry():
            push_target = push_until_var.get().strip()
            if not push_target.isdigit():
                messagebox.showerror(
                    "Missing Target",
                    "Enter a numeric target value before saving this brawler.",
                    parent=top,
                )
                return

            payload = self._build_brawler_payload(
                brawler=brawler,
                push_until_raw=push_until_var.get(),
                trophies_raw=trophies_var.get(),
                wins_raw=wins_var.get(),
                win_streak_raw=current_win_streak_var.get(),
                goal_type=goal_type_var.get(),
                automatically_pick=auto_pick_var.get(),
            )
            self._upsert_brawler_data(payload)

            if debug:
                print("Selected Brawler Data :", self.brawlers_data)
            self.refresh_roster_summary()
            top.destroy()

        cancel_button = ctk.CTkButton(
            action_row,
            text="Cancel",
            command=top.destroy,
            fg_color=COLORS["surface_alt"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            corner_radius=S(10),
            font=font(14, "bold"),
            height=S(42)
        )
        cancel_button.grid(row=0, column=0, sticky="ew", padx=(0, S(6)))

        save_button = ctk.CTkButton(
            action_row,
            text="Save Brawler to Roster",
            command=save_brawler_entry,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["text"],
            corner_radius=S(10),
            font=font(16, "bold"),
            height=S(54)
        )
        save_button.grid(row=0, column=1, sticky="ew", padx=(S(6), 0))

        refresh_goal_buttons()
        self._bind_scroll_handler_recursive(body, popup_scroll_handler)
        self._bind_scroll_handler_recursive(action_panel, popup_scroll_handler)
        save_button.focus()

    def remove_brawler(self, brawler):
        self.brawlers_data = [item for item in self.brawlers_data if item["brawler"] != brawler]
        self.refresh_roster_summary()

    def update_images(self, filter_text):
        normalized_filter = filter_text.strip().lower()
        row_num = 0
        col_num = 0
        available_width = max(self.image_frame.winfo_width(), self.left_panel.winfo_width())
        column_count = max(2, min(6, available_width // max(S(150), 1)))
        for column_index in range(column_count):
            self.image_frame.grid_columnconfigure(column_index, weight=1)

        for brawler, button in self.brawler_buttons:
            if normalized_filter and normalized_filter not in brawler:
                button.grid_remove()
                continue

            button.grid(row=row_num, column=col_num, padx=S(8), pady=S(8), sticky="nsew")

            col_num += 1
            if col_num == column_count:
                col_num = 0
                row_num += 1

    def refresh_roster_summary(self):
        roster_count = len(self.brawlers_data)
        self.roster_count_label.configure(
            text=f"{roster_count} configured" if roster_count else "0 configured"
        )
        self.roster_empty_label.grid() if roster_count == 0 else self.roster_empty_label.grid_remove()

        active_brawlers = set()
        for row, item in enumerate(self.brawlers_data):
            brawler_name = item["brawler"]
            active_brawlers.add(brawler_name)
            card_info = self.roster_cards.get(brawler_name)
            if card_info is None:
                card_info = self._create_roster_card(brawler_name)
                self.roster_cards[brawler_name] = card_info

            self._update_roster_card(card_info, item)
            card_info["frame"].grid(row=row, column=0, sticky="ew", padx=S(4), pady=S(6))

        for brawler_name, card_info in self.roster_cards.items():
            if brawler_name not in active_brawlers:
                card_info["frame"].grid_remove()

    def _create_roster_card(self, brawler_name):
        card = ctk.CTkFrame(
            self.roster_frame,
            fg_color=COLORS["surface_alt_2"],
            corner_radius=S(14),
            border_width=1,
            border_color=COLORS["border"]
        )
        card.grid_columnconfigure(1, weight=1)

        icon = ctk.CTkLabel(card, image=self.image_lookup.get(brawler_name), text="")
        icon.grid(row=0, column=0, rowspan=3, padx=S(12), pady=S(12), sticky="n")

        title = ctk.CTkLabel(
            card,
            text=brawler_name.title(),
            font=font(15, "bold"),
            text_color=COLORS["text"]
        )
        title.grid(row=0, column=1, sticky="w", padx=(0, S(8)), pady=(S(12), S(4)))

        subtitle = ctk.CTkLabel(
            card,
            font=font(12),
            text_color=COLORS["muted"],
            justify="left",
            wraplength=S(270)
        )
        subtitle.grid(row=1, column=1, sticky="w", padx=(0, S(8)), pady=(0, S(10)))

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.grid(row=2, column=1, sticky="w", padx=(0, S(8)), pady=(0, S(12)))

        edit_button = ctk.CTkButton(
            actions,
            text="Edit",
            command=lambda b=brawler_name: self.open_brawler_entry(b),
            width=S(72),
            height=S(32),
            corner_radius=S(8),
            fg_color=COLORS["surface"],
            hover_color=COLORS["border"],
            border_width=1,
            border_color=COLORS["border"],
            text_color=COLORS["text"],
            font=font(12, "bold")
        )
        edit_button.pack(side="left", padx=(0, S(8)))

        remove_button = ctk.CTkButton(
            actions,
            text="Remove",
            command=lambda b=brawler_name: self.remove_brawler(b),
            width=S(86),
            height=S(32),
            corner_radius=S(8),
            fg_color=COLORS["accent_soft"],
            hover_color=COLORS["accent_hover"],
            text_color=COLORS["text"],
            font=font(12, "bold")
        )
        remove_button.pack(side="left")

        self._bind_scroll_handler_recursive(card, self.roster_scroll_handler)

        return {
            "frame": card,
            "title": title,
            "subtitle": subtitle,
        }

    @staticmethod
    def _update_roster_card(card_info, item):
        goal_type = str(item["type"]).title() if item["type"] else "Auto"
        target_value = item["push_until"] if item["push_until"] != "" else "Not set"
        stats_line = f"Current trophies: {item['trophies']} | Current wins: {item['wins']}"
        card_info["title"].configure(text=item["brawler"].title())
        card_info["subtitle"].configure(
            text=f"Goal: {goal_type} -> {target_value}\n{stats_line}\nAuto-pick: {'Yes' if item['automatically_pick'] else 'No'} | Win streak: {item['win_streak']}"
        )

    def _save_runtime_limit(self, _event=None):
        value = self.timer_var.get()
        try:
            minutes = int(value)
            self.general_config["run_for_minutes"] = minutes
            self.general_config = save_config("general", self.general_config)
            self.timer_var.set(str(self.general_config["run_for_minutes"]))
        except ValueError:
            self.timer_var.set(str(self.general_config.get("run_for_minutes", 600)))
