import customtkinter as ctk
import webbrowser
import os
from PIL import Image
import tkinter as tk
from gui.config_store import load_config, save_config, update_config_value
from gui.theme import COLORS, FONT_FAMILY, S, apply_appearance, center_window
from utils import get_discord_link
from packaging import version


class Hub:
    """
    Updated, more user-friendly interface for the Pyla bot.
    """

    def __init__(self,
                 version_str,
                 latest_version_str,
                 correct_zoom=True,
                 on_close_callback=None):

        self.version_str = version_str
        self.latest_version_str = latest_version_str
        self.correct_zoom = correct_zoom
        self.on_close_callback = on_close_callback

        # -----------------------------------------------------------------------------------------
        # Load configs
        # -----------------------------------------------------------------------------------------
        self.bot_config = load_config("bot")
        self.time_tresholds = load_config("time")
        self.match_history = load_config("match_history")
        self.general_config = load_config("general")

        # -----------------------------------------------------------------------------------------
        # Appearance
        # -----------------------------------------------------------------------------------------
        apply_appearance()

        # For showing tooltips in Toplevel windows
        # For showing tooltips
        self.tooltip_window = None
        self._tooltip_after_id = None
        self._tooltip_owner = None
        self._tooltip_text = ""
        self._closing = False
        self._should_continue = False

        # -----------------------------------------------------------------------------------------
        # Main window
        # -----------------------------------------------------------------------------------------
        self.app = ctk.CTk()
        self.app.title(f"Pyla Hub - {self.version_str}")
        hub_width = S(1100)
        hub_height = S(820)
        self.app.geometry(f"{hub_width}x{hub_height}")
        self.app.minsize(S(980), S(760))
        self.app.resizable(True, True)
        self.app.configure(fg_color=COLORS["bg"])
        self.app.protocol("WM_DELETE_WINDOW", self._request_close)
        center_window(self.app, hub_width, hub_height)

        # Hide tooltip on "global" interactions (tab switch, clicks, scroll, key press, focus loss, etc.)
        for seq in ("<ButtonPress>", "<MouseWheel>", "<KeyPress>", "<FocusOut>"):
            self.app.bind_all(seq, self._hide_tooltip, add="+")
        self.app.bind("<Configure>", self._hide_tooltip, add="+")  # window move/resize

        # -----------------------------------------------------------------------------------------
        # Main TabView
        # -----------------------------------------------------------------------------------------
        self.tabview = ctk.CTkTabview(
            self.app,
            width=S(1040),
            height=S(700),
            corner_radius=S(10)
        )
        self.tabview.pack(pady=S(14), padx=S(14), fill="both", expand=True)

        # Enlarge the segmented tab buttons
        self.tabview._segmented_button.configure(
            corner_radius=S(10),
            border_width=2,
            fg_color=COLORS["surface_alt"],
            selected_color=COLORS["accent"],
            selected_hover_color=COLORS["accent_hover"],
            unselected_color=COLORS["surface"],
            unselected_hover_color=COLORS["border"],
            text_color=COLORS["text"],
            font=(FONT_FAMILY, S(16), "bold"),
            height=S(40)
        )

        # Add tabs
        self.tab_overview = self.tabview.add("Setup")
        self.tab_additional = self.tabview.add("Advanced")
        self.tab_timers = self.tabview.add("Detection Timing")
        self.tab_history = self.tabview.add("History")

        # Init each tab
        self._init_overview_tab()
        self._init_additional_tab()
        self._init_timers_tab()
        self._init_history_tab()

        # Main loop
        try:
            self.app.mainloop()
        finally:
            self._finalize_window()
        if self._should_continue and callable(self.on_close_callback):
            self.on_close_callback()

    def _save_config(self, config_name, config_data):
        sanitized = save_config(config_name, config_data)
        if config_name == "bot":
            self.bot_config = sanitized
        elif config_name == "time":
            self.time_tresholds = sanitized
        elif config_name == "general":
            self.general_config = sanitized
        elif config_name == "match_history":
            self.match_history = sanitized
        return sanitized

    def _get_config(self, config_name):
        if config_name == "bot":
            return self.bot_config
        if config_name == "time":
            return self.time_tresholds
        if config_name == "general":
            return self.general_config
        if config_name == "match_history":
            return self.match_history
        raise ValueError(f"Unknown config name: {config_name}")

    def _create_section_header(self, parent, row_idx, text):
        header = ctk.CTkLabel(
            parent,
            text=text,
            font=(FONT_FAMILY, S(20), "bold"),
            text_color=COLORS["text"]
        )
        header.grid(row=row_idx, column=0, columnspan=2, pady=(S(14), S(4)))
        return row_idx + 1

    def _create_config_entry(
        self,
        parent,
        row_idx,
        label_text,
        config_name,
        config_key,
        convert_func,
        tooltip_text=None,
    ):
        label = ctk.CTkLabel(parent, text=label_text, font=(FONT_FAMILY, S(18)))
        label.grid(row=row_idx, column=0, sticky="e", padx=S(20), pady=S(10))

        initial_value = str(self._get_config(config_name)[config_key])
        value_var = tk.StringVar(value=initial_value)

        def on_save(*_):
            current_config = self._get_config(config_name)
            raw_value = value_var.get().strip()
            if raw_value == "":
                value_var.set(str(current_config[config_key]))
                return
            try:
                converted_value = convert_func(raw_value)
                current_config[config_key] = converted_value
                self._save_config(config_name, current_config)
                value_var.set(str(self._get_config(config_name)[config_key]))
            except ValueError:
                value_var.set(str(current_config[config_key]))

        entry = ctk.CTkEntry(
            parent, textvariable=value_var, width=S(120), font=(FONT_FAMILY, S(16))
        )
        entry.grid(row=row_idx, column=1, sticky="w", padx=S(20), pady=S(10))
        entry.bind("<FocusOut>", on_save)
        entry.bind("<Return>", on_save)

        if tooltip_text:
            self.attach_tooltip(entry, tooltip_text)

        return row_idx + 1

    def _create_timer_setting(self, parent, row_idx, param_name, label_text, tooltip_text=None, disabled=False):
        label = ctk.CTkLabel(parent, text=label_text, font=(FONT_FAMILY, S(18)))
        label.grid(row=row_idx, column=0, padx=S(20), pady=S(10), sticky="e")

        slider_entry_frame = ctk.CTkFrame(parent, fg_color="transparent")
        slider_entry_frame.grid(row=row_idx, column=1, padx=S(20), pady=S(10), sticky="w")

        value_var = tk.StringVar(value=str(self.time_tresholds[param_name]))

        slider = ctk.CTkSlider(
            slider_entry_frame,
            from_=0.1,
            to=10,
            number_of_steps=99,
            width=S(200),
            command=lambda value: on_slider_change(value),
            state=("disabled" if disabled else "normal")
        )
        slider.pack(side="left", padx=S(5))

        entry = ctk.CTkEntry(
            slider_entry_frame,
            textvariable=value_var,
            width=S(80),
            font=(FONT_FAMILY, S(16)),
            state=("disabled" if disabled else "normal")
        )
        entry.pack(side="left", padx=S(10))

        def normalize_slider_value(value):
            return min(max(value, 0.1), 10)

        def on_save(_):
            if disabled:
                return
            new_value_str = value_var.get().strip()
            if new_value_str == "":
                value_var.set(str(self.time_tresholds[param_name]))
                return
            try:
                value = float(new_value_str)
                self.time_tresholds[param_name] = value
                self._save_config("time", self.time_tresholds)
                slider.set(normalize_slider_value(value))
            except ValueError:
                value_var.set(str(self.time_tresholds[param_name]))

        def on_slider_change(value):
            if disabled:
                return
            value = float(value)
            value_var.set(f"{value:.2f}")
            self.time_tresholds[param_name] = value
            self._save_config("time", self.time_tresholds)

        entry.bind("<FocusOut>", on_save)
        entry.bind("<Return>", on_save)

        try:
            slider.set(normalize_slider_value(float(self.time_tresholds[param_name])))
        except Exception:
            slider.set(1.0)

        if tooltip_text and not disabled:
            self.attach_tooltip(slider, tooltip_text)
            self.attach_tooltip(entry, tooltip_text)

        return row_idx + 1

    def _update_config_value(self, config_name, config_data, key, value):
        sanitized = update_config_value(config_name, config_data, key, value)
        if config_name == "bot":
            self.bot_config = sanitized
        elif config_name == "time":
            self.time_tresholds = sanitized
        elif config_name == "general":
            self.general_config = sanitized
        elif config_name == "match_history":
            self.match_history = sanitized
        return sanitized

    # ---------------------------------------------------------------------------------------------
    #  Tooltip Handler
    # ---------------------------------------------------------------------------------------------
    def _pointer_over_widget(self, widget) -> bool:
        if widget is None or not widget.winfo_exists():
            return False
        try:
            px, py = widget.winfo_pointerx(), widget.winfo_pointery()
            x, y = widget.winfo_rootx(), widget.winfo_rooty()
            w, h = widget.winfo_width(), widget.winfo_height()
            return x <= px <= x + w and y <= py <= y + h
        except tk.TclError:
            return False

    def _hide_tooltip(self, _event=None):
        # cancel delayed show if pending
        if self._tooltip_after_id is not None:
            try:
                self.app.after_cancel(self._tooltip_after_id)
            except Exception:
                pass
            self._tooltip_after_id = None

        # destroy current tooltip window
        if self.tooltip_window is not None:
            try:
                self.tooltip_window.destroy()
            except Exception:
                pass
            self.tooltip_window = None

        self._tooltip_owner = None
        self._tooltip_text = ""

    def _request_close(self):
        if self._closing:
            return
        self._closing = True
        self._hide_tooltip()
        try:
            self.app.quit()
        except Exception:
            pass

    def _finalize_window(self):
        self._hide_tooltip()
        if self.app.winfo_exists():
            try:
                self.app.destroy()
            except Exception:
                pass

    def attach_tooltip(self, widget, text, delay_ms: int = 250):
        """
        Robust tooltip:
        - shows after delay
        - hides on Leave, Unmap (tab switch), Destroy, clicks/scroll/keys (via global binds)
        - prevents stuck tooltips when switching tabs
        """

        def schedule_show(event=None):
            # reset any existing tooltip
            self._hide_tooltip()

            self._tooltip_owner = widget
            self._tooltip_text = text

            def do_show():
                # widget may have disappeared / tab switched
                if (self._tooltip_owner is None
                        or not self._tooltip_owner.winfo_exists()
                        or not self._tooltip_owner.winfo_viewable()
                        or not self._pointer_over_widget(self._tooltip_owner)):
                    self._hide_tooltip()
                    return

                # create tooltip
                self.tooltip_window = ctk.CTkToplevel(self.app)
                self.tooltip_window.overrideredirect(True)
                self.tooltip_window.attributes("-topmost", True)

                # position near cursor
                px = self.app.winfo_pointerx()
                py = self.app.winfo_pointery()
                self.tooltip_window.geometry(f"+{px + 12}+{py + 12}")

                label = ctk.CTkLabel(
                    self.tooltip_window,
                    text=self._tooltip_text,
                    fg_color=COLORS["surface"],
                    text_color=COLORS["text"],
                    corner_radius=S(6),
                    font=(FONT_FAMILY, S(12))
                )
                label.pack(padx=S(6), pady=S(4))

                # if mouse enters tooltip itself, hide (avoids "stuck" hovering on tooltip)
                self.tooltip_window.bind("<Enter>", self._hide_tooltip)
                self.tooltip_window.bind("<Leave>", self._hide_tooltip)

            self._tooltip_after_id = self.app.after(delay_ms, do_show)

        def on_leave(_event=None):
            self._hide_tooltip()

        # Bindings
        widget.bind("<Enter>", schedule_show, add="+")
        widget.bind("<Leave>", on_leave, add="+")
        widget.bind("<Unmap>", on_leave, add="+")  # IMPORTANT: tab switching / frame hidden
        widget.bind("<Destroy>", on_leave, add="+")  # safety
        widget.bind("<ButtonPress>", on_leave, add="+")  # click on the widget -> hide

    # ---------------------------------------------------------------------------------------------
    #  Overview Tab
    # ---------------------------------------------------------------------------------------------
    def _init_overview_tab(self):
        frame = self.tab_overview

        container = ctk.CTkFrame(frame, fg_color="transparent")
        container.pack(expand=True, fill="both")

        row_ = 0

        # -----------------------------------------------------------------
        # 1) Warnings at the top (bigger, red), if any
        # -----------------------------------------------------------------
        w_list = []
        if not self.correct_zoom:
            w_list.append("Warning: Your Windows zoom isn't 100% (DPI != 96).")
        if self.latest_version_str and version.parse(self.version_str) < version.parse(self.latest_version_str):
            w_list.append(f"Warning: You are not on the latest version ({self.latest_version_str}).")

        if w_list:
            warn_text = "\n".join(w_list)
            warn_label = ctk.CTkLabel(
                container,
                text=warn_text,
                text_color="#e74c3c",
                font=(FONT_FAMILY, S(16), "bold")
            )
            warn_label.grid(row=row_, column=0, columnspan=2, pady=S(10))
            row_ += 1

        # -----------------------------------------------------------------
        # 2) Map Orientation selection
        # -----------------------------------------------------------------
        self.gamemode_type_var = tk.IntVar(value=self.bot_config["gamemode_type"])

        orientation_frame = ctk.CTkFrame(container, fg_color="transparent")
        orientation_frame.grid(row=row_, column=0, columnspan=2, pady=S(10))

        label_type = ctk.CTkLabel(
            orientation_frame,
            text="Map Layout:",
            font=(FONT_FAMILY, S(20), "bold")
        )
        label_type.pack(side="left", padx=S(15))

        def set_gamemode_type(t):
            """Only change the local var & refresh everything so frames swap."""
            self.gamemode_type_var.set(t)
            self._refresh_gamemode_buttons()

        self.btn_type_vertical = ctk.CTkButton(
            orientation_frame,
            text="Vertical",
            command=lambda: set_gamemode_type(3),
            font=(FONT_FAMILY, S(16), "bold"),
            corner_radius=S(6),
            width=S(120),
            height=S(40)
        )
        self.btn_type_vertical.pack(side="left", padx=S(10))

        self.btn_type_horizontal = ctk.CTkButton(
            orientation_frame,
            text="Horizontal",
            command=lambda: set_gamemode_type(5),
            font=(FONT_FAMILY, S(16), "bold"),
            corner_radius=S(6),
            width=S(120),
            height=S(40)
        )
        self.btn_type_horizontal.pack(side="left", padx=S(10))

        row_ += 1

        # -----------------------------------------------------------------
        # 3) Gamemode Selection as rectangular buttons
        # -----------------------------------------------------------------
        gm_label = ctk.CTkLabel(container, text="Game Mode:", font=(FONT_FAMILY, S(20), "bold"))
        gm_label.grid(row=row_, column=0, columnspan=2, pady=S(10))
        row_ += 1

        gm_buttons_frame = ctk.CTkFrame(container, fg_color="transparent")
        gm_buttons_frame.grid(row=row_, column=0, columnspan=2, pady=S(10))

        self.gm3_frame = ctk.CTkFrame(gm_buttons_frame, fg_color="transparent")
        self.gm5_frame = ctk.CTkFrame(gm_buttons_frame, fg_color="transparent")

        self.gamemode_var = tk.StringVar(value=self.bot_config["gamemode"])

        def create_gamemode_button(parent, gm_value, text_display, disabled=False, orientation=3):
            """Creates a rectangular toggle button for a gamemode."""

            def on_click():
                if disabled:
                    return
                # Set orientation + gamemode in config
                self.bot_config["gamemode_type"] = orientation
                self.bot_config["gamemode"] = gm_value
                self._save_config("bot", self.bot_config)

                self.gamemode_type_var.set(orientation)
                self.gamemode_var.set(gm_value)
                self._refresh_gamemode_buttons()

            btn = ctk.CTkButton(
                parent,
                text=text_display,
                command=on_click,
                corner_radius=S(6),
                width=S(150),
                height=S(40),
                font=(FONT_FAMILY, S(16), "bold"),
                state=("disabled" if disabled else "normal")
            )
            return btn

        # For type=3 (vertical)
        self.rb_brawlball_3 = create_gamemode_button(
            self.gm3_frame, "brawlball", "Brawlball", orientation=3
        )
        self.rb_other_3 = create_gamemode_button(
            self.gm3_frame, "other", "Other Modes", orientation=3
        )

        self.rb_brawlball_3.grid(row=0, column=0, padx=S(10), pady=S(5))
        self.rb_other_3.grid(row=0, column=1, padx=S(10), pady=S(5))

        # For type=5 (horizontal)
        self.rb_basketbrawl_5 = create_gamemode_button(
            self.gm5_frame, "basketbrawl", "Basket Brawl", orientation=5
        )
        self.rb_bb5v5_5 = create_gamemode_button(
            self.gm5_frame, "brawlball_5v5", "Brawlball 5v5", orientation=5
        )

        self.rb_basketbrawl_5.grid(row=0, column=0, padx=S(10), pady=S(5))
        self.rb_bb5v5_5.grid(row=0, column=1, padx=S(10), pady=S(5))

        def refresh_gm_buttons():
            """Refresh button colors to highlight the currently selected gamemode."""
            gm_now = self.gamemode_var.get()

            def set_button_color(btn, val):
                if val == gm_now:
                    btn.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"])
                else:
                    btn.configure(fg_color=COLORS["surface"], hover_color=COLORS["accent_hover"])

            # For vertical set
            set_button_color(self.rb_brawlball_3, "brawlball")
            set_button_color(self.rb_other_3, "other")

            # For horizontal set
            set_button_color(self.rb_basketbrawl_5, "basketbrawl")
            set_button_color(self.rb_bb5v5_5, "brawlball_5v5")

        def refresh_orientation_buttons():
            """Refresh the orientation buttons' color based on self.gamemode_type_var."""
            t = self.gamemode_type_var.get()
            if t == 3:
                self.btn_type_vertical.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"])
                self.btn_type_horizontal.configure(fg_color=COLORS["surface"], hover_color=COLORS["accent_hover"])
            else:
                self.btn_type_vertical.configure(fg_color=COLORS["surface"], hover_color=COLORS["accent_hover"])
                self.btn_type_horizontal.configure(fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"])

        self._refresh_orientation_buttons = refresh_orientation_buttons

        def _refresh_gm_frames():
            """Show/hide frames depending on orientation."""
            self.gm3_frame.pack_forget()
            self.gm5_frame.pack_forget()

            if self.gamemode_type_var.get() == 3:
                self.gm3_frame.pack(side="top")
            else:
                self.gm5_frame.pack(side="top")

        def full_refresh():
            self._refresh_orientation_buttons()
            _refresh_gm_frames()
            refresh_gm_buttons()

        self._refresh_gamemode_buttons = full_refresh
        full_refresh()

        row_ += 1

        setup_note = ctk.CTkLabel(
            container,
            text="Device connection is detected automatically when runtime starts.",
            font=(FONT_FAMILY, S(14)),
            text_color=COLORS["muted"]
        )
        setup_note.grid(row=row_, column=0, columnspan=2, pady=(S(8), S(2)))
        row_ += 1

        # -----------------------------------------------------------------
        # 5) Start Button
        # -----------------------------------------------------------------
        start_button = ctk.CTkButton(
            container,
            text="Continue To Brawler Setup",
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, S(24), "bold"),
            command=self._on_start,
            width=S(220),
            height=S(60),
            corner_radius=S(10),
            text_color=COLORS["text"]
        )
        start_button.grid(row=row_, column=0, columnspan=2, padx=S(20), pady=S(30))
        row_ += 1

        # -----------------------------------------------------------------
        # 6) "Pyla is free..." label at bottom, link in blue only
        # -----------------------------------------------------------------
        disclaim_frame = ctk.CTkFrame(container, fg_color="transparent")
        disclaim_frame.grid(row=row_, column=0, columnspan=2, pady=S(10))

        disclaim_label = ctk.CTkLabel(
            disclaim_frame,
            text="Community: ",
            font=(FONT_FAMILY, S(18), "bold"),
            text_color=COLORS["text"]
        )
        disclaim_label.pack(side="left")

        discord_link = get_discord_link()

        def open_discord_link():
            webbrowser.open(discord_link)

        link_label = ctk.CTkLabel(
            disclaim_frame,
            text=discord_link,
            font=(FONT_FAMILY, S(18), "bold"),
            text_color=COLORS["link"],
            cursor="hand2"
        )
        link_label.pack(side="left")
        link_label.bind("<Button-1>", lambda e: open_discord_link())

        row_ += 1

        ad_frame = ctk.CTkFrame(container, fg_color="transparent")
        ad_frame.grid(row=row_, column=0, columnspan=2, pady=S(10))

        ad_label = ctk.CTkLabel(
            ad_frame,
            text="Support: ",
            font=(FONT_FAMILY, S(18), "bold"),
            text_color=COLORS["text"]
        )
        ad_label.pack(side="left")

        shown_patreon_link = "www.patreon.com/c/pyla"
        patreon_link = "https://www.patreon.com/pyla/membership"
        def open_patreon_link():
            webbrowser.open(patreon_link)
        patreon_label = ctk.CTkLabel(
            ad_frame,
            text=shown_patreon_link,
            font=(FONT_FAMILY, S(18), "bold"),
            text_color=COLORS["link"],
            cursor="hand2"
        )
        patreon_label.pack(side="left")
        patreon_label.bind("<Button-1>", lambda e: open_patreon_link())

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

    # ---------------------------------------------------------------------------------------------
    #  Additional Settings Tab
    # ---------------------------------------------------------------------------------------------
    def _init_additional_tab(self):
        frame = self.tab_additional
        container = ctk.CTkFrame(frame, fg_color="transparent")
        container.pack(expand=True, fill="both")

        # Extra space to avoid tooltip clipping
        container.grid_rowconfigure(0, minsize=S(10))

        row_idx = 0

        row_idx = self._create_section_header(container, row_idx, "Movement")

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Minimum Movement Hold:",
            config_name="bot",
            config_key="minimum_movement_delay",
            convert_func=float,
            tooltip_text="How long, in seconds, the bot keeps a movement before switching."
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Unstuck Delay:",
            config_name="bot",
            config_key="unstuck_movement_delay",
            convert_func=float,
            tooltip_text="How long, in seconds, the bot keeps trying the same movement before attempting to unstick itself."
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Unstuck Duration:",
            config_name="bot",
            config_key="unstuck_movement_hold_time",
            convert_func=float,
            tooltip_text="How long, in seconds, the bot keeps the unstuck movement before returning to normal control."
        )

        row_idx = self._create_section_header(container, row_idx, "Detection")

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Wall Detection Confidence:",
            config_name="bot",
            config_key="wall_detection_confidence",
            convert_func=float,
            tooltip_text="Confidence threshold for wall detection. Lower values can detect more objects, but may increase false positives."
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Player / Enemy Confidence:",
            config_name="bot",
            config_key="entity_detection_confidence",
            convert_func=float,
            tooltip_text="Confidence threshold for player, enemy, and teammate detection. Lower values can detect more objects, but may increase false positives."
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Super Ready Pixels:",
            config_name="bot",
            config_key="super_pixels_minimum",
            convert_func=float,
            tooltip_text='Minimum yellow pixel count used to treat Super as ready.'
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Gadget Ready Pixels:",
            config_name="bot",
            config_key="gadget_pixels_minimum",
            convert_func=float,
            tooltip_text='Minimum green pixel count used to treat Gadget as ready.'
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Hypercharge Ready Pixels:",
            config_name="bot",
            config_key="hypercharge_pixels_minimum",
            convert_func=float,
            tooltip_text='Minimum purple pixel count used to treat Hypercharge as ready.'
        )

        row_idx = self._create_section_header(container, row_idx, "Runtime")

        lbl_gpu = ctk.CTkLabel(container, text="Execution Device:", font=(FONT_FAMILY, S(18)))
        lbl_gpu.grid(row=row_idx, column=0, sticky="e", padx=S(20), pady=S(10))

        gpu_values = ["auto", "gpu", "cpu"]
        gpu_var = tk.StringVar(value=self.general_config["cpu_or_gpu"])

        def on_gpu_change(choice):
            self._update_config_value("general", self.general_config, "cpu_or_gpu", choice)

        gpu_menu = ctk.CTkOptionMenu(
            container,
            values=gpu_values,
            command=on_gpu_change,
            variable=gpu_var,
            font=(FONT_FAMILY, S(16)),
            fg_color=COLORS["accent"],
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            width=S(100),
            height=S(35)
        )
        gpu_menu.grid(row=row_idx, column=1, padx=S(20), pady=S(10), sticky="w")
        self.attach_tooltip(gpu_menu, "Choose Auto to prefer the best available device, GPU to force a GPU provider when available, or CPU to stay on the processor.")
        row_idx += 1

        lbl_long_press = ctk.CTkLabel(container, text="Hold To Open Starr Drop:", font=(FONT_FAMILY, S(18)))
        lbl_long_press.grid(row=row_idx, column=0, sticky="e", padx=S(20), pady=S(10))
        long_press_var = tk.BooleanVar(
            value=(str(self.general_config["long_press_star_drop"]).lower() in ["yes", "true"])
        )

        def toggle_long_press_detection():
            new_value = "yes" if long_press_var.get() else "no"
            self._update_config_value("general", self.general_config, "long_press_star_drop", new_value)

        long_press_cb = ctk.CTkCheckBox(
            container,
            text="",
            variable=long_press_var,
            command=toggle_long_press_detection,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            width=S(30),
            height=S(30)
        )
        long_press_cb.grid(row=row_idx, column=1, sticky="w", padx=S(20), pady=S(10))
        self.attach_tooltip(long_press_cb, "When enabled, the bot holds the confirm button longer on Starr Drop screens.")
        row_idx += 1

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Run Time Limit (Minutes):",
            config_name="general",
            config_key="run_for_minutes",
            convert_func=int,
            tooltip_text="How long the bot should run before entering its cooldown stop behavior. Use 0 for no limit."
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Trophies Multiplier:",
            config_name="general",
            config_key="trophies_multiplier",
            convert_func=int,
            tooltip_text="Multiplier applied to trophy gains for custom modes such as doubled-trophy environments."
        )

        row_idx = self._create_config_entry(
            container,
            row_idx,
            label_text="Rate Limit (IPS):",
            config_name="general",
            config_key="max_ips",
            convert_func=lambda s: s if s.lower() == "auto" else int(s),
            tooltip_text="Maximum images per second to process. 'auto' leaves the runtime uncapped."
        )

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

    # ---------------------------------------------------------------------------------------------
    #  Timers Tab
    # ---------------------------------------------------------------------------------------------
    def _init_timers_tab(self):
        frame = self.tab_timers
        container = ctk.CTkFrame(frame, fg_color="transparent")
        container.pack(expand=True, fill="both")

        container.grid_rowconfigure(0, minsize=S(70))  # extra top space for tooltips

        row_idx = 1

        row_idx = self._create_timer_setting(
            container,
            row_idx,
            param_name="super",
            label_text="Super Check Interval:",
            tooltip_text="How often, in seconds, the bot checks whether Super is ready."
        )
        row_idx = self._create_timer_setting(
            container,
            row_idx,
            param_name="hypercharge",
            label_text="Hypercharge Check Interval:",
            tooltip_text="How often, in seconds, the bot checks whether Hypercharge is ready."
        )
        row_idx = self._create_timer_setting(
            container,
            row_idx,
            param_name="gadget",
            label_text="Gadget Check Interval:",
            tooltip_text="How often, in seconds, the bot checks whether Gadget is ready."
        )
        row_idx = self._create_timer_setting(
            container,
            row_idx,
            param_name="wall_detection",
            label_text="Wall Detection Interval:",
            tooltip_text="How often, in seconds, the bot refreshes wall detections."
        )
        row_idx = self._create_timer_setting(
            container,
            row_idx,
            param_name="no_detection_proceed",
            label_text="Recover When Player Missing:",
            tooltip_text="How often, in seconds, the bot tries to proceed when the player is missing but the game state still looks like a match."
        )

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=1)

    # ---------------------------------------------------------------------------------------------
    #  Match History Tab
    # ---------------------------------------------------------------------------------------------
    def _init_history_tab(self):
        frame = self.tab_history

        scroll_frame = ctk.CTkScrollableFrame(
            frame, width=S(900), height=S(600), fg_color="transparent", corner_radius=S(10)
        )
        scroll_frame.pack(fill="both", expand=True, padx=S(10), pady=S(10))

        max_cols = 4
        row_idx = 0
        col_idx = 0

        icon_size = S(100)  # bigger icons
        for brawler, stats in self.match_history.items():
            if brawler == "total":
                continue
            icon_path = f"./api/assets/brawler_icons/{brawler}.png"
            if not os.path.exists(icon_path):
                icon_img = None
            else:
                pil_img = Image.open(icon_path).resize((icon_size, icon_size))
                icon_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(icon_size, icon_size))

            total_games = stats["victory"] + stats["defeat"] + stats.get("draw", 0)
            if total_games == 0:
                wr = lr = dr = 0
            else:
                wr = round(100 * stats["victory"] / total_games, 1)
                lr = round(100 * stats["defeat"] / total_games, 1)
                dr = round(100 * stats.get("draw", 0) / total_games, 1)

            cell_frame = ctk.CTkFrame(
                scroll_frame,
                width=S(200),
                height=S(220),
                corner_radius=S(8)
            )
            cell_frame.grid(row=row_idx, column=col_idx, padx=S(15), pady=S(15))

            # Icon
            if icon_img:
                icon_label = ctk.CTkLabel(cell_frame, image=icon_img, text="")
                icon_label.pack(pady=S(5))

            # Brawler name & total games
            text_label = ctk.CTkLabel(
                cell_frame,
                text=f"{brawler}\n{total_games} games",
                font=(FONT_FAMILY, S(16), "bold")
            )
            text_label.pack()

            stats_frame = ctk.CTkFrame(cell_frame, fg_color="transparent")
            stats_frame.pack(pady=S(5))

            # Win in green
            color_win = "#2ecc71"

            # Loss in red
            color_loss = "#e74c3c"
            color_draw = COLORS["muted"]

            lbl_win = ctk.CTkLabel(
                stats_frame,
                text=f"{wr}%",
                font=(FONT_FAMILY, S(14), "bold"),
                text_color=color_win
            )
            lbl_win.pack(side="left", padx=S(5))

            lbl_loss = ctk.CTkLabel(
                stats_frame,
                text=f"{lr}%",
                font=(FONT_FAMILY, S(14), "bold"),
                text_color=color_loss
            )
            lbl_loss.pack(side="left", padx=S(5))

            if dr > 0:
                lbl_draw = ctk.CTkLabel(
                    stats_frame,
                    text=f"{dr}%",
                    font=(FONT_FAMILY, S(14), "bold"),
                    text_color=color_draw
                )
                lbl_draw.pack(side="left", padx=S(5))

            col_idx += 1
            if col_idx >= max_cols:
                col_idx = 0
                row_idx += 1

    # ---------------------------------------------------------------------------------------------
    #  On Start => close window + callback
    # ---------------------------------------------------------------------------------------------
    def _on_start(self):
        self._should_continue = True
        self._request_close()

