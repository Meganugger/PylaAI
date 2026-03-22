import os
import sys

import customtkinter as ctk  # Import the customtkinter library
from gui.api import check_if_exists
from gui.theme import COLORS, S, apply_appearance, center_window, font
from utils import api_base_url
sys.path.append(os.path.abspath('../'))
from utils import update_toml_file, load_toml_as_dict


def login(logged_in_setter=None):
    login_succeeded = False

    if api_base_url == "localhost":
        if callable(logged_in_setter):
            logged_in_setter(True)
        return True

    def validate_api_key(api_key):
        return check_if_exists(api_key)

    def finish_login(success, app=None):
        nonlocal login_succeeded
        login_succeeded = success
        if callable(logged_in_setter):
            logged_in_setter(success)
        if app is not None and app.winfo_exists():
            app.destroy()

    def on_login_button_click():
        api_key = api_key_entry.get()
        if validate_api_key(api_key):
            result_label.configure(text="API key accepted.", text_color=COLORS["success"])
            update_toml_file("./cfg/login.toml", {"key": api_key})
            finish_login(True, app)
            return
        else:
            result_label.configure(text="API key not recognized.", text_color=COLORS["danger"])

    login_data = load_toml_as_dict('./cfg/login.toml')
    auth_key = login_data.get('key', "")
    if auth_key:
        if validate_api_key(auth_key):
            if callable(logged_in_setter):
                logged_in_setter(True)
            return True

    app = ctk.CTk()
    app.title('Pyla Sign In')
    login_width = S(620)
    login_height = S(290)
    app.geometry(f'{login_width}x{login_height}')
    app.resizable(False, False)
    apply_appearance()
    app.configure(fg_color=COLORS["bg"])
    center_window(app, login_width, login_height)

    card = ctk.CTkFrame(
        app,
        fg_color=COLORS["surface"],
        corner_radius=S(16),
        border_width=1,
        border_color=COLORS["border"]
    )
    card.pack(expand=True, fill="both", padx=S(24), pady=S(24))

    label = ctk.CTkLabel(card, text="Enter your API key", font=font(22, "bold"), text_color=COLORS["text"])
    label.pack(pady=(S(22), S(6)))

    helper_label = ctk.CTkLabel(
        card,
        text="Paste the key linked to your Pyla account to continue.",
        font=font(13),
        text_color=COLORS["muted"],
        wraplength=S(460),
        justify="center"
    )
    helper_label.pack(pady=(0, S(14)))

    api_key_entry = ctk.CTkEntry(
        card,
        placeholder_text="Paste your API key",
        font=font(15),
        width=S(420),
        height=S(42),
        fg_color=COLORS["surface_alt"],
        border_color=COLORS["border"],
        text_color=COLORS["text"]
    )
    api_key_entry.pack(pady=(0, S(14)))

    login_button = ctk.CTkButton(
        card,
        text="Continue",
        command=on_login_button_click,
        font=font(16, "bold"),
        width=S(180),
        height=S(42),
        corner_radius=S(10),
        fg_color=COLORS["accent"],
        hover_color=COLORS["accent_hover"],
        text_color=COLORS["text"]
    )
    login_button.pack()

    result_label = ctk.CTkLabel(card, text="", font=font(13), text_color=COLORS["muted"])
    result_label.pack(pady=(S(12), 0))

    app.mainloop()
    return login_succeeded
