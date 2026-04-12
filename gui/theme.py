import customtkinter as ctk
import pyautogui

from utils import get_dpi_scale

ORIG_SCREEN_WIDTH = 1920
ORIG_SCREEN_HEIGHT = 1080
FONT_FAMILY = "Bahnschrift"
FONT_FAMILY_ALT = "Segoe UI"

COLORS = {
    "bg": "#090A0E",
    "bg_alt": "#0D1016",
    "sidebar": "#07080C",
    "surface": "#11141B",
    "surface_alt": "#171C25",
    "surface_alt_2": "#1E2530",
    "surface_elevated": "#242D3A",
    "border": "#2C3442",
    "border_strong": "#3C4657",
    "text": "#F4F7FB",
    "text_bright": "#FFFFFF",
    "muted": "#9DA7B8",
    "muted_alt": "#778194",
    "accent": "#ED2A2A",
    "accent_hover": "#FF4B36",
    "accent_soft": "#5D1E23",
    "accent_muted": "#F18E54",
    "accent_dim": "#A3645C",
    "success": "#4FD08E",
    "danger": "#F16C6C",
    "warning": "#F5BC52",
    "info": "#72B7FF",
    "gold": "#FFD166",
    "link": "#93C5FD",
}

TYPOGRAPHY = {
    "eyebrow": 10,
    "label": 12,
    "body": 13,
    "body_large": 15,
    "title": 24,
    "hero": 30,
}

RADIUS = {
    "sm": 8,
    "md": 12,
    "lg": 18,
    "xl": 24,
}

SPACING = {
    "xs": 6,
    "sm": 10,
    "md": 16,
    "lg": 24,
    "xl": 32,
}


def _compute_scale():
    width, height = pyautogui.size()
    width_ratio = width / ORIG_SCREEN_WIDTH
    height_ratio = height / ORIG_SCREEN_HEIGHT
    scale_factor = min(width_ratio, height_ratio)
    return scale_factor * (96 / get_dpi_scale())


UI_SCALE = _compute_scale()


def S(value):
    return max(1, int(value * UI_SCALE))


def font(size, weight="normal"):
    return (FONT_FAMILY, S(size), weight)


def ui_font(size, weight="normal"):
    return (FONT_FAMILY_ALT, S(size), weight)


def apply_appearance():
    ctk.set_appearance_mode("dark")


def center_window(window, width=None, height=None):
    window.update_idletasks()
    if width is None:
        width = max(window.winfo_width(), window.winfo_reqwidth())
    if height is None:
        height = max(window.winfo_height(), window.winfo_reqheight())

    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    pos_x = max((screen_width - width) // 2, 0)
    pos_y = max((screen_height - height) // 2, 0)
    window.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
