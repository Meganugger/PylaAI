import customtkinter as ctk
import pyautogui

from utils import get_dpi_scale

ORIG_SCREEN_WIDTH = 1920
ORIG_SCREEN_HEIGHT = 1080
FONT_FAMILY = "Segoe UI"

COLORS = {
    "bg": "#0F141B",
    "surface": "#171E28",
    "surface_alt": "#212A36",
    "surface_alt_2": "#273241",
    "border": "#313D4E",
    "text": "#F3F6FA",
    "muted": "#AEB8C6",
    "accent": "#4E86FF",
    "accent_hover": "#6A99FF",
    "accent_soft": "#284C8F",
    "success": "#60C58A",
    "danger": "#F06F6F",
    "warning": "#F1B860",
    "link": "#8FBCFF",
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
