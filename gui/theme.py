import customtkinter as ctk
import pyautogui

from utils import get_dpi_scale

ORIG_SCREEN_WIDTH = 1920
ORIG_SCREEN_HEIGHT = 1080
FONT_FAMILY = "Segoe UI"

COLORS = {
    "bg": "#14171D",
    "surface": "#1D222B",
    "surface_alt": "#242B36",
    "border": "#39414F",
    "text": "#F4F6F8",
    "muted": "#B8C0CC",
    "accent": "#C7514A",
    "accent_hover": "#D8625A",
    "accent_soft": "#8F312E",
    "success": "#5EBB73",
    "danger": "#E46B6B",
    "link": "#6EB4FF",
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
