# Win32 overlay for debug visualization (runs in a thread, ~30fps)

import ctypes
import ctypes.wintypes as wt
import math
import threading
import time
from collections import deque

# dPI Awareness (call BEFORE any Win32 window ops)
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)   # PROCESS_SYSTEM_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

import numpy as np
import win32gui  # type: ignore[import-untyped]
import win32con  # type: ignore[import-untyped]

from PIL import Image, ImageDraw, ImageFont
from utils import load_toml_as_dict

# win32 constants / types
AC_SRC_OVER  = 0x00
AC_SRC_ALPHA = 0x01
ULW_ALPHA    = 0x02
WS_EX_LAYERED      = 0x00080000
WS_EX_TRANSPARENT   = 0x00000020
WS_EX_TOPMOST       = 0x00000008
WS_EX_TOOLWINDOW    = 0x00000080
WS_EX_NOACTIVATE    = 0x08000000
WS_POPUP             = 0x80000000
GWL_EXSTYLE          = -20
HWND_TOPMOST         = -1
SWP_NOACTIVATE       = 0x0010
SWP_NOMOVE           = 0x0002
SWP_NOSIZE           = 0x0001
SWP_SHOWWINDOW       = 0x0040

user32 = ctypes.windll.user32
gdi32  = ctypes.windll.gdi32


class _BLENDFUNCTION(ctypes.Structure):
    _fields_ = [
        ("BlendOp",             ctypes.c_byte),
        ("BlendFlags",          ctypes.c_byte),
        ("SourceConstantAlpha", ctypes.c_byte),
        ("AlphaFormat",         ctypes.c_byte),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize",          ctypes.c_uint32),
        ("biWidth",         ctypes.c_int32),
        ("biHeight",        ctypes.c_int32),
        ("biPlanes",        ctypes.c_uint16),
        ("biBitCount",      ctypes.c_uint16),
        ("biCompression",   ctypes.c_uint32),
        ("biSizeImage",     ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed",       ctypes.c_uint32),
        ("biClrImportant",  ctypes.c_uint32),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", _BITMAPINFOHEADER),
    ]


# overlay colours (RGBA)
CLR_PLAYER_DOT   = (0, 255, 0, 230)
CLR_PLAYER_RING  = (255, 255, 255, 230)
CLR_ATTACK_RANGE = (255, 140, 0, 100)
CLR_SAFE_RANGE   = (0, 220, 100, 70)
CLR_SUPER_RANGE  = (170, 102, 255, 100)
CLR_SUPER_GLOW   = (255, 220, 50)           # pulsating gold (alpha varies)
CLR_MOVE_ARROW   = (0, 191, 255, 200)
CLR_LOS_CLEAR    = (0, 255, 0, 120)
CLR_LOS_BLOCKED  = (255, 68, 68, 120)
CLR_TARGET_DOT   = (255, 0, 255, 220)
CLR_ENEMY_DOT    = (255, 68, 68, 200)
CLR_ENEMY_BOX    = (255, 50, 50, 160)
CLR_TEAMMATE_BOX = (50, 140, 255, 160)
CLR_WALL_FILL    = (100, 70, 35, 45)
CLR_WALL_OUTLINE = (200, 160, 70, 120)
CLR_DESTROYED_WALL = (255, 60, 60, 25)
CLR_HUD_BG       = (0, 0, 0, 160)
CLR_HUD_TEXT     = (255, 255, 255, 220)
CLR_HUD_NAME     = (255, 220, 50, 255)
CLR_HUD_LABEL    = (180, 180, 180, 200)
CLR_HUD_VALUE    = (255, 255, 255, 230)
CLR_STORM_RING   = (100, 60, 200, 140)
CLR_IN_STORM_WRN = (255, 40, 40, 220)
CLR_CHOKE_PT     = (255, 200, 0, 120)
CLR_DANGER_BUSH  = (200, 100, 0, 40)
CLR_HP_BG        = (40, 40, 40, 180)
CLR_HP_HIGH      = (50, 220, 50, 230)
CLR_HP_MID       = (255, 200, 0, 230)
CLR_HP_LOW       = (255, 50, 50, 230)
CLR_AMMO_FULL    = (255, 210, 40, 230)
CLR_AMMO_EMPTY   = (80, 80, 80, 130)
CLR_GHOST_DOT    = (255, 80, 80)            # base colour (alpha varies)
CLR_DECISION_BG  = (0, 0, 0, 170)
CLR_DECISION_TXT = (255, 255, 255, 240)
CLR_BADGE_DISENGAGE  = (255, 80, 80, 200)
CLR_BADGE_ADVANTAGE  = (50, 220, 50, 200)
CLR_BADGE_STUTTER    = (60, 160, 255, 200)
CLR_BADGE_PEEK       = (180, 100, 255, 200)
CLR_TARGET_OUTLINE   = (255, 0, 255, 220)
CLR_FPS_TEXT         = (120, 255, 120, 160)
CLR_OBJECTIVE        = (255, 230, 50, 200)
CLR_LANE_LINE        = (80, 200, 255, 80)
CLR_REGEN            = (60, 255, 120, 200)
CLR_SHIELD           = (80, 180, 255, 200)
CLR_ENEMY_VEL        = (255, 120, 50, 160)
CLR_RELOAD_WINDOW    = (255, 255, 50, 200)
CLR_AGGR_LOW         = (80, 180, 255, 200)
CLR_AGGR_HIGH        = (255, 80, 50, 200)
CLR_MATE_HP_BG       = (40, 40, 40, 160)
CLR_MATE_HP_FILL     = (60, 160, 255, 200)
CLR_SHADOW           = (0, 0, 0, 180)       # text drop-shadow

# decision / BT-state colour map (banner + BT path label)
_BT_COLOR_MAP: dict[str, tuple[int, ...]] = {
    "DODGE":      (0, 220, 255, 230),
    "EMERGENCY":  (255, 50, 50, 240),
    "COMBAT":     (255, 160, 40, 230),
    "RUSH":       (255, 100, 30, 230),
    "RETREAT":    (200, 120, 255, 220),
    "RANGED":     (80, 200, 255, 230),
    "LOWH":       (255, 80, 80, 230),
    "FINISH":     (255, 200, 50, 230),
    "NOENEMY":    (180, 180, 180, 200),
    "DEAD":       (130, 130, 130, 160),
    "FALLBACK":   (180, 180, 180, 200),
    "DEFAULT":    (220, 220, 220, 220),
}

def _bt_color_for(text: str) -> tuple[int, ...]:
    """Pick banner colour based on keywords in the text."""
    up = text.upper()
    for key, clr in _BT_COLOR_MAP.items():
        if key in up:
            return clr
    return _BT_COLOR_MAP["DEFAULT"]

# Isometric camera correction (Y compressed)
ISO_Y_FACTOR = 0.8

# Direction vectors for WASD
_DIR_VECTORS = {
    "W": (0, -1), "A": (-1, 0), "S": (0, 1), "D": (1, 0),
    "AW": (-0.707, -0.707), "DW": (0.707, -0.707),
    "AS": (-0.707, 0.707), "DS": (0.707, 0.707),
}

_CONFIG_RELOAD_INTERVAL = 5.0
_HWND_SEARCH_INTERVAL   = 2.0
_MAX_LOS = 8
_MAX_WALLS = 120
_GHOST_MAX_AGE = 3.0

# Emulator window title patterns
_EMULATOR_HINTS: dict[str, list[str]] = {
    "bluestacks": ["bluestacks"],
    "ldplayer":   ["ldplayer", "dnplayer"],
    "nox":        ["noxplayer", "nox app player"],
    "mumu":       ["mumu", "nemu"],
    "memu":       ["memu"],
    "scrcpy":     ["scrcpy"],
}


# helpers
def _find_game_hwnd(emulator_name: str = ""):
    emu_key = emulator_name.strip().lower().replace(" ", "")
    hints: list[str] = []
    for key, values in _EMULATOR_HINTS.items():
        if not emu_key or key in emu_key or emu_key in key:
            hints.extend(values)
    if not hints:
        hints = [emu_key] if emu_key else [
            v for vals in _EMULATOR_HINTS.values() for v in vals
        ]
    result = []
    def _enum(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd).lower()
            if title and any(h in title for h in hints):
                result.append(hwnd)
    win32gui.EnumWindows(_enum, None)
    return result[0] if result else None


def _create_overlay_hwnd():
    """Create an invisible layered + click-through + topmost popup window."""
    wc_name = "PylaAIOverlay"
    wndclass = win32gui.WNDCLASS()
    wndclass.lpfnWndProc = {win32con.WM_DESTROY: lambda *a: None}
    wndclass.lpszClassName = wc_name
    wndclass.hCursor = user32.LoadCursorW(0, 32512)
    try:
        win32gui.RegisterClass(wndclass)
    except Exception:
        pass  # already registered

    ex_style = (WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST |
                WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE)
    hwnd = win32gui.CreateWindowEx(
        ex_style, wc_name, "PylaAI Overlay",
        WS_POPUP,
        0, 0, 1, 1,
        0, 0, 0, None,
    )
    user32.ShowWindow(hwnd, 8)  # SW_SHOWNA
    return hwnd


def _update_layered(hwnd, img: Image.Image, x: int, y: int, opacity: int):
    """Atomically blit a PIL RGBA image onto a layered window at (x, y)."""
    w, h = img.size
    if w < 1 or h < 1:
        return

    raw = img.tobytes("raw", "BGRA")

    hdc_screen = user32.GetDC(0)
    hdc_mem = gdi32.CreateCompatibleDC(hdc_screen)

    bi = _BITMAPINFO()
    bi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
    bi.bmiHeader.biWidth = w
    bi.bmiHeader.biHeight = -h  # top-down
    bi.bmiHeader.biPlanes = 1
    bi.bmiHeader.biBitCount = 32
    bi.bmiHeader.biCompression = 0

    ppvBits = ctypes.c_void_p()
    hbmp = gdi32.CreateDIBSection(
        hdc_mem, ctypes.byref(bi), 0, ctypes.byref(ppvBits), None, 0
    )
    if not hbmp:
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)
        return

    old_bmp = 0
    try:
        ctypes.memmove(ppvBits, raw, len(raw))
        old_bmp = gdi32.SelectObject(hdc_mem, hbmp)

        pt_pos = wt.POINT(x, y)
        sz = wt.SIZE(w, h)
        pt_src = wt.POINT(0, 0)
        blend = _BLENDFUNCTION(AC_SRC_OVER, 0, opacity, AC_SRC_ALPHA)

        user32.UpdateLayeredWindow(
            hwnd, hdc_screen, ctypes.byref(pt_pos), ctypes.byref(sz),
            hdc_mem, ctypes.byref(pt_src), 0, ctypes.byref(blend), ULW_ALPHA,
        )
    finally:
        # Always release GDI resources to prevent handle leaks over days
        if old_bmp:
            gdi32.SelectObject(hdc_mem, old_bmp)
        gdi32.DeleteObject(hbmp)
        gdi32.DeleteDC(hdc_mem)
        user32.ReleaseDC(0, hdc_screen)


def _hp_color(pct):
    if pct > 60:
        return CLR_HP_HIGH
    elif pct > 30:
        return CLR_HP_MID
    return CLR_HP_LOW


def _fmt_time(seconds):
    """Format seconds to M:SS string."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


# main overlay class
class VisualOverlay:
    """
    Flicker-free transparent overlay - Win32 layered window +
    PIL offscreen rendering + atomic UpdateLayeredWindow blit.
    """

    def __init__(self):
        self._config = self._load_config()
        self._running = True
        self._data: dict = {}
        self._data_lock = threading.Lock()
        self._last_config_reload = time.time()
        self._last_hwnd_search = 0.0
        self._game_hwnd = None
        self._overlay_hwnd = None
        self._prev_client = (0, 0, 0, 0)

        # cached fonts (sized for readability on 720p–1080p overlay)
        self._font_8 = self._load_font(11)
        self._font_9 = self._load_font(12)
        self._font_10 = self._load_font(14)
        self._font_11 = self._load_font(15)
        self._font_13 = self._load_font(17)
        self._font_15 = self._load_font(19)
        self._font_18 = self._load_font(22)
        self._font_20 = self._load_font(24)

        # fPS tracking
        self._frame_times: deque = deque(maxlen=20)
        self._fps = 0.0
        self._data_gen = 0          # bumped on each .update()
        self._last_drawn_gen = -1   # skip draw if data unchanged

        # position smoothing
        self._smooth_px = 0.0
        self._smooth_py = 0.0
        self._smooth_init = False
        self._smooth_enemies: dict = {}   # idx -> (sx, sy)

        # Wall smoothing: keep a small history of displayed walls so they
        # don't flicker.  Each entry is (x1, y1, x2, y2, age) where age
        # counts how many frames since first seen.  Walls are matched by
        # IoU overlap across frames and lerped toward new positions.
        self._smooth_walls: list = []           # [(x1,y1,x2,y2, age, alpha)]
        self._WALL_LERP = 0.55                  # positional smoothing factor
        self._WALL_FADE_IN  = 4                 # frames to reach full opacity
        self._WALL_FADE_OUT = 6                 # frames to fade out when gone
        self._WALL_IOU_THRESH = 0.20            # min IoU to consider same wall

        # reusable canvas
        self._canvas_size = (0, 0)
        self._canvas: Image.Image | None = None

        # decision banner
        self._last_decision = ""
        self._decision_update_time = 0.0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _load_font(size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype("segoeui.ttf", size)
        except Exception:
            try:
                return ImageFont.truetype("arial.ttf", size)
            except Exception:
                return ImageFont.load_default()

    # public API
    def update(self, *, player_pos=None, player_bbox=None, enemies=None,
               walls=None, brawler_ranges=None, movement="",
               is_super_ready=False, teammates=None, hittable_map=None,
               target_bbox=None, frame_w=None, frame_h=None,
               is_dead=False, game_state="match",
               brawler_name="", brawler_info=None,
               storm_center=None, storm_radius=9999, gas_active=False,
               gas_density_map=None, in_storm=False,
               choke_points=None, bushes=None,
               player_hp=100, enemy_hp=-1,
               per_enemy_hp=None,
               hp_confidence_player=1.0, hp_confidence_enemy=1.0,
               ammo=3, max_ammo=3,
               decision_reason="",
               match_phase="early", our_score=0, their_score=0,
               death_count=0, kills=0,
               target_info=None,
               last_known_enemies=None,
               behavior_flags=None,
               # extended fields
               enemy_velocity=None, enemy_move_dir="none",
               teammate_hp_data=None,
               is_gadget_ready=False, is_hypercharge_ready=False,
               hold_super=False,
               respawn_shield_active=False, respawn_shield_until=0.0,
               is_regenerating=False,
               match_start_time=0.0,
               game_mode_name="",
               assigned_lane="", lane_center=None,
               objective_pos=None,
               spawn_side="",
               aggression_modifier=1.0,
               destroyed_wall_zones=None,
               patrol_phase="idle", no_enemy_duration=0.0,
               enemy_reload_window=False,
               stuck_level=0,
               bot_ips=0,
               # aI Subsystem fields
               bt_active_path=None,
               projectiles=None,
               spatial_grid=None,
               combo_state="",
               aim_stats=None,
               enemy_tracks=None,
               opponent_summary=None,
               dodge_direction=None,
               pathfinder_path=None,
               **kwargs):
        with self._data_lock:
            self._data_gen += 1
            self._data = {
                "player_pos": player_pos,
                "player_bbox": player_bbox,
                "enemies": enemies or [],
                "walls": walls or [],
                "brawler_ranges": brawler_ranges,
                "movement": movement.upper() if movement else "",
                "is_super_ready": is_super_ready,
                "teammates": teammates or [],
                "hittable_map": hittable_map or {},
                "target_bbox": target_bbox,
                "frame_w": frame_w,
                "frame_h": frame_h,
                "is_dead": is_dead,
                "game_state": game_state,
                "brawler_name": brawler_name or "",
                "brawler_info": brawler_info or {},
                "storm_center": storm_center,
                "storm_radius": storm_radius,
                "gas_active": gas_active,
                "gas_density_map": gas_density_map,
                "in_storm": in_storm,
                "choke_points": choke_points or [],
                "bushes": bushes or [],
                "player_hp": player_hp,
                "enemy_hp": enemy_hp,
                "per_enemy_hp": per_enemy_hp or {},
                "hp_confidence_player": hp_confidence_player,
                "hp_confidence_enemy": hp_confidence_enemy,
                "ammo": ammo,
                "max_ammo": max_ammo,
                "decision_reason": decision_reason or "",
                "match_phase": match_phase or "early",
                "our_score": our_score,
                "their_score": their_score,
                "death_count": death_count,
                "kills": kills,
                "target_info": target_info or {},
                "last_known_enemies": last_known_enemies or [],
                "behavior_flags": behavior_flags or {},
                # Extended
                "enemy_velocity": enemy_velocity or (0, 0),
                "enemy_move_dir": enemy_move_dir or "none",
                "teammate_hp_data": teammate_hp_data or [],
                "is_gadget_ready": is_gadget_ready,
                "is_hypercharge_ready": is_hypercharge_ready,
                "hold_super": hold_super,
                "respawn_shield_active": respawn_shield_active,
                "respawn_shield_until": respawn_shield_until,
                "is_regenerating": is_regenerating,
                "match_start_time": match_start_time,
                "game_mode_name": game_mode_name or "",
                "assigned_lane": assigned_lane or "",
                "lane_center": lane_center,
                "objective_pos": objective_pos,
                "spawn_side": spawn_side or "",
                "aggression_modifier": aggression_modifier,
                "destroyed_wall_zones": destroyed_wall_zones or [],
                "patrol_phase": patrol_phase or "idle",
                "no_enemy_duration": no_enemy_duration,
                "enemy_reload_window": enemy_reload_window,
                "stuck_level": stuck_level,
                "bot_ips": bot_ips,
                # AI subsystem data
                "bt_active_path": bt_active_path or [],
                "projectiles": projectiles or [],
                "spatial_grid": spatial_grid,
                "combo_state": combo_state or "",
                "aim_stats": aim_stats or {},
                "enemy_tracks": enemy_tracks or [],
                "opponent_summary": opponent_summary or {},
                "dodge_direction": dodge_direction,
                "pathfinder_path": pathfinder_path or [],
            }

    def set_game_state(self, state: str):
        with self._data_lock:
            self._data["game_state"] = state

    def reload_config(self):
        self._config = self._load_config()

    def stop(self):
        self._running = False

    # config
    @staticmethod
    def _load_config():
        gc = load_toml_as_dict("cfg/general_config.toml")
        _yn = lambda v: str(v).lower() in ("yes", "true", "1")
        return {
            "enabled":           _yn(gc.get("visual_overlay_enabled", "no")),
            "player_dot":        _yn(gc.get("visual_overlay_player_dot", "yes")),
            "attack_range":      _yn(gc.get("visual_overlay_attack_range", "yes")),
            "safe_range":        _yn(gc.get("visual_overlay_safe_range", "yes")),
            "super_range":       _yn(gc.get("visual_overlay_super_range", "yes")),
            "movement_arrow":    _yn(gc.get("visual_overlay_movement_arrow", "yes")),
            "los_all_enemies":   _yn(gc.get("visual_overlay_los_all_enemies", "yes")),
            "enemies":           _yn(gc.get("visual_overlay_enemies", "yes")),
            "teammates":         _yn(gc.get("visual_overlay_teammates", "yes")),
            "walls":             _yn(gc.get("visual_overlay_walls", "yes")),
            "hide_when_dead":    _yn(gc.get("visual_overlay_hide_when_dead", "yes")),
            "brawler_hud":       _yn(gc.get("visual_overlay_brawler_hud", "yes")),
            "gas_zone":          _yn(gc.get("visual_overlay_gas_zone", "yes")),
            "danger_zones":      _yn(gc.get("visual_overlay_danger_zones", "yes")),
            "decision_banner":   _yn(gc.get("visual_overlay_decision_banner", "yes")),
            "hp_bars":           _yn(gc.get("visual_overlay_hp_bars", "yes")),
            "target_info":       _yn(gc.get("visual_overlay_target_info", "yes")),
            "ghost_dots":        _yn(gc.get("visual_overlay_ghost_dots", "yes")),
            "opacity":           max(30, min(255, int(gc.get("visual_overlay_opacity", 180)))),
            "emulator":          str(gc.get("current_emulator", "")),
            # Emulator chrome offsets (toolbar / sidebar pixels to exclude)
            "chrome_top":        max(0, int(gc.get("emulator_chrome_top", 0))),
            "chrome_right":      max(0, int(gc.get("emulator_chrome_right", 0))),
            "chrome_bottom":     max(0, int(gc.get("emulator_chrome_bottom", 0))),
            "chrome_left":       max(0, int(gc.get("emulator_chrome_left", 0))),
            "hud_position":      str(gc.get("visual_overlay_hud_position", "top-left")).lower(),
            # AI subsystem overlays
            "bt_path":           _yn(gc.get("visual_overlay_bt_path", "yes")),
            "projectiles":       _yn(gc.get("visual_overlay_projectiles", "yes")),
            "spatial_grid":      _yn(gc.get("visual_overlay_spatial_grid", "no")),
            "combo_queue":       _yn(gc.get("visual_overlay_combo_queue", "yes")),
            "aim_stats":         _yn(gc.get("visual_overlay_aim_stats", "yes")),
        }

    # main loop
    def _run(self):
        self._overlay_hwnd = _create_overlay_hwnd()
        self._loop()

    def _loop(self):
        # --- Windows message pump structures ---
        # Every thread that owns a window MUST pump its message queue.
        # Without this, Windows marks the whole process "Not Responding"
        # after ~5 s of unprocessed messages.
        _PM_REMOVE = 0x0001

        class _MSG(ctypes.Structure):
            _fields_ = [
                ("hwnd",    wt.HWND),
                ("message", wt.UINT),
                ("wParam",  wt.WPARAM),
                ("lParam",  wt.LPARAM),
                ("time",    wt.DWORD),
                ("pt",      wt.POINT),
            ]

        _PeekMessageW    = user32.PeekMessageW
        _TranslateMessage = user32.TranslateMessage
        _DispatchMessageW = user32.DispatchMessageW
        _msg = _MSG()
        _msg_ptr = ctypes.byref(_msg)

        _tick_errors = 0

        while self._running:
            # Drain all pending window messages (non-blocking)
            while _PeekMessageW(_msg_ptr, 0, 0, 0, _PM_REMOVE):
                _TranslateMessage(_msg_ptr)
                _DispatchMessageW(_msg_ptr)

            t0 = time.perf_counter()
            try:
                self._tick()
                _tick_errors = 0  # reset on success
            except Exception as _te:
                _tick_errors += 1
                if _tick_errors <= 3 or _tick_errors % 500 == 0:
                    print(f"[OVERLAY] Tick error ({_tick_errors}): {_te}")
                if _tick_errors >= 3000:
                    print("[OVERLAY] Too many consecutive tick errors, stopping overlay thread")
                    self._running = False
                    break
            elapsed = time.perf_counter() - t0
            self._frame_times.append(elapsed)
            if len(self._frame_times) >= 5:
                avg = sum(self._frame_times) / len(self._frame_times)
                self._fps = 1.0 / avg if avg > 0 else 0.0
            # Target ~30 FPS (0.033s) for smooth visual updates
            time.sleep(max(0.002, 0.033 - elapsed))

        try:
            if self._overlay_hwnd:
                win32gui.DestroyWindow(self._overlay_hwnd)
        except Exception:
            pass

    # --- mAIN DRAW TICK ---
    def _tick(self):
        now = time.time()

        # Hot-reload config
        if now - self._last_config_reload > _CONFIG_RELOAD_INTERVAL:
            self._last_config_reload = now
            try:
                self._config = self._load_config()
            except Exception:
                pass

        cfg = self._config
        if not cfg["enabled"]:
            self._hide()
            return

        # Find game window
        if not self._game_hwnd or now - self._last_hwnd_search > _HWND_SEARCH_INTERVAL:
            self._last_hwnd_search = now
            self._game_hwnd = _find_game_hwnd(cfg.get("emulator", ""))

        if not self._game_hwnd:
            self._hide()
            return

        try:
            client_xy = win32gui.ClientToScreen(self._game_hwnd, (0, 0))
            crect = win32gui.GetClientRect(self._game_hwnd)
            raw_cx, raw_cy = client_xy
            raw_cw, raw_ch = crect[2], crect[3]
        except Exception:
            self._game_hwnd = None
            self._hide()
            return

        if raw_cw < 50 or raw_ch < 50:
            self._hide()
            return

        # apply emulator chrome offsets
        # Subtract toolbar (top), sidebar (right), etc. so the overlay
        # canvas covers ONLY the actual game area inside the emulator.
        ct = cfg.get("chrome_top", 0)
        cr = cfg.get("chrome_right", 0)
        cb = cfg.get("chrome_bottom", 0)
        cl = cfg.get("chrome_left", 0)
        cx = raw_cx + cl
        cy = raw_cy + ct
        cw = max(50, raw_cw - cl - cr)
        ch = max(50, raw_ch - ct - cb)

        with self._data_lock:
            d = dict(self._data)
            gen = self._data_gen

        # Allow periodic redraws even when data hasn't changed so that
        # smoothing animations (wall fade-in/out, lerp) keep running.
        # Skip only if data is stale AND no wall animations are active.
        data_changed = gen != self._last_drawn_gen
        has_anim = bool(self._smooth_walls)
        if not data_changed and not has_anim and self._prev_client != (0, 0, 0, 0):
            return
        self._last_drawn_gen = gen

        game_state = d.get("game_state", "match")
        if game_state and game_state != "match":
            self._hide()
            return

        is_dead = d.get("is_dead", False)
        if is_dead and cfg.get("hide_when_dead", True):
            self._hide()
            return

        player_pos = d.get("player_pos")
        if not player_pos:
            self._hide()
            return

        # canvas
        if self._canvas_size != (cw, ch):
            self._canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            self._canvas_size = (cw, ch)
        else:
            self._canvas.paste((0, 0, 0, 0), (0, 0, cw, ch))
        img = self._canvas
        draw = ImageDraw.Draw(img)

        # Scaling
        frame_w = d.get("frame_w") or cw
        frame_h = d.get("frame_h") or ch
        sx = cw / frame_w if frame_w else 1.0
        sy = ch / frame_h if frame_h else 1.0

        # Player foot position (75% down bbox)
        player_bbox = d.get("player_bbox")
        if player_bbox:
            raw_px = ((player_bbox[0] + player_bbox[2]) / 2) * sx
            raw_py = (player_bbox[1] * 0.25 + player_bbox[3] * 0.75) * sy
        else:
            raw_px = player_pos[0] * sx
            raw_py = player_pos[1] * sy

        # Lerp for smooth tracking
        _LERP = 0.85
        if not self._smooth_init:
            self._smooth_px = raw_px
            self._smooth_py = raw_py
            self._smooth_init = True
        else:
            self._smooth_px += (raw_px - self._smooth_px) * _LERP
            self._smooth_py += (raw_py - self._smooth_py) * _LERP
        px, py = self._smooth_px, self._smooth_py

        # Extract data
        enemies = d.get("enemies", [])
        raw_walls = d.get("walls", [])
        if len(raw_walls) > _MAX_WALLS:
            raw_walls.sort(key=lambda w: (w[2]-w[0])*(w[3]-w[1]), reverse=True)
            walls = raw_walls[:_MAX_WALLS]
        else:
            walls = raw_walls
        ranges_ = d.get("brawler_ranges")
        movement = d.get("movement", "")
        is_super_ready = d.get("is_super_ready", False)
        hittable_map = d.get("hittable_map", {})
        target_bbox = d.get("target_bbox")
        teammates = d.get("teammates", [])

        player_hp = d.get("player_hp", 100)
        enemy_hp = d.get("enemy_hp", -1)
        per_enemy_hp = d.get("per_enemy_hp", {})
        hp_conf_player = d.get("hp_confidence_player", 1.0)
        hp_conf_enemy = d.get("hp_confidence_enemy", 1.0)
        ammo = d.get("ammo", 3)
        max_ammo = d.get("max_ammo", 3)
        decision_reason = d.get("decision_reason", "")
        match_phase = d.get("match_phase", "early")
        our_score = d.get("our_score", 0)
        their_score = d.get("their_score", 0)
        death_count = d.get("death_count", 0)
        kills = d.get("kills", 0)
        target_info = d.get("target_info", {})
        last_known = d.get("last_known_enemies", [])
        behavior = d.get("behavior_flags", {})

        # Extended data
        enemy_vel = d.get("enemy_velocity", (0, 0))
        enemy_move_dir = d.get("enemy_move_dir", "none")
        teammate_hp_data = d.get("teammate_hp_data", [])
        is_gadget_ready = d.get("is_gadget_ready", False)
        is_hypercharge_ready = d.get("is_hypercharge_ready", False)
        hold_super = d.get("hold_super", False)
        respawn_shield = d.get("respawn_shield_active", False)
        respawn_shield_until = d.get("respawn_shield_until", 0.0)
        is_regen = d.get("is_regenerating", False)
        match_start = d.get("match_start_time", 0.0)
        game_mode_name = d.get("game_mode_name", "")
        assigned_lane = d.get("assigned_lane", "")
        lane_center = d.get("lane_center")
        objective_pos = d.get("objective_pos")
        aggression = d.get("aggression_modifier", 1.0)
        destroyed_zones = d.get("destroyed_wall_zones", [])
        patrol_phase = d.get("patrol_phase", "idle")
        no_enemy_dur = d.get("no_enemy_duration", 0.0)
        enemy_reload_win = d.get("enemy_reload_window", False)
        stuck_level = d.get("stuck_level", 0)
        bot_ips = d.get("bot_ips", 0)

        # --- l1: Storm / Gas zone ---
        storm_center = d.get("storm_center")
        storm_radius = d.get("storm_radius", 9999)
        gas_active = d.get("gas_active", False)
        gas_map = d.get("gas_density_map")
        in_storm = d.get("in_storm", False)

        if cfg.get("gas_zone") and storm_center and storm_radius < 5000:
            scx, scy = storm_center[0] * sx, storm_center[1] * sy
            self._draw_ellipse(draw, scx, scy, storm_radius * sx,
                               storm_radius * sy, CLR_STORM_RING, 2, "SAFE")

        if cfg.get("gas_zone") and gas_active and gas_map:
            rw, rh = cw / 3, ch / 3
            for (gx, gy), density in gas_map.items():
                if density < 0.12:
                    continue
                a = int(min(1.0, density * 2.0) * 80)
                fill = (int(40 + density * 160), int(200 - density * 80), 30, a)
                draw.rectangle([gx*rw, gy*rh, gx*rw+rw, gy*rh+rh], fill=fill)

        if cfg.get("gas_zone") and in_storm:
            txt = "\u26a0 IN STORM!"
            try:
                tw = int(draw.textlength(txt, font=self._font_20)) + 30
            except Exception:
                tw = 170
            th = 36
            sx0 = (cw - tw) // 2
            # Pulsating red warning banner
            pulse = int((math.sin(now * 6.0) + 1.0) * 0.5 * 60 + 30)
            draw.rounded_rectangle([sx0-4, 28, sx0+tw+4, 28+th+4],
                                   radius=th//2+2, fill=(255, 0, 0, pulse))
            draw.rounded_rectangle([sx0, 30, sx0+tw, 30+th],
                                   radius=th//2, fill=(40, 0, 0, 200),
                                   outline=(255, 50, 50, 180), width=2)
            self._text_shadow(draw, (sx0+14, 35), txt,
                              CLR_IN_STORM_WRN, self._font_20)

        # --- l2: Danger zones ---
        chokes = d.get("choke_points", [])
        if cfg.get("danger_zones") and chokes:
            for cp in chokes:
                cpx, cpy = cp[0] * sx, cp[1] * sy
                gw = cp[2] * sx * 0.4 if len(cp) > 2 else 16
                draw.polygon([
                    (cpx, cpy - gw * 0.6), (cpx + gw * 0.4, cpy),
                    (cpx, cpy + gw * 0.6), (cpx - gw * 0.4, cpy),
                ], fill=CLR_CHOKE_PT, outline=(255, 180, 0, 180))

        bushes = d.get("bushes", [])
        if cfg.get("danger_zones") and bushes:
            for b in bushes:
                draw.rectangle([b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy],
                               fill=CLR_DANGER_BUSH, outline=(180, 100, 0, 60))

        # --- l3: Destroyed wall zones ---
        if cfg.get("walls") and destroyed_zones:
            for dz in destroyed_zones:
                draw.rectangle([dz[0]*sx, dz[1]*sy, dz[2]*sx, dz[3]*sy],
                               fill=CLR_DESTROYED_WALL,
                               outline=(255, 80, 80, 80), width=1)

        # --- l3b: Wall rectangles (drawn early so they appear BEHIND circles/enemies) ---
        if cfg["walls"]:
            # Scale raw walls to overlay coords
            new_walls = []
            for w in (walls or []):
                wx1, wy1 = w[0]*sx, w[1]*sy
                wx2, wy2 = w[2]*sx, w[3]*sy
                if (wx2 - wx1) < 3 or (wy2 - wy1) < 3:
                    continue
                new_walls.append((wx1, wy1, wx2, wy2))

            # Match new walls to existing smooth-wall slots by IoU
            matched_new = set()
            matched_old = set()

            def _iou(a, b):
                ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
                ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
                inter = max(0, ix2-ix1)*max(0, iy2-iy1)
                aa = max(1, (a[2]-a[0])*(a[3]-a[1]))
                ab = max(1, (b[2]-b[0])*(b[3]-b[1]))
                return inter / (aa + ab - inter) if (aa+ab-inter) > 0 else 0

            for ni, nw in enumerate(new_walls):
                best_iou, best_oi = 0, -1
                for oi, ow in enumerate(self._smooth_walls):
                    if oi in matched_old:
                        continue
                    v = _iou(nw, ow[:4])
                    if v > best_iou:
                        best_iou, best_oi = v, oi
                if best_iou >= self._WALL_IOU_THRESH and best_oi >= 0:
                    matched_new.add(ni)
                    matched_old.add(best_oi)
                    ow = self._smooth_walls[best_oi]
                    L = self._WALL_LERP
                    sx1 = ow[0] + (nw[0]-ow[0])*L
                    sy1 = ow[1] + (nw[1]-ow[1])*L
                    sx2 = ow[2] + (nw[2]-ow[2])*L
                    sy2 = ow[3] + (nw[3]-ow[3])*L
                    new_age = min(ow[4]+1, self._WALL_FADE_IN)
                    new_alpha = min(1.0, ow[5] + 1.0/self._WALL_FADE_IN)
                    self._smooth_walls[best_oi] = (sx1, sy1, sx2, sy2, new_age, new_alpha)

            for ni, nw in enumerate(new_walls):
                if ni not in matched_new:
                    self._smooth_walls.append((*nw, 1, 1.0/self._WALL_FADE_IN))

            kept = []
            for oi, ow in enumerate(self._smooth_walls):
                if oi in matched_old:
                    kept.append(ow)
                else:
                    new_alpha = ow[5] - 1.0 / self._WALL_FADE_OUT
                    if new_alpha > 0.02:
                        kept.append((ow[0], ow[1], ow[2], ow[3], ow[4], new_alpha))
            self._smooth_walls = kept[:_MAX_WALLS]

            for sw in self._smooth_walls:
                a = max(0.0, min(1.0, sw[5]))
                if a < 0.03:
                    continue
                fill = (CLR_WALL_FILL[0], CLR_WALL_FILL[1], CLR_WALL_FILL[2],
                        int(CLR_WALL_FILL[3] * a))
                outline = (CLR_WALL_OUTLINE[0], CLR_WALL_OUTLINE[1], CLR_WALL_OUTLINE[2],
                           int(CLR_WALL_OUTLINE[3] * a))
                draw.rectangle([sw[0], sw[1], sw[2], sw[3]],
                               fill=fill, outline=outline)

        # --- l4: Lane indicator ---
        if assigned_lane and lane_center:
            lcx, lcy = lane_center[0] * sx, lane_center[1] * sy
            if assigned_lane in ("left", "center", "right"):
                draw.line([(lcx, 0), (lcx, ch)],
                          fill=CLR_LANE_LINE, width=2)
            else:
                draw.line([(0, lcy), (cw, lcy)],
                          fill=CLR_LANE_LINE, width=2)
            draw.text((lcx + 6, 6), assigned_lane.upper(),
                      fill=CLR_LANE_LINE, font=self._font_11)

        # --- l5: Objective waypoint ---
        if objective_pos:
            ox, oy = objective_pos[0] * sx, objective_pos[1] * sy
            s = 13
            draw.polygon([(ox, oy-s), (ox+s, oy), (ox, oy+s), (ox-s, oy)],
                         outline=CLR_OBJECTIVE, width=3)
            pulse = int((math.sin(now * 3.0) + 1.0) * 0.5 * 80 + 40)
            ring_r = 20
            draw.ellipse([ox-ring_r, oy-ring_r, ox+ring_r, oy+ring_r],
                         outline=(*CLR_OBJECTIVE[:3], pulse), width=2)

        # --- l6: Ghost dots ---
        if cfg.get("ghost_dots") and last_known:
            vis_buckets = set()
            for en in enemies:
                vis_buckets.add((int((en[0]+en[2])/2) // 20,
                                 int((en[1]+en[3])/2) // 20))
            for entry in last_known:
                if len(entry) < 3:
                    continue
                gx, gy, gt = entry[0], entry[1], entry[2]
                age = now - gt
                if age > _GHOST_MAX_AGE or age < 0.5:
                    continue
                if (int(gx)//20, int(gy)//20) in vis_buckets:
                    continue
                a = int(max(0, 1.0 - age / _GHOST_MAX_AGE) * 100)
                if a < 20:
                    continue
                gsx, gsy = gx * sx, gy * sy
                clr = (*CLR_GHOST_DOT, a)
                draw.ellipse([gsx-7, gsy-7, gsx+7, gsy+7],
                             outline=clr, width=2)
                draw.text((gsx+9, gsy-7), "?", fill=clr, font=self._font_10)

        # --- l7: Range ellipses + super glow ---
        # Ranges are pre-scaled by scale_factor in play.py (device->1920×1080).
        # Use uniform scale to keep circles proportional (not squished).
        rs = min(sx, sy) if sx and sy else 1.0
        if ranges_:
            safe_r, atk_r, sup_r = ranges_
            if cfg["safe_range"] and safe_r > 0:
                self._draw_ellipse(draw, px, py, safe_r*rs, safe_r*rs,
                                   CLR_SAFE_RANGE, 2)
            if cfg["attack_range"] and atk_r > 0:
                self._draw_ellipse(draw, px, py, atk_r*rs, atk_r*rs,
                                   CLR_ATTACK_RANGE, 3, "ATK")
            if cfg["super_range"] and sup_r > 0 and is_super_ready:
                self._draw_ellipse(draw, px, py, sup_r*rs, sup_r*rs,
                                   CLR_SUPER_RANGE, 3, "SUPER")

        if is_super_ready and cfg.get("super_range"):
            pulse = int((math.sin(now * 4.0) + 1.0) * 0.5 * 120 + 50)
            draw.ellipse([px-16, py-16, px+16, py+16],
                         outline=(*CLR_SUPER_GLOW, pulse), width=3)

        # --- l8: LoS lines + enemy dots (smoothed) ---
        _E_LERP = 0.80
        smooth_ec = {}
        for i, en in enumerate(enemies[:_MAX_LOS]):
            raw_ex = ((en[0]+en[2])/2) * sx
            raw_ey = ((en[1]+en[3])/2) * sy
            prev = self._smooth_enemies.get(i)
            if prev:
                se_x = prev[0] + (raw_ex - prev[0]) * _E_LERP
                se_y = prev[1] + (raw_ey - prev[1]) * _E_LERP
            else:
                se_x, se_y = raw_ex, raw_ey
            self._smooth_enemies[i] = (se_x, se_y)
            smooth_ec[i] = (se_x, se_y)
        self._smooth_enemies = {k: v for k, v in self._smooth_enemies.items()
                                if k < len(enemies)}

        if cfg["los_all_enemies"] and enemies:
            for i, en in enumerate(enemies[:_MAX_LOS]):
                ex, ey = smooth_ec.get(i, (((en[0]+en[2])/2)*sx,
                                            ((en[1]+en[3])/2)*sy))
                is_tgt = (target_bbox is not None and
                          abs(en[0]-target_bbox[0]) < 5 and
                          abs(en[1]-target_bbox[1]) < 5)
                hittable = hittable_map.get(i, True)
                if is_tgt:
                    # Target LoS: solid, bright, thick
                    color = CLR_LOS_CLEAR if hittable else CLR_LOS_BLOCKED
                    draw.line([(px, py), (ex, ey)], fill=color, width=3)
                    r = 7
                    draw.ellipse([ex-r, ey-r, ex+r, ey+r], fill=CLR_TARGET_DOT)
                else:
                    # Non-target LoS: dashed, subtle
                    color = (0, 255, 0, 70) if hittable else (255, 68, 68, 70)
                    self._draw_dashed_line(draw, px, py, ex, ey, color, dash=12, gap=8)
                    r = 5
                    draw.ellipse([ex-r, ey-r, ex+r, ey+r], fill=CLR_ENEMY_DOT)

        # --- l9: Enemy boxes + target annotation + velocity arrow + per-enemy HP ---
        if cfg["enemies"] and enemies:
            for i, en in enumerate(enemies):
                x1, y1 = en[0]*sx, en[1]*sy
                x2, y2 = en[2]*sx, en[3]*sy
                is_tgt = (target_bbox is not None and
                          abs(en[0]-target_bbox[0]) < 5 and
                          abs(en[1]-target_bbox[1]) < 5)

                # Look up per-enemy HP from the map (bbox center -> key)
                _en_cx = (en[0] + en[2]) / 2
                _en_cy = (en[1] + en[3]) / 2
                _en_hp = -1
                if per_enemy_hp:
                    _best_d = 999999
                    for _k, (_h, _c, _t) in per_enemy_hp.items():
                        try:
                            _kx, _ky = [float(_v) for _v in _k.split(",")]
                        except (ValueError, AttributeError):
                            continue
                        _d = abs(_kx - _en_cx) + abs(_ky - _en_cy)
                        if _d < _best_d:
                            _best_d = _d
                            _en_hp = _h
                    if _best_d > 80:
                        _en_hp = -1  # No close match

                if is_tgt:
                    # Pulsating glow behind target box
                    pulse_a = int((math.sin(now * 5.0) + 1.0) * 0.5 * 80 + 40)
                    glow_clr = (*CLR_TARGET_OUTLINE[:3], pulse_a)
                    draw.rounded_rectangle([x1-5, y1-5, x2+5, y2+5],
                                           radius=6, outline=glow_clr, width=3)
                    draw.rounded_rectangle([x1, y1, x2, y2],
                                           radius=4, outline=CLR_TARGET_OUTLINE, width=3)
                    if cfg.get("target_info"):
                        self._draw_target_annotation(draw, x1, y1, x2, y2,
                                                     target_info)
                    # HP bar: prefer per-enemy HP, fall back to global enemy_hp
                    _tgt_hp = _en_hp if _en_hp >= 0 else enemy_hp
                    if cfg.get("hp_bars") and _tgt_hp >= 0:
                        bar_w = max(40, x2 - x1)
                        self._draw_hp_bar(draw, x1, y1-10, bar_w, 5, _tgt_hp)

                    # Enemy velocity arrow on target
                    if enemy_vel and (abs(enemy_vel[0]) > 2 or abs(enemy_vel[1]) > 2):
                        ecx, ecy = (x1+x2)/2, (y1+y2)/2
                        vscale = 0.18
                        vex = ecx + enemy_vel[0] * vscale
                        vey = ecy + enemy_vel[1] * vscale
                        draw.line([(ecx, ecy), (vex, vey)],
                                  fill=CLR_ENEMY_VEL, width=3)
                        self._draw_arrowhead(draw, ecx, ecy, vex, vey,
                                             CLR_ENEMY_VEL, size=8)

                    # Enemy reload window
                    if enemy_reload_win:
                        draw.text((x2+4, y1), "\u26a1",
                                  fill=CLR_RELOAD_WINDOW, font=self._font_11)
                else:
                    # Non-target: corner brackets instead of full rectangle
                    self._draw_corner_brackets(draw, x1, y1, x2, y2,
                                               CLR_ENEMY_BOX, width=2)
                    # HP bar for non-target enemies too
                    if cfg.get("hp_bars") and _en_hp >= 0:
                        bar_w = max(35, x2 - x1)
                        self._draw_hp_bar(draw, x1, y1-10, bar_w, 4, _en_hp)

        # --- l10: Teammate boxes + HP bars ---
        if cfg["teammates"] and teammates:
            for mate in teammates:
                mx1, my1 = mate[0]*sx, mate[1]*sy
                mx2, my2 = mate[2]*sx, mate[3]*sy
                draw.rectangle([mx1, my1, mx2, my2],
                               outline=CLR_TEAMMATE_BOX, width=2)

            if cfg.get("hp_bars") and teammate_hp_data:
                for thd in teammate_hp_data:
                    if len(thd) < 3:
                        continue
                    tcx, tcy, thp = thd[0]*sx, thd[1]*sy, thd[2]
                    bw = 38
                    self._draw_hp_bar(draw, tcx-bw/2, tcy-24, bw, 5, thp,
                                      bg_clr=CLR_MATE_HP_BG,
                                      fill_clr=CLR_MATE_HP_FILL)

        # (walls already drawn at l3b)

        # --- l12: Movement arrow (with trail effect) ---
        if cfg["movement_arrow"] and movement:
            norm = "".join(sorted(set(movement.upper())))
            vec = _DIR_VECTORS.get(norm)
            if vec:
                arrow_len = 65
                ax = px + vec[0] * arrow_len
                ay = py + vec[1] * arrow_len
                # Trail segments (fading from origin)
                for t in range(3):
                    frac = (t + 1) / 4.0
                    trail_a = int(50 + frac * 120)
                    tx = px + vec[0] * arrow_len * frac
                    ty = py + vec[1] * arrow_len * frac
                    tw = max(2, int(4 - t))
                    draw.line([(px + vec[0]*arrow_len*(t/4.0),
                                py + vec[1]*arrow_len*(t/4.0)),
                               (tx, ty)],
                              fill=(*CLR_MOVE_ARROW[:3], trail_a), width=tw)
                # Main line
                draw.line([(px, py), (ax, ay)],
                          fill=CLR_MOVE_ARROW, width=3)
                self._draw_arrowhead(draw, px, py, ax, ay, CLR_MOVE_ARROW, 13)

        # --- l13: Player dot + HP + ammo + shield/regen ---
        if cfg["player_dot"]:
            # Respawn shield ring
            if respawn_shield:
                shield_left = max(0, respawn_shield_until - now)
                if shield_left > 0:
                    sa = int(min(200, shield_left * 80))
                    draw.ellipse([px-18, py-18, px+18, py+18],
                                 outline=(*CLR_SHIELD[:3], sa), width=3)

            # Regen pulsating ring
            if is_regen:
                rpulse = int((math.sin(now * 5.0) + 1.0) * 0.5 * 100 + 60)
                draw.ellipse([px-17, py-17, px+17, py+17],
                             outline=(*CLR_REGEN[:3], rpulse), width=2)

            dot_r = 10
            # Outer glow
            draw.ellipse([px-dot_r-4, py-dot_r-4, px+dot_r+4, py+dot_r+4],
                         fill=(0, 255, 0, 50))
            draw.ellipse([px-dot_r-2, py-dot_r-2, px+dot_r+2, py+dot_r+2],
                         fill=CLR_PLAYER_RING)
            draw.ellipse([px-dot_r, py-dot_r, px+dot_r, py+dot_r],
                         fill=CLR_PLAYER_DOT)

        # Player HP bar (above player)
        if cfg.get("hp_bars") and player_hp >= 0:
            bar_w = 52
            self._draw_hp_bar(draw, px - bar_w//2, py - 22, bar_w, 6, player_hp)

        # Ammo pips (horizontal row below player)
        if cfg.get("hp_bars") and max_ammo > 0:
            pip_r = 4
            pip_sp = 12
            total_w = max_ammo * pip_sp
            pip_x0 = px - total_w / 2 + pip_sp / 2
            pip_y = py + 18
            for i in range(max_ammo):
                clr = CLR_AMMO_FULL if i < ammo else CLR_AMMO_EMPTY
                xx = pip_x0 + i * pip_sp
                draw.ellipse([xx-pip_r, pip_y-pip_r,
                              xx+pip_r, pip_y+pip_r], fill=clr)

        # --- l14: HUD panel ---
        brawler_name = d.get("brawler_name", "")
        brawler_info = d.get("brawler_info", {})
        if cfg.get("brawler_hud") and brawler_name:
            hud_rect = self._draw_hud(
                draw, cw, ch, cfg, brawler_name, brawler_info, ranges_,
                match_phase, our_score, their_score, kills, death_count,
                ammo, max_ammo, game_mode_name, match_start, now,
                is_gadget_ready, is_super_ready, is_hypercharge_ready,
                hold_super, aggression, bot_ips, no_enemy_dur, patrol_phase,
                stuck_level, self._fps,
                player_hp=player_hp, enemy_hp=enemy_hp)

            if behavior:
                self._draw_badges(draw, hud_rect[0], hud_rect[1] + 4, behavior)

        # --- l15: Decision banner (modern centered pill) ---
        if cfg.get("decision_banner") and decision_reason:
            if decision_reason != self._last_decision:
                self._last_decision = decision_reason
                self._decision_update_time = now
            age = now - self._decision_update_time
            if age < 4.0:
                a = int(max(0, 1.0 - age / 4.0) * 230)
                if a > 20:
                    self._draw_banner(draw, cw, ch, decision_reason, a)

        # FPS/IPS is now integrated into HUD panel, no separate counter needed

        # --- l17: AI - Projectile Arrows + Dodge Direction ---
        if cfg.get("projectiles"):
            projectiles = d.get("projectiles", [])
            for proj in projectiles:
                # Each projectile: {"pos": (x,y), "velocity": (vx,vy), "threat": float}
                ppx = proj.get("pos", (0, 0))
                pvel = proj.get("velocity", (0, 0))
                threat = proj.get("threat", 0)
                prx = ppx[0] * sx
                pry = ppx[1] * sy
                # Draw projectile dot
                pr = max(3, int(4 + threat * 4))
                pa = int(min(255, 120 + threat * 135))
                clr = (255, 80, 0, pa) if threat > 0.5 else (255, 180, 0, pa)
                draw.ellipse([prx - pr, pry - pr, prx + pr, pry + pr], fill=clr)
                # Velocity arrow
                if abs(pvel[0]) + abs(pvel[1]) > 0:
                    avx = prx + pvel[0] * sx * 0.3
                    avy = pry + pvel[1] * sy * 0.3
                    draw.line([(prx, pry), (avx, avy)], fill=clr, width=1)

            # Dodge direction arrow (if dodging is recommended)
            dodge_dir = d.get("dodge_direction")
            if dodge_dir and (abs(dodge_dir[0]) + abs(dodge_dir[1]) > 0):
                ddx = px + dodge_dir[0] * 60
                ddy = py + dodge_dir[1] * 60
                draw.line([(px, py), (ddx, ddy)], fill=(0, 255, 255, 210), width=3)
                self._draw_arrowhead(draw, px, py, ddx, ddy, (0, 255, 255, 210), 12)

        # --- l18: AI - BT Active Path (colour-coded pill) ---
        if cfg.get("bt_path"):
            bt_path = d.get("bt_active_path", [])
            if bt_path:
                truncated = bt_path[-4:]
                path_str = " -> ".join(truncated)
                label = f"BT: {path_str}"
                clr = _bt_color_for(truncated[-1]) if truncated else (180, 220, 255, 200)
                try:
                    tw = int(draw.textlength(label, font=self._font_11)) + 24
                except Exception:
                    tw = len(label) * 7 + 24
                th = 26
                bt_x = (cw - tw) // 2
                bt_y = ch - 74
                draw.rounded_rectangle([bt_x, bt_y, bt_x+tw, bt_y+th],
                                       radius=th//2,
                                       fill=(10, 12, 18, 170),
                                       outline=(*clr[:3], 100), width=1)
                self._text_shadow(draw, (bt_x + 12, bt_y + 5), label,
                                  clr, self._font_11)

        # --- l19: AI - Combo State (styled pill) ---
        if cfg.get("combo_queue"):
            combo_state = d.get("combo_state", "")
            if combo_state:
                combo_lbl = f">> {combo_state}"
                try:
                    ctw = int(draw.textlength(combo_lbl, font=self._font_11)) + 20
                except Exception:
                    ctw = len(combo_lbl) * 7 + 20
                cth = 24
                ccx = (cw - ctw) // 2
                ccy = ch - 46
                draw.rounded_rectangle([ccx, ccy, ccx+ctw, ccy+cth],
                                       radius=cth//2,
                                       fill=(40, 30, 10, 170),
                                       outline=(255, 200, 80, 100), width=1)
                self._text_shadow(draw, (ccx + 10, ccy + 4), combo_lbl,
                                  (255, 220, 100, 230), self._font_11)

        # --- l20: AI - Enemy Track IDs + Reload Indicator ---
        if cfg.get("enemies"):
            enemy_tracks = d.get("enemy_tracks", [])
            for trk in enemy_tracks:
                # trk: {"id": int, "pos": (x,y), "reloading": bool, "style": str}
                tpos = trk.get("pos", (0, 0))
                etx = tpos[0] * sx
                ety = tpos[1] * sy
                tid = trk.get("id", "?")
                draw.text((etx + 14, ety - 24), f"#{tid}",
                          fill=(255, 200, 200, 190), font=self._font_9)
                if trk.get("reloading"):
                    draw.text((etx + 14, ety - 12), "⟳ reload",
                              fill=(100, 255, 100, 200), font=self._font_9)
                style = trk.get("style", "")
                if style and style != "unknown":
                    draw.text((etx + 14, ety), style[:6],
                              fill=(200, 200, 255, 170), font=self._font_9)

        # --- l21: AI - Aim Stats (manual vs auto ratio) ---
        if cfg.get("aim_stats"):
            aim = d.get("aim_stats", {})
            if aim:
                manual_pct = aim.get("manual_pct", 0)
                total_shots = aim.get("total_shots", 0)
                if total_shots > 0:
                    aim_txt = f"Aim: {manual_pct:.0f}% manual ({total_shots} shots)"
                    try:
                        atw = int(draw.textlength(aim_txt, font=self._font_9)) + 16
                    except Exception:
                        atw = len(aim_txt) * 6 + 16
                    ax0 = cw - atw - 8
                    ay0 = ch - 38
                    draw.rounded_rectangle([ax0, ay0, ax0+atw, ay0+20],
                                           radius=10, fill=(10, 10, 10, 140))
                    self._text_shadow(draw, (ax0+8, ay0+3), aim_txt,
                                      (200, 200, 200, 200), self._font_9)

        # --- l22: AI - Opponent Model Summary ---
        opp = d.get("opponent_summary", {})
        if opp and cfg.get("brawler_hud"):
            team_style = opp.get("team_style", "unknown")
            team_agg = opp.get("team_aggression", 0)
            if team_style != "unknown":
                opp_txt = f"Enemy: {team_style} ({team_agg:.0f}% agg)"
                try:
                    otw = int(draw.textlength(opp_txt, font=self._font_10)) + 16
                except Exception:
                    otw = len(opp_txt) * 7 + 16
                draw.rounded_rectangle([8, ch-38, 8+otw, ch-16],
                                       radius=10, fill=(10, 10, 10, 150))
                self._text_shadow(draw, (16, ch-36), opp_txt,
                                  (255, 180, 180, 210), self._font_10)

        # --- l23: AI - Spatial Grid (minimap, fast numpy-rendered) ---
        if cfg.get("spatial_grid"):
            grid_data = d.get("spatial_grid")
            if grid_data is not None:
                try:
                    rows, cols = grid_data.shape
                    cell_w = max(2, min(5, cw // cols))
                    cell_h = max(2, min(5, ch // rows))
                    ox = cw - cols * cell_w - 8
                    oy = ch - rows * cell_h - 40

                    # Build RGBA minimap image via numpy (much faster
                    # than 1000+ individual PIL draw.rectangle calls).
                    _GRID_LUT = np.array([
                        [40, 40, 40, 60],     # 0 UNKNOWN
                        [0, 0, 0, 0],         # 1 EMPTY
                        [120, 80, 40, 100],   # 2 WALL
                        [30, 120, 30, 80],    # 3 BUSH
                        [100, 100, 100, 80],  # 4 DESTROYED
                        [40, 80, 160, 80],    # 5 WATER
                        [160, 40, 160, 100],  # 6 GAS
                    ], dtype=np.uint8)

                    safe = np.clip(grid_data, 0, 6).astype(np.intp)
                    mini_rgba = _GRID_LUT[safe]  # (rows, cols, 4)

                    # Scale up by cell size using repeat (nearest-neighbor)
                    mini_rgba = np.repeat(mini_rgba, cell_h, axis=0)
                    mini_rgba = np.repeat(mini_rgba, cell_w, axis=1)

                    mini_img = Image.fromarray(mini_rgba, "RGBA")
                    img.paste(mini_img, (ox, oy), mini_img)

                    # Draw A* pathfinder path on minimap (cyan line)
                    pf_path = d.get("pathfinder_path", [])
                    if pf_path and len(pf_path) >= 2:
                        try:
                            for i in range(len(pf_path) - 1):
                                px1, py1 = pf_path[i]
                                px2, py2 = pf_path[i + 1]
                                mx1 = ox + int(px1 / 40) * cell_w + cell_w // 2
                                my1 = oy + int(py1 / 40) * cell_h + cell_h // 2
                                mx2 = ox + int(px2 / 40) * cell_w + cell_w // 2
                                my2 = oy + int(py2 / 40) * cell_h + cell_h // 2
                                draw.line([(mx1, my1), (mx2, my2)],
                                         fill=(0, 255, 255, 200), width=2)
                            gpx, gpy = pf_path[-1]
                            gmx = ox + int(gpx / 40) * cell_w
                            gmy = oy + int(gpy / 40) * cell_h
                            draw.ellipse([gmx - 2, gmy - 2, gmx + cell_w + 2, gmy + cell_h + 2],
                                        outline=(255, 255, 0, 220), width=1)
                        except Exception:
                            pass
                except Exception:
                    pass

        # --- premultiply + blit ---
        out = self._premultiply_alpha(img)
        _update_layered(self._overlay_hwnd, out, cx, cy, cfg["opacity"])
        self._prev_client = (cx, cy, cw, ch)

    # --- dRAWING HELPERS ---

    def _hide(self):
        if self._prev_client != (0, 0, 0, 0):
            try:
                blank = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
                _update_layered(self._overlay_hwnd, blank, -10, -10, 0)
            except Exception:
                pass
            self._prev_client = (0, 0, 0, 0)
        self._smooth_init = False

    def _draw_ellipse(self, draw, cx_e, cy_e, rx, ry, color,
                      width=2, label=None):
        ry_iso = ry * ISO_Y_FACTOR
        draw.ellipse([cx_e-rx, cy_e-ry_iso, cx_e+rx, cy_e+ry_iso],
                     outline=color, width=width)
        if label:
            draw.text((cx_e-16, cy_e-ry_iso-18), label,
                      fill=color, font=self._font_11)

    def _draw_hp_bar(self, draw, x, y, w, h, pct,
                     bg_clr=None, fill_clr=None):
        pct = max(0, min(100, pct))
        # Rounded background
        draw.rounded_rectangle([x, y, x+w, y+h], radius=h//2+1,
                               fill=bg_clr or CLR_HP_BG)
        if pct > 0:
            fw = max(h, w * pct / 100.0)  # min width = height for rounded cap
            draw.rounded_rectangle([x, y, x+fw, y+h], radius=h//2+1,
                                   fill=fill_clr or _hp_color(pct))

    def _draw_target_annotation(self, draw, x1, y1, x2, y2, info):
        if not info:
            return
        parts = []
        name = info.get("name")
        if name:
            parts.append(str(name))
        dist = info.get("distance", 0)
        if dist and int(dist) > 0:
            parts.append(f"{int(dist)}px")
        hp = info.get("hp", -1)
        if hp is not None and int(hp) >= 0:
            parts.append(f"{int(hp)}%")
        conf = info.get("confidence", 0)
        if conf and float(conf) > 0:
            parts.append(f"{float(conf)*100:.0f}%")
        if not parts:
            return

        text = " \u2022 ".join(parts)
        ty = y1 - 32
        try:
            tw = int(draw.textlength(text, font=self._font_11)) + 24
        except Exception:
            tw = len(text) * 8 + 24
        tx0 = int((x1 + x2) / 2 - tw / 2)
        # Outer glow
        draw.rounded_rectangle([tx0-2, ty-5, tx0+tw+2, ty+20],
                               radius=11, fill=(255, 0, 255, 25))
        draw.rounded_rectangle([tx0, ty-3, tx0+tw, ty+18],
                               radius=10, fill=(10, 10, 10, 200),
                               outline=(255, 140, 200, 100), width=1)
        self._text_shadow(draw, (tx0+12, ty), text,
                          (255, 220, 220, 250), self._font_11)

    def _draw_hud(self, draw, cw, ch, cfg, name, info, ranges_,
                  phase, our, their, kills, deaths, ammo, max_ammo,
                  mode_name, match_start, now,
                  gadget, super_rdy, hyper, hold_sup,
                  aggr, ips, no_enemy, patrol, stuck, fps,
                  player_hp=100, enemy_hp=-1):
        """Draw modern compact HUD panel. Returns (badge_x, badge_y)."""
        pad = 10
        lh = 20
        disp = name.upper()

        lines = []  # (label, value, color_override_or_None)

        # Mode + timer header
        header_parts = []
        if mode_name:
            header_parts.append(mode_name.upper())
        if match_start and match_start > 0:
            elapsed = now - match_start
            if elapsed > 0:
                header_parts.append(_fmt_time(elapsed))
        if header_parts:
            lines.append(("", " \u2502 ".join(header_parts), (180, 200, 220, 200)))

        # Game state first (most important)
        if phase:
            pclr = {
                "LATE": (255, 100, 100, 230),
                "MID": (255, 220, 80, 230),
            }.get(phase.upper(), (100, 255, 100, 230))
            lines.append(("PHASE", phase.upper(), pclr))
        # HP display (prominent, color-coded)
        hp_clr = (100, 255, 100, 230) if player_hp > 60 else (255, 220, 80, 230) if player_hp > 30 else (255, 80, 80, 230)
        hp_txt = f"{player_hp}%" if player_hp >= 0 else "?"
        if enemy_hp >= 0:
            ehp_clr = (255, 100, 100, 230) if enemy_hp < 40 else (255, 200, 100, 230) if enemy_hp < 70 else (200, 200, 200, 200)
            hp_txt += f"  \u2694 {enemy_hp}%"
        lines.append(("HP", hp_txt, hp_clr))
        lines.append(("SCORE", f"{our} \u2013 {their}", None))
        lines.append(("K/D", f"{kills} / {deaths}", None))

        # Reward score (from adaptive learning)
        reward_score = getattr(self, '_last_reward_score', 0.0)
        aggr_mod = getattr(self, '_last_aggr_mod', 1.0)
        if reward_score != 0.0:
            r_clr = (100, 255, 100, 230) if reward_score > 0 else (255, 80, 80, 230)
            lines.append(("REWARD", f"{reward_score:+.1f}", r_clr))
        # Aggression modifier from rewards
        if aggr_mod != 1.0:
            aggr_pct = int(aggr_mod * 100)
            a_clr = (255, 180, 50, 230) if aggr_mod > 1.0 else (100, 150, 255, 230)
            lines.append(("ADAPT", f"{aggr_pct}%", a_clr))

        # Abilities row (compact)
        ab_parts = []
        if gadget:
            ab_parts.append("\u2699")
        if super_rdy:
            ab_parts.append("\u2b50H" if hold_sup else "\u2b50")
        if hyper:
            ab_parts.append("\u26a1")
        if ab_parts:
            lines.append(("ABILITIES", "  ".join(ab_parts), (255, 220, 50, 240)))

        # Compact brawler stats
        stat_parts = []
        if info.get("health"):
            stat_parts.append(f"\u2764{info['health']}")
        if info.get("attack_damage"):
            stat_parts.append(f"\u2694{info['attack_damage']}")
        if stat_parts:
            lines.append(("", "  ".join(stat_parts), (200, 200, 200, 180)))

        # Aggression + Stuck (only when notable)
        if aggr < 0.85 or aggr > 1.15:
            aggr_pct = int(aggr * 100)
            aclr = CLR_AGGR_HIGH if aggr > 1.0 else CLR_AGGR_LOW
            lines.append(("AGGR", f"{aggr_pct}%", aclr))
        if stuck > 0:
            lines.append(("STUCK", f"LVL {stuck}", (255, 80, 80, 200)))

        # Search state
        if no_enemy > 2.0:
            lines.append(("PATROL", f"{no_enemy:.0f}s {patrol.upper()[:5]}",
                          (200, 160, 80, 200)))

        # FPS/IPS integrated
        perf_parts = []
        if fps and fps > 0:
            perf_parts.append(f"{fps:.0f}fps")
        if ips and ips > 0:
            perf_parts.append(f"{ips:.0f}ips")
        if perf_parts:
            lines.append(("", " ".join(perf_parts), CLR_FPS_TEXT))

        # compute layout
        max_lw = 0
        max_vw = 0
        for lb, val, _ in lines:
            if lb:
                try:
                    tw = draw.textlength(f"{lb}:", font=self._font_9)
                except Exception:
                    tw = len(f"{lb}:") * 5
                max_lw = max(max_lw, tw)
            if val:
                try:
                    tw = draw.textlength(val, font=self._font_10)
                except Exception:
                    tw = len(val) * 5
                max_vw = max(max_vw, tw)

        try:
            nw = draw.textlength(disp, font=self._font_15)
        except Exception:
            nw = len(disp) * 10
        bw = max(int(max_lw + max_vw + pad*3 + 12), int(nw + pad*2), 160)

        n_rows = sum(1 for l, v, _ in lines if l or v)
        bh = pad*2 + lh + 6 + n_rows*lh + 4

        # Position: configurable (top-left, top-right, bottom-left, bottom-right)
        # When at top, offset Y by ~70px to avoid the game's own team UI icons
        _GAME_UI_TOP_OFFSET = 70
        hud_pos = cfg.get("hud_position", "top-left")
        if "right" in hud_pos:
            x0 = cw - bw - 10
        else:
            x0 = 10
        if "bottom" in hud_pos:
            y0 = ch - bh - 10
        else:
            y0 = _GAME_UI_TOP_OFFSET

        # Background: frosted dark panel with soft glow edge
        draw.rounded_rectangle([x0-1, y0-1, x0+bw+1, y0+bh+1],
                               radius=11,
                               fill=(40, 50, 70, 40))  # subtle outer glow
        draw.rounded_rectangle([x0, y0, x0+bw, y0+bh],
                               radius=10,
                               fill=(10, 12, 18, 195),
                               outline=(80, 90, 110, 80))
        # Gradient-like header band (2 lines: bright + fade)
        draw.line([(x0+10, y0+2), (x0+bw-10, y0+2)],
                  fill=(255, 220, 50, 120), width=2)
        draw.line([(x0+14, y0+4), (x0+bw-14, y0+4)],
                  fill=(255, 220, 50, 40), width=1)

        # Brawler name header (larger font + shadow)
        self._text_shadow(draw, (x0+pad, y0+pad), disp,
                          CLR_HUD_NAME, self._font_15)
        # Thin separator under name
        sep_y = y0 + pad + lh + 2
        draw.line([(x0+pad, sep_y), (x0+bw-pad, sep_y)],
                  fill=(80, 90, 110, 60), width=1)

        vx = x0 + int(max_lw) + pad + 14
        ty = y0 + pad + lh + 6
        _sep_after = {"K/D", "ADAPT", "REWARD"}  # draw separator after these
        for lb, val, cov in lines:
            if not lb and not val:
                continue
            if lb:
                self._text_shadow(draw, (x0+pad, ty), f"{lb}:",
                                  CLR_HUD_LABEL, self._font_10)
            self._text_shadow(draw, (vx, ty), val,
                              cov or CLR_HUD_VALUE, self._font_11)
            ty += lh
            if lb in _sep_after:
                draw.line([(x0+pad, ty-4), (x0+bw-pad, ty-4)],
                          fill=(60, 70, 90, 50), width=1)

        return (x0, y0 + bh)

    def _draw_badges(self, draw, x0, y0, flags):
        badges = []
        if flags.get("disengage"):
            badges.append(("\u2190 DISENGAGE", CLR_BADGE_DISENGAGE))
        if flags.get("number_advantage"):
            badges.append(("\u2714 ADVANTAGE", CLR_BADGE_ADVANTAGE))
        if flags.get("stutter_step"):
            badges.append(("\u21c4 STUTTER", CLR_BADGE_STUTTER))
        if flags.get("peek"):
            badges.append(("\u25ce PEEK", CLR_BADGE_PEEK))
        if not badges:
            return
        bx = x0
        for label, color in badges:
            try:
                tw = int(draw.textlength(label, font=self._font_9)) + 16
            except Exception:
                tw = len(label) * 6 + 16
            # Outer glow
            draw.rounded_rectangle([bx-1, y0-1, bx+tw+1, y0+22],
                                   radius=12, fill=(*color[:3], 40))
            draw.rounded_rectangle([bx, y0, bx+tw, y0+21],
                                   radius=11, fill=(*color[:3], int(color[3]*0.85)),
                                   outline=(*color[:3], 60))
            self._text_shadow(draw, (bx+8, y0+3), label,
                              (255, 255, 255, 250), self._font_9,
                              shadow=(0, 0, 0, 120))
            bx += tw + 6

    def _draw_banner(self, draw, cw, ch, reason, alpha):
        """Modern decision banner - color-coded pill above bottom controls."""
        text = reason[:50].upper()
        clr = _bt_color_for(text)
        try:
            tw = int(draw.textlength(text, font=self._font_15)) + 36
        except Exception:
            tw = len(text) * 9 + 36
        th = 36
        x0 = (cw - tw) // 2
        y0 = int(ch * 0.78)
        # Compute alpha-scaled versions
        f = min(1.0, alpha / 230.0)
        bg = (10, 12, 18, int(210 * f))
        # Accent border matching BT state colour
        border = (*clr[:3], int(min(160, alpha * 0.65)))
        txt = (255, 255, 255, min(250, alpha))
        # Outer glow (subtle)
        glow_a = int(40 * f)
        if glow_a > 8:
            draw.rounded_rectangle([x0-3, y0-3, x0+tw+3, y0+th+3],
                                   radius=th//2+3,
                                   fill=(*clr[:3], glow_a))
        draw.rounded_rectangle([x0, y0, x0+tw, y0+th],
                               radius=th//2, fill=bg, outline=border, width=2)
        # Colour pip indicator
        pip_r = 5
        pip_x = x0 + 14
        pip_y = y0 + th // 2
        draw.ellipse([pip_x-pip_r, pip_y-pip_r, pip_x+pip_r, pip_y+pip_r],
                     fill=(*clr[:3], min(250, alpha)))
        self._text_shadow(draw, (x0 + 26, y0 + 8), text, txt, self._font_15,
                          shadow=(0, 0, 0, int(180 * f)))

    # text-shadow helper (1px offset, big readability boost)
    @staticmethod
    def _text_shadow(draw, xy, text, fill, font, shadow=CLR_SHADOW, offset=1):
        """Draw text with a dark drop-shadow for legibility over bright backgrounds."""
        x, y = xy
        draw.text((x + offset, y + offset), text, fill=shadow, font=font)
        draw.text((x, y), text, fill=fill, font=font)

    @staticmethod
    def _draw_arrowhead(draw, x1, y1, x2, y2, color, size=8):
        angle = math.atan2(y2 - y1, x2 - x1)
        a1 = angle + math.radians(150)
        a2 = angle - math.radians(150)
        p1 = (x2 + size * math.cos(a1), y2 + size * math.sin(a1))
        p2 = (x2 + size * math.cos(a2), y2 + size * math.sin(a2))
        draw.polygon([(x2, y2), p1, p2], fill=color)

    @staticmethod
    def _draw_corner_brackets(draw, x1, y1, x2, y2, color, width=1):
        """Draw L-shaped corners instead of full rectangle - less visual clutter."""
        bw = max(6, int((x2 - x1) * 0.25))
        bh = max(6, int((y2 - y1) * 0.25))
        # Top-left
        draw.line([(x1, y1), (x1 + bw, y1)], fill=color, width=width)
        draw.line([(x1, y1), (x1, y1 + bh)], fill=color, width=width)
        # Top-right
        draw.line([(x2, y1), (x2 - bw, y1)], fill=color, width=width)
        draw.line([(x2, y1), (x2, y1 + bh)], fill=color, width=width)
        # Bottom-left
        draw.line([(x1, y2), (x1 + bw, y2)], fill=color, width=width)
        draw.line([(x1, y2), (x1, y2 - bh)], fill=color, width=width)
        # Bottom-right
        draw.line([(x2, y2), (x2 - bw, y2)], fill=color, width=width)
        draw.line([(x2, y2), (x2, y2 - bh)], fill=color, width=width)

    @staticmethod
    def _draw_dashed_line(draw, x1, y1, x2, y2, color, dash=12, gap=10):
        """Draw a dashed line between two points (optimised: fewer segments)."""
        dx = x2 - x1
        dy = y2 - y1
        dist = math.hypot(dx, dy)
        if dist < 1:
            return
        inv = 1.0 / dist
        ux, uy = dx * inv, dy * inv
        step = dash + gap
        segments = []
        pos = 0.0
        while pos < dist:
            end = min(pos + dash, dist)
            segments.append((x1 + ux * pos, y1 + uy * pos,
                             x1 + ux * end, y1 + uy * end))
            pos += step
        for sx, sy, ex, ey in segments:
            draw.line([(sx, sy), (ex, ey)], fill=color, width=1)

    @staticmethod
    def _premultiply_alpha(img: Image.Image) -> Image.Image:
        arr = np.array(img)  # writable copy
        alpha = arr[:, :, 3:4].astype(np.uint16)
        arr[:, :, :3] = (arr[:, :, :3].astype(np.uint16) * alpha >> 8).astype(np.uint8)
        return Image.fromarray(arr, "RGBA")
