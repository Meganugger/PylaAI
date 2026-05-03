import atexit
import ctypes
import math
import socket
import threading
import time
import cv2
import numpy as np
import win32gui
import win32con
import win32ui
import pyautogui
from PIL import Image
from typing import List

# New libraries
import scrcpy
from adbutils import adb

from utils import load_toml_as_dict

# --- Configuration ---
brawl_stars_width, brawl_stars_height = 1920, 1080
BRAWL_STARS_PACKAGE = load_toml_as_dict("cfg/general_config.toml").get(
    "brawlstars_package",
    "com.supercell.brawlstars"
)

key_coords_dict = {
    "H": (1400, 990),
    "G": (1640, 990),
    "M": (1725, 800),
    "Q": (1740, 1000),
    "E": (1510, 880),
    "F": (1390, 930),
}

continue_fallback_targets = (
    (1660, 980),
    (1450, 950),
    (1180, 950),
    (960, 950),
    (960, 540),
)

play_again_targets = (
    (1390, 930),
    (1470, 930),
    (1300, 930),
    (1390, 980),
)

directions_xy_deltas_dict = {
    "w": (0, -100),
    "a": (-100, 0),
    "s": (0, 100),
    "d": (100, 0),
}

EMULATOR_PORTS = {
    "BlueStacks": [5555, 5556, 5557, 5565],
    "LDPlayer": [5555, 5557, 5559, 5554],
    "MEmu": [21503, 21513, 21523, 5555],
    "MuMu": [16384, 16416, 16448, 7555, 5558, 5557, 5556, 5555, 5554],
    "Others": [5555, 5558, 7555, 16384, 16416, 16448, 21503, 5635],
}


def _normalize_emulator_name(name):
    normalized = str(name or "").strip().lower()
    if "blue" in normalized:
        return "BlueStacks"
    if "ld" in normalized:
        return "LDPlayer"
    if "memu" in normalized:
        return "MEmu"
    if "mumu" in normalized:
        return "MuMu"
    return "Others"


def _unique_ports(ports):
    unique = []
    for port in ports:
        try:
            port = int(port)
        except (TypeError, ValueError):
            continue
        if port == 5037:
            continue
        if port not in unique:
            unique.append(port)
    return unique


def _is_port_open(host, port, timeout=0.05):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((host, int(port))) == 0
    except OSError:
        return False


def _serial_port(serial):
    if serial.startswith("emulator-"):
        try:
            return int(serial.rsplit("-", 1)[1])
        except ValueError:
            return None
    if ":" in serial:
        try:
            return int(serial.rsplit(":", 1)[1])
        except ValueError:
            return None
    return None


def _serial_candidates(serial, fallback_port=0):
    candidates = []
    serial = str(serial or "").strip()
    port = _serial_port(serial)
    if serial:
        candidates.append(serial)
    if port is None:
        try:
            port = int(fallback_port)
        except (TypeError, ValueError):
            port = None
    if port:
        tcp_serial = f"127.0.0.1:{port}"
        if tcp_serial not in candidates:
            candidates.append(tcp_serial)
    return candidates

class WindowController:
    def __init__(self):
        self.scale_factor = None
        self.width = None
        self.height = None
        self.width_ratio = None
        self.height_ratio = None
        self.joystick_x, self.joystick_y = None, None
        time_config = load_toml_as_dict("cfg/time_tresholds.toml")
        # --- 2. ADB & Scrcpy Connection ---
        print("Connecting to ADB...")
        try:
            def list_online_devices():
                devices = []
                for dev in adb.device_list():
                    try:
                        state = dev.get_state()
                    except Exception:
                        state = "unknown"
                    if state == "device":
                        devices.append(dev)
                    else:
                        print(f"Skipping ADB device {dev.serial} (state: {state})")
                return devices

            def prefer_selected_devices(devices, selected_emulator, configured_port):
                preferred_ports = set(
                    _unique_ports([configured_port] + EMULATOR_PORTS.get(selected_emulator, EMULATOR_PORTS["Others"]))
                )
                preferred_serials = {f"127.0.0.1:{port}" for port in preferred_ports}
                return [
                    dev for dev in devices
                    if _serial_port(dev.serial) in preferred_ports or dev.serial in preferred_serials
                ]

            general_config = load_toml_as_dict("cfg/general_config.toml")
            self.selected_emulator = _normalize_emulator_name(general_config.get("current_emulator", "Others"))
            self.configured_port = general_config.get("emulator_port", 0)
            self.candidate_ports = _unique_ports(
                [self.configured_port]
                + EMULATOR_PORTS.get(self.selected_emulator, EMULATOR_PORTS["Others"])
                + EMULATOR_PORTS["Others"]
                + list(range(5565, 5756, 10))
            )

            device_list = list_online_devices()
            preferred_devices = prefer_selected_devices(device_list, self.selected_emulator, self.configured_port)

            should_probe_ports = not device_list or (self.selected_emulator != "Others" and not preferred_devices)
            if should_probe_ports:
                open_ports = [port for port in self.candidate_ports if _is_port_open("127.0.0.1", port)]
                for port in open_ports:
                    try:
                        adb.connect(f"127.0.0.1:{port}")
                    except Exception:
                        pass
                device_list = list_online_devices()
                preferred_devices = prefer_selected_devices(device_list, self.selected_emulator, self.configured_port)

            if not device_list:
                tried_ports = ", ".join(str(port) for port in self.candidate_ports)
                raise ConnectionError(f"No online ADB devices found. Tried ports: {tried_ports}")

            self.device = preferred_devices[0] if preferred_devices else device_list[0]
            self.device_serial = self.device.serial
            if self.selected_emulator != "Others" and not preferred_devices:
                print(
                    f"Could not identify a {self.selected_emulator} device by port; "
                    f"using first online ADB device instead."
                )
            print(f"Connected to {self.selected_emulator}: {self.device.serial}")

            self.frame_condition = threading.Condition()
            self.scrcpy_client = None
            self.last_frame = None
            self.last_frame_time = 0.0
            self.last_joystick_pos = (None, None)
            self.FRAME_STALE_TIMEOUT = 10.0  # MuMu can have periodic frame delays; 5s was too aggressive
            self._last_frame_hash = None
            self._consecutive_identical_frames = 0
            self._frozen_frame_threshold = int(time_config.get("frozen_frame_min_identical_frames", 30))
            self._frozen_frame_seconds = float(time_config.get("frozen_frame_seconds", 15.0))
            self._frozen_frame_restart_cooldown = float(time_config.get("frozen_frame_restart_cooldown", 20.0))
            self._last_scrcpy_restart_at = 0.0
            self._last_healthy_frame_time = 0.0
            self.APP_STATE_CHECK_INTERVAL = float(time_config.get("check_if_brawl_stars_crashed", 10.0))
            self.APP_RELAUNCH_WAIT = 3.0
            self.last_app_state_check = 0.0
            self.SCRCPY_START_RETRIES = 3
            self.SCRCPY_RETRY_DELAY = 0.5
            self.scrcpy_max_fps = int(general_config.get("scrcpy_max_fps", 30))
            if self.scrcpy_max_fps <= 0:
                self.scrcpy_max_fps = None
            self.scrcpy_max_width = int(general_config.get("scrcpy_max_width", 960))
            if self.scrcpy_max_width < 0:
                self.scrcpy_max_width = 0
            self.scrcpy_bitrate = int(general_config.get("scrcpy_bitrate", 3000000))
            if self.scrcpy_bitrate <= 0:
                self.scrcpy_bitrate = 3000000
            self.capture_fallback_level = 0
            self.start_scrcpy_client()
            atexit.register(self.close)
            print("Scrcpy client started successfully.")

        except Exception as e:
            raise ConnectionError(f"Failed to initialize Scrcpy: {e}")
        self.are_we_moving = False
        self.PID_JOYSTICK = 1  # ID for WASD movement
        self.PID_ATTACK = 2  # ID for clicks/attacks

    def _list_online_devices(self):
        devices = []
        for dev in adb.device_list():
            try:
                state = dev.get_state()
            except Exception:
                state = "unknown"
            if state == "device":
                devices.append(dev)
        return devices

    def _refresh_device_handle(self):
        candidates = _serial_candidates(self.device_serial, self.configured_port)
        online_devices = self._list_online_devices()

        for serial in candidates:
            for dev in online_devices:
                if dev.serial == serial:
                    self.device = dev
                    self.device_serial = dev.serial
                    return True

        target_port = _serial_port(self.device_serial)
        if target_port is None:
            try:
                target_port = int(self.configured_port)
            except (TypeError, ValueError):
                target_port = None

        if target_port is not None:
            for dev in online_devices:
                if _serial_port(dev.serial) == target_port:
                    self.device = dev
                    self.device_serial = dev.serial
                    return True

        return False

    def _recover_adb_transport(self, error):
        print(f"Recovering ADB transport after scrcpy start failure: {error}")
        for serial in _serial_candidates(self.device_serial, self.configured_port):
            try:
                adb.disconnect(serial)
            except Exception:
                pass

        time.sleep(0.15)

        for serial in _serial_candidates(self.device_serial, self.configured_port):
            if serial.startswith("127.0.0.1:"):
                try:
                    adb.connect(serial)
                except Exception:
                    pass

        if not self._refresh_device_handle():
            for port in self.candidate_ports:
                if not _is_port_open("127.0.0.1", port):
                    continue
                try:
                    adb.connect(f"127.0.0.1:{port}")
                except Exception:
                    pass
            self._refresh_device_handle()

    def _reset_frame_state(self):
        with self.frame_condition:
            self.last_frame = None
            self.last_frame_time = 0.0
            self._last_frame_hash = None
            self._consecutive_identical_frames = 0
            self._last_healthy_frame_time = time.time()
            self.frame_condition.notify_all()
        self.last_joystick_pos = (None, None)
        self.are_we_moving = False

    def _build_scrcpy_client(self):
        def on_frame(frame):
            if frame is not None:
                with self.frame_condition:
                    # Frozen-frame detection: check if this frame is identical
                    # to the previous one. MuMu sometimes delivers the same
                    # frozen frame repeatedly while the game is stalled.
                    frame_hash = hash(frame.data.tobytes()[:4096])  # fast partial hash
                    if frame_hash == self._last_frame_hash:
                        self._consecutive_identical_frames += 1
                    else:
                        self._consecutive_identical_frames = 0
                        self._last_frame_hash = frame_hash
                        self._last_healthy_frame_time = time.time()
                    self.last_frame = frame
                    self.last_frame_time = time.time()
                    self.frame_condition.notify_all()

        client_kwargs = {
            "device": self.device,
            "max_width": self.scrcpy_max_width,
            "bitrate": self.scrcpy_bitrate,
        }
        if self.scrcpy_max_fps:
            client_kwargs["max_fps"] = self.scrcpy_max_fps
        client = scrcpy.Client(**client_kwargs)
        client.add_listener(scrcpy.EVENT_FRAME, on_frame)
        return client

    def start_scrcpy_client(self, retries=None):
        retries = max(int(retries or self.SCRCPY_START_RETRIES), 1)
        last_error = None
        self._reset_frame_state()

        for attempt in range(1, retries + 1):
            client = None
            try:
                self._refresh_device_handle()
                client = self._build_scrcpy_client()
                client.start(threaded=True)
                self.scrcpy_client = client
                return
            except Exception as exc:
                last_error = exc
                self.scrcpy_client = None
                if client is not None:
                    try:
                        client.stop()
                    except Exception:
                        pass
                if attempt >= retries:
                    break
                print(f"scrcpy start attempt {attempt}/{retries} failed: {exc}")
                self._recover_adb_transport(exc)
                time.sleep(self.SCRCPY_RETRY_DELAY * attempt)

        raise ConnectionError(f"Failed to start scrcpy after {retries} attempt(s): {last_error}")

    def reduce_capture_load_for_slow_feed(self):
        """Lower scrcpy capture cost after repeated low-IPS recoveries."""
        previous = (self.scrcpy_max_width, self.scrcpy_max_fps, self.scrcpy_bitrate)
        if self.capture_fallback_level == 0:
            self.scrcpy_max_width = min(self.scrcpy_max_width or 960, 854)
            self.scrcpy_max_fps = min(self.scrcpy_max_fps or 60, 30)
            self.scrcpy_bitrate = min(self.scrcpy_bitrate or 3000000, 2000000)
            self.capture_fallback_level = 1
        elif self.capture_fallback_level == 1:
            self.scrcpy_max_width = min(self.scrcpy_max_width or 854, 720)
            self.scrcpy_max_fps = min(self.scrcpy_max_fps or 30, 30)
            self.scrcpy_bitrate = min(self.scrcpy_bitrate or 2000000, 1500000)
            self.capture_fallback_level = 2
        else:
            return False

        current = (self.scrcpy_max_width, self.scrcpy_max_fps, self.scrcpy_bitrate)
        if current == previous:
            return False
        print(
            "Slow emulator feed fallback:",
            f"scrcpy_max_width={self.scrcpy_max_width}",
            f"scrcpy_max_fps={self.scrcpy_max_fps}",
            f"scrcpy_bitrate={self.scrcpy_bitrate}",
        )
        return True

    def restart_scrcpy_client(self):
        print("Restarting scrcpy client...")
        old_client = self.scrcpy_client
        self.scrcpy_client = None
        if old_client is not None:
            try:
                old_client.stop()
            except Exception as exc:
                print(f"Could not stop old scrcpy client cleanly: {exc}")
        time.sleep(0.25)
        self.start_scrcpy_client()
        self._last_scrcpy_restart_at = time.time()
        print("Scrcpy client restarted successfully.")

    def _ensure_frame_geometry(self, frame):
        if self.width and self.height:
            return
        self.width = frame.shape[1]
        self.height = frame.shape[0]
        self.width_ratio = self.width / brawl_stars_width
        self.height_ratio = self.height / brawl_stars_height
        self.joystick_x, self.joystick_y = 220 * self.width_ratio, 870 * self.height_ratio
        self.scale_factor = min(self.width_ratio, self.height_ratio)

    def get_latest_frame(self, copy_frame=True):
        with self.frame_condition:
            if self.last_frame is None:
                return None, 0.0
            frame = self.last_frame.copy() if copy_frame else self.last_frame
            return frame, self.last_frame_time

    def get_current_frame(self, copy_frame=True, timeout=15.0):
        frame, frame_time = self.get_latest_frame(copy_frame=copy_frame)
        if frame is None:
            frame, frame_time = self.wait_for_next_frame(timeout=timeout, copy_frame=copy_frame)
            if frame is None:
                return None, frame_time

        self._ensure_frame_geometry(frame)
        return frame, frame_time

    def ensure_brawl_stars_running(self, force=False):
        now = time.monotonic()
        if not force and now - self.last_app_state_check < self.APP_STATE_CHECK_INTERVAL:
            return False

        self.last_app_state_check = now
        try:
            current_app = self.device.app_current()
            current_package = getattr(current_app, "package", "") if current_app else ""
        except Exception as exc:
            print(f"Could not check the current Android app: {exc}")
            return False

        if current_package == BRAWL_STARS_PACKAGE:
            return False

        print(f"Brawl Stars is not foregrounded (found '{current_package or 'unknown'}'). Relaunching...")
        try:
            self.device.app_start(BRAWL_STARS_PACKAGE)
        except Exception as exc:
            print(f"Failed to relaunch Brawl Stars: {exc}")
            return False

        time.sleep(self.APP_RELAUNCH_WAIT)
        return True

    def wait_for_next_frame(self, last_frame_time=0.0, timeout=None, copy_frame=True):
        if timeout is None:
            timeout = 15.0 if last_frame_time <= 0 else self.FRAME_STALE_TIMEOUT

        deadline = time.monotonic() + timeout
        waiting_for_first_frame = last_frame_time <= 0
        did_log_wait = False
        latest_frame_time = 0.0
        checked_app_state = False
        restarted_scrcpy = False

        while True:
            if waiting_for_first_frame and not did_log_wait:
                print("Waiting for first frame...")
                did_log_wait = True

            with self.frame_condition:
                frame_time = self.last_frame_time
                latest_frame_time = frame_time
                if self.last_frame is not None and frame_time > last_frame_time:
                    frame = self.last_frame.copy() if copy_frame else self.last_frame
                    self._ensure_frame_geometry(frame)
                    return frame, frame_time

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    if not checked_app_state and self.ensure_brawl_stars_running(force=True):
                        checked_app_state = True
                        deadline = time.monotonic() + self.APP_RELAUNCH_WAIT + 2.0
                        continue
                    if not restarted_scrcpy:
                        try:
                            self.restart_scrcpy_client()
                            restarted_scrcpy = True
                            deadline = time.monotonic() + min(timeout, 5.0)
                            continue
                        except Exception as exc:
                            print(f"Could not restart scrcpy client after frame timeout: {exc}")
                    return None, latest_frame_time

                self.frame_condition.wait(timeout=remaining)

    def is_connection_healthy(self):
        """Check if the scrcpy connection is delivering fresh, non-frozen frames."""
        if self.scrcpy_client is None:
            return False
        if self.last_frame is None:
            return False
        frame_age = time.time() - self.last_frame_time if self.last_frame_time > 0 else float("inf")
        if frame_age > self.FRAME_STALE_TIMEOUT:
            return False
        frozen_age = time.time() - self._last_healthy_frame_time if self._last_healthy_frame_time > 0 else 0.0
        if (
            self._consecutive_identical_frames >= self._frozen_frame_threshold
            and frozen_age >= self._frozen_frame_seconds
        ):
            return False
        return True

    def screenshot(self, array=False):
        frame, frame_time = self.get_current_frame(copy_frame=True, timeout=15.0)
        if frame is None:
            raise ConnectionError(
                "No frame received from scrcpy within 15s. "
                "Check USB/emulator connection."
            )

        age = time.time() - frame_time
        if frame_time > 0 and age > self.FRAME_STALE_TIMEOUT:
            print(f"WARNING: scrcpy frame is {age:.1f}s stale -- feed may be frozen")

        # Frozen-frame auto-recovery
        frozen_age = time.time() - self._last_healthy_frame_time if self._last_healthy_frame_time > 0 else 0.0
        restart_cooldown_ok = (time.time() - self._last_scrcpy_restart_at) >= self._frozen_frame_restart_cooldown
        if (
            self._consecutive_identical_frames >= self._frozen_frame_threshold
            and frozen_age >= self._frozen_frame_seconds
            and restart_cooldown_ok
        ):
            print(
                f"WARNING: {self._consecutive_identical_frames} consecutive identical frames "
                f"over {frozen_age:.1f}s detected -- restarting scrcpy to recover"
            )
            try:
                self.restart_scrcpy_client()
                frame, frame_time = self.get_current_frame(copy_frame=True, timeout=10.0)
                if frame is None:
                    raise ConnectionError("No frame after frozen-frame scrcpy restart")
            except Exception as exc:
                print(f"Could not recover from frozen frames: {exc}")

        if array:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def touch_down(self, x, y, pointer_id=0):
        # We explicitly pass the pointer_id
        self.scrcpy_client.control.touch(int(x), int(y), scrcpy.ACTION_DOWN, pointer_id)

    def touch_move(self, x, y, pointer_id=0):
        self.scrcpy_client.control.touch(int(x), int(y), scrcpy.ACTION_MOVE, pointer_id)

    def touch_up(self, x, y, pointer_id=0):
        self.scrcpy_client.control.touch(int(x), int(y), scrcpy.ACTION_UP, pointer_id)

    def move_joystick_angle(self, angle_degrees: float, radius: float = 145.0):
        angle_rad = math.radians(float(angle_degrees) % 360.0)
        scaled_radius = float(radius) * max(float(self.scale_factor or 1.0), 0.5)
        target_x = self.joystick_x + math.cos(angle_rad) * scaled_radius
        target_y = self.joystick_y + math.sin(angle_rad) * scaled_radius

        if not self.are_we_moving:
            self.touch_down(self.joystick_x, self.joystick_y, pointer_id=self.PID_JOYSTICK)
            self.are_we_moving = True

        if self.last_joystick_pos != (target_x, target_y):
            self.touch_move(target_x, target_y, pointer_id=self.PID_JOYSTICK)
            self.last_joystick_pos = (target_x, target_y)

    def stop_joystick(self):
        if self.are_we_moving:
            self.touch_up(self.joystick_x, self.joystick_y, pointer_id=self.PID_JOYSTICK)
            self.are_we_moving = False
            self.last_joystick_pos = (None, None)

    def keys_up(self, keys: List[str]):
        if "".join(keys).lower() == "wasd":
            self.stop_joystick()

    def keys_down(self, keys: List[str]):

        delta_x, delta_y = 0, 0
        for key in keys:
            if key in directions_xy_deltas_dict:
                dx, dy = directions_xy_deltas_dict[key]
                delta_x += dx
                delta_y += dy

        if not self.are_we_moving:
            self.touch_down(self.joystick_x, self.joystick_y, pointer_id=self.PID_JOYSTICK)
            self.are_we_moving = True
            self.last_joystick_pos = (self.joystick_x + delta_x, self.joystick_y + delta_y)

        if self.last_joystick_pos != (self.joystick_x + delta_x, self.joystick_y + delta_y):
            self.touch_move(self.joystick_x + delta_x, self.joystick_y + delta_y, pointer_id=self.PID_JOYSTICK)
            self.last_joystick_pos = (self.joystick_x + delta_x, self.joystick_y + delta_y)

    def click(self, x: int, y: int, delay=0.005, already_include_ratio=True):
        if not already_include_ratio:
            x = x * self.width_ratio
            y = y * self.height_ratio
        # Use PID_ATTACK for clicks so we don't interrupt movement
        self.touch_down(x, y, pointer_id=self.PID_ATTACK)
        time.sleep(delay)
        self.touch_up(x, y, pointer_id=self.PID_ATTACK)

    def press_key(self, key, delay=0.005, touch_up=True, touch_down=True):
        if key not in key_coords_dict:
            return
        x, y = key_coords_dict[key]
        target_x = x * self.width_ratio
        target_y = y * self.height_ratio
        if touch_down and touch_up:
            self.click(target_x, target_y, delay)
        elif touch_down:
            self.touch_down(int(target_x), int(target_y), pointer_id=self.PID_ATTACK)
        elif touch_up:
            self.touch_up(int(target_x), int(target_y), pointer_id=self.PID_ATTACK)

    def press_continue(self, hold_seconds=0.0, include_fallback_clicks=True):
        delay = float(hold_seconds) if hold_seconds and hold_seconds > 0 else 0.005
        self.press_key("Q", delay)
        if not include_fallback_clicks:
            return

        for x, y in continue_fallback_targets:
            self.click(x, y, 0.02, already_include_ratio=False)
            time.sleep(0.5)

    def press_play_again(self):
        for x, y in play_again_targets:
            self.click(x, y, 0.02, already_include_ratio=False)
            time.sleep(0.12)

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.2):
        dist_x = end_x - start_x
        dist_y = end_y - start_y
        distance = math.sqrt(dist_x ** 2 + dist_y ** 2)

        if distance == 0:
            return

        step_len = 25
        steps = max(int(distance / step_len), 1)
        step_delay = duration / steps

        self.touch_down(int(start_x), int(start_y), pointer_id=self.PID_ATTACK)
        for i in range(1, steps + 1):
            t = i / steps
            cx = start_x + dist_x * t
            cy = start_y + dist_y * t
            time.sleep(step_delay)
            self.touch_move(int(cx), int(cy), pointer_id=self.PID_ATTACK)
        self.touch_up(int(end_x), int(end_y), pointer_id=self.PID_ATTACK)

    def close(self):
        if hasattr(self, 'scrcpy_client'):
            client = self.scrcpy_client
            self.scrcpy_client = None
            if client is not None:
                try:
                    client.stop()
                except Exception:
                    pass
