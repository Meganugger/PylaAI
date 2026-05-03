import asyncio
import ctypes
import gc
import threading
import time
import traceback

from gui.hub import Hub
from gui.login import login
from gui.main import App
from gui.select_brawler import SelectBrawler
from lobby_automation import LobbyAutomation
from play import Play
from stage_manager import StageManager
from state_finder.main import get_state, find_game_result
from time_management import TimeManagement
from utils import FrameData, load_toml_as_dict, current_wall_model_is_latest, api_base_url, update_icons, reader
from utils import get_brawler_list, update_missing_brawlers_info, check_version, notify_user, \
    update_wall_model_classes, get_latest_wall_model_file, get_latest_version, cprint
from window_controller import WindowController

pyla_version = load_toml_as_dict("./cfg/general_config.toml")['pyla_version']
BRAWL_STARS_PACKAGE = load_toml_as_dict("cfg/general_config.toml").get("brawlstars_package", "com.supercell.brawlstars")

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"

# gPU provider check
try:
    import onnxruntime as _ort
    _providers = _ort.get_available_providers()
    if "DmlExecutionProvider" in _providers:
        cprint("[GPU] DirectML (AMD/Intel) available", "#AA66FF")
    elif "CUDAExecutionProvider" in _providers:
        cprint("[GPU] CUDA (NVIDIA) available", "#00DC64")
    else:
        cprint("[CPU] No GPU provider detected - running on CPU", "#FFD700")
except Exception:
    pass

# Global reference to the active dashboard (set by dashboard.py when bot starts)
_active_dashboard = None
_active_stage_manager = None


def _print_session_summary(s):
    """Print a formatted session summary to the console."""
    bar = "=" * 56
    cprint(f"\n{bar}", "#FF6B00")
    cprint("         â˜…  SESSION SUMMARY  â˜…", "#FF6B00")
    cprint(bar, "#FF6B00")

    cprint(f"  Duration:     {s['duration']}", "#E0E0E0")
    cprint(f"  Matches:      {s['total_matches']}", "#E0E0E0")
    cprint(f"  Victories:    {s['victories']}  |  Defeats: {s['defeats']}  |  Draws: {s['draws']}", "#E0E0E0")
    wr = s['winrate']
    wr_color = "#00DC64" if wr >= 55 else "#FFD700" if wr >= 45 else "#FF4444"
    cprint(f"  Win Rate:     {wr:.0f}%", wr_color)

    net = s['net_trophies']
    trophy_sign = "+" if net >= 0 else ""
    trophy_color = "#00DC64" if net > 0 else "#FF4444" if net < 0 else "#E0E0E0"
    cprint(f"  Trophies:     {trophy_sign}{net} ", trophy_color)

    tk = s.get('total_kills', 0)
    ta = s.get('total_assists', 0)
    td = s.get('total_damage', 0)
    if tk or ta or td:
        cprint(f"  Kills: {tk}  |  Assists: {ta}  |  Damage: {td:,}", "#00BFFF")

    brawlers = s.get('brawlers', [])
    if brawlers:
        cprint(f"\n  {'Brawler':<14} {'W/D/Dr':<12} {'WR':>5}  {'Trophies':>12}", "#AA66FF")
        cprint(f"  {'-'*14} {'-'*12} {'-'*5}  {'-'*12}", "#555555")
        for b in brawlers:
            name = b['name'].title()
            wldr = f"{b['victories']}/{b['defeats']}/{b['draws']}"
            wr_b = f"{b['winrate']:.0f}%"
            delta = b['trophy_delta']
            dsign = "+" if delta >= 0 else ""
            d_color = "#00DC64" if delta > 0 else "#FF4444" if delta < 0 else "#E0E0E0"
            line = f"  {name:<14} {wldr:<12} {wr_b:>5}  {dsign}{delta:>5} ({b['trophy_start']}->{b['trophy_end']})"
            cprint(line, d_color)

    cprint(bar, "#FF6B00")
    cprint("", "#E0E0E0")


def pyla_main(data, external_stop_event=None, external_pause_event=None):
    class Main:

        def __init__(self):
            self._stop_event = external_stop_event if external_stop_event is not None else threading.Event()
            self._pause_event = external_pause_event if external_pause_event is not None else threading.Event()
            self.window_controller = WindowController()
            self.Play = Play(*self.load_models(), self.window_controller)
            self.Play._runtime_state = "starting"
            self.Time_management = TimeManagement()
            self.lobby_automator = LobbyAutomation(self.window_controller)
            self.Stage_manager = StageManager(data, self.lobby_automator, self.window_controller)
            # Expose for external access (e.g., brawler scanner)
            import sys
            main_module = sys.modules.get('__main__')
            if main_module:
                main_module._active_stage_manager = self.Stage_manager
            self._reward_context_states = {
                "reward_claim",
                "trophy_reward",
                "player_title_reward",
                "prestige_reward",
                "star_drop",
            }
            self.states_requiring_data = [
                "lobby",
                "popup",
                "end",
                *sorted(self._reward_context_states),
            ]
            active_entry = data[0] if data else {}
            startup_brawler = str(active_entry.get('brawler', '') or '').strip().lower()
            self._startup_brawler_target = startup_brawler
            self._pending_initial_brawler_select = bool(
                startup_brawler and active_entry.get('automatically_pick', True)
            )
            self._skip_play_cycle = False
            self.no_detections_action_threshold = 60 * 5  # 5 min (was 2)
            self.initialize_stage_manager()
            # Register first brawler for session summary
            self.Stage_manager.Trophy_observer.start_session_brawler(
                data[0]['brawler'], data[0].get('trophies', 0))
            self._easyocr_warmup_started = False
            if self._pending_initial_brawler_select:
                self.Stage_manager._delay_lobby_start(2.0, "waiting for initial brawler auto-select")
            self.state = None
            try:
                self.Time_management.states["state_check"] = 0.0
            except Exception:
                pass
            try:
                self.max_ips = int(load_toml_as_dict("cfg/general_config.toml")['max_ips'])
            except ValueError:
                self.max_ips = None
            self.run_for_minutes = int(load_toml_as_dict("cfg/general_config.toml")['run_for_minutes'])
            self.start_time = time.time()
            self.time_to_stop = False
            self.in_cooldown = False
            self.cooldown_start_time = 0
            self.cooldown_duration = 3 * 60
            self.last_processed_frame_time = 0.0
            self._was_paused = False
            # Expose stop/pause events to dashboard so it can control the bot
            if _active_dashboard is not None:
                _active_dashboard._bot_stop_event = self._stop_event
                _active_dashboard._bot_pause_event = self._pause_event
            self._start_hotkey_listener()
            self.current_ips = 0.0
            self.last_console_stats_time = 0
            self._last_dashboard_match_counter = int(getattr(self.Stage_manager.Trophy_observer, "match_counter", 0) or 0)
            self._last_roster_signature = None
            self._last_roster_push_time = 0.0
            self._last_live_exception_time = 0.0
            self._last_live_stats_push = 0.0
            self._last_fast_result_probe = 0.0
            self._last_dashboard_history_revision = int(
                getattr(self.Stage_manager.Trophy_observer, "history_revision", 0) or 0
            )
            self._live_push_interval = 0.35
            self._live_stats_push_interval = 0.35
            self._roster_push_interval = 2.5
            self._out_of_match_since = 0.0
            self._out_of_match_latched = False
            self._last_match_phase_reset_time = 0.0
            # Match duration watchdog: detect stuck-in-match (idle disconnect, etc.)
            self._match_watchdog_start = 0.0   # when we entered 'match' state
            self._match_watchdog_max = 6 * 60  # 6 minutes max per match
            time_thresholds = load_toml_as_dict("cfg/time_tresholds.toml")
            self.match_ready_at = 0.0
            self.low_ips_threshold = float(time_thresholds.get("low_ips_recovery_threshold", 4.0))
            self.low_ips_startup_grace_seconds = float(time_thresholds.get("low_ips_startup_grace_seconds", 60))
            self.low_ips_match_grace_seconds = float(time_thresholds.get("low_ips_match_grace_seconds", 12))
            self.low_ips_recovery_seconds = float(time_thresholds.get("low_ips_recovery_seconds", 45))
            self.low_ips_recovery_cooldown = float(time_thresholds.get("low_ips_recovery_cooldown", 35))
            self.low_ips_app_restart_after = int(time_thresholds.get("low_ips_app_restart_after", 3))
            self.low_ips_emulator_restart_after = int(time_thresholds.get("low_ips_emulator_restart_after", 6))
            self.low_ips_since = None
            self.last_low_ips_recovery = 0.0
            self.low_ips_recovery_attempts = 0
            # Periodic garbage collection
            self._last_gc_time = time.time()

        def initialize_stage_manager(self):
            active = data[0] if data else {}
            self.Stage_manager.Trophy_observer.win_streak = self.Stage_manager._coerce_int(active.get('win_streak'), 0)
            self.Stage_manager.Trophy_observer.current_trophies = self.Stage_manager._coerce_int(active.get('trophies'), 0)
            self.Stage_manager.Trophy_observer.current_wins = self.Stage_manager._coerce_int(active.get('wins'), 0)

        @staticmethod
        def _start_easyocr_warmup():
            def warm():
                reader.warm_up()

            threading.Thread(target=warm, name="easyocr-warmup", daemon=True).start()

        def _ensure_easyocr_warmup(self):
            if self._easyocr_warmup_started:
                return
            self._start_easyocr_warmup()
            self._easyocr_warmup_started = True

        def _post_match_context_active(self, now=None, runtime_state=None):
            if now is None:
                now = time.time()
            if runtime_state is None:
                runtime_state = str(getattr(self.Play, "_runtime_state", "") or "")
            if runtime_state.startswith("end") or runtime_state in self._reward_context_states:
                return True
            pending = getattr(self.Stage_manager, "is_post_match_resolution_pending", None)
            if callable(pending):
                return bool(pending(now))
            return bool(
                getattr(self.Stage_manager, "_awaiting_lobby_result_sync", False)
                and not getattr(self.Stage_manager, "_match_in_progress", False)
            )

        def _should_warm_post_match_ocr(self, now=None):
            checker = getattr(self.Stage_manager, "should_warm_post_match_ocr", None)
            if callable(checker):
                return bool(checker(now))
            return bool(
                getattr(self.Stage_manager, "_awaiting_lobby_result_sync", False)
                and not getattr(self.Stage_manager, "_result_applied_for_active_match", False)
            )

        def _normalize_post_match_state(self, state, runtime_state, now):
            if (
                state == "match"
                and getattr(self.Stage_manager, "_awaiting_lobby_result_sync", False)
                and not getattr(self.Stage_manager, "_match_in_progress", False)
            ):
                should_hold = getattr(self.Stage_manager, "should_hold_match_probe", None)
                if callable(should_hold) and not should_hold(now):
                    return state
                held_state = ""
                getter = getattr(self.Stage_manager, "get_end_transition_state", None)
                if callable(getter):
                    held_state = getter()
                return held_state or (runtime_state if str(runtime_state).startswith("end") else "end")
            return state

        def _run_initial_brawler_select(self):
            if not self._pending_initial_brawler_select:
                return True

            target_brawler = str(self._startup_brawler_target or "").strip().lower()
            if not target_brawler:
                self._pending_initial_brawler_select = False
                return True

            self._ensure_easyocr_warmup()

            print(f"[STARTUP] Selecting brawler '{target_brawler}' before first match")
            try:
                selected = self.lobby_automator.select_brawler(target_brawler)
                if not selected:
                    print(
                        f"[STARTUP] Could not confirm selection for '{target_brawler}'. "
                        "Keeping lobby start blocked and retrying."
                    )
                    self.Stage_manager._delay_lobby_start(2.0, "waiting for initial brawler auto-select")
                    return False
            except Exception as exc:
                print(
                    f"[STARTUP] Brawler selection failed for '{target_brawler}': {exc}. "
                    "Keeping lobby start blocked and retrying."
                )
                self.Stage_manager._delay_lobby_start(2.0, "waiting for initial brawler auto-select")
                return False

            self._pending_initial_brawler_select = False
            self.Stage_manager._lobby_start_blocked_until = 0.0
            self.Stage_manager._lobby_start_block_reason = ""
            now = time.time()
            self.Play.time_since_last_proceeding = now
            for key in getattr(self.Play, "time_since_detections", {}):
                self.Play.time_since_detections[key] = now
            print(f"[STARTUP] Ready to play with '{target_brawler}'")
            return True

        def _cancel_initial_brawler_select(self, reason):
            if not self._pending_initial_brawler_select:
                return
            self._pending_initial_brawler_select = False
            target_brawler = str(self._startup_brawler_target or "").strip().lower()
            if reason:
                print(
                    f"[STARTUP] Skipping initial brawler selection for "
                    f"'{target_brawler}' because {reason}."
                )

        def _start_hotkey_listener(self):
            """Poll F9 (stop), F8 (pause/resume), F7 (toggle overlay) via Windows API.
            Uses key-up detection to prevent rapid toggling when a key is held."""
            def _listen():
                user32 = ctypes.windll.user32
                VK_F9, VK_F8, VK_F7 = 0x78, 0x77, 0x76
                # Track previous key states (True = was pressed last poll)
                prev_f9 = False
                prev_f8 = False
                prev_f7 = False
                while not self._stop_event.is_set():
                    f9_down = bool(user32.GetAsyncKeyState(VK_F9) & 0x8000)
                    f8_down = bool(user32.GetAsyncKeyState(VK_F8) & 0x8000)
                    f7_down = bool(user32.GetAsyncKeyState(VK_F7) & 0x8000)

                    # F9 - stop (trigger on key DOWN, one-shot)
                    if f9_down and not prev_f9:
                        cprint("[HOTKEY] F9 pressed -> STOPPING", "#FF4444")
                        self._stop_event.set()
                        break

                    # F8 - pause/resume (trigger on key DOWN edge only)
                    if f8_down and not prev_f8:
                        if self._pause_event.is_set():
                            self._pause_event.clear()
                            cprint("[HOTKEY] F8 pressed -> RESUMED", "#00DC64")
                        else:
                            self._pause_event.set()
                            cprint("[HOTKEY] F8 pressed -> PAUSED", "#FFD700")

                    # F7 - toggle overlay (trigger on key DOWN edge only)
                    if f7_down and not prev_f7:
                        self._toggle_visual_overlay()

                    prev_f9 = f9_down
                    prev_f8 = f8_down
                    prev_f7 = f7_down
                    time.sleep(0.05)
            threading.Thread(target=_listen, daemon=True).start()

        def _toggle_visual_overlay(self):
            """Toggle the transparent visual overlay on/off and update config."""
            try:
                gc = load_toml_as_dict("cfg/general_config.toml")
                current = str(gc.get("visual_overlay_enabled", "no")).lower() in ("yes", "true", "1")
                new_val = "no" if current else "yes"
                gc["visual_overlay_enabled"] = new_val
                from utils import save_dict_as_toml
                save_dict_as_toml(gc, "cfg/general_config.toml")

                # Hot-reload the overlay if it exists
                overlay = getattr(self.Play, 'visual_overlay', None)
                if overlay:
                    overlay.reload_config()
                elif new_val == "yes":
                    # Create overlay if it doesn't exist yet
                    from visual_overlay import VisualOverlay
                    self.Play.visual_overlay = VisualOverlay()

                state_text = "ON" if new_val == "yes" else "OFF"
                color = "#AA66FF" if new_val == "yes" else "#555555"
                cprint(f"[HOTKEY] F7 pressed -> Visual Overlay {state_text}", color)
            except Exception as e:
                print(f"[OVERLAY] Toggle failed: {e}")

        def reset_low_ips_watchdog(self, recovered=True):
            self.low_ips_since = None
            if recovered:
                self.low_ips_recovery_attempts = 0

        def recover_low_ips(self, current_ips):
            if current_ips <= 0:
                return False
            if self.max_ips and self.max_ips <= self.low_ips_threshold:
                return False
            now = time.time()
            if now - self.start_time < self.low_ips_startup_grace_seconds:
                return False
            if self.state == "match" and self.match_ready_at and now - self.match_ready_at < self.low_ips_match_grace_seconds:
                return False
            if current_ips >= self.low_ips_threshold:
                if self.low_ips_since is not None:
                    print(f"IPS recovered to {current_ips:.2f}; clearing low-IPS watchdog.")
                self.reset_low_ips_watchdog(recovered=True)
                return False
            if self.low_ips_since is None:
                self.low_ips_since = now
                return False
            low_for = now - self.low_ips_since
            if low_for < self.low_ips_recovery_seconds or now - self.last_low_ips_recovery < self.low_ips_recovery_cooldown:
                return False
            self.last_low_ips_recovery = now
            self.low_ips_recovery_attempts += 1
            self.window_controller.keys_up(list("wasd"))
            print(f"IPS stayed low ({current_ips:.2f}) for {low_for:.1f}s; recovery attempt {self.low_ips_recovery_attempts}.")
            if self.low_ips_recovery_attempts >= 2 and hasattr(self.window_controller, "reduce_capture_load_for_slow_feed"):
                try:
                    self.window_controller.reduce_capture_load_for_slow_feed()
                except Exception as exc:
                    print(f"Could not reduce scrcpy capture load: {exc}")
            if self.low_ips_recovery_attempts >= self.low_ips_app_restart_after:
                print("Low IPS persisted; restarting Brawl Stars and scrcpy.")
                self.restart_brawl_stars()
                try:
                    self.window_controller.restart_scrcpy_client()
                except Exception:
                    pass
            else:
                print("Low IPS detected; restarting scrcpy feed.")
                try:
                    self.window_controller.restart_scrcpy_client()
                except Exception:
                    self.restart_brawl_stars()
            self.last_processed_frame_time = 0.0
            self.low_ips_since = now
            gc.collect()
            return True

        @staticmethod
        def load_models():
            folder_path = "./models/"
            model_names = ['mainInGameModel.onnx', 'tileDetector.onnx']
            loaded_models = []

            for name in model_names:
                loaded_models.append(folder_path + name)
            return loaded_models

        def restart_brawl_stars(self):
            """Restart Brawl Stars via ADB after a crash/stuck situation."""
            print("[RECOVERY] Restarting Brawl Stars...")
            self.window_controller.keys_up(list("wasd"))

            try:
                # Force stop Brawl Stars
                self.window_controller.device.shell(f"am force-stop {BRAWL_STARS_PACKAGE}")
                print("[RECOVERY] Brawl Stars force-stopped")
                time.sleep(3)

                # Reconnect scrcpy if the stream died
                if not self.window_controller.is_stream_alive():
                    print("[RECOVERY] Scrcpy stream dead, reconnecting...")
                    self.window_controller.reconnect_scrcpy()
                    time.sleep(2)

                # Restart Brawl Stars
                self.window_controller.device.shell(
                    f"monkey -p {BRAWL_STARS_PACKAGE} -c android.intent.category.LAUNCHER 1"
                )
                print("[RECOVERY] Brawl Stars restarting...")
                time.sleep(15)  # Wait for game to load

                # Verify we're getting frames
                _, ft = self.window_controller.get_latest_frame()
                if ft > 0 and (time.time() - ft) < self.window_controller.FRAME_STALE_TIMEOUT:
                    print("[RECOVERY] Game restarted successfully, resuming bot")
                    return
                else:
                    print("[RECOVERY] Still no fresh frames after restart")

            except Exception as e:
                print(f"[RECOVERY] Restart failed: {e}")

            # Fall back to notification + exit if recovery failed
            try:
                screenshot = self.window_controller.screenshot()
                notify_user("bot_stuck", screenshot=screenshot)
            except Exception:
                pass
            print("Recovery failed. User notified. Shutting down.")
            self.window_controller.close()
            import sys
            sys.exit(1)

        def manage_time_tasks(self, frame):
            now = time.time()
            self._skip_play_cycle = False
            runtime_state = str(getattr(self.Play, "_runtime_state", "") or "")
            post_match_context = self._post_match_context_active(now, runtime_state)
            if post_match_context and self._should_warm_post_match_ocr(now) and not reader.is_ready():
                self._ensure_easyocr_warmup()
            if post_match_context and now - self._last_fast_result_probe >= 0.35:
                self._last_fast_result_probe = now
                reward_ocr_ready = reader.is_ready()
                fast_state = get_state(frame, allow_reward_ocr=reward_ocr_ready)
                fast_state = self._normalize_post_match_state(fast_state, runtime_state, now)
                if fast_state != "match" or (
                    getattr(self.Stage_manager, "_awaiting_lobby_result_sync", False)
                    and not getattr(self.Stage_manager, "_match_in_progress", False)
                ):
                    self.state = fast_state
                    self.Play._runtime_state = fast_state
                    if fast_state != "match":
                        self.Play.time_since_last_proceeding = now
                    frame_data = frame if (
                        fast_state in self.states_requiring_data
                        or str(fast_state).startswith("end")
                    ) else None
                    self.Stage_manager.do_state(fast_state, frame_data)
                    self._skip_play_cycle = True
                    return
            if (
                runtime_state == "match"
                and now - self._last_fast_result_probe >= 0.6
            ):
                self._last_fast_result_probe = now
                fast_result = find_game_result(frame)
                if fast_result:
                    print(f"[RESULT] fast probe detected {fast_result} in manage_time_tasks")
                    end_state = f"end_{fast_result}"
                    self.state = end_state
                    self.Play._runtime_state = end_state
                    self.Stage_manager.do_state(end_state, frame)
                    self._skip_play_cycle = True
                    return

            if self.Time_management.state_check():
                try:
                    reward_ocr_ready = self.Stage_manager._is_easyocr_ready()
                except Exception:
                    reward_ocr_ready = bool(reader.is_ready())
                allow_reward_ocr = (
                    reward_ocr_ready
                    and self._post_match_context_active(now, runtime_state)
                )
                state = get_state(frame, allow_reward_ocr=allow_reward_ocr)
                state = self._normalize_post_match_state(state, runtime_state, now)
                if self.Stage_manager._awaiting_lobby_result_sync and state in {"lobby", "match"}:
                    try:
                        screenshot = self.window_controller.screenshot()
                        confirmed_state = get_state(screenshot, allow_reward_ocr=reward_ocr_ready)
                        confirmed_state = self._normalize_post_match_state(confirmed_state, runtime_state, time.time())
                        if isinstance(confirmed_state, str) and confirmed_state.startswith("end_"):
                            state = confirmed_state
                            frame = screenshot
                        elif confirmed_state in self._reward_context_states or confirmed_state == "end":
                            state = confirmed_state
                            frame = screenshot
                        elif confirmed_state == "lobby":
                            frame = screenshot
                    except Exception:
                        pass
                if (
                    self._pending_initial_brawler_select
                    and (
                        state == "match"
                        or runtime_state == "match"
                        or self.Stage_manager._match_in_progress
                    )
                ):
                    self._cancel_initial_brawler_select("match context was already detected")
                if state == "lobby" and self._pending_initial_brawler_select:
                    if not self._run_initial_brawler_select():
                        self.state = "lobby"
                        self.Play._runtime_state = "lobby"
                        self.Play.time_since_last_proceeding = time.time()
                        self._skip_play_cycle = True
                        return
                    try:
                        frame = self.window_controller.screenshot()
                        state = get_state(frame)
                    except Exception:
                        state = "lobby"
                previous_state = self.state
                self.state = state
                self.Play._runtime_state = state
                if state != "match":
                    self.Play.time_since_last_proceeding = time.time()
                    self._match_watchdog_start = 0.0  # reset watchdog
                else:
                    if previous_state != "match" or not getattr(self.Stage_manager, "_match_in_progress", False):
                        if self.Stage_manager.mark_match_started():
                            self.match_ready_at = time.time()
                    # Track how long we've been continuously in 'match'
                    if self._match_watchdog_start <= 0:
                        self._match_watchdog_start = time.time()
                        # Reset detection timestamps so the no-detections watchdog
                        # counts from the start of this match, not from a previous one.
                        now_t = time.time()
                        for k in self.Play.time_since_detections:
                            self.Play.time_since_detections[k] = now_t
                    elif (time.time() - self._match_watchdog_start) > self._match_watchdog_max:
                        print(f"[WATCHDOG] Stuck in 'match' state for >{self._match_watchdog_max/60:.0f} min, forcing restart")
                        self._match_watchdog_start = time.time()  # reset to avoid rapid re-triggers
                        self.restart_brawl_stars()
                        return
                frame_data = frame if (state in self.states_requiring_data or str(state).startswith("end_")) else None
                self.Stage_manager.do_state(state, frame_data)
                if (
                    state != "match"
                    or (
                        getattr(self.Stage_manager, "_awaiting_lobby_result_sync", False)
                        and not getattr(self.Stage_manager, "_match_in_progress", False)
                    )
                ):
                    self._skip_play_cycle = True

            if self.state == "match" and self.Time_management.no_detections_check():
                frame_data = self.Play.time_since_detections
                for key, value in frame_data.items():
                    if time.time() - value > self.no_detections_action_threshold:
                        print(f"[NO-DETECT] No '{key}' detection for >{self.no_detections_action_threshold//60:.0f} min in match state - restarting")
                        self.restart_brawl_stars()
                        return

            if self.Time_management.idle_check():
                #print("check for idle!")
                self.lobby_automator.check_for_idle(frame)

            # Periodic garbage collection (every 5 minutes)
            if (time.time() - self._last_gc_time) > 300:
                self._last_gc_time = time.time()
                gc.collect()

        def main(self): #this is for timer to stop after time
            s_time = time.time()
            c = 0
            _consecutive_errors = 0  # crash recovery counter
            while True:
                # stop / Pause controls
                if self._stop_event.is_set():
                    cprint("Bot stopped by user", "#FF4444")
                    self.window_controller.keys_up(list("wasd"))
                    break

                if self._pause_event.is_set():
                    if not self._was_paused:
                        self.window_controller.keys_up(list("wasd"))
                        cprint("Bot paused (F8 or button)", "#FFD700")
                        self._was_paused = True
                    time.sleep(0.3)
                    continue
                elif self._was_paused:
                    cprint("Bot resumed", "#00DC64")
                    self._was_paused = False

                if self.max_ips:
                    frame_start = time.perf_counter()
                if self.run_for_minutes > 0 and not self.in_cooldown:
                    elapsed_time = (time.time() - self.start_time) / 60
                    if elapsed_time >= self.run_for_minutes:
                        cprint(f"timer is done, {self.run_for_minutes} is over. continuing for 3 minutes if in game", "#AAE5A4")
                        self.in_cooldown = True # tries to finish game if in game
                        self.cooldown_start_time = time.time()
                        self.Stage_manager.states['lobby'] = lambda data: 0

                if self.in_cooldown:
                    if time.time() - self.cooldown_start_time >= self.cooldown_duration:
                        cprint("stopping bot fully", "#AAE5A4")
                        break

                if abs(s_time - time.time()) > 1:
                    elapsed = time.time() - s_time
                    if elapsed > 0:
                        self.current_ips = c / elapsed
                        if debug:
                            print(f"{self.current_ips:.2f} IPS")
                        if self.recover_low_ips(self.current_ips):
                            s_time = time.time()
                            c = 0
                            continue
                    s_time = time.time()
                    c = 0

                frame_bgr, frame_time = self.window_controller.wait_for_next_frame(
                    self.last_processed_frame_time,
                    copy_frame=False,
                )
                if frame_bgr is None:
                    self.Play.window_controller.keys_up(list("wasd"))
                    print("No fresh frame received -- pausing actions until feed resumes")
                    if not self.window_controller.is_stream_alive():
                        print("[RECOVERY] Scrcpy stream died, attempting reconnect...")
                        if self.window_controller.reconnect_scrcpy():
                            print("[RECOVERY] Stream reconnected, resuming")
                        else:
                            print("[RECOVERY] Reconnect failed, restarting Brawl Stars...")
                            self.restart_brawl_stars()
                    time.sleep(1)
                    continue
                if (
                    self.state == "match"
                    and hasattr(self.window_controller, "is_connection_healthy")
                    and not self.window_controller.is_connection_healthy()
                ):
                    self.Play.window_controller.keys_up(list("wasd"))
                    print("Frozen scrcpy feed detected -- restarting feed before issuing inputs")
                    try:
                        self.window_controller.restart_scrcpy_client()
                    except Exception as exc:
                        print(f"Could not restart scrcpy after frozen frame detection: {exc}")
                        self.restart_brawl_stars()
                    self.last_processed_frame_time = 0.0
                    time.sleep(1)
                    continue

                self.last_processed_frame_time = frame_time
                frame = FrameData(frame_bgr[:, :, ::-1].copy())

                try:
                    self.manage_time_tasks(frame)
                    if self._skip_play_cycle:
                        c += 1
                        continue

                    runtime_state = str(getattr(self.Play, "_runtime_state", "") or "")
                    post_match_pending = (
                        getattr(self.Stage_manager, "_awaiting_lobby_result_sync", False)
                        and not getattr(self.Stage_manager, "_match_in_progress", False)
                    )
                    if (
                        self.state != "match"
                        or runtime_state.startswith("end")
                        or runtime_state in self._reward_context_states
                        or post_match_pending
                    ):
                        self.window_controller.keys_up(list("wasd"))
                        c += 1
                        continue

                    # push current state to visual overlay so it auto-hides
                    #    outside of a match (lobby, shop, brawler_selection, etc.)
                    _vo = getattr(self.Play, 'visual_overlay', None)
                    if _vo is not None:
                        _vo.set_game_state(self.state or 'starting')

                    # Apply auto-detected game mode if available
                    if self.Stage_manager.detected_game_mode_type is not None:
                        if self.Play.game_mode != self.Stage_manager.detected_game_mode_type:
                            print(f"[AUTO-MODE] Updating game_mode: {self.Play.game_mode} -> {self.Stage_manager.detected_game_mode_type} ({self.Stage_manager.detected_game_mode})")
                        resolved_mode_name = self.Stage_manager.detected_game_mode
                        if self.Play._showdown_detected_in_match and not self.Stage_manager.is_showdown:
                            resolved_mode_name = self.Play.game_mode_name
                        self.Play.apply_gamemode_context(
                            gamemode_name=resolved_mode_name or "unknown",
                            gamemode_type=self.Stage_manager.detected_game_mode_type,
                            is_showdown=self.Stage_manager.is_showdown or self.Play._showdown_detected_in_match,
                        )

                        # Pass objective waypoint from GAMEMODE_MAP to Play
                        from stage_manager import GAMEMODE_MAP
                        mode_key = (self.Stage_manager.detected_game_mode or '').lower()
                        mode_info = GAMEMODE_MAP.get(mode_key)
                        if mode_info and mode_info.get("objective"):
                            ox, oy = mode_info["objective"]
                            self.Play.objective_pos = (
                                ox * self.Play.window_controller.width_ratio,
                                oy * self.Play.window_controller.height_ratio
                            )
                        else:
                            self.Play.objective_pos = None

                    brawler = self.Stage_manager.brawlers_pick_data[0]['brawler']
                    # Update Play's current brawler if it changed (e.g. trophy farm mode)
                    if self.Play.current_brawler != brawler:
                        print(f"[MAIN] Brawler changed: {self.Play.current_brawler} -> {brawler}")
                        self.Play.current_brawler = brawler
                        # Register new brawler for session summary
                        tobs = self.Stage_manager.Trophy_observer
                        tobs.start_session_brawler(
                            brawler, self.Stage_manager.brawlers_pick_data[0].get('trophies', 0))

                    # Pass stats info to Play so debug overlay shows everything
                    target_val = self.Stage_manager.brawlers_pick_data[0].get('push_until', '?')
                    self.Play._stats_info = {
                        'ips': self.current_ips,
                        'start_time': self.start_time,
                        'state': self.state or 'starting',
                        'trophy_observer': self.Stage_manager.Trophy_observer,
                        'target': target_val,
                    }

                    # RL match boundary guard:
                    # Reset exactly once on stable out-of-match transitions.
                    # Includes stable "end" (new round / end screen), but uses
                    # latching + cooldown to prevent duplicate resets from flicker.
                    stable_out_states = {"lobby", "brawler_selection", "shop", "trophy_reward", "popup", "end"}
                    is_end_state = str(self.state).startswith("end")
                    if self.state == "match":
                        self._out_of_match_since = 0.0
                        self._out_of_match_latched = False
                    elif self.Play._match_phase_set and (self.state in stable_out_states or is_end_state):
                        if self._out_of_match_since <= 0:
                            self._out_of_match_since = time.time()
                        stable_needed = 0.5 if is_end_state else 1.0
                        cooldown_ok = (time.time() - self._last_match_phase_reset_time) >= 4.0
                        if (not self._out_of_match_latched
                                and cooldown_ok
                                and (time.time() - self._out_of_match_since) >= stable_needed):
                            self.Play._match_phase_set = False
                            self.Play._spawn_detected = False
                            self.Play._spawn_detect_frames = 0
                            self._out_of_match_since = 0.0
                            self._out_of_match_latched = True
                            self._last_match_phase_reset_time = time.time()
                            print(f"[MAIN] Stable out-of-match state '{self.state}' -> match phase flags reset")
                    else:
                        self._out_of_match_since = 0.0
                        self._out_of_match_latched = False

                    self.Play.main(frame, brawler)
                    pending_result = getattr(self.Play, "_pending_end_result", None)
                    if pending_result:
                        print(f"[RESULT] play.py queued pending result {pending_result}")
                        end_state = f"end_{pending_result}"
                        self.Play._pending_end_result = None
                        self.state = end_state
                        self.Play._runtime_state = end_state
                        self.Stage_manager.do_state(end_state, frame)
                        c += 1
                        continue
                    c += 1

                    tobs = self.Stage_manager.Trophy_observer
                    bname = brawler
                    hist = tobs.match_history.get(bname, {})
                    # push live data to the dashboard
                    _now = time.time()
                    should_push_live = (_now - getattr(self, '_last_live_push', 0)) >= self._live_push_interval
                    if should_push_live:
                        self._last_live_push = _now

                        all_brawler_stats = getattr(tobs, 'brawler_stats', {}) or {}
                        brawler_perf = (
                            all_brawler_stats.get(bname, {})
                            or all_brawler_stats.get(str(bname).lower(), {})
                            or all_brawler_stats.get(str(bname).upper(), {})
                        )
                        bt = getattr(self.Play, '_bt_combat', None)
                        rc = getattr(bt, '_reward_calculator', None) if bt else None
                        trainer = getattr(bt, '_rl_trainer', None) if bt else None
                        rl_summary = {}
                        if rc and hasattr(rc, 'episode_summary'):
                            try:
                                rl_summary = rc.episode_summary() or {}
                            except Exception:
                                rl_summary = {}
                        rl_buffer_size = 0
                        rl_buffer_capacity = 0
                        if trainer and getattr(trainer, 'buffer', None) is not None:
                            try:
                                rl_buffer_size = len(trainer.buffer)
                                rl_buffer_capacity = int(getattr(trainer.buffer, 'max_size', 0) or 0)
                            except Exception:
                                rl_buffer_size = 0
                                rl_buffer_capacity = 0

                        play_kills = int(getattr(self.Play, '_enemies_killed_this_match', 0) or 0)
                        play_deaths = int(getattr(self.Play, '_death_count', 0) or 0)
                        bt_damage = float(getattr(bt, '_session_damage_dealt', 0.0) if bt else 0.0)
                        rl_kills_live = int(rl_summary.get('kills', 0) or 0)
                        rl_deaths_live = int(rl_summary.get('deaths', 0) or 0)
                        rl_damage_live = float(rl_summary.get('damage_dealt', 0.0) or 0.0)

                        current_kills_live = max(play_kills, rl_kills_live)
                        current_deaths_live = max(play_deaths, rl_deaths_live)
                        current_damage_live = int(max(bt_damage, rl_damage_live))
                        if (
                            hasattr(tobs, 'update_live_match_stats')
                            and (_now - self._last_live_stats_push) >= self._live_stats_push_interval
                        ):
                            try:
                                tobs.update_live_match_stats(
                                    bname,
                                    kills=current_kills_live,
                                    assists=0,
                                    damage=current_damage_live,
                                    deaths=current_deaths_live,
                                )
                                self._last_live_stats_push = _now
                            except Exception:
                                pass
                        session_stats = getattr(tobs, 'session_stats', {}) or {}

                        perf_source_parts = []
                        if bt and bt_damage > 0:
                            perf_source_parts.append("BT")
                        if rl_damage_live > 0 or rl_kills_live > 0 or rl_deaths_live > 0:
                            perf_source_parts.append("RL")
                        if play_kills > 0 or play_deaths > 0:
                            perf_source_parts.append("PLAY")
                        perf_source = "+".join(perf_source_parts) if perf_source_parts else "INIT"

                        global _active_dashboard
                        if _active_dashboard is not None:
                            active_entry = self.Stage_manager.brawlers_pick_data[0] if self.Stage_manager.brawlers_pick_data else {}
                            coerce_int = getattr(self.Stage_manager, "_coerce_int", None)
                            if not callable(coerce_int):
                                def coerce_int(value, default=0):
                                    try:
                                        return int(value)
                                    except (TypeError, ValueError):
                                        return default
                            current_match_counter = int(getattr(tobs, "match_counter", 0) or 0)
                            current_history_revision = int(getattr(tobs, "history_revision", 0) or 0)
                            awaiting_verified_sync = bool(
                                getattr(self.Stage_manager, "_awaiting_lobby_result_sync", False)
                                and not getattr(tobs, "_lobby_trophy_verified", False)
                            )
                            if awaiting_verified_sync:
                                live_trophies = coerce_int(active_entry.get("trophies", 0), 0)
                                live_wins = coerce_int(active_entry.get("wins", 0), 0)
                                live_streak = coerce_int(active_entry.get("win_streak", 0), 0)
                            else:
                                live_trophies = coerce_int(
                                    tobs.current_trophies if getattr(tobs, "current_trophies", None) is not None else active_entry.get("trophies", 0),
                                    0,
                                )
                                live_wins = coerce_int(
                                    getattr(tobs, "current_wins", None) if getattr(tobs, "current_wins", None) is not None else active_entry.get("wins", 0),
                                    0,
                                )
                                live_streak = coerce_int(
                                    tobs.win_streak if getattr(tobs, "win_streak", None) is not None else active_entry.get("win_streak", 0),
                                    0,
                                )
                            if active_entry and not awaiting_verified_sync:
                                if coerce_int(active_entry.get("trophies", 0), 0) != live_trophies:
                                    active_entry["trophies"] = live_trophies
                                if coerce_int(active_entry.get("wins", 0), 0) != live_wins:
                                    active_entry["wins"] = live_wins
                                if coerce_int(active_entry.get("win_streak", 0), 0) != live_streak:
                                    active_entry["win_streak"] = live_streak
                            roster_signature = tuple(
                                (
                                    str(entry.get('brawler', '')),
                                    int(entry.get('trophies', 0) or 0),
                                    int(entry.get('wins', 0) or 0),
                                    int(entry.get('win_streak', 0) or 0),
                                    int(entry.get('push_until', 0) or 0),
                                    str(entry.get('type', 'trophies') or 'trophies'),
                                    bool(entry.get('automatically_pick', True)),
                                    bool(entry.get('manual_trophies', False)),
                                )
                                for entry in (self.Stage_manager.brawlers_pick_data or [])
                                if isinstance(entry, dict)
                            )
                            should_sync_roster = (
                                roster_signature != self._last_roster_signature
                                or current_match_counter != self._last_dashboard_match_counter
                                or current_history_revision != self._last_dashboard_history_revision
                                or (_now - self._last_roster_push_time) >= self._roster_push_interval
                            )
                            if should_sync_roster:
                                try:
                                    _active_dashboard.sync_runtime_roster(
                                        self.Stage_manager.brawlers_pick_data,
                                        emit_history=(
                                            current_match_counter != self._last_dashboard_match_counter
                                            or current_history_revision != self._last_dashboard_history_revision
                                        ),
                                    )
                                    self._last_roster_signature = roster_signature
                                    self._last_roster_push_time = _now
                                    self._last_dashboard_match_counter = current_match_counter
                                    self._last_dashboard_history_revision = current_history_revision
                                except Exception as _roster_exc:
                                    if _now - self._last_live_exception_time >= 5.0:
                                        print(f"[LIVE] Roster sync error: {_roster_exc}")
                                        self._last_live_exception_time = _now
                            try:
                                session_victories = int(session_stats.get('victories', 0) or 0)
                                session_defeats = int(session_stats.get('defeats', 0) or 0)
                                session_draws = int(session_stats.get('draws', 0) or 0)
                                session_matches = int(session_stats.get('total_matches', 0) or 0)
                                if session_matches <= 0:
                                    session_matches = current_match_counter
                                _active_dashboard.update_live(
                                    start_time=self.start_time,
                                    ips=self.current_ips,
                                    state=self.state or 'starting',
                                    brawler=bname,
                                    trophies=live_trophies,
                                    target=target_val,
                                    victories=session_victories,
                                    defeats=session_defeats,
                                    draws=session_draws,
                                    streak=live_streak,
                                    game_mode=self.Play.game_mode_name if self.Play.game_mode_name != 'unknown' else (self.Stage_manager.detected_game_mode or ''),
                                    playstyle='hold' if self.Play.must_brawler_hold_attack(bname, self.Play.brawlers_info) else 'tap',
                                    player_hp=self.Play.player_hp_percent,
                                    enemy_hp=self.Play.enemy_hp_percent,
                                    hp_confidence_player=getattr(self.Play, '_hp_confidence_player', 1.0),
                                    hp_confidence_enemy=getattr(self.Play, '_hp_confidence_enemy', 1.0),
                                    is_dead=self.Play._is_dead,
                                    target_name=self.Play.target_info.get('name'),
                                    target_hp=self.Play.target_info.get('hp', -1),
                                    n_enemies=self.Play.target_info.get('n_enemies', 0),
                                    enemies=1 if (time.time() - self.Play.time_since_detections.get('enemy', 0)) < 0.5 else 0,
                                    movement=self.Play.last_movement,
                                    walls=len(self.Play.last_walls_data),
                                    gadget_ready=self.Play.is_gadget_ready,
                                    super_ready=self.Play.is_super_ready,
                                    hypercharge_ready=self.Play.is_hypercharge_ready,
                                    decision=self.Play.last_decision_reason,
                                    ammo=getattr(self.Play, "current_ammo", getattr(self.Play, "_ammo", 0)),
                                    our_score=self.Play._our_score,
                                    their_score=self.Play._their_score,
                                    score_diff=self.Play._score_diff,
                                    deaths=self.Play._death_count,
                                    current_kills=current_kills_live,
                                    current_deaths=current_deaths_live,
                                    current_assists=0,
                                    current_damage=current_damage_live,
                                    kills=current_kills_live,
                                    assists=0,
                                    damage=current_damage_live,
                                    match_active=(self.state == 'match'),
                                    perf_source=perf_source,
                                    total_kills=session_stats.get('total_kills', 0),
                                    total_assists=session_stats.get('total_assists', 0),
                                    total_damage=session_stats.get('total_damage', 0),
                                    total_matches=session_matches,
                                    session_matches=session_matches,
                                    session_victories=session_victories,
                                    session_defeats=session_defeats,
                                    session_draws=session_draws,
                                    last_kills=session_stats.get('last_match_kills', 0),
                                    last_assists=session_stats.get('last_match_assists', 0),
                                    last_damage=session_stats.get('last_match_damage', 0),
                                    last_result=getattr(tobs, "last_match_result", None),
                                    last_trophy_delta=getattr(tobs, "last_match_trophy_delta", 0),
                                    last_trophy_delta_verified=getattr(tobs, "last_match_trophies_verified", False),
                                    last_streak_bonus=getattr(tobs, "last_match_streak_bonus", 0),
                                    last_underdog_bonus=getattr(tobs, "last_match_underdog_bonus", 0),
                                    last_trophy_adjustment=getattr(tobs, "last_match_trophy_adjustment", 0),
                                    avg_kills=brawler_perf.get('avg_kills', 0),
                                    avg_assists=brawler_perf.get('avg_assists', 0),
                                    avg_damage=brawler_perf.get('avg_damage', 0),
                                    farm_mode=self.Stage_manager.smart_trophy_farm,
                                    farm_remaining=len(self.Stage_manager.brawlers_pick_data),
                                    rl_training_enabled=trainer is not None,
                                    rl_total_episodes=int(getattr(trainer, 'total_episodes', 0) if trainer else 0),
                                    rl_total_updates=int(getattr(trainer, 'total_updates', 0) if trainer else 0),
                                    rl_buffer_size=int(rl_buffer_size),
                                    rl_buffer_capacity=int(rl_buffer_capacity),
                                    rl_episode_reward=float(rl_summary.get('total_reward', 0.0) or 0.0),
                                    rl_kills=int(rl_summary.get('kills', 0) or 0),
                                    rl_deaths=int(rl_summary.get('deaths', 0) or 0),
                                    rl_damage_dealt=int(rl_summary.get('damage_dealt', 0) or 0),
                                    rl_damage_taken=int(rl_summary.get('damage_taken', 0) or 0),
                                    rl_hit_rate=float(rl_summary.get('hit_rate', -1) or -1),
                                )
                            except Exception as _live_exc:
                                if _now - self._last_live_exception_time >= 5.0:
                                    print(f"[LIVE] Dashboard update error: {_live_exc}")
                                    self._last_live_exception_time = _now

                    # --- ADAPTIVE AGGRESSION based on win rate ---
                    total_games = hist.get('victory', 0) + hist.get('defeat', 0) + hist.get('draw', 0)
                    if total_games >= 3:
                        wr = hist.get('victory', 0) / total_games
                        if wr < 0.35:
                            self.Play.aggression_modifier = 0.7   # Losing a lot -> play defensive
                        elif wr < 0.45:
                            self.Play.aggression_modifier = 0.85  # Below average -> slightly defensive
                        elif wr > 0.65:
                            self.Play.aggression_modifier = 1.3   # Winning a lot -> push aggressively
                        elif wr > 0.55:
                            self.Play.aggression_modifier = 1.15  # Above average -> slightly aggressive
                        else:
                            self.Play.aggression_modifier = 1.0   # Balanced

                    # Console stats every 30 seconds
                    now = time.time()
                    if now - self.last_console_stats_time >= 30:
                        self.last_console_stats_time = now
                        tobs = self.Stage_manager.Trophy_observer
                        elapsed = now - self.start_time
                        m, s = divmod(int(elapsed), 60)
                        hrs, m = divmod(m, 60)
                        bname = brawler
                        hist = tobs.match_history.get(bname, {})
                        v = hist.get('victory', 0)
                        d_count = hist.get('defeat', 0)
                        dr = hist.get('draw', 0)
                        total = v + d_count + dr
                        wr = (v / total * 100) if total > 0 else 0
                        print(f"\n=== STATS [{hrs:02d}:{m:02d}:{s:02d}] ===")
                        print(f"  Brawler: {bname} | Trophies: {tobs.current_trophies} -> {target_val}")
                        print(f"  Wins: {tobs.current_wins} | Streak: {tobs.win_streak} | Matches: {tobs.match_counter}")
                        print(f"  History: {v}V / {d_count}D / {dr}Dr | WR: {wr:.0f}%")
                        print(f"  IPS: {self.current_ips:.1f} | State: {self.state}")
                        print(f"====================\n")

                    if self.max_ips:
                        target_period = 1 / self.max_ips
                        work_time = time.perf_counter() - frame_start
                        if work_time < target_period:
                            time.sleep(target_period - work_time)

                    _consecutive_errors = 0  # reset on successful iteration

                except Exception as _loop_err:
                    _consecutive_errors += 1
                    print(f"[ERROR] Main loop exception ({_consecutive_errors}): {_loop_err}")
                    traceback.print_exc()
                    try:
                        self.window_controller.keys_up(list("wasd"))
                    except Exception:
                        pass
                    if _consecutive_errors >= 10:
                        print("[RECOVERY] 10 consecutive errors, restarting Brawl Stars...")
                        _consecutive_errors = 0
                        try:
                            self.restart_brawl_stars()
                        except Exception as _re:
                            print(f"[RECOVERY] Restart also failed: {_re}")
                    time.sleep(2)

    main = Main()
    main.main()
    # session Summary
    try:
        summary = main.Stage_manager.Trophy_observer.get_session_summary()
        _print_session_summary(summary)
        # Push summary to dashboard for popup
        global _active_dashboard
        if _active_dashboard is not None:
            try:
                _active_dashboard._session_summary = summary
                _active_dashboard.after(100, _active_dashboard._show_session_summary)
            except Exception:
                pass
    except Exception as e:
        print(f"[SUMMARY] Error generating summary: {e}")
    # clean shutdown
    cprint("Cleaning up...", "#888888")
    try:
        main.window_controller.keys_up(list("wasd"))
    except Exception:
        pass
    try:

        main.window_controller.close()
    except Exception:
        pass
    cprint("Bot shutdown complete.", "#00DC64")


if __name__ == "__main__":
    all_brawlers = get_brawler_list()
    update_icons()
    if api_base_url != "localhost":
        update_missing_brawlers_info(all_brawlers)

        check_version()
        update_wall_model_classes()
        if not current_wall_model_is_latest():
            print("New Wall detection model found, downloading... (this might take a few minutes depending on your internet speed)")
            get_latest_wall_model_file()

    # Use the smaller ratio to maintain aspect ratio
    app = App(login, SelectBrawler, pyla_main, all_brawlers, Hub)
    app.start(pyla_version, get_latest_version)
