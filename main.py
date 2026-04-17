import asyncio
import sys
import threading
import time

from runtime_threads import apply_process_thread_limits

apply_process_thread_limits()

from gui.hub import Hub
from gui.login import login
from gui.main import App
from gui.select_brawler import SelectBrawler
from lobby_automation import LobbyAutomation
from play import Play
from stage_manager import StageManager
from state_finder.main import get_state, find_game_result
from time_management import TimeManagement
from utils import load_toml_as_dict, current_wall_model_is_latest, api_base_url, ensure_state_icons_present
from utils import get_brawler_list, update_missing_brawlers_info, check_version, async_notify_user, \
    update_wall_model_classes, get_latest_wall_model_file, get_latest_version, cprint, record_timing, reader
from window_controller import WindowController

pyla_version = load_toml_as_dict("./cfg/general_config.toml")['pyla_version']

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"
_active_dashboard = None
_active_stage_manager = None


def _set_active_dashboard_instance(instance):
    global _active_dashboard
    _active_dashboard = instance
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "_active_dashboard", instance)


def _get_active_dashboard_instance():
    if _active_dashboard is not None:
        return _active_dashboard
    main_module = sys.modules.get("__main__")
    return getattr(main_module, "_active_dashboard", None) if main_module is not None else None


def _set_active_stage_manager_instance(instance):
    global _active_stage_manager
    _active_stage_manager = instance
    main_module = sys.modules.get("__main__")
    if main_module is not None:
        setattr(main_module, "_active_stage_manager", instance)


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
            _set_active_stage_manager_instance(self.Stage_manager)
            self.states_requiring_frame_data = ["lobby", "popup", "end", "reward_claim"]
            active_entry = data[0] if data else {}
            self._startup_brawler_target = str(active_entry.get('brawler', '') or '').strip().lower()
            self._pending_initial_brawler_select = bool(
                self._startup_brawler_target and active_entry.get('automatically_pick', True)
            )
            self.Play.current_brawler = self._startup_brawler_target
            self.no_detections_action_threshold = 60 * 8
            self.initialize_stage_manager()
            self._easyocr_warmup_started = False
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
            self.in_cooldown = False
            self.cooldown_start_time = 0
            self.cooldown_duration = 3 * 60
            self.last_processed_frame_time = 0.0
            self.current_ips = 0.0
            self.current_state = "starting"
            self._last_fast_result_probe = 0.0
            self._last_dashboard_match_counter = int(getattr(self.Stage_manager.Trophy_observer, "match_counter", 0) or 0)
            self._last_dashboard_history_revision = int(
                getattr(self.Stage_manager.Trophy_observer, "history_revision", 0) or 0
            )
            self._last_live_push = 0.0
            self._last_live_stats_push = 0.0
            self._last_roster_signature = None
            self._last_roster_push_time = 0.0
            self._last_live_exception_time = 0.0
            self._live_push_interval = 0.25
            self._live_stats_push_interval = 0.25
            self._roster_push_interval = 2.0

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

        @staticmethod
        def load_models():
            folder_path = "./models/"
            model_names = ['mainInGameModel.onnx', 'tileDetector.onnx']
            loaded_models = []

            for name in model_names:
                loaded_models.append(folder_path + name)
            return loaded_models

        def restart_brawl_stars(self):
            print("Trying to recover Brawl Stars...")
            self.window_controller.keys_up(list("wasd"))

            self.window_controller.ensure_brawl_stars_running(force=True)

            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            now = time.time()
            self.Play.time_since_last_proceeding = now
            for key in self.Play.time_since_detections:
                self.Play.time_since_detections[key] = now
            print(f"Brawl Stars recovered successfully (state: {current_state}).")
            return

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(async_notify_user("bot_is_stuck", screenshot))
            finally:
                loop.close()
            print("Bot got stuck. User notified. Shutting down.")
            self.window_controller.keys_up(list("wasd"))
            self.window_controller.close()
            import sys
            sys.exit(1)

        def _run_initial_brawler_select(self):
            if not self._pending_initial_brawler_select:
                return False

            self._pending_initial_brawler_select = False
            if not self._easyocr_warmup_started:
                self._start_easyocr_warmup()
                self._easyocr_warmup_started = True
            target_brawler = str(self._startup_brawler_target or "").strip().lower()
            if not target_brawler:
                return False

            print(f"[STARTUP] Selecting brawler '{target_brawler}' before first match")
            try:
                selected = self.lobby_automator.select_brawler(target_brawler)
                if selected is False:
                    print(
                        f"[STARTUP] Could not confirm selection for '{target_brawler}'. "
                        "Continuing with the current lobby brawler."
                    )
                else:
                    print(f"[STARTUP] Ready to play with '{target_brawler}'")
            except Exception as exc:
                print(
                    f"[STARTUP] Brawler selection failed for '{target_brawler}': {exc}. "
                    "Continuing with the current lobby brawler."
                )

            now = time.time()
            self.Play.time_since_last_proceeding = now
            self.Play.time_since_player_last_found = now
            for key in self.Play.time_since_detections:
                self.Play.time_since_detections[key] = now
            return True

        def manage_time_tasks(self, frame):
            now = time.time()
            runtime_state = str(getattr(self.Play, "_runtime_state", "") or "")
            player_missing_for = max(
                0.0,
                now - float(getattr(self.Play, "time_since_player_last_found", now) or now),
            )
            if (
                runtime_state == "match"
                and now - self._last_fast_result_probe >= 0.6
            ):
                self._last_fast_result_probe = now
                fast_result = find_game_result(frame)
                if fast_result:
                    print(f"[RESULT] fast probe detected {fast_result} in manage_time_tasks")
                    end_state = f"end_{fast_result}"
                    self.current_state = end_state
                    self.Play._runtime_state = end_state
                    self.Stage_manager.do_state(end_state, frame)
                    return

            if self.Time_management.state_check():
                allow_reward_ocr = (
                    runtime_state.startswith("end_")
                    or runtime_state == "reward_claim"
                )
                state = get_state(frame, allow_reward_ocr=allow_reward_ocr)
                if state == "match" and hasattr(self.Play, "note_confirmed_match_state"):
                    self.Play.note_confirmed_match_state(now)
                if not self._easyocr_warmup_started and self._pending_initial_brawler_select:
                    self._start_easyocr_warmup()
                    self._easyocr_warmup_started = True
                if state == "lobby" and self._pending_initial_brawler_select:
                    self._run_initial_brawler_select()
                    try:
                        frame = self.window_controller.screenshot()
                        state = get_state(frame)
                    except Exception:
                        state = "lobby"
                if (
                    state == "starting"
                    and runtime_state == "match"
                    and getattr(self.Play, "has_recent_match_context", lambda *_args, **_kwargs: False)(now)
                ):
                    if debug:
                        print(f"[MATCH] keeping runtime state as match despite state probe '{state}'")
                    state = "match"
                if self.Stage_manager._awaiting_lobby_result_sync and state in {"lobby", "match"}:
                    try:
                        screenshot = self.window_controller.screenshot()
                        confirmed_state = get_state(
                            screenshot,
                            allow_reward_ocr=allow_reward_ocr,
                        )
                        if confirmed_state == "match" and hasattr(self.Play, "note_confirmed_match_state"):
                            self.Play.note_confirmed_match_state(time.time())
                        if isinstance(confirmed_state, str) and confirmed_state.startswith("end_"):
                            state = confirmed_state
                            frame = screenshot
                        elif confirmed_state == "lobby":
                            frame = screenshot
                    except Exception:
                        pass
                self.current_state = state
                self.Play._runtime_state = state
                if state == "match":
                    self.Stage_manager.mark_match_started()
                if state != "match":
                    self.Play.time_since_last_proceeding = time.time()
                frame_data = frame if (state in self.states_requiring_frame_data or str(state).startswith("end_")) else None
                self.Stage_manager.do_state(state, frame_data)

            if self.Time_management.no_detections_check():
                frame_data = self.Play.time_since_detections
                for key, value in frame_data.items():
                    if time.time() - value > self.no_detections_action_threshold:
                        self.restart_brawl_stars()

            if self.Time_management.idle_check():
                #print("check for idle!")
                self.lobby_automator.check_for_idle(frame)

        def enter_cooldown_mode(self):
            cprint(
                f"timer is done, {self.run_for_minutes} is over. continuing for 3 minutes if in game",
                "#AAE5A4",
            )
            self.in_cooldown = True
            self.cooldown_start_time = time.time()
            self.Stage_manager.set_lobby_start_enabled(False)

        def main(self): #this is for timer to stop after time
            s_time = time.time()
            c = 0
            while True:
                if self._stop_event.is_set():
                    break
                if self._pause_event.is_set():
                    try:
                        self.window_controller.keys_up(list("wasd"))
                    except Exception:
                        pass
                    time.sleep(0.1)
                    continue
                loop_started_at = time.perf_counter()
                if self.max_ips:
                    frame_start = time.perf_counter()
                if self.run_for_minutes > 0 and not self.in_cooldown:
                    elapsed_time = (time.time() - self.start_time) / 60
                    if elapsed_time >= self.run_for_minutes:
                        self.enter_cooldown_mode()

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
                    s_time = time.time()
                    c = 0

                wait_started_at = time.perf_counter()
                frame_timeout = 15.0 if self.last_processed_frame_time <= 0 else 1.0
                frame, frame_time = self.window_controller.get_current_frame(
                    copy_frame=False,
                    timeout=frame_timeout,
                )
                record_timing("frame_fetch", time.perf_counter() - wait_started_at, print_every=120)
                if frame is None:
                    self.Play.window_controller.keys_up(list("wasd"))
                    print("Stale frame detected -- pausing actions until feed resumes")
                    time.sleep(1)
                    continue
                frame_age = time.time() - frame_time if frame_time > 0 else float("inf")
                if frame_age > self.window_controller.FRAME_STALE_TIMEOUT:
                    self.Play.window_controller.keys_up(list("wasd"))
                    print("Stale frame detected -- pausing actions until feed resumes")
                    self.window_controller.ensure_brawl_stars_running(force=True)
                    time.sleep(1)
                    continue
                self.last_processed_frame_time = frame_time

                tasks_started_at = time.perf_counter()
                self.manage_time_tasks(frame)
                record_timing("time_tasks", time.perf_counter() - tasks_started_at, print_every=120)

                runtime_state = str(getattr(self.Play, "_runtime_state", "") or "")
                if runtime_state.startswith("end_") or runtime_state == "reward_claim":
                    c += 1
                    continue

                brawler = self.Stage_manager.brawlers_pick_data[0]['brawler']
                if self.Play.current_brawler != brawler:
                    self.Play.current_brawler = brawler
                    self.Stage_manager.Trophy_observer.start_session_brawler(
                        brawler,
                        self.Stage_manager.brawlers_pick_data[0].get("trophies", 0),
                    )
                play_started_at = time.perf_counter()
                self.Play.main(frame, brawler)
                record_timing("play_main", time.perf_counter() - play_started_at, print_every=120)
                if getattr(self.Play, "_runtime_state", "") == "match":
                    self.Stage_manager.mark_match_started()
                pending_result = getattr(self.Play, "_pending_end_result", None)
                if pending_result:
                    print(f"[RESULT] play.py queued pending result {pending_result}")
                    end_state = f"end_{pending_result}"
                    self.Play._pending_end_result = None
                    self.current_state = end_state
                    self.Play._runtime_state = end_state
                    self.Stage_manager.do_state(end_state, frame)
                    c += 1
                    continue
                c += 1

                active_dashboard = _get_active_dashboard_instance()
                if active_dashboard is not None:
                    tobs = self.Stage_manager.Trophy_observer
                    active_entry = self.Stage_manager.brawlers_pick_data[0] if self.Stage_manager.brawlers_pick_data else {}
                    session_stats = getattr(tobs, "session_stats", {}) or {}
                    coerce_int = getattr(self.Stage_manager, "_coerce_int", None)
                    if not callable(coerce_int):
                        def coerce_int(value, default=0):
                            try:
                                return int(value)
                            except (TypeError, ValueError):
                                return default
                    current_match_counter = int(getattr(tobs, "match_counter", 0) or 0)
                    current_history_revision = int(getattr(tobs, "history_revision", 0) or 0)
                    current_kills = int(getattr(self.Play, "_enemies_killed_this_match", 0) or 0)
                    current_deaths = int(getattr(self.Play, "_death_count", 0) or 0)
                    current_damage = int(
                        getattr(self.Play, "_current_damage", 0)
                        or getattr(self.Play, "_damage_dealt", 0)
                        or 0
                    )
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
                    if active_entry:
                        if coerce_int(active_entry.get("trophies", 0), 0) != live_trophies:
                            active_entry["trophies"] = live_trophies
                        if coerce_int(active_entry.get("wins", 0), 0) != live_wins:
                            active_entry["wins"] = live_wins
                        if coerce_int(active_entry.get("win_streak", 0), 0) != live_streak:
                            active_entry["win_streak"] = live_streak
                    now = time.time()
                    if (
                        hasattr(tobs, "update_live_match_stats")
                        and (now - self._last_live_stats_push) >= self._live_stats_push_interval
                    ):
                        try:
                            tobs.update_live_match_stats(
                                brawler,
                                kills=current_kills,
                                assists=0,
                                damage=current_damage,
                                deaths=current_deaths,
                            )
                            self._last_live_stats_push = now
                        except Exception as live_stats_exc:
                            if time.time() - self._last_live_exception_time >= 5.0:
                                print(f"[LIVE] Match stat sync error: {live_stats_exc}")
                                self._last_live_exception_time = time.time()

                    roster_signature = tuple(
                        (
                            str(entry.get("brawler", "")),
                            int(entry.get("trophies", 0) or 0),
                            int(entry.get("wins", 0) or 0),
                            int(entry.get("win_streak", 0) or 0),
                            int(entry.get("push_until", 0) or 0),
                            str(entry.get("type", "trophies") or "trophies"),
                            bool(entry.get("automatically_pick", True)),
                            bool(entry.get("manual_trophies", False)),
                        )
                        for entry in (self.Stage_manager.brawlers_pick_data or [])
                        if isinstance(entry, dict)
                    )
                    should_sync_roster = (
                        roster_signature != self._last_roster_signature
                        or current_match_counter != self._last_dashboard_match_counter
                        or current_history_revision != self._last_dashboard_history_revision
                        or (now - self._last_roster_push_time) >= self._roster_push_interval
                    )
                    if should_sync_roster:
                        try:
                            active_dashboard.sync_runtime_roster(
                                self.Stage_manager.brawlers_pick_data,
                                emit_history=(
                                    current_match_counter != self._last_dashboard_match_counter
                                    or current_history_revision != self._last_dashboard_history_revision
                                ),
                            )
                            self._last_roster_signature = roster_signature
                            self._last_roster_push_time = now
                            self._last_dashboard_match_counter = current_match_counter
                            self._last_dashboard_history_revision = current_history_revision
                        except Exception as roster_exc:
                            if now - self._last_live_exception_time >= 5.0:
                                print(f"[LIVE] Roster sync error: {roster_exc}")
                                self._last_live_exception_time = now

                    if (now - self._last_live_push) >= self._live_push_interval:
                        self._last_live_push = now
                        try:
                            session_victories = int(session_stats.get("victories", 0) or 0)
                            session_defeats = int(session_stats.get("defeats", 0) or 0)
                            session_draws = int(session_stats.get("draws", 0) or 0)
                            session_matches = int(session_stats.get("total_matches", 0) or 0)
                            if session_matches <= 0:
                                session_matches = current_match_counter
                            active_dashboard.update_live(
                                start_time=self.start_time,
                                ips=self.current_ips,
                                state=self.current_state,
                                brawler=brawler,
                                trophies=live_trophies,
                                target=active_entry.get("push_until", 0),
                                victories=session_victories,
                                defeats=session_defeats,
                                draws=session_draws,
                                streak=live_streak,
                                game_mode=getattr(self.Play, "game_mode_name", ""),
                                gadget_ready=getattr(self.Play, "is_gadget_ready", False),
                                super_ready=getattr(self.Play, "is_super_ready", False),
                                hypercharge_ready=getattr(self.Play, "is_hypercharge_ready", False),
                                movement=getattr(self.Play, "last_movement", ""),
                                ammo=getattr(self.Play, "current_ammo", getattr(self.Play, "_ammo", 0)),
                                current_kills=current_kills,
                                current_deaths=current_deaths,
                                current_assists=0,
                                current_damage=current_damage,
                                kills=current_kills,
                                assists=0,
                                damage=current_damage,
                                total_kills=session_stats.get("total_kills", 0),
                                total_assists=session_stats.get("total_assists", 0),
                                total_damage=session_stats.get("total_damage", 0),
                                total_matches=session_matches,
                                session_matches=session_matches,
                                session_victories=session_victories,
                                session_defeats=session_defeats,
                                session_draws=session_draws,
                                last_kills=session_stats.get("last_match_kills", 0),
                                last_assists=session_stats.get("last_match_assists", 0),
                                last_damage=session_stats.get("last_match_damage", 0),
                                last_result=getattr(tobs, "last_match_result", None),
                                last_trophy_delta=getattr(tobs, "last_match_trophy_delta", 0),
                                last_trophy_delta_verified=getattr(tobs, "last_match_trophies_verified", False),
                                last_streak_bonus=getattr(tobs, "last_match_streak_bonus", 0),
                                last_underdog_bonus=getattr(tobs, "last_match_underdog_bonus", 0),
                                last_trophy_adjustment=getattr(tobs, "last_match_trophy_adjustment", 0),
                                match_active=self.current_state == "match",
                                farm_mode=str(getattr(self.Stage_manager, "smart_trophy_farm", False)).lower() in ("yes", "true", "1"),
                                farm_remaining=len(self.Stage_manager.brawlers_pick_data),
                            )
                        except Exception as live_exc:
                            if now - self._last_live_exception_time >= 5.0:
                                print(f"[LIVE] Dashboard update error: {live_exc}")
                                self._last_live_exception_time = now

                if self.max_ips:
                    target_period = 1 / self.max_ips
                    work_time = time.perf_counter() - frame_start
                    if work_time < target_period:
                        time.sleep(target_period - work_time)
                record_timing("runtime_loop", time.perf_counter() - loop_started_at, print_every=120)

            try:
                self.window_controller.keys_up(list("wasd"))
            except Exception:
                pass
            try:
                self.window_controller.close()
            except Exception:
                pass
            _set_active_dashboard_instance(None)
            _set_active_stage_manager_instance(None)

    main = Main()
    main.main()


all_brawlers = get_brawler_list()
ensure_state_icons_present()
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
