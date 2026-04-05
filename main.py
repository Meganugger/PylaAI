import asyncio
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
from state_finder.main import get_state
from time_management import TimeManagement
from utils import load_toml_as_dict, current_wall_model_is_latest, api_base_url, ensure_state_icons_present
from utils import get_brawler_list, update_missing_brawlers_info, check_version, async_notify_user, \
    update_wall_model_classes, get_latest_wall_model_file, get_latest_version, cprint, record_timing
from window_controller import WindowController

pyla_version = load_toml_as_dict("./cfg/general_config.toml")['pyla_version']

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"


def pyla_main(data):
    class Main:

        def __init__(self):
            self.window_controller = WindowController()
            self.Play = Play(*self.load_models(), self.window_controller)
            self.Time_management = TimeManagement()
            self.lobby_automator = LobbyAutomation(self.window_controller)
            self.Stage_manager = StageManager(data, self.lobby_automator, self.window_controller)
            self.states_requiring_frame_data = ["play_store", "lobby", "popup", "end", "reward_claim"]
            if data[0]['automatically_pick']:
                if debug: print("Picking brawler automatically")
                self.lobby_automator.select_brawler(data[0]['brawler'])
            self.Play.current_brawler = data[0]['brawler']
            self.no_detections_action_threshold = 60 * 8
            self.initialize_stage_manager()
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

        def initialize_stage_manager(self):
            self.Stage_manager.Trophy_observer.win_streak = data[0]['win_streak']
            self.Stage_manager.Trophy_observer.current_trophies = data[0]['trophies']
            self.Stage_manager.Trophy_observer.current_wins = data[0]['wins'] if data[0]['wins'] != "" else 0

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
            if current_state == "play_store":
                self.Stage_manager.click_brawl_stars(screenshot)
                time.sleep(10)
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)

            if current_state != "play_store":
                now = time.time()
                self.Play.time_since_last_proceeding = now
                for key in self.Play.time_since_detections:
                    self.Play.time_since_detections[key] = now
                print("Brawl Stars recovered successfully.")
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

        def manage_time_tasks(self, frame):
            if self.Time_management.state_check():
                state = get_state(frame)
                if state != "match":
                    self.Play.time_since_last_proceeding = time.time()
                frame_data = frame if state in self.states_requiring_frame_data else None
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
                        print(f"{c / elapsed:.2f} IPS")
                    s_time = time.time()
                    c = 0

                wait_started_at = time.perf_counter()
                frame, frame_time = self.window_controller.wait_for_next_frame(
                    self.last_processed_frame_time,
                    copy_frame=True
                )
                record_timing("frame_wait", time.perf_counter() - wait_started_at, print_every=120)
                if frame is None:
                    self.Play.window_controller.keys_up(list("wasd"))
                    print("Stale frame detected -- pausing actions until feed resumes")
                    time.sleep(1)
                    continue
                self.last_processed_frame_time = frame_time

                tasks_started_at = time.perf_counter()
                self.manage_time_tasks(frame)
                record_timing("time_tasks", time.perf_counter() - tasks_started_at, print_every=120)


                brawler = self.Stage_manager.brawlers_pick_data[0]['brawler']
                play_started_at = time.perf_counter()
                self.Play.main(frame, brawler)
                record_timing("play_main", time.perf_counter() - play_started_at, print_every=120)
                c += 1

                if self.max_ips:
                    target_period = 1 / self.max_ips
                    work_time = time.perf_counter() - frame_start
                    if work_time < target_period:
                        time.sleep(target_period - work_time)
                record_timing("runtime_loop", time.perf_counter() - loop_started_at, print_every=120)

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
