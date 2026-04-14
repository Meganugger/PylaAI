import os.path
import sys

import asyncio
import time

import cv2
import numpy as np

from state_finder.main import get_state
from trophy_observer import TrophyObserver
from utils import find_template_center, load_toml_as_dict, async_notify_user, \
    save_brawler_data, reader

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"


def load_image(image_path, scale_factor):
    # Load the image
    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]

    # Calculate the new dimensions based on the scale factor
    new_width = int(orig_width * scale_factor)
    new_height = int(orig_height * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

class StageManager:

    def __init__(self, brawlers_data, lobby_automator, window_controller):
        self.state_handlers = {
            'shop': self.quit_shop,
            'brawler_selection': self.quit_shop,
            'popup': self.close_pop_up,
            'reward_claim': self.claim_reward,
            'match': lambda: 0,
            'end': self.end_game,
            'end_victory': self.end_game,
            'end_defeat': self.end_game,
            'end_draw': self.end_game,
            'lobby': self.start_game,
            'star_drop': self.click_star_drop
        }
        self.Lobby_automation = lobby_automator
        self.lobby_config = load_toml_as_dict("./cfg/lobby_config.toml")
        self.brawl_stars_icon = None
        self.close_popup_icon = None
        self.brawlers_pick_data = brawlers_data
        brawler_list = [brawler["brawler"] for brawler in brawlers_data]
        self.Trophy_observer = TrophyObserver(brawler_list)
        self.time_since_last_stat_change = time.time()
        self.long_press_star_drop = load_toml_as_dict("./cfg/general_config.toml")["long_press_star_drop"]
        self.window_controller = window_controller
        self.lobby_start_enabled = True
        self._awaiting_lobby_result_sync = False

    def _sync_active_brawler_progress(self):
        if not self.brawlers_pick_data:
            return
        active = self.brawlers_pick_data[0]
        active['trophies'] = self.Trophy_observer.current_trophies
        active['wins'] = self.Trophy_observer.current_wins
        active['win_streak'] = self.Trophy_observer.win_streak

    @staticmethod
    def validate_trophies(trophies_string):
        trophies_string = str(trophies_string or "").lower()
        replacements = {
            "s": "5",
            "o": "0",
            "i": "1",
            "l": "1",
            "|": "1",
            "b": "8",
        }
        for source, target in replacements.items():
            trophies_string = trophies_string.replace(source, target)
        numbers = ''.join(filter(str.isdigit, trophies_string))
        if not numbers:
            return False
        return int(numbers)

    def mark_match_started(self):
        self._awaiting_lobby_result_sync = True

    def _read_lobby_trophies(self, frame):
        region = (self.lobby_config.get("lobby") or {}).get("trophy_observer")
        if frame is None or not region or len(region) != 4:
            return None

        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        x, y, width, height = region
        x1 = max(0, int(x * wr))
        y1 = max(0, int(y * hr))
        x2 = min(frame.shape[1], int((x + width) * wr))
        y2 = min(frame.shape[0], int((y + height) * hr))
        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return None

        try:
            ocr_result = reader.readtext(cropped)
        except Exception as exc:
            if debug:
                print(f"Lobby trophy OCR failed: {exc}")
            return None

        baseline = int(self.Trophy_observer.current_trophies or self.brawlers_pick_data[0].get("trophies", 0) or 0)
        candidates = []
        for _bbox, text, _prob in ocr_result:
            value = self.validate_trophies(text)
            if value is not False:
                candidates.append(int(value))

        if not candidates:
            return None

        expected_values = {baseline}
        try:
            expected_values.add(max(0, baseline - int(self.Trophy_observer.calc_lost_decrement() or 0)))
        except Exception:
            pass
        try:
            expected_values.add(baseline + int(self.Trophy_observer.calc_win_increment() or 0))
        except Exception:
            pass

        best = min(candidates, key=lambda value: min(abs(value - expected) for expected in expected_values))
        best_delta = min(abs(best - expected) for expected in expected_values)
        if best_delta > 35:
            return None
        return best

    def _sync_lobby_result(self, frame):
        if not self._awaiting_lobby_result_sync or not self.brawlers_pick_data:
            return False

        screenshot = frame
        try:
            screenshot = self.window_controller.screenshot()
        except Exception:
            pass

        verified_trophies = self._read_lobby_trophies(screenshot)
        if verified_trophies is None:
            return False

        current_trophies = int(self.Trophy_observer.current_trophies or self.brawlers_pick_data[0].get("trophies", 0) or 0)
        if verified_trophies > current_trophies:
            inferred_result = "victory"
        elif verified_trophies < current_trophies:
            inferred_result = "defeat"
        else:
            inferred_result = "draw"

        applied = self._apply_match_result(inferred_result)
        if not applied:
            return False

        if self.Trophy_observer.current_trophies != verified_trophies:
            self.Trophy_observer.change_trophies(verified_trophies)
        self._sync_active_brawler_progress()
        save_brawler_data(self.brawlers_pick_data)
        self._awaiting_lobby_result_sync = False
        if debug:
            print(f"Lobby result sync applied as '{inferred_result}' ({current_trophies} -> {verified_trophies})")
        return True

    def _apply_match_result(self, game_result):
        if not self.brawlers_pick_data or not game_result:
            return False
        if not self._awaiting_lobby_result_sync:
            return False

        current_brawler = self.brawlers_pick_data[0]['brawler']
        applied = self.Trophy_observer.add_trophies(game_result, current_brawler)
        self.Trophy_observer.add_win(game_result)
        self.time_since_last_stat_change = time.time()

        values = {
            "trophies": self.Trophy_observer.current_trophies,
            "wins": self.Trophy_observer.current_wins
        }
        type_to_push = self.brawlers_pick_data[0]['type']
        if type_to_push not in values:
            type_to_push = "trophies"
        value = values[type_to_push]

        self._sync_active_brawler_progress()
        self.brawlers_pick_data[0][type_to_push] = value
        save_brawler_data(self.brawlers_pick_data)
        self._awaiting_lobby_result_sync = False
        return applied

    def set_lobby_start_enabled(self, enabled):
        self.lobby_start_enabled = enabled

    def start_game(self, data):
        print("state is lobby, starting game")
        self._sync_lobby_result(data)
        values = {
            "trophies": self.Trophy_observer.current_trophies,
            "wins": self.Trophy_observer.current_wins
        }

        type_of_push = self.brawlers_pick_data[0]['type']
        if type_of_push not in values:
            type_of_push = "trophies"
        value = values[type_of_push]
        if value == "" and type_of_push == "wins":
            value = 0
        push_current_brawler_till = self.brawlers_pick_data[0]['push_until']
        if push_current_brawler_till == "" and type_of_push == "wins":
            push_current_brawler_till = 300
        if push_current_brawler_till == "" and type_of_push == "trophies":
            push_current_brawler_till = 1000

        if value >= push_current_brawler_till:
            if len(self.brawlers_pick_data) <= 1:
                print("Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                      "Bot will now pause itself until closed.", value, push_current_brawler_till)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    screenshot = self.window_controller.screenshot()
                    loop.run_until_complete(async_notify_user("bot_is_stuck", screenshot))
                finally:
                    loop.close()
                print("Bot stopping: all targets completed with no more brawlers.")
                self.window_controller.keys_up(list("wasd"))
                self.window_controller.close()
                sys.exit(0)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                screenshot = self.window_controller.screenshot()
                loop.run_until_complete(async_notify_user(self.brawlers_pick_data[0]["brawler"], screenshot))
            finally:
                loop.close()
            self.brawlers_pick_data.pop(0)
            self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
            self.Trophy_observer.current_wins = self.brawlers_pick_data[0]['wins'] if self.brawlers_pick_data[0]['wins'] != "" else 0
            self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
            next_brawler_name = self.brawlers_pick_data[0]['brawler']
            if self.brawlers_pick_data[0]["automatically_pick"]:
                if debug: print("Picking next automatically picked brawler")
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                if current_state != "lobby":
                    print("Trying to reach the lobby to switch brawler")

                max_attempts = 30
                attempts = 0
                while current_state != "lobby" and attempts < max_attempts:
                    self.window_controller.press_continue()
                    if debug: print("Pressed Q to return to lobby")
                    time.sleep(1)
                    screenshot = self.window_controller.screenshot()
                    current_state = get_state(screenshot)
                    attempts += 1
                if attempts >= max_attempts:
                    print("Failed to reach lobby after max attempts")
                else:
                    self.Lobby_automation.select_brawler(next_brawler_name)
            else:
                print("Next brawler is in manual mode, waiting 10 seconds to let user switch.")

        # q btn is over the start btn
        self.window_controller.keys_up(list("wasd"))
        self.window_controller.press_key("Q")
        print("Pressed Q to start a match")

    def click_brawl_stars(self, frame):
        if isinstance(frame, np.ndarray):
            screenshot = frame[4:31, 50:900]
        else:
            screenshot = frame.crop((50, 4, 900, 31))
        if self.brawl_stars_icon is None:
            self.brawl_stars_icon = load_image("state_finder/images_to_detect/brawl_stars_icon.png",
                                               self.window_controller.scale_factor)
        detection = find_template_center(screenshot, self.brawl_stars_icon)
        if detection:
            x, y = detection
            self.window_controller.click(x=x + 50, y=y)
    def click_star_drop(self):
        if self.long_press_star_drop == "yes":
            self.window_controller.press_continue(hold_seconds=10, include_fallback_clicks=False)
        else:
            self.window_controller.press_continue()

    def claim_reward(self, frame=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()
        if self.close_popup_icon is None:
            self.close_popup_icon = load_image(
                "state_finder/images_to_detect/close_popup.png",
                self.window_controller.scale_factor
            )

        popup_location = find_template_center(screenshot, self.close_popup_icon)
        if popup_location:
            self.window_controller.click(*popup_location)
            return

        self.window_controller.press_continue()

    def end_game(self, frame=None, known_result=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()

        found_game_result = False
        current_state = get_state(screenshot)
        if known_result in {"victory", "defeat", "draw"}:
            found_game_result = known_result if self._apply_match_result(known_result) else False
            current_state = f"end_{known_result}"
        max_end_attempts = 30
        end_attempts = 0
        while str(current_state).startswith("end") and end_attempts < max_end_attempts:
            state_result = None
            if isinstance(current_state, str) and current_state.startswith("end_"):
                state_result = current_state.split("_", 1)[1]

            should_probe_result = (
                not found_game_result
                and (state_result is not None or time.time() - self.time_since_last_stat_change > 10)
            )
            if should_probe_result:
                if state_result is not None:
                    found_game_result = state_result if self._apply_match_result(state_result) else False
                else:
                    detected = self.Trophy_observer.find_game_result(
                        screenshot,
                        current_brawler=self.brawlers_pick_data[0]['brawler'],
                        game_result=state_result,
                    )
                    if detected:
                        found_game_result = getattr(self.Trophy_observer, "_last_game_result", False)
                        self.time_since_last_stat_change = time.time()
                        self._sync_active_brawler_progress()
                        save_brawler_data(self.brawlers_pick_data)
                push_current_brawler_till = self.brawlers_pick_data[0]['push_until']
                values = {
                    "trophies": self.Trophy_observer.current_trophies,
                    "wins": self.Trophy_observer.current_wins
                }
                type_to_push = self.brawlers_pick_data[0]['type']
                if type_to_push not in values:
                    type_to_push = "trophies"
                value = values[type_to_push]

                if value == "" and type_to_push == "wins":
                    value = 0
                if push_current_brawler_till == "" and type_to_push == "wins":
                    push_current_brawler_till = 300
                if push_current_brawler_till == "" and type_to_push == "trophies":
                    push_current_brawler_till = 1000

                if value >= push_current_brawler_till:
                    if len(self.brawlers_pick_data) <= 1:
                        print(
                            "Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                            "Bot will now pause itself until closed.")
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            screenshot = self.window_controller.screenshot()
                            loop.run_until_complete(async_notify_user("completed", screenshot))
                        finally:
                            loop.close()
                        if os.path.exists("latest_brawler_data.json"):
                            os.remove("latest_brawler_data.json")
                        print("Bot stopping: all targets completed.")
                        self.window_controller.keys_up(list("wasd"))
                        self.window_controller.close()
                        sys.exit(0)
            self.window_controller.press_continue()
            if debug: print("Game has ended, pressing Q")
            time.sleep(1)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            end_attempts += 1
        if end_attempts >= max_end_attempts:
            print("End game screen stuck for too long, forcing continue")
        if debug: print("Game has ended", current_state)

    def quit_shop(self):
        self.window_controller.click(100*self.window_controller.width_ratio, 60*self.window_controller.height_ratio)

    def close_pop_up(self, frame=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()
        if self.close_popup_icon is None:
            self.close_popup_icon = load_image("state_finder/images_to_detect/close_popup.png", self.window_controller.scale_factor)
        popup_location = find_template_center(screenshot, self.close_popup_icon)
        if popup_location:
            self.window_controller.click(*popup_location)

    def do_state(self, state, data=None):
        known_result = None
        if isinstance(state, str) and state.startswith("end_"):
            known_result = state.split("_", 1)[1]
            state = "end"
        if state == "lobby" and not self.lobby_start_enabled:
            return

        if state == "end":
            self.state_handlers[state](data, known_result)
            return
        if data is not None:
            self.state_handlers[state](data)
            return
        self.state_handlers[state]()

