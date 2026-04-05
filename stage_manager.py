import os.path
import sys

import asyncio
import time

import cv2
import numpy as np
import requests

from state_finder.main import get_state
from trophy_observer import TrophyObserver
from utils import find_template_center, extract_text_and_positions, load_toml_as_dict, async_notify_user, \
    save_brawler_data

from difflib import SequenceMatcher

# Game mode name -> gamemode_type mapping for movement logic
GAMEMODE_MAP = {
    # Vertical-priority modes (gamemode_type=3): focus on pushing forward/up
    "knockout":    {"type": 3, "walls": True,  "showdown": False, "objective": None},
    "bounty":      {"type": 3, "walls": True,  "showdown": False, "objective": None},
    "gemgrab":     {"type": 3, "walls": True,  "showdown": False, "objective": (960, 400)},
    "gem grab":    {"type": 3, "walls": True,  "showdown": False, "objective": (960, 400)},
    "hotzone":     {"type": 3, "walls": True,  "showdown": False, "objective": (960, 540)},
    "hot zone":    {"type": 3, "walls": True,  "showdown": False, "objective": (960, 540)},
    # Horizontal-priority modes (gamemode_type=5): focus on pushing right
    "brawlball":   {"type": 5, "walls": True,  "showdown": False, "objective": (1700, 540)},
    "brawl ball":  {"type": 5, "walls": True,  "showdown": False, "objective": (1700, 540)},
    "heist":       {"type": 5, "walls": True,  "showdown": False, "objective": (1700, 540)},
    # Showdown modes - survival, stay near teammate
    "showdown":      {"type": 3, "walls": True, "showdown": True,  "objective": (960, 540)},
    "duo showdown":  {"type": 3, "walls": True, "showdown": True,  "objective": (960, 540)},
    "duoshowdown":   {"type": 3, "walls": True, "showdown": True,  "objective": (960, 540)},
    "solo showdown": {"type": 3, "walls": True, "showdown": True,  "objective": (960, 540)},
    "soloshowdown":  {"type": 3, "walls": True, "showdown": True,  "objective": (960, 540)},
    # Other
    "duels":       {"type": 3, "walls": True,  "showdown": False, "objective": None},
    "wipeout":     {"type": 3, "walls": True,  "showdown": False, "objective": None},
}


def detect_game_mode_from_frame(frame, window_controller):
    """Detect game mode from the lobby screen using OCR on the mode name area."""
    try:
        wr = window_controller.width_ratio
        hr = window_controller.height_ratio
        # The game mode name is displayed in the center-top of the lobby screen
        # Try a few crop regions where mode text commonly appears
        crop_regions = [
            (int(700 * wr), int(0 * hr), int(1220 * wr), int(80 * hr)),   # top center
            (int(600 * wr), int(50 * hr), int(1320 * wr), int(150 * hr)),  # slightly lower
            (int(500 * wr), int(0 * hr), int(1400 * wr), int(120 * hr)),   # wider top
        ]

        all_texts = {}
        for region in crop_regions:
            cropped = np.asarray(frame.crop(region))
            texts = extract_text_and_positions(cropped)
            all_texts.update(texts)

        if not all_texts:
            return None

        # Fuzzy match detected text against known game modes
        best_mode = None
        best_score = 0.0
        for text_key in all_texts:
            text_clean = text_key.lower().replace(' ', '').replace('-', '')
            for mode_name in GAMEMODE_MAP:
                mode_clean = mode_name.lower().replace(' ', '')
                # Exact or substring match
                if mode_clean in text_clean or text_clean in mode_clean:
                    # Use length similarity as a confidence score
                    shorter = min(len(mode_clean), len(text_clean))
                    longer = max(len(mode_clean), len(text_clean))
                    score = shorter / longer if longer > 0 else 0
                    if len(text_clean) >= 3 and score > best_score:
                        best_score = score
                        best_mode = mode_name
                # Fuzzy match
                ratio = SequenceMatcher(None, mode_clean, text_clean).ratio()
                if ratio > best_score and ratio >= 0.6:
                    best_score = ratio
                    best_mode = mode_name

        if best_mode:
            mode_info = GAMEMODE_MAP[best_mode]
            print(f"[AUTO-DETECT] Game mode detected: '{best_mode}' -> type={mode_info['type']}, walls={mode_info['walls']} (confidence={best_score:.2f})")
            return best_mode, mode_info
        else:
            print(f"[AUTO-DETECT] Could not detect game mode from text: {list(all_texts.keys())}")
            return None
    except Exception as e:
        print(f"[AUTO-DETECT] Error detecting game mode: {e}")
        return None

user_id = load_toml_as_dict("cfg/general_config.toml")['discord_id']
debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"
user_webhook = load_toml_as_dict("cfg/general_config.toml")['personal_webhook']


def notify_user(message_type):
    # message type will be used to have conditions determining the message
    # but for now there's only one possible type of message
    message_data = {
        'content': f"<@{user_id}> Pyla Bot has completed all it's targets !"
    }

    response = requests.post(user_webhook, json=message_data)

    if response.status_code != 204:
        print(
            f'Failed to send message. Be sure to have put a valid webhook url in the config. Status code: {response.status_code}')


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
        self.states = {
            'shop': self.quit_shop,
            'brawler_selection': self.quit_shop,
            'popup': self.close_pop_up,
            'match': lambda: 0,
            'end': self.end_game,
            'lobby': self.start_game,
            'play_store': self.click_brawl_stars,
            'star_drop': self.click_star_drop,
            'reward_claim': self.claim_reward,
            'trophy_reward': self.dismiss_trophy_reward,
            'idle_disconnect': self.handle_idle_disconnect,
            'mode_selection': self.handle_mode_selection,
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
        self.play_again_on_win = load_toml_as_dict("./cfg/bot_config.toml").get("play_again_on_win", "no") == "yes"
        self.window_controller = window_controller
        # Default to config values, auto-detection will override if successful
        _bot_cfg = load_toml_as_dict("./cfg/bot_config.toml")
        self.detected_game_mode = _bot_cfg.get("gamemode", "knockout")
        self.detected_game_mode_type = _bot_cfg.get("gamemode_type", 3)
        self.is_showdown = False  # Updated by detection

        # Smart Trophy Farm settings
        self.smart_trophy_farm = _bot_cfg.get("smart_trophy_farm", "no") == "yes"
        self.trophy_farm_target = int(_bot_cfg.get("trophy_farm_target", 500))
        self.trophy_farm_strategy = _bot_cfg.get("trophy_farm_strategy", "lowest_first")
        # Dynamic rotation: switch to lowest brawler every N matches
        self.dynamic_rotation_enabled = _bot_cfg.get("dynamic_rotation_enabled", "no") == "yes"
        self.dynamic_rotation_every = int(_bot_cfg.get("dynamic_rotation_every", 20))
        # Flag set by end_game when a brawler switch is queued for the next lobby visit
        self._pending_brawler_switch = False

    def start_brawl_stars(self, frame):
        data = extract_text_and_positions(np.asarray(frame))
        for key in list(data.keys()):
            if key.replace(" ", "") in ["brawl", "brawlstars", "stars"]:
                x, y = data[key]['center']
                self.window_controller.click(x, y)
                return

        brawl_stars_icon_coords = self.lobby_config['lobby'].get('brawl_stars_icon', [960, 540])
        x, y = brawl_stars_icon_coords[0]*self.window_controller.width_ratio, brawl_stars_icon_coords[1]*self.window_controller.height_ratio
        self.window_controller.click(x, y)

    @staticmethod
    def validate_trophies(trophies_string):
        trophies_string = trophies_string.lower()
        while "s" in trophies_string:
            trophies_string = trophies_string.replace("s", "5")
        numbers = ''.join(filter(str.isdigit, trophies_string))

        if not numbers:
            return False

        trophy_value = int(numbers)
        return trophy_value

    def _pick_next_farm_brawler(self):
        """Smart Trophy Farm: re-sort remaining brawlers and pick the best next one."""
        if len(self.brawlers_pick_data) <= 1:
            return False  # No more brawlers

        # Remove the current (completed) brawler
        completed = self.brawlers_pick_data.pop(0)
        print(f"[FARM] {completed['brawler'].title()} reached target. "
              f"{len(self.brawlers_pick_data)} brawlers remaining.")

        # Update trophies for remaining brawlers from their saved data
        # (they may have been updated by OCR corrections)
        remaining = self.brawlers_pick_data

        # Re-sort based on strategy
        if self.trophy_farm_strategy == "lowest_first":
            remaining.sort(key=lambda x: x.get("trophies", 0))
        elif self.trophy_farm_strategy == "highest_winrate":
            hist = self.Trophy_observer.match_history
            def wr_key(x):
                h = hist.get(x["brawler"], {})
                v = h.get("victory", 0) if isinstance(h, dict) else 0
                d = h.get("defeat", 0) if isinstance(h, dict) else 0
                total = v + d
                return -(v / total * 100) if total > 0 else 0
            remaining.sort(key=wr_key)
        elif self.trophy_farm_strategy == "sequential":
            remaining.sort(key=lambda x: x["brawler"])

        self.brawlers_pick_data = remaining
        next_b = self.brawlers_pick_data[0]
        print(f"[FARM] Next brawler: {next_b['brawler'].title()} "
              f"({next_b.get('trophies', 0)} trophies)")
        return True

    def _dynamic_rotate_to_lowest(self):
        """Dynamic rotation: re-sort all brawlers by trophies (ascending) and move
        the one with the fewest trophies to the front, WITHOUT removing any brawler
        from the queue.  Only triggers an actual brawler switch when the head changes.

        Returns True if the active brawler changed, False otherwise.
        """
        if len(self.brawlers_pick_data) <= 1:
            return False

        current_brawler = self.brawlers_pick_data[0]['brawler']
        self.brawlers_pick_data.sort(key=lambda x: x.get('trophies', 0))
        new_brawler = self.brawlers_pick_data[0]['brawler']

        if new_brawler == current_brawler:
            print(f"[FARM] Dynamic rotation: {current_brawler.title()} is already the lowest — no switch needed.")
            return False

        print(f"[FARM] Dynamic rotation: switching from {current_brawler.title()} "
              f"to {new_brawler.title()} ({self.brawlers_pick_data[0].get('trophies', 0)} trophies).")
        self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
        self.Trophy_observer.current_wins = (
            self.brawlers_pick_data[0]['wins']
            if self.brawlers_pick_data[0]['wins'] != "" else 0
        )
        self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
        save_brawler_data(self.brawlers_pick_data)
        return True

    def _handle_quest_rotation(self):
        """Quest Farm: check if current brawler's quest is completed (icon gone),
        and if so, switch to the next quest brawler.

        This is called at the start of each lobby visit (start_game) when
        type == 'quest'. On the very first call for a brawler (no matches
        played yet), it skips the check and just starts the game.
        """
        current = self.brawlers_pick_data[0]
        brawler_name = current['brawler']

        # Track matches played per quest brawler
        if not hasattr(self, '_quest_matches_played'):
            self._quest_matches_played = {}

        matches = self._quest_matches_played.get(brawler_name, 0)

        # Skip quest check on first game (no match has been played yet)
        if matches == 0:
            print(f"[QUEST] First game with {brawler_name.title()}, skipping quest check.")
            return

        # Check if the quest icon is still present for this brawler
        print(f"[QUEST] Checking if {brawler_name.title()} still has quest after {matches} match(es)...")
        still_has_quest = self.Lobby_automation.check_brawler_has_quest(brawler_name)

        if still_has_quest:
            print(f"[QUEST] {brawler_name.title()} quest still active. Continuing...")
            # Make sure we're back in lobby
            self._ensure_lobby()
            return

        # Quest is done! Switch to next brawler
        print(f"[QUEST] {brawler_name.title()} quest COMPLETED after {matches} match(es)!")
        self._quest_matches_played.pop(brawler_name, None)

        if len(self.brawlers_pick_data) <= 1:
            # All quests done!
            print("[QUEST] All quest brawlers completed! Bot stopping.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                screenshot = self.window_controller.screenshot()
                loop.run_until_complete(
                    asyncio.wait_for(async_notify_user("completed", screenshot), timeout=15))
            except Exception:
                pass
            finally:
                loop.close()
            if os.path.exists("latest_brawler_data.json"):
                os.remove("latest_brawler_data.json")
            self.window_controller.keys_up(list("wasd"))
            self.window_controller.close()
            sys.exit(0)

        # Pop completed brawler and move to next
        completed = self.brawlers_pick_data.pop(0)
        print(f"[QUEST] Switching from {completed['brawler'].title()} -> "
              f"{self.brawlers_pick_data[0]['brawler'].title()} "
              f"({len(self.brawlers_pick_data)} remaining)")

        # Load next brawler data
        next_data = self.brawlers_pick_data[0]
        self.Trophy_observer.change_trophies(next_data.get('trophies', 0))
        self.Trophy_observer.current_wins = 0
        self.Trophy_observer.win_streak = 0
        save_brawler_data(self.brawlers_pick_data)

        # Navigate to lobby and select next brawler
        self._ensure_lobby()
        self.Lobby_automation.select_brawler(next_data['brawler'])

    def _ensure_lobby(self):
        """Navigate back to the lobby using the back arrow (NOT Q, which starts a match)."""
        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        screenshot = self.window_controller.screenshot()
        current_state = get_state(screenshot)
        attempts = 0
        while current_state != "lobby" and attempts < 30:
            # Use back arrow instead of Q (Q maps to START MATCH button)
            self.window_controller.click(int(100 * wr), int(60 * hr))
            if debug:
                print("Pressed back arrow to return to lobby")
            time.sleep(1)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            attempts += 1

    def _read_lobby_trophies(self, screenshot):
        """Read trophy text from the lobby screen."""
        try:
            wr = self.window_controller.width_ratio
            hr = self.window_controller.height_ratio
            region = (int(140 * wr), int(10 * hr), int(380 * wr), int(60 * hr))
            crop = np.asarray(screenshot.crop(region))
            data = extract_text_and_positions(crop)
            for key in data:
                nums = ''.join(filter(str.isdigit, key))
                if nums:
                    return key
        except Exception:
            pass
        return None

    def _is_manual_trophy_locked(self):
        if not self.brawlers_pick_data:
            return False
        return bool(self.brawlers_pick_data[0].get('manual_trophies', False))

    def _reset_to_manual_trophies(self):
        if not self._is_manual_trophy_locked() or not self.brawlers_pick_data:
            return
        manual_value = self.brawlers_pick_data[0].get('trophies')
        if isinstance(manual_value, int):
            self.Trophy_observer.current_trophies = manual_value
            self.Trophy_observer._lobby_trophy_verified = True

    def start_game(self, data):
        print("state is lobby, starting game")

        # quest Farm Mode: check if current brawler's quest is done
        type_of_push = self.brawlers_pick_data[0]['type']
        if type_of_push == "quest":
            self._handle_quest_rotation()
            # After quest handling, start the game if we still have brawlers
            if not self.brawlers_pick_data:
                return
            # q btn is over the start btn
            self.window_controller.keys_up(list("wasd"))
            # Auto-detect game mode
            try:
                lobby_screenshot = self.window_controller.screenshot()
                result = detect_game_mode_from_frame(lobby_screenshot, self.window_controller)
                if result:
                    mode_name, mode_info = result
                    self.detected_game_mode = mode_name
                    self.detected_game_mode_type = mode_info['type']
                    self.is_showdown = mode_info.get('showdown', False)
                    print(f"[AUTO-DETECT] Using: {mode_name} (type={mode_info['type']})")
            except Exception as e:
                print(f"[AUTO-DETECT] Error: {e}")

            # Verify trophies before starting
            try:
                lobby_screenshot = self.window_controller.screenshot()
                trophy_text = self._read_lobby_trophies(lobby_screenshot)
                if trophy_text:
                    verified = self.validate_trophies(trophy_text)
                    if verified and isinstance(verified, int):
                        if self._is_manual_trophy_locked():
                            self._reset_to_manual_trophies()
                        else:
                            self.Trophy_observer.change_trophies(verified)
                            self.brawlers_pick_data[0]['trophies'] = verified
            except Exception:
                pass

            time.sleep(0.3)
            self.window_controller.press_key("Q")
            return

        # If end_game() already rotated to a new brawler, select it immediately
        if self._pending_brawler_switch:
            self._pending_brawler_switch = False
            next_brawler_name = self.brawlers_pick_data[0]['brawler']
            print(f"[FARM] Executing queued brawler switch → {next_brawler_name.title()}")
            self.Lobby_automation.select_brawler(next_brawler_name)
            # Start the match immediately — do NOT re-check targets (avoids double rotation)
            self.window_controller.keys_up(list("wasd"))
            try:
                lobby_screenshot = self.window_controller.screenshot()
                result = detect_game_mode_from_frame(lobby_screenshot, self.window_controller)
                if result:
                    mode_name, mode_info = result
                    self.detected_game_mode = mode_name
                    self.detected_game_mode_type = mode_info['type']
                    self.is_showdown = mode_info.get('showdown', False)
                    print(f"[AUTO-DETECT] Using: {mode_name} (type={mode_info['type']})")
            except Exception as e:
                print(f"[AUTO-DETECT] Error: {e}")
            self.window_controller.press_key("Q")
            print("[FARM] Pressed Q to start match after brawler switch")
            return

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
            # smart Trophy Farm: use intelligent rotation
            if self.smart_trophy_farm:
                has_next = self._pick_next_farm_brawler()
                if not has_next:
                    print("[FARM] All brawlers reached target! Bot stopping.")
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        screenshot = self.window_controller.screenshot()
                        loop.run_until_complete(
                            asyncio.wait_for(async_notify_user("completed", screenshot), timeout=15))
                    except Exception:
                        pass
                    finally:
                        loop.close()
                    if os.path.exists("latest_brawler_data.json"):
                        os.remove("latest_brawler_data.json")
                    self.window_controller.keys_up(list("wasd"))
                    self.window_controller.close()
                    sys.exit(0)

                # Load next brawler's data
                self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
                self.Trophy_observer.current_wins = self.brawlers_pick_data[0]['wins'] if self.brawlers_pick_data[0]['wins'] != "" else 0
                self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
                next_brawler_name = self.brawlers_pick_data[0]['brawler']
                save_brawler_data(self.brawlers_pick_data)

                # Always auto-pick in farm mode
                print(f"[FARM] Switching to {next_brawler_name.title()}")
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                max_attempts = 30
                attempts = 0
                wr = self.window_controller.width_ratio or 1.0
                hr = self.window_controller.height_ratio or 1.0
                while current_state != "lobby" and attempts < max_attempts:
                    self.window_controller.click(int(100 * wr), int(60 * hr))
                    if debug: print("Pressed back arrow to return to lobby")
                    time.sleep(1)
                    screenshot = self.window_controller.screenshot()
                    current_state = get_state(screenshot)
                    attempts += 1
                if attempts < max_attempts:
                    self.Lobby_automation.select_brawler(next_brawler_name)
                else:
                    print("[FARM] Failed to reach lobby for brawler switch")
            else:
                # original sequential rotation
                if len(self.brawlers_pick_data) <= 1:
                    print("Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                          "Bot will now pause itself until closed.", value, push_current_brawler_till)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        screenshot = self.window_controller.screenshot()
                        loop.run_until_complete(
                            asyncio.wait_for(async_notify_user("bot_is_stuck", screenshot), timeout=15))
                    except Exception:
                        pass
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
                    loop.run_until_complete(
                        asyncio.wait_for(async_notify_user(self.brawlers_pick_data[0]["brawler"], screenshot), timeout=15))
                except Exception:
                    pass
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
                    wr = self.window_controller.width_ratio or 1.0
                    hr = self.window_controller.height_ratio or 1.0
                    while current_state != "lobby" and attempts < max_attempts:
                        self.window_controller.click(int(100 * wr), int(60 * hr))
                        if debug: print("Pressed back arrow to return to lobby")
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

        # Auto-detect game mode from lobby screen before starting
        try:
            lobby_screenshot = self.window_controller.screenshot()
            result = detect_game_mode_from_frame(lobby_screenshot, self.window_controller)
            if result:
                mode_name, mode_info = result
                self.detected_game_mode = mode_name
                self.detected_game_mode_type = mode_info['type']
                self.is_showdown = mode_info.get('showdown', False)
                print(f"[LOBBY] Auto-detected game mode: {mode_name} (type={mode_info['type']}, showdown={self.is_showdown})")
            else:
                print("[LOBBY] Could not auto-detect game mode, using config default")
        except Exception as e:
            print(f"[LOBBY] Game mode detection error: {e}")

        # --- Verify trophies from lobby before starting match ---
        try:
            if lobby_screenshot is None:
                lobby_screenshot = self.window_controller.screenshot()
            wr = self.window_controller.width_ratio or 1.0
            hr = self.window_controller.height_ratio or 1.0
            self.Trophy_observer.verify_lobby_trophies(lobby_screenshot, wr=wr, hr=hr)
            # Save if corrected
            if self.Trophy_observer._corrections_log:
                if self._is_manual_trophy_locked():
                    self._reset_to_manual_trophies()
                    print("[VERIFY] Manual trophy lock active - skipped pre-match correction persistence")
                else:
                    self.brawlers_pick_data[0]['trophies'] = self.Trophy_observer.current_trophies
                    save_brawler_data(self.brawlers_pick_data)
                    print(f"[VERIFY] Pre-match trophy correction saved")
                self.Trophy_observer._corrections_log.clear()  # Reset for this match
        except Exception as e:
            print(f"[VERIFY] Pre-match trophy check error: {e}")

        self.window_controller.press_key("Q")
        print("Pressed Q to start a match")

    def click_brawl_stars(self, frame):
        screenshot = frame.crop((50, 4, 900, 31))
        if self.brawl_stars_icon is None:
            self.brawl_stars_icon = load_image("state_finder/images_to_detect/brawl_stars_icon.png",
                                               self.window_controller.scale_factor)
        detection = find_template_center(screenshot, self.brawl_stars_icon)
        if detection:
            x, y = detection
            self.window_controller.click(x=x + 50, y=y)

    def handle_mode_selection(self):
        """Handle the game mode selection screen (e.g. TROPHY GAME MODES).
        Click the back arrow to return to the main lobby."""
        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        print("[STAGE] Mode selection screen detected — pressing back to lobby")
        for attempt in range(5):
            self.window_controller.click(int(100 * wr), int(60 * hr))
            time.sleep(1.5)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            if current_state == "lobby":
                print("[STAGE] Back in lobby after mode selection")
                return
            if current_state != "mode_selection":
                print(f"[STAGE] Left mode selection -> {current_state}")
                return
        print("[STAGE] Could not leave mode selection after 5 attempts")

    def handle_idle_disconnect(self):
        """Handle the 'Idle Disconnect' dialog by clicking the RELOAD button.
        
        The RELOAD button is typically centered horizontally, in the lower
        half of the dialog (roughly 55-65% down the screen).
        """
        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        print("[STAGE] Idle disconnect detected - clicking RELOAD")

        for attempt in range(5):
            # RELOAD button is approximately at center-x, ~58% down
            btn_x = int(960 * wr)
            btn_y = int(625 * hr)
            self.window_controller.click(btn_x, btn_y)
            time.sleep(0.8)

            # Try slightly different positions for different resolutions
            btn_x2 = int(960 * wr)
            btn_y2 = int(590 * hr)
            self.window_controller.click(btn_x2, btn_y2)
            time.sleep(0.5)

            # Check if we left the idle disconnect screen
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            if current_state != "idle_disconnect":
                print(f"[STAGE] Idle disconnect resolved -> {current_state}")
                return

            # Also try pressing Q as fallback
            self.window_controller.press_key("Q")
            time.sleep(0.5)

        print("[STAGE] Idle disconnect: RELOAD click failed after 5 attempts")

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

    def end_game(self, frame=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()

        found_game_result = False
        read_match_stats = False
        verified_trophies = False
        current_state = get_state(screenshot)
        max_end_attempts = 30
        end_attempts = 0
        while current_state == "end" and end_attempts < max_end_attempts:
            if not found_game_result and time.time() - self.time_since_last_stat_change > 10:

                found_game_result = self.Trophy_observer.find_game_result(screenshot, current_brawler=self.brawlers_pick_data[0]['brawler'])

                self.time_since_last_stat_change = time.time()
                if found_game_result:
                    # Track quest matches played
                    if self.brawlers_pick_data[0].get('type') == 'quest':
                        if not hasattr(self, '_quest_matches_played'):
                            self._quest_matches_played = {}
                        bn = self.brawlers_pick_data[0]['brawler']
                        self._quest_matches_played[bn] = self._quest_matches_played.get(bn, 0) + 1
                        print(f"[QUEST] {bn.title()} match #{self._quest_matches_played[bn]} completed")

                    values = {
                        "trophies": self.Trophy_observer.current_trophies,
                        "wins": self.Trophy_observer.current_wins
                    }
                    type_to_push = self.brawlers_pick_data[0]['type']
                    if type_to_push not in values:
                        type_to_push = "trophies"
                    value = values[type_to_push]
                    self.brawlers_pick_data[0][type_to_push] = value
                    save_brawler_data(self.brawlers_pick_data)
                    push_current_brawler_till = self.brawlers_pick_data[0]['push_until']

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
                                loop.run_until_complete(
                                    asyncio.wait_for(async_notify_user("completed", screenshot), timeout=15))
                            except Exception:
                                pass
                            finally:
                                loop.close()
                            if os.path.exists("latest_brawler_data.json"):
                                os.remove("latest_brawler_data.json")
                            print("Bot stopping: all targets completed.")
                            self.window_controller.keys_up(list("wasd"))
                            self.window_controller.close()
                            sys.exit(0)
                        elif self.smart_trophy_farm:
                            # Rotate NOW so the next lobby visit just selects the right brawler
                            # instead of playing one extra match above the target.
                            print(f"[FARM] Target reached in end_game — rotating brawler immediately.")
                            has_next = self._pick_next_farm_brawler()
                            if has_next:
                                self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
                                self.Trophy_observer.current_wins = (
                                    self.brawlers_pick_data[0]['wins']
                                    if self.brawlers_pick_data[0]['wins'] != "" else 0
                                )
                                self.Trophy_observer.win_streak = self.brawlers_pick_data[0]['win_streak']
                                save_brawler_data(self.brawlers_pick_data)
                                self._pending_brawler_switch = True
                                print(f"[FARM] Queued switch to {self.brawlers_pick_data[0]['brawler'].title()} for next lobby.")
                    elif (self.smart_trophy_farm and self.dynamic_rotation_enabled
                            and len(self.brawlers_pick_data) > 1):
                        # Dynamic rotation: periodically switch to the brawler with the
                        # fewest trophies so the farm stays balanced and dynamic.
                        mc = self.Trophy_observer.match_counter
                        if mc > 0 and mc % self.dynamic_rotation_every == 0:
                            print(f"[FARM] Dynamic rotation triggered after {mc} total matches.")
                            switched = self._dynamic_rotate_to_lowest()
                            if switched:
                                self._pending_brawler_switch = True

            # Read match performance stats - retry every loop iteration until success
            if found_game_result and not read_match_stats:
                try:
                    wr = self.window_controller.width_ratio or 1.0
                    hr = self.window_controller.height_ratio or 1.0
                    self.Trophy_observer.read_end_screen_stats(
                        screenshot,
                        self.brawlers_pick_data[0]['brawler'],
                        wr=wr, hr=hr
                    )
                    read_match_stats = True
                except Exception as e:
                    print(f"[END] Error reading match stats: {e}")

            # --- AUTO-VERIFY trophies & game result from end screen ---
            if found_game_result and not verified_trophies:
                try:
                    wr = self.window_controller.width_ratio or 1.0
                    hr = self.window_controller.height_ratio or 1.0
                    current_brawler = self.brawlers_pick_data[0]['brawler']

                    # 1) Verify game result is correct (victory vs defeat)
                    self.Trophy_observer.verify_game_result_consistency(
                        screenshot, current_brawler, wr=wr, hr=hr
                    )
                    # 2) Verify trophy delta matches OCR
                    self.Trophy_observer.verify_trophy_delta(
                        screenshot, wr=wr, hr=hr
                    )

                    # If corrections were made, re-save the updated data
                    if self.Trophy_observer._corrections_log:
                        if self._is_manual_trophy_locked():
                            self._reset_to_manual_trophies()
                            print("[VERIFY] Manual trophy lock active - skipped end-screen correction persistence")
                        else:
                            values = {
                                "trophies": self.Trophy_observer.current_trophies,
                                "wins": self.Trophy_observer.current_wins
                            }
                            type_to_push = self.brawlers_pick_data[0]['type']
                            if type_to_push not in values:
                                type_to_push = "trophies"
                            self.brawlers_pick_data[0][type_to_push] = values[type_to_push]
                            save_brawler_data(self.brawlers_pick_data)
                            print(f"[VERIFY] Saved corrected data after {len(self.Trophy_observer._corrections_log)} correction(s)")

                    verified_trophies = True
                except Exception as e:
                    print(f"[VERIFY] Error during end-screen verification: {e}")
                    verified_trophies = True  # Don't retry on error

            # --- Stale frame guard: don't press Q on frozen feed ---
            _, frame_time = self.window_controller.get_latest_frame()
            if frame_time > 0 and (time.time() - frame_time) > self.window_controller.FRAME_STALE_TIMEOUT:
                print("[END] Frame is stale, waiting for fresh feed before pressing Q...")
                stale_wait_start = time.time()
                feed_recovered = False
                while (time.time() - stale_wait_start) < 15:
                    time.sleep(1)
                    _, frame_time = self.window_controller.get_latest_frame()
                    if frame_time > 0 and (time.time() - frame_time) < self.window_controller.FRAME_STALE_TIMEOUT:
                        print("[END] Feed recovered, resuming")
                        feed_recovered = True
                        break
                if not feed_recovered:
                    print("[END] Feed still stale after 15s, breaking out of end_game")
                    break
                # Re-evaluate with fresh frame
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                continue

            # --- Play Again on Win: press F instead of Q ---
            if self.play_again_on_win and found_game_result and self.Trophy_observer._last_game_result == "victory":
                self.window_controller.press_key("F")
                if debug: print("Victory - pressing F (Play Again)")
            else:
                self.window_controller.press_continue()
                if debug: print("Game has ended, pressing Q")
            time.sleep(1)  # Reduced from 3s to avoid long main-thread blocks
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            end_attempts += 1
        if end_attempts >= max_end_attempts:
            print("End game screen stuck for too long, forcing continue")
        if debug: print("Game has ended", current_state)

        # --- PLAY AGAIN ON WIN: wait for new match to start ---
        if self.play_again_on_win and self.Trophy_observer._last_game_result == "victory":
            print("[PLAY-AGAIN] Waiting for new match to start...")
            start_wait_time = time.time()
            while time.time() - start_wait_time < 25:
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                if current_state == "match":
                    print("[PLAY-AGAIN] New match started successfully!")
                    return
                time.sleep(0.5)
            print("[PLAY-AGAIN] Match did not start within 25s, pressing Q to return to lobby.")
            self.window_controller.press_continue()
            time.sleep(2)
            self.window_controller.press_continue()

        # --- LOBBY TROPHY VERIFICATION ---
        # After exiting the end screen, we should be in the lobby.
        # OCR-read the actual trophy count and compare with our internal value.
        try:
            time.sleep(1)  # Wait for lobby to fully render
            lobby_screenshot = self.window_controller.screenshot()
            lobby_state = get_state(lobby_screenshot)
            if lobby_state == "lobby":
                wr = self.window_controller.width_ratio or 1.0
                hr = self.window_controller.height_ratio or 1.0
                self.Trophy_observer.verify_lobby_trophies(
                    lobby_screenshot, wr=wr, hr=hr
                )
                # If lobby correction was made, update saved data
                if self.Trophy_observer._corrections_log:
                    latest_corrections = len(self.Trophy_observer._corrections_log)
                    if self._is_manual_trophy_locked():
                        self._reset_to_manual_trophies()
                        print("[VERIFY] Manual trophy lock active - skipped lobby correction persistence")
                    else:
                        values = {
                            "trophies": self.Trophy_observer.current_trophies,
                            "wins": self.Trophy_observer.current_wins
                        }
                        type_to_push = self.brawlers_pick_data[0]['type']
                        if type_to_push not in values:
                            type_to_push = "trophies"
                        self.brawlers_pick_data[0][type_to_push] = values[type_to_push]
                        save_brawler_data(self.brawlers_pick_data)
                        print(f"[VERIFY] Lobby verification complete - {latest_corrections} total correction(s) this match")
                    self.Trophy_observer._corrections_log.clear()
        except Exception as e:
            print(f"[VERIFY] Lobby trophy verification error: {e}")

    def quit_shop(self):
        self.window_controller.click(100*self.window_controller.width_ratio, 60*self.window_controller.height_ratio)

    def close_pop_up(self, frame=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()
        if self.close_popup_icon is None:
            self.close_popup_icon = load_image("state_finder/images_to_detect/close_popup.png", self.window_controller.scale_factor)
        popup_location = find_template_center(screenshot, self.close_popup_icon)
        if popup_location:
            self.window_controller.click(*popup_location)

    def dismiss_trophy_reward(self):
        """Dismiss the trophy milestone reward screen by clicking 'LET'S GO' button."""
        print("[STAGE] Trophy reward screen detected - clicking LET'S GO")
        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        # "LET'S GO" button center is approximately at (1720, 960) at 1920x1080
        btn_x = int(1720 * wr)
        btn_y = int(960 * hr)
        self.window_controller.click(btn_x, btn_y)
        time.sleep(0.8)
        # Click again - sometimes the first click doesn't register
        self.window_controller.click(btn_x, btn_y)
        time.sleep(0.5)
        # Try a slightly different position in case of offset
        btn_x2 = int(1700 * wr)
        btn_y2 = int(940 * hr)
        self.window_controller.click(btn_x2, btn_y2)
        time.sleep(0.3)
        # Press ESC/Q as fallback dismissal
        self.window_controller.press_continue(include_fallback_clicks=False)
        time.sleep(0.2)
        # Also try clicking center of screen as last resort
        self.window_controller.click(int(960 * wr), int(540 * hr))
        print("[STAGE] Trophy reward dismiss attempts completed")

    def do_state(self, state, data=None):
        if state not in self.states:
            print(f"[STAGE] Unknown state '{state}', pressing back arrow as fallback.")
            wr = self.window_controller.width_ratio or 1.0
            hr = self.window_controller.height_ratio or 1.0
            self.window_controller.click(int(100 * wr), int(60 * hr))
            return
        if data is not None:
            self.states[state](data)
            return
        self.states[state]()

