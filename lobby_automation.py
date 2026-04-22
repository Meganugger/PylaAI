from difflib import SequenceMatcher
import time

import cv2
import numpy as np

from stage_manager import load_image
from state_finder.main import get_state
from utils import extract_text_and_positions, count_hsv_pixels, load_toml_as_dict, find_template_center

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"
GRAY_PIXELS_THRESHOLD = int(load_toml_as_dict("./cfg/bot_config.toml").get("idle_pixels_minimum", 1000))

OCR_BRAWLER_ALIASES = {
    'shey': 'shelly',
    'shlly': 'shelly',
    'larryslawrie': 'larrylawrie',
    '[eon': 'leon',
}

class LobbyAutomation:

    def __init__(self, window_controller):
        self.coords_cfg = load_toml_as_dict("./cfg/lobby_config.toml")
        self.window_controller = window_controller
        self.brawler_menu_template = None
        self._last_idle_debug_time = 0.0

    def check_for_idle(self, frame):
        wr = self.window_controller.width_ratio
        hr = self.window_controller.height_ratio
        # Tight ROI over the reconnect dialog body to avoid gameplay pixels.
        x1, y1 = int(700 * wr), int(470 * hr)
        x2, y2 = int(1220 * wr), int(620 * hr)
        if isinstance(frame, np.ndarray):
            screenshot = frame[y1:y2, x1:x2]
        else:
            screenshot = frame.crop((x1, y1, x2, y2))
        gray_pixels = count_hsv_pixels(screenshot, (0, 0, 18), (10, 20, 100))
        if debug and (gray_pixels > GRAY_PIXELS_THRESHOLD or time.time() - self._last_idle_debug_time >= 5.0):
            print(f"gray pixels (if > {GRAY_PIXELS_THRESHOLD} then bot will try to unidle) :", gray_pixels)
            self._last_idle_debug_time = time.time()
        if gray_pixels > GRAY_PIXELS_THRESHOLD:
            self.window_controller.click(int(535 * wr), int(615 * hr))

    @staticmethod
    def _can_select_brawler_in_state(state):
        return state in {"lobby", "brawler_selection"}

    def select_brawler(self, brawler):
        brawler = str(brawler or "").strip().lower()
        if not brawler:
            return False
        general_config = load_toml_as_dict("cfg/general_config.toml")
        debug_enabled = str(general_config.get("super_debug", "no")).lower() in ("yes", "true", "1")
        try:
            ocr_scale = float(general_config.get("ocr_scale_down_factor", 0.65))
        except (TypeError, ValueError):
            ocr_scale = 0.65
        ocr_scale = max(0.35, min(1.0, ocr_scale))
        target_key = self.normalize_ocr_name(brawler)
        brawler_menu_treshold = 0.8
        found = False
        current_frame = self.window_controller.screenshot()
        current_state = get_state(current_frame)
        if not self._can_select_brawler_in_state(current_state):
            print(
                f"WARNING: Skipping brawler selection for '{brawler}' because "
                f"the current state is '{current_state}'."
            )
            return False
        if self.brawler_menu_template is None:
            self.brawler_menu_template = load_image(
                r'state_finder/images_to_detect/brawler_menu_btn.png',
                self.window_controller.scale_factor
            )
        while not found:
            brawler_menu_btn_coords = find_template_center(
                current_frame,
                self.brawler_menu_template,
                brawler_menu_treshold
            )
            if brawler_menu_btn_coords:
                found = True
            else:
                if debug: print("Brawler menu button not found, retrying...")
                brawler_menu_treshold -= 0.1
                time.sleep(1)
                current_frame = self.window_controller.screenshot()
                current_state = get_state(current_frame)
                if not self._can_select_brawler_in_state(current_state):
                    print(
                        f"WARNING: Aborting brawler selection for '{brawler}' because "
                        f"the state changed to '{current_state}'."
                    )
                    return False
            if not found and brawler_menu_treshold < 0.5:
                current_frame.save(r'brawler_menu_btn_not_found.png')
                raise ValueError("Brawler menu button not found on screen, even at low threshold.")
        x, y = brawler_menu_btn_coords
        self.window_controller.click(x, y)
        c = 0
        found_brawler = False
        for i in range(50):
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            if not self._can_select_brawler_in_state(current_state):
                print(
                    f"WARNING: Aborting brawler selection for '{brawler}' because "
                    f"the state changed to '{current_state}'."
                )
                return False
            screenshot_full = np.array(screenshot)
            screenshot = cv2.resize(
                screenshot_full,
                (int(screenshot_full.shape[1] * ocr_scale), int(screenshot_full.shape[0] * ocr_scale)),
                interpolation=cv2.INTER_AREA,
            )
            if debug_enabled: print("extracting text on current screen...")
            results = extract_text_and_positions(screenshot)
            reworked_results = {}
            for key in results.keys():
                orig_key = key
                key = self.normalize_ocr_name(key)
                key = self.resolve_ocr_typos(key)
                reworked_results[key] = results[orig_key]
            if debug_enabled:
                print("All detected text while looking for brawler name:", reworked_results.keys())
                print()
            matches = []
            for detected_name, text_box in reworked_results.items():
                if self.names_match(detected_name, target_key):
                    score = self.name_match_score(detected_name, target_key)
                    matches.append((score, detected_name, text_box))
            if matches:
                matches.sort(key=lambda item: item[0], reverse=True)
                _, detected_name, text_box = matches[0]
                x, y = text_box['center']
                click_x = int(x / ocr_scale)
                click_y = int((y / ocr_scale) - (95 * self.window_controller.height_ratio))
                click_y = max(0, min(screenshot_full.shape[0] - 1, click_y))
                if debug_enabled:
                    print(f"Found brawler {brawler} (OCR: {detected_name})")
                self.window_controller.click(click_x, click_y)
                time.sleep(1)
                select_x, select_y = self.coords_cfg['lobby']['select_btn'][0], self.coords_cfg['lobby']['select_btn'][1]
                self.window_controller.click(select_x, select_y, already_include_ratio=False)
                time.sleep(0.5)
                if debug_enabled: print("Selected brawler ", brawler)
                found_brawler = True
                break
            if c == 0:
                wr = self.window_controller.width_ratio
                hr = self.window_controller.height_ratio
                self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(850 * hr), duration=0.8)
                c += 1
                continue
            wr = self.window_controller.width_ratio
            hr = self.window_controller.height_ratio
            self.window_controller.swipe(int(1700 * wr), int(900 * hr), int(1700 * wr), int(650 * hr), duration=0.8)
            time.sleep(1)
        if not found_brawler:
            print(f"WARNING: Brawler '{brawler}' was not found after 50 scroll attempts. "
                  f"The bot will continue with the currently selected brawler.")
            return False
        return True

    @staticmethod
    def resolve_ocr_typos(potential_brawler_name: str) -> str:
        """
        Matches well known 'typos' from OCR to the correct brawler's name
        or returns the original string
        """
        normalized_name = str(potential_brawler_name or "").lower()
        return OCR_BRAWLER_ALIASES.get(normalized_name, normalized_name)

    @staticmethod
    def normalize_ocr_name(value: str) -> str:
        normalized = str(value).lower()
        for symbol in [' ', '-', '.', "&", "'", "`", "_"]:
            normalized = normalized.replace(symbol, "")
        return normalized

    @staticmethod
    def bounded_edit_distance(left: str, right: str, limit: int) -> int:
        if abs(len(left) - len(right)) > limit:
            return limit + 1
        previous = list(range(len(right) + 1))
        for i, left_char in enumerate(left, 1):
            current = [i]
            best = current[0]
            for j, right_char in enumerate(right, 1):
                cost = 0 if left_char == right_char else 1
                value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
                current.append(value)
                best = min(best, value)
            if best > limit:
                return limit + 1
            previous = current
        return previous[-1]

    @classmethod
    def names_match(cls, detected_name: str, target_name: str) -> bool:
        if detected_name == target_name:
            return True
        if len(target_name) >= 4 and (target_name in detected_name or detected_name in target_name):
            return True
        limit = 1 if len(target_name) <= 5 else 2
        if cls.bounded_edit_distance(detected_name, target_name, limit) <= limit:
            return True
        return SequenceMatcher(None, detected_name, target_name).ratio() >= 0.84

    @classmethod
    def name_match_score(cls, detected_name: str, target_name: str) -> float:
        if detected_name == target_name:
            return 2.0
        ratio = SequenceMatcher(None, detected_name, target_name).ratio()
        distance = cls.bounded_edit_distance(detected_name, target_name, 3)
        return ratio - (distance * 0.05)
