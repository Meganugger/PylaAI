import time

import numpy as np

from stage_manager import load_image
from state_finder.main import get_state
from utils import extract_text_and_positions, count_hsv_pixels, load_toml_as_dict, find_template_center

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"

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
        x1, y1 = int(400 * wr), int(380 * hr)
        x2, y2 = int(1500 * wr), int(700 * hr)
        if isinstance(frame, np.ndarray):
            screenshot = frame[y1:y2, x1:x2]
        else:
            screenshot = frame.crop((x1, y1, x2, y2))
        gray_pixels = count_hsv_pixels(screenshot, (0, 0, 55), (10, 15, 77))
        if debug and (gray_pixels > 1000 or time.time() - self._last_idle_debug_time >= 5.0):
            print("gray pixels (if > 1000 then bot will try to unidle) :", gray_pixels)
            self._last_idle_debug_time = time.time()
        if gray_pixels > 1000:
            self.window_controller.click(int(535 * wr), int(615 * hr))

    @staticmethod
    def _can_select_brawler_in_state(state):
        return state in {"lobby", "brawler_selection"}

    def select_brawler(self, brawler):
        brawler = str(brawler or "").strip().lower()
        if not brawler:
            return False
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
            screenshot = screenshot.resize((int(screenshot.width * 0.65), int(screenshot.height * 0.65)))
            screenshot = np.array(screenshot)
            if debug: print("extracting text on current screen...")
            results = extract_text_and_positions(screenshot)
            reworked_results = {}
            for key in results.keys():
                orig_key = key
                for symbol in [' ', '-', '.', "&"]:
                    key = key.replace(symbol, "")
                
                key = self.resolve_ocr_typos(key)
                reworked_results[key] = results[orig_key]
            if debug:
                print("All detected text while looking for brawler name:", reworked_results.keys())
                print()
            if brawler in reworked_results.keys():
                if debug: print("Found brawler ", brawler)
                x, y = reworked_results[brawler]['center']
                self.window_controller.click(int(x * 1.5385), int(y * 1.5385))
                time.sleep(1)
                select_x, select_y = self.coords_cfg['lobby']['select_btn'][0], self.coords_cfg['lobby']['select_btn'][1]
                self.window_controller.click(select_x, select_y, already_include_ratio=False)
                time.sleep(0.5)
                if debug: print("Selected brawler ", brawler)
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
