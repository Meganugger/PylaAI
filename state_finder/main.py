import os
import sys
from functools import lru_cache
import time

import cv2
import numpy as np

sys.path.append(os.path.abspath('../'))
from utils import load_toml_as_dict, to_bgr_array, record_timing

orig_screen_width, orig_screen_height = 1920, 1080

path = r"./state_finder/images_to_detect/"
end_results_path = os.path.join(path, "end_results")
images_with_star_drop = []
end_result_names = ("victory", "defeat", "draw")

for file in os.listdir("./state_finder/images_to_detect"):
    if "star_drop" in file:
        images_with_star_drop.append(file)

region_data = load_toml_as_dict("./cfg/lobby_config.toml")['template_matching']
region_data.setdefault("reward_claim_corner", [0, 0, 190, 120])
debug = load_toml_as_dict("./cfg/general_config.toml").get("super_debug", "no") == "yes"
_last_state_debug_value = None
_last_state_debug_time = 0.0
crop_region = load_toml_as_dict("./cfg/lobby_config.toml")['lobby']['trophy_observer']


def is_template_in_region(image, template_path, region):
    current_height, current_width = image.shape[:2]
    orig_x, orig_y, orig_width, orig_height = region
    width_ratio = current_width / orig_screen_width
    height_ratio = current_height / orig_screen_height

    new_x = int(orig_x * width_ratio)
    new_y = int(orig_y * height_ratio)
    new_width = int(orig_width * width_ratio)
    new_height = int(orig_height * height_ratio)
    cropped_image = image[new_y:new_y + new_height, new_x:new_x + new_width]
    loaded_template = load_template(template_path, current_width, current_height)
    result = cv2.matchTemplate(cropped_image, loaded_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val > 0.7


@lru_cache(maxsize=256)
def load_template(image_path, width, height):
    current_width_ratio = width / orig_screen_width
    current_height_ratio = height / orig_screen_height
    image = cv2.imread(image_path)
    orig_height, orig_width = image.shape[:2]
    resized_image = cv2.resize(
        image,
        (int(orig_width * current_width_ratio), int(orig_height * current_height_ratio))
    )
    return resized_image


def find_game_result(screenshot):
    screenshot_bgr = to_bgr_array(screenshot)

    for result_name in end_result_names:
        template_path = os.path.join(end_results_path, f"{result_name}.png")
        if is_template_in_region(screenshot_bgr, template_path, crop_region):
            return result_name

    return False


def get_in_game_state(image):
    game_result = find_game_result(image)
    if game_result:
        return f"end_{game_result}"
    if is_in_shop(image):
        return "shop"
    if is_in_offer_popup(image):
        return "popup"
    if is_in_reward_claim(image):
        return "reward_claim"
    if is_in_lobby(image):
        return "lobby"
    if is_in_brawler_selection(image):
        return "brawler_selection"
    if is_in_brawl_pass(image) or is_in_star_road(image):
        return "shop"
    if is_in_star_drop(image):
        return "star_drop"
    return "match"


def is_in_shop(image) -> bool:
    return is_template_in_region(image, path + 'powerpoint.png', region_data["powerpoint"])


def is_in_brawler_selection(image) -> bool:
    return is_template_in_region(image, path + 'brawler_menu_task.png', region_data["brawler_menu_task"])


def is_in_offer_popup(image) -> bool:
    return is_template_in_region(image, path + 'close_popup.png', region_data["close_popup"])


def is_in_reward_claim(image) -> bool:
    return is_template_in_region(
        image,
        path + 'end_battle_top_left_continue_corner.png',
        region_data["reward_claim_corner"]
    )


def is_in_lobby(image) -> bool:
    return is_template_in_region(image, path + 'lobby_menu.png', region_data["lobby_menu"])


def is_in_end_of_a_match(image):
    return bool(find_game_result(image))


def is_in_brawl_pass(image):
    return is_template_in_region(image, path + 'brawl_pass_house.PNG', region_data['brawl_pass_house'])


def is_in_star_road(image):
    return is_template_in_region(image, path + "go_back_arrow.png", region_data['go_back_arrow'])


def is_in_star_drop(image):
    for image_filename in images_with_star_drop:
        if is_template_in_region(image, path + image_filename, region_data['star_drop']):
            return True
    return False


def get_state(screenshot):
    global _last_state_debug_value, _last_state_debug_time
    started_at = time.perf_counter()
    screenshot_bgr = to_bgr_array(screenshot)
    state = get_in_game_state(screenshot_bgr)
    record_timing("state_detection", time.perf_counter() - started_at, print_every=60)
    now = time.time()
    if debug and (state != _last_state_debug_value or now - _last_state_debug_time >= 3.0):
        print(f"State: {state}")
        _last_state_debug_value = state
        _last_state_debug_time = now
    return state
