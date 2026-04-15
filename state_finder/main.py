import os
import sys
from functools import lru_cache
import time

import cv2
import numpy as np

sys.path.append(os.path.abspath('../'))
from utils import load_toml_as_dict, to_bgr_array, record_timing, reader

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
region_data.setdefault("reward_claim_button", [620, 860, 680, 180])
region_data.setdefault("reward_claim_title", [420, 120, 1080, 220])
debug = load_toml_as_dict("./cfg/general_config.toml").get("super_debug", "no") == "yes"
_last_state_debug_value = None
_last_state_debug_time = 0.0
_reward_claim_cache = {
    "checked_at": 0.0,
    "shape": None,
    "detected": False,
    "button_center": None,
}
crop_region = region_data.get("end_result")
if not crop_region:
    crop_region = load_toml_as_dict("./cfg/lobby_config.toml")['lobby']['trophy_observer']


def template_score_in_region(image, template_path, region):
    current_height, current_width = image.shape[:2]
    orig_x, orig_y, orig_width, orig_height = region
    width_ratio = current_width / orig_screen_width
    height_ratio = current_height / orig_screen_height

    new_x = int(orig_x * width_ratio)
    new_y = int(orig_y * height_ratio)
    new_width = int(orig_width * width_ratio)
    new_height = int(orig_height * height_ratio)
    cropped_image = image[new_y:new_y + new_height, new_x:new_x + new_width]
    if cropped_image.size == 0:
        return 0.0
    loaded_template = load_template(template_path, current_width, current_height)
    result = cv2.matchTemplate(cropped_image, loaded_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def is_template_in_region(image, template_path, region, threshold=0.7):
    return template_score_in_region(image, template_path, region) > threshold


@lru_cache(maxsize=256)
def load_template(image_path, width, height):
    current_width_ratio = width / orig_screen_width
    current_height_ratio = height / orig_screen_height
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Template image could not be loaded: {image_path}")
    orig_height, orig_width = image.shape[:2]
    new_width = max(1, int(orig_width * current_width_ratio))
    new_height = max(1, int(orig_height * current_height_ratio))
    resized_image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized_image


def _crop_region(image, region):
    current_height, current_width = image.shape[:2]
    orig_x, orig_y, orig_width, orig_height = region
    width_ratio = current_width / orig_screen_width
    height_ratio = current_height / orig_screen_height

    x1 = max(0, int(orig_x * width_ratio))
    y1 = max(0, int(orig_y * height_ratio))
    x2 = min(current_width, int((orig_x + orig_width) * width_ratio))
    y2 = min(current_height, int((orig_y + orig_height) * height_ratio))
    cropped = image[y1:y2, x1:x2]
    return cropped, (x1, y1, x2, y2)


def _region_center(image, region):
    _, (x1, y1, x2, y2) = _crop_region(image, region)
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def _normalize_reward_text(text):
    normalized = str(text or "").strip().lower()
    replacements = {
        "0": "o",
        "1": "l",
        "5": "s",
        "6": "g",
        "8": "b",
        "!": "l",
        "|": "l",
        "$": "s",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return "".join(ch for ch in normalized if ch.isalnum())


def _ocr_text_tokens(image, region, allowlist=None):
    cropped, _ = _crop_region(image, region)
    if cropped.size == 0:
        return []

    variants = [cropped]
    try:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        enlarged = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        _, otsu = cv2.threshold(enlarged, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.extend([gray, enlarged, otsu])
    except Exception:
        pass

    tokens = []
    seen = set()
    for variant in variants:
        try:
            result = reader.readtext(variant, allowlist=allowlist)
        except TypeError:
            result = reader.readtext(variant)
        except Exception:
            continue
        for _bbox, text, _prob in result:
            normalized = _normalize_reward_text(text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                tokens.append(normalized)
    return tokens


def _is_mastery_reward_screen(image):
    now = time.time()
    shape = tuple(image.shape[:2])
    if _reward_claim_cache["shape"] == shape and (now - _reward_claim_cache["checked_at"]) < 0.6:
        return _reward_claim_cache["detected"], _reward_claim_cache["button_center"]

    button_tokens = _ocr_text_tokens(
        image,
        region_data["reward_claim_button"],
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' ",
    )
    title_tokens = _ocr_text_tokens(
        image,
        region_data["reward_claim_title"],
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' ",
    )

    has_lets_go = any(
        "letsgo" in token or ("let" in token and token.endswith("go"))
        for token in button_tokens
    )
    has_mastery_title = any(
        "master" in token or "reward" in token
        for token in title_tokens
    )
    detected = has_lets_go or (has_mastery_title and any(token in {"go", "letsgo"} for token in button_tokens))
    button_center = _region_center(image, region_data["reward_claim_button"]) if detected else None

    _reward_claim_cache["checked_at"] = now
    _reward_claim_cache["shape"] = shape
    _reward_claim_cache["detected"] = detected
    _reward_claim_cache["button_center"] = button_center
    return detected, button_center


def find_reward_claim_action(screenshot):
    screenshot_bgr = to_bgr_array(screenshot)
    _detected, button_center = _is_mastery_reward_screen(screenshot_bgr)
    return button_center


def find_game_result(screenshot):
    screenshot_bgr = to_bgr_array(screenshot)
    scores = {}
    for result_name in end_result_names:
        template_path = os.path.join(end_results_path, f"{result_name}.png")
        scores[result_name] = template_score_in_region(screenshot_bgr, template_path, crop_region)

    best_result = max(scores, key=scores.get)
    best_score = scores[best_result]
    sorted_scores = sorted(scores.values(), reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = best_score - second_best

    if best_result == "draw":
        if best_score >= 0.78 and margin >= 0.05:
            return "draw"
        return False

    if best_score >= 0.70 and margin >= 0.02:
        return best_result

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
    if is_template_in_region(
        image,
        path + 'end_battle_top_left_continue_corner.png',
        region_data["reward_claim_corner"]
    ):
        return True
    detected, _button_center = _is_mastery_reward_screen(image)
    return detected


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
