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
showdown_place_templates = {
    "1st": ("sd1st.png",),
    "2nd": ("sd2nd.png",),
    "3rd": ("sd3rd.png", "sd3rd_alt.png"),
    "4th": ("sd4th.png",),
}
SHOWDOWN_PLACE_THRESHOLD = 0.85
SHOWDOWN_PLACE_MARGIN = 0.02

for file in os.listdir("./state_finder/images_to_detect"):
    if "star_drop" in file:
        images_with_star_drop.append(file)

region_data = load_toml_as_dict("./cfg/lobby_config.toml")['template_matching']
region_data.setdefault("reward_claim_corner", [0, 0, 190, 120])
region_data.setdefault("reward_claim_button", [620, 860, 680, 180])
region_data.setdefault("reward_claim_title", [420, 120, 1080, 220])
region_data.setdefault("trophies_screen", [1545, 915, 365, 168])
debug = load_toml_as_dict("./cfg/general_config.toml").get("super_debug", "no") == "yes"
_last_state_debug_value = None
_last_state_debug_time = 0.0
_reward_claim_cache = {
    "checked_at": 0.0,
    "signature": None,
    "detected": False,
    "button_center": None,
}
_player_title_reward_cache = {
    "checked_at": 0.0,
    "detected": False,
}
_selected_gamemode_cache = {
    "checked_at": 0.0,
    "value": "",
}
crop_region = region_data.get("end_result")
if not crop_region:
    crop_region = load_toml_as_dict("./cfg/lobby_config.toml")['lobby']['trophy_observer']


def _selected_gamemode():
    now = time.time()
    if now - _selected_gamemode_cache["checked_at"] < 1.0:
        return _selected_gamemode_cache["value"]
    try:
        value = str(load_toml_as_dict("./cfg/bot_config.toml").get("gamemode", "") or "").strip().lower()
    except Exception:
        value = ""
    _selected_gamemode_cache["checked_at"] = now
    _selected_gamemode_cache["value"] = value
    return value


def _is_showdown_selected():
    return "showdown" in _selected_gamemode()


@lru_cache(maxsize=1)
def _available_showdown_place_templates():
    available = {}
    for place_name, template_files in showdown_place_templates.items():
        resolved_paths = []
        for template_file in template_files:
            template_path = os.path.join(end_results_path, template_file)
            if os.path.exists(template_path):
                resolved_paths.append(template_path)
        if resolved_paths:
            available[place_name] = tuple(resolved_paths)
    return available


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


def _reward_claim_signature(image):
    button_crop, _ = _crop_region(image, region_data["reward_claim_button"])
    title_crop, _ = _crop_region(image, region_data["reward_claim_title"])
    if button_crop.size == 0 or title_crop.size == 0:
        return None

    button_preview = cv2.resize(button_crop, (8, 4), interpolation=cv2.INTER_AREA)
    title_preview = cv2.resize(title_crop, (8, 4), interpolation=cv2.INTER_AREA)
    button_stats = tuple(int(value) for value in button_preview.mean(axis=(0, 1)))
    title_stats = tuple(int(value) for value in title_preview.mean(axis=(0, 1)))
    return (
        tuple(image.shape[:2]),
        button_stats,
        title_stats,
        int(float(button_preview.std())),
        int(float(title_preview.std())),
    )


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


def _is_mastery_reward_screen(image, allow_ocr=False):
    now = time.time()
    signature = _reward_claim_signature(image)
    if (
        signature is not None
        and _reward_claim_cache["signature"] == signature
        and (now - _reward_claim_cache["checked_at"]) < 0.6
    ):
        return _reward_claim_cache["detected"], _reward_claim_cache["button_center"]

    if not allow_ocr:
        _reward_claim_cache["checked_at"] = now
        _reward_claim_cache["signature"] = signature
        _reward_claim_cache["detected"] = False
        _reward_claim_cache["button_center"] = None
        return False, None

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
    _reward_claim_cache["signature"] = signature
    _reward_claim_cache["detected"] = detected
    _reward_claim_cache["button_center"] = button_center
    return detected, button_center


def find_reward_claim_action(screenshot):
    screenshot_bgr = to_bgr_array(screenshot)
    _detected, button_center = _is_mastery_reward_screen(screenshot_bgr, allow_ocr=True)
    return button_center


def get_reward_claim_button_center(screenshot):
    screenshot_bgr = to_bgr_array(screenshot)
    return _region_center(screenshot_bgr, region_data["reward_claim_button"])


def find_game_result(screenshot):
    screenshot_bgr = to_bgr_array(screenshot)
    if _is_showdown_selected():
        available_showdown_templates = _available_showdown_place_templates()
        showdown_scores = {}
        for place_name, template_paths in available_showdown_templates.items():
            best_score = 0.0
            for template_path in template_paths:
                best_score = max(best_score, template_score_in_region(screenshot_bgr, template_path, crop_region))
            showdown_scores[place_name] = best_score

        if showdown_scores:
            best_place = max(showdown_scores, key=showdown_scores.get)
            best_place_score = showdown_scores[best_place]
            sorted_place_scores = sorted(showdown_scores.values(), reverse=True)
            second_best_place = sorted_place_scores[1] if len(sorted_place_scores) > 1 else 0.0
            if best_place_score >= SHOWDOWN_PLACE_THRESHOLD and (best_place_score - second_best_place) >= SHOWDOWN_PLACE_MARGIN:
                return best_place

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


def get_in_game_state(image, allow_reward_ocr=False):
    game_result = find_game_result(image)
    if game_result:
        return f"end_{game_result}"
    if is_in_shop(image):
        return "shop"
    if is_in_offer_popup(image):
        return "popup"
    if is_in_brawler_selection(image):
        return "brawler_selection"
    if is_in_brawl_pass(image) or is_in_star_road(image):
        return "shop"
    # Star drops are dismissed by the normal post-match continue flow. Surfacing
    # them as a runtime state caused false positives during matches and lobby
    # transitions.
    if allow_reward_ocr and is_in_trophy_reward(image):
        return "trophy_reward"
    if is_in_player_title_reward(image, allow_ocr=allow_reward_ocr):
        return "player_title_reward"
    if is_in_reward_claim(image, allow_ocr=allow_reward_ocr):
        return "reward_claim"
    if is_in_lobby(image):
        return "lobby"
    return "match"


def is_in_shop(image) -> bool:
    return is_template_in_region(image, path + 'powerpoint.png', region_data["powerpoint"])


def is_in_brawler_selection(image) -> bool:
    return is_template_in_region(image, path + 'brawler_menu_task.png', region_data["brawler_menu_task"])


def is_in_offer_popup(image) -> bool:
    return is_template_in_region(image, path + 'close_popup.png', region_data["close_popup"])


def is_in_reward_claim(image, allow_ocr=False) -> bool:
    if is_template_in_region(
        image,
        path + 'end_battle_top_left_continue_corner.png',
        region_data["reward_claim_corner"]
    ):
        return True
    detected, _button_center = _is_mastery_reward_screen(image, allow_ocr=allow_ocr)
    return detected


def is_in_trophy_reward(image) -> bool:
    return is_template_in_region(image, path + 'trophies_screen.png', region_data["trophies_screen"])


def is_in_player_title_reward(image, allow_ocr=False) -> bool:
    if not allow_ocr:
        return False
    now = time.time()
    if now - _player_title_reward_cache["checked_at"] < 1.0:
        return _player_title_reward_cache["detected"]

    _player_title_reward_cache["checked_at"] = now
    screenshot_bgr = to_bgr_array(image)
    height, width = screenshot_bgr.shape[:2]
    top_half = screenshot_bgr[0:int(height * 0.62), int(width * 0.16):int(width * 0.84)]
    if top_half.size == 0:
        _player_title_reward_cache["detected"] = False
        return False

    try:
        hsv = cv2.cvtColor(top_half, cv2.COLOR_BGR2HSV)
    except Exception:
        _player_title_reward_cache["detected"] = False
        return False

    blue_ratio = cv2.inRange(hsv, (90, 65, 70), (125, 255, 255)).mean() / 255.0
    if blue_ratio < 0.22:
        _player_title_reward_cache["detected"] = False
        return False

    try:
        text = " ".join(result[1] for result in reader.readtext(top_half)).lower()
    except Exception as exc:
        if debug:
            print(f"Could not OCR player title reward screen: {exc}")
        _player_title_reward_cache["detected"] = False
        return False

    normalized = "".join(ch for ch in text if ch.isalnum())
    detected = (
        "newplayertitle" in normalized
        or ("playertitle" in normalized and "battlecard" in normalized)
    )
    _player_title_reward_cache["detected"] = detected
    return detected


def is_in_lobby(image) -> bool:
    return (
        is_template_in_region(image, path + 'lobby_menu.png', region_data["lobby_menu"])
        or is_lobby_play_button_visible(image)
    )


def is_lobby_play_button_visible(image) -> bool:
    current_height, current_width = image.shape[:2]
    width_ratio = current_width / orig_screen_width
    height_ratio = current_height / orig_screen_height

    region = [1260, 820, 610, 225]
    x = int(region[0] * width_ratio)
    y = int(region[1] * height_ratio)
    w = int(region[2] * width_ratio)
    h = int(region[3] * height_ratio)
    crop = image[y:y + h, x:x + w]
    if crop.size == 0:
        return False

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv,
        np.array((15, 90, 120), dtype=np.uint8),
        np.array((42, 255, 255), dtype=np.uint8),
    )
    yellow_pixels = cv2.countNonZero(yellow_mask)
    yellow_ratio = yellow_pixels / max(1, crop.shape[0] * crop.shape[1])
    if yellow_ratio < 0.28:
        return False

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    bx, by, bw, bh = cv2.boundingRect(largest)
    return bw > w * 0.45 and bh > h * 0.35


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


def get_state(screenshot, allow_reward_ocr=False):
    global _last_state_debug_value, _last_state_debug_time
    started_at = time.perf_counter()
    screenshot_bgr = to_bgr_array(screenshot)
    state = get_in_game_state(screenshot_bgr, allow_reward_ocr=allow_reward_ocr)
    record_timing("state_detection", time.perf_counter() - started_at, print_every=60)
    now = time.time()
    if debug and (state != _last_state_debug_value or now - _last_state_debug_time >= 3.0):
        print(f"State: {state}")
        _last_state_debug_value = state
        _last_state_debug_time = now
    return state
