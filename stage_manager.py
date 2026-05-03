import os.path
import sys

import time
import threading

import cv2
import numpy as np

from state_finder.main import (
    get_state,
    find_game_result,
    find_reward_claim_action,
    get_reward_claim_button_center,
    get_star_drop_type,
    is_lobby_play_button_visible,
)
from trophy_observer import TrophyObserver
from utils import find_template_center, extract_text_and_positions, load_toml_as_dict, notify_user, has_notification_webhook, \
    save_brawler_data, reader, to_bgr_array, load_brawl_stars_api_config, fetch_brawl_stars_player, \
    normalize_brawler_name

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
    "brawlball":   {"type": 5, "walls": True,  "showdown": False, "objective": (960, 540)},
    "brawl ball":  {"type": 5, "walls": True,  "showdown": False, "objective": (960, 540)},
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

debug = load_toml_as_dict("cfg/general_config.toml")['super_debug'] == "yes"


def load_image(image_path, scale_factor):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Template image could not be loaded: {image_path}")
    orig_height, orig_width = image.shape[:2]

    try:
        scale = float(scale_factor)
    except (TypeError, ValueError):
        scale = 1.0
    if scale <= 0:
        scale = 1.0

    # Calculate the new dimensions based on the scale factor
    new_width = max(1, int(orig_width * scale))
    new_height = max(1, int(orig_height * scale))

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return resized_image

class StageManager:

    def __init__(self, brawlers_data, lobby_automator, window_controller):
        self.states = {
            'shop': self.quit_shop,
            'brawler_selection': self.quit_shop,
            'popup': self.close_pop_up,
            'match': lambda: 0,
            'end': self.end_game,
            'end_victory': self.end_game,
            'end_defeat': self.end_game,
            'end_draw': self.end_game,
            'end_1st': self.end_game,
            'end_2nd': self.end_game,
            'end_3rd': self.end_game,
            'end_4th': self.end_game,
            'lobby': self.start_game,
            'star_drop': self.click_star_drop,
            'reward_claim': self.claim_reward,
            'trophy_reward': self.dismiss_trophy_reward,
            'player_title_reward': self.handle_player_title_reward,
            'prestige_reward': self.handle_prestige_reward,
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
        configured_mode = str(_bot_cfg.get("gamemode", "knockout") or "knockout").strip().lower()
        configured_mode_info = GAMEMODE_MAP.get(configured_mode, {})
        self.detected_game_mode = configured_mode
        self.detected_game_mode_type = int(_bot_cfg.get("gamemode_type", configured_mode_info.get("type", 3)))
        self.is_showdown = bool(configured_mode_info.get("showdown", "showdown" in configured_mode))

        # Smart Trophy Farm settings
        self.smart_trophy_farm = _bot_cfg.get("smart_trophy_farm", "no") == "yes"
        self.trophy_farm_target = int(_bot_cfg.get("trophy_farm_target", 500))
        self.trophy_farm_strategy = _bot_cfg.get("trophy_farm_strategy", "lowest_first")
        # Dynamic rotation: switch to lowest brawler every N matches
        self.dynamic_rotation_enabled = _bot_cfg.get("dynamic_rotation_enabled", "no") == "yes"
        self.dynamic_rotation_every = int(_bot_cfg.get("dynamic_rotation_every", 20))
        # Flag set by end_game when a brawler switch is queued for the next lobby visit
        self._pending_brawler_switch = False
        self._pending_farm_completed_restore = None
        self._pending_dynamic_rotation_restore = None
        self._awaiting_lobby_result_sync = False
        self._result_applied_for_active_match = False
        self._match_in_progress = False
        self._lobby_sync_started_at = 0.0
        self._pending_verified_result = None
        self._api_lobby_sync_attempts = 0
        self._last_api_lobby_sync_attempt_at = 0.0
        self._lobby_ocr_warmup_started = False
        self._pending_webhook_milestone_summary = None
        self._lobby_visible_since = 0.0
        self._last_start_press_at = 0.0
        self._last_result_applied_at = 0.0
        self._start_press_attempts = 0
        self._start_wait_logged_at = 0.0
        self._lobby_start_settle_delay = 0.45
        self._lobby_start_retry_delay = 1.5
        self._lobby_start_blocked_until = 0.0
        self._lobby_start_block_reason = ""
        self._post_result_lobby_delay = 1.5
        self._unexpected_brawler_selection_delay = 2.5
        self._last_brawler_menu_recovery_at = 0.0
        self._end_transition_started_at = 0.0
        self._end_transition_last_action_at = 0.0
        self._end_transition_last_result = None
        self._end_transition_hold_match_until = 0.0
        self._end_transition_action_interval = 0.75
        self._end_transition_hold_seconds = float(self.bot_config.get("post_match_dismiss_hold_seconds", 10.0))
        self._end_transition_timeout = 12.0
        self._lobby_sync_max_timeout = 8.0  # hard max for lobby result sync
        self._consecutive_lobby_start_fails = 0
        self.push_all_needs_selection = False
        self._last_push_all_refresh_at = 0.0
        self._push_all_refresh_interval = float(self.bot_config.get("push_all_api_refresh_interval", 8.0))

    def _is_easyocr_ready(self):
        try:
            return bool(reader.is_ready())
        except Exception:
            return bool(getattr(reader, "reader", None))

    def _ensure_lobby_ocr_warmup(self, delay_seconds=0.0):
        if self._lobby_ocr_warmup_started or self._is_easyocr_ready():
            return
        self._lobby_ocr_warmup_started = True

        def warm():
            try:
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                if self._match_in_progress or not self._awaiting_lobby_result_sync:
                    self._lobby_ocr_warmup_started = False
                    return
                if self._last_start_press_at and (time.time() - self._last_start_press_at) < 1.0:
                    self._lobby_ocr_warmup_started = False
                    return
                warmed = reader.warm_up()
                if not warmed and not self._is_easyocr_ready():
                    self._lobby_ocr_warmup_started = False
            except Exception:
                self._lobby_ocr_warmup_started = False

        threading.Thread(
            target=warm,
            name="easyocr-lobby-sync-warmup",
            daemon=True,
        ).start()

    def _note_lobby_visible(self, now=None):
        if now is None:
            now = time.time()
        if not self._lobby_visible_since:
            self._lobby_visible_since = now

    def _reset_lobby_start_tracking(self, reset_last_press=True):
        self._lobby_visible_since = 0.0
        if reset_last_press:
            self._last_start_press_at = 0.0
        self._start_press_attempts = 0
        self._start_wait_logged_at = 0.0

    def _restart_lobby_settle_window(self, now=None):
        if now is None:
            now = time.time()
        self._lobby_visible_since = now
        self._last_start_press_at = 0.0
        self._start_press_attempts = 0
        self._start_wait_logged_at = 0.0

    def _delay_lobby_start(self, seconds, reason=""):
        now = time.time()
        self._lobby_start_blocked_until = max(self._lobby_start_blocked_until, now + max(0.0, float(seconds)))
        self._lobby_start_block_reason = reason or self._lobby_start_block_reason
        self._start_wait_logged_at = 0.0

    @staticmethod
    def _is_endish_state(state):
        return isinstance(state, str) and (
            state.startswith("end")
            or state in {"reward_claim", "trophy_reward", "player_title_reward", "prestige_reward", "star_drop"}
        )

    def _begin_end_transition(self, result=None, now=None):
        if now is None:
            now = time.time()
        result = str(result or "").strip() or None
        if result != self._end_transition_last_result:
            self._end_transition_last_action_at = 0.0
        if not self._end_transition_started_at or result != self._end_transition_last_result:
            self._end_transition_started_at = now
        self._end_transition_last_result = result
        # Some post-match reward/proceed screens fall through to "match" while
        # EasyOCR or API sync is still catching up. Keep the result context for
        # the full transition window so gameplay never starts on those screens.
        self._end_transition_hold_match_until = max(
            self._end_transition_hold_match_until,
            now + max(4.0, self._end_transition_hold_seconds, self._end_transition_timeout),
        )

    def _clear_end_transition(self):
        self._end_transition_started_at = 0.0
        self._end_transition_last_action_at = 0.0
        self._end_transition_last_result = None
        self._end_transition_hold_match_until = 0.0

    def has_recent_end_transition(self, now=None):
        if now is None:
            now = time.time()
        if not self._end_transition_started_at:
            return False
        return (now - self._end_transition_started_at) < self._end_transition_timeout

    def get_end_transition_state(self):
        if self._end_transition_last_result:
            return f"end_{self._end_transition_last_result}"
        if self._end_transition_started_at:
            return "end"
        return ""

    def should_hold_match_probe(self, now=None):
        if now is None:
            now = time.time()
        return (
            self._awaiting_lobby_result_sync
            and self.has_recent_end_transition(now)
            and now < self._end_transition_hold_match_until
        )

    def _reset_lobby_result_sync_state(self):
        self._awaiting_lobby_result_sync = False
        self._match_in_progress = False
        self._lobby_sync_started_at = 0.0
        self._pending_verified_result = None
        self._api_lobby_sync_attempts = 0
        self._last_api_lobby_sync_attempt_at = 0.0

    def _finish_pending_result_sync(self, reason=""):
        result_ready = bool(self._result_applied_for_active_match)
        if self._pending_verified_result and not result_ready:
            print(
                f"[RESULT] lobby verification skipped; applying pending "
                f"{self._pending_verified_result}"
            )
            result_ready = bool(self._apply_match_result(self._pending_verified_result))

        if result_ready:
            if reason:
                print(f"[RESULT] {reason}; keeping predicted progress")
            self._commit_active_brawler_progress(queue_milestone=True)
        elif reason:
            print(f"[RESULT] {reason}; no committed result was available")

        self._reset_lobby_result_sync_state()
        return result_ready

    def is_post_match_resolution_pending(self, now=None):
        return bool(self._awaiting_lobby_result_sync and not self._match_in_progress)

    def _select_brawler_or_delay(self, brawler_name, reason="brawler auto-select", prefix="[SELECT]"):
        brawler_name = str(brawler_name or "").strip().lower()
        if not brawler_name:
            return True

        self.window_controller.keys_up(list("wasd"))
        screenshot = self.window_controller.screenshot()
        current_state = get_state(screenshot, allow_reward_ocr=self._is_easyocr_ready())
        max_attempts = 30
        attempts = 0
        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        while current_state != "lobby" and attempts < max_attempts:
            self.window_controller.click(int(100 * wr), int(60 * hr))
            if debug:
                print("Pressed back arrow to return to lobby")
            time.sleep(1)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot, allow_reward_ocr=self._is_easyocr_ready())
            attempts += 1

        if current_state != "lobby":
            print(f"{prefix} Failed to reach lobby before selecting {brawler_name}.")
            self._delay_lobby_start(5.0, reason)
            return False

        if not self.Lobby_automation.select_brawler(brawler_name):
            print(
                f"{prefix} Could not confirm brawler switch to '{brawler_name}'. "
                "Keeping lobby start blocked until retry."
            )
            self._delay_lobby_start(5.0, reason)
            return False

        return True

    def _recover_from_brawler_selection(self):
        now = time.time()
        if now - self._last_brawler_menu_recovery_at < 0.9:
            return False
        region = (self.lobby_config.get("template_matching") or {}).get("go_back_arrow") or [0, 0, 175, 110]
        if len(region) != 4:
            region = [0, 0, 175, 110]
        x, y, w, h = [int(value) for value in region]
        target_x = max(1, x + (w // 2))
        target_y = max(1, y + (h // 2))
        self.window_controller.keys_up(list("wasd"))
        self.window_controller.click(target_x, target_y, already_include_ratio=False)
        self._last_brawler_menu_recovery_at = now
        self._last_start_press_at = now
        self._start_wait_logged_at = 0.0
        print("[START] Unexpected brawler menu opened; backing out before retrying")
        return True

    def _is_safe_lobby_to_start(self):
        try:
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
        except Exception as exc:
            print(f"[START] Could not verify lobby before pressing Q: {exc}")
            return False

        if current_state == "brawler_selection":
            self._recover_from_brawler_selection()
            return False
        if current_state != "lobby":
            print(f"[START] Start blocked because verified state is '{current_state}', not lobby.")
            return False

        try:
            if not is_lobby_play_button_visible(to_bgr_array(screenshot)):
                print("[START] Start blocked because the lobby PLAY button was not confirmed.")
                return False
        except Exception as exc:
            print(f"[START] Could not verify lobby PLAY button before pressing Q: {exc}")
            return False
        return True

    def _try_press_lobby_start(self, prefix="[RESULT]"):
        now = time.time()
        self._note_lobby_visible(now)
        lobby_visible_for = now - self._lobby_visible_since

        if now < self._lobby_start_blocked_until:
            if now - self._start_wait_logged_at >= 0.75:
                remaining = self._lobby_start_blocked_until - now
                reason = f" after {self._lobby_start_block_reason}" if self._lobby_start_block_reason else ""
                print(
                    f"[START] delaying lobby Q press{reason} "
                    f"({remaining:.2f}s remaining)"
                )
                self._start_wait_logged_at = now
            return False

        if lobby_visible_for < self._lobby_start_settle_delay:
            if now - self._start_wait_logged_at >= 0.75:
                print(
                    f"[START] waiting for lobby UI to settle before pressing Q "
                    f"({lobby_visible_for:.2f}s)"
                )
                self._start_wait_logged_at = now
            return False

        if self._last_start_press_at and (now - self._last_start_press_at) < self._lobby_start_retry_delay:
            return False

        if not self._is_safe_lobby_to_start():
            self._delay_lobby_start(1.0, "waiting for confirmed lobby play button")
            return False

        self.window_controller.keys_up(list("wasd"))
        self.window_controller.press_key("Q")
        self._last_start_press_at = now
        self._start_press_attempts += 1
        self._start_wait_logged_at = 0.0
        self._lobby_start_block_reason = ""
        self._consecutive_lobby_start_fails = 0
        if self._start_press_attempts == 1:
            print(f"{prefix} Pressed Q to start a match")
        else:
            print(f"{prefix} Lobby still visible; retrying Q press ({self._start_press_attempts})")
        return True

    @staticmethod
    def _coerce_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _number_or_default(cls, value, default=0):
        return cls._coerce_int(value, default)

    def _resolve_push_progress(self):
        active = self.brawlers_pick_data[0] if self.brawlers_pick_data else {}
        type_to_push = str(active.get("type", "trophies") or "trophies")
        if type_to_push not in {"trophies", "wins"}:
            type_to_push = "trophies"

        use_verified_progress = (
            self._awaiting_lobby_result_sync
            and not getattr(self.Trophy_observer, "_lobby_trophy_verified", False)
        )

        if type_to_push == "wins":
            raw_value = active.get("wins") if use_verified_progress else self.Trophy_observer.current_wins
            default_target = 300
        else:
            raw_value = active.get("trophies") if use_verified_progress else self.Trophy_observer.current_trophies
            default_target = 1000

        if raw_value is None:
            raw_value = active.get(type_to_push)

        value = self._coerce_int(raw_value, 0)
        push_target = self._coerce_int(active.get("push_until"), default_target)
        return type_to_push, value, push_target

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

    def _pick_next_farm_brawler(self):
        """Smart Trophy Farm: re-sort remaining brawlers and pick the best next one."""
        if len(self.brawlers_pick_data) <= 1:
            return False  # No more brawlers

        # Remove the current (completed) brawler
        completed = self.brawlers_pick_data.pop(0)
        self._pending_farm_completed_restore = completed
        completed_summary = self._build_live_summary(completed['brawler'])
        self._send_webhook(
            "brawler_completed",
            subject=completed['brawler'],
            live_summary=completed_summary,
        )
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

    def _restore_pending_farm_rotation(self):
        completed = getattr(self, "_pending_farm_completed_restore", None)
        if not completed:
            return False
        completed_name = completed.get("brawler")
        if completed_name and (
            not self.brawlers_pick_data
            or self.brawlers_pick_data[0].get("brawler") != completed_name
        ):
            self.brawlers_pick_data.insert(0, completed)
            self.Trophy_observer.change_trophies(completed.get('trophies', 0))
            self.Trophy_observer.current_wins = self._coerce_int(completed.get('wins'), 0)
            self.Trophy_observer.win_streak = self._coerce_int(completed.get('win_streak'), 0)
            save_brawler_data(self.brawlers_pick_data)
            print(f"[FARM] Restored active brawler to {completed_name.title()} until selection succeeds.")
        self._pending_farm_completed_restore = None
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
        previous_order = list(self.brawlers_pick_data)
        self.brawlers_pick_data.sort(key=lambda x: x.get('trophies', 0))
        new_brawler = self.brawlers_pick_data[0]['brawler']

        if new_brawler == current_brawler:
            print(f"[FARM] Dynamic rotation: {current_brawler.title()} is already the lowest — no switch needed.")
            return False

        print(f"[FARM] Dynamic rotation: switching from {current_brawler.title()} "
              f"to {new_brawler.title()} ({self.brawlers_pick_data[0].get('trophies', 0)} trophies).")
        self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
        self.Trophy_observer.current_wins = self._coerce_int(self.brawlers_pick_data[0].get('wins'), 0)
        self.Trophy_observer.win_streak = self._coerce_int(self.brawlers_pick_data[0].get('win_streak'), 0)
        save_brawler_data(self.brawlers_pick_data)
        self._pending_dynamic_rotation_restore = previous_order
        return True

    def _restore_pending_dynamic_rotation(self):
        previous_order = getattr(self, "_pending_dynamic_rotation_restore", None)
        if not previous_order:
            return False
        self.brawlers_pick_data = previous_order
        active = self.brawlers_pick_data[0]
        self.Trophy_observer.change_trophies(active.get('trophies', 0))
        self.Trophy_observer.current_wins = self._coerce_int(active.get('wins'), 0)
        self.Trophy_observer.win_streak = self._coerce_int(active.get('win_streak'), 0)
        save_brawler_data(self.brawlers_pick_data)
        self._pending_dynamic_rotation_restore = None
        print(f"[FARM] Restored dynamic rotation to {active.get('brawler', 'current').title()} until selection succeeds.")
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
            screenshot = self.window_controller.screenshot()
            self._flush_webhook_milestone(screenshot)
            self._send_webhook(
                "quests_completed",
                screenshot=screenshot,
                current_brawler=self._current_brawler_name(),
            )
            if os.path.exists("latest_brawler_data.json"):
                os.remove("latest_brawler_data.json")
            self.window_controller.keys_up(list("wasd"))
            self.window_controller.close()
            sys.exit(0)

        completed = self.brawlers_pick_data[0]
        next_data = self.brawlers_pick_data[1]
        if not self._select_brawler_or_delay(next_data['brawler'], "waiting for quest brawler auto-select", prefix="[QUEST]"):
            return

        completed_summary = self._build_live_summary(completed['brawler'])
        self._send_webhook(
            "brawler_completed",
            subject=completed['brawler'],
            live_summary=completed_summary,
        )
        self.brawlers_pick_data.pop(0)
        print(f"[QUEST] Switching from {completed['brawler'].title()} -> "
              f"{self.brawlers_pick_data[0]['brawler'].title()} "
              f"({len(self.brawlers_pick_data)} remaining)")

        # Load next brawler data
        self.Trophy_observer.change_trophies(next_data.get('trophies', 0))
        self.Trophy_observer.current_wins = 0
        self.Trophy_observer.win_streak = 0
        save_brawler_data(self.brawlers_pick_data)

    def _ensure_lobby(self):
        """Navigate back to the lobby using the back arrow (NOT Q, which starts a match)."""
        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        screenshot = self.window_controller.screenshot()
        current_state = get_state(screenshot, allow_reward_ocr=True)
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
        region = (self.lobby_config.get("lobby") or {}).get("trophy_observer")
        if screenshot is None:
            return None
        if not region or len(region) != 4:
            try:
                wr = self.window_controller.width_ratio or 1.0
                hr = self.window_controller.height_ratio or 1.0
                region = (
                    int(140 * wr),
                    int(10 * hr),
                    int(380 * wr),
                    int(60 * hr),
                )
                frame = to_bgr_array(screenshot)
                cropped = frame[region[1]:region[3], region[0]:region[2]]
            except Exception:
                return None
        else:
            try:
                frame = to_bgr_array(screenshot)
            except Exception:
                return None
            wr = self.window_controller.width_ratio or 1.0
            hr = self.window_controller.height_ratio or 1.0
            x, y, width, height = region
            pad_x = max(8, int(width * wr * 0.16))
            pad_y = max(6, int(height * hr * 0.20))
            x1 = max(0, int(x * wr) - pad_x)
            y1 = max(0, int(y * hr) - pad_y)
            x2 = min(frame.shape[1], int((x + width) * wr) + pad_x)
            y2 = min(frame.shape[0], int((y + height) * hr) + pad_y)
            cropped = frame[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        baseline = int(self.Trophy_observer.current_trophies or self.brawlers_pick_data[0].get("trophies", 0) or 0)
        candidates = []
        variants = [cropped]
        try:
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            for scale in (2.0, 3.0):
                resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                variants.append(resized)
                _, thresh_dark = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, thresh_light = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                variants.extend([thresh_dark, thresh_light])
                adaptive = cv2.adaptiveThreshold(
                    resized,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    31,
                    11,
                )
                variants.append(adaptive)
        except Exception:
            pass

        for variant in variants:
            try:
                ocr_result = reader.readtext(variant, allowlist="0123456789soibl|")
            except TypeError:
                ocr_result = reader.readtext(variant)
            except Exception:
                continue

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

    def _sync_active_brawler_progress(self):
        if not self.brawlers_pick_data:
            return
        active = self.brawlers_pick_data[0]
        active['trophies'] = self._coerce_int(
            self.Trophy_observer.current_trophies,
            self._coerce_int(active.get('trophies'), 0),
        )
        active['wins'] = self._coerce_int(
            self.Trophy_observer.current_wins,
            self._coerce_int(active.get('wins'), 0),
        )
        active['win_streak'] = self._coerce_int(
            self.Trophy_observer.win_streak,
            self._coerce_int(active.get('win_streak'), 0),
        )

    def _commit_active_brawler_progress(self, queue_milestone=True):
        if not self.brawlers_pick_data:
            return False
        current_brawler = self.brawlers_pick_data[0].get('brawler')
        self._sync_active_brawler_progress()
        if queue_milestone:
            self._queue_webhook_milestone(current_brawler)
        save_brawler_data(self.brawlers_pick_data)
        return True

    def _prepare_next_push_all_brawler(self, target, type_of_push="trophies"):
        """Drop completed rows and promote the next brawler, sorting Push All by lowest trophies."""
        if not self.brawlers_pick_data:
            return False

        target = self._number_or_default(target, 1000 if type_of_push == "trophies" else 300)
        current_row = dict(self.brawlers_pick_data[0])
        current_row[type_of_push] = self._number_or_default(
            getattr(self.Trophy_observer, f"current_{type_of_push}", current_row.get(type_of_push, 0)),
            current_row.get(type_of_push, 0),
        )
        current_row["win_streak"] = self._number_or_default(self.Trophy_observer.win_streak, 0)

        remaining = [dict(row) for row in self.brawlers_pick_data[1:]]
        if type_of_push == "trophies":
            remaining = [
                row for row in remaining
                if self._number_or_default(row.get("trophies", 0), 0) < target
            ]
        else:
            remaining = [
                row for row in remaining
                if self._number_or_default(row.get("wins", 0), 0) < target
            ]

        if not remaining:
            self.brawlers_pick_data = []
            save_brawler_data(self.brawlers_pick_data)
            return False

        if any(row.get("selection_method") == "lowest_trophies" for row in remaining):
            remaining.sort(
                key=lambda row: (
                    self._number_or_default(row.get(type_of_push, 0), 0),
                    str(row.get("brawler", "")),
                )
            )
            for row in remaining:
                row["selection_method"] = "lowest_trophies"
                row["automatically_pick"] = True

        self.brawlers_pick_data = remaining
        next_data = self.brawlers_pick_data[0]
        self.Trophy_observer.change_trophies(self._number_or_default(next_data.get("trophies", 0), 0))
        self.Trophy_observer.current_wins = self._number_or_default(next_data.get("wins", 0), 0)
        self.Trophy_observer.win_streak = self._number_or_default(next_data.get("win_streak", 0), 0)
        save_brawler_data(self.brawlers_pick_data)
        return True

    @staticmethod
    def fetch_push_all_player_data(force_token_refresh=False):
        api_config = load_brawl_stars_api_config(
            "cfg/brawl_stars_api.toml",
            force_refresh=force_token_refresh,
        )
        return fetch_brawl_stars_player(
            api_config.get("api_token", "").strip(),
            api_config.get("player_tag", "").strip(),
            int(api_config.get("timeout_seconds", 15)),
        )

    def refresh_push_all_trophies_from_api(self):
        if not self.brawlers_pick_data:
            return False
        if self.brawlers_pick_data[0].get("type", "trophies") != "trophies":
            return False
        if not any(row.get("selection_method") == "lowest_trophies" for row in self.brawlers_pick_data):
            return False
        now = time.time()
        if now - self._last_push_all_refresh_at < self._push_all_refresh_interval:
            return False
        self._last_push_all_refresh_at = now

        old_front_brawler = self.brawlers_pick_data[0].get("brawler")
        try:
            player_data = self.fetch_push_all_player_data(force_token_refresh=False)
        except ValueError as exc:
            if debug:
                print(f"Push All API trophy refresh skipped: {exc}")
            return False
        except RuntimeError as exc:
            if "accessDenied" not in str(exc):
                print(f"Push All API trophy refresh failed; using local trophies. {exc}")
                return False
            try:
                print("Push All API token was rejected; refreshing token for current public IP and retrying.")
                player_data = self.fetch_push_all_player_data(force_token_refresh=True)
            except Exception as retry_error:
                print(f"Push All API trophy refresh failed after token refresh; using local trophies. {retry_error}")
                return False
        except Exception as exc:
            print(f"Push All API trophy refresh failed; using local trophies. {exc}")
            return False

        trophies_by_brawler = {
            normalize_brawler_name(brawler.get("name", "")): int(brawler.get("trophies", 0))
            for brawler in player_data.get("brawlers", [])
        }
        target = self._number_or_default(self.brawlers_pick_data[0].get("push_until", 1000), 1000)
        refreshed_rows = []
        changed = False

        for row in self.brawlers_pick_data:
            refreshed_row = dict(row)
            key = normalize_brawler_name(refreshed_row.get("brawler", ""))
            if key in trophies_by_brawler:
                api_trophies = trophies_by_brawler[key]
                if refreshed_row.get("brawler") == old_front_brawler:
                    local_trophies = self._number_or_default(
                        getattr(self.Trophy_observer, "current_trophies", refreshed_row.get("trophies", 0)),
                        refreshed_row.get("trophies", 0),
                    )
                    api_trophies = max(api_trophies, local_trophies)
                if refreshed_row.get("trophies") != api_trophies:
                    refreshed_row["trophies"] = api_trophies
                    changed = True
            if self._number_or_default(refreshed_row.get("trophies", 0), 0) < target:
                refreshed_rows.append(refreshed_row)

        current_row = next((row for row in refreshed_rows if row.get("brawler") == old_front_brawler), None)
        remaining_rows = [row for row in refreshed_rows if row.get("brawler") != old_front_brawler]

        if current_row is not None:
            remaining_rows.sort(
                key=lambda row: (
                    self._number_or_default(row.get("trophies", 0), 0),
                    str(row.get("brawler", "")),
                )
            )
            refreshed_rows = [current_row] + remaining_rows
            self.push_all_needs_selection = False
        else:
            remaining_rows.sort(
                key=lambda row: (
                    self._number_or_default(row.get("trophies", 0), 0),
                    str(row.get("brawler", "")),
                )
            )
            refreshed_rows = remaining_rows
            self.push_all_needs_selection = bool(refreshed_rows)

        if refreshed_rows:
            refreshed_rows[0]["automatically_pick"] = False
            refreshed_rows[0]["selection_method"] = "lowest_trophies"
            for row in refreshed_rows[1:]:
                if row.get("automatically_pick") is not True:
                    changed = True
                row["automatically_pick"] = True
                row["selection_method"] = "lowest_trophies"

        old_order = [row.get("brawler") for row in self.brawlers_pick_data]
        new_order = [row.get("brawler") for row in refreshed_rows]
        if old_order != new_order or len(refreshed_rows) != len(self.brawlers_pick_data):
            changed = True

        if not refreshed_rows:
            self.brawlers_pick_data = []
            save_brawler_data(self.brawlers_pick_data)
            print("Push All API trophies refreshed: all brawlers reached target.")
            return True

        self.brawlers_pick_data = refreshed_rows
        current_trophies = self._number_or_default(self.brawlers_pick_data[0].get("trophies", 0), 0)
        if getattr(self.Trophy_observer, "current_trophies", None) != current_trophies:
            self.Trophy_observer.change_trophies(current_trophies)
            changed = True

        if changed:
            if self.push_all_needs_selection:
                print("Push All API trophies refreshed; current brawler reached target, selecting next lowest.")
            else:
                print("Push All API trophies refreshed; keeping current brawler until target.")
            save_brawler_data(self.brawlers_pick_data)
        return changed

    def _current_brawler_name(self):
        if not self.brawlers_pick_data:
            return None
        return self.brawlers_pick_data[0].get('brawler')

    def _build_live_summary(self, brawler_name=None):
        target_brawler = brawler_name or self._current_brawler_name()
        return self.Trophy_observer.build_live_notification_summary(target_brawler)

    def _queue_webhook_milestone(self, brawler_name=None):
        target_brawler = brawler_name or self._current_brawler_name()
        if not target_brawler:
            self._pending_webhook_milestone_summary = None
            return None
        self._pending_webhook_milestone_summary = self.Trophy_observer.preview_trophy_milestone(target_brawler)
        return self._pending_webhook_milestone_summary

    def _flush_webhook_milestone(self, screenshot=None):
        summary = self._pending_webhook_milestone_summary
        self._pending_webhook_milestone_summary = None
        if not summary or not has_notification_webhook():
            return False

        brawler_name = summary.get("brawler")
        if not self.Trophy_observer.commit_trophy_milestone(brawler_name, summary.get("milestone_bucket")):
            return False

        if screenshot is None:
            try:
                screenshot = self.window_controller.screenshot()
            except Exception:
                screenshot = None
        return notify_user(
            "milestone_reached",
            screenshot=screenshot,
            subject=brawler_name,
            live_summary=summary,
        )

    def _send_webhook(self, message_type, screenshot=None, subject=None, current_brawler=None, live_summary=None):
        if not has_notification_webhook():
            return False
        if live_summary is None:
            live_summary = self._build_live_summary(current_brawler or subject)
        if screenshot is None:
            try:
                screenshot = self.window_controller.screenshot()
            except Exception:
                screenshot = None
        return notify_user(
            message_type,
            screenshot=screenshot,
            subject=subject,
            live_summary=live_summary,
        )

    def mark_match_started(self):
        if self._match_in_progress:
            return False
        now = time.time()
        recent_start_press = bool(self._last_start_press_at and (now - self._last_start_press_at) <= 12.0)
        sync_started_at = max(
            self._lobby_sync_started_at,
            self._last_result_applied_at,
        )
        new_start_press = recent_start_press and (
            sync_started_at <= 0.0 or self._last_start_press_at >= sync_started_at
        )
        if self._awaiting_lobby_result_sync:
            if self._result_applied_for_active_match or self._pending_verified_result:
                if not new_start_press:
                    if debug:
                        print("[RESULT] ignoring match-start probe while post-match sync is still pending")
                    return False
                self._finish_pending_result_sync("new match started before lobby verification")
            else:
                if not new_start_press:
                    if debug:
                        print("[RESULT] ignoring match-start probe with unresolved lobby sync")
                    return False
                print("[RESULT] clearing unresolved lobby sync because a new match started")
                self._reset_lobby_result_sync_state()
        if debug:
            active_name = self.brawlers_pick_data[0]['brawler'] if self.brawlers_pick_data else "unknown"
            print(f"[RESULT] mark_match_started for {active_name}")
        self._match_in_progress = True
        self._awaiting_lobby_result_sync = True
        self._result_applied_for_active_match = False
        self._last_result_applied_at = 0.0
        self._lobby_sync_started_at = 0.0
        self._pending_verified_result = None
        self._api_lobby_sync_attempts = 0
        self._last_api_lobby_sync_attempt_at = 0.0
        self._reset_lobby_start_tracking()
        if self.brawlers_pick_data:
            self.Trophy_observer.begin_match(self.brawlers_pick_data[0]['brawler'])
        return True

    def _apply_or_defer_detected_result(self, game_result, source="detector"):
        if not game_result:
            return False
        if game_result == "draw":
            self._pending_verified_result = "draw"
            print(f"[RESULT] deferring draw from {source} until lobby verification")
            return True
        self._pending_verified_result = None
        return self._apply_match_result(game_result)

    def _sync_lobby_result(self, frame, allow_ocr=True, api_timeout=1.25):
        if not self._awaiting_lobby_result_sync or not self.brawlers_pick_data:
            return False

        current_brawler = self.brawlers_pick_data[0]['brawler']
        screenshot = frame
        if screenshot is None:
            try:
                screenshot = self.window_controller.screenshot()
            except Exception:
                screenshot = None

        verified_trophies = None
        verification_source = None
        elapsed = time.time() - self._lobby_sync_started_at if self._lobby_sync_started_at else 0.0
        has_api_settings = self.Trophy_observer.has_brawlstars_api_settings()
        if has_api_settings:
            should_attempt_api = False
            force_api = False
            now = time.time()
            if self._api_lobby_sync_attempts == 0:
                should_attempt_api = True
            elif (
                self._api_lobby_sync_attempts == 1
                and elapsed >= 2.5
                and now - self._last_api_lobby_sync_attempt_at >= 1.25
            ):
                should_attempt_api = True
                force_api = True
            if should_attempt_api:
                self._api_lobby_sync_attempts += 1
                self._last_api_lobby_sync_attempt_at = now
                verified_trophies = self.Trophy_observer.fetch_brawler_trophies_from_brawlstars_api(
                    current_brawler,
                    force=force_api,
                    timeout=api_timeout,
                )
                if verified_trophies is not None:
                    verification_source = "api"
                    print(f"[RESULT] API fallback trophies for {current_brawler} -> {verified_trophies}")

        if verified_trophies is None and (
            not has_api_settings
            or self._api_lobby_sync_attempts >= 2
            or elapsed >= 3.5
        ):
            if not allow_ocr:
                return False
            verified_trophies = self._read_lobby_trophies(screenshot)
            verification_source = "ocr" if verified_trophies is not None else None
            if verified_trophies is None:
                if debug:
                    print("[RESULT] lobby trophy OCR did not find a usable value")
                return False

        verified_trophies = self._coerce_int(verified_trophies, None)
        if verified_trophies is None:
            return False

        match_start_trophies = self.Trophy_observer.get_active_match_start_trophies(current_brawler)
        if match_start_trophies is None:
            match_start_trophies = self._coerce_int(
                self.brawlers_pick_data[0].get("trophies"),
                self._coerce_int(self.Trophy_observer.current_trophies, 0),
            )
        else:
            match_start_trophies = self._coerce_int(
                match_start_trophies,
                self._coerce_int(self.Trophy_observer.current_trophies, 0),
            )

        if verification_source == "api" and verified_trophies == match_start_trophies:
            elapsed = time.time() - self._lobby_sync_started_at if self._lobby_sync_started_at else 0.0
            if elapsed < 2.0:
                if debug:
                    print("[RESULT] API fallback still matches start trophies; waiting before resolving draw")
                return False

        if verified_trophies > match_start_trophies:
            inferred_result = "victory"
        elif verified_trophies < match_start_trophies:
            inferred_result = "defeat"
        else:
            inferred_result = "draw"

        if not self._result_applied_for_active_match:
            applied = self._apply_match_result(inferred_result)
            if not applied:
                return False

        self.Trophy_observer.reconcile_verified_trophies(current_brawler, verified_trophies)
        self._commit_active_brawler_progress(queue_milestone=True)
        self._awaiting_lobby_result_sync = False
        self._match_in_progress = False
        self._lobby_sync_started_at = 0.0
        self._pending_verified_result = None
        self._api_lobby_sync_attempts = 0
        self._last_api_lobby_sync_attempt_at = 0.0
        if debug:
            print(
                f"Lobby result sync applied as '{inferred_result}' via {verification_source} "
                f"({match_start_trophies} -> {verified_trophies})"
            )
        return True

    def _apply_match_result(self, game_result):
        if not self.brawlers_pick_data or not game_result:
            return False
        if self._result_applied_for_active_match:
            if debug:
                print(f"[RESULT] skipping duplicate apply for {game_result}")
            return False

        current_brawler = self.brawlers_pick_data[0]['brawler']
        print(f"[RESULT] applying {game_result} for {current_brawler}")
        applied = self.Trophy_observer.add_trophies(game_result, current_brawler)
        self.Trophy_observer.add_win(game_result)
        self.time_since_last_stat_change = time.time()

        type_to_push, value, _ = self._resolve_push_progress()
        self._result_applied_for_active_match = True
        self._last_result_applied_at = time.time()
        self._match_in_progress = False
        print(f"[RESULT] applied={applied} predicted_value={value} type={type_to_push}")
        return applied

    def start_game(self, data):
        print("state is lobby, starting game")

        if self._awaiting_lobby_result_sync:
            synced = False
            if not self._lobby_sync_started_at:
                self._lobby_sync_started_at = time.time()
            elapsed = time.time() - self._lobby_sync_started_at

            if not self._result_applied_for_active_match:
                print("[RESULT] entering lobby fallback sync because no result was committed yet")
                direct_result = False
                try:
                    if data is not None:
                        direct_result = find_game_result(data)
                except Exception:
                    direct_result = False
                if direct_result:
                    print(f"[RESULT] lobby entry direct probe recovered {direct_result}")
                    self._apply_or_defer_detected_result(direct_result, source="lobby-entry")

            ocr_ready = self._is_easyocr_ready()
            has_api_settings = self.Trophy_observer.has_brawlstars_api_settings()
            finalized_without_verification = False
            if (
                not ocr_ready
                and not has_api_settings
                and (self._result_applied_for_active_match or self._pending_verified_result)
            ):
                self._finish_pending_result_sync("lobby verification unavailable without API/OCR")
                self._restart_lobby_settle_window()
                self._delay_lobby_start(self._post_result_lobby_delay, "post-match UI sync")
                finalized_without_verification = True
            elif not ocr_ready and not has_api_settings:
                self._ensure_lobby_ocr_warmup(delay_seconds=0.0)

            synced = finalized_without_verification
            if not synced and self._awaiting_lobby_result_sync:
                synced = self._sync_lobby_result(
                    data,
                    allow_ocr=ocr_ready,
                    api_timeout=3.0,
                )
                elapsed = time.time() - self._lobby_sync_started_at if self._lobby_sync_started_at else elapsed

            if synced:
                self._restart_lobby_settle_window()
                self._delay_lobby_start(self._post_result_lobby_delay, "post-match UI sync")
            else:
                verification_window = 2.25 if (has_api_settings or ocr_ready) else 1.25
                if elapsed < verification_window:
                    if elapsed >= self._lobby_sync_max_timeout:
                        print(f"[RESULT] lobby sync hit hard max timeout ({elapsed:.1f}s), forcing through")
                    else:
                        if debug:
                            print(
                                f"[RESULT] waiting for verified trophy sync "
                                f"({elapsed:.2f}s/{verification_window:.2f}s)"
                            )
                        self._delay_lobby_start(self._post_result_lobby_delay, "waiting for verified trophy sync")
                        return

                if self._pending_verified_result and not self._result_applied_for_active_match:
                    print(f"[RESULT] lobby verification unavailable; falling back to pending {self._pending_verified_result}")
                    self._apply_match_result(self._pending_verified_result)
                    self._pending_verified_result = None
                elif not self._result_applied_for_active_match:
                    print("[RESULT] WARNING: match result could not be verified (no API, OCR failed, no pending result). Result dropped.")

                if self._result_applied_for_active_match:
                    print("[RESULT] verified trophy sync unavailable; keeping predicted progress for this match")
                    self._commit_active_brawler_progress(queue_milestone=True)

                self._awaiting_lobby_result_sync = False
                self._match_in_progress = False
                self._lobby_sync_started_at = 0.0
                self._pending_verified_result = None
                self._restart_lobby_settle_window()
                self._delay_lobby_start(self._post_result_lobby_delay, "post-match UI sync")
        self.push_all_needs_selection = False
        self.refresh_push_all_trophies_from_api()
        if not self.brawlers_pick_data:
            print("Bot stopping: all Push All targets completed.")
            self.window_controller.keys_up(list("wasd"))
            self.window_controller.close()
            sys.exit(0)
        self._flush_webhook_milestone(data)

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
                verified = self._read_lobby_trophies(lobby_screenshot)
                if verified is not None:
                    if self._is_manual_trophy_locked():
                        self._reset_to_manual_trophies()
                    else:
                        self.Trophy_observer.change_trophies(verified)
                        self.brawlers_pick_data[0]['trophies'] = verified
            except Exception:
                pass

            time.sleep(0.3)
            self._try_press_lobby_start(prefix="[FARM]")
            return

        # If end_game() already rotated to a new brawler, select it immediately
        if self._pending_brawler_switch:
            self._pending_brawler_switch = False
            next_brawler_name = self.brawlers_pick_data[0]['brawler']
            print(f"[FARM] Executing queued brawler switch → {next_brawler_name.title()}")
            if not self._select_brawler_or_delay(next_brawler_name, "waiting for queued brawler auto-select", prefix="[FARM]"):
                restored = self._restore_pending_farm_rotation() or self._restore_pending_dynamic_rotation()
                self._pending_brawler_switch = not restored
                return
            self._pending_farm_completed_restore = None
            self._pending_dynamic_rotation_restore = None
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
            self._try_press_lobby_start(prefix="[FARM]")
            return

        type_of_push, value, push_current_brawler_till = self._resolve_push_progress()

        if value >= push_current_brawler_till:
            # smart Trophy Farm: use intelligent rotation
            if self.smart_trophy_farm:
                has_next = self._pick_next_farm_brawler()
                if not has_next:
                    print("[FARM] All brawlers reached target! Bot stopping.")
                    screenshot = self.window_controller.screenshot()
                    self._flush_webhook_milestone(screenshot)
                    self._send_webhook(
                        "farm_completed",
                        screenshot=screenshot,
                        current_brawler=self._current_brawler_name(),
                    )
                    if os.path.exists("latest_brawler_data.json"):
                        os.remove("latest_brawler_data.json")
                    self.window_controller.keys_up(list("wasd"))
                    self.window_controller.close()
                    sys.exit(0)

                # Load next brawler's data
                self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
                self.Trophy_observer.current_wins = self._coerce_int(self.brawlers_pick_data[0].get('wins'), 0)
                self.Trophy_observer.win_streak = self._coerce_int(self.brawlers_pick_data[0].get('win_streak'), 0)
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
                    if not self.Lobby_automation.select_brawler(next_brawler_name):
                        self._restore_pending_farm_rotation()
                        self._delay_lobby_start(5.0, "waiting for farm brawler auto-select")
                        return
                    self._pending_farm_completed_restore = None
                else:
                    print("[FARM] Failed to reach lobby for brawler switch")
                    self._restore_pending_farm_rotation()
                    self._delay_lobby_start(5.0, "waiting for farm lobby recovery")
                    return
            else:
                # original sequential rotation
                if len(self.brawlers_pick_data) <= 1:
                    print("Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                          "Bot will now pause itself until closed.", value, push_current_brawler_till)
                    screenshot = self.window_controller.screenshot()
                    self._flush_webhook_milestone(screenshot)
                    self._send_webhook(
                        "all_targets_completed",
                        screenshot=screenshot,
                        current_brawler=self._current_brawler_name(),
                    )
                    print("Bot stopping: all targets completed with no more brawlers.")
                    self.window_controller.keys_up(list("wasd"))
                    self.window_controller.close()
                    sys.exit(0)
                completed_brawler = self.brawlers_pick_data[0]["brawler"]
                completed_summary = self._build_live_summary(completed_brawler)
                next_data = self.brawlers_pick_data[1]
                next_brawler_name = next_data['brawler']
                screenshot = self.window_controller.screenshot()
                if next_data.get("automatically_pick", True):
                    if debug:
                        print("Picking next automatically picked brawler")
                    if next_data.get("selection_method") == "lowest_trophies":
                        if not self.Lobby_automation.select_lowest_trophy_brawler():
                            print(
                                "WARNING: Could not confirm lowest-trophy brawler selection. "
                                "Keeping lobby start blocked until retry."
                            )
                            self.window_controller.keys_up(list("wasd"))
                            self._delay_lobby_start(5.0, "waiting for lowest-trophy brawler auto-select")
                            return
                    elif not self._select_brawler_or_delay(next_brawler_name, "waiting for brawler auto-select", prefix="[RESULT]"):
                        return
                else:
                    print("Next brawler is in manual mode, waiting 10 seconds to let user switch.")
                    time.sleep(10)

                self._send_webhook(
                    "brawler_completed",
                    screenshot=screenshot,
                    subject=completed_brawler,
                    live_summary=completed_summary,
                )
                self.brawlers_pick_data.pop(0)
                self.Trophy_observer.change_trophies(next_data.get('trophies', 0))
                self.Trophy_observer.current_wins = self._coerce_int(next_data.get('wins'), 0)
                self.Trophy_observer.win_streak = self._coerce_int(next_data.get('win_streak'), 0)
                save_brawler_data(self.brawlers_pick_data)
        elif self.push_all_needs_selection:
            print("Push All queue changed from API; selecting the new lowest trophy brawler.")
            selected = bool(self.Lobby_automation.select_lowest_trophy_brawler())
            if not selected:
                print("Could not confirm the API-refreshed brawler selection reached lobby; delaying match start.")
                self.window_controller.keys_up(list("wasd"))
                self._delay_lobby_start(5.0, "waiting for API-refreshed brawler auto-select")
                return
            self.push_all_needs_selection = False

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
                    self._sync_active_brawler_progress()
                    save_brawler_data(self.brawlers_pick_data)
                    print(f"[VERIFY] Pre-match trophy correction saved")
                self.Trophy_observer._corrections_log.clear()  # Reset for this match
        except Exception as e:
            print(f"[VERIFY] Pre-match trophy check error: {e}")

        self._try_press_lobby_start(prefix="[RESULT]")

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
        screenshot = self.window_controller.screenshot()
        star_drop_type = get_star_drop_type(to_bgr_array(screenshot))
        if star_drop_type in ("angelic", "demonic"):
            print(f"{star_drop_type.capitalize()} star drop detected; forcing long press.")
            self.window_controller.press_key("Q", 10)
            return

        if star_drop_type == "standard":
            print("Standard star drop detected; fast tapping.")
            for _ in range(5):
                self.window_controller.press_key("Q")
                time.sleep(0.08)
            return

        if self.long_press_star_drop == "yes":
            self.window_controller.press_continue(hold_seconds=10, include_fallback_clicks=False)
        else:
            self.window_controller.press_continue()

    def handle_prestige_reward(self, frame=None):
        """Handle the prestige (1000-trophy) reward screen.

        Clicks NEXT, advances to the next queued brawler if available,
        then returns to the lobby and selects the new brawler.
        """
        print("[PRESTIGE] Prestige reward screen detected; clicking NEXT.")
        wr = self.window_controller.width_ratio or 1.0
        hr = self.window_controller.height_ratio or 1.0
        self.window_controller.keys_up(list("wasd"))
        self.window_controller.click(int(1410 * wr), int(990 * hr))
        time.sleep(1.0)

        if not self.brawlers_pick_data:
            self.window_controller.press_key("Q")
            return

        current_brawler = self.brawlers_pick_data[0].get("brawler", "current")
        print(f"[PRESTIGE] Treating {current_brawler} as completed (reached prestige).")

        if len(self.brawlers_pick_data) <= 1:
            print("[PRESTIGE] No next brawler queued. Continuing.")
            self.window_controller.press_key("Q")
            return

        next_data = self.brawlers_pick_data[1]
        next_name = next_data.get("brawler")
        if next_name and next_data.get("automatically_pick"):
            if not self._select_brawler_or_delay(next_name, "waiting for prestige brawler auto-select", prefix="[PRESTIGE]"):
                print(f"[PRESTIGE] Could not confirm switch to {next_name}; keeping current queue active.")
                return
            print(f"[PRESTIGE] Switched to {next_name}.")

        self.brawlers_pick_data.pop(0)
        self.Trophy_observer.change_trophies(next_data.get("trophies", 0))
        self.Trophy_observer.current_wins = self._coerce_int(next_data.get("wins"), 0)
        self.Trophy_observer.win_streak = self._coerce_int(next_data.get("win_streak"), 0)
        save_brawler_data(self.brawlers_pick_data)

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

        fallback_center = get_reward_claim_button_center(screenshot)
        if fallback_center:
            self.window_controller.click(*fallback_center)
            time.sleep(0.08)
            self.window_controller.press_continue(include_fallback_clicks=False)
            return

        reward_action = find_reward_claim_action(screenshot)
        if reward_action:
            self.window_controller.click(*reward_action)
            time.sleep(0.08)
            self.window_controller.press_continue(include_fallback_clicks=False)
            return

        self.window_controller.press_continue()

    def _legacy_end_game(self, frame=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()

        found_game_result = False
        read_match_stats = False
        verified_trophies = False
        current_state = get_state(screenshot)
        max_end_attempts = 30
        end_attempts = 0
        while (
            (
                str(current_state).startswith("end")
                or current_state in {"reward_claim", "trophy_reward", "star_drop"}
            )
            and end_attempts < max_end_attempts
        ):
            if current_state in {"reward_claim", "trophy_reward"}:
                self.states[current_state](screenshot)
                time.sleep(0.2)
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot, allow_reward_ocr=True)
                end_attempts += 1
                continue
            if current_state == "star_drop":
                self.click_star_drop()
                time.sleep(0.2)
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot, allow_reward_ocr=True)
                end_attempts += 1
                continue
            state_result = None
            if isinstance(current_state, str) and current_state.startswith("end_"):
                state_result = current_state.split("_", 1)[1]

            should_probe_result = (
                not found_game_result
                and (state_result is not None or time.time() - self.time_since_last_stat_change > 10)
            )
            if should_probe_result:
                found_game_result = self.Trophy_observer.find_game_result(
                    screenshot,
                    current_brawler=self.brawlers_pick_data[0]['brawler'],
                    game_result=state_result,
                )
                self.time_since_last_stat_change = time.time()
                if found_game_result:
                    # Track quest matches played
                    if self.brawlers_pick_data[0].get('type') == 'quest':
                        if not hasattr(self, '_quest_matches_played'):
                            self._quest_matches_played = {}
                        bn = self.brawlers_pick_data[0]['brawler']
                        self._quest_matches_played[bn] = self._quest_matches_played.get(bn, 0) + 1
                        print(f"[QUEST] {bn.title()} match #{self._quest_matches_played[bn]} completed")

                    type_to_push, value, push_current_brawler_till = self._resolve_push_progress()
                    self._sync_active_brawler_progress()
                    self.brawlers_pick_data[0][type_to_push] = value
                    save_brawler_data(self.brawlers_pick_data)

                    if value >= push_current_brawler_till:
                        if len(self.brawlers_pick_data) <= 1:
                            print(
                                "Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                                "Bot will now pause itself until closed.")
                            screenshot = self.window_controller.screenshot()
                            self._flush_webhook_milestone(screenshot)
                            self._send_webhook(
                                "all_targets_completed",
                                screenshot=screenshot,
                                current_brawler=self._current_brawler_name(),
                            )
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
                                self.Trophy_observer.current_wins = self._coerce_int(self.brawlers_pick_data[0].get('wins'), 0)
                                self.Trophy_observer.win_streak = self._coerce_int(self.brawlers_pick_data[0].get('win_streak'), 0)
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
                            self._sync_active_brawler_progress()
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

            # --- Play Again on Win: press the real Play Again button instead of Proceed ---
            if self.play_again_on_win and found_game_result and self.Trophy_observer._last_game_result == "victory":
                self.window_controller.press_play_again()
                print("[PLAY-AGAIN] Pressed Play Again")
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
                        self._sync_active_brawler_progress()
                        save_brawler_data(self.brawlers_pick_data)
                        print(f"[VERIFY] Lobby verification complete - {latest_corrections} total correction(s) this match")
                    self.Trophy_observer._corrections_log.clear()
        except Exception as e:
            print(f"[VERIFY] Lobby trophy verification error: {e}")

    def end_game(self, frame=None, known_result=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()

        found_game_result = False
        observed_result_processed = False
        result_post_processed = False
        read_match_stats = False
        current_state = get_state(screenshot, allow_reward_ocr=True)
        print(f"[RESULT] end_game entered known_result={known_result} current_state={current_state}")
        if known_result in {"victory", "defeat", "draw", "1st", "2nd", "3rd", "4th"}:
            self._begin_end_transition(known_result, time.time())
            found_game_result = known_result if self._apply_or_defer_detected_result(known_result, source="known-result") else False
            current_state = f"end_{known_result}"

        max_end_attempts = 30
        end_attempts = 0
        while (
            (
                str(current_state).startswith("end")
                or current_state in {"reward_claim", "trophy_reward", "player_title_reward", "star_drop"}
            )
            and end_attempts < max_end_attempts
        ):
            if current_state in {"reward_claim", "trophy_reward", "player_title_reward"}:
                self.states[current_state](screenshot)
                time.sleep(0.2)
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot, allow_reward_ocr=True)
                end_attempts += 1
                continue
            if current_state == "star_drop":
                self.click_star_drop()
                time.sleep(0.2)
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot, allow_reward_ocr=True)
                end_attempts += 1
                continue
            state_result = None
            if isinstance(current_state, str) and current_state.startswith("end_"):
                state_result = current_state.split("_", 1)[1]
                self._begin_end_transition(state_result, time.time())

            should_probe_result = (
                not found_game_result
                and (state_result is not None or time.time() - self.time_since_last_stat_change > 10)
            )
            if should_probe_result:
                if state_result is not None:
                    print(f"[RESULT] end_game state result probe -> {state_result}")
                    found_game_result = state_result if self._apply_or_defer_detected_result(state_result, source="state-probe") else False
                else:
                    detected_result = self.Trophy_observer.find_game_result(
                        screenshot,
                        current_brawler=self.brawlers_pick_data[0]['brawler'],
                        game_result=state_result,
                    )
                    if detected_result:
                        self._begin_end_transition(detected_result, time.time())
                        found_game_result = (
                            detected_result
                            if self._apply_or_defer_detected_result(detected_result, source="ocr-fallback")
                            else False
                        )
                        if detected_result == "draw" and not self._result_applied_for_active_match:
                            print("[RESULT] OCR fallback detected draw; awaiting lobby verification")
                self.time_since_last_stat_change = time.time()

            if found_game_result and not observed_result_processed:
                observed_result_processed = True
                if self.brawlers_pick_data[0].get('type') == 'quest':
                    if not hasattr(self, '_quest_matches_played'):
                        self._quest_matches_played = {}
                    bn = self.brawlers_pick_data[0]['brawler']
                    self._quest_matches_played[bn] = self._quest_matches_played.get(bn, 0) + 1
                    print(f"[QUEST] {bn.title()} match #{self._quest_matches_played[bn]} completed")

            if self._result_applied_for_active_match and not result_post_processed:
                result_post_processed = True
                type_to_push, value, push_current_brawler_till = self._resolve_push_progress()
                self._sync_active_brawler_progress()
                self.brawlers_pick_data[0][type_to_push] = value
                save_brawler_data(self.brawlers_pick_data)

                if value >= push_current_brawler_till:
                    if len(self.brawlers_pick_data) <= 1:
                        print(
                            "Brawler reached required trophies/wins. No more brawlers selected for pushing in the menu. "
                            "Bot will now pause itself until closed.")
                        screenshot = self.window_controller.screenshot()
                        self._flush_webhook_milestone(screenshot)
                        self._send_webhook(
                            "all_targets_completed",
                            screenshot=screenshot,
                            current_brawler=self._current_brawler_name(),
                        )
                        if os.path.exists("latest_brawler_data.json"):
                            os.remove("latest_brawler_data.json")
                        print("Bot stopping: all targets completed.")
                        self.window_controller.keys_up(list("wasd"))
                        self.window_controller.close()
                        sys.exit(0)
                    elif self.smart_trophy_farm:
                        print("[FARM] Target reached in end_game - rotating brawler immediately.")
                        has_next = self._pick_next_farm_brawler()
                        if has_next:
                            self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
                            self.Trophy_observer.current_wins = self._coerce_int(self.brawlers_pick_data[0].get('wins'), 0)
                            self.Trophy_observer.win_streak = self._coerce_int(self.brawlers_pick_data[0].get('win_streak'), 0)
                            save_brawler_data(self.brawlers_pick_data)
                            self._pending_brawler_switch = True
                            print(f"[FARM] Queued switch to {self.brawlers_pick_data[0]['brawler'].title()} for next lobby.")
                elif (self.smart_trophy_farm and self.dynamic_rotation_enabled
                        and len(self.brawlers_pick_data) > 1):
                    mc = self.Trophy_observer.match_counter
                    if mc > 0 and mc % self.dynamic_rotation_every == 0:
                        print(f"[FARM] Dynamic rotation triggered after {mc} total matches.")
                        switched = self._dynamic_rotate_to_lowest()
                        if switched:
                            self._pending_brawler_switch = True

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
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot, allow_reward_ocr=True)
                continue

            if (
                self.play_again_on_win
                and self._result_applied_for_active_match
                and self.Trophy_observer._last_game_result == "victory"
            ):
                self.window_controller.press_play_again()
                print("[PLAY-AGAIN] Pressed Play Again")
            else:
                self.window_controller.press_key("Q")
                if debug:
                    print("Game has ended, pressing Q")
            time.sleep(0.45)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot, allow_reward_ocr=True)
            end_attempts += 1

        if end_attempts >= max_end_attempts:
            print("End game screen stuck for too long, forcing continue")
        print(f"[RESULT] end_game exiting current_state={current_state} found={found_game_result}")
        if debug:
            print("Game has ended", current_state)

        if (
            self.play_again_on_win
            and self._result_applied_for_active_match
            and self.Trophy_observer._last_game_result == "victory"
        ):
            print("[PLAY-AGAIN] Waiting for new match to start...")
            start_wait_time = time.time()
            while time.time() - start_wait_time < 25:
                screenshot = self.window_controller.screenshot()
                current_state = get_state(screenshot)
                if current_state == "match":
                    print("[PLAY-AGAIN] New match started successfully!")
                    self._awaiting_lobby_result_sync = False
                    self._result_applied_for_active_match = False
                    self._match_in_progress = False
                    self._lobby_sync_started_at = 0.0
                    self._pending_verified_result = None
                    return
                time.sleep(0.5)
            print("[PLAY-AGAIN] Match did not start within 25s, pressing Q to return to lobby.")
            self.window_controller.press_continue()
            time.sleep(2)
            self.window_controller.press_continue()

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

    def handle_player_title_reward(self, frame=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()
        print("[RESULT] Player title reward detected; dismissing reward screen")
        fallback_center = get_reward_claim_button_center(screenshot)
        if fallback_center:
            self.window_controller.click(*fallback_center)
            time.sleep(0.08)
            self.window_controller.press_continue(include_fallback_clicks=False)
            return
        self.window_controller.press_key("Q")

    def do_state(self, state, data=None):
        known_result = None
        if isinstance(state, str) and state.startswith("end_"):
            known_result = state.split("_", 1)[1]
            state = "end"
        now = time.time()
        if state == "lobby":
            self._clear_end_transition()
        elif state == "match" and (
            self.should_hold_match_probe(now)
            or (self._awaiting_lobby_result_sync and not self._match_in_progress)
        ):
            state = "end"
            known_result = self._end_transition_last_result or known_result
        elif not self._is_endish_state(state):
            self._clear_end_transition()
        if state == "brawler_selection":
            recent_start_press = self._last_start_press_at and (time.time() - self._last_start_press_at) <= 4.0
            self._delay_lobby_start(self._unexpected_brawler_selection_delay, "recovering from accidental brawler menu")
            if recent_start_press:
                self._recover_from_brawler_selection()
        if state == "lobby":
            self._note_lobby_visible()
        else:
            self._reset_lobby_start_tracking(reset_last_press=(state != "brawler_selection"))
        if state not in self.states:
            print(f"[STAGE] Unknown state '{state}', pressing back arrow as fallback.")
            wr = self.window_controller.width_ratio or 1.0
            hr = self.window_controller.height_ratio or 1.0
            self.window_controller.click(int(100 * wr), int(60 * hr))
            return
        if state == "end":
            print(f"[RESULT] do_state -> end (known_result={known_result})")
            self.states[state](data, known_result)
            return
        if data is not None:
            self.states[state](data)
            return
        self.states[state]()

