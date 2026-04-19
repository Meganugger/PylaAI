import os.path
import sys

import time
import threading

import cv2
import numpy as np

from state_finder.main import get_state, find_game_result, find_reward_claim_action, get_reward_claim_button_center
from trophy_observer import TrophyObserver
from utils import find_template_center, load_toml_as_dict, notify_user, has_notification_webhook, \
    save_brawler_data, reader, to_bgr_array

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
            'end_1st': self.end_game,
            'end_2nd': self.end_game,
            'end_3rd': self.end_game,
            'end_4th': self.end_game,
            'lobby': self.start_game,
            'star_drop': self.click_star_drop
        }
        self.Lobby_automation = lobby_automator
        self.lobby_config = load_toml_as_dict("./cfg/lobby_config.toml")
        self.bot_config = load_toml_as_dict("./cfg/bot_config.toml")
        self.brawl_stars_icon = None
        self.close_popup_icon = None
        self.brawlers_pick_data = brawlers_data
        self.smart_trophy_farm = self.bot_config.get("smart_trophy_farm", "no")
        brawler_list = [brawler["brawler"] for brawler in brawlers_data]
        self.Trophy_observer = TrophyObserver(brawler_list)
        self.time_since_last_stat_change = time.time()
        self.long_press_star_drop = load_toml_as_dict("./cfg/general_config.toml")["long_press_star_drop"]
        self.window_controller = window_controller
        self.lobby_start_enabled = True
        self._awaiting_lobby_result_sync = False
        self._result_applied_for_active_match = False
        self._match_in_progress = False
        self._lobby_sync_started_at = 0.0
        self._pending_verified_result = None
        self._api_lobby_sync_attempts = 0
        self._last_api_lobby_sync_attempt_at = 0.0
        self._lobby_ocr_warmup_started = False
        self._pending_webhook_milestone_summary = None

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

    @staticmethod
    def _coerce_int(value, default=0):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _resolve_push_progress(self):
        active = self.brawlers_pick_data[0] if self.brawlers_pick_data else {}
        type_to_push = str(active.get("type", "trophies") or "trophies")
        if type_to_push not in {"trophies", "wins"}:
            type_to_push = "trophies"

        if type_to_push == "wins":
            raw_value = self.Trophy_observer.current_wins
            default_target = 300
        else:
            raw_value = self.Trophy_observer.current_trophies
            default_target = 1000

        if raw_value is None:
            raw_value = active.get(type_to_push)

        value = self._coerce_int(raw_value, 0)
        push_target = self._coerce_int(active.get("push_until"), default_target)
        return type_to_push, value, push_target

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
        if self._match_in_progress or self._awaiting_lobby_result_sync:
            return False
        if debug:
            active_name = self.brawlers_pick_data[0]['brawler'] if self.brawlers_pick_data else "unknown"
            print(f"[RESULT] mark_match_started for {active_name}")
        self._match_in_progress = True
        self._awaiting_lobby_result_sync = True
        self._result_applied_for_active_match = False
        self._lobby_sync_started_at = 0.0
        self._pending_verified_result = None
        self._api_lobby_sync_attempts = 0
        self._last_api_lobby_sync_attempt_at = 0.0
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

    def _read_lobby_trophies(self, frame):
        region = (self.lobby_config.get("lobby") or {}).get("trophy_observer")
        if frame is None or not region or len(region) != 4:
            return None
        try:
            frame = to_bgr_array(frame)
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
            except Exception as exc:
                if debug:
                    print(f"Lobby trophy OCR failed: {exc}")
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
        self._sync_active_brawler_progress()
        self._queue_webhook_milestone(current_brawler)
        save_brawler_data(self.brawlers_pick_data)
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

        self._sync_active_brawler_progress()
        self._queue_webhook_milestone(current_brawler)
        self.brawlers_pick_data[0][type_to_push] = value
        save_brawler_data(self.brawlers_pick_data)
        self._result_applied_for_active_match = True
        self._match_in_progress = False
        print(f"[RESULT] applied={applied} value={value} type={type_to_push}")
        return applied

    def set_lobby_start_enabled(self, enabled):
        self.lobby_start_enabled = enabled

    def start_game(self, data):
        print("state is lobby, starting game")
        if self._awaiting_lobby_result_sync and not self._result_applied_for_active_match:
            print("[RESULT] entering lobby fallback sync because no result was committed yet")
            synced = False
            if not self._lobby_sync_started_at:
                self._lobby_sync_started_at = time.time()
            direct_result = False
            try:
                if data is not None:
                    direct_result = find_game_result(data)
            except Exception:
                direct_result = False
            if direct_result:
                print(f"[RESULT] lobby entry direct probe recovered {direct_result}")
                self._apply_or_defer_detected_result(direct_result, source="lobby-entry")
                synced = self._result_applied_for_active_match
            if not synced:
                ocr_ready = self._is_easyocr_ready()
                has_api_settings = self.Trophy_observer.has_brawlstars_api_settings()
                if not ocr_ready:
                    self._ensure_lobby_ocr_warmup(delay_seconds=1.25)
                synced = self._sync_lobby_result(
                    data,
                    allow_ocr=ocr_ready,
                    api_timeout=0.35 if not ocr_ready else 0.75,
                )
                if not synced and not has_api_settings and not ocr_ready:
                    print("[RESULT] Brawl Stars API settings are empty; skipping slow lobby OCR wait and starting the next match")
            if debug and not synced:
                print("Lobby result sync did not resolve before next match start.")
            if synced and self._awaiting_lobby_result_sync:
                print("[RESULT] lobby fallback resolved after direct result commit; skipping OCR verification")
                self._awaiting_lobby_result_sync = False
                self._match_in_progress = False
                self._lobby_sync_started_at = 0.0
                self._pending_verified_result = None
            if not synced:
                if self._pending_verified_result:
                    print(f"[RESULT] lobby verification unavailable; falling back to pending {self._pending_verified_result}")
                    self._apply_match_result(self._pending_verified_result)
                    self._pending_verified_result = None
                self._awaiting_lobby_result_sync = False
                self._match_in_progress = False
                self._lobby_sync_started_at = 0.0
        elif self._awaiting_lobby_result_sync:
            print("[RESULT] lobby reached after direct result commit; skipping OCR fallback")
            self._awaiting_lobby_result_sync = False
            self._match_in_progress = False
            self._lobby_sync_started_at = 0.0
            self._pending_verified_result = None
        self._flush_webhook_milestone(data)
        type_of_push, value, push_current_brawler_till = self._resolve_push_progress()

        if value >= push_current_brawler_till:
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
            screenshot = self.window_controller.screenshot()
            completed_brawler = self.brawlers_pick_data[0]["brawler"]
            completed_summary = self._build_live_summary(completed_brawler)
            self._send_webhook(
                "brawler_completed",
                screenshot=screenshot,
                subject=completed_brawler,
                live_summary=completed_summary,
            )
            self.brawlers_pick_data.pop(0)
            self.Trophy_observer.change_trophies(self.brawlers_pick_data[0]['trophies'])
            self.Trophy_observer.current_wins = self._coerce_int(self.brawlers_pick_data[0].get('wins'), 0)
            self.Trophy_observer.win_streak = self._coerce_int(self.brawlers_pick_data[0].get('win_streak'), 0)
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
        print("[RESULT] Pressed Q to start a match")

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

        reward_action = find_reward_claim_action(screenshot)
        if reward_action:
            self.window_controller.click(*reward_action)
            return

        fallback_center = get_reward_claim_button_center(screenshot)
        if fallback_center:
            self.window_controller.click(*fallback_center)
            return

        self.window_controller.press_continue()

    def end_game(self, frame=None, known_result=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()

        found_game_result = False
        current_state = get_state(screenshot, allow_reward_ocr=True)
        print(f"[RESULT] end_game entered known_result={known_result} current_state={current_state}")
        if known_result in {"victory", "defeat", "draw", "1st", "2nd", "3rd", "4th"}:
            found_game_result = known_result if self._apply_or_defer_detected_result(known_result, source="known-result") else False
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
                    print(f"[RESULT] end_game state result probe -> {state_result}")
                    found_game_result = state_result if self._apply_or_defer_detected_result(state_result, source="state-probe") else False
                else:
                    detected_result = self.Trophy_observer.find_game_result(
                        screenshot,
                        current_brawler=self.brawlers_pick_data[0]['brawler'],
                        game_result=state_result,
                    )
                    if detected_result:
                        found_game_result = (
                            detected_result
                            if self._apply_or_defer_detected_result(detected_result, source="ocr-fallback")
                            else False
                        )
                        if self._result_applied_for_active_match:
                            self.time_since_last_stat_change = time.time()
                            self._sync_active_brawler_progress()
                            save_brawler_data(self.brawlers_pick_data)
                            print(f"[RESULT] OCR fallback committed {found_game_result}")
                        elif detected_result == "draw":
                            print("[RESULT] OCR fallback detected draw; awaiting lobby verification")
                type_to_push, value, push_current_brawler_till = self._resolve_push_progress()

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
            self.window_controller.press_continue()
            if debug: print("Game has ended, pressing Q")
            time.sleep(1)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot, allow_reward_ocr=True)
            end_attempts += 1
        if end_attempts >= max_end_attempts:
            print("End game screen stuck for too long, forcing continue")
        print(f"[RESULT] end_game exiting current_state={current_state} found={found_game_result}")
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
            print(f"[RESULT] do_state -> end (known_result={known_result})")
            self.state_handlers[state](data, known_result)
            return
        if data is not None:
            self.state_handlers[state](data)
            return
        self.state_handlers[state]()
