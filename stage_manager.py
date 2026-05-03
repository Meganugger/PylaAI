import os.path
import sys

import time
import threading

import cv2
import numpy as np

from adaptive_brain import AdaptiveBrain

from state_finder.main import (
    get_state,
    find_game_result,
    find_reward_claim_action,
    get_reward_claim_button_center,
    is_lobby_play_button_visible,
)
from trophy_observer import TrophyObserver
from utils import find_template_center, load_toml_as_dict, notify_user, has_notification_webhook, \
    save_brawler_data, reader, to_bgr_array, load_brawl_stars_api_config, fetch_brawl_stars_player, \
    normalize_brawler_name

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
            'trophy_reward': self.claim_reward,
            'prestige_reward': self.handle_prestige_reward,
            'player_title_reward': self.handle_player_title_reward,
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
        # ── Adaptive brain ──────────────────────────────────────
        bot_config = self.bot_config
        adaptive_enabled = str(bot_config.get("adaptive_brain_enabled", "yes")).lower() in ("yes", "true", "1")
        adaptive_window = int(bot_config.get("adaptive_brain_window", 20))
        self.adaptive_brain = AdaptiveBrain(enabled=adaptive_enabled, window_size=adaptive_window)
        print(self.adaptive_brain.summary())
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
                # Avoid spiking CPU by waking EasyOCR during a live match or
                # after we've already moved on from the unresolved lobby state.
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

    def _reset_lobby_result_sync_state(self):
        self._awaiting_lobby_result_sync = False
        self._match_in_progress = False
        self._lobby_sync_started_at = 0.0
        self._pending_verified_result = None
        self._api_lobby_sync_attempts = 0
        self._last_api_lobby_sync_attempt_at = 0.0
        self._clear_end_transition()

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

    def is_post_match_resolution_pending(self, now=None):
        if now is None:
            now = time.time()
        return (
            (self._awaiting_lobby_result_sync and not self._match_in_progress)
            or self.has_recent_end_transition(now)
        )

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

    def _try_press_lobby_start(self):
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
            print("[RESULT] Pressed Q to start a match")
        else:
            print(f"[RESULT] Lobby still visible; retrying Q press ({self._start_press_attempts})")
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
        now = time.time()
        recent_start_press = bool(self._last_start_press_at and (now - self._last_start_press_at) <= 12.0)
        sync_started_at = max(
            self._lobby_sync_started_at,
            self._last_result_applied_at,
            self._end_transition_started_at,
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
        if self._match_in_progress:
            return False
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
        self._clear_end_transition()
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

        # ── Feed result to adaptive brain for parameter tuning ──
        try:
            self.adaptive_brain.record_result(game_result, brawler=current_brawler)
        except Exception as exc:
            print(f"[ADAPTIVE] Could not record result: {exc}")

        type_to_push, value, _ = self._resolve_push_progress()
        self._result_applied_for_active_match = True
        self._last_result_applied_at = time.time()
        self._match_in_progress = False
        print(f"[RESULT] applied={applied} predicted_value={value} type={type_to_push}")
        return applied

    def set_lobby_start_enabled(self, enabled):
        self.lobby_start_enabled = enabled

    def start_game(self, data):
        print("state is lobby, starting game")
        if self._awaiting_lobby_result_sync:
            synced = False
            if not self._lobby_sync_started_at:
                self._lobby_sync_started_at = time.time()
            elapsed = time.time() - self._lobby_sync_started_at
            finalized_without_verification = False

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
                    # Hard max timeout: never wait longer than _lobby_sync_max_timeout
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
            completed_brawler = self.brawlers_pick_data[0]["brawler"]
            completed_summary = self._build_live_summary(completed_brawler)
            screenshot = self.window_controller.screenshot()
            if not self._prepare_next_push_all_brawler(push_current_brawler_till, type_of_push):
                print("Brawler reached required trophies/wins. No remaining brawlers are below the Push All target.")
                self._send_webhook(
                    "all_targets_completed",
                    screenshot=screenshot,
                    current_brawler=completed_brawler,
                )
                print("Bot stopping: all targets completed.")
                self.window_controller.keys_up(list("wasd"))
                self.window_controller.close()
                sys.exit(0)

            next_data = self.brawlers_pick_data[0]
            next_brawler_name = next_data['brawler']
            selection_confirmed = True

            if next_data.get("automatically_pick", True):
                if debug: print("Picking next automatically picked brawler")
                current_state = get_state(screenshot)
                if current_state != "lobby":
                    print("Trying to reach the lobby to switch brawler")

                max_attempts = 30
                attempts = 0
                while current_state != "lobby" and attempts < max_attempts:
                    self.window_controller.press_key("Q")
                    if debug: print("Pressed Q to return to lobby")
                    time.sleep(1)
                    screenshot = self.window_controller.screenshot()
                    current_state = get_state(screenshot)
                    attempts += 1
                if attempts >= max_attempts:
                    print("Failed to reach lobby after max attempts")
                    selection_confirmed = False
                else:
                    selection_method = next_data.get("selection_method", "named_brawler")
                    if selection_method == "lowest_trophies":
                        selection_confirmed = bool(self.Lobby_automation.select_lowest_trophy_brawler())
                    else:
                        selection_confirmed = bool(self.Lobby_automation.select_brawler(next_brawler_name))
                if not selection_confirmed:
                    print(
                        f"WARNING: Could not confirm brawler switch to '{next_brawler_name}'. "
                        "Keeping the current brawler active and blocking lobby start until retry."
                    )
                    self._delay_lobby_start(5.0, "waiting for brawler auto-select")
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

        self._try_press_lobby_start()

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

    def handle_prestige_reward(self, frame=None):
        """Handle the prestige reward screen and advance the queue if needed."""
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
        print(f"[PRESTIGE] Treating {current_brawler} as completed.")
        self.brawlers_pick_data[0]["trophies"] = max(
            1000,
            self._coerce_int(self.brawlers_pick_data[0].get("trophies"), 0),
        )
        self.brawlers_pick_data[0]["push_until"] = max(
            1000,
            self._coerce_int(self.brawlers_pick_data[0].get("push_until"), 1000),
        )

        if len(self.brawlers_pick_data) <= 1:
            print("[PRESTIGE] No next brawler queued. Saving and continuing.")
            save_brawler_data(self.brawlers_pick_data)
            self.window_controller.press_key("Q")
            return

        next_data = self.brawlers_pick_data[1]

        screenshot = self.window_controller.screenshot()
        current_state = get_state(screenshot)
        attempts = 0
        while current_state != "lobby" and attempts < 30:
            self.window_controller.press_key("Q")
            time.sleep(1.0)
            screenshot = self.window_controller.screenshot()
            current_state = get_state(screenshot)
            attempts += 1

        if current_state == "lobby":
            next_name = next_data.get("brawler")
            if next_name and next_data.get("automatically_pick"):
                if not self.Lobby_automation.select_brawler(next_name):
                    print(f"[PRESTIGE] Could not confirm switch to {next_name}; keeping current queue active.")
                    self._delay_lobby_start(5.0, "waiting for prestige brawler auto-select")
                    return
                print(f"[PRESTIGE] Switched to {next_name}.")
        else:
            print("[PRESTIGE] Could not reach lobby after prestige reward.")
            self._delay_lobby_start(5.0, "waiting for prestige lobby recovery")
            return

        self.brawlers_pick_data.pop(0)
        self.Trophy_observer.change_trophies(next_data.get("trophies", 0))
        self.Trophy_observer.current_wins = self._coerce_int(next_data.get("wins"), 0)
        self.Trophy_observer.win_streak = self._coerce_int(next_data.get("win_streak"), 0)
        save_brawler_data(self.brawlers_pick_data)

    def end_game(self, frame=None, known_result=None):
        screenshot = frame if frame is not None else self.window_controller.screenshot()
        now = time.time()
        current_state = (
            f"end_{known_result}"
            if known_result in {"victory", "defeat", "draw", "1st", "2nd", "3rd", "4th"}
            else get_state(
                screenshot,
                allow_reward_ocr=self.should_hold_match_probe(now) and self._is_easyocr_ready(),
            )
        )
        print(f"[RESULT] end_game entered known_result={known_result} current_state={current_state}")

        state_result = None
        if isinstance(current_state, str) and current_state.startswith("end_"):
            state_result = current_state.split("_", 1)[1]
        result_to_apply = known_result if known_result in {"victory", "defeat", "draw", "1st", "2nd", "3rd", "4th"} else state_result

        if result_to_apply:
            self._begin_end_transition(result_to_apply, now)
            print(f"[RESULT] end_game state result probe -> {result_to_apply}")
            found_game_result = (
                result_to_apply
                if self._apply_or_defer_detected_result(result_to_apply, source="known-result" if known_result else "state-probe")
                else False
            )
        else:
            found_game_result = False
            should_probe_result = time.time() - self.time_since_last_stat_change > 10
            if should_probe_result:
                detected_result = self.Trophy_observer.find_game_result(
                    screenshot,
                    current_brawler=self.brawlers_pick_data[0]['brawler'],
                    game_result=None,
                )
                if detected_result:
                    self._begin_end_transition(detected_result, now)
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

        if self._result_applied_for_active_match:
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

        if not self._is_endish_state(current_state):
            print(f"[RESULT] end_game exiting current_state={current_state} found={found_game_result}")
            if debug:
                print("Game has ended", current_state)
            return

        if (now - self._end_transition_last_action_at) < self._end_transition_action_interval:
            return

        if current_state in {"reward_claim", "trophy_reward"}:
            self.claim_reward(screenshot)
        elif current_state == "player_title_reward":
            self.handle_player_title_reward(screenshot)
        elif current_state == "prestige_reward":
            self.handle_prestige_reward(screenshot)
        elif current_state == "star_drop":
            self.click_star_drop()
        else:
            self.window_controller.press_key("Q")
            if debug:
                print("Game has ended, pressing Q")
        self._end_transition_last_action_at = now
        print(f"[RESULT] end_game exiting current_state={current_state} found={found_game_result}")
        if debug:
            print("Game has ended", current_state)

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
