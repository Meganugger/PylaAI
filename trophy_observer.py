import copy
import os
import time
from difflib import SequenceMatcher

import numpy as np
import requests

from utils import load_toml_as_dict, save_dict_as_toml, api_base_url, reader


class TrophyObserver:
    def __init__(self, brawler_list):
        self.history_file = "./cfg/match_history.toml"
        self.current_trophies = None
        self.current_wins = None
        self.match_history = self.load_history(brawler_list)
        total_history = self.match_history.get("total", {}) if isinstance(self.match_history.get("total"), dict) else {}
        self.match_history["total"] = {
            "defeat": self._safe_int(total_history.get("defeat", 0), 0),
            "victory": self._safe_int(total_history.get("victory", 0), 0),
            "draw": self._safe_int(total_history.get("draw", 0), 0),
        }
        self._session_start_history = copy.deepcopy(self.match_history)
        self.sent_match_history = {
            brawler: {
                "defeat": self.match_history[brawler]["defeat"],
                "victory": self.match_history[brawler]["victory"],
                "draw": 0,
            }
            for brawler in brawler_list
        }
        self.win_streak = 0
        self.match_counter = 0
        self.trophy_lose_ranges = [
            (49, 0), (299, 1), (599, 2), (799, 3), (999, 4), (1099, 5),
            (1199, 6), (1299, 7), (1499, 8), (1799, 9), (3999, 10),
            (float("inf"), 15),
        ]
        self.trophy_win_ranges = [
            (1999, 10), (2499, 8), (2799, 6), (2999, 4), (3099, 2),
            (float("inf"), 1),
        ]
        self.crop_region = load_toml_as_dict("./cfg/lobby_config.toml")["lobby"]["trophy_observer"]
        self.trophies_multiplier = int(load_toml_as_dict("./cfg/general_config.toml")["trophies_multiplier"])

        self.session_started_at = time.time()
        self.session_stats = {
            "victories": 0,
            "defeats": 0,
            "draws": 0,
            "total_matches": 0,
            "total_kills": 0,
            "total_assists": 0,
            "total_damage": 0,
            "total_deaths": 0,
            "last_match_kills": 0,
            "last_match_assists": 0,
            "last_match_damage": 0,
            "last_match_deaths": 0,
        }
        self.brawler_stats = {}
        self._current_match_stats = {
            "brawler": None,
            "kills": 0,
            "assists": 0,
            "damage": 0,
            "deaths": 0,
        }
        self._session_start_trophies = {}
        self._session_end_trophies = {}
        self._last_game_result = None
        self._corrections_log = []
        self._lobby_trophy_verified = False
        self._active_match_brawler = None
        self._active_match_start_trophies = None
        self._active_match_start_streak = 0
        self.last_match_result = None
        self.last_match_trophy_delta = 0
        self.last_match_predicted_trophy_delta = 0
        self.last_match_trophy_adjustment = 0
        self.last_match_streak_bonus = 0
        self.last_match_underdog_bonus = 0
        self.last_match_trophies_verified = False
        self.last_match_start_trophies = 0
        self.last_match_end_trophies = 0
        self.history_revision = 0
        self._api_player_brawlers_cache = None
        self._api_player_brawlers_cache_time = 0.0
        self._last_api_lookup_error = ""
        self._last_api_lookup_error_time = 0.0

        for brawler in brawler_list:
            self._ensure_brawler_stats(brawler)

        try:
            if not os.path.exists(self.history_file) or os.path.getsize(self.history_file) == 0:
                self.save_history()
        except OSError:
            pass

    @staticmethod
    def rework_game_result(res_string):
        res_string = res_string.lower()
        if res_string in ["victory", "defeat", "draw"]:
            return res_string, 1.0

        ratios = {
            "victory": SequenceMatcher(None, res_string, "victory").ratio(),
            "defeat": SequenceMatcher(None, res_string, "defeat").ratio(),
            "draw": SequenceMatcher(None, res_string, "draw").ratio(),
        }
        highest_ratio_string = max(ratios, key=ratios.get)
        return highest_ratio_string, ratios[highest_ratio_string]

    @staticmethod
    def _safe_int(value, default=0):
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _slug_name(value):
        return "".join(ch for ch in str(value or "").lower() if ch.isalnum())

    def _log_api_lookup_issue(self, message):
        now = time.time()
        if message != self._last_api_lookup_error or now - self._last_api_lookup_error_time >= 5.0:
            print(message)
            self._last_api_lookup_error = message
            self._last_api_lookup_error_time = now

    @staticmethod
    def _load_brawlstars_api_settings():
        general_config = load_toml_as_dict("./cfg/general_config.toml")
        api_key = str(general_config.get("brawlstars_api_key", "") or "").strip()
        player_tag = str(general_config.get("brawlstars_player_tag", "") or "").strip().upper().replace("#", "")
        return api_key, player_tag

    def has_brawlstars_api_settings(self):
        api_key, player_tag = self._load_brawlstars_api_settings()
        return bool(api_key and player_tag)

    def _fetch_player_brawlers_from_brawlstars_api(self, force=False, timeout=3.0):
        api_key, player_tag = self._load_brawlstars_api_settings()
        if not api_key or not player_tag:
            return None

        now = time.time()
        if (
            not force
            and isinstance(self._api_player_brawlers_cache, list)
            and now - self._api_player_brawlers_cache_time < 2.5
        ):
            return self._api_player_brawlers_cache

        try:
            response = requests.get(
                f"https://api.brawlstars.com/v1/players/%23{player_tag}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=timeout,
            )
            if response.status_code == 403:
                raise ValueError("the Brawl Stars API key was rejected")
            if response.status_code == 404:
                raise ValueError("the player tag was not found")
            response.raise_for_status()
            payload = response.json()
        except (requests.exceptions.RequestException, ValueError) as exc:
            self._log_api_lookup_issue(f"[RESULT] API fallback unavailable: {exc}")
            return None

        brawlers = payload.get("brawlers", [])
        if not isinstance(brawlers, list):
            self._log_api_lookup_issue("[RESULT] API fallback unavailable: invalid brawler payload")
            return None

        self._api_player_brawlers_cache = brawlers
        self._api_player_brawlers_cache_time = now
        self._last_api_lookup_error = ""
        self._last_api_lookup_error_time = 0.0
        return brawlers

    def fetch_brawler_trophies_from_brawlstars_api(self, current_brawler, force=False, timeout=3.0):
        target_slug = self._slug_name(current_brawler)
        if not target_slug:
            return None

        brawlers = self._fetch_player_brawlers_from_brawlstars_api(force=force, timeout=timeout)
        if not isinstance(brawlers, list):
            return None

        for entry in brawlers:
            if self._slug_name(entry.get("name", "")) != target_slug:
                continue
            trophies = self._safe_int(entry.get("trophies"), None)
            if trophies is not None:
                return trophies
            break

        self._log_api_lookup_issue(f"[RESULT] API fallback could not find trophies for {current_brawler}")
        return None

    def _ensure_brawler_stats(self, brawler):
        key = str(brawler).lower()
        if key not in self.brawler_stats:
            self.brawler_stats[key] = {
                "matches": 0,
                "total_kills": 0,
                "total_assists": 0,
                "total_damage": 0,
                "total_deaths": 0,
                "avg_kills": 0.0,
                "avg_assists": 0.0,
                "avg_damage": 0.0,
                "avg_deaths": 0.0,
            }
        return self.brawler_stats[key]

    def start_session_brawler(self, brawler, starting_trophies):
        if not brawler:
            return
        key = str(brawler).lower()
        self._ensure_brawler_stats(key)
        if key not in self._session_start_trophies:
            try:
                start_value = int(round(float(starting_trophies)))
            except (TypeError, ValueError):
                start_value = 0
            self._session_start_trophies[key] = start_value
        self._session_end_trophies[key] = self._session_start_trophies.get(key, 0)

    def begin_match(self, current_brawler):
        if not current_brawler:
            return
        key = str(current_brawler).lower()
        current_trophies = self._safe_int(self.current_trophies, 0)
        self.start_session_brawler(key, current_trophies)
        if self._active_match_brawler == key and self._active_match_start_trophies is not None:
            return
        self._active_match_brawler = key
        self._active_match_start_trophies = current_trophies
        self._active_match_start_streak = self._safe_int(self.win_streak, 0)
        self._lobby_trophy_verified = False

    def get_active_match_start_trophies(self, current_brawler=None):
        if current_brawler and self._active_match_brawler:
            if str(current_brawler).lower() != self._active_match_brawler:
                return None
        return self._active_match_start_trophies

    def _set_last_match_trophy_summary(
        self,
        game_result,
        start_trophies,
        end_trophies,
        predicted_delta,
        streak_bonus=0,
        verified=False,
    ):
        start_value = self._safe_int(start_trophies, 0)
        end_value = self._safe_int(end_trophies, 0)
        predicted_value = self._safe_int(predicted_delta, 0)
        actual_delta = end_value - start_value
        adjustment = actual_delta - predicted_value

        self.last_match_result = game_result
        self.last_match_start_trophies = start_value
        self.last_match_end_trophies = end_value
        self.last_match_predicted_trophy_delta = predicted_value
        self.last_match_trophy_delta = actual_delta
        self.last_match_trophy_adjustment = adjustment
        self.last_match_streak_bonus = max(0, self._safe_int(streak_bonus, 0)) if game_result == "victory" else 0
        self.last_match_underdog_bonus = adjustment if game_result == "victory" and adjustment > 0 else 0
        self.last_match_trophies_verified = bool(verified)

    @staticmethod
    def _session_result_key(game_result):
        mapping = {
            "victory": "victories",
            "defeat": "defeats",
            "draw": "draws",
        }
        return mapping.get(str(game_result or "").lower())

    def _loss_decrement_for(self, trophies):
        trophy_value = self._safe_int(trophies, 0)
        for max_trophies, loss in self.trophy_lose_ranges:
            if float(trophy_value) <= float(max_trophies):
                return self._safe_int(loss, 0)
        return 0

    def _win_increment_for(self, trophies, start_streak=0):
        trophy_value = self._safe_int(trophies, 0)
        streak_bonus = max(0, min(self._safe_int(start_streak, 0), 5))
        for max_trophies, gain in self.trophy_win_ranges:
            if float(trophy_value) <= float(max_trophies):
                return self._safe_int(gain, 0) * self.trophies_multiplier + streak_bonus
        return streak_bonus

    def _reclassify_verified_result(self, current_brawler, corrected_result, start_value, verified_value):
        previous_result = str(self.last_match_result or "").lower()
        corrected_result = str(corrected_result or "").lower()
        if not current_brawler or not previous_result or previous_result == corrected_result:
            return False

        key = str(current_brawler).lower()
        if key not in self.match_history:
            self.match_history[key] = {"defeat": 0, "victory": 0, "draw": 0}
        if "total" not in self.match_history:
            self.match_history["total"] = {"defeat": 0, "victory": 0, "draw": 0}

        if previous_result in self.match_history[key] and self.match_history[key][previous_result] > 0:
            self.match_history[key][previous_result] -= 1
        if previous_result in self.match_history["total"] and self.match_history["total"][previous_result] > 0:
            self.match_history["total"][previous_result] -= 1
        self.match_history[key][corrected_result] = self._safe_int(self.match_history[key].get(corrected_result, 0), 0) + 1
        self.match_history["total"][corrected_result] = self._safe_int(self.match_history["total"].get(corrected_result, 0), 0) + 1

        previous_session_key = self._session_result_key(previous_result)
        corrected_session_key = self._session_result_key(corrected_result)
        if previous_session_key:
            self.session_stats[previous_session_key] = max(0, self._safe_int(self.session_stats.get(previous_session_key, 0), 0) - 1)
        if corrected_session_key:
            self.session_stats[corrected_session_key] = self._safe_int(self.session_stats.get(corrected_session_key, 0), 0) + 1

        start_streak = self._safe_int(self._active_match_start_streak, 0)
        current_wins = self._safe_int(self.current_wins, 0)
        if previous_result == "victory" and corrected_result != "victory":
            current_wins = max(0, current_wins - 1)
        elif corrected_result == "victory" and previous_result != "victory":
            current_wins += 1
        self.current_wins = current_wins

        if corrected_result == "victory":
            self.win_streak = start_streak + 1
            predicted_delta = self._win_increment_for(start_value, start_streak)
            streak_bonus = max(0, min(start_streak, 5))
        elif corrected_result == "defeat":
            self.win_streak = 0
            predicted_delta = -self._loss_decrement_for(start_value)
            streak_bonus = 0
        else:
            self.win_streak = start_streak
            predicted_delta = 0
            streak_bonus = 0

        self.last_match_result = corrected_result
        self._last_game_result = corrected_result
        self._set_last_match_trophy_summary(
            corrected_result,
            start_value,
            verified_value,
            predicted_delta,
            streak_bonus=streak_bonus,
            verified=True,
        )
        self.save_history()
        self.history_revision += 1
        print(f"[RESULT] corrected result {previous_result} -> {corrected_result} for {key}")
        return True

    def reconcile_verified_trophies(self, current_brawler, verified_trophies):
        if not current_brawler:
            return False

        key = str(current_brawler).lower()
        verified_value = self._safe_int(verified_trophies, self._safe_int(self.current_trophies, 0))
        start_value = self.get_active_match_start_trophies(current_brawler)
        if start_value is None:
            start_value = self.last_match_start_trophies

        corrected_result = str(self.last_match_result or "").lower()
        if verified_value > start_value:
            corrected_result = "victory"
        elif verified_value < start_value:
            corrected_result = "defeat"
        elif not corrected_result:
            corrected_result = "draw"

        self._reclassify_verified_result(current_brawler, corrected_result, start_value, verified_value)

        previous_value = self.current_trophies
        if previous_value != verified_value:
            print(f"Trophies changed from {previous_value} to {verified_value}")
        self.current_trophies = verified_value
        self._session_end_trophies[key] = verified_value
        self._set_last_match_trophy_summary(
            corrected_result or self.last_match_result,
            start_value,
            verified_value,
            self.last_match_predicted_trophy_delta,
            streak_bonus=self.last_match_streak_bonus,
            verified=True,
        )
        if self.last_match_trophy_adjustment != 0:
            self._corrections_log.append({
                "brawler": key,
                "result": self.last_match_result,
                "adjustment": self.last_match_trophy_adjustment,
                "verified_trophies": verified_value,
                "timestamp": time.time(),
            })
        self._lobby_trophy_verified = True
        self._active_match_brawler = None
        self._active_match_start_trophies = None
        self._active_match_start_streak = 0
        return True

    def update_live_match_stats(self, current_brawler, kills=0, assists=0, damage=0, deaths=0):
        if not current_brawler:
            return
        key = str(current_brawler).lower()
        if key != self._current_match_stats["brawler"]:
            self._current_match_stats = {
                "brawler": key,
                "kills": 0,
                "assists": 0,
                "damage": 0,
                "deaths": 0,
            }
        self.start_session_brawler(key, self.current_trophies if self.current_trophies is not None else 0)
        self._current_match_stats["kills"] = max(self._current_match_stats["kills"], int(kills or 0))
        self._current_match_stats["assists"] = max(self._current_match_stats["assists"], int(assists or 0))
        self._current_match_stats["damage"] = max(self._current_match_stats["damage"], int(damage or 0))
        self._current_match_stats["deaths"] = max(self._current_match_stats["deaths"], int(deaths or 0))

    def _finalize_current_match(self, current_brawler, game_result):
        key = str(current_brawler).lower()
        pending = self._current_match_stats if self._current_match_stats["brawler"] == key else {
            "kills": 0,
            "assists": 0,
            "damage": 0,
            "deaths": 0,
        }

        stats = self._ensure_brawler_stats(key)
        kills = int(pending.get("kills", 0) or 0)
        assists = int(pending.get("assists", 0) or 0)
        damage = int(pending.get("damage", 0) or 0)
        deaths = int(pending.get("deaths", 0) or 0)

        self.session_stats["last_match_kills"] = kills
        self.session_stats["last_match_assists"] = assists
        self.session_stats["last_match_damage"] = damage
        self.session_stats["last_match_deaths"] = deaths

        self.session_stats["total_matches"] += 1
        self.session_stats["total_kills"] += kills
        self.session_stats["total_assists"] += assists
        self.session_stats["total_damage"] += damage
        self.session_stats["total_deaths"] += deaths

        if game_result == "victory":
            self.session_stats["victories"] += 1
        elif game_result == "defeat":
            self.session_stats["defeats"] += 1
        elif game_result == "draw":
            self.session_stats["draws"] += 1

        stats["matches"] += 1
        stats["total_kills"] += kills
        stats["total_assists"] += assists
        stats["total_damage"] += damage
        stats["total_deaths"] += deaths
        match_count = max(1, stats["matches"])
        stats["avg_kills"] = stats["total_kills"] / match_count
        stats["avg_assists"] = stats["total_assists"] / match_count
        stats["avg_damage"] = stats["total_damage"] / match_count
        stats["avg_deaths"] = stats["total_deaths"] / match_count

        current_trophies = self.current_trophies if self.current_trophies is not None else 0
        try:
            self._session_end_trophies[key] = int(round(float(current_trophies)))
        except (TypeError, ValueError):
            self._session_end_trophies[key] = self._session_end_trophies.get(key, 0)

        self._current_match_stats = {
            "brawler": None,
            "kills": 0,
            "assists": 0,
            "damage": 0,
            "deaths": 0,
        }

    def win_streak_gain(self):
        return min(self.win_streak - 1, 5)

    def calc_lost_decrement(self):
        for max_trophies, loss in self.trophy_lose_ranges:
            if float(self.current_trophies) <= float(max_trophies):
                return loss

    def calc_win_increment(self):
        for max_trophies, gain in self.trophy_win_ranges:
            if float(self.current_trophies) <= float(max_trophies):
                return gain * self.trophies_multiplier + self.win_streak_gain()

    def load_history(self, brawler_list):
        if os.path.exists(self.history_file):
            loaded_data = load_toml_as_dict(self.history_file)
        else:
            loaded_data = {}

        for brawler in brawler_list:
            if brawler not in loaded_data:
                loaded_data[brawler] = {"defeat": 0, "victory": 0, "draw": 0}

        if "total" not in loaded_data:
            loaded_data["total"] = {"defeat": 0, "victory": 0, "draw": 0}

        return loaded_data

    def save_history(self):
        save_dict_as_toml(self.match_history, self.history_file)

    def add_trophies(self, game_result, current_brawler):
        key = str(current_brawler).lower()
        if current_brawler not in self.sent_match_history:
            self.sent_match_history[current_brawler] = {"defeat": 0, "victory": 0, "draw": 0}
        if current_brawler not in self.match_history:
            self.match_history[current_brawler] = {"defeat": 0, "victory": 0, "draw": 0}

        self.begin_match(key)

        print(f"[RESULT] TrophyObserver.add_trophies({game_result}) win_streak={self.win_streak}")
        old = self._safe_int(self.current_trophies, 0)
        predicted_delta = 0
        streak_bonus = 0
        self._lobby_trophy_verified = False
        if game_result == "victory":
            self.win_streak += 1
            predicted_delta = self._safe_int(self.calc_win_increment(), 0)
            streak_bonus = max(0, self.win_streak_gain())
            self.current_trophies = old + predicted_delta
        elif game_result == "defeat":
            predicted_delta = -self._safe_int(self.calc_lost_decrement(), 0)
            self.win_streak = 0
            self.current_trophies = old + predicted_delta
        elif game_result == "draw":
            self.current_trophies = old
            print("Nothing changed. Draw detected")
        else:
            print("Catastrophic failure")

        match_start_trophies = self.get_active_match_start_trophies(current_brawler)
        if match_start_trophies is None:
            match_start_trophies = old
        self._set_last_match_trophy_summary(
            game_result,
            match_start_trophies,
            self.current_trophies,
            predicted_delta,
            streak_bonus=streak_bonus,
            verified=False,
        )
        print(f"[RESULT] trophies {old} -> {self.current_trophies}")
        print(f"[RESULT] current wins before increment: {self.current_wins}")
        self.match_history[current_brawler][game_result] += 1
        self.match_history["total"][game_result] += 1
        self._last_game_result = game_result

        self.match_counter += 1
        if self.match_counter % 4 == 0:
            self.send_results_to_api()

        self.save_history()
        self.history_revision += 1
        self._finalize_current_match(current_brawler, game_result)
        return True

    def add_win(self, game_result):
        if game_result == "victory":
            self.current_wins = self._safe_int(self.current_wins, 0) + 1
            print(f"[RESULT] current_wins incremented to {self.current_wins}")

    def find_game_result(self, screenshot, current_brawler, game_result=None):
        if not game_result:
            if isinstance(screenshot, np.ndarray):
                x1, y1, x2, y2 = self.crop_region
                array_screenshot = screenshot[y1:y2, x1:x2]
            else:
                screenshot = screenshot.crop(self.crop_region)
                array_screenshot = np.array(screenshot)
            result = reader.readtext(array_screenshot)

            if len(result) == 0:
                return False

            _, text, _conf = result[0]
            game_result, ratio = self.rework_game_result(text)
            if ratio < 0.55:
                if ratio > 0:
                    print("Couldn't find game result", game_result, ratio)
                return False

        if game_result not in {"victory", "defeat", "draw"}:
            return False
        return game_result

    def change_trophies(self, new):
        print(f"Trophies changed from {self.current_trophies} to {new}")
        self.current_trophies = self._safe_int(new, self.current_trophies)

    def read_end_screen_stats(self, screenshot, current_brawler, wr=1.0, hr=1.0):
        self.start_session_brawler(current_brawler, self.current_trophies if self.current_trophies is not None else 0)
        return {
            "kills": self.session_stats.get("last_match_kills", 0),
            "assists": self.session_stats.get("last_match_assists", 0),
            "damage": self.session_stats.get("last_match_damage", 0),
            "deaths": self.session_stats.get("last_match_deaths", 0),
        }

    def verify_game_result_consistency(self, screenshot, current_brawler, wr=1.0, hr=1.0):
        return False

    def verify_trophy_delta(self, screenshot, wr=1.0, hr=1.0):
        return False

    def verify_lobby_trophies(self, screenshot, wr=1.0, hr=1.0):
        self._lobby_trophy_verified = True
        return self.current_trophies

    def _history_delta(self, brawler, key):
        start = self._session_start_history.get(brawler, {}).get(key, 0)
        current = self.match_history.get(brawler, {}).get(key, 0)
        return max(0, current - start)

    def get_session_summary(self):
        duration_seconds = max(1, int(time.time() - self.session_started_at))
        hours, rem = divmod(duration_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        victories = self.session_stats.get("victories", 0)
        defeats = self.session_stats.get("defeats", 0)
        draws = self.session_stats.get("draws", 0)
        total_matches = victories + defeats + draws
        winrate = (victories / total_matches * 100) if total_matches > 0 else 0.0

        brawlers = []
        tracked_brawlers = sorted({
            *(str(b).lower() for b in self._session_start_trophies.keys()),
            *(str(b).lower() for b in self._session_end_trophies.keys()),
            *(str(b).lower() for b in self.brawler_stats.keys()),
        })
        for brawler in tracked_brawlers:
            stats = self.brawler_stats.get(brawler, {})
            b_v = self._history_delta(brawler, "victory")
            b_d = self._history_delta(brawler, "defeat")
            b_dr = self._history_delta(brawler, "draw")
            trophy_start = self._session_start_trophies.get(brawler, self._session_end_trophies.get(brawler, 0))
            trophy_end = self._session_end_trophies.get(brawler, trophy_start)
            trophy_delta = trophy_end - trophy_start
            b_matches = b_v + b_d + b_dr
            if b_matches == 0 and trophy_delta == 0 and stats.get("matches", 0) == 0:
                continue
            b_winrate = (b_v / b_matches * 100) if b_matches > 0 else 0.0
            brawlers.append({
                "name": brawler,
                "victories": b_v,
                "defeats": b_d,
                "draws": b_dr,
                "winrate": b_winrate,
                "trophy_start": trophy_start,
                "trophy_end": trophy_end,
                "trophy_delta": trophy_delta,
                "kills": int(stats.get("total_kills", 0) or 0),
                "assists": int(stats.get("total_assists", 0) or 0),
                "damage": int(stats.get("total_damage", 0) or 0),
                "deaths": int(stats.get("total_deaths", 0) or 0),
            })

        net_trophies = sum(item["trophy_delta"] for item in brawlers)
        return {
            "duration": duration,
            "total_matches": total_matches,
            "victories": victories,
            "defeats": defeats,
            "draws": draws,
            "winrate": winrate,
            "net_trophies": net_trophies,
            "total_kills": int(self.session_stats.get("total_kills", 0) or 0),
            "total_assists": int(self.session_stats.get("total_assists", 0) or 0),
            "total_damage": int(self.session_stats.get("total_damage", 0) or 0),
            "brawlers": brawlers,
        }

    def send_results_to_api(self):
        data = {}
        for brawler, stats in self.match_history.items():
            if brawler != "total":
                if brawler not in self.sent_match_history:
                    self.sent_match_history[brawler] = {"defeat": 0, "victory": 0, "draw": 0}
                new_stats = {
                    "wins": stats["victory"] - self.sent_match_history[brawler]["victory"],
                    "defeats": stats["defeat"] - self.sent_match_history[brawler]["defeat"],
                    "draws": 0,
                }
                if any(new_stats.values()):
                    data[brawler] = new_stats

        if not data:
            return

        if api_base_url != "localhost":
            try:
                response = requests.post(f"https://{api_base_url}/api/brawlers", json=data)
                if response.status_code == 200:
                    print("Results successfully sent to API")
                    for brawler, stats in self.match_history.items():
                        if brawler != "total":
                            self.sent_match_history[brawler]["victory"] = stats["victory"]
                            self.sent_match_history[brawler]["defeat"] = stats["defeat"]
                            self.sent_match_history[brawler]["draw"] = 0
                else:
                    print(f"Failed to send results to API. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Error sending results to API: {e}")
