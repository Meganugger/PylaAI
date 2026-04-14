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
        self.match_history["total"] = {"defeat": 0, "victory": 0, "draw": 0}
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

        self.start_session_brawler(key, self.current_trophies if self.current_trophies is not None else 0)

        print(f"Found game result!: {game_result} win streak: {self.win_streak}")
        old = self.current_trophies
        if game_result == "victory":
            self.win_streak += 1
            self.current_trophies += self.calc_win_increment()
        elif game_result == "defeat":
            self.win_streak = 0
            self.current_trophies -= self.calc_lost_decrement()
        elif game_result == "draw":
            print("Nothing changed. Draw detected")
        else:
            print("Catastrophic failure")

        print(f"Trophies : {old} -> {self.current_trophies}")
        print("Current wins:", self.current_wins)
        self.match_history[current_brawler][game_result] += 1
        self.match_history["total"][game_result] += 1
        self._last_game_result = game_result

        self.match_counter += 1
        if self.match_counter % 4 == 0:
            self.send_results_to_api()

        self.save_history()
        self._finalize_current_match(current_brawler, game_result)
        return True

    def add_win(self, game_result):
        if game_result == "victory":
            self.current_wins += 1

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

        self.add_trophies(game_result, current_brawler)
        self.add_win(game_result)
        return True

    def change_trophies(self, new):
        print(f"Trophies changed from {self.current_trophies} to {new}")
        self.current_trophies = new

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
