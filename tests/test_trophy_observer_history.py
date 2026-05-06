import copy
import unittest
from unittest.mock import Mock, patch

from trophy_observer import TrophyObserver


def make_observer(history=None):
    history = copy.deepcopy(history or {
        "darryl": {"victory": 0, "defeat": 0, "draw": 0},
        "total": {"victory": 0, "defeat": 0, "draw": 0},
    })

    def fake_load_toml(path):
        path = str(path)
        if path.endswith("match_history.toml"):
            return copy.deepcopy(history)
        if path.endswith("lobby_config.toml"):
            return {"lobby": {"trophy_observer": [0, 0, 1, 1]}}
        if path.endswith("general_config.toml"):
            return {"trophies_multiplier": 1}
        return {}

    with patch("trophy_observer.load_toml_as_dict", side_effect=fake_load_toml):
        with patch("trophy_observer.load_brawlers_info", return_value={"darryl": {}, "shelly": {}}):
            with patch.object(TrophyObserver, "save_history", lambda self: None):
                observer = TrophyObserver(["darryl"])
    observer.save_history = lambda: None
    observer.current_trophies = 500
    observer.current_wins = 0
    observer.match_counter = 3
    return observer


class TrophyObserverHistoryTests(unittest.TestCase):
    def test_sent_match_history_int_migrates_and_api_send_does_not_crash(self):
        observer = make_observer({
            "darryl": {"victory": 3, "defeat": 1, "draw": 0},
            "total": {"victory": 3, "defeat": 1, "draw": 0},
        })
        observer.sent_match_history = {"darryl": 1}
        response = Mock(status_code=200)

        with patch("trophy_observer.api_base_url", "example.test"):
            with patch("trophy_observer.requests.post", return_value=response) as post:
                self.assertTrue(observer.send_results_to_api())

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["darryl"]["wins"], 2)
        self.assertEqual(payload["darryl"]["defeats"], 1)
        self.assertIsInstance(observer.sent_match_history["darryl"], dict)
        self.assertEqual(observer.sent_match_history["darryl"]["victory"], 3)

    def test_missing_sent_match_history_entry_initializes_default(self):
        observer = make_observer({
            "darryl": {"victory": 1, "defeat": 0, "draw": 0},
            "total": {"victory": 1, "defeat": 0, "draw": 0},
        })
        observer.sent_match_history = {}

        with patch("trophy_observer.api_base_url", "localhost"):
            self.assertTrue(observer.send_results_to_api())

        self.assertEqual(observer.sent_match_history["darryl"], {"defeat": 0, "victory": 0, "draw": 0})

    def test_partial_dict_fills_missing_keys(self):
        observer = make_observer()

        normalized = observer.normalize_match_history_entry({"victory": "2"}, brawler="darryl")

        self.assertEqual(normalized, {"defeat": 0, "victory": 2, "draw": 0})

    def test_mixed_case_history_keys_are_preserved_during_migration(self):
        observer = make_observer({
            "Darryl": {"victory": 4, "defeat": 2, "draw": 1},
            "total": {"victory": 4, "defeat": 2, "draw": 1},
        })

        self.assertEqual(observer.match_history["darryl"], {"victory": 4, "defeat": 2, "draw": 1})

    def test_send_results_handles_malformed_history_without_crashing(self):
        observer = make_observer()
        observer.match_history = {"darryl": 3, "total": {}}
        observer.sent_match_history = {"darryl": {"victory": 1}}

        with patch("trophy_observer.api_base_url", "localhost"):
            self.assertTrue(observer.send_results_to_api())

        self.assertEqual(observer.match_history["darryl"], {"defeat": 0, "victory": 3, "draw": 0})
        self.assertEqual(observer.sent_match_history["darryl"], {"defeat": 0, "victory": 1, "draw": 0})

    def test_add_trophies_applies_locally_when_api_send_fails(self):
        observer = make_observer()

        with patch("trophy_observer.api_base_url", "example.test"):
            with patch("trophy_observer.requests.post", side_effect=RuntimeError("offline")):
                self.assertTrue(observer.add_trophies("victory", "Darryl"))

        self.assertEqual(observer.match_history["darryl"]["victory"], 1)
        self.assertEqual(observer.match_history["total"]["victory"], 1)
        self.assertEqual(observer.last_match_bucket, "victory")

    def test_corrupted_result_name_sections_are_removed(self):
        observer = make_observer({
            "darryl": {"victory": 2, "defeat": 1, "draw": 0},
            "victory": {"victory": 9, "defeat": 0, "draw": 0},
            "defeat": {"victory": 0, "defeat": 9, "draw": 0},
            "draw": {"victory": 0, "defeat": 0, "draw": 9},
            "total": {"victory": 2, "defeat": 1, "draw": 0},
        })

        self.assertNotIn("victory", observer.match_history)
        self.assertNotIn("defeat", observer.match_history)
        self.assertNotIn("draw", observer.match_history)
        self.assertEqual(observer.match_history["darryl"], {"victory": 2, "defeat": 1, "draw": 0})
        self.assertEqual(observer.match_history["total"], {"victory": 2, "defeat": 1, "draw": 0})

    def test_add_trophies_never_creates_result_named_brawler_section(self):
        observer = make_observer()

        self.assertTrue(observer.add_trophies("victory", "darryl"))
        self.assertTrue(observer.add_trophies("defeat", "darryl"))

        self.assertEqual(observer.match_history["darryl"]["victory"], 1)
        self.assertEqual(observer.match_history["darryl"]["defeat"], 1)
        self.assertEqual(observer.match_history["total"]["victory"], 1)
        self.assertEqual(observer.match_history["total"]["defeat"], 1)
        self.assertNotIn("victory", observer.match_history)
        self.assertNotIn("defeat", observer.match_history)

    def test_invalid_current_brawler_result_name_falls_back_to_resolved_brawler(self):
        observer = make_observer()
        observer._active_match_brawler = "darryl"

        self.assertTrue(observer.add_trophies("victory", "victory"))

        self.assertEqual(observer.match_history["darryl"]["victory"], 1)
        self.assertEqual(observer.match_history["total"]["victory"], 1)
        self.assertNotIn("victory", observer.match_history)

    def test_api_delta_ignores_invalid_result_name_sections(self):
        observer = make_observer()
        observer.match_history = {
            "darryl": {"victory": 3, "defeat": 0, "draw": 0},
            "victory": {"victory": 10, "defeat": 0, "draw": 0},
            "total": {"victory": 13, "defeat": 0, "draw": 0},
        }
        observer.sent_match_history = {"darryl": {"victory": 1, "defeat": 0, "draw": 0}}
        response = Mock(status_code=200)

        with patch("trophy_observer.api_base_url", "example.test"):
            with patch("trophy_observer.requests.post", return_value=response) as post:
                self.assertTrue(observer.send_results_to_api())

        payload = post.call_args.kwargs["json"]
        self.assertEqual(set(payload.keys()), {"darryl"})
        self.assertEqual(payload["darryl"]["wins"], 2)

    def test_trio_showdown_first_place_starts_at_eleven_trophies(self):
        observer = make_observer()
        observer.current_trophies = 0

        self.assertEqual(observer._showdown_delta_for("1st", trophies=0), 11)


if __name__ == "__main__":
    unittest.main()
