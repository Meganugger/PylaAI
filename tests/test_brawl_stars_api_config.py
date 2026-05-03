import tempfile
import unittest
from unittest.mock import patch

from utils import load_brawl_stars_api_config, load_toml_as_dict, normalize_brawler_name


class BrawlStarsApiConfigTests(unittest.TestCase):
    def test_malformed_token_file_still_recovers_api_token(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as handle:
            handle.write('api_token = "abc\\n  def"\n')
            handle.write('player_tag = "#P123"\n')
            handle.write('timeout_seconds = 7\n')
            path = handle.name

        config = load_brawl_stars_api_config(path)

        self.assertEqual(config["api_token"], "abcdef")
        self.assertEqual(config["player_tag"], "#P123")
        self.assertEqual(config["timeout_seconds"], 7)

    def test_toml_loader_accepts_utf8_bom(self):
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8-sig") as handle:
            handle.write('idle_pixels_minimum = 1234\n')
            path = handle.name

        config = load_toml_as_dict(path)

        self.assertEqual(config["idle_pixels_minimum"], 1234)

    @patch("utils.refresh_brawl_stars_api_token_if_enabled")
    def test_force_refresh_flows_to_token_refresher(self, mock_refresh):
        mock_refresh.side_effect = lambda config, file_path, force=False: {
            **config,
            "force": force,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False, encoding="utf-8") as handle:
            handle.write('api_token = "old"\nplayer_tag = "#P123"\nauto_refresh_token = true\n')
            path = handle.name

        config = load_brawl_stars_api_config(path, force_refresh=True)

        self.assertTrue(config["force"])

    def test_brawler_name_normalization_matches_api_names(self):
        self.assertEqual(normalize_brawler_name("Buzz Lightyear"), "buzzlightyear")
        self.assertEqual(normalize_brawler_name("Larry & Lawrie"), "larrylawrie")


if __name__ == "__main__":
    unittest.main()
