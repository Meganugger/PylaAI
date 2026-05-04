import tempfile
import unittest
import os
from pathlib import Path

from tools.updater import (
    backup_preserved_files,
    build_update_status,
    copy_update_files,
    merge_toml_text,
    repo_branch,
    read_installed_update_sha,
    read_local_update_sha,
    restore_preserved_files,
    write_local_update_info,
)


def workspace_tempdir():
    root = Path.cwd() / ".test-tmp"
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root)


class UpdaterTest(unittest.TestCase):
    def test_copy_update_preserves_user_api_config_and_skips_protected_binaries(self):
        with workspace_tempdir() as tmp:
            root = Path(tmp)
            project = root / "project"
            source = root / "source"
            backup = root / "backup"

            (project / "cfg").mkdir(parents=True)
            (project / "cfg" / "brawl_stars_api.toml").write_text('api_token = "USER"\n', encoding="utf-8")
            (project / "cfg" / "general_config.toml").write_text(
                'max_ips = 24\nplayer_tag = "USER_TAG"\nold_local_key = "keep"\n',
                encoding="utf-8",
            )
            (project / "cfg" / "adaptive_state.json").write_text(
                '{"matches": 12, "old_only": true, "nested": {"user": 1}}',
                encoding="utf-8",
            )
            (project / "updater.exe").write_text("old updater", encoding="utf-8")
            (project / "main.py").write_text("old", encoding="utf-8")

            (source / "cfg").mkdir(parents=True)
            (source / "cfg" / "brawl_stars_api.toml").write_text('api_token = ""\n', encoding="utf-8")
            (source / "cfg" / "general_config.toml").write_text(
                'max_ips = 30\nplayer_tag = ""\nnew_key = "added"\n',
                encoding="utf-8",
            )
            (source / "cfg" / "adaptive_state.json").write_text(
                '{"matches": 0, "new_only": true, "nested": {"default": 2}}',
                encoding="utf-8",
            )
            (source / "updater.exe").write_text("new updater", encoding="utf-8")
            (source / "adb.exe").write_text("new adb", encoding="utf-8")
            (source / "main.py").write_text("new", encoding="utf-8")
            (source / "new_file.py").write_text("added", encoding="utf-8")

            backup_preserved_files(project, backup)
            copy_update_files(source, project)
            restore_preserved_files(project, backup)

            self.assertEqual((project / "cfg" / "brawl_stars_api.toml").read_text(encoding="utf-8"), 'api_token = "USER"\n')
            general_config = (project / "cfg" / "general_config.toml").read_text(encoding="utf-8")
            self.assertIn("max_ips = 24", general_config)
            self.assertIn('player_tag = "USER_TAG"', general_config)
            self.assertIn('new_key = "added"', general_config)
            self.assertIn('old_local_key = "keep"', general_config)
            adaptive_state = (project / "cfg" / "adaptive_state.json").read_text(encoding="utf-8")
            self.assertIn('"matches": 12', adaptive_state)
            self.assertIn('"new_only": true', adaptive_state)
            self.assertIn('"old_only": true', adaptive_state)
            self.assertIn('"default": 2', adaptive_state)
            self.assertIn('"user": 1', adaptive_state)
            self.assertEqual((project / "updater.exe").read_text(encoding="utf-8"), "old updater")
            self.assertFalse((project / "adb.exe").exists())
            self.assertEqual((project / "main.py").read_text(encoding="utf-8"), "new")
            self.assertEqual((project / "new_file.py").read_text(encoding="utf-8"), "added")

    def test_toml_merge_keeps_user_values_and_adds_new_defaults(self):
        merged = merge_toml_text(
            'api_token = ""\ntimeout_seconds = 15\nnew_key = true\n',
            'api_token = "USER_TOKEN"\nold_key = "kept"\n',
        )

        self.assertIn('api_token = "USER_TOKEN"', merged)
        self.assertIn("timeout_seconds = 15", merged)
        self.assertIn("new_key = true", merged)
        self.assertIn('old_key = "kept"', merged)

    def test_update_info_marker_round_trips_branch_sha(self):
        with workspace_tempdir() as tmp:
            project = Path(tmp)

            self.assertIsNone(read_local_update_sha(project))
            write_local_update_info(project, "abc123")

            self.assertEqual(read_local_update_sha(project), "abc123")

    def test_update_status_reports_failure_without_throwing(self):
        with workspace_tempdir() as tmp:
            project = Path(tmp)
            (project / "cfg").mkdir()
            (project / "cfg" / "general_config.toml").write_text('pyla_version = "1.0.0+test"\n', encoding="utf-8")

            import tools.updater as updater_module

            original = updater_module.latest_branch_info
            updater_module.latest_branch_info = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
            try:
                status = build_update_status(project)
            finally:
                updater_module.latest_branch_info = original

            self.assertFalse(status["ok"])
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["currentVersion"], "1.0.0+test")

    def test_update_status_does_not_prompt_when_local_marker_matches_latest(self):
        with workspace_tempdir() as tmp:
            project = Path(tmp)
            (project / "cfg").mkdir()
            (project / "cfg" / "general_config.toml").write_text('pyla_version = "1.0.0+main"\n', encoding="utf-8")
            write_local_update_info(project, "abc123")

            import tools.updater as updater_module

            original = updater_module.latest_branch_info
            updater_module.latest_branch_info = lambda _project=None: {
                "sha": "abc123",
                "short_sha": "abc123",
                "repo": "Meganugger/PylaAI",
                "branch": "main",
                "summary": "same commit",
                "message": "same commit",
                "html_url": "",
            }
            try:
                status = build_update_status(project)
            finally:
                updater_module.latest_branch_info = original

            self.assertTrue(status["ok"])
            self.assertFalse(status["updateAvailable"])
            self.assertEqual(status["state"], "up to date")
            self.assertEqual(read_installed_update_sha(project), "abc123")

    def test_update_status_does_not_prompt_when_local_git_head_is_ahead(self):
        with workspace_tempdir() as tmp:
            project = Path(tmp)
            (project / "cfg").mkdir()
            (project / "cfg" / "general_config.toml").write_text('pyla_version = "1.0.0+main"\n', encoding="utf-8")

            import tools.updater as updater_module

            originals = (
                updater_module.latest_branch_info,
                updater_module.read_local_update_sha,
                updater_module.read_current_git_sha,
                updater_module.is_git_ancestor,
            )
            updater_module.latest_branch_info = lambda _project=None: {
                "sha": "remote123",
                "short_sha": "remote123",
                "repo": "Meganugger/PylaAI",
                "branch": "main",
                "summary": "remote ancestor",
                "message": "remote ancestor",
                "html_url": "",
            }
            updater_module.read_local_update_sha = lambda _project: None
            updater_module.read_current_git_sha = lambda _project: "local456"
            updater_module.is_git_ancestor = lambda _project, ancestor, descendant: (
                ancestor == "remote123" and descendant == "local456"
            )
            try:
                status = build_update_status(project)
            finally:
                (
                    updater_module.latest_branch_info,
                    updater_module.read_local_update_sha,
                    updater_module.read_current_git_sha,
                    updater_module.is_git_ancestor,
                ) = originals

            self.assertTrue(status["ok"])
            self.assertFalse(status["updateAvailable"])
            self.assertEqual(status["state"], "up to date")

    def test_repo_branch_infers_branch_specific_version_when_env_is_not_set(self):
        with workspace_tempdir() as tmp:
            project = Path(tmp)
            (project / "cfg").mkdir()
            (project / "cfg" / "general_config.toml").write_text(
                'pyla_version = "1.0.0+strongestbotfull"\n',
                encoding="utf-8",
            )

            original = os.environ.pop("PYLA_UPDATE_BRANCH", None)
            try:
                self.assertEqual(repo_branch(project), "strongest-bot-full")
            finally:
                if original is not None:
                    os.environ["PYLA_UPDATE_BRANCH"] = original


if __name__ == "__main__":
    unittest.main()
