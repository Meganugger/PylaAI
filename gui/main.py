import json
import os
import sys

import utils
from utils import api_base_url, get_brawler_data_path

sys.path.append(os.path.abspath('../'))

BRAWLER_DATA_FILE = get_brawler_data_path()


class App:
    """
    Unified dashboard-based app.
    Always shows the dashboard UI -- no more autonomous mode bypass.
    """

    def __init__(self, login_page, select_brawler_page, pyla_main, brawlers, hub_menu):
        self.login = login_page
        self.select_brawler = select_brawler_page
        self.logged_in = False
        self.brawler_data = None
        self.pyla_main = pyla_main
        self.brawlers = brawlers
        self.hub_menu = hub_menu

    def set_is_logged(self, value):
        self.logged_in = value

    def set_data(self, value):
        self.brawler_data = value

    @staticmethod
    def _load_saved_brawler_data():
        """Try to load saved brawler data for pre-filling the dashboard."""
        if not os.path.exists(BRAWLER_DATA_FILE):
            return None
        try:
            with open(BRAWLER_DATA_FILE, "r") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) > 0 and "brawler" in data[0]:
                return data
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None

    def start(self, pyla_version, get_latest_version):
        saved = self._load_saved_brawler_data()
        try:
            from qt_ui.app import run_qt_app

            run_qt_app(
                version_str=pyla_version,
                brawlers=self.brawlers,
                pyla_main_fn=self.pyla_main,
                login_fn=self.login,
                saved_brawler_data=saved,
            )
            return
        except ImportError as exc:
            print(f"[UI] PySide6 UI unavailable, falling back to CustomTkinter: {exc}")
        except Exception as exc:
            print(f"[UI] Qt UI failed to launch, falling back to CustomTkinter: {exc}")
        from dashboard import Dashboard

        dashboard = Dashboard(
            version_str=pyla_version,
            brawlers=self.brawlers,
            pyla_main_fn=self.pyla_main,
            login_fn=self.login,
        )

        if saved:
            dashboard.brawlers_data = saved
            dashboard._update_sidebar_brawler()
            if hasattr(dashboard, "_brawler_scroll"):
                dashboard._refresh_brawler_grid()
            print(f"[DASHBOARD] Pre-loaded saved brawler: "
                  f"{', '.join(d['brawler'] for d in saved)}")

        dashboard.run()
