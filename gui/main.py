import os
import sys

import utils
from utils import api_base_url

sys.path.append(os.path.abspath('../'))


class App:

    def __init__(self, login_page, select_brawler_page, pyla_main, brawlers, hub_menu):
        self.login = login_page
        self.select_brawler = select_brawler_page
        self.pyla_main = pyla_main
        self.brawlers = brawlers
        self.hub_menu = hub_menu

    def run_login(self):
        return bool(self.login())

    def run_hub(self, pyla_version, get_latest_version):
        latest_version = pyla_version if api_base_url == "localhost" else get_latest_version()
        self.hub_menu(pyla_version, latest_version)

    def run_brawler_setup(self):
        screen = self.select_brawler(brawlers=self.brawlers)
        return getattr(screen, "result_data", None)

    def start(self, pyla_version, get_latest_version):
        if not self.run_login():
            return

        self.run_hub(pyla_version, get_latest_version)
        brawler_data = self.run_brawler_setup()
        if not brawler_data:
            return

        utils.save_brawler_data(brawler_data)
        self.pyla_main(brawler_data)

