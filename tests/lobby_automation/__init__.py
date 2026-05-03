"""Test package shim for `python -m unittest discover -s tests`.

When unittest puts the tests directory first on sys.path, this package can
shadow the project-level lobby_automation.py module. Re-export the real class
so discovery works the same way as direct module test runs.
"""

import importlib.util
from pathlib import Path


_ROOT_MODULE = Path(__file__).resolve().parents[2] / "lobby_automation.py"
_SPEC = importlib.util.spec_from_file_location("_pyla_lobby_automation", _ROOT_MODULE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

LobbyAutomation = _MODULE.LobbyAutomation
load_toml_as_dict = _MODULE.load_toml_as_dict
