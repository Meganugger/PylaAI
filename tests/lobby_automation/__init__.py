"""Test-package shim that re-exports the project lobby_automation module.

When unittest discovery imports test modules from tests/lobby_automation, the
package name can otherwise shadow the real top-level lobby_automation.py.
"""

import importlib.util
from pathlib import Path

_ROOT_MODULE = Path(__file__).resolve().parents[2] / "lobby_automation.py"
_SPEC = importlib.util.spec_from_file_location("_pyla_lobby_automation", _ROOT_MODULE)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

LobbyAutomation = _MODULE.LobbyAutomation
load_toml_as_dict = _MODULE.load_toml_as_dict

__all__ = ["LobbyAutomation", "load_toml_as_dict"]
