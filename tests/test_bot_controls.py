import threading
import unittest

from qt_ui.bridge import QtBridge
from stage_manager import StageManager


class DummySignal:
    def __init__(self):
        self.payloads = []

    def emit(self, *args):
        self.payloads.append(args)


class FakeThread:
    def __init__(self, alive=True):
        self._alive = alive

    def is_alive(self):
        return self._alive


class DummyController:
    def __init__(self):
        self.released = []

    def release_all_inputs(self, reason=""):
        self.released.append(reason)


class BotControlTests(unittest.TestCase):
    def make_bridge(self):
        bridge = QtBridge.__new__(QtBridge)
        bridge._bot_thread = FakeThread(True)
        bridge._bot_stop_event = threading.Event()
        bridge._bot_pause_event = threading.Event()
        bridge._bot_stop_requested = False
        bridge._bot_paused = False
        bridge._bot_control_state = "running"
        bridge._live_data = {}
        bridge._live_lock = threading.Lock()
        bridge._event_log = []
        bridge._pyla_main = lambda data: None
        bridge.liveDataChanged = DummySignal()
        bridge.logsChanged = DummySignal()
        bridge.notificationRaised = DummySignal()
        bridge._release_calls = []
        bridge._release_active_bot_inputs = lambda reason: bridge._release_calls.append(reason)
        return bridge

    def test_pause_resume_and_stop_update_control_events(self):
        bridge = self.make_bridge()

        bridge.pauseBot()
        self.assertTrue(bridge._bot_pause_event.is_set())
        self.assertTrue(bridge._bot_paused)
        self.assertEqual(bridge._bot_control_state, "paused")
        self.assertIn("pause requested", bridge._release_calls)

        bridge.resumeBot()
        self.assertFalse(bridge._bot_pause_event.is_set())
        self.assertFalse(bridge._bot_paused)
        self.assertEqual(bridge._bot_control_state, "running")

        bridge.pauseBot()
        bridge.stopBot()
        self.assertTrue(bridge._bot_stop_event.is_set())
        self.assertFalse(bridge._bot_pause_event.is_set())
        self.assertEqual(bridge._bot_control_state, "stopping")
        self.assertIn("stop requested", bridge._release_calls)

    def test_stop_when_not_running_is_safe(self):
        bridge = self.make_bridge()
        bridge._bot_thread = FakeThread(False)

        bridge.stopBot()

        self.assertEqual(bridge._bot_control_state, "stopped")
        self.assertFalse(bridge._bot_stop_requested)

    def test_stage_manager_suppresses_lobby_start_when_paused_or_stopped(self):
        manager = object.__new__(StageManager)
        manager.window_controller = DummyController()
        manager._last_control_log_at = 0.0
        stop_event = threading.Event()
        pause_event = threading.Event()
        manager.set_control_events(stop_event, pause_event)

        pause_event.set()
        self.assertFalse(manager._try_press_lobby_start())
        self.assertEqual(manager.window_controller.released[-1], "pause requested")

        pause_event.clear()
        stop_event.set()
        self.assertFalse(manager._try_press_lobby_start())
        self.assertEqual(manager.window_controller.released[-1], "stop requested")


if __name__ == "__main__":
    unittest.main()
