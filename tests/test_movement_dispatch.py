import unittest

import scrcpy

from window_controller import WindowController


class FakeControl:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def touch(self, x, y, action, pointer_id):
        if self.fail:
            raise RuntimeError("touch backend offline")
        self.calls.append((x, y, action, pointer_id))


class FakeClient:
    def __init__(self, fail=False):
        self.control = FakeControl(fail=fail)


def movement_controller(fail=False):
    wc = object.__new__(WindowController)
    wc.scrcpy_client = FakeClient(fail=fail)
    wc.joystick_x = 220.0
    wc.joystick_y = 870.0
    wc.width = 1920
    wc.height = 1080
    wc.scale_factor = 1.0
    wc.are_we_moving = False
    wc.PID_JOYSTICK = 1
    wc.PID_ATTACK = 2
    wc.last_joystick_pos = (None, None)
    wc.last_joystick_down_time = 0.0
    wc.last_joystick_move_time = 0.0
    wc.last_movement_log_time = 0.0
    wc.last_movement_error_log_time = 0.0
    wc.joystick_refresh_seconds = 0.35
    wc.joystick_repress_seconds = 1.8
    wc.joystick_down_move_delay = 0.0
    wc.input_debug = False
    wc.last_attack_pos = (None, None)
    wc.attack_pointer_down = False
    return wc


class MovementDispatchTests(unittest.TestCase):
    def test_angle_to_joystick_coordinates_uses_screen_space_direction(self):
        wc = movement_controller()
        endpoint = wc.calculate_joystick_endpoint(270, radius=150)

        self.assertEqual(endpoint["start"], (220.0, 870.0))
        self.assertEqual(round(endpoint["end"][0]), 220)
        self.assertEqual(round(endpoint["end"][1]), 720)

    def test_movement_holds_joystick_with_dedicated_pointer(self):
        wc = movement_controller()
        result = wc.move_joystick_angle(270, radius=150)

        self.assertTrue(result["ok"])
        self.assertTrue(result["attempted"])
        self.assertEqual(wc.scrcpy_client.control.calls[0], (220, 870, scrcpy.ACTION_DOWN, 1))
        self.assertEqual(wc.scrcpy_client.control.calls[1], (220, 720, scrcpy.ACTION_MOVE, 1))
        self.assertTrue(wc.are_we_moving)

    def test_movement_refreshes_same_angle_after_refresh_interval(self):
        wc = movement_controller()
        wc.move_joystick_angle(270, radius=150)
        wc.last_joystick_move_time -= 1.0

        result = wc.move_joystick_angle(270, radius=150)

        self.assertTrue(result["ok"])
        self.assertTrue(result["attempted"])
        self.assertEqual(wc.scrcpy_client.control.calls[-1], (220, 720, scrcpy.ACTION_MOVE, 1))

    def test_attack_click_does_not_release_held_movement_pointer(self):
        wc = movement_controller()
        wc.move_joystick_angle(270, radius=150)

        wc.click(1600, 870)

        self.assertTrue(wc.are_we_moving)
        self.assertEqual(wc.scrcpy_client.control.calls[-2], (1600, 870, scrcpy.ACTION_DOWN, 2))
        self.assertEqual(wc.scrcpy_client.control.calls[-1], (1600, 870, scrcpy.ACTION_UP, 2))

    def test_movement_reports_backend_failure_without_claiming_success(self):
        wc = movement_controller(fail=True)
        result = wc.move_joystick_angle(90, radius=150)

        self.assertFalse(result["ok"])
        self.assertTrue(result["attempted"])
        self.assertIn("touch backend offline", result["error"])

    def test_release_all_inputs_releases_movement_and_held_attack_pointer(self):
        wc = movement_controller()
        wc.move_joystick_angle(270, radius=150)
        wc.last_attack_pos = (1600.0, 870.0)
        wc.attack_pointer_down = True

        wc.release_all_inputs("test")

        self.assertFalse(wc.are_we_moving)
        self.assertFalse(wc.attack_pointer_down)
        self.assertEqual(wc.scrcpy_client.control.calls[-2], (220, 720, scrcpy.ACTION_UP, 1))
        self.assertEqual(wc.scrcpy_client.control.calls[-1], (1600, 870, scrcpy.ACTION_UP, 2))


if __name__ == "__main__":
    unittest.main()
