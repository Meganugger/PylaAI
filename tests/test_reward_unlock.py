import unittest

import cv2
import numpy as np

from state_finder.main import get_in_game_state, is_in_reward_unlock


class RewardUnlockTests(unittest.TestCase):
    @staticmethod
    def draw_reward_unlock_screen():
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        blue = cv2.cvtColor(
            np.full((1, 1, 3), (104, 210, 215), dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        card_blue = cv2.cvtColor(
            np.full((1, 1, 3), (98, 70, 230), dtype=np.uint8),
            cv2.COLOR_HSV2BGR,
        )[0, 0]
        image[:, :] = blue
        image[150:235, 760:1180] = (245, 245, 245)
        image[300:520, 780:1140] = card_blue
        image[520:620, 820:1100] = (20, 20, 20)
        image[635:720, 790:1140] = (245, 245, 245)
        return image

    def test_reward_unlock_detector_accepts_blue_unlock_screen(self):
        image = self.draw_reward_unlock_screen()

        self.assertTrue(is_in_reward_unlock(image))
        self.assertEqual(get_in_game_state(image, allow_reward_ocr=True), "reward_unlock")

    def test_reward_unlock_detector_rejects_match_like_screen(self):
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        image[:, :] = (70, 60, 80)
        image[150:235, 760:1180] = (245, 245, 245)
        image[635:720, 790:1140] = (245, 245, 245)

        self.assertFalse(is_in_reward_unlock(image))
        self.assertNotEqual(get_in_game_state(image, allow_reward_ocr=True), "reward_unlock")


if __name__ == "__main__":
    unittest.main()
