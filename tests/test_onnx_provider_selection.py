import unittest
from unittest.mock import patch

import detect


class OnnxProviderSelectionTests(unittest.TestCase):
    def test_cuda_is_skipped_when_runtime_dependency_is_missing(self):
        with patch("detect.get_missing_cuda_runtime_dependencies", return_value=["nvrtc64_120_0.dll"]):
            providers, selected, reason, available = detect.build_onnx_providers(
                "auto",
                "auto",
                available_providers=[
                    "CUDAExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )

        self.assertEqual(selected, "DmlExecutionProvider")
        self.assertEqual(providers[0], "DmlExecutionProvider")
        self.assertIn("nvrtc64_120_0.dll", reason)
        self.assertIn("CUDAExecutionProvider", available)

    def test_cuda_is_used_when_dependencies_are_present_and_requested(self):
        with patch("detect.get_missing_cuda_runtime_dependencies", return_value=[]):
            providers, selected, reason, _available = detect.build_onnx_providers(
                "auto",
                "cuda",
                available_providers=[
                    "CUDAExecutionProvider",
                    "DmlExecutionProvider",
                    "CPUExecutionProvider",
                ],
            )

        self.assertEqual(selected, "CUDAExecutionProvider")
        self.assertEqual(providers[0], "CUDAExecutionProvider")
        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
