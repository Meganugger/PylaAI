import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import detect


class FakeIo:
    name = "input"


class FakeModel:
    def __init__(self, provider):
        self.provider = provider

    def get_providers(self):
        return [self.provider]

    def get_inputs(self):
        return [FakeIo()]

    def get_outputs(self):
        return [FakeIo()]

    def run(self, _output_names, _feed):
        if self.provider == "CUDAExecutionProvider":
            raise RuntimeError("cuda warmup failed")
        return [np.zeros((1, 5, 1), dtype=np.float32)]


class OnnxProviderSelectionTests(unittest.TestCase):
    def test_cuda_missing_nvrtc_path_scan_still_tries_cuda_by_default(self):
        with patch("detect.get_missing_cuda_runtime_dependencies", return_value=["nvrtc64_120_0.dll"]):
            with patch("detect.cuda_missing_nvrtc_allowed", return_value=True):
                providers, selected, reason, available = detect.build_onnx_providers(
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
        self.assertIn("CUDAExecutionProvider", available)

    def test_cuda_missing_nvrtc_can_be_strictly_blocked_by_config(self):
        with patch("detect.get_missing_cuda_runtime_dependencies", return_value=["nvrtc64_120_0.dll"]):
            with patch("detect.cuda_missing_nvrtc_allowed", return_value=False):
                providers, selected, reason, _available = detect.build_onnx_providers(
                    "auto",
                    "cuda",
                    available_providers=[
                        "CUDAExecutionProvider",
                        "DmlExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                )

        self.assertEqual(selected, "DmlExecutionProvider")
        self.assertEqual(providers[0], "DmlExecutionProvider")
        self.assertIn("onnx_allow_cuda_with_missing_nvrtc=false", reason)

    def test_load_model_falls_back_when_cuda_session_creation_fails(self):
        calls = []

        def fake_session(_path, sess_options=None, providers=None):
            calls.append(list(providers or []))
            if providers and providers[0] == "CUDAExecutionProvider":
                raise RuntimeError("cuda session failed")
            return FakeModel(providers[0])

        det = object.__new__(detect.Detect)
        det.model_path = "model.onnx"
        det.preferred_device = "gpu"
        det.preferred_backend = "cuda"

        with patch("detect.ort.get_available_providers", return_value=[
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]):
            with patch("detect.ort.InferenceSession", side_effect=fake_session):
                with patch("detect.preload_onnxruntime_gpu_dlls"):
                    model, provider = detect.Detect.load_model(det)

        self.assertEqual(provider, "DmlExecutionProvider")
        self.assertEqual(model.get_providers()[0], "DmlExecutionProvider")
        self.assertEqual(calls[0][0], "CUDAExecutionProvider")
        self.assertEqual(calls[1][0], "DmlExecutionProvider")

    def test_detect_init_falls_back_when_cuda_warmup_fails(self):
        calls = []

        def fake_session(_path, sess_options=None, providers=None):
            calls.append(list(providers or []))
            return FakeModel(providers[0])

        with patch("detect.ort.get_available_providers", return_value=[
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]):
            with patch("detect.ort.InferenceSession", side_effect=fake_session):
                with patch("detect.preload_onnxruntime_gpu_dlls"):
                    with patch("detect.load_toml_as_dict", return_value={
                        "cpu_or_gpu": "gpu",
                        "preferred_backend": "cuda",
                    }):
                        det = detect.Detect("model.onnx", classes=["enemy"])

        self.assertEqual(det.device, "DmlExecutionProvider")
        self.assertEqual(calls[0][0], "CUDAExecutionProvider")
        self.assertEqual(calls[1][0], "DmlExecutionProvider")

    def test_discover_nvidia_cuda_wheel_dll_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            bin_dir = Path(tmp) / "nvidia" / "cuda_nvrtc" / "bin"
            bin_dir.mkdir(parents=True)
            (bin_dir / "nvrtc64_120_0.dll").write_bytes(b"")

            dirs = detect.discover_nvidia_cuda_dll_directories(site_roots=[tmp])

        self.assertIn(str(bin_dir.resolve()), dirs)


if __name__ == "__main__":
    unittest.main()
