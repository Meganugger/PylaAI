import argparse
from importlib import metadata
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CUDA_WHEEL_DISTS = (
    "nvidia-cuda-runtime-cu12",
    "nvidia-cublas-cu12",
    "nvidia-cuda-nvrtc-cu12",
    "nvidia-cufft-cu12",
    "nvidia-cudnn-cu12",
)


def _dist_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Verify the installed ONNX Runtime backend.")
    parser.add_argument("--backend", choices=("cuda", "directml", "cpu", "auto"), default="auto")
    args = parser.parse_args()

    import onnxruntime as ort

    if args.backend == "cuda":
        try:
            from detect import preload_onnxruntime_gpu_dlls

            preload_onnxruntime_gpu_dlls()
        except Exception as exc:
            print(f"[ONNX][WARN] CUDA preload check failed: {exc}")

    print(f"Selected backend: {args.backend}")
    print(f"onnxruntime package version: {getattr(ort, '__version__', 'unknown')}")

    providers = list(ort.get_available_providers())
    print(f"Available providers: {', '.join(providers) if providers else '(none)'}")

    if args.backend == "cuda":
        print("CUDA runtime wheel versions:")
        for dist_name in CUDA_WHEEL_DISTS:
            version = _dist_version(dist_name)
            print(f"  {dist_name}: {version or 'missing'}")
        cuda_available = "CUDAExecutionProvider" in providers
        print(f"CUDAExecutionProvider available: {'yes' if cuda_available else 'no'}")
        if not cuda_available:
            print("CUDA was selected, but ONNX Runtime did not expose CUDAExecutionProvider.")
            print("Repair the CUDA install or rerun setup with DirectML/CPU.")
            return 2

    if args.backend == "directml":
        dml_available = "DmlExecutionProvider" in providers
        print(f"DmlExecutionProvider available: {'yes' if dml_available else 'no'}")
        if not dml_available:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
