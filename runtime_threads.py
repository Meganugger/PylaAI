import os

import toml

_GENERAL_CONFIG_PATH = "cfg/general_config.toml"
_config_cache = None
_process_limits_applied = False
_opencv_threads_applied = False
_torch_threads_applied = False

_THREAD_PRESETS = {
    "cuda": {
        "process_threads": 4,
        "opencv_threads": 2,
        "onnx_intra_threads": 2,
        "onnx_inter_threads": 1,
        "torch_threads": 2,
        "torch_interop_threads": 1,
    },
    "directml": {
        "process_threads": 4,
        "opencv_threads": 2,
        "onnx_intra_threads": 3,
        "onnx_inter_threads": 1,
        "torch_threads": 2,
        "torch_interop_threads": 1,
    },
    "cpu": {
        "process_threads": 6,
        "opencv_threads": 2,
        "onnx_intra_threads": 4,
        "onnx_inter_threads": 1,
        "torch_threads": 2,
        "torch_interop_threads": 1,
    },
    "auto": {
        "process_threads": 4,
        "opencv_threads": 2,
        "onnx_intra_threads": 2,
        "onnx_inter_threads": 1,
        "torch_threads": 2,
        "torch_interop_threads": 1,
    },
}


def _load_general_config():
    global _config_cache
    if _config_cache is None:
        if os.path.exists(_GENERAL_CONFIG_PATH):
            with open(_GENERAL_CONFIG_PATH, "r") as file:
                _config_cache = toml.load(file)
        else:
            _config_cache = {}
    return _config_cache


def _cpu_count():
    return max(2, os.cpu_count() or 4)


def get_preferred_backend(config=None):
    config = _load_general_config() if config is None else config
    preferred_device = str(config.get("cpu_or_gpu", "auto")).lower()
    preferred_backend = str(config.get("preferred_backend", "auto")).lower()

    if preferred_device == "cpu":
        return "cpu"
    if preferred_backend in ("cuda", "directml"):
        return preferred_backend
    if preferred_device in ("gpu", "auto"):
        return "auto"
    return "cpu"


def get_thread_preset(backend=None):
    normalized_backend = str(backend or get_preferred_backend()).lower()
    return dict(_THREAD_PRESETS.get(normalized_backend, _THREAD_PRESETS["auto"]))


def _default_thread_setting(config_key):
    preset = get_thread_preset()
    default_value = preset.get(config_key)
    if default_value is not None:
        return default_value

    cpu_count = _cpu_count()
    if get_preferred_backend() == "cpu":
        return max(2, min(8, cpu_count - 1))
    return max(2, min(4, cpu_count // 2))


def _resolve_thread_setting(config_key, default_value):
    config = _load_general_config()
    raw_value = config.get(config_key, "auto")
    if isinstance(raw_value, int):
        return max(1, raw_value)

    value = str(raw_value).strip().lower()
    if value in ("", "auto", "none"):
        return max(1, default_value)
    try:
        return max(1, int(value))
    except ValueError:
        return max(1, default_value)


def apply_process_thread_limits():
    global _process_limits_applied
    if _process_limits_applied:
        return

    env_thread_limit = str(_resolve_thread_setting("process_threads", _default_thread_setting("process_threads")))
    for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_name, env_thread_limit)

    _process_limits_applied = True


def configure_opencv_threads(cv2_module):
    global _opencv_threads_applied
    if _opencv_threads_applied:
        return

    thread_count = _resolve_thread_setting("opencv_threads", _default_thread_setting("opencv_threads"))
    cv2_module.setNumThreads(thread_count)
    _opencv_threads_applied = True


def configure_onnx_session_options(ort_module, session_options):
    intra_threads = _resolve_thread_setting("onnx_intra_threads", _default_thread_setting("onnx_intra_threads"))
    inter_threads = _resolve_thread_setting("onnx_inter_threads", _default_thread_setting("onnx_inter_threads"))

    session_options.intra_op_num_threads = intra_threads
    session_options.inter_op_num_threads = inter_threads
    if inter_threads <= 1:
        session_options.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL


def configure_torch_threads(torch_module):
    global _torch_threads_applied
    if _torch_threads_applied:
        return

    torch_threads = _resolve_thread_setting("torch_threads", _default_thread_setting("torch_threads"))
    torch_interop_threads = _resolve_thread_setting("torch_interop_threads", _default_thread_setting("torch_interop_threads"))

    torch_module.set_num_threads(torch_threads)
    try:
        torch_module.set_num_interop_threads(torch_interop_threads)
    except RuntimeError:
        pass

    _torch_threads_applied = True
