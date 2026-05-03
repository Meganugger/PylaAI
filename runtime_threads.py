import os

import toml

_config_cache = None
_process_limits_applied = False
_opencv_threads_applied = False
_torch_threads_applied = False
_detected_backend = None

_THREAD_PRESETS = {
    "cuda": {
        "process_threads": 3,
        "opencv_threads": 1,
        "onnx_intra_threads": 2,
        "onnx_inter_threads": 1,
        "torch_threads": 1,
        "torch_interop_threads": 1,
    },
    "directml": {
        "process_threads": 3,
        "opencv_threads": 1,
        "onnx_intra_threads": 2,
        "onnx_inter_threads": 1,
        "torch_threads": 1,
        "torch_interop_threads": 1,
    },
}


def _load_toml_file(file_path):
    with open(file_path, "r", encoding="utf-8-sig") as file:
        text = file.read()
    toml_decode_error = getattr(toml, "TomlDecodeError", ValueError)
    try:
        return toml.loads(text.lstrip("\ufeffï»¿"))
    except toml_decode_error:
        sanitized = text.encode("utf-8", errors="ignore").decode("utf-8-sig", errors="ignore")
        return toml.loads(sanitized.lstrip("\ufeffï»¿\x00\r\n\t "))


def _load_general_config():
    global _config_cache
    general_config_path = os.path.join(os.environ.get("PYLA_CONFIG_DIR", "cfg"), "general_config.toml")
    if _config_cache is None:
        if os.path.exists(general_config_path):
            _config_cache = _load_toml_file(general_config_path)
        else:
            _config_cache = {}
    return _config_cache


def _cpu_count():
    return max(2, os.cpu_count() or 4)


def _instance_count(config=None):
    config = _load_general_config() if config is None else config
    try:
        return max(1, min(16, int(config.get("instance_count", 1) or 1)))
    except Exception:
        return 1


def _detect_available_backend():
    global _detected_backend
    if _detected_backend is not None:
        return _detected_backend

    try:
        import onnxruntime as ort  # Imported lazily to keep module import cheap.

        available_providers = set(ort.get_available_providers())
    except Exception:
        available_providers = set()

    if "CUDAExecutionProvider" in available_providers:
        _detected_backend = "cuda"
    elif "DmlExecutionProvider" in available_providers:
        _detected_backend = "directml"
    else:
        _detected_backend = "cpu"
    return _detected_backend


def get_preferred_backend(config=None):
    config = _load_general_config() if config is None else config
    preferred_device = str(config.get("cpu_or_gpu", "auto")).lower()
    preferred_backend = str(config.get("preferred_backend", "auto")).lower()

    if preferred_device == "cpu":
        return "cpu"
    if preferred_backend in ("cuda", "directml"):
        return preferred_backend
    if preferred_device in ("gpu", "auto"):
        return _detect_available_backend()
    return "cpu"


def get_thread_preset(backend=None):
    config = _load_general_config()
    normalized_backend = str(backend or get_preferred_backend()).lower()
    if normalized_backend in ("auto", ""):
        return get_thread_preset(get_preferred_backend())
    if normalized_backend == "cpu":
        cpu_count = _cpu_count()
        instance_count = _instance_count(config)
        if instance_count > 1:
            per_instance_budget = max(2, cpu_count // instance_count)
            return {
                "process_threads": max(2, min(4, per_instance_budget)),
                "opencv_threads": 1,
                "onnx_intra_threads": max(1, min(3, per_instance_budget)),
                "onnx_inter_threads": 1,
                "torch_threads": max(1, min(2, per_instance_budget)),
                "torch_interop_threads": 1,
            }
        return {
            "process_threads": max(4, min(12, cpu_count)),
            "opencv_threads": max(1, min(4, cpu_count // 2)),
            "onnx_intra_threads": max(2, min(8, cpu_count - 1)),
            "onnx_inter_threads": 2 if cpu_count >= 12 else 1,
            "torch_threads": max(2, min(6, cpu_count - 1)),
            "torch_interop_threads": max(1, min(3, cpu_count // 4)),
        }
    return dict(_THREAD_PRESETS.get(normalized_backend, _THREAD_PRESETS["cuda"]))


def _default_thread_setting(config_key):
    preset = get_thread_preset()
    default_value = preset.get(config_key)
    if default_value is not None:
        return default_value

    cpu_count = _cpu_count()
    return max(2, min(8, cpu_count - 1))


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
