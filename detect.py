import ctypes.util
import os
import sys
import time

from runtime_threads import apply_process_thread_limits, configure_onnx_session_options, configure_opencv_threads

apply_process_thread_limits()

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from utils import load_toml_as_dict, record_timing

configure_opencv_threads(cv2)

_ONNX_PROVIDER_STATUS_LOGGED = False
_CUDA_DEPENDENCY_WARNED = False
_CUDA_DEPENDENCY_NAMES = ("nvrtc64_120_0.dll",)


def preload_onnxruntime_gpu_dlls():
    if not hasattr(ort, "preload_dlls"):
        return

    try:
        ort.preload_dlls(cuda=True, cudnn=True, msvc=True, directory="")
        return
    except TypeError:
        pass
    except Exception:
        return

    try:
        ort.preload_dlls(directory="")
    except Exception:
        pass


def _dll_exists_on_path(dll_name):
    if os.name != "nt":
        return True
    if ctypes.util.find_library(dll_name):
        return True
    search_dirs = [
        os.getcwd(),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(sys.executable),
    ]
    search_dirs.extend(str(part) for part in os.environ.get("PATH", "").split(os.pathsep) if part)
    for directory in search_dirs:
        try:
            if os.path.exists(os.path.join(directory, dll_name)):
                return True
        except Exception:
            continue
    return False


def get_missing_cuda_runtime_dependencies():
    if os.name != "nt":
        return []
    return [dll for dll in _CUDA_DEPENDENCY_NAMES if not _dll_exists_on_path(dll)]


def build_onnx_providers(preferred_device, preferred_backend, available_providers=None):
    """Build a provider list without selecting CUDA when its runtime DLLs are missing."""
    preferred_device = str(preferred_device or "auto").lower()
    preferred_backend = str(preferred_backend or "auto").lower()
    available = list(available_providers if available_providers is not None else ort.get_available_providers())
    providers = ["CPUExecutionProvider"]
    selected = "CPUExecutionProvider"
    fallback_reason = ""

    if preferred_device in ("gpu", "auto"):
        if preferred_backend == "directml":
            provider_order = ["DmlExecutionProvider", "CUDAExecutionProvider", "AzureExecutionProvider"]
        elif preferred_backend == "cuda":
            provider_order = ["CUDAExecutionProvider", "DmlExecutionProvider", "AzureExecutionProvider"]
        else:
            provider_order = ["CUDAExecutionProvider", "DmlExecutionProvider", "AzureExecutionProvider"]

        missing_cuda_deps = get_missing_cuda_runtime_dependencies()
        for provider_name in provider_order:
            if provider_name not in available:
                continue
            if provider_name == "CUDAExecutionProvider" and missing_cuda_deps:
                fallback_reason = (
                    f"CUDA runtime dependency {', '.join(missing_cuda_deps)} missing"
                )
                continue
            selected = provider_name
            providers = [provider_name, "CPUExecutionProvider"]
            break

        if selected == "CPUExecutionProvider" and "CUDAExecutionProvider" in available:
            missing_cuda_deps = get_missing_cuda_runtime_dependencies()
            if missing_cuda_deps:
                fallback_reason = (
                    f"CUDA runtime dependency {', '.join(missing_cuda_deps)} missing"
                )

    return providers, selected, fallback_reason, available


def log_onnx_provider_status(model_path, selected_provider, available_providers=None, fallback_reason=""):
    global _ONNX_PROVIDER_STATUS_LOGGED, _CUDA_DEPENDENCY_WARNED
    available = available_providers if available_providers is not None else ort.get_available_providers()
    model_name = os.path.basename(model_path)
    if fallback_reason and not _CUDA_DEPENDENCY_WARNED:
        print(f"[ONNX][WARN] {fallback_reason}; falling back to {selected_provider}.")
        print("Install a matching CUDA runtime or switch preferred_backend to directml/cpu to remove this warning.")
        _CUDA_DEPENDENCY_WARNED = True
    if not _ONNX_PROVIDER_STATUS_LOGGED:
        if selected_provider == "CUDAExecutionProvider":
            print("[ONNX] CUDA provider active")
        elif selected_provider == "DmlExecutionProvider":
            print("[ONNX] DirectML provider active")
        else:
            print(f"[ONNX] provider active: {selected_provider}")
        print(f"[ONNX] available providers: {', '.join(available)}")
        _ONNX_PROVIDER_STATUS_LOGGED = True
    print(f"ONNX Runtime provider for {model_name}: {selected_provider}")

class Detect:
    def __init__(self, model_path, ignore_classes=None, classes=None, input_size=(640, 640)):
        config = load_toml_as_dict("cfg/general_config.toml")
        self.preferred_device = str(config.get("cpu_or_gpu", "auto")).lower()
        self.preferred_backend = str(config.get("preferred_backend", "auto")).lower()
        self.model_path = model_path
        self.classes = classes
        self.ignore_classes = ignore_classes if ignore_classes else []
        self.input_size = input_size
        self.model, self.device = self.load_model()
        self.input_name = self.model.get_inputs()[0].name
        self.output_names = [output.name for output in self.model.get_outputs()]
        self._padded_bgr = None
        self._input_blob = None
        self._normalization_scale = np.float32(1.0 / 255.0)
        self._use_iobinding = self.device == "CUDAExecutionProvider" and hasattr(self.model, "io_binding")
        self._warmup()

    def _allocate_preprocess_buffers(self):
        padded_shape = (self.input_size[0], self.input_size[1], 3)
        input_shape = (1, 3, self.input_size[0], self.input_size[1])
        if self._padded_bgr is None or self._padded_bgr.shape != padded_shape:
            self._padded_bgr = np.empty(padded_shape, dtype=np.uint8)
        if self._input_blob is None or self._input_blob.shape != input_shape:
            self._input_blob = np.empty(input_shape, dtype=np.float32)

    def _warmup(self):
        self._allocate_preprocess_buffers()
        self._padded_bgr.fill(128)
        np.multiply(
            self._padded_bgr[:, :, ::-1].transpose(2, 0, 1),
            self._normalization_scale,
            out=self._input_blob[0],
            casting="unsafe",
        )
        for _ in range(2):
            if self._use_iobinding:
                self._run_with_iobinding(self._input_blob)
            else:
                self.model.run(self.output_names, {self.input_name: self._input_blob})

    def load_model(self):
        available_providers = ort.get_available_providers()
        if (
            self.preferred_device in ("gpu", "auto")
            and "CUDAExecutionProvider" in available_providers
            and not get_missing_cuda_runtime_dependencies()
        ):
            preload_onnxruntime_gpu_dlls()
        providers, selected_provider, fallback_reason, available_providers = build_onnx_providers(
            self.preferred_device,
            self.preferred_backend,
            available_providers=available_providers,
        )
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        configure_onnx_session_options(ort, so)
        model = ort.InferenceSession(self.model_path, sess_options=so, providers=providers)
        active_provider = model.get_providers()[0]
        log_onnx_provider_status(
            self.model_path,
            active_provider,
            available_providers=available_providers,
            fallback_reason=fallback_reason if active_provider != "CUDAExecutionProvider" else "",
        )

        return model, active_provider

    @staticmethod
    def _ensure_bgr_image(img):
        if img is None:
            raise ValueError("detect_objects received an empty frame.")

        if isinstance(img, np.ndarray):
            arr = img
        elif isinstance(img, Image.Image):
            arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            return arr
        else:
            arr = np.asarray(img)
            if not isinstance(arr, np.ndarray) or arr.size == 0:
                raise TypeError(f"Unsupported frame type for detection: {type(img)}")
            # FrameData and similar wrappers store RGB data and expose it via np.asarray().
            if hasattr(img, "arr"):
                if arr.ndim == 3 and arr.shape[2] == 3:
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Unsupported frame shape for detection: {arr.shape}")
        return arr

    def preprocess_image(self, img):
        img = self._ensure_bgr_image(img)

        h, w, _ = img.shape
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        new_w = int(w * scale)
        new_h = int(h * scale)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        self._allocate_preprocess_buffers()

        self._padded_bgr.fill(128)
        self._padded_bgr[:new_h, :new_w, :] = resized_img
        np.multiply(
            self._padded_bgr[:, :, ::-1].transpose(2, 0, 1),
            self._normalization_scale,
            out=self._input_blob[0],
            casting="unsafe",
        )
        return self._input_blob, new_w, new_h

    @staticmethod
    def _xywh_to_xyxy(boxes):
        converted = np.empty_like(boxes)
        half_width = boxes[:, 2] * 0.5
        half_height = boxes[:, 3] * 0.5
        converted[:, 0] = boxes[:, 0] - half_width
        converted[:, 1] = boxes[:, 1] - half_height
        converted[:, 2] = boxes[:, 0] + half_width
        converted[:, 3] = boxes[:, 1] + half_height
        return converted

    @staticmethod
    def _nms_numpy(boxes, scores, iou_thres):
        if boxes.size == 0:
            return np.empty((0,), dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            remaining = order[1:]
            xx1 = np.maximum(x1[i], x1[remaining])
            yy1 = np.maximum(y1[i], y1[remaining])
            xx2 = np.minimum(x2[i], x2[remaining])
            yy2 = np.minimum(y2[i], y2[remaining])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h
            union = areas[i] + areas[remaining] - intersection
            iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
            order = remaining[iou <= iou_thres]

        return np.asarray(keep, dtype=np.int32)

    def postprocess(self, preds, orig_img_shape, resized_shape, conf_tresh=0.6):
        max_det = 300
        if preds.ndim == 2:
            preds = np.expand_dims(preds, axis=0)

        expected_attrs_no_objectness = 4 + len(self.classes)
        expected_attrs_with_objectness = 5 + len(self.classes)

        if preds.ndim != 3:
            return []

        if preds.shape[1] in (expected_attrs_no_objectness, expected_attrs_with_objectness) and preds.shape[2] not in (expected_attrs_no_objectness, expected_attrs_with_objectness):
            preds = np.transpose(preds, (0, 2, 1))

        if preds.shape[2] == expected_attrs_no_objectness:
            boxes_xywh = preds[..., :4]
            class_scores = preds[..., 4:]
        elif preds.shape[2] == expected_attrs_with_objectness:
            boxes_xywh = preds[..., :4]
            class_scores = preds[..., 5:] * preds[..., 4:5]
        else:
            return []

        orig_h, orig_w = orig_img_shape
        resized_w, resized_h = resized_shape

        scale_w = orig_w / resized_w
        scale_h = orig_h / resized_h

        results = []
        for image_boxes_xywh, image_class_scores in zip(boxes_xywh, class_scores):
            if image_boxes_xywh.size == 0:
                continue

            class_ids = np.argmax(image_class_scores, axis=1)
            confidences = image_class_scores[np.arange(image_class_scores.shape[0]), class_ids]
            keep_mask = confidences > conf_tresh
            if not np.any(keep_mask):
                continue

            filtered_boxes = self._xywh_to_xyxy(image_boxes_xywh[keep_mask]).astype(np.float32, copy=False)
            filtered_scores = confidences[keep_mask].astype(np.float32, copy=False)
            filtered_class_ids = class_ids[keep_mask]
            detections = []

            for class_id in np.unique(filtered_class_ids):
                class_mask = filtered_class_ids == class_id
                class_boxes = filtered_boxes[class_mask]
                class_scores_filtered = filtered_scores[class_mask]
                keep_indices = self._nms_numpy(class_boxes, class_scores_filtered, iou_thres=0.6)
                if keep_indices.size == 0:
                    continue

                selected_boxes = class_boxes[keep_indices]
                selected_scores = class_scores_filtered[keep_indices, None]
                selected_class_ids = np.full((keep_indices.size, 1), class_id, dtype=np.float32)
                detections.append(np.concatenate((selected_boxes, selected_scores, selected_class_ids), axis=1))

            if detections:
                pred = np.vstack(detections)
                pred = pred[np.argsort(-pred[:, 4])]
                if pred.shape[0] > max_det:
                    pred = pred[:max_det]
                pred[:, 0] *= scale_w
                pred[:, 1] *= scale_h
                pred[:, 2] *= scale_w
                pred[:, 3] *= scale_h
                results.append(pred)

        return results

    def detect_objects(self, img, conf_tresh=0.6):
        started_at = time.perf_counter()
        img = self._ensure_bgr_image(img)
        orig_h, orig_w = img.shape[:2]
        orig_img_shape = (orig_h, orig_w)

        preprocess_started_at = time.perf_counter()
        preprocessed_img, resized_w, resized_h = self.preprocess_image(img)
        resized_shape = (resized_w, resized_h)
        record_timing(f"detect_preprocess:{os.path.basename(self.model_path)}", time.perf_counter() - preprocess_started_at, print_every=60)

        inference_started_at = time.perf_counter()
        if self._use_iobinding:
            outputs = self._run_with_iobinding(preprocessed_img)
        else:
            outputs = self.model.run(self.output_names, {self.input_name: preprocessed_img})
        record_timing(f"detect_inference:{os.path.basename(self.model_path)}", time.perf_counter() - inference_started_at, print_every=60)

        postprocess_started_at = time.perf_counter()
        detections = self.postprocess(outputs[0], orig_img_shape, resized_shape, conf_tresh)
        record_timing(f"detect_postprocess:{os.path.basename(self.model_path)}", time.perf_counter() - postprocess_started_at, print_every=60)

        results = {}
        for detection in detections:
            for *xyxy, conf, cls in detection:
                x1, y1, x2, y2 = map(int, xyxy)
                class_id = int(cls)
                class_name = self.classes[class_id]

                if class_id in self.ignore_classes or class_name in self.ignore_classes:
                    continue
                if class_name not in results:
                    results[class_name] = []
                results[class_name].append([x1, y1, x2, y2])

        record_timing(f"detect:{os.path.basename(self.model_path)}", time.perf_counter() - started_at, print_every=60)
        return results

    def _run_with_iobinding(self, input_blob):
        try:
            io_binding = self.model.io_binding()
            io_binding.bind_cpu_input(self.input_name, input_blob)
            for output_name in self.output_names:
                io_binding.bind_output(output_name, "cpu")
            self.model.run_with_iobinding(io_binding)
            return io_binding.copy_outputs_to_cpu()
        except Exception:
            self._use_iobinding = False
            return self.model.run(self.output_names, {self.input_name: input_blob})


