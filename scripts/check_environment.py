"""Phase 1 — verify the Python environment, compute devices, MediaPipe and webcam.

Run from the project root:
    python -m scripts.check_environment
    python -m scripts.check_environment --skip-webcam
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys

# Silence TensorFlow's C++ start-up chatter before it is imported anywhere.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import utils.native_preload  # noqa: E402,F401  (load order: before pandas/pyarrow)

import config
from utils.logger import get_logger

log = get_logger("check_env")

# (import name, pip package name)
REQUIRED_PACKAGES = [
    ("mediapipe", "mediapipe"),
    ("onnxruntime", "onnxruntime"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("cv2", "opencv-contrib-python"),
    ("streamlit", "streamlit"),
    ("plotly", "plotly"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("psutil", "psutil"),
    ("fpdf", "fpdf2"),
    ("pyarrow", "pyarrow"),
]


def check_python() -> bool:
    version = sys.version_info
    ok = (3, 11) <= (version.major, version.minor) <= (3, 12)
    log.info("Python %s on %s (%s)", platform.python_version(), platform.system(), platform.machine())
    if not ok:
        log.error("Python 3.11 or 3.12 is required; mediapipe/tensorflow wheels do not exist for this version.")
    return ok


def check_packages() -> bool:
    all_ok = True
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(module_name)
            log.info("  [OK]   %-22s %s", pip_name, getattr(module, "__version__", "unknown"))
        except Exception as exc:  # ImportError, or DLL load failures on Windows
            all_ok = False
            log.error("  [FAIL] %-22s %s: %s", pip_name, type(exc).__name__, exc)
    return all_ok


def check_cv2_conflict() -> bool:
    """opencv-python and opencv-contrib-python installed together break `cv2`."""
    from importlib.metadata import PackageNotFoundError, version

    installed = []
    for dist in ("opencv-python", "opencv-contrib-python", "opencv-python-headless"):
        try:
            installed.append(f"{dist}=={version(dist)}")
        except PackageNotFoundError:
            pass
    if len(installed) > 1:
        log.warning("Multiple OpenCV distributions installed: %s", ", ".join(installed))
        log.warning("Fix: pip uninstall -y opencv-python opencv-python-headless && "
                    "pip install --force-reinstall opencv-contrib-python==4.10.0.84")
        return False
    return True


def check_compute() -> None:
    from utils.device import configure_tensorflow

    info = configure_tensorflow()
    log.info("CPU: %s | physical cores: %s | logical cores: %s | RAM: %s GB",
             info.cpu_name or "unknown", info.physical_cores, info.logical_cores, info.ram_gb)
    log.info("ONNX Runtime %s providers: %s", info.onnxruntime_version, info.onnx_providers)
    log.info("TensorFlow: %s (only needed for training, which runs in Colab)", info.tensorflow_version)


def check_mediapipe() -> bool:
    import mediapipe as mp
    import numpy as np

    try:
        with mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as detector:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            result = detector.process(blank)
        n = len(result.detections) if result.detections else 0
        log.info("MediaPipe face detector loaded (detections on a blank frame: %d, expected 0)", n)
        return True
    except Exception as exc:
        log.error("MediaPipe face detector failed to initialise: %s", exc)
        return False


def check_webcam(index: int) -> bool:
    import cv2

    # DirectShow opens much faster than MSMF on most Windows laptops.
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    try:
        if not cap.isOpened():
            log.error("Could not open webcam index %d (in use by another app, or privacy setting blocks it).", index)
            return False
        ok, frame = False, None
        for _ in range(10):  # first frames are often empty while the sensor warms up
            ok, frame = cap.read()
            if ok and frame is not None:
                break
        if not ok or frame is None:
            log.error("Webcam opened but returned no frames.")
            return False
        log.info("Webcam %d OK: frame %dx%d", index, frame.shape[1], frame.shape[0])
        return True
    finally:
        cap.release()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the EmotionLens AI environment.")
    parser.add_argument("--skip-webcam", action="store_true", help="Do not try to open the camera.")
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    config.ensure_dirs()
    results: dict[str, bool] = {}

    log.info("=== Python ===")
    results["python"] = check_python()

    log.info("=== Packages ===")
    results["packages"] = check_packages()
    results["opencv_single_install"] = check_cv2_conflict()

    if results["packages"]:
        log.info("=== Compute ===")
        check_compute()
        log.info("=== MediaPipe ===")
        results["mediapipe"] = check_mediapipe()
        if not args.skip_webcam:
            log.info("=== Webcam ===")
            results["webcam"] = check_webcam(args.camera_index)

    log.info("=== Summary ===")
    for name, ok in results.items():
        log.info("  %-22s %s", name, "PASS" if ok else "FAIL")

    all_ok = all(results.values())
    log.info("Environment ready." if all_ok else "Fix the FAIL items above before continuing.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
