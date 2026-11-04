"""CPU/GPU detection and TensorFlow runtime configuration."""
from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import dataclass, field
from functools import lru_cache

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from utils.logger import get_logger

log = get_logger("device")


@dataclass
class DeviceInfo:
    device: str                      # "GPU" or "CPU"
    gpu_names: list[str] = field(default_factory=list)
    tensorflow_version: str = "not installed"
    onnxruntime_version: str = "not installed"
    onnx_providers: list[str] = field(default_factory=list)
    cpu_name: str = ""
    physical_cores: int | None = None
    logical_cores: int | None = None
    ram_gb: float | None = None
    os_name: str = ""


@lru_cache(maxsize=1)
def configure_tensorflow() -> DeviceInfo:
    """Detect compute devices once per process (ONNX Runtime providers and, if installed,
    TensorFlow GPUs with memory growth enabled).

    Falls back to CPU silently: every model in this project is small enough to run
    in real time on a laptop CPU.
    """
    import psutil

    info = DeviceInfo(
        device="CPU",
        cpu_name=platform.processor() or platform.machine(),
        physical_cores=psutil.cpu_count(logical=False),
        logical_cores=psutil.cpu_count(logical=True),
        ram_gb=round(psutil.virtual_memory().total / 1024**3, 1),
        os_name=f"{platform.system()} {platform.release()}",
    )
    try:
        import onnxruntime as ort

        info.onnxruntime_version = ort.__version__
        info.onnx_providers = ort.get_available_providers()
        if any(p in info.onnx_providers for p in ("CUDAExecutionProvider", "DmlExecutionProvider")):
            info.device = "GPU"
    except Exception as exc:
        log.warning("onnxruntime unavailable: %s", exc)

    # TensorFlow is optional on the laptop (the app uses ONNX Runtime). Checking find_spec first
    # avoids a slow, failing import where Windows Smart App Control blocks TensorFlow's DLLs.
    if importlib.util.find_spec("tensorflow") is None:
        return info
    try:
        import tensorflow as tf
    except Exception as exc:  # DLL errors on Windows surface as ImportError/OSError
        log.warning("TensorFlow is installed but could not be imported: %s", exc)
        return info

    info.tensorflow_version = tf.__version__
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:  # raised if TF already initialised the GPU
            log.warning("Could not enable memory growth on %s: %s", gpu.name, exc)
    if gpus:
        info.device = "GPU"
        info.gpu_names = [g.name for g in gpus]
        log.info("TensorFlow GPU: %s", info.gpu_names)
    else:
        log.info("No GPU visible to TensorFlow; using CPU (%s logical cores).", info.logical_cores)
    return info
