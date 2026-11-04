"""CPU/GPU detection and TensorFlow runtime configuration."""
from __future__ import annotations

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
    tensorflow_version: str = "unavailable"
    cpu_name: str = ""
    physical_cores: int | None = None
    logical_cores: int | None = None
    ram_gb: float | None = None
    os_name: str = ""


@lru_cache(maxsize=1)
def configure_tensorflow() -> DeviceInfo:
    """Detect devices once per process and enable GPU memory growth when a GPU exists.

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
        import tensorflow as tf
    except Exception as exc:  # DLL errors on Windows surface as ImportError/OSError
        log.error("TensorFlow could not be imported: %s", exc)
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
        log.info("Using GPU: %s", info.gpu_names)
    else:
        log.info("No GPU visible to TensorFlow; using CPU (%s logical cores).", info.logical_cores)
    return info
