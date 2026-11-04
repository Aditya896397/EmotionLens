"""Expression classifier: loads a trained model and runs batched inference.

Two interchangeable backends share one contract, (N, 48, 48, 1) float32 in [0, 255]
-> (N, 7) softmax probabilities:
  * ONNX Runtime (`<name>.onnx`) is the default on the laptop. Its DLLs are Microsoft-signed,
    so it runs where Windows Smart App Control blocks TensorFlow, and it is lighter and
    faster on CPU.
  * TensorFlow/Keras (`<name>.keras`) is used on the training machine (Colab/Linux) or
    anywhere TensorFlow imports.

Inference optimisations: every face in a frame goes through ONE batched forward pass;
the Keras path uses a tf.function with a fixed input signature (traced once); a warm-up
call at load time moves one-off initialisation out of the first live frame.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import config
from utils.logger import get_logger

log = get_logger("classifier")

BACKENDS = ("auto", "onnx", "keras")


class ModelNotFoundError(FileNotFoundError):
    pass


def list_models() -> list[str]:
    """Checkpoint names that have an .onnx and/or .keras file."""
    stems = {p.stem for p in config.CHECKPOINT_DIR.glob("*.onnx")}
    stems |= {p.stem for p in config.CHECKPOINT_DIR.glob("*.keras")}
    return sorted(stems)


def best_model_name() -> str | None:
    """Model with the highest measured test accuracy, else 'cnn', else any checkpoint."""
    models = list_models()
    if not models:
        return None
    scored = []
    for name in models:
        path = config.EXPERIMENTS_DIR / f"eval_{name}.json"
        if path.exists():
            try:
                scored.append((json.loads(path.read_text(encoding="utf-8"))["accuracy"], name))
            except (KeyError, json.JSONDecodeError):
                continue
    if scored:
        return max(scored)[1]
    return "cnn" if "cnn" in models else models[0]


def resolve_model_name(requested: str) -> str:
    name = requested or best_model_name()
    if not name or name not in list_models():
        raise ModelNotFoundError(
            "No trained model found in models/checkpoints/. Train one with the Colab notebook "
            "(notebooks/train_colab.ipynb) and copy the .onnx files into models/checkpoints/."
        )
    return name


def _pick_backend(name: str, backend: str) -> str:
    onnx_path = config.CHECKPOINT_DIR / f"{name}.onnx"
    keras_path = config.CHECKPOINT_DIR / f"{name}.keras"
    if backend == "onnx" or (backend == "auto" and onnx_path.exists()):
        if not onnx_path.exists():
            raise ModelNotFoundError(f"{onnx_path} not found. Export it with models/export_onnx.py.")
        return "onnx"
    if not keras_path.exists():
        raise ModelNotFoundError(f"{keras_path} not found.")
    return "keras"


class EmotionClassifier:
    def __init__(self, model_name: str = "", backend: str = "auto") -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.name = resolve_model_name(model_name)
        self.backend = _pick_backend(self.name, backend)
        self.labels = config.EMOTION_LABELS
        self.path: Path = config.CHECKPOINT_DIR / f"{self.name}.{self.backend}"
        self.last_latency_ms = 0.0

        start = time.perf_counter()
        if self.backend == "onnx":
            self._load_onnx()
        else:
            self._load_keras()
        self.predict_batch([np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)])  # warm-up
        self.load_seconds = time.perf_counter() - start
        log.info("Loaded model '%s' (%s, %s) in %.2fs", self.name, self.backend, self.device, self.load_seconds)

    # ------------------------------------------------------------------ loading
    def _load_onnx(self) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        providers = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in available]
        providers.append("CPUExecutionProvider")
        self._session = ort.InferenceSession(str(self.path), sess_options=opts, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        used = self._session.get_providers()[0]
        self.device = "GPU" if used != "CPUExecutionProvider" else "CPU"
        self.runtime_version = f"onnxruntime {ort.__version__}"

        def run(batch: np.ndarray) -> np.ndarray:
            return self._session.run(None, {self._input_name: batch})[0]

        self._run = run

    def _load_keras(self) -> None:
        import tensorflow as tf
        from tensorflow import keras

        from utils.device import configure_tensorflow

        self.device = configure_tensorflow().device
        self.runtime_version = f"tensorflow {tf.__version__}"
        model = keras.models.load_model(self.path, compile=False)
        self.model = model

        @tf.function(
            input_signature=[tf.TensorSpec([None, config.IMG_SIZE, config.IMG_SIZE, 1], tf.float32)],
            reduce_retracing=True,
        )
        def infer(x):
            return model(x, training=False)

        def run(batch: np.ndarray) -> np.ndarray:
            return infer(tf.constant(batch)).numpy()

        self._run = run

    # ------------------------------------------------------------------ inference
    def predict_batch(self, faces: list[np.ndarray]) -> np.ndarray:
        """faces: list of uint8 (48, 48) grayscale crops -> (N, 7) probabilities."""
        if not faces:
            return np.zeros((0, config.NUM_CLASSES), dtype=np.float32)
        batch = np.stack(faces).astype(np.float32)[..., np.newaxis]
        start = time.perf_counter()
        probs = self._run(batch)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return probs

    def predict_array(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Batched prediction over a whole array: uint8/float (N, 48, 48[, 1]) -> (N, 7)."""
        if x.ndim == 3:
            x = x[..., np.newaxis]
        x = x.astype(np.float32)
        out = [self._run(x[i:i + batch_size]) for i in range(0, len(x), batch_size)]
        return np.concatenate(out) if out else np.zeros((0, config.NUM_CLASSES), dtype=np.float32)

    def predict_one(self, face: np.ndarray) -> np.ndarray:
        return self.predict_batch([face])[0]
