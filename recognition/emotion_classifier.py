"""Expression classifier: loads a trained .keras checkpoint and runs batched inference.

Inference optimisations:
  * every face in a frame is classified in ONE batched forward pass;
  * the forward pass is wrapped in a tf.function with a fixed input signature, so the
    graph is traced once and reused (no Python overhead or retracing per frame);
  * a warm-up call at load time moves graph tracing out of the first live frame.
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


class ModelNotFoundError(FileNotFoundError):
    pass


def list_models() -> list[str]:
    return sorted(p.stem for p in config.CHECKPOINT_DIR.glob("*.keras"))


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
    if not name or not (config.CHECKPOINT_DIR / f"{name}.keras").exists():
        raise ModelNotFoundError(
            "No trained model found in models/checkpoints/. "
            "Train one first: python -m models.train --model cnn"
        )
    return name


class EmotionClassifier:
    def __init__(self, model_name: str = "") -> None:
        import tensorflow as tf
        from tensorflow import keras

        from utils.device import configure_tensorflow

        self.device = configure_tensorflow()
        self.name = resolve_model_name(model_name)
        self.path: Path = config.CHECKPOINT_DIR / f"{self.name}.keras"
        self.labels = config.EMOTION_LABELS

        start = time.perf_counter()
        self.model = keras.models.load_model(self.path, compile=False)
        model = self.model

        @tf.function(
            input_signature=[tf.TensorSpec([None, config.IMG_SIZE, config.IMG_SIZE, 1], tf.float32)],
            reduce_retracing=True,
        )
        def _infer(x):
            return model(x, training=False)

        self._infer = _infer
        self._tf = tf
        self.predict_batch([np.zeros((config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)])  # warm-up
        self.load_seconds = time.perf_counter() - start
        self.last_latency_ms = 0.0
        log.info("Loaded model '%s' on %s in %.2fs", self.name, self.device.device, self.load_seconds)

    def predict_batch(self, faces: list[np.ndarray]) -> np.ndarray:
        """faces: list of uint8 (48, 48) grayscale crops -> (N, 7) probabilities."""
        if not faces:
            return np.zeros((0, config.NUM_CLASSES), dtype=np.float32)
        batch = np.stack(faces).astype(np.float32)[..., np.newaxis]
        start = time.perf_counter()
        probs = self._infer(self._tf.constant(batch)).numpy()
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return probs

    def predict_one(self, face: np.ndarray) -> np.ndarray:
        return self.predict_batch([face])[0]
