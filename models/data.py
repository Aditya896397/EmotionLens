"""Load the prepared FER-2013 arrays and build tf.data pipelines."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config


class PreparedDataMissingError(FileNotFoundError):
    pass


@dataclass
class FERData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

    def split(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        return getattr(self, f"x_{name}"), getattr(self, f"y_{name}")


def load_prepared(npz_path: Path = config.DATASET_NPZ) -> FERData:
    """Load uint8 arrays and add the channel axis -> (N, 48, 48, 1) float32 in [0, 255]."""
    if not npz_path.exists():
        raise PreparedDataMissingError(
            f"{npz_path} not found. Run `python -m scripts.prepare_dataset` first."
        )
    data = np.load(npz_path)

    def prep(x: np.ndarray) -> np.ndarray:
        return x[..., np.newaxis].astype(np.float32)

    return FERData(
        prep(data["X_train"]), data["y_train"].astype(np.int64),
        prep(data["X_val"]), data["y_val"].astype(np.int64),
        prep(data["X_test"]), data["y_test"].astype(np.int64),
    )


def load_class_weights(report_path: Path = config.DATASET_REPORT) -> dict[int, float] | None:
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    weights = report.get("class_weights_balanced")
    if not weights:
        return None
    return {i: float(weights[name]) for i, name in enumerate(config.EMOTION_LABELS)}


def make_dataset(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = config.RANDOM_SEED):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
