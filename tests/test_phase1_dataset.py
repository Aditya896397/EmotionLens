"""Phase 1 tests: dataset loading and cleaning on small synthetic FER-2013 copies.

Run from the project root:
    python -m pytest tests -v
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import config
from scripts import prepare_dataset as prep

RNG = np.random.default_rng(0)


def _random_face() -> np.ndarray:
    return RNG.integers(0, 256, size=(config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)


def _to_pixels(img: np.ndarray) -> str:
    return " ".join(map(str, img.flatten().tolist()))


@pytest.fixture
def csv_dataset(tmp_path: Path) -> Path:
    rows = []
    for usage, per_class in (("Training", 6), ("PublicTest", 2), ("PrivateTest", 2)):
        for label in range(config.NUM_CLASSES):
            for _ in range(per_class):
                rows.append({"emotion": label, "pixels": _to_pixels(_random_face()), "Usage": usage})
    leaked = _random_face()
    rows.append({"emotion": 3, "pixels": _to_pixels(leaked), "Usage": "PrivateTest"})
    rows.append({"emotion": 3, "pixels": _to_pixels(leaked), "Usage": "Training"})       # leakage
    blank = np.full((config.IMG_SIZE, config.IMG_SIZE), 128, dtype=np.uint8)
    rows.append({"emotion": 0, "pixels": _to_pixels(blank), "Usage": "Training"})        # blank
    rows.append({"emotion": 1, "pixels": "1 2 3", "Usage": "Training"})                  # malformed

    raw = tmp_path / "raw" / "fer2013"
    raw.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(raw / "fer2013.csv", index=False)
    return tmp_path / "raw"


@pytest.fixture
def folder_dataset(tmp_path: Path) -> Path:
    raw = tmp_path / "raw" / "archive"
    for split, per_class in (("train", 10), ("test", 3)):
        for name in config.EMOTION_LABELS:
            class_dir = raw / split / name.lower()
            class_dir.mkdir(parents=True)
            for i in range(per_class):
                cv2.imwrite(str(class_dir / f"{i}.png"), _random_face())
    return tmp_path / "raw"


def test_detects_csv_layout(csv_dataset: Path) -> None:
    fmt, path = prep.detect_format(csv_dataset)
    assert fmt == "csv" and path.name == "fer2013.csv"


def test_detects_folder_layout(folder_dataset: Path) -> None:
    fmt, path = prep.detect_format(folder_dataset)
    assert fmt == "folders" and (path / "train").is_dir()


def test_missing_dataset_raises(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(prep.DatasetNotFoundError):
        prep.detect_format(tmp_path / "raw")


def test_csv_pipeline_cleans_training_split(csv_dataset: Path) -> None:
    splits, report = prep.build_dataset(csv_dataset)
    assert report["rejected_rows_or_unreadable_files"] == 1
    assert report["cleaning_train_only"]["blank_or_low_variance"] == 1
    assert report["cleaning_train_only"]["identical_to_val_or_test"] == 1
    assert report["splits"]["train"]["total"] == 6 * config.NUM_CLASSES
    assert report["splits"]["val"]["total"] == 2 * config.NUM_CLASSES
    assert report["splits"]["test"]["total"] == 2 * config.NUM_CLASSES + 1
    x_train, y_train = splits["train"]
    assert x_train.dtype == np.uint8 and x_train.shape[1:] == (48, 48)
    assert y_train.dtype == np.int64


def test_folder_pipeline_creates_stratified_val(folder_dataset: Path) -> None:
    splits, report = prep.build_dataset(folder_dataset, val_fraction=0.2)
    assert report["splits"]["val"]["total"] == 14
    assert report["splits"]["train"]["total"] == 56
    assert all(v == 2 for v in report["splits"]["val"]["per_class"].values())
    assert len(report["class_weights_balanced"]) == config.NUM_CLASSES


def test_save_outputs_round_trip(folder_dataset: Path, tmp_path: Path) -> None:
    splits, report = prep.build_dataset(folder_dataset, val_fraction=0.2)
    npz = tmp_path / "out" / "fer2013.npz"
    prep.save_outputs(splits, report, npz, npz.with_name("dataset_report.json"))
    data = np.load(npz)
    assert data["X_train"].shape == splits["train"][0].shape
    assert list(data["labels"]) == list(config.EMOTION_LABELS)


def test_save_outputs_leaves_public_report_untouched(folder_dataset: Path, tmp_path: Path) -> None:
    """Regression: save_outputs once also rewrote reports/dataset_report.json with test data."""
    public = config.DATASET_REPORT_PUBLIC
    before = public.read_bytes() if public.exists() else None
    splits, report = prep.build_dataset(folder_dataset, val_fraction=0.2)
    npz = tmp_path / "x" / "fer2013.npz"
    prep.save_outputs(splits, report, npz, npz.with_name("dataset_report.json"))
    after = public.read_bytes() if public.exists() else None
    assert before == after
