"""Phase 1 — FER-2013 dataset preparation.

Accepts either public layout of FER-2013 placed anywhere under data/raw/:
  * CSV layout     : fer2013.csv with columns emotion, pixels, Usage
                     (Training -> train, PublicTest -> val, PrivateTest -> test)
  * Folder layout  : train/<class>/*.png and test/<class>/*.png (Kaggle msambare/fer2013);
                     a stratified validation split is carved from train.

Cleaning (training split only, so val/test stay comparable with published results):
  * near-blank images (pixel std below a threshold)
  * training images that are byte-identical to a val/test image (leakage)
  * exact duplicates inside the training split (first copy kept)

Outputs:
  data/processed/fer2013.npz          uint8 images (N, 48, 48) + int64 labels per split
  data/processed/dataset_report.json  counts, imbalance ratio, class weights, cleaning stats
  reports/figures/class_distribution.png
  reports/figures/sample_grid.png

Run from the project root:
    python -m scripts.prepare_dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # file output only; no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from utils.logger import get_logger

log = get_logger("prepare_dataset")

FOLDER_TO_INDEX = {name.lower(): i for i, name in enumerate(config.EMOTION_LABELS)}
USAGE_TO_SPLIT = {"Training": "train", "PublicTest": "val", "PrivateTest": "test"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS = ("train", "val", "test")
PIXELS_PER_IMAGE = config.IMG_SIZE * config.IMG_SIZE

Split = tuple[np.ndarray, np.ndarray]  # (images uint8 [N,48,48], labels int64 [N])


class DatasetNotFoundError(RuntimeError):
    """Raised when no recognisable FER-2013 layout exists under the raw directory."""


# ---------------------------------------------------------------------------
# Format detection and loading
# ---------------------------------------------------------------------------
def _has_class_folders(directory: Path) -> bool:
    names = {p.name.lower() for p in directory.iterdir() if p.is_dir()}
    return set(FOLDER_TO_INDEX) <= names


def detect_format(raw_dir: Path) -> tuple[str, Path]:
    """Return ("csv", csv_path) or ("folders", dir_containing_train_and_test)."""
    if not raw_dir.exists():
        raise DatasetNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    csv_files = sorted(raw_dir.rglob("fer2013.csv"))
    if csv_files:
        return "csv", csv_files[0]

    for train_dir in sorted(raw_dir.rglob("train")):
        test_dir = train_dir.parent / "test"
        if train_dir.is_dir() and test_dir.is_dir() and _has_class_folders(train_dir):
            return "folders", train_dir.parent

    raise DatasetNotFoundError(
        f"No FER-2013 data found under {raw_dir}. Expected fer2013.csv or "
        "train/ and test/ folders containing angry, disgust, fear, happy, neutral, sad, surprise."
    )


def load_csv(csv_path: Path) -> tuple[dict[str, Split], int]:
    """Load the original challenge CSV. Returns (splits, number_of_rejected_rows)."""
    df = pd.read_csv(csv_path)
    missing = {"emotion", "pixels", "Usage"} - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")

    buckets: dict[str, tuple[list[np.ndarray], list[int]]] = {s: ([], []) for s in SPLITS}
    rejected = 0
    for emotion, pixels, usage in zip(df["emotion"], df["pixels"], df["Usage"]):
        split = USAGE_TO_SPLIT.get(str(usage).strip())
        try:
            values = np.array(str(pixels).split(), dtype=np.int16)
        except ValueError:
            values = np.empty(0, dtype=np.int16)
        if (
            split is None
            or values.size != PIXELS_PER_IMAGE
            or values.min() < 0
            or values.max() > 255
            or not 0 <= int(emotion) < config.NUM_CLASSES
        ):
            rejected += 1
            continue
        buckets[split][0].append(values.astype(np.uint8).reshape(config.IMG_SIZE, config.IMG_SIZE))
        buckets[split][1].append(int(emotion))

    splits: dict[str, Split] = {}
    for split, (images, labels) in buckets.items():
        if not images:
            raise ValueError(f"CSV contains no valid rows for split '{split}'.")
        splits[split] = (np.stack(images), np.asarray(labels, dtype=np.int64))
    return splits, rejected


def _read_gray(path: Path) -> np.ndarray | None:
    # np.fromfile + imdecode handles non-ASCII Windows paths that cv2.imread cannot.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def load_folders(root: Path) -> tuple[dict[str, Split], int]:
    """Load train/ and test/ image folders. Returns ({'train','test'}, unreadable_count)."""
    splits: dict[str, Split] = {}
    unreadable = 0
    for split in ("train", "test"):
        images: list[np.ndarray] = []
        labels: list[int] = []
        for class_dir in sorted((root / split).iterdir()):
            if not class_dir.is_dir():
                continue
            label = FOLDER_TO_INDEX.get(class_dir.name.lower())
            if label is None:
                log.warning("Skipping unknown class folder: %s", class_dir)
                continue
            for file in sorted(class_dir.iterdir()):
                if file.suffix.lower() not in IMAGE_EXTS:
                    continue
                img = _read_gray(file)
                if img is None:
                    unreadable += 1
                    continue
                if img.shape != (config.IMG_SIZE, config.IMG_SIZE):
                    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE), interpolation=cv2.INTER_AREA)
                images.append(img)
                labels.append(label)
        if not images:
            raise ValueError(f"No readable images found in {root / split}")
        splits[split] = (np.stack(images).astype(np.uint8), np.asarray(labels, dtype=np.int64))
    return splits, unreadable


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def find_low_variance(images: np.ndarray, std_threshold: float) -> np.ndarray:
    """Boolean mask of near-constant (blank / corrupt) images."""
    return images.reshape(len(images), -1).std(axis=1) < std_threshold


def image_hashes(images: np.ndarray) -> np.ndarray:
    return np.array([hashlib.md5(img.tobytes()).hexdigest() for img in images])


def clean_training_split(
    train: Split, eval_images: list[np.ndarray], std_threshold: float
) -> tuple[Split, dict[str, int]]:
    """Remove blank, leaked and duplicated images from the training split."""
    x, y = train
    blank = find_low_variance(x, std_threshold)

    hashes = image_hashes(x)
    eval_hashes: set[str] = set()
    for images in eval_images:
        eval_hashes.update(image_hashes(images))
    leaked = np.array([h in eval_hashes for h in hashes], dtype=bool)

    _, first_index = np.unique(hashes, return_index=True)
    duplicate = np.ones(len(x), dtype=bool)
    duplicate[first_index] = False

    remove = blank | leaked | duplicate
    stats = {
        "train_before": int(len(x)),
        "blank_or_low_variance": int(blank.sum()),
        "identical_to_val_or_test": int(leaked.sum()),
        "duplicate_within_train": int(duplicate.sum()),
        "total_removed": int(remove.sum()),
        "train_after": int((~remove).sum()),
    }
    return (x[~remove], y[~remove]), stats


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def class_counts(labels: np.ndarray) -> dict[str, int]:
    return {name: int((labels == i).sum()) for i, name in enumerate(config.EMOTION_LABELS)}


def build_dataset(
    raw_dir: Path,
    val_fraction: float = config.VAL_FRACTION,
    std_threshold: float = config.LOW_VARIANCE_STD_THRESHOLD,
    seed: int = config.RANDOM_SEED,
) -> tuple[dict[str, Split], dict]:
    fmt, source = detect_format(raw_dir)
    log.info("Detected %s layout at %s", fmt, source)

    if fmt == "csv":
        splits, rejected = load_csv(source)
        train, cleaning = clean_training_split(
            splits["train"], [splits["val"][0], splits["test"][0]], std_threshold
        )
        splits["train"] = train
        val_origin = "PublicTest rows of fer2013.csv"
    else:
        loaded, rejected = load_folders(source)
        train, cleaning = clean_training_split(loaded["train"], [loaded["test"][0]], std_threshold)
        x_tr, x_val, y_tr, y_val = train_test_split(
            train[0], train[1], test_size=val_fraction, stratify=train[1], random_state=seed
        )
        splits = {"train": (x_tr, y_tr), "val": (x_val, y_val), "test": loaded["test"]}
        val_origin = f"stratified {val_fraction:.0%} of cleaned train (seed={seed})"

    y_train = splits["train"][1]
    present = np.unique(y_train)
    if len(present) != config.NUM_CLASSES:
        raise ValueError(f"Training split is missing classes; found indices {present.tolist()}")

    weights = compute_class_weight("balanced", classes=np.arange(config.NUM_CLASSES), y=y_train)
    train_counts = class_counts(y_train)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_format": fmt,
        "source_path": str(source),
        "image_size": config.IMG_SIZE,
        "labels": list(config.EMOTION_LABELS),
        "validation_split_origin": val_origin,
        "rejected_rows_or_unreadable_files": int(rejected),
        "cleaning_train_only": cleaning,
        "eval_low_variance_kept": {
            s: int(find_low_variance(splits[s][0], std_threshold).sum()) for s in ("val", "test")
        },
        "splits": {
            s: {"total": int(len(splits[s][1])), "per_class": class_counts(splits[s][1])} for s in SPLITS
        },
        "train_imbalance_ratio_max_over_min": round(max(train_counts.values()) / min(train_counts.values()), 2),
        "class_weights_balanced": {
            name: round(float(w), 4) for name, w in zip(config.EMOTION_LABELS, weights)
        },
    }
    return splits, report


def save_outputs(splits: dict[str, Split], report: dict, npz_path: Path, report_path: Path) -> None:
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X_train=splits["train"][0], y_train=splits["train"][1],
        X_val=splits["val"][0], y_val=splits["val"][1],
        X_test=splits["test"][0], y_test=splits["test"][1],
        labels=np.array(config.EMOTION_LABELS),
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Saved %s (%.1f MB)", npz_path, npz_path.stat().st_size / 1024**2)
    log.info("Saved %s", report_path)


def plot_class_distribution(report: dict, out_path: Path) -> None:
    labels = report["labels"]
    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(10, 5))
    for offset, split in zip((-width, 0, width), SPLITS):
        counts = [report["splits"][split]["per_class"][name] for name in labels]
        bars = ax.bar(x + offset, counts, width, label=f"{split} (n={report['splits'][split]['total']})")
        ax.bar_label(bars, fontsize=7, padding=2)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Images")
    ax.set_title("FER-2013 class distribution after cleaning")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", out_path)


def plot_sample_grid(train: Split, out_path: Path, per_class: int = 8, seed: int = config.RANDOM_SEED) -> None:
    x, y = train
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(config.NUM_CLASSES, per_class, figsize=(per_class * 1.2, config.NUM_CLASSES * 1.3))
    for row, name in enumerate(config.EMOTION_LABELS):
        idx = np.flatnonzero(y == row)
        chosen = rng.choice(idx, size=min(per_class, len(idx)), replace=False) if len(idx) else []
        for col in range(per_class):
            ax = axes[row, col]
            ax.axis("off")
            if col < len(chosen):
                ax.imshow(x[chosen[col]], cmap="gray", vmin=0, vmax=255)
        axes[row, 0].set_title(name, loc="left", fontsize=9)
    fig.suptitle("Random training samples per class", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved %s", out_path)


def print_summary(report: dict) -> None:
    labels = report["labels"]
    header = f"{'Class':<10}" + "".join(f"{s:>8}" for s in SPLITS) + f"{'weight':>9}"
    print("\n" + header)
    print("-" * len(header))
    for name in labels:
        row = f"{name:<10}" + "".join(f"{report['splits'][s]['per_class'][name]:>8}" for s in SPLITS)
        print(row + f"{report['class_weights_balanced'][name]:>9.3f}")
    print("-" * len(header))
    print(f"{'Total':<10}" + "".join(f"{report['splits'][s]['total']:>8}" for s in SPLITS))
    print(f"\nTrain imbalance ratio (max/min): {report['train_imbalance_ratio_max_over_min']}")
    print(f"Cleaning (train only): {report['cleaning_train_only']}")
    print(f"Low-variance images kept in eval splits: {report['eval_low_variance_kept']}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare FER-2013 for EmotionLens AI.")
    parser.add_argument("--raw-dir", type=Path, default=config.RAW_DATA_DIR)
    parser.add_argument("--out", type=Path, default=config.DATASET_NPZ)
    parser.add_argument("--val-fraction", type=float, default=config.VAL_FRACTION)
    parser.add_argument("--std-threshold", type=float, default=config.LOW_VARIANCE_STD_THRESHOLD)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    config.ensure_dirs()
    try:
        splits, report = build_dataset(args.raw_dir, args.val_fraction, args.std_threshold, args.seed)
    except DatasetNotFoundError as exc:
        log.error("%s", exc)
        log.error("See Phase 1 'Dataset acquisition' for download instructions.")
        return 1
    except ValueError as exc:
        log.error("Dataset is malformed: %s", exc)
        return 1

    save_outputs(splits, report, args.out, args.out.with_name("dataset_report.json"))
    plot_class_distribution(report, config.FIGURES_DIR / "class_distribution.png")
    plot_sample_grid(splits["train"], config.FIGURES_DIR / "sample_grid.png", seed=args.seed)
    print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
