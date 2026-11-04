"""Research experiments. Every table is computed from real model runs on this machine.

    python -m scripts.run_experiments all
    python -m scripts.run_experiments lighting --models cnn mobilenet
    python -m scripts.run_experiments smoothing --limit 300
    python -m scripts.run_experiments multiface
    python -m scripts.run_experiments preprocessing
    python -m scripts.run_experiments ablation

Experiments
  lighting       accuracy on the test set after synthetic darkening, with and without the
                 low-light enhancement used by the live pipeline
  preprocessing  test accuracy with raw input vs CLAHE vs global histogram equalisation
  smoothing      flip rate and accuracy of raw vs EMA vs moving-window predictions on
                 synthetic "webcam jitter" sequences built from static test images
  multiface      detection recall and end-to-end latency/FPS for frames with 1..9 faces
  ablation       collects every evaluated checkpoint (e.g. cnn, cnn_noaug, cnn_noweights)
                 into one table: effect of augmentation and class weighting

Outputs: reports/experiments/<name>.csv, <name>_notes.md and reports/figures/experiment_<name>.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import utils.native_preload  # noqa: E402,F401  (load order: before pandas/pyarrow)

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

import config
from models.data import PreparedDataMissingError, load_prepared
from recognition.emotion_classifier import EmotionClassifier, best_model_name, list_models
from recognition.temporal_analyzer import TemporalAnalyzer
from utils.image_utils import darken, enhance_low_light
from utils.logger import get_logger

log = get_logger("experiments")

LIGHTING_CONDITIONS = [  # (name, intensity factor, gamma)
    ("normal", 1.00, 1.0),
    ("dim", 0.60, 1.2),
    ("dark", 0.35, 1.5),
    ("very dark", 0.20, 1.8),
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def predict_array(clf: EmotionClassifier, x: np.ndarray, batch: int = 512) -> np.ndarray:
    """x: uint8 (N, 48, 48) -> probabilities (N, 7), on whichever backend clf uses."""
    return clf.predict_array(x, batch_size=batch)


def metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(axis=1)
    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)}


def test_set(limit: int | None, seed: int = config.RANDOM_SEED) -> tuple[np.ndarray, np.ndarray]:
    data = load_prepared()
    x = data.x_test[..., 0].astype(np.uint8)
    y = data.y_test
    if limit and limit < len(x):
        idx = np.random.default_rng(seed).choice(len(x), size=limit, replace=False)
        x, y = x[idx], y[idx]
    return x, y


def save(name: str, df: pd.DataFrame, notes: str) -> None:
    config.ensure_dirs()
    df.to_csv(config.EXPERIMENTS_DIR / f"{name}.csv", index=False)
    (config.EXPERIMENTS_DIR / f"{name}_notes.md").write_text(notes, encoding="utf-8")
    log.info("Saved reports/experiments/%s.csv", name)
    print(f"\n=== {name} ===\n{df.to_string(index=False)}\n")


def save_figure(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"experiment_{name}.png", dpi=150)
    plt.close(fig)


def resolve_models(names: list[str] | None) -> list[str]:
    available = list_models()
    if names:
        return [n for n in names if n in available]
    preferred = [n for n in ("cnn", "mobilenet") if n in available]
    return preferred or available[:2]


# ---------------------------------------------------------------------------
# experiments
# ---------------------------------------------------------------------------
def exp_lighting(model_names: list[str], limit: int | None) -> None:
    x, y = test_set(limit)
    settings = config.AppSettings()
    rows = []
    for name in model_names:
        clf = EmotionClassifier(name)
        for cond, factor, gamma in LIGHTING_CONDITIONS:
            dark = np.stack([darken(img, factor, gamma) for img in x])
            enhanced = np.stack([
                enhance_low_light(img) if img.mean() < settings.low_light_threshold else img for img in dark
            ])
            for mode, data in (("off", dark), ("on", enhanced)):
                rows.append({"model": name, "condition": cond, "mean_brightness": round(float(dark.mean()), 1),
                             "enhancement": mode, **metrics(y, predict_array(clf, data))})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (model, mode), g in df.groupby(["model", "enhancement"]):
        ax.plot(g["condition"], g["accuracy"], marker="o", linestyle="-" if mode == "on" else "--",
                label=f"{model} / enhancement {mode}")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy under synthetic low light")
    ax.legend()
    save_figure(fig, "lighting")
    save("lighting", df, (
        "**Method.** Test images were darkened as `I' = factor * I^gamma` (gamma > 1 crushes shadows). "
        "With enhancement 'on', images whose mean gray level fell below the low-light threshold "
        f"({settings.low_light_threshold}) were processed with the pipeline's adaptive gamma + CLAHE. "
        "**Limitation.** Synthetic darkening does not reproduce real sensor noise, colour casts or "
        "motion blur in low light, so real-world degradation is likely larger."
    ))


def exp_preprocessing(model_names: list[str], limit: int | None) -> None:
    x, y = test_set(limit)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    variants = {
        "raw (training distribution)": x,
        "CLAHE always": np.stack([clahe.apply(i) for i in x]),
        "global histogram equalisation": np.stack([cv2.equalizeHist(i) for i in x]),
    }
    rows = []
    for name in model_names:
        clf = EmotionClassifier(name)
        for variant, data in variants.items():
            rows.append({"model": name, "preprocessing": variant, **metrics(y, predict_array(clf, data))})
    df = pd.DataFrame(rows)
    save("preprocessing", df, (
        "Models are trained on raw FER-2013 pixels. Applying contrast normalisation to *every* face "
        "shifts the input distribution; this table shows whether that helps or hurts, and justifies "
        "applying enhancement only to dark faces in the live pipeline. Face alignment cannot be ablated "
        "on FER-2013 because its images are already roughly aligned crops."
    ))


def _jitter_sequence(img: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate a static face seen by a webcam: small shifts/rotations, brightness flicker, sensor noise."""
    frames = []
    h, w = img.shape
    for _ in range(length):
        angle = rng.uniform(-6, 6)
        tx, ty = rng.uniform(-3, 3, size=2)
        scale = rng.uniform(0.95, 1.05)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        m[:, 2] += (tx, ty)
        f = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        f = f * rng.uniform(0.8, 1.2) + rng.normal(0, 8, size=f.shape)
        frames.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(frames)


def exp_smoothing(model_name: str, limit: int | None, length: int = 40) -> None:
    x, y = test_set(limit or 300)
    rng = np.random.default_rng(config.RANDOM_SEED)
    clf = EmotionClassifier(model_name)
    seqs = np.stack([_jitter_sequence(img, length, rng) for img in x])          # (K, T, 48, 48)
    probs = predict_array(clf, seqs.reshape(-1, config.IMG_SIZE, config.IMG_SIZE)).reshape(len(x), length, -1)

    configs = [("raw (no smoothing)", "none", 1.0, 1, 1)]
    configs += [(f"EMA alpha={a}", "ema", a, 8, 1) for a in (0.6, 0.35, 0.2)]
    configs += [(f"window={w}", "window", 0.35, w, 1) for w in (5, 10)]
    configs += [("EMA alpha=0.35 + hysteresis(3)", "ema", 0.35, 8, 3)]

    rows = []
    for label, method, alpha, window, persist in configs:
        flips, correct, total = [], 0, 0
        for k in range(len(x)):
            ta = TemporalAnalyzer(method=method, alpha=alpha, window=window, min_persist=persist,
                                  confidence_threshold=0.0)
            seq_labels = []
            for t in range(length):
                r = ta.update(0, probs[k, t], t / 30.0)
                seq_labels.append(r.label_index)
            seq_labels = np.array(seq_labels)
            flips.append(int(np.sum(seq_labels[1:] != seq_labels[:-1])))
            correct += int(np.sum(seq_labels == y[k]))
            total += length
        rows.append({
            "method": label,
            "flips_per_sequence": round(float(np.mean(flips)), 3),
            "flips_per_second_at_30fps": round(float(np.mean(flips)) / (length / 30.0), 3),
            "sequences_with_no_flip": round(float(np.mean(np.array(flips) == 0)), 3),
            "frame_accuracy": round(correct / total, 4),
        })
    df = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.bar(df["method"], df["flips_per_sequence"], color="#7C5CFF")
    ax1.set_ylabel("label flips per 40-frame sequence")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    ax2.plot(df["method"], df["frame_accuracy"], color="#F97316", marker="o")
    ax2.set_ylabel("frame accuracy", color="#F97316")
    ax1.set_title(f"Temporal smoothing on synthetic jitter sequences ({model_name})")
    save_figure(fig, "smoothing")
    save("smoothing", df, (
        f"**Method.** {len(x)} test faces were each turned into a {length}-frame sequence with random "
        "rotation (+-6 deg), shift (+-3 px), scale (+-5 %), brightness (+-20 %) and Gaussian noise "
        "(sigma 8). The true expression is constant, so every label change is a flicker. "
        "**Limitation.** Real webcam footage also contains genuine expression changes; heavier "
        "smoothing reduces flicker but delays the response to real changes (latency is not measured here)."
    ))


def _face_canvas(faces: np.ndarray, tile: int = 144, pad: int = 56) -> np.ndarray:
    """Arrange faces on a grid; replicate-padding gives the detector some surrounding context."""
    n = len(faces)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cell = tile + 2 * pad
    canvas = np.full((rows * cell, cols * cell, 3), 90, dtype=np.uint8)
    for i, f in enumerate(faces):
        big = cv2.resize(f, (tile, tile), interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        r, c = divmod(i, cols)
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    return canvas


def exp_multiface(model_name: str, repeats: int = 20) -> None:
    from detection.face_detector import FaceDetector
    from recognition.pipeline import FrameProcessor

    x, _ = test_set(None)
    rng = np.random.default_rng(config.RANDOM_SEED)
    clf = EmotionClassifier(model_name)
    settings = config.AppSettings(detection_input_width=1280)
    detector = FaceDetector(settings.detector_backend, settings.detection_confidence,
                            settings.mediapipe_model_selection, settings.detection_input_width)
    processor = FrameProcessor(settings, clf, detector=detector, use_tracking=False)
    rows = []
    try:
        for n in (1, 2, 4, 6, 9):
            found, det_ms, total_ms, cls_ms = [], [], [], []
            for _ in range(repeats):
                canvas = _face_canvas(x[rng.choice(len(x), n, replace=False)])
                t0 = time.perf_counter()
                detections = detector.detect(canvas)
                det_ms.append((time.perf_counter() - t0) * 1000)
                _, results = processor.process(canvas, t=0.0, draw=False)
                total_ms.append(processor.last_pipeline_ms)
                cls_ms.append(clf.last_latency_ms if results else 0.0)
                found.append(min(len(detections), n))
            rows.append({
                "faces_in_frame": n,
                "detector": detector.backend,
                "detection_recall": round(float(np.mean(found)) / n, 3),
                "detect_ms": round(float(np.median(det_ms)), 2),
                "classify_ms": round(float(np.median(cls_ms)), 2),
                "pipeline_ms": round(float(np.median(total_ms)), 2),
                "pipeline_fps": round(1000.0 / max(float(np.median(total_ms)), 1e-6), 1),
            })
    finally:
        processor.close()
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["faces_in_frame"], df["pipeline_ms"], marker="o", label="full pipeline")
    ax.plot(df["faces_in_frame"], df["detect_ms"], marker="o", label="detection")
    ax.plot(df["faces_in_frame"], df["classify_ms"], marker="o", label="classification (batched)")
    ax.set_xlabel("faces in frame")
    ax.set_ylabel("median ms per frame")
    ax.set_title(f"Latency vs number of faces ({model_name}, {df['detector'].iloc[0]})")
    ax.legend()
    save_figure(fig, "multiface")
    save("multiface", df, (
        "**Method.** Synthetic frames were built by tiling N random FER-2013 test faces (upscaled to "
        "144 px with replicated borders) and run through the same FrameProcessor the app uses. "
        "pipeline_ms covers detection, alignment, batched classification and smoothing, not "
        "camera capture or display. **Limitation.** Tiled, upscaled 48x48 crops are easier/harder "
        "than real scenes in different ways (no occlusion or pose variety, but low resolution), so "
        "recall here is indicative only. Measure real webcam FPS on the Live page."
    ))


def exp_ablation() -> None:
    rows = []
    for path in sorted(config.EXPERIMENTS_DIR.glob("eval_*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": r["name"],
            "architecture": r.get("architecture"),
            "augmentation": r.get("augmentation"),
            "class_weights": r.get("class_weights"),
            "accuracy": r["accuracy"],
            "f1_macro": r["f1_macro"],
            "recall_Disgust": r["per_class"]["Disgust"]["recall"],
            "recall_Fear": r["per_class"]["Fear"]["recall"],
            "recall_Happy": r["per_class"]["Happy"]["recall"],
        })
    if not rows:
        log.error("No eval_*.json files. Run python -m models.evaluate first.")
        return
    df = pd.DataFrame(rows)
    save("ablation", df, (
        "Train the ablation variants with `python -m models.train --model cnn --no-augment --name cnn_noaug` "
        "and `--no-class-weights --name cnn_noweights`, evaluate them, then rerun this experiment. "
        "Compare accuracy (dominated by frequent classes) with macro-F1 and minority-class recall "
        "(Disgust, Fear) to see what class weighting actually changes."
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EmotionLens research experiments.")
    parser.add_argument("experiment", choices=["all", "lighting", "preprocessing", "smoothing", "multiface", "ablation"])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Subsample the test set (0 = all).")
    args = parser.parse_args()
    config.ensure_dirs()

    models = resolve_models(args.models)
    needs_model = args.experiment != "ablation"
    if needs_model and not models:
        log.error("No trained checkpoints found. Run python -m models.train --model cnn first.")
        return 1
    best = best_model_name() if not args.models else models[0]
    limit = args.limit or None

    try:
        if args.experiment in ("all", "lighting"):
            exp_lighting(models, limit)
        if args.experiment in ("all", "preprocessing"):
            exp_preprocessing(models, limit)
        if args.experiment in ("all", "smoothing"):
            exp_smoothing(best, args.limit or 300)
        if args.experiment in ("all", "multiface"):
            exp_multiface(best)
        if args.experiment in ("all", "ablation"):
            exp_ablation()
    except PreparedDataMissingError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
