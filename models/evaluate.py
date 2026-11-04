"""Evaluate trained models on the held-out FER-2013 test split and compare them.

Every number written here is computed from the model's predictions on the test set
and from timings measured on this machine. Works with either backend: Keras on the
training machine (Colab) and ONNX Runtime on the laptop, so latency is measured on the
hardware that will actually run the app.

    python -m models.evaluate                       # every checkpoint found
    python -m models.evaluate --models cnn mobilenet
    python -m models.evaluate --backend keras        # in Colab, before exporting to ONNX

Outputs:
    reports/experiments/eval_<name>.json         metrics, per-class report, confusion matrix, latency
    reports/experiments/model_comparison.csv/.md side-by-side table
    reports/figures/confusion_<name>.png
    reports/figures/training_curves_<name>.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

import config
from models.data import PreparedDataMissingError, load_prepared
from recognition.emotion_classifier import EmotionClassifier, list_models
from utils.logger import get_logger

log = get_logger("evaluate")


def available_checkpoints() -> list[str]:
    return list_models()


def measure_latency(clf: EmotionClassifier, sample: np.ndarray, batch_sizes=(1, 4, 8), runs: int = 100,
                    warmup: int = 10) -> dict:
    """Median/p95 milliseconds per forward pass for several batch sizes (face counts)."""
    results = {}
    for bs in batch_sizes:
        faces = [sample[0, ..., 0].astype(np.uint8)] * bs
        for _ in range(warmup):
            clf.predict_batch(faces)
        times = []
        for _ in range(runs):
            start = time.perf_counter()
            clf.predict_batch(faces)
            times.append((time.perf_counter() - start) * 1000)
        results[f"batch_{bs}"] = {
            "median_ms": round(float(np.median(times)), 3),
            "p95_ms": round(float(np.percentile(times, 95)), 3),
            "per_face_ms": round(float(np.median(times)) / bs, 3),
        }
    return results


def plot_confusion(cm: np.ndarray, name: str, out: Path) -> None:
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels = config.EMOTION_LABELS
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7, color=color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (row-normalised) - {name}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_training_curves(name: str, out: Path) -> bool:
    hist_path = config.CHECKPOINT_DIR / f"{name}_history.csv"
    if not hist_path.exists():
        return False
    df = pd.read_csv(hist_path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for metric, ax in zip(("loss", "accuracy"), axes):
        ax.plot(df["epoch"], df[metric], label=f"train {metric}")
        ax.plot(df["epoch"], df[f"val_{metric}"], label=f"val {metric}")
        if "stage" in df and df["stage"].nunique() > 1:
            boundary = df.loc[df["stage"] != df["stage"].iloc[0], "epoch"].min()
            ax.axvline(boundary - 0.5, color="gray", linestyle="--", label="fine-tune starts")
        ax.set_xlabel("epoch")
        ax.set_title(metric)
        ax.legend()
    fig.suptitle(f"Training curves - {name}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def evaluate_model(name: str, x_test: np.ndarray, y_test: np.ndarray, latency_runs: int,
                   backend: str = "auto") -> dict:
    clf = EmotionClassifier(name, backend=backend)

    start = time.perf_counter()
    probs = clf.predict_array(x_test)
    total_seconds = time.perf_counter() - start
    y_pred = probs.argmax(axis=1)

    labels = list(range(config.NUM_CLASSES))
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="macro", zero_division=0)
    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(
        y_test, y_pred, labels=labels, target_names=list(config.EMOTION_LABELS),
        output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    meta_path = config.CHECKPOINT_DIR / f"{name}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    result = {
        "name": name,
        "architecture": meta.get("architecture", "unknown"),
        "test_samples": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_macro": round(float(precision_m), 4),
        "recall_macro": round(float(recall_m), 4),
        "f1_macro": round(float(f1_m), 4),
        "precision_weighted": round(float(precision_w), 4),
        "recall_weighted": round(float(recall_w), 4),
        "f1_weighted": round(float(f1_w), 4),
        "per_class": {k: {m: round(float(v), 4) for m, v in report[k].items()} for k in config.EMOTION_LABELS},
        "confusion_matrix": cm.tolist(),
        "labels": list(config.EMOTION_LABELS),
        "parameters": int(meta.get("parameters") or (clf.model.count_params() if clf.backend == "keras" else 0)),
        "model_file_mb": round(clf.path.stat().st_size / 1024**2, 2),
        "inference_backend": clf.backend,
        "runtime": clf.runtime_version,
        "training_seconds": meta.get("training_seconds"),
        "epochs_run": meta.get("epochs_run"),
        "augmentation": meta.get("augmentation"),
        "class_weights": meta.get("class_weights"),
        "test_set_throughput_img_per_s": round(len(x_test) / total_seconds, 1),
        "latency": measure_latency(clf, x_test, runs=latency_runs),
        "device": clf.device,
    }
    plot_confusion(cm, name, config.FIGURES_DIR / f"confusion_{name}.png")
    plot_training_curves(name, config.FIGURES_DIR / f"training_curves_{name}.png")
    (config.EXPERIMENTS_DIR / f"eval_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def comparison_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "model": r["name"],
            "accuracy": r["accuracy"],
            "precision_macro": r["precision_macro"],
            "recall_macro": r["recall_macro"],
            "f1_macro": r["f1_macro"],
            "f1_weighted": r["f1_weighted"],
            "params_M": round(r["parameters"] / 1e6, 2),
            "train_min": round(r["training_seconds"] / 60, 1) if r.get("training_seconds") else None,
            "latency_1face_ms": r["latency"]["batch_1"]["median_ms"],
            "latency_8faces_ms": r["latency"]["batch_8"]["median_ms"],
            "backend": r.get("inference_backend", "keras"),
            "device": r["device"],
        })
    return pd.DataFrame(rows)


def write_comparison(results: list[dict]) -> pd.DataFrame:
    """Merge with previously evaluated models so the table always covers every eval_*.json."""
    all_results = {r["name"]: r for r in load_all_results()}
    all_results.update({r["name"]: r for r in results})
    df = comparison_table(list(all_results.values()))
    df.to_csv(config.EXPERIMENTS_DIR / "model_comparison.csv", index=False)
    (config.EXPERIMENTS_DIR / "model_comparison.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    return df


def load_all_results() -> list[dict]:
    out = []
    for path in sorted(config.EXPERIMENTS_DIR.glob("eval_*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            log.warning("Skipping unreadable %s", path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EmotionLens models on the test split.")
    parser.add_argument("--models", nargs="*", default=None, help="Checkpoint names (default: all).")
    parser.add_argument("--latency-runs", type=int, default=100)
    parser.add_argument("--backend", choices=["auto", "onnx", "keras"], default="auto",
                        help="auto = ONNX if <name>.onnx exists, else Keras.")
    args = parser.parse_args()

    config.ensure_dirs()
    names = args.models or available_checkpoints()
    if not names:
        log.error("No checkpoints in %s. Train first: python -m models.train --model cnn", config.CHECKPOINT_DIR)
        return 1
    try:
        data = load_prepared()
    except PreparedDataMissingError as exc:
        log.error("%s", exc)
        return 1

    results = []
    for name in names:
        log.info("Evaluating %s on %d test images ...", name, len(data.y_test))
        try:
            r = evaluate_model(name, data.x_test, data.y_test, args.latency_runs, args.backend)
        except FileNotFoundError as exc:
            log.error("%s", exc)
            continue
        results.append(r)
        log.info("%s: accuracy=%.4f macro-F1=%.4f latency(1 face)=%.2f ms",
                 name, r["accuracy"], r["f1_macro"], r["latency"]["batch_1"]["median_ms"])

    if not results:
        return 1
    df = write_comparison(results)
    print("\n" + df.to_string(index=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
