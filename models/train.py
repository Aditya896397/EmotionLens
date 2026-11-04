"""Train the custom CNN or the MobileNetV2 transfer model on prepared FER-2013.

Examples (run from the project root):
    python -m models.train --model cnn
    python -m models.train --model mobilenet
    python -m models.train --model cnn --epochs 3 --limit 2000      # quick smoke test
    python -m models.train --model cnn --no-augment --name cnn_noaug          # ablation
    python -m models.train --model cnn --no-class-weights --name cnn_noweights  # ablation

Outputs in models/checkpoints/:
    <name>.keras            best weights (highest val_accuracy)
    <name>_history.csv      per-epoch loss/accuracy/learning rate
    <name>_meta.json        settings, parameter count, measured training time
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

import config
from models.cnn_model import build_custom_cnn, build_mobilenetv2, unfreeze_top_layers
from models.data import PreparedDataMissingError, load_class_weights, load_prepared, make_dataset
from utils.device import configure_tensorflow
from utils.logger import get_logger

log = get_logger("train")


class EpochTimer(keras.callbacks.Callback):
    """Records wall-clock seconds per epoch (measured, not estimated)."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_seconds: list[float] = []
        self._start = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_seconds.append(time.perf_counter() - self._start)
        if logs is not None:
            logs["epoch_seconds"] = self.epoch_seconds[-1]
            logs["learning_rate"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))


def make_callbacks(best_path, monitor_best: float | None = None) -> tuple[list, EpochTimer]:
    timer = EpochTimer()
    callbacks = [
        timer,
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_path), monitor="val_accuracy", mode="max",
            save_best_only=True, initial_value_threshold=monitor_best, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=config.LR_PLATEAU_PATIENCE, min_lr=1e-6, verbose=1,
        ),
    ]
    return callbacks, timer


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )


def subsample(x: np.ndarray, y: np.ndarray, limit: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not limit or limit >= len(x):
        return x, y
    idx = np.random.default_rng(seed).choice(len(x), size=limit, replace=False)
    return x[idx], y[idx]


def history_frame(histories: list[keras.callbacks.History], stages: list[str]) -> pd.DataFrame:
    frames = []
    epoch_offset = 0
    for hist, stage in zip(histories, stages):
        df = pd.DataFrame(hist.history)
        df.insert(0, "epoch", np.arange(epoch_offset + 1, epoch_offset + len(df) + 1))
        df.insert(1, "stage", stage)
        epoch_offset += len(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def train(args: argparse.Namespace) -> int:
    config.ensure_dirs()
    device = configure_tensorflow()
    keras.utils.set_random_seed(args.seed)

    try:
        data = load_prepared()
    except PreparedDataMissingError as exc:
        log.error("%s", exc)
        return 1

    x_train, y_train = subsample(data.x_train, data.y_train, args.limit, args.seed)
    x_val, y_val = subsample(data.x_val, data.y_val, args.limit // 5 if args.limit else None, args.seed)
    train_ds = make_dataset(x_train, y_train, args.batch_size, shuffle=True, seed=args.seed)
    val_ds = make_dataset(x_val, y_val, args.batch_size, shuffle=False)

    class_weight = None
    if not args.no_class_weights:
        class_weight = load_class_weights()
        if class_weight is None:
            log.warning("dataset_report.json has no class weights; training without them.")
    log.info("Train %d | Val %d | class weights: %s", len(x_train), len(x_val), class_weight)

    name = args.name or args.model
    best_path = config.CHECKPOINT_DIR / f"{name}.keras"
    histories: list[keras.callbacks.History] = []
    stages: list[str] = []
    epoch_seconds: list[float] = []

    start = time.perf_counter()
    if args.model == "cnn":
        model = build_custom_cnn(augment=not args.no_augment)
        compile_model(model, args.lr or config.CNN_LEARNING_RATE)
        model.summary(print_fn=lambda line, *_, **__: log.info(line))
        callbacks, timer = make_callbacks(best_path)
        histories.append(model.fit(
            train_ds, validation_data=val_ds, epochs=args.epochs or config.CNN_EPOCHS,
            class_weight=class_weight, callbacks=callbacks, verbose=2,
        ))
        stages.append("full")
        epoch_seconds += timer.epoch_seconds
    else:
        model, backbone = build_mobilenetv2(img_size=args.img_size, augment=not args.no_augment)
        # Stage 1: train only the new head on top of frozen ImageNet features.
        compile_model(model, config.MOBILENET_HEAD_LR)
        model.summary(print_fn=lambda line, *_, **__: log.info(line))
        callbacks, timer = make_callbacks(best_path)
        histories.append(model.fit(
            train_ds, validation_data=val_ds, epochs=args.head_epochs,
            class_weight=class_weight, callbacks=callbacks, verbose=2,
        ))
        stages.append("head")
        epoch_seconds += timer.epoch_seconds
        stage1_best = float(np.max(histories[-1].history["val_accuracy"]))

        # Stage 2: unfreeze the top of the backbone and fine-tune with a 10x lower LR.
        n = unfreeze_top_layers(backbone, args.trainable_layers)
        log.info("Fine-tuning: %d backbone layers unfrozen", n)
        compile_model(model, args.lr or config.MOBILENET_FINETUNE_LR)
        callbacks, timer = make_callbacks(best_path, monitor_best=stage1_best)
        epochs = args.epochs or config.MOBILENET_FINETUNE_EPOCHS
        if epochs > 0:
            histories.append(model.fit(
                train_ds, validation_data=val_ds, epochs=epochs,
                class_weight=class_weight, callbacks=callbacks, verbose=2,
            ))
            stages.append("finetune")
            epoch_seconds += timer.epoch_seconds
    train_seconds = time.perf_counter() - start

    if not best_path.exists():  # can only happen if every epoch failed to improve on -inf
        model.save(best_path)
    best_model = keras.models.load_model(best_path, compile=False)
    compile_model(best_model, 1e-4)
    val_loss, val_acc = best_model.evaluate(val_ds, verbose=0)

    hist_df = history_frame(histories, stages)
    hist_df.to_csv(config.CHECKPOINT_DIR / f"{name}_history.csv", index=False)

    meta = {
        "name": name,
        "architecture": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": int(best_model.count_params()),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "epochs_run": int(len(hist_df)),
        "batch_size": args.batch_size,
        "augmentation": not args.no_augment,
        "class_weights": class_weight is not None,
        "mobilenet_img_size": args.img_size if args.model == "mobilenet" else None,
        "training_seconds": round(train_seconds, 1),
        "mean_epoch_seconds": round(float(np.mean(epoch_seconds)), 2) if epoch_seconds else None,
        "best_val_accuracy": round(float(val_acc), 4),
        "best_val_loss": round(float(val_loss), 4),
        "device": device.device,
        "tensorflow_version": tf.__version__,
        "seed": args.seed,
    }
    (config.CHECKPOINT_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Saved %s", best_path)
    log.info("Best validation accuracy %.4f | training time %.1f min", val_acc, train_seconds / 60)
    log.info("Next: python -m models.evaluate --models %s", name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Train an EmotionLens expression model.")
    parser.add_argument("--model", choices=["cnn", "mobilenet"], required=True)
    parser.add_argument("--name", default="", help="Checkpoint name (default: model type).")
    parser.add_argument("--epochs", type=int, default=0,
                        help="CNN epochs, or MobileNet fine-tune epochs (0 = config default).")
    parser.add_argument("--head-epochs", type=int, default=config.MOBILENET_HEAD_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=0.0, help="0 = config default.")
    parser.add_argument("--img-size", type=int, default=config.MOBILENET_IMG_SIZE)
    parser.add_argument("--trainable-layers", type=int, default=config.MOBILENET_TRAINABLE_LAYERS)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Use only N training images (smoke tests).")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    return train(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
