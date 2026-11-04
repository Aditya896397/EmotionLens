"""Export trained Keras checkpoints to ONNX and verify the conversion numerically.

Run on the training machine (Colab/Linux), where TensorFlow is available:
    python -m models.export_onnx                 # every .keras checkpoint
    python -m models.export_onnx --models cnn

Steps per model: .keras -> TensorFlow SavedModel (Keras 3 `model.export`) -> ONNX (tf2onnx),
then both models predict the same test images and the maximum probability difference and
the label agreement are recorded in <name>_meta.json. The laptop app loads the .onnx file
with ONNX Runtime, so it does not need TensorFlow at all.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import config
from models.data import PreparedDataMissingError, load_prepared
from utils.logger import get_logger

log = get_logger("export_onnx")

OPSET = 17


def export_one(name: str, x_check: np.ndarray | None) -> dict:
    import onnxruntime as ort
    from tensorflow import keras

    keras_path = config.CHECKPOINT_DIR / f"{name}.keras"
    onnx_path = config.CHECKPOINT_DIR / f"{name}.onnx"
    model = keras.models.load_model(keras_path, compile=False)

    tmp = Path(tempfile.mkdtemp(prefix=f"{name}_savedmodel_"))
    try:
        model.export(str(tmp), format="tf_saved_model")
        cmd = [sys.executable, "-m", "tf2onnx.convert", "--saved-model", str(tmp),
               "--output", str(onnx_path), "--opset", str(OPSET)]
        log.info("Running: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"tf2onnx failed for {name}:\n{proc.stderr[-3000:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    result = {"onnx_file": onnx_path.name, "onnx_opset": OPSET,
              "onnx_size_mb": round(onnx_path.stat().st_size / 1024**2, 2)}
    if x_check is not None:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        p_onnx = np.concatenate([session.run(None, {input_name: x_check[i:i + 256]})[0]
                                 for i in range(0, len(x_check), 256)])
        p_keras = model.predict(x_check, batch_size=256, verbose=0)
        result["verify_images"] = int(len(x_check))
        result["max_abs_prob_diff"] = float(np.max(np.abs(p_onnx - p_keras)))
        result["label_agreement"] = float(np.mean(p_onnx.argmax(1) == p_keras.argmax(1)))
        if result["label_agreement"] < 0.999:
            log.warning("ONNX and Keras disagree on %.2f%% of labels for %s",
                        100 * (1 - result["label_agreement"]), name)

    meta_path = config.CHECKPOINT_DIR / f"{name}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"name": name}
    meta.update(result)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("Exported %s: %s", name, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Keras checkpoints to ONNX.")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--verify", type=int, default=1000, help="Test images used to compare outputs (0 = skip).")
    args = parser.parse_args()

    names = args.models or sorted(p.stem for p in config.CHECKPOINT_DIR.glob("*.keras"))
    if not names:
        log.error("No .keras checkpoints in %s", config.CHECKPOINT_DIR)
        return 1

    x_check = None
    if args.verify:
        try:
            x_check = load_prepared().x_test[:args.verify]
        except PreparedDataMissingError:
            log.warning("Prepared data missing; exporting without numerical verification.")

    failed = 0
    for name in names:
        try:
            export_one(name, x_check)
        except Exception as exc:
            failed += 1
            log.error("%s", exc)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
