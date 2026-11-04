"""Export trained Keras checkpoints to ONNX and verify the conversion numerically.

Run on the training machine (Colab/Linux), where TensorFlow is available:
    python -m models.export_onnx                 # every .keras checkpoint
    python -m models.export_onnx --models cnn

Steps per model:
  1. Rebuild the same architecture WITHOUT the augmentation block and copy the trained weights
     into it. The augmentation layers hold no weights, so the weight lists match one-to-one.
     This step is required: Keras' random augmentation layers leave stateless random-number ops
     in the traced graph even in inference mode, and tf2onnx rejects that graph ("InvalidGraph").
  2. Run the inference model once (Keras refuses to export a model that has never been called),
     then export it with Keras 3's built-in exporter, `model.export(path, format="onnx")`,
     which uses tf2onnx and patches its NumPy 2 incompatibilities.
  3. Predict the same test images with the original model and with ONNX Runtime, and record the
     maximum probability difference and label agreement in <name>_meta.json.

tf2onnx 1.16.1 pins protobuf~=3.20, which conflicts with current TensorFlow. Install it with
`pip install --no-deps tf2onnx==1.16.1`; it works with the protobuf that TensorFlow brings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

import config
from models.data import PreparedDataMissingError, load_prepared
from utils.logger import get_logger

log = get_logger("export_onnx")

OPSET = 17


def inference_model(name: str, trained):
    """Same architecture as `trained`, without augmentation, carrying the trained weights."""
    from models.cnn_model import build_custom_cnn, build_mobilenetv2

    meta_path = config.CHECKPOINT_DIR / f"{name}_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    architecture = meta.get("architecture") or ("mobilenet" if "mobilenet" in name else "cnn")
    if architecture == "mobilenet":
        img_size = meta.get("mobilenet_img_size") or config.MOBILENET_IMG_SIZE
        model = build_mobilenetv2(img_size=img_size, augment=False, weights=None)[0]
    else:
        model = build_custom_cnn(augment=False)
    model.set_weights(trained.get_weights())
    return model


def export_one(name: str, x_check: np.ndarray | None) -> dict:
    import onnxruntime as ort
    from tensorflow import keras

    keras_path = config.CHECKPOINT_DIR / f"{name}.keras"
    onnx_path = config.CHECKPOINT_DIR / f"{name}.onnx"
    model = keras.models.load_model(keras_path, compile=False)
    serving = inference_model(name, model)

    sample = x_check[:8] if x_check is not None else np.zeros((1, config.IMG_SIZE, config.IMG_SIZE, 1), np.float32)
    serving.predict(sample, verbose=0)  # build/call once before export
    serving.export(str(onnx_path), format="onnx", opset_version=OPSET, verbose=False)
    if not onnx_path.exists():
        raise RuntimeError(f"ONNX export produced no file for {name}")

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
