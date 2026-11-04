"""ONNX backend tests: a tiny model with the project's input/output contract is built with the
onnx helper API, so the real loading and inference path runs without a trained model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config
from recognition import emotion_classifier as ec


def write_tiny_onnx(path: Path, favoured_class: int = 3) -> None:
    """(N,48,48,1) -> mean pixel -> 7 logits -> softmax. Biased so `favoured_class` wins."""
    from onnx import TensorProto, helper, numpy_helper, save

    w = numpy_helper.from_array(np.full((1, config.NUM_CLASSES), 0.001, dtype=np.float32), "w")
    bias = np.zeros(config.NUM_CLASSES, dtype=np.float32)
    bias[favoured_class] = 2.0
    b = numpy_helper.from_array(bias, "b")
    nodes = [
        helper.make_node("ReduceMean", ["x"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["m", "ax"], ["m2"]),
        helper.make_node("MatMul", ["m2", "w"], ["l"]),
        helper.make_node("Add", ["l", "b"], ["lb"]),
        helper.make_node("Softmax", ["lb"], ["y"], axis=1),
    ]
    ax = numpy_helper.from_array(np.array([1], dtype=np.int64), "ax")
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, config.IMG_SIZE, config.IMG_SIZE, 1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, config.NUM_CLASSES])],
        [w, b, ax],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    save(model, str(path))


@pytest.fixture
def checkpoint_dir(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(config, "CHECKPOINT_DIR", tmp_path)
    monkeypatch.setattr(config, "EXPERIMENTS_DIR", tmp_path)
    return tmp_path


def test_no_model_raises(checkpoint_dir: Path) -> None:
    with pytest.raises(ec.ModelNotFoundError):
        ec.EmotionClassifier()


def test_onnx_backend_predicts(checkpoint_dir: Path) -> None:
    write_tiny_onnx(checkpoint_dir / "cnn.onnx")
    clf = ec.EmotionClassifier()
    assert clf.name == "cnn" and clf.backend == "onnx"
    faces = [np.random.default_rng(i).integers(0, 255, (48, 48), dtype=np.uint8) for i in range(5)]
    probs = clf.predict_batch(faces)
    assert probs.shape == (5, config.NUM_CLASSES)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs.argmax(axis=1) == 3).all()
    assert clf.predict_array(np.stack(faces)).shape == (5, config.NUM_CLASSES)
    assert clf.last_latency_ms > 0


def test_best_model_uses_eval_accuracy(checkpoint_dir: Path) -> None:
    import json

    write_tiny_onnx(checkpoint_dir / "cnn.onnx")
    write_tiny_onnx(checkpoint_dir / "mobilenet.onnx")
    (checkpoint_dir / "eval_cnn.json").write_text(json.dumps({"accuracy": 0.61}))
    (checkpoint_dir / "eval_mobilenet.json").write_text(json.dumps({"accuracy": 0.64}))
    assert ec.list_models() == ["cnn", "mobilenet"]
    assert ec.best_model_name() == "mobilenet"
