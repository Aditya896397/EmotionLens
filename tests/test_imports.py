"""Every project module must at least import. This catches syntax errors in code that the other
tests never execute, such as the Streamlit pages."""
from __future__ import annotations

import importlib
import py_compile

import pytest

import config

TENSORFLOW_ONLY = {"models.train", "models.cnn_model", "models.export_onnx"}  # need TensorFlow
SCRIPT_FILES = sorted(p for p in config.ROOT_DIR.rglob("*.py") if ".venv" not in p.parts)
MODULES = sorted(
    ".".join(p.relative_to(config.ROOT_DIR).with_suffix("").parts)
    for p in SCRIPT_FILES
    if p.parent != config.ROOT_DIR and p.name != "__init__.py" and p.parts[-2] != "tests"
)


@pytest.mark.parametrize("path", SCRIPT_FILES, ids=lambda p: str(p.relative_to(config.ROOT_DIR)))
def test_compiles(path) -> None:
    py_compile.compile(str(path), doraise=True)


@pytest.mark.parametrize("module", [m for m in MODULES if m not in TENSORFLOW_ONLY])
def test_imports(module: str) -> None:
    importlib.import_module(module)
