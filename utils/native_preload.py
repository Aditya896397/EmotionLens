"""Load native libraries in a safe order. Import this module FIRST in every entry point.

pyarrow (pulled in by pandas and Streamlit tables) ships its own, older copy of the Visual C++
runtime (msvcp140.dll). If pyarrow loads first, Windows reuses that old DLL for the whole
process, and MediaPipe's native module then fails with "DLL initialization routine failed".
Loading MediaPipe and ONNX Runtime first puts the newer runtime in memory, which pyarrow then
reuses without problems.

This module must not import pandas, pyarrow or anything that imports them.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")  # quieter MediaPipe start-up

PRELOAD_ERRORS: dict[str, str] = {}

for _module in ("mediapipe", "onnxruntime"):
    try:
        __import__(_module)
    except Exception as exc:  # reported later by the detector / classifier, with fallbacks
        PRELOAD_ERRORS[_module] = f"{type(exc).__name__}: {exc}"
