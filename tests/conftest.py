"""Load native libraries before pandas/pyarrow (see utils/native_preload.py)."""
import utils.native_preload  # noqa: F401

try:
    import onnx  # noqa: F401  (also sensitive to pyarrow's bundled MSVC runtime)
except ImportError:
    pass
