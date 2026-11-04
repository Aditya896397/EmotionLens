"""Image helpers: decoding, brightness, low-light enhancement and result overlays."""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

import config

if TYPE_CHECKING:
    from recognition.pipeline import FaceResult

# BGR colours per label, matching the Plotly palette in analytics/charts.py.
LABEL_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "Angry": (68, 68, 239),
    "Disgust": (94, 197, 34),
    "Fear": (247, 85, 168),
    "Happy": (21, 204, 250),
    "Sad": (246, 130, 59),
    "Surprise": (22, 115, 249),
    "Neutral": (184, 163, 148),
    config.UNCERTAIN_LABEL: (128, 128, 128),
}

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_CLAHE_SMALL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


def decode_image(data: bytes) -> np.ndarray | None:
    """Decode uploaded bytes into a BGR image (None if the bytes are not an image)."""
    array = np.frombuffer(data, dtype=np.uint8)
    if array.size == 0:
        return None
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def resize_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)


def mean_brightness(image: np.ndarray) -> float:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _gamma_lut(gamma: float) -> np.ndarray:
    return np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)


def enhance_low_light(image: np.ndarray, target_brightness: float = 110.0) -> np.ndarray:
    """Brighten a dark frame: adaptive gamma followed by CLAHE on luminance.

    Works on BGR frames (CLAHE applied to the L channel of LAB, so colours are kept)
    and on single-channel grayscale face crops.
    """
    brightness = max(mean_brightness(image), 1.0)
    # Choose gamma so the mean moves toward the target: (b/255)^g = t/255.
    gamma = float(np.clip(np.log(target_brightness / 255.0) / np.log(brightness / 255.0), 0.35, 1.0))
    lut = _gamma_lut(gamma)

    if image.ndim == 2:
        clahe = _CLAHE_SMALL if min(image.shape) <= 64 else _CLAHE
        return clahe.apply(cv2.LUT(image, lut))

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _CLAHE.apply(cv2.LUT(l_channel, lut))
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def darken(image: np.ndarray, factor: float, gamma: float = 1.0) -> np.ndarray:
    """Simulate low light: scale intensities and optionally apply a gamma > 1."""
    out = image.astype(np.float32) / 255.0
    out = np.power(out, gamma) * factor
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _put_label(img: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int],
               scale: float = 0.5) -> int:
    """Draw text on a filled background; returns the height used."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, 1)
    x, y = origin
    y = max(y, th + baseline + 2)
    cv2.rectangle(img, (x, y - th - baseline - 4), (x + tw + 6, y), color, thickness=-1)
    luminance = 0.114 * color[0] + 0.587 * color[1] + 0.299 * color[2]
    text_color = (20, 20, 20) if luminance > 140 else (245, 245, 245)
    cv2.putText(img, text, (x + 3, y - baseline - 2), font, scale, text_color, 1, cv2.LINE_AA)
    return th + baseline + 4


def draw_results(frame: np.ndarray, results: list["FaceResult"], show_ids: bool = True) -> np.ndarray:
    """Return a copy of the frame with boxes, IDs, labels and both confidences drawn."""
    out = frame.copy()
    scale = max(0.45, min(out.shape[1] / 1280.0, 1.0) * 0.7)
    thickness = max(2, out.shape[1] // 400)
    for r in results:
        x1, y1, x2, y2 = r.bbox
        color = LABEL_COLORS_BGR.get(r.label, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        prefix = f"ID {r.track_id} | " if show_ids else ""
        line1 = f"{prefix}{r.label} {r.confidence * 100:.0f}%"
        det = "n/a" if r.detection_confidence is None else f"{r.detection_confidence:.2f}"
        line2 = f"face det {det}"
        used = _put_label(out, line1, (x1, y1 - 2), color, scale)
        _put_label(out, line2, (x1, y1 - 2 - used), (60, 60, 60), scale * 0.8)
    return out


def draw_hud(frame: np.ndarray, faces: int, fps: float | None, extra: str = "") -> np.ndarray:
    """Small heads-up display in the top-left corner (draws in place)."""
    text = f"Faces: {faces}"
    if fps is not None:
        text += f"  |  FPS: {fps:.1f}"
    if extra:
        text += f"  |  {extra}"
    _put_label(frame, text, (8, 26), (30, 30, 30), 0.6)
    return frame


def encode_png(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode image as PNG")
    return buffer.tobytes()
