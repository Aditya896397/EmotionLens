"""Turn a detected face into the 48x48 grayscale input the classifier expects.

Steps: (1) expand the box by a margin, (2) rotate so the eyes are horizontal
(when eye keypoints exist), (3) crop, (4) grayscale, (5) brighten dark faces,
(6) resize to 48x48. Rotation and crop happen in one cv2.warpAffine call that only
renders the output box, which is much cheaper than rotating the whole frame.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

import config
from detection.face_detector import Detection
from utils.image_utils import enhance_low_light

MAX_ALIGN_ANGLE = 30.0  # larger angles are usually keypoint errors, not real head roll


def eye_angle(keypoints: np.ndarray) -> float:
    """Roll angle in degrees of the line through both eyes (keypoints 0 and 1).

    The eyes are ordered by x so mirrored frames give the same result.
    """
    a, b = keypoints[0], keypoints[1]
    left_in_image, right_in_image = (a, b) if a[0] <= b[0] else (b, a)
    dx = right_in_image[0] - left_in_image[0]
    dy = right_in_image[1] - left_in_image[1]
    return math.degrees(math.atan2(dy, dx))


def square_box(bbox: np.ndarray, margin: float) -> tuple[float, float, float]:
    """Centre and side length of a square box around the face, expanded by margin."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    return cx, cy, side


def preprocess_face(
    frame_bgr: np.ndarray,
    detection: Detection,
    align: bool = True,
    margin: float = 0.15,
    low_light_enhance: bool = True,
    low_light_threshold: float = 70.0,
    size: int = config.IMG_SIZE,
) -> np.ndarray:
    """Return a uint8 (size, size) grayscale face crop."""
    gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cx, cy, side = square_box(detection.bbox, margin)

    angle = 0.0
    if align and detection.keypoints is not None:
        angle = eye_angle(detection.keypoints)
        if abs(angle) > MAX_ALIGN_ANGLE:
            angle = 0.0

    # Work on a region around the face (large enough for any rotation) instead of the full frame.
    h, w = gray.shape[:2]
    half = side * 0.75  # > side / sqrt(2), so a rotated square still fits
    rx1, ry1 = max(0, int(cx - half)), max(0, int(cy - half))
    rx2, ry2 = min(w, int(cx + half) + 1), min(h, int(cy + half) + 1)
    roi = gray[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return np.zeros((size, size), dtype=np.uint8)
    cx, cy = cx - rx1, cy - ry1

    scale = size / side
    if scale < 0.5:
        # Pre-blur large faces so the single-step downscale does not alias.
        roi = cv2.GaussianBlur(roi, (0, 0), sigmaX=0.5 * (1.0 / scale - 1.0))

    # Rotate about the face centre and scale, then translate the centre to the middle
    # of the size x size output, all in one affine warp.
    m = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    m[0, 2] += size / 2 - cx
    m[1, 2] += size / 2 - cy
    face = cv2.warpAffine(roi, m, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if low_light_enhance and float(face.mean()) < low_light_threshold:
        face = enhance_low_light(face)
    return face
