"""Preprocessing, image utility and report tests."""
from __future__ import annotations

import numpy as np

from analytics.report_generator import build_pdf_report
from detection.face_detector import Detection
from detection.preprocessing import eye_angle, preprocess_face
from utils.image_utils import darken, enhance_low_light, mean_brightness


def test_preprocess_returns_48x48_uint8() -> None:
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    det = Detection(bbox=np.array([200, 150, 320, 290], dtype=np.float32), score=0.9)
    face = preprocess_face(frame, det)
    assert face.shape == (48, 48) and face.dtype == np.uint8


def test_preprocess_handles_face_at_frame_edge() -> None:
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    det = Detection(bbox=np.array([0, 0, 60, 70], dtype=np.float32), score=0.9)
    assert preprocess_face(frame, det).shape == (48, 48)


def test_eye_angle_sign_and_mirroring() -> None:
    level = np.array([[100, 100], [140, 100]] + [[0, 0]] * 4, dtype=np.float32)
    tilted = np.array([[100, 100], [140, 120]] + [[0, 0]] * 4, dtype=np.float32)
    mirrored = tilted[[1, 0, 2, 3, 4, 5]]
    assert abs(eye_angle(level)) < 1e-6
    assert eye_angle(tilted) > 0
    assert abs(eye_angle(tilted) - eye_angle(mirrored)) < 1e-6


def test_alignment_levels_rotated_pattern() -> None:
    """A bright horizontal bar rotated by the eye angle should come out horizontal again."""
    import cv2

    frame = np.zeros((400, 400), dtype=np.uint8)
    cv2.rectangle(frame, (150, 195), (250, 205), 255, -1)
    m = cv2.getRotationMatrix2D((200, 200), -20, 1.0)  # tilt the bar by +20 degrees (clockwise on screen)
    rotated = cv2.warpAffine(frame, m, (400, 400))
    left, right = m @ np.array([150, 200, 1]), m @ np.array([250, 200, 1])
    kps = np.array([left, right] + [[0, 0]] * 4, dtype=np.float32)
    det = Detection(bbox=np.array([140, 140, 260, 260], dtype=np.float32), score=1.0, keypoints=kps)
    face = preprocess_face(rotated, det, align=True, margin=0.0, low_light_enhance=False)
    rows = face.mean(axis=1)
    cols_profile = face.mean(axis=0)
    # After alignment the bar is horizontal: energy concentrated in few rows, spread across columns.
    assert (rows > 40).sum() <= 8
    assert (cols_profile > 10).sum() >= 30


def test_low_light_enhancement_brightens() -> None:
    img = np.random.default_rng(1).integers(0, 255, (100, 100, 3), dtype=np.uint8)
    dark = darken(img, 0.2, 1.8)
    assert mean_brightness(enhance_low_light(dark)) > mean_brightness(dark) + 20
    gray = dark[..., 0]
    assert enhance_low_light(gray).shape == gray.shape


def test_pdf_report_builds() -> None:
    pdf = build_pdf_report("Test report", {"session_id": "abc", "faces": 2, "note": "naive – unicode ok"},
                           {"Happy": 3, "Sad": 1, "Angry": 0, "Disgust": 0, "Fear": 0, "Surprise": 0, "Neutral": 2})
    assert pdf[:4] == b"%PDF"
