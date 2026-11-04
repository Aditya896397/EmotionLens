"""Face detection: MediaPipe BlazeFace with an OpenCV Haar-cascade fallback.

`Detection.score` is the *face detection* confidence (is this a face?). It is a
different quantity from the expression classifier's softmax confidence.
Haar cascades produce no calibrated score, so their detections carry score=None.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("detector")


@dataclass
class Detection:
    bbox: np.ndarray                  # float32 [x1, y1, x2, y2] in frame pixels
    score: float | None               # detection confidence (None for Haar)
    keypoints: np.ndarray | None = None   # (6, 2) pixels: R eye, L eye, nose, mouth, R ear, L ear

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class FaceDetector:
    """Detect faces in BGR frames.

    backend: "mediapipe", "haar" or "auto" (MediaPipe, falling back to Haar if
    MediaPipe cannot be initialised on this machine).
    """

    def __init__(
        self,
        backend: str = "auto",
        min_confidence: float = 0.5,
        model_selection: int = 0,
        input_width: int = 640,
        min_face_size: int = 24,
    ) -> None:
        self.min_confidence = min_confidence
        self.input_width = input_width
        self.min_face_size = min_face_size
        self._mp_detector = None
        self._haar = None
        self.backend = ""

        if backend in ("auto", "mediapipe"):
            try:
                import mediapipe as mp

                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=model_selection, min_detection_confidence=min_confidence
                )
                self.backend = "mediapipe"
            except Exception as exc:
                if backend == "mediapipe":
                    raise RuntimeError(f"MediaPipe face detection unavailable: {exc}") from exc
                log.warning("MediaPipe unavailable (%s); falling back to Haar cascade.", exc)

        if not self.backend:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade_path)
            if self._haar.empty():
                raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
            self.backend = "haar"
        log.info("Face detector backend: %s", self.backend)

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        if frame_bgr is None or frame_bgr.size == 0:
            return []
        h, w = frame_bgr.shape[:2]
        scale = 1.0
        small = frame_bgr
        if w > self.input_width:
            scale = self.input_width / w
            small = cv2.resize(frame_bgr, (self.input_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)

        try:
            raw = self._detect_mediapipe(small) if self.backend == "mediapipe" else self._detect_haar(small)
        except Exception as exc:  # never let one bad frame crash a live session
            log.error("Face detection failed on a frame: %s", exc)
            return []

        detections = []
        for det in raw:
            bbox = det.bbox / scale
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, h - 1)
            det.bbox = bbox.astype(np.float32)
            if det.keypoints is not None:
                det.keypoints = (det.keypoints / scale).astype(np.float32)
            if det.width >= self.min_face_size and det.height >= self.min_face_size:
                detections.append(det)
        return detections

    def _detect_mediapipe(self, frame_bgr: np.ndarray) -> list[Detection]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._mp_detector.process(rgb)
        out = []
        for d in result.detections or []:
            box = d.location_data.relative_bounding_box
            x1, y1 = box.xmin * w, box.ymin * h
            bbox = np.array([x1, y1, x1 + box.width * w, y1 + box.height * h], dtype=np.float32)
            kps = np.array([[kp.x * w, kp.y * h] for kp in d.location_data.relative_keypoints], dtype=np.float32)
            out.append(Detection(bbox=bbox, score=float(d.score[0]), keypoints=kps if len(kps) == 6 else None))
        return out

    def _detect_haar(self, frame_bgr: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [
            Detection(bbox=np.array([x, y, x + fw, y + fh], dtype=np.float32), score=None)
            for (x, y, fw, fh) in faces
        ]

    def close(self) -> None:
        if self._mp_detector is not None:
            self._mp_detector.close()
            self._mp_detector = None
