"""Camera access, video reading/writing and FPS measurement."""
from __future__ import annotations

import platform
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger

log = get_logger("video")


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


class Camera:
    """Webcam wrapper used as a context manager so the device is always released.

        with Camera(0, 640, 480) as cam:
            frame = cam.read()
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self.index = index
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._failures = 0

    def open(self) -> "Camera":
        # DirectShow opens far faster than MSMF on most Windows laptops.
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open camera {self.index}. Close other apps using it and check "
                "Windows Settings > Privacy & security > Camera."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low: always read the newest frame
        self._cap = cap
        log.info("Camera %d opened", self.index)
        return self

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise CameraError("Camera is not open")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= 30:
                raise CameraError("Camera stopped delivering frames.")
            time.sleep(0.01)
            return self.read()
        self._failures = 0
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Camera %d released", self.index)

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.release()


class FPSMeter:
    """Exponential moving average of frames per second."""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self._last: float | None = None
        self.fps: float | None = None

    def tick(self) -> float | None:
        now = time.perf_counter()
        if self._last is not None:
            dt = now - self._last
            if dt > 0:
                inst = 1.0 / dt
                self.fps = inst if self.fps is None else self.alpha * inst + (1 - self.alpha) * self.fps
        self._last = now
        return self.fps


@dataclass
class VideoProperties:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def video_properties(path: Path) -> VideoProperties:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps > 240:  # 0, NaN or nonsense metadata
            fps = 25.0
        return VideoProperties(
            fps=float(fps),
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def iterate_video(path: Path, max_seconds: float | None = None) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_seconds, frame) using the file's own frame rate."""
    props = video_properties(path)
    max_frames = int(max_seconds * props.fps) if max_seconds else None
    cap = cv2.VideoCapture(str(path))
    try:
        index = 0
        while True:
            if max_frames is not None and index >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            yield index, index / props.fps, frame
            index += 1
    finally:
        cap.release()


def create_video_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    """Open an MP4 writer. H.264 ('avc1') plays in browsers; 'mp4v' is the universal fallback."""
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            log.info("Video writer opened with codec %s", codec)
            return writer, codec
        writer.release()
    raise RuntimeError("No usable MP4 codec found in this OpenCV build.")
