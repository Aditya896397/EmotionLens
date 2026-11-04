"""FrameProcessor: the end-to-end per-frame pipeline shared by webcam, video and image modes.

    frame -> [low-light check] -> FaceDetector -> ByteTracker -> preprocess_face (align, crop)
          -> EmotionClassifier (one batch for all faces) -> TemporalAnalyzer -> SessionStats
          -> annotated frame + list[FaceResult]

Frame skipping: detection + classification run every (frame_skip + 1) frames; frames in
between reuse the previous results for the overlay, which keeps the display smooth
while cutting compute roughly by that factor.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

import config
from analytics.session import SessionStats
from detection.face_detector import Detection, FaceDetector
from detection.face_tracker import ByteTracker
from detection.preprocessing import preprocess_face
from recognition.emotion_classifier import EmotionClassifier
from recognition.temporal_analyzer import ChangeEvent, TemporalAnalyzer
from utils.image_utils import draw_hud, draw_results, enhance_low_light, mean_brightness
from utils.logger import get_logger
from utils.video_utils import FPSMeter

log = get_logger("pipeline")

HAAR_TRACKING_SCORE = 0.9  # Haar gives no score; treat its boxes as confident for the tracker only


@dataclass
class FaceResult:
    track_id: int
    bbox: tuple[int, int, int, int]
    detection_confidence: float | None     # "is this a face?" (detector)
    raw_label: str                         # single-frame argmax
    raw_confidence: float                  # single-frame softmax probability
    label: str                             # temporally smoothed stable label (or "Uncertain")
    confidence: float                      # smoothed probability of `label`
    probabilities: np.ndarray              # smoothed (or raw, for images) 7-class vector
    stability: float
    change: ChangeEvent | None = None

    def top_k(self, k: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(self.probabilities)[::-1][:k]
        return [(config.EMOTION_LABELS[i], float(self.probabilities[i])) for i in order]


def build_detector(settings: config.AppSettings) -> FaceDetector:
    return FaceDetector(
        backend=settings.detector_backend,
        min_confidence=settings.detection_confidence,
        model_selection=settings.mediapipe_model_selection,
        input_width=settings.detection_input_width,
        min_face_size=settings.min_face_size,
    )


class FrameProcessor:
    def __init__(
        self,
        settings: config.AppSettings,
        classifier: EmotionClassifier,
        detector: FaceDetector | None = None,
        use_tracking: bool = True,
        session: SessionStats | None = None,
    ) -> None:
        self.settings = settings
        self.classifier = classifier
        self.detector = detector or build_detector(settings)
        self.use_tracking = use_tracking
        self.tracker = ByteTracker(
            high_thresh=settings.track_high_thresh,
            low_thresh=settings.track_low_thresh,
            new_track_thresh=settings.track_new_thresh,
            iou_thresh=settings.track_iou_thresh,
            max_lost=settings.track_max_lost,
            min_hits=settings.track_min_hits,
        ) if use_tracking else None
        self.temporal = TemporalAnalyzer(
            method=settings.smoothing_method if use_tracking else "none",
            alpha=settings.ema_alpha,
            window=settings.smoothing_window,
            min_persist=settings.change_min_persist if use_tracking else 1,
            confidence_threshold=settings.expression_confidence_threshold,
            stability_window=settings.stability_window,
        )
        self.session = session
        self.fps_meter = FPSMeter()
        self.frame_index = 0
        self.last_results: list[FaceResult] = []
        self.last_brightness = 0.0
        self.last_enhanced = False
        self.last_pipeline_ms = 0.0
        self.detection_failures = 0
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    def _detect(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """Detect faces, enhancing dark frames; retries on an enhanced frame if nothing is found."""
        s = self.settings
        self.last_brightness = mean_brightness(frame)
        work = frame
        self.last_enhanced = False
        if s.low_light_enhance and self.last_brightness < s.low_light_threshold:
            work = enhance_low_light(frame)
            self.last_enhanced = True
        detections = self.detector.detect(work)
        if not detections and s.low_light_enhance and not self.last_enhanced \
                and self.last_brightness < s.low_light_threshold * 1.5:
            enhanced = enhance_low_light(frame)
            detections = self.detector.detect(enhanced)
            if detections:
                work, self.last_enhanced = enhanced, True
        if not detections:
            self.detection_failures += 1
        return work, detections

    def _infer(self, frame: np.ndarray, t: float) -> list[FaceResult]:
        s = self.settings
        work, detections = self._detect(frame)

        if self.use_tracking:
            scores = np.array([d.score if d.score is not None else HAAR_TRACKING_SCORE for d in detections])
            boxes = np.array([d.bbox for d in detections]).reshape(-1, 4)
            tracked = [(o.track_id, o.det_index) for o in self.tracker.update(boxes, scores)]
        else:
            # Single image: number faces left to right.
            order = sorted(range(len(detections)), key=lambda i: detections[i].bbox[0])
            tracked = [(rank + 1, i) for rank, i in enumerate(order)]

        if not tracked:
            if self.use_tracking:
                self.temporal.prune(t)
            return []

        gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        crops = [
            preprocess_face(gray, detections[di], align=s.align_faces, margin=s.face_margin,
                            low_light_enhance=s.low_light_enhance, low_light_threshold=s.low_light_threshold)
            for _, di in tracked
        ]
        probs = self.classifier.predict_batch(crops)

        h, w = frame.shape[:2]
        results = []
        for (track_id, di), p in zip(tracked, probs):
            det = detections[di]
            tr = self.temporal.update(track_id, p, t)
            raw_idx = int(np.argmax(p))
            x1, y1, x2, y2 = det.bbox
            results.append(FaceResult(
                track_id=track_id,
                bbox=(int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2))),
                detection_confidence=det.score,
                raw_label=config.EMOTION_LABELS[raw_idx],
                raw_confidence=float(p[raw_idx]),
                label=tr.label,
                confidence=tr.confidence,
                probabilities=tr.probabilities,
                stability=tr.stability,
                change=tr.change,
            ))
        if self.use_tracking:
            self.temporal.prune(t)
        return results

    # ------------------------------------------------------------------
    def process(self, frame: np.ndarray, t: float | None = None, draw: bool = True
                ) -> tuple[np.ndarray, list[FaceResult]]:
        """Process one BGR frame. `t` = timestamp in seconds (wall-clock if omitted)."""
        if t is None:
            t = time.perf_counter() - self._t0
        run_inference = self.frame_index % (self.settings.frame_skip + 1) == 0
        self.frame_index += 1

        if run_inference:
            start = time.perf_counter()
            try:
                self.last_results = self._infer(frame, t)
            except Exception as exc:  # keep live sessions alive on unexpected per-frame errors
                log.exception("Frame processing failed: %s", exc)
                self.last_results = []
            self.last_pipeline_ms = (time.perf_counter() - start) * 1000

        fps = self.fps_meter.tick()
        if self.session is not None:
            self.session.record(t, self.last_results, processed=run_inference, fps=fps)

        if not draw:
            return frame, self.last_results
        annotated = draw_results(frame, self.last_results, show_ids=self.use_tracking)
        extra = "low-light boost" if self.last_enhanced else ""
        draw_hud(annotated, len(self.last_results), fps if self.use_tracking else None, extra)
        return annotated, self.last_results

    @property
    def fps(self) -> float | None:
        return self.fps_meter.fps

    def close(self) -> None:
        self.detector.close()
