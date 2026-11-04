"""In-memory accumulator for one analysis session (webcam, video or image).

Only labels, confidences, timestamps and anonymous integer track IDs are recorded.
No face pixels and no embeddings are ever kept here.
"""
from __future__ import annotations

import time
import uuid
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import config

if TYPE_CHECKING:
    from recognition.pipeline import FaceResult
    from recognition.temporal_analyzer import TemporalAnalyzer

MAX_TIMELINE_ROWS = 200_000  # memory bound for very long sessions


class SessionStats:
    def __init__(self, source: str, model_name: str = "") -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.source = source
        self.model_name = model_name
        self.started_at = datetime.now()
        self._start_perf = time.perf_counter()
        self.frames_seen = 0
        self.frames_processed = 0
        self.track_ids: set[int] = set()
        self.max_concurrent = 0
        self.counts: Counter = Counter()
        self.uncertain = 0
        self._conf_sum = 0.0
        self._det_conf_sum = 0.0
        self._det_conf_n = 0
        self.predictions = 0
        self.timeline: list[tuple] = []
        self.changes: list[dict] = []
        self.fps_samples: list[float] = []
        self.media_seconds: float | None = None   # set for videos (file time, not wall time)
        self.last_t = 0.0

    # ------------------------------------------------------------------
    def record(self, t: float, results: list["FaceResult"], processed: bool, fps: float | None) -> None:
        self.frames_seen += 1
        self.last_t = max(self.last_t, t)
        if fps:
            self.fps_samples.append(fps)
        if not processed:
            return
        self.frames_processed += 1
        self.max_concurrent = max(self.max_concurrent, len(results))
        for r in results:
            self.track_ids.add(r.track_id)
            self.predictions += 1
            self._conf_sum += r.raw_confidence
            if r.detection_confidence is not None:
                self._det_conf_sum += r.detection_confidence
                self._det_conf_n += 1
            if r.label == config.UNCERTAIN_LABEL:
                self.uncertain += 1
            else:
                self.counts[r.label] += 1
            if len(self.timeline) < MAX_TIMELINE_ROWS:
                self.timeline.append((round(t, 3), r.track_id, r.label, round(r.confidence, 4),
                                      r.raw_label, round(r.raw_confidence, 4), round(r.stability, 3)))
            if r.change is not None:
                self.changes.append({"t": round(t, 2), "track_id": r.track_id,
                                     "from": r.change.from_label, "to": r.change.to_label})

    # ------------------------------------------------------------------
    @property
    def duration_s(self) -> float:
        if self.media_seconds is not None:
            return self.media_seconds
        if self.source == "image":
            return 0.0
        return time.perf_counter() - self._start_perf

    @property
    def avg_confidence(self) -> float | None:
        return self._conf_sum / self.predictions if self.predictions else None

    @property
    def avg_detection_confidence(self) -> float | None:
        return self._det_conf_sum / self._det_conf_n if self._det_conf_n else None

    @property
    def dominant_expression(self) -> str | None:
        return self.counts.most_common(1)[0][0] if self.counts else None

    def expression_counts(self) -> dict[str, int]:
        return {name: int(self.counts.get(name, 0)) for name in config.EMOTION_LABELS}

    def timeline_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.timeline,
            columns=["t", "track_id", "expression", "confidence", "raw_expression", "raw_confidence", "stability"],
        )

    def per_second_df(self) -> pd.DataFrame:
        """One row per (second, track): most frequent label and mean confidence (what the DB stores)."""
        df = self.timeline_df()
        if df.empty:
            return pd.DataFrame(columns=["t_second", "track_id", "expression", "confidence"])
        df["t_second"] = np.floor(df["t"]).astype(int)
        grouped = df.groupby(["t_second", "track_id"])
        out = grouped["expression"].agg(lambda s: s.value_counts().idxmax()).to_frame()
        out["confidence"] = grouped["confidence"].mean().round(4)
        return out.reset_index()

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "source": self.source,
            "model_name": self.model_name,
            "duration_s": round(self.duration_s, 2),
            "frames_processed": self.frames_processed,
            "total_faces": len(self.track_ids),
            "max_concurrent_faces": self.max_concurrent,
            "total_predictions": self.predictions,
            "uncertain_predictions": self.uncertain,
            "avg_confidence": round(self.avg_confidence, 4) if self.avg_confidence is not None else None,
            "avg_detection_confidence": (round(self.avg_detection_confidence, 4)
                                         if self.avg_detection_confidence is not None else None),
            "avg_fps": round(float(np.mean(self.fps_samples)), 2) if self.fps_samples else None,
            "dominant_expression": self.dominant_expression,
            "expression_changes": len(self.changes),
        }

    def track_summaries(self, temporal: "TemporalAnalyzer | None") -> list[dict]:
        return temporal.track_summaries() if temporal is not None else []
