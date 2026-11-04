"""Temporal analysis of per-track expression predictions.

Single-frame predictions flicker because of motion blur, lighting and pose changes.
For every tracked face this module keeps:
  * a smoothed probability vector (EMA or moving-window average);
  * a *stable* label that changes only after a new label has been the smoothed
    top class for `min_persist` consecutive updates (hysteresis), which turns noisy
    frame-level output into discrete "expression change" events;
  * a stability score = share of recent raw predictions that agree with the stable label;
  * time spent in each stable label.

These are statistics about classifier outputs over time, not evidence of a person's
internal emotional state.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

import config

MAX_DT = 1.0  # seconds; gaps longer than this (e.g. occlusion) are not counted as time in a label


@dataclass
class ChangeEvent:
    track_id: int
    t: float
    from_label: str
    to_label: str


@dataclass
class TemporalResult:
    probabilities: np.ndarray        # smoothed (7,)
    label: str                       # stable label, or UNCERTAIN_LABEL
    label_index: int | None
    confidence: float                # smoothed probability of the stable label
    stability: float                 # 0..1
    change: ChangeEvent | None


@dataclass
class TrackState:
    track_id: int
    first_t: float
    last_t: float
    window: deque
    raw_labels: deque
    ema: np.ndarray | None = None
    stable: int | None = None
    candidate: int | None = None
    candidate_count: int = 0
    updates: int = 0
    seconds_per_label: np.ndarray = field(default_factory=lambda: np.zeros(config.NUM_CLASSES))
    counts_per_label: np.ndarray = field(default_factory=lambda: np.zeros(config.NUM_CLASSES, dtype=int))
    changes: int = 0
    stability: float = 1.0


class TemporalAnalyzer:
    def __init__(
        self,
        method: str = "ema",
        alpha: float = 0.35,
        window: int = 8,
        min_persist: int = 3,
        confidence_threshold: float = 0.4,
        stability_window: int = 15,
        ttl_seconds: float = 10.0,
    ) -> None:
        if method not in ("ema", "window", "none"):
            raise ValueError(f"Unknown smoothing method: {method}")
        self.method = method
        self.alpha = alpha
        self.window = max(1, window)
        self.min_persist = max(1, min_persist)
        self.confidence_threshold = confidence_threshold
        self.stability_window = max(1, stability_window)
        self.ttl_seconds = ttl_seconds
        self.tracks: dict[int, TrackState] = {}
        self.finished: list[TrackState] = []
        self.events: list[ChangeEvent] = []

    def _smooth(self, state: TrackState, probs: np.ndarray) -> np.ndarray:
        state.window.append(probs)
        if self.method == "ema":
            state.ema = probs.copy() if state.ema is None else self.alpha * probs + (1 - self.alpha) * state.ema
            return state.ema
        if self.method == "window":
            return np.mean(state.window, axis=0)
        return probs

    def update(self, track_id: int, probs: np.ndarray, t: float) -> TemporalResult:
        probs = np.asarray(probs, dtype=np.float64)
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackState(
                track_id=track_id, first_t=t, last_t=t,
                window=deque(maxlen=self.window), raw_labels=deque(maxlen=self.stability_window),
            )
            self.tracks[track_id] = state

        smoothed = self._smooth(state, probs)
        top = int(np.argmax(smoothed))
        state.raw_labels.append(int(np.argmax(probs)))
        dt = min(max(t - state.last_t, 0.0), MAX_DT)

        change = None
        if state.stable is None:
            state.stable = top
        elif top != state.stable:
            if top == state.candidate:
                state.candidate_count += 1
            else:
                state.candidate, state.candidate_count = top, 1
            if state.candidate_count >= self.min_persist:
                change = ChangeEvent(track_id, t, config.EMOTION_LABELS[state.stable], config.EMOTION_LABELS[top])
                self.events.append(change)
                state.changes += 1
                state.stable, state.candidate, state.candidate_count = top, None, 0
        else:
            state.candidate, state.candidate_count = None, 0

        if state.updates > 0:
            state.seconds_per_label[state.stable] += dt
        state.counts_per_label[state.stable] += 1
        state.updates += 1
        state.last_t = t
        state.stability = float(np.mean(np.array(state.raw_labels) == state.stable))

        confidence = float(smoothed[state.stable])
        confident = confidence >= self.confidence_threshold
        return TemporalResult(
            probabilities=smoothed.astype(np.float32),
            label=config.EMOTION_LABELS[state.stable] if confident else config.UNCERTAIN_LABEL,
            label_index=state.stable if confident else None,
            confidence=confidence,
            stability=state.stability,
            change=change,
        )

    def prune(self, now: float) -> None:
        """Move tracks not updated for ttl_seconds into `finished` (their summaries are kept)."""
        stale = [tid for tid, s in self.tracks.items() if now - s.last_t > self.ttl_seconds]
        for tid in stale:
            self.finished.append(self.tracks.pop(tid))

    def track_summaries(self) -> list[dict]:
        out = []
        for s in [*self.finished, *self.tracks.values()]:
            dominant = int(np.argmax(s.counts_per_label)) if s.updates else None
            out.append({
                "track_id": s.track_id,
                "first_seen_s": round(s.first_t, 2),
                "last_seen_s": round(s.last_t, 2),
                "predictions": int(s.updates),
                "dominant_expression": config.EMOTION_LABELS[dominant] if dominant is not None else None,
                "stability": round(s.stability, 3),
                "changes": int(s.changes),
                **{f"seconds_{name}": round(float(sec), 2)
                   for name, sec in zip(config.EMOTION_LABELS, s.seconds_per_label)},
            })
        return sorted(out, key=lambda d: d["track_id"])

    def reset(self) -> None:
        self.tracks.clear()
        self.finished.clear()
        self.events.clear()
