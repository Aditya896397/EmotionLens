"""Temporal analyzer tests: smoothing, hysteresis, uncertainty, stability, time accounting."""
from __future__ import annotations

import numpy as np

import config
from recognition.temporal_analyzer import TemporalAnalyzer

HAPPY = config.EMOTION_LABELS.index("Happy")
SAD = config.EMOTION_LABELS.index("Sad")


def one_hot(idx: int, p: float = 0.9) -> np.ndarray:
    v = np.full(config.NUM_CLASSES, (1 - p) / (config.NUM_CLASSES - 1))
    v[idx] = p
    return v


def test_single_outlier_frame_does_not_change_label() -> None:
    ta = TemporalAnalyzer(method="ema", alpha=0.35, min_persist=3)
    for t in range(10):
        ta.update(1, one_hot(HAPPY), t * 0.1)
    r = ta.update(1, one_hot(SAD), 1.0)
    assert r.label == "Happy" and r.change is None


def test_persistent_change_is_detected_once() -> None:
    ta = TemporalAnalyzer(method="ema", alpha=0.5, min_persist=3)
    for t in range(10):
        ta.update(1, one_hot(HAPPY), t * 0.1)
    changes = [ta.update(1, one_hot(SAD), 1 + t * 0.1).change for t in range(15)]
    events = [c for c in changes if c is not None]
    assert len(events) == 1
    assert (events[0].from_label, events[0].to_label) == ("Happy", "Sad")


def test_raw_mode_flips_immediately_with_no_persistence() -> None:
    ta = TemporalAnalyzer(method="none", min_persist=1)
    ta.update(1, one_hot(HAPPY), 0.0)
    assert ta.update(1, one_hot(SAD), 0.1).label == "Sad"


def test_low_confidence_is_uncertain() -> None:
    ta = TemporalAnalyzer(confidence_threshold=0.5)
    r = ta.update(1, one_hot(HAPPY, p=0.3), 0.0)
    assert r.label == config.UNCERTAIN_LABEL and r.label_index is None


def test_stability_and_time_accounting() -> None:
    ta = TemporalAnalyzer(method="ema", stability_window=10)
    for t in range(11):
        r = ta.update(7, one_hot(HAPPY), t * 0.5)
    assert r.stability == 1.0
    summary = ta.track_summaries()[0]
    assert summary["dominant_expression"] == "Happy"
    assert abs(summary["seconds_Happy"] - 5.0) < 1e-6


def test_window_smoothing_averages() -> None:
    ta = TemporalAnalyzer(method="window", window=2)
    ta.update(1, one_hot(HAPPY, 1.0), 0.0)
    r = ta.update(1, one_hot(SAD, 1.0), 0.1)
    assert abs(r.probabilities[HAPPY] - 0.5) < 1e-6


def test_prune_moves_stale_tracks_to_finished() -> None:
    ta = TemporalAnalyzer(ttl_seconds=1.0)
    ta.update(1, one_hot(HAPPY), 0.0)
    ta.prune(5.0)
    assert 1 not in ta.tracks and len(ta.finished) == 1
    assert len(ta.track_summaries()) == 1
