"""SQLite manager and SessionStats tests (no TensorFlow needed)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import config
from analytics.session import SessionStats
from database.db_manager import DatabaseManager
from recognition.pipeline import FaceResult


def make_result(track_id: int, label: str, conf: float = 0.8) -> FaceResult:
    probs = np.full(config.NUM_CLASSES, 0.02, dtype=np.float32)
    if label in config.EMOTION_LABELS:
        probs[config.EMOTION_LABELS.index(label)] = conf
    return FaceResult(track_id=track_id, bbox=(0, 0, 10, 10), detection_confidence=0.9,
                      raw_label=label if label in config.EMOTION_LABELS else "Neutral",
                      raw_confidence=conf, label=label, confidence=conf, probabilities=probs, stability=1.0)


@pytest.fixture
def session() -> SessionStats:
    s = SessionStats("video", "cnn")
    for i in range(30):
        t = i / 10
        results = [make_result(1, "Happy"), make_result(2, "Sad" if i < 20 else config.UNCERTAIN_LABEL, 0.3)]
        s.record(t, results, processed=True, fps=25.0)
    s.media_seconds = 3.0
    return s


def test_session_summary(session: SessionStats) -> None:
    summary = session.summary()
    assert summary["total_faces"] == 2
    assert summary["max_concurrent_faces"] == 2
    assert summary["total_predictions"] == 60
    assert summary["uncertain_predictions"] == 10
    assert summary["dominant_expression"] == "Happy"
    assert summary["duration_s"] == 3.0
    assert session.expression_counts()["Sad"] == 20


def test_per_second_aggregation(session: SessionStats) -> None:
    per_sec = session.per_second_df()
    assert set(per_sec["t_second"]) == {0, 1, 2}
    assert len(per_sec) == 6  # 3 seconds x 2 tracks


def test_db_save_read_delete(tmp_path: Path, session: SessionStats) -> None:
    db = DatabaseManager(tmp_path / "test.db")
    tracks = [{"track_id": 1, "first_seen_s": 0.0, "last_seen_s": 2.9, "predictions": 30,
               "dominant_expression": "Happy", "stability": 1.0, "changes": 0}]
    sid = db.save_session(session.summary(), session.expression_counts(), session.per_second_df(), tracks)

    assert db.get_session(sid)["total_faces"] == 2
    assert db.get_expression_counts(sid)["Happy"] == 30
    assert len(db.get_timeline(sid)) == 6
    assert len(db.get_tracks(sid)) == 1
    assert db.overall_stats()["sessions"] == 1
    assert db.aggregate_counts()["Happy"] == 30

    assert db.delete_session(sid)
    assert db.get_session(sid) is None
    assert db.get_timeline(sid).empty  # ON DELETE CASCADE removed child rows


def test_db_delete_all(tmp_path: Path) -> None:
    db = DatabaseManager(tmp_path / "test.db")
    for _ in range(3):
        s = SessionStats("image")
        s.record(0.0, [make_result(1, "Neutral")], processed=True, fps=None)
        db.save_session(s.summary(), s.expression_counts())
    assert len(db.list_sessions()) == 3
    assert db.delete_all() == 3
    assert db.list_sessions().empty
