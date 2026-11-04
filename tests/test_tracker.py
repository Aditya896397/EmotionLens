"""ByteTrack tests: ID consistency, multiple faces, occlusion recovery, low-score rescue."""
from __future__ import annotations

import numpy as np

from detection.face_tracker import ByteTracker, iou_matrix


def box(x: float, y: float, s: float = 80) -> list[float]:
    return [x, y, x + s, y + s]


def test_iou_matrix_basic() -> None:
    a = np.array([box(0, 0, 10)])
    b = np.array([box(0, 0, 10), box(5, 0, 10), box(100, 100, 10)])
    iou = iou_matrix(a, b)[0]
    assert iou[0] == 1.0
    assert abs(iou[1] - 50 / 150) < 1e-6
    assert iou[2] == 0.0


def test_single_face_keeps_id_while_moving() -> None:
    tracker = ByteTracker(min_hits=1)
    ids = set()
    for t in range(30):
        out = tracker.update(np.array([box(100 + 4 * t, 120)]), np.array([0.95]))
        assert len(out) == 1
        ids.add(out[0].track_id)
    assert ids == {1}


def test_two_faces_get_distinct_stable_ids() -> None:
    tracker = ByteTracker(min_hits=1)
    history = {0: set(), 1: set()}
    for t in range(20):
        boxes = np.array([box(50 + 2 * t, 100), box(400 - 2 * t, 100)])
        out = tracker.update(boxes, np.array([0.9, 0.9]))
        for o in out:
            history[o.det_index].add(o.track_id)
    assert all(len(v) == 1 for v in history.values())
    assert history[0] != history[1]


def test_reappearance_after_short_occlusion_keeps_id() -> None:
    tracker = ByteTracker(min_hits=1, max_lost=10)
    for t in range(10):
        first = tracker.update(np.array([box(100 + 3 * t, 100)]), np.array([0.9]))
    original = first[0].track_id
    for _ in range(5):  # face hidden
        assert tracker.update(np.zeros((0, 4)), np.zeros(0)) == []
    back = tracker.update(np.array([box(100 + 3 * 15, 100)]), np.array([0.9]))
    assert back and back[0].track_id == original


def test_track_deleted_after_max_lost_gets_new_id() -> None:
    tracker = ByteTracker(min_hits=1, max_lost=3)
    first = tracker.update(np.array([box(100, 100)]), np.array([0.9]))[0].track_id
    for _ in range(6):
        tracker.update(np.zeros((0, 4)), np.zeros(0))
    again = tracker.update(np.array([box(100, 100)]), np.array([0.9]))
    assert again[0].track_id != first


def test_low_score_detection_keeps_existing_track_alive() -> None:
    tracker = ByteTracker(min_hits=1, high_thresh=0.6, low_thresh=0.1)
    tid = tracker.update(np.array([box(100, 100)]), np.array([0.9]))[0].track_id
    out = tracker.update(np.array([box(102, 100)]), np.array([0.3]))  # blurred frame: low confidence
    assert out and out[0].track_id == tid


def test_low_score_detection_does_not_start_new_track() -> None:
    tracker = ByteTracker(min_hits=1)
    assert tracker.update(np.array([box(100, 100)]), np.array([0.3])) == []
    assert tracker.tracks == []


def test_tentative_track_needs_min_hits_after_first_frame() -> None:
    tracker = ByteTracker(min_hits=2)
    tracker.update(np.zeros((0, 4)), np.zeros(0))          # frame 1: empty
    assert tracker.update(np.array([box(10, 10)]), np.array([0.9])) == []   # frame 2: tentative
    out = tracker.update(np.array([box(12, 10)]), np.array([0.9]))          # frame 3: confirmed
    assert len(out) == 1
