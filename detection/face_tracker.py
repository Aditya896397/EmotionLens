"""Multi-face tracking with ByteTrack (Zhang et al., ECCV 2022), implemented from scratch.

Why ByteTrack instead of DeepSORT: ByteTrack associates boxes using motion (a Kalman
filter) and overlap (IoU) only. It needs no appearance embedding network, so no
biometric face embeddings are computed or stored, and it stays fast on a CPU.

Per frame:
  1. Predict every existing track forward with its Kalman filter.
  2. Split detections into high-score and low-score sets.
  3. First association: high-score detections <-> all tracks (active + lost), IoU + Hungarian.
  4. Second association: low-score detections <-> remaining *active* tracks. This keeps
     tracks alive through blur / partial occlusion, when detector confidence drops.
  5. Unmatched tracks become "lost"; lost tracks are deleted after `max_lost` frames.
     A lost track that is matched again keeps its original ID (re-appearance handling).
  6. Unmatched high-score detections start new tentative tracks, confirmed after `min_hits`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------------------------
# Kalman filter: state = [cx, cy, w, h, vcx, vcy, vw, vh], constant-velocity model
# ---------------------------------------------------------------------------
class KalmanBoxFilter:
    _F = np.eye(8)
    _F[:4, 4:] = np.eye(4)          # position += velocity each step
    _H = np.eye(4, 8)               # we observe position only
    _STD_POS = 1.0 / 20             # noise scaled by box height, as in ByteTrack/DeepSORT
    _STD_VEL = 1.0 / 160

    def __init__(self, box_xyxy: np.ndarray) -> None:
        z = self.xyxy_to_cxcywh(box_xyxy)
        self.x = np.concatenate([z, np.zeros(4)])
        h = max(z[3], 1.0)
        std = np.array([2 * self._STD_POS * h] * 4 + [10 * self._STD_VEL * h] * 4)
        self.P = np.diag(std ** 2)

    @staticmethod
    def xyxy_to_cxcywh(b: np.ndarray) -> np.ndarray:
        w, h = b[2] - b[0], b[3] - b[1]
        return np.array([b[0] + w / 2, b[1] + h / 2, w, h], dtype=np.float64)

    def predict(self) -> None:
        h = max(self.x[3], 1.0)
        q = np.array([self._STD_POS * h] * 4 + [self._STD_VEL * h] * 4)
        self.x = self._F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = self._F @ self.P @ self._F.T + np.diag(q ** 2)

    def update(self, box_xyxy: np.ndarray) -> None:
        z = self.xyxy_to_cxcywh(box_xyxy)
        h = max(self.x[3], 1.0)
        r = np.diag((np.array([self._STD_POS * h] * 4)) ** 2)
        s = self._H @ self.P @ self._H.T + r
        k = self.P @ self._H.T @ np.linalg.inv(s)
        self.x = self.x + k @ (z - self._H @ self.x)
        self.P = (np.eye(8) - k @ self._H) @ self.P

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between boxes a (N,4) and b (M,4) in xyxy format."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = a[:, None, :]
    b = b[None, :, :]
    ix1, iy1 = np.maximum(a[..., 0], b[..., 0]), np.maximum(a[..., 1], b[..., 1])
    ix2, iy2 = np.minimum(a[..., 2], b[..., 2]), np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def associate(tracks_boxes: np.ndarray, det_boxes: np.ndarray, iou_thresh: float):
    """Hungarian matching on (1 - IoU). Returns (matches, unmatched_tracks, unmatched_dets)."""
    if len(tracks_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(tracks_boxes))), list(range(len(det_boxes)))
    iou = iou_matrix(tracks_boxes, det_boxes)
    rows, cols = linear_sum_assignment(1.0 - iou)
    matches, matched_t, matched_d = [], set(), set()
    for r, c in zip(rows, cols):
        if iou[r, c] >= iou_thresh:
            matches.append((r, c))
            matched_t.add(r)
            matched_d.add(c)
    unmatched_t = [i for i in range(len(tracks_boxes)) if i not in matched_t]
    unmatched_d = [j for j in range(len(det_boxes)) if j not in matched_d]
    return matches, unmatched_t, unmatched_d


# ---------------------------------------------------------------------------
# Tracks
# ---------------------------------------------------------------------------
@dataclass
class Track:
    track_id: int
    kf: KalmanBoxFilter
    score: float
    hits: int = 1
    lost_frames: int = 0
    confirmed: bool = False
    det_index: int | None = None   # index into the current frame's detections (None if not matched)
    age: int = 1

    @property
    def box(self) -> np.ndarray:
        return self.kf.box


@dataclass
class TrackOutput:
    track_id: int
    bbox: np.ndarray       # smoothed box from the Kalman filter (xyxy)
    det_index: int         # which detection of this frame it matched
    score: float           # detection confidence of the matched detection


@dataclass
class ByteTracker:
    high_thresh: float = 0.6
    low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    iou_thresh: float = 0.3
    second_iou_thresh: float = 0.5
    max_lost: int = 30
    min_hits: int = 2
    tracks: list[Track] = field(default_factory=list)
    frame_id: int = 0
    _next_id: int = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.frame_id = 0
        self._next_id = 1

    @property
    def active_count(self) -> int:
        return sum(1 for t in self.tracks if t.confirmed and t.lost_frames == 0)

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[TrackOutput]:
        """Feed one frame of detections; returns confirmed tracks matched in this frame."""
        self.frame_id += 1
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        for t in self.tracks:
            t.kf.predict()
            t.det_index = None
            t.age += 1

        high = np.flatnonzero(scores >= self.high_thresh)
        low = np.flatnonzero((scores >= self.low_thresh) & (scores < self.high_thresh))

        # 1st association: high-score detections with every track (active and lost).
        pool = list(range(len(self.tracks)))
        track_boxes = np.array([self.tracks[i].box for i in pool]).reshape(-1, 4)
        matches, unmatched_t, unmatched_high = associate(track_boxes, boxes[high], self.iou_thresh)
        for ti, di in matches:
            self._apply_match(self.tracks[pool[ti]], high[di], boxes, scores)
        remaining = [pool[i] for i in unmatched_t]

        # 2nd association: low-score detections with tracks that were active last frame.
        active_remaining = [i for i in remaining if self.tracks[i].lost_frames == 0]
        track_boxes = np.array([self.tracks[i].box for i in active_remaining]).reshape(-1, 4)
        matches2, unmatched_t2, _ = associate(track_boxes, boxes[low], self.second_iou_thresh)
        for ti, di in matches2:
            self._apply_match(self.tracks[active_remaining[ti]], low[di], boxes, scores)
        still_unmatched = {active_remaining[i] for i in unmatched_t2}
        still_unmatched |= {i for i in remaining if self.tracks[i].lost_frames > 0}

        # Unmatched tracks: lost (kept for re-identification) or deleted.
        for i in still_unmatched:
            t = self.tracks[i]
            t.lost_frames += 1
            if not t.confirmed:
                t.lost_frames = self.max_lost + 1  # tentative tracks die immediately
        self.tracks = [t for t in self.tracks if t.lost_frames <= self.max_lost]

        # New tracks from confident unmatched detections.
        for j in unmatched_high:
            d = high[j]
            if scores[d] >= self.new_track_thresh:
                track = Track(self._next_id, KalmanBoxFilter(boxes[d]), float(scores[d]), det_index=int(d))
                # On the very first frame there is no history to confirm against.
                track.confirmed = self.frame_id == 1 or self.min_hits <= 1
                self._next_id += 1
                self.tracks.append(track)

        return [
            TrackOutput(t.track_id, t.box, t.det_index, t.score)
            for t in self.tracks
            if t.confirmed and t.det_index is not None
        ]

    def _apply_match(self, track: Track, det: int, boxes: np.ndarray, scores: np.ndarray) -> None:
        track.kf.update(boxes[det])
        track.score = float(scores[det])
        track.det_index = int(det)
        track.hits += 1
        track.lost_frames = 0
        if track.hits >= self.min_hits:
            track.confirmed = True
