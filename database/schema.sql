-- EmotionLens AI session analytics schema.
-- Privacy: only aggregate statistics, labels, confidences and anonymous integer
-- track IDs are stored. No face images, crops or embeddings.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id               TEXT PRIMARY KEY,
    started_at               TEXT    NOT NULL,
    source                   TEXT    NOT NULL CHECK (source IN ('webcam', 'video', 'image')),
    model_name               TEXT,
    duration_s               REAL    NOT NULL DEFAULT 0,
    frames_processed         INTEGER NOT NULL DEFAULT 0,
    total_faces              INTEGER NOT NULL DEFAULT 0,
    max_concurrent_faces     INTEGER NOT NULL DEFAULT 0,
    total_predictions        INTEGER NOT NULL DEFAULT 0,
    uncertain_predictions    INTEGER NOT NULL DEFAULT 0,
    avg_confidence           REAL,
    avg_detection_confidence REAL,
    avg_fps                  REAL,
    dominant_expression      TEXT,
    expression_changes       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS expression_counts (
    session_id TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    expression TEXT    NOT NULL,
    count      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, expression)
);

-- One row per (second, track): most frequent smoothed label and mean confidence.
CREATE TABLE IF NOT EXISTS session_timeline (
    session_id TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    t_second   INTEGER NOT NULL,
    track_id   INTEGER NOT NULL,
    expression TEXT    NOT NULL,
    confidence REAL,
    PRIMARY KEY (session_id, t_second, track_id)
);

CREATE TABLE IF NOT EXISTS track_summaries (
    session_id          TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    track_id            INTEGER NOT NULL,
    first_seen_s        REAL,
    last_seen_s         REAL,
    predictions         INTEGER,
    dominant_expression TEXT,
    stability           REAL,
    changes             INTEGER,
    PRIMARY KEY (session_id, track_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_timeline_session ON session_timeline(session_id);
