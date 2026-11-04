"""SQLite persistence for session analytics (aggregates only, never face data)."""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

import config
from utils.logger import get_logger

log = get_logger("db")

SESSION_COLUMNS = (
    "session_id", "started_at", "source", "model_name", "duration_s", "frames_processed",
    "total_faces", "max_concurrent_faces", "total_predictions", "uncertain_predictions",
    "avg_confidence", "avg_detection_confidence", "avg_fps", "dominant_expression", "expression_changes",
)


class DatabaseManager:
    def __init__(self, db_path: Path = config.DB_PATH, schema_path: Path = config.SCHEMA_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # A short-lived connection per operation: safe with Streamlit's threads.
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes
    def save_session(
        self,
        summary: dict,
        expression_counts: dict[str, int],
        timeline: pd.DataFrame | None = None,
        track_summaries: list[dict] | None = None,
    ) -> str:
        """Insert (or replace) one session with all its child rows in a single transaction."""
        row = tuple(summary.get(c) for c in SESSION_COLUMNS)
        placeholders = ", ".join("?" for _ in SESSION_COLUMNS)
        sid = summary["session_id"]
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
            conn.execute(f"INSERT INTO sessions ({', '.join(SESSION_COLUMNS)}) VALUES ({placeholders})", row)
            conn.executemany(
                "INSERT INTO expression_counts (session_id, expression, count) VALUES (?, ?, ?)",
                [(sid, k, int(v)) for k, v in expression_counts.items()],
            )
            if timeline is not None and not timeline.empty:
                conn.executemany(
                    "INSERT INTO session_timeline (session_id, t_second, track_id, expression, confidence) "
                    "VALUES (?, ?, ?, ?, ?)",
                    [(sid, int(r.t_second), int(r.track_id), str(r.expression), float(r.confidence))
                     for r in timeline.itertuples(index=False)],
                )
            if track_summaries:
                conn.executemany(
                    "INSERT INTO track_summaries (session_id, track_id, first_seen_s, last_seen_s, predictions, "
                    "dominant_expression, stability, changes) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(sid, t["track_id"], t["first_seen_s"], t["last_seen_s"], t["predictions"],
                      t["dominant_expression"], t["stability"], t["changes"]) for t in track_summaries],
                )
        log.info("Saved session %s (%s)", sid, summary.get("source"))
        return sid

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            deleted = cur.rowcount > 0
        log.info("Deleted session %s: %s", session_id, deleted)
        return deleted

    def delete_all(self) -> int:
        with self._connect() as conn:
            n = conn.execute("DELETE FROM sessions").rowcount
        log.info("Deleted all %d sessions", n)
        return n

    # ------------------------------------------------------------------ reads
    def _frame(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=params)

    def list_sessions(self, source: str | None = None) -> pd.DataFrame:
        if source:
            return self._frame("SELECT * FROM sessions WHERE source = ? ORDER BY started_at DESC", (source,))
        return self._frame("SELECT * FROM sessions ORDER BY started_at DESC")

    def get_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
        return dict(row) if row else None

    def get_expression_counts(self, session_id: str) -> dict[str, int]:
        df = self._frame("SELECT expression, count FROM expression_counts WHERE session_id = ?", (session_id,))
        counts = dict(zip(df["expression"], df["count"].astype(int)))
        return {name: counts.get(name, 0) for name in config.EMOTION_LABELS}

    def get_timeline(self, session_id: str) -> pd.DataFrame:
        return self._frame(
            "SELECT t_second, track_id, expression, confidence FROM session_timeline "
            "WHERE session_id = ? ORDER BY t_second, track_id", (session_id,))

    def get_tracks(self, session_id: str) -> pd.DataFrame:
        return self._frame("SELECT * FROM track_summaries WHERE session_id = ? ORDER BY track_id", (session_id,))

    def aggregate_counts(self) -> dict[str, int]:
        df = self._frame("SELECT expression, SUM(count) AS total FROM expression_counts GROUP BY expression")
        counts = dict(zip(df["expression"], df["total"].fillna(0).astype(int)))
        return {name: counts.get(name, 0) for name in config.EMOTION_LABELS}

    def overall_stats(self) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS sessions, COALESCE(SUM(total_faces), 0) AS faces, "
                "COALESCE(SUM(total_predictions), 0) AS predictions, "
                "COALESCE(SUM(duration_s), 0) AS seconds, "
                "SUM(avg_confidence * total_predictions) / NULLIF(SUM(total_predictions), 0) AS avg_conf "
                "FROM sessions"
            ).fetchone()
        return dict(row)
