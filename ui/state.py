"""Shared Streamlit state: settings, cached resources and session persistence helpers."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from analytics.session import SessionStats
from database.db_manager import DatabaseManager
from recognition.emotion_classifier import EmotionClassifier, ModelNotFoundError, resolve_model_name
from recognition.temporal_analyzer import TemporalAnalyzer
from utils.logger import get_logger

log = get_logger("ui")


def get_settings() -> config.AppSettings:
    if "settings" not in st.session_state:
        st.session_state.settings = config.load_settings()
    return st.session_state.settings


@st.cache_resource(show_spinner=False)
def get_db() -> DatabaseManager:
    return DatabaseManager()


@st.cache_resource(show_spinner="Loading expression model ...")
def _load_classifier(name: str) -> EmotionClassifier:
    return EmotionClassifier(name)


def get_classifier() -> EmotionClassifier | None:
    """Return the configured classifier, or None (with an on-page message) if none is trained."""
    settings = get_settings()
    try:
        name = resolve_model_name(settings.model_name)
        return _load_classifier(name)
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.code("python -m scripts.prepare_dataset\npython -m models.train --model cnn\n"
                "python -m models.evaluate", language="bash")
        return None
    except Exception as exc:
        log.exception("Model loading failed")
        st.error(f"Could not load the expression model: {exc}")
        return None


def save_session(session: SessionStats, temporal: TemporalAnalyzer | None) -> str:
    db = get_db()
    tracks = temporal.track_summaries() if temporal is not None else []
    sid = db.save_session(session.summary(), session.expression_counts(), session.per_second_df(), tracks)
    st.session_state.last_session = {
        "summary": session.summary(),
        "counts": session.expression_counts(),
        "timeline": session.timeline_df(),
        "tracks": pd.DataFrame(tracks),
        "changes": list(session.changes),
    }
    return sid


def load_eval_results() -> list[dict]:
    results = []
    for path in sorted(config.EXPERIMENTS_DIR.glob("eval_*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return results
