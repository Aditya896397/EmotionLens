"""Home Dashboard page."""
from __future__ import annotations

import streamlit as st

from analytics import charts
from analytics.dashboard import disclaimer, fmt_duration, fmt_pct, hero, kpi_row
from recognition.emotion_classifier import best_model_name, list_models
from ui.state import get_db, get_settings
from utils.device import configure_tensorflow


def render() -> None:
    hero("EmotionLens AI",
         "Real-time facial expression recognition, multi-face tracking and expression analytics.")
    disclaimer()

    db = get_db()
    stats = db.overall_stats()
    counts = db.aggregate_counts()
    top = max(counts, key=counts.get) if sum(counts.values()) else "-"
    kpi_row([
        ("Saved sessions", stats["sessions"]),
        ("Faces tracked", stats["faces"], "unique track IDs across sessions"),
        ("Predictions", f"{stats['predictions']:,}"),
        ("Avg expression confidence", fmt_pct(stats["avg_conf"])),
        ("Analysed time", fmt_duration(stats["seconds"])),
        ("Most frequent", top),
    ])

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recent sessions")
        sessions = db.list_sessions()
        if sessions.empty:
            st.info("No sessions yet. Start with **Live Webcam Detection**, **Image Analysis** or **Video Analysis**.")
        else:
            cols = ["started_at", "source", "duration_s", "total_faces", "dominant_expression", "avg_confidence"]
            st.dataframe(sessions[cols].head(8), hide_index=True, use_container_width=True)
            st.plotly_chart(charts.sessions_bar(sessions.head(30)), use_container_width=True)
    with right:
        st.subheader("All-time expression distribution")
        st.plotly_chart(charts.distribution_donut(counts, title=""), use_container_width=True)

    st.subheader("System status")
    device = configure_tensorflow()
    settings = get_settings()
    models = list_models()
    active = settings.model_name or best_model_name() or "none trained"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Compute device", device.device, help=", ".join(device.gpu_names) or device.cpu_name)
    c2.metric("Expression model", active, help=f"{len(models)} checkpoint(s) available")
    c3.metric("Detector", settings.detector_backend, help="auto = MediaPipe, Haar cascade fallback")
    c4.metric("TensorFlow", device.tensorflow_version)

    with st.expander("How the pipeline works"):
        st.markdown(
            "1. **Detect** faces with MediaPipe BlazeFace (Haar cascade fallback). Dark frames are "
            "enhanced first.\n"
            "2. **Track** faces across frames with ByteTrack (Kalman filter + IoU + Hungarian matching). "
            "No appearance embeddings are used.\n"
            "3. **Align & crop**: rotate so the eyes are level, then crop to 48x48 grayscale.\n"
            "4. **Classify** all faces in one batched CNN forward pass.\n"
            "5. **Smooth** per track (EMA), with hysteresis for expression-change events.\n"
            "6. **Store** aggregate statistics only (no images) in SQLite."
        )
