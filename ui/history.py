"""Session History page: browse, export and delete stored session records."""
from __future__ import annotations

import streamlit as st

from analytics import charts
from analytics.dashboard import fmt_duration, fmt_pct, hero, kpi_row, note
from analytics.report_generator import build_pdf_report, dataframe_csv
from ui.state import get_db


def render() -> None:
    hero("Session History", "Stored session statistics. No face images or embeddings are ever saved.")
    note("Each record holds counts, confidences, durations and anonymous track IDs only. "
         "You can delete single sessions or all records at any time.")
    db = get_db()

    stats = db.overall_stats()
    kpi_row([
        ("Sessions", stats["sessions"]),
        ("Faces tracked", stats["faces"]),
        ("Predictions", f"{stats['predictions']:,}"),
        ("Avg expression conf.", fmt_pct(stats["avg_conf"])),
        ("Total time", fmt_duration(stats["seconds"])),
    ])
    st.write("")

    sessions = db.list_sessions()
    if sessions.empty:
        st.info("No sessions stored yet.")
        return

    sources = st.multiselect("Source", ["webcam", "video", "image"], default=["webcam", "video", "image"])
    view = sessions[sessions["source"].isin(sources)]
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.download_button("Export table as CSV", dataframe_csv(view), file_name="emotionlens_sessions.csv",
                       mime="text/csv")

    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.sessions_bar(view), use_container_width=True)
    c2.plotly_chart(charts.distribution_donut(db.aggregate_counts(), title="All sessions"),
                    use_container_width=True)

    st.subheader("Session details")
    if view.empty:
        return
    labels = [f"{r.started_at} | {r.source} | {r.session_id}" for r in view.itertuples()]
    choice = st.selectbox("Select a session", labels)
    sid = choice.split(" | ")[-1]
    summary = db.get_session(sid)
    counts = db.get_expression_counts(sid)
    tracks = db.get_tracks(sid)

    d1, d2 = st.columns([2, 3])
    d1.json(summary)
    d2.plotly_chart(charts.distribution_bar(counts), use_container_width=True)
    if not tracks.empty:
        st.dataframe(tracks, hide_index=True, use_container_width=True)

    a1, a2, a3 = st.columns(3)
    a1.download_button("PDF report", build_pdf_report("Session report", summary, counts, tracks=tracks),
                       file_name=f"emotionlens_session_{sid}.pdf", mime="application/pdf")
    confirm_one = a2.checkbox("Confirm delete of this session", key=f"confirm_{sid}")
    if a2.button("Delete session", disabled=not confirm_one):
        db.delete_session(sid)
        last = st.session_state.get("last_session") or {}
        if last.get("summary", {}).get("session_id") == sid:
            st.session_state.pop("last_session", None)
        st.success(f"Deleted {sid}")
        st.rerun()

    with a3.expander("Delete all records"):
        confirm_all = st.checkbox("I understand this permanently deletes every session record")
        if st.button("Delete all sessions", type="primary", disabled=not confirm_all):
            n = db.delete_all()
            st.session_state.pop("last_session", None)
            st.success(f"Deleted {n} sessions")
            st.rerun()
