"""Expression Analytics page: interactive analysis of the latest or any stored session."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from analytics import charts
from analytics.dashboard import disclaimer, fmt_duration, fmt_pct, hero, kpi_row, note
from analytics.report_generator import build_pdf_report, dataframe_csv
from ui.state import get_db

LATEST = "Latest analysed session (full resolution)"


def _load(choice: str, sessions: pd.DataFrame) -> dict | None:
    if choice == LATEST:
        return st.session_state.get("last_session")
    db = get_db()
    sid = choice.split(" | ")[-1]
    summary = db.get_session(sid)
    if summary is None:
        return None
    timeline = db.get_timeline(sid).rename(columns={"t_second": "t"})
    return {
        "summary": summary,
        "counts": db.get_expression_counts(sid),
        "timeline": timeline,
        "tracks": db.get_tracks(sid),
        "changes": [],
    }


def render() -> None:
    hero("Expression Analytics", "Distributions, timelines, per-track history and stability, with filters.")
    disclaimer()

    sessions = get_db().list_sessions()
    options = []
    if st.session_state.get("last_session"):
        options.append(LATEST)
    options += [f"{r.started_at} | {r.source} | {r.session_id}" for r in sessions.itertuples()]
    if not options:
        st.info("No sessions yet. Run a webcam, image or video analysis first.")
        return

    choice = st.selectbox("Session", options)
    data = _load(choice, sessions)
    if data is None:
        st.error("Session not found (it may have been deleted).")
        return
    if choice != LATEST:
        note("Stored sessions keep one aggregated row per second per track (most frequent label), "
             "not every frame.")

    s = data["summary"]
    timeline: pd.DataFrame = data["timeline"]
    tracks: pd.DataFrame = data["tracks"]

    kpi_row([
        ("Faces tracked", s["total_faces"]),
        ("Max faces at once", s["max_concurrent_faces"]),
        ("Predictions", f"{s['total_predictions']:,}"),
        ("Uncertain", f"{s['uncertain_predictions']:,}"),
        ("Avg expression conf.", fmt_pct(s["avg_confidence"])),
        ("Duration", fmt_duration(s["duration_s"])),
    ])
    st.write("")

    if timeline.empty:
        st.plotly_chart(charts.distribution_bar(data["counts"]), use_container_width=True)
        st.caption("This session has no timeline (e.g. a single image).")
        return

    # ---- Filters
    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([2, 2, 3, 1])
        labels = [l for l in [*config.EMOTION_LABELS, config.UNCERTAIN_LABEL] if l in set(timeline["expression"])]
        chosen = f1.multiselect("Expressions", labels, default=labels)
        track_ids = sorted(timeline["track_id"].unique().tolist())
        chosen_tracks = f2.multiselect("Track IDs", track_ids, default=track_ids)
        t_min, t_max = float(timeline["t"].min()), float(timeline["t"].max())
        if t_max > t_min:
            t_range = f3.slider("Time range (s)", t_min, t_max, (t_min, t_max))
        else:
            t_range = (t_min, t_max)
        bin_s = f4.number_input("Bin (s)", 0.5, 60.0, 2.0, 0.5)

    view = timeline[
        timeline["expression"].isin(chosen)
        & timeline["track_id"].isin(chosen_tracks)
        & timeline["t"].between(*t_range)
    ]
    counts = view["expression"].value_counts().to_dict()
    filtered_counts = {l: int(counts.get(l, 0)) for l in labels}

    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.distribution_donut(filtered_counts), use_container_width=True)
    c2.plotly_chart(charts.distribution_bar(filtered_counts), use_container_width=True)
    st.plotly_chart(charts.timeline_area(view, bin_seconds=bin_s), use_container_width=True)
    st.plotly_chart(charts.track_history(view), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(charts.confidence_histogram(view), use_container_width=True)
    if not tracks.empty:
        c4.plotly_chart(charts.stability_bar(tracks[tracks["track_id"].isin(chosen_tracks)]),
                        use_container_width=True)

    seconds_cols = [c for c in tracks.columns if c.startswith("seconds_")]
    if seconds_cols:
        st.subheader("Time in each smoothed expression per track (s)")
        st.dataframe(tracks[["track_id", *seconds_cols]].rename(columns=lambda c: c.replace("seconds_", "")),
                     hide_index=True, use_container_width=True)

    if data["changes"]:
        st.subheader("Expression change events")
        st.dataframe(pd.DataFrame(data["changes"]), hide_index=True, use_container_width=True)

    st.subheader("Export")
    e1, e2 = st.columns(2)
    e1.download_button("Filtered timeline CSV", dataframe_csv(view),
                       file_name=f"emotionlens_timeline_{s['session_id']}.csv", mime="text/csv")
    e2.download_button("PDF report", build_pdf_report("Expression analytics report", dict(s), data["counts"],
                                                      tracks=tracks, changes=data["changes"][:200]),
                       file_name=f"emotionlens_report_{s['session_id']}.pdf", mime="application/pdf")
