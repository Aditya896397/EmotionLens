"""Video Analysis page: frame-by-frame detection, tracking and classification of an upload."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from analytics import charts
from analytics.dashboard import disclaimer, fmt_duration, fmt_pct, hero, kpi_row
from analytics.report_generator import build_pdf_report, dataframe_csv, summary_json
from analytics.session import SessionStats
from recognition.pipeline import FrameProcessor
from ui.state import get_classifier, get_settings, save_session
from utils.image_utils import resize_max_width
from utils.logger import get_logger
from utils.video_utils import create_video_writer, iterate_video, video_properties

log = get_logger("ui.video")


def _remove(path: Path | None) -> None:
    if path and path.exists():
        try:
            os.remove(path)
        except OSError as exc:
            log.warning("Could not delete temp file %s: %s", path, exc)


def _process(upload, classifier, settings, max_seconds: int) -> dict:
    suffix = Path(upload.name).suffix or ".mp4"
    in_path = out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(upload.getvalue())
            in_path = Path(f.name)
        props = video_properties(in_path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            out_path = Path(f.name)

        session = SessionStats("video", classifier.name)
        processor = FrameProcessor(settings, classifier, use_tracking=True, session=session)
        writer, codec = create_video_writer(out_path, props.fps, (props.width, props.height))
        total = min(props.frame_count or 1, int(max_seconds * props.fps)) or 1
        progress = st.progress(0.0, text="Processing frames ...")
        preview = st.empty()
        last_t = 0.0
        try:
            for idx, t, frame in iterate_video(in_path, max_seconds=max_seconds):
                annotated, _ = processor.process(frame, t=t)
                writer.write(annotated)
                last_t = t
                if idx % 10 == 0:
                    progress.progress(min(idx / total, 1.0), text=f"Frame {idx}/{total}  ({t:.1f}s)")
                if idx % 30 == 0:
                    preview.image(resize_max_width(annotated, 720), channels="BGR")
        finally:
            writer.release()
            processor.close()
        progress.progress(1.0, text="Done")
        preview.empty()

        session.media_seconds = last_t + 1.0 / props.fps
        save_session(session, processor.temporal)
        return {
            "name": upload.name,
            "video": out_path.read_bytes(),
            "codec": codec,
            "props": props,
            "summary": session.summary(),
            "counts": session.expression_counts(),
            "timeline": session.timeline_df(),
            "tracks": pd.DataFrame(processor.temporal.track_summaries()),
            "changes": list(session.changes),
            "truncated": props.duration_s > max_seconds,
        }
    finally:
        _remove(in_path)
        _remove(out_path)


def render() -> None:
    hero("Video Analysis", "Process an MP4/AVI file: detect, track and classify every face, then export.")
    disclaimer()
    classifier = get_classifier()
    if classifier is None:
        return
    settings = get_settings()

    upload = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    c1, c2 = st.columns(2)
    max_seconds = c1.slider("Analyse the first N seconds", 5, 600, settings.video_max_seconds, step=5)
    c2.caption(f"Frame skip: {settings.frame_skip} (change on the Settings page). Higher values are faster; "
               "skipped frames reuse the previous overlay.")

    if upload is not None and st.button("Analyse video", type="primary"):
        try:
            st.session_state.video_result = _process(upload, classifier, settings, max_seconds)
        except (ValueError, RuntimeError) as exc:
            st.error(f"Video processing failed: {exc}")
            return

    result = st.session_state.get("video_result")
    if not result:
        st.info("Upload a video and click **Analyse video**. The processed video and a summary are "
                "available for download afterwards.")
        return

    s = result["summary"]
    st.subheader(f"Results: {result['name']}")
    if result["truncated"]:
        st.caption(f"Only the first {max_seconds}s were analysed.")
    kpi_row([
        ("Faces tracked", s["total_faces"]),
        ("Max faces at once", s["max_concurrent_faces"]),
        ("Predictions", f"{s['total_predictions']:,}"),
        ("Avg expression conf.", fmt_pct(s["avg_confidence"])),
        ("Expression changes", s["expression_changes"]),
        ("Duration", fmt_duration(s["duration_s"])),
    ])
    st.write("")

    v1, v2 = st.columns([3, 2])
    with v1:
        if result["codec"] == "avc1":
            st.video(result["video"], format="video/mp4")
        else:
            st.info("This OpenCV build wrote MPEG-4 Part 2 video, which most browsers cannot preview. "
                    "Download it below and play it in VLC or Windows Media Player.")
    with v2:
        st.plotly_chart(charts.distribution_donut(result["counts"]), use_container_width=True)

    st.plotly_chart(charts.timeline_area(result["timeline"]), use_container_width=True)
    st.plotly_chart(charts.track_history(result["timeline"]), use_container_width=True)
    if not result["tracks"].empty:
        st.plotly_chart(charts.stability_bar(result["tracks"]), use_container_width=True)

    st.subheader("Export")
    d1, d2, d3, d4 = st.columns(4)
    sid = s["session_id"]
    d1.download_button("Processed video (MP4)", result["video"], file_name=f"emotionlens_{sid}.mp4",
                       mime="video/mp4")
    d2.download_button("PDF summary", build_pdf_report(
        f"Video analysis report - {result['name']}", s, result["counts"],
        tracks=result["tracks"], changes=result["changes"][:200]),
        file_name=f"emotionlens_video_{sid}.pdf", mime="application/pdf")
    d3.download_button("Timeline CSV", dataframe_csv(result["timeline"]),
                       file_name=f"emotionlens_timeline_{sid}.csv", mime="text/csv")
    d4.download_button("Summary JSON", summary_json(s, result["counts"], result["tracks"].to_dict("records")),
                       file_name=f"emotionlens_summary_{sid}.json", mime="application/json")
