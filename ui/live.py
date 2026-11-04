"""Live Webcam Detection page.

Streamlit reruns the whole script when a widget changes. Turning the "Start camera"
toggle off triggers a rerun that interrupts the capture loop below; the `with Camera`
block guarantees the device is released, and the next run saves the session.
"""
from __future__ import annotations

import time

import streamlit as st

from analytics import charts
from analytics.dashboard import disclaimer, fmt_duration, fmt_pct, hero, kpi_row_into, note
from analytics.session import SessionStats
from recognition.pipeline import FrameProcessor
from ui.state import get_classifier, get_settings, save_session
from utils.image_utils import resize_max_width
from utils.logger import get_logger
from utils.video_utils import Camera, CameraError

log = get_logger("ui.live")

KPI_EVERY_N_FRAMES = 10


def _finalize() -> None:
    live = st.session_state.get("live")
    if not live:
        return
    try:
        if live["session"].frames_seen > 0:
            sid = save_session(live["session"], live["processor"].temporal)
            st.success(f"Session `{sid}` saved: {live['session'].frames_processed} frames analysed.")
    finally:
        live["processor"].close()
        st.session_state.live = None


def _kpis(processor: FrameProcessor, session: SessionStats, results) -> list[tuple]:
    return [
        ("Faces now", len(results)),
        ("Faces tracked", len(session.track_ids), "unique IDs this session"),
        ("FPS", f"{processor.fps:.1f}" if processor.fps else "-"),
        ("Pipeline latency", f"{processor.last_pipeline_ms:.0f} ms"),
        ("Avg expression conf.", fmt_pct(session.avg_confidence)),
        ("Session time", fmt_duration(session.duration_s)),
    ]


def _show_last_session() -> None:
    last = st.session_state.get("last_session")
    if not last or last["summary"]["source"] != "webcam":
        return
    st.subheader("Last webcam session")
    c1, c2 = st.columns(2)
    c1.plotly_chart(charts.distribution_donut(last["counts"]), use_container_width=True)
    c2.plotly_chart(charts.timeline_area(last["timeline"]), use_container_width=True)
    st.caption("Open **Expression Analytics** for per-track history, filters and reports.")


def render() -> None:
    hero("Live Webcam Detection", "Multi-face detection, tracking and smoothed expression labels in real time.")
    disclaimer()

    classifier = get_classifier()
    if classifier is None:
        return
    settings = get_settings()

    run = st.toggle("Start camera", key="live_run",
                    help="Turn off to stop the camera and save the session summary.")
    note("Overlay: 'ID n | Label xx%' = smoothed expression confidence; 'face det' = face detection "
         "confidence. Labels below the confidence threshold show as Uncertain.")

    if not run:
        _finalize()
        _show_last_session()
        return

    if not st.session_state.get("live"):
        session = SessionStats("webcam", classifier.name)
        processor = FrameProcessor(settings, classifier, use_tracking=True, session=session)
        st.session_state.live = {"session": session, "processor": processor}
    live = st.session_state.live
    processor: FrameProcessor = live["processor"]
    session: SessionStats = live["session"]

    kpi_placeholder = st.empty()
    frame_col, side_col = st.columns([3, 1])
    frame_placeholder = frame_col.empty()
    side_placeholder = side_col.empty()

    try:
        with Camera(settings.camera_index, settings.camera_width, settings.camera_height) as cam:
            i = 0
            while True:
                frame = cam.read()
                annotated, results = processor.process(frame)
                frame_placeholder.image(resize_max_width(annotated, 960), channels="BGR",
                                        use_container_width=True)
                if i % KPI_EVERY_N_FRAMES == 0:
                    kpi_row_into(kpi_placeholder, _kpis(processor, session, results))
                    lines = [f"**ID {r.track_id}**: {r.label} ({r.confidence * 100:.0f}%), "
                             f"stability {r.stability:.2f}" for r in results]
                    side_placeholder.markdown("#### Tracked faces\n" + ("\n\n".join(lines) or "_none_"))
                i += 1
                if session.duration_s > settings.max_live_seconds:
                    st.warning(f"Reached the {settings.max_live_seconds}s session limit (see Settings). "
                               "Turn the camera toggle off to save, then start again.")
                    break
                if i % 300 == 0 and not results:
                    log.info("No faces in recent frames (brightness %.0f)", processor.last_brightness)
                time.sleep(0.001)  # yield so Streamlit can deliver the toggle change
    except CameraError as exc:
        st.error(str(exc))
        st.info("Try another camera index on the Settings page, or close apps that use the camera.")
        _finalize()
    except MemoryError as exc:
        st.error(f"Out of memory while processing frames: {exc}")
        _finalize()
