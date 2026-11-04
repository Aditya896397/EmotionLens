"""Live Webcam Detection page.

Two camera sources:
  * Browser camera: st.camera_input captures snapshots in the visitor's browser and sends
    them to the server. This is the only option on a hosted deployment (Streamlit Cloud),
    because the server has no camera. Each snapshot goes through the full pipeline, and
    tracking + smoothing carry over between snapshots of one session.
  * This computer's camera: a real-time OpenCV capture loop, only when the app runs locally.
    Streamlit reruns the whole script when a widget changes. Turning the "Start camera"
    toggle off triggers a rerun that interrupts the loop; the `with Camera` block guarantees
    the device is released, and the next run saves the session.
"""
from __future__ import annotations

import hashlib
import time

import streamlit as st

import config

from analytics import charts
from analytics.dashboard import disclaimer, fmt_duration, fmt_pct, hero, kpi_row_into, note
from analytics.session import SessionStats
from recognition.pipeline import FrameProcessor
from ui.state import get_classifier, get_settings, save_session
from utils.image_utils import decode_image, resize_max_width
from utils.logger import get_logger
from utils.video_utils import Camera, CameraError

log = get_logger("ui.live")

KPI_EVERY_N_FRAMES = 10
SOURCE_BROWSER = "Browser camera (snapshots, works when deployed)"
SOURCE_LOCAL = "This computer's camera (real-time, local run only)"


def _finalize(key: str = "live") -> None:
    live = st.session_state.get(key)
    if not live:
        return
    try:
        if live["session"].frames_seen > 0:
            sid = save_session(live["session"], live["processor"].temporal)
            st.success(f"Session `{sid}` saved: {live['session'].frames_processed} frames analysed.")
    finally:
        live["processor"].close()
        st.session_state[key] = None


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


def _render_browser_camera(classifier, settings) -> None:
    note("Take snapshots with your browser's camera. Each one goes through detection, tracking, "
         "classification and smoothing; track IDs and smoothing carry over between snapshots. "
         "Snapshots are processed in memory and never stored.")
    snap = st.session_state.get("snap")
    if not snap:
        session = SessionStats("webcam", classifier.name)
        processor = FrameProcessor(settings, classifier, use_tracking=True, session=session)
        snap = st.session_state.snap = {"session": session, "processor": processor, "key": None,
                                        "annotated": None, "results": []}

    picture = st.camera_input("Snapshot", label_visibility="collapsed")
    if picture is not None:
        data = picture.getvalue()
        key = hashlib.md5(data).hexdigest()
        if key != snap["key"]:
            frame = decode_image(data)
            if frame is None:
                st.error("The camera image could not be decoded.")
            else:
                snap["annotated"], snap["results"] = snap["processor"].process(frame)
                snap["key"] = key

    session: SessionStats = snap["session"]
    processor: FrameProcessor = snap["processor"]
    kpi_row_into(st.empty(), [
        ("Faces in snapshot", len(snap["results"])),
        ("Faces tracked", len(session.track_ids), "unique IDs this session"),
        ("Snapshots", session.frames_processed),
        ("Pipeline latency", f"{processor.last_pipeline_ms:.0f} ms"),
        ("Avg expression conf.", fmt_pct(session.avg_confidence)),
    ])
    if snap["annotated"] is not None:
        c1, c2 = st.columns([3, 1])
        c1.image(resize_max_width(snap["annotated"], 960), channels="BGR", use_container_width=True)
        lines = [f"**ID {r.track_id}**: {r.label} ({r.confidence * 100:.0f}%)" for r in snap["results"]]
        c2.markdown("#### Faces\n" + ("\n\n".join(lines) or "_none detected_"))

    b1, b2 = st.columns(2)
    if b1.button("Save session", type="primary", disabled=session.frames_processed == 0):
        _finalize("snap")
        st.rerun()
    if b2.button("Discard session", disabled=session.frames_processed == 0):
        snap["processor"].close()
        st.session_state.snap = None
        st.rerun()


def render() -> None:
    hero("Live Webcam Detection", "Multi-face detection, tracking and smoothed expression labels in real time.")
    disclaimer()

    classifier = get_classifier()
    if classifier is None:
        return
    settings = get_settings()

    sources = [SOURCE_BROWSER, SOURCE_LOCAL]
    source = st.radio("Camera source", sources, index=0 if config.IS_CLOUD else 1, horizontal=True,
                      help="On a hosted deployment the server has no camera, so use the browser camera.")
    if source == SOURCE_BROWSER:
        _finalize("live")
        _render_browser_camera(classifier, settings)
        _show_last_session()
        return
    _finalize("snap")
    if config.IS_CLOUD:
        st.warning("This app is running on a server, which has no camera. Use the browser camera.")
        return

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
