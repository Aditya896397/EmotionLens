"""Settings page: every runtime parameter, persisted to user_settings.json."""
from __future__ import annotations

import streamlit as st

import config
from analytics.dashboard import hero, note
from recognition.emotion_classifier import best_model_name, list_models
from ui.state import get_settings
from utils.device import configure_tensorflow


def render() -> None:
    hero("Settings", "Detection, classification, tracking, smoothing and camera parameters.")
    note("Changes apply to the next analysis or camera session. A running webcam session keeps "
         "the settings it started with.")
    s = get_settings()
    models = list_models()

    with st.form("settings_form"):
        st.subheader("Expression model")
        model_options = ["(best evaluated)"] + models
        current = s.model_name if s.model_name in models else "(best evaluated)"
        model_choice = st.selectbox("Model checkpoint", model_options, index=model_options.index(current),
                                    help=f"Best evaluated right now: {best_model_name() or 'none'}")
        threshold = st.slider("Expression confidence threshold", 0.0, 0.9, s.expression_confidence_threshold, 0.05,
                              help="Smoothed softmax probability below this is shown as 'Uncertain'.")
        frame_skip = st.slider("Frame skip", 0, 5, s.frame_skip,
                               help="Run detection + classification every (skip + 1) frames.")

        st.subheader("Face detection")
        c1, c2, c3 = st.columns(3)
        backend = c1.selectbox("Detector backend", ["auto", "mediapipe", "haar"],
                               index=["auto", "mediapipe", "haar"].index(s.detector_backend))
        det_conf = c2.slider("Detection confidence", 0.1, 0.95, s.detection_confidence, 0.05,
                             help="Minimum face-detector score (not the expression confidence).")
        mp_range = c3.selectbox("MediaPipe range", [0, 1], index=s.mediapipe_model_selection,
                                format_func=lambda v: "Short (< 2 m)" if v == 0 else "Full (< 5 m)")
        c4, c5 = st.columns(2)
        widths = [320, 480, 640, 800, 960, 1280]
        det_width = c4.select_slider("Detection input width", widths,
                                     value=s.detection_input_width if s.detection_input_width in widths else 640)
        min_face = c5.slider("Minimum face size (px)", 12, 120, s.min_face_size)

        st.subheader("Preprocessing")
        c6, c7, c8, c9 = st.columns(4)
        align = c6.checkbox("Align faces (level eyes)", s.align_faces)
        low_light = c7.checkbox("Low-light enhancement", s.low_light_enhance)
        low_thr = c8.slider("Low-light threshold", 20.0, 140.0, s.low_light_threshold, 5.0)
        margin = c9.slider("Face crop margin", 0.0, 0.4, s.face_margin, 0.05)

        st.subheader("Temporal smoothing")
        c10, c11, c12, c13 = st.columns(4)
        method = c10.selectbox("Method", ["ema", "window", "none"],
                               index=["ema", "window", "none"].index(s.smoothing_method))
        alpha = c11.slider("EMA alpha", 0.05, 1.0, s.ema_alpha, 0.05, help="Higher = reacts faster, smooths less.")
        window = c12.slider("Window size", 1, 30, s.smoothing_window)
        persist = c13.slider("Change persistence", 1, 15, s.change_min_persist,
                             help="Updates a new label must persist before an expression change is recorded.")

        st.subheader("Tracking (ByteTrack)")
        c14, c15, c16, c17 = st.columns(4)
        high = c14.slider("High score threshold", 0.3, 0.95, s.track_high_thresh, 0.05)
        iou = c15.slider("IoU match threshold", 0.1, 0.8, s.track_iou_thresh, 0.05)
        max_lost = c16.slider("Keep lost tracks (frames)", 1, 120, s.track_max_lost)
        min_hits = c17.slider("Frames to confirm a track", 1, 5, s.track_min_hits)

        st.subheader("Camera and limits")
        c18, c19, c20, c21 = st.columns(4)
        cam = c18.number_input("Camera index", 0, 10, s.camera_index)
        res = c19.selectbox("Resolution", ["640x480", "1280x720"],
                            index=0 if s.camera_width <= 640 else 1)
        max_live = c20.number_input("Max live session (s)", 30, 3600, s.max_live_seconds, 30)
        max_video = c21.number_input("Default video limit (s)", 5, 600, s.video_max_seconds, 5)

        saved = st.form_submit_button("Save settings", type="primary")

    if saved:
        s.model_name = "" if model_choice == "(best evaluated)" else model_choice
        s.expression_confidence_threshold = float(threshold)
        s.frame_skip = int(frame_skip)
        s.detector_backend = backend
        s.detection_confidence = float(det_conf)
        s.mediapipe_model_selection = int(mp_range)
        s.detection_input_width = int(det_width)
        s.min_face_size = int(min_face)
        s.align_faces = bool(align)
        s.low_light_enhance = bool(low_light)
        s.low_light_threshold = float(low_thr)
        s.face_margin = float(margin)
        s.smoothing_method = method
        s.ema_alpha = float(alpha)
        s.smoothing_window = int(window)
        s.change_min_persist = int(persist)
        s.track_high_thresh = float(high)
        s.track_new_thresh = float(high)
        s.track_iou_thresh = float(iou)
        s.track_max_lost = int(max_lost)
        s.track_min_hits = int(min_hits)
        s.camera_index = int(cam)
        s.camera_width, s.camera_height = (int(v) for v in res.split("x"))
        s.max_live_seconds = int(max_live)
        s.video_max_seconds = int(max_video)
        config.save_settings(s)
        st.success("Settings saved to user_settings.json")

    if st.button("Reset to defaults"):
        st.session_state.settings = config.AppSettings()
        config.save_settings(st.session_state.settings)
        st.rerun()

    st.subheader("System")
    d = configure_tensorflow()
    st.json({
        "device": d.device, "gpus": d.gpu_names, "onnxruntime": d.onnxruntime_version,
        "onnx_providers": d.onnx_providers, "tensorflow": d.tensorflow_version, "cpu": d.cpu_name,
        "cores": f"{d.physical_cores} physical / {d.logical_cores} logical", "ram_gb": d.ram_gb, "os": d.os_name,
        "checkpoints": models, "database": str(config.DB_PATH),
        "privacy": "No face images or embeddings are stored; only aggregate statistics.",
    })
