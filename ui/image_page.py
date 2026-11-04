"""Image Analysis page."""
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from analytics import charts
from analytics.dashboard import disclaimer, fmt_pct, hero, kpi_row
from analytics.report_generator import build_pdf_report, dataframe_csv
from analytics.session import SessionStats
from recognition.pipeline import FrameProcessor
from ui.state import get_classifier, get_db, get_settings
from utils.image_utils import decode_image, encode_png, resize_max_width

MAX_WIDTH = 1600


def _analyse(data: bytes, classifier, settings) -> dict | None:
    image = decode_image(data)
    if image is None:
        return None
    image = resize_max_width(image, MAX_WIDTH)
    session = SessionStats("image", classifier.name)
    processor = FrameProcessor(settings, classifier, use_tracking=False, session=session)
    try:
        annotated, results = processor.process(image, t=0.0)
    finally:
        processor.close()

    faces = pd.DataFrame([{
        "face": r.track_id,
        "expression": r.label,
        "expression_conf": round(r.confidence, 3),
        "top3": ", ".join(f"{l} {p:.2f}" for l, p in r.top_k(3)),
        "detection_conf": None if r.detection_confidence is None else round(r.detection_confidence, 3),
        "box_xyxy": str(r.bbox),
    } for r in results])
    return {
        "image": image, "annotated": annotated, "results": results, "faces": faces,
        "session": session, "brightness": processor.last_brightness,
        "enhanced": processor.last_enhanced, "backend": processor.detector.backend,
    }


def render() -> None:
    hero("Image Analysis", "Detect every visible face in a photo and classify its expression.")
    disclaimer()
    classifier = get_classifier()
    if classifier is None:
        return
    settings = get_settings()

    upload = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if upload is None:
        st.info("Group photos work too. Each face is numbered left to right.")
        return

    data = upload.getvalue()
    key = hashlib.md5(data).hexdigest() + str(hash(settings.to_json()))
    if st.session_state.get("image_key") != key:
        with st.spinner("Analysing ..."):
            result = _analyse(data, classifier, settings)
        if result is None:
            st.error("This file could not be decoded as an image.")
            return
        st.session_state.image_key = key
        st.session_state.image_result = result
        st.session_state.image_saved = False
    result = st.session_state.image_result
    session: SessionStats = result["session"]

    kpi_row([
        ("Faces detected", len(result["results"])),
        ("Avg expression conf.", fmt_pct(session.avg_confidence)),
        ("Avg detection conf.", fmt_pct(session.avg_detection_confidence)),
        ("Image brightness", f"{result['brightness']:.0f}/255", "low-light boost applied" if result["enhanced"] else ""),
        ("Detector", result["backend"]),
    ])
    st.write("")

    c1, c2 = st.columns(2)
    c1.image(result["image"], channels="BGR", caption="Original", use_container_width=True)
    c2.image(result["annotated"], channels="BGR", caption="Analysis", use_container_width=True)

    if not result["results"]:
        st.warning("No faces detected. Try a larger or front-facing face, better lighting, a lower detection "
                   "confidence on the Settings page, or MediaPipe full-range mode for distant faces.")
        return

    st.subheader("Faces")
    st.dataframe(result["faces"], hide_index=True, use_container_width=True)

    st.subheader("Class probabilities per face")
    cols = st.columns(min(3, len(result["results"])))
    for i, r in enumerate(result["results"]):
        cols[i % len(cols)].plotly_chart(
            charts.probability_bar(r.probabilities, title=f"Face {r.track_id}: {r.label}"),
            use_container_width=True, key=f"prob_{i}")

    st.subheader("Export")
    summary = session.summary()
    counts = session.expression_counts()
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("PDF report", build_pdf_report("Image analysis report", summary, counts,
                                                      faces=result["faces"], image_bgr=result["annotated"]),
                       file_name=f"emotionlens_image_{summary['session_id']}.pdf", mime="application/pdf")
    d2.download_button("Faces CSV", dataframe_csv(result["faces"]),
                       file_name=f"emotionlens_faces_{summary['session_id']}.csv", mime="text/csv")
    d3.download_button("Annotated PNG", encode_png(result["annotated"]),
                       file_name=f"emotionlens_annotated_{summary['session_id']}.png", mime="image/png")
    if d4.button("Save to session history", disabled=st.session_state.get("image_saved", False)):
        get_db().save_session(summary, counts)
        st.session_state.image_saved = True
        st.success("Saved (statistics only, the image is not stored).")
