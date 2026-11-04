"""EmotionLens AI: Streamlit entry point.

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations

import utils.native_preload  # noqa: F401  (must come before streamlit/pandas, see module docstring)

import streamlit as st

st.set_page_config(
    page_title="EmotionLens AI",
    page_icon="🙂",
    layout="wide",
    initial_sidebar_state="expanded",
)

import config  # noqa: E402  (set_page_config must be the first Streamlit call)
from analytics.dashboard import inject_css  # noqa: E402
from ui import (  # noqa: E402
    analytics_page,
    history,
    home,
    image_page,
    live,
    model_page,
    settings_page,
    video_page,
)
from utils.logger import get_logger  # noqa: E402

log = get_logger("app")
config.ensure_dirs()
inject_css()

LOGO_SVG = (
    '<svg width="44" height="44" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#7C5CFF"/><stop offset="1" stop-color="#22D3EE"/></linearGradient></defs>'
    '<rect x="2" y="2" width="44" height="44" rx="12" fill="url(#g)"/>'
    '<circle cx="24" cy="24" r="12" fill="none" stroke="white" stroke-width="2.5"/>'
    '<circle cx="19.5" cy="21" r="1.8" fill="white"/><circle cx="28.5" cy="21" r="1.8" fill="white"/>'
    '<path d="M18.5 27.5 Q24 32 29.5 27.5" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round"/>'
    '<path d="M6 12 V6 H12 M36 6 H42 V12 M42 36 V42 H36 M12 42 H6 V36" fill="none" stroke="white" '
    'stroke-width="2" stroke-linecap="round"/></svg>'
)

PAGES = {
    "Home Dashboard": home.render,
    "Live Webcam Detection": live.render,
    "Image Analysis": image_page.render,
    "Video Analysis": video_page.render,
    "Expression Analytics": analytics_page.render,
    "Model Performance": model_page.render,
    "Session History": history.render,
    "Settings": settings_page.render,
}

with st.sidebar:
    st.markdown(LOGO_SVG + "<h3 style='margin:.4rem 0 0 0'>EmotionLens AI</h3>", unsafe_allow_html=True)
    st.caption("Facial expression recognition & analytics")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Predictions describe visible facial expressions, not inner emotional states. "
               "No face images are stored.")

# Leaving the webcam page with a session open: stop the camera and save the session.
if page != "Live Webcam Detection":
    if st.session_state.get("live"):
        st.session_state.live_run = False
        live._finalize("live")
    if st.session_state.get("snap"):
        live._finalize("snap")

try:
    PAGES[page]()
except Exception as exc:  # last-resort guard so one page error does not blank the app
    log.exception("Unhandled error on page %s", page)
    st.error(f"Something went wrong on this page: {exc}")
    st.caption("Details were written to logs/emotionlens.log")
