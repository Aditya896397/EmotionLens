"""Downloadable reports: PDF (fpdf2) and CSV/JSON exports.

Charts inside the PDF are rendered with Matplotlib (Plotly static export would need
the heavy Kaleido dependency). fpdf2's built-in fonts are Latin-1 only, so all text
is sanitised before it is written.
"""
from __future__ import annotations

import io
import json
from datetime import datetime

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.enums import XPos, YPos

import config

MPL_COLORS = {
    "Angry": "#EF4444", "Disgust": "#22C55E", "Fear": "#A855F7", "Happy": "#FACC15",
    "Sad": "#3B82F6", "Surprise": "#F97316", "Neutral": "#94A3B8",
}


def _latin1(text) -> str:
    return str(text).encode("latin-1", "replace").decode("latin-1")


def _fmt(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _counts_chart_png(counts: dict[str, int]) -> bytes:
    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    bars = ax.bar(labels, values, color=[MPL_COLORS.get(l, "#7C5CFF") for l in labels])
    ax.bar_label(bars, fontsize=8)
    ax.set_ylabel("predictions")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


class _Report(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(124, 92, 255)
        self.cell(0, 8, "EmotionLens AI", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}  |  Generated {datetime.now():%Y-%m-%d %H:%M}", align="C")


def _heading(pdf: FPDF, text: str) -> None:
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, _latin1(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)


def _table(pdf: FPDF, df: pd.DataFrame, max_rows: int = 40) -> None:
    if df.empty:
        pdf.cell(0, 6, "No data.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        return
    pdf.set_font("Helvetica", "", 8)
    with pdf.table(text_align="LEFT", line_height=5) as table:
        header = table.row()
        for col in df.columns:
            header.cell(_latin1(col))
        for _, r in df.head(max_rows).iterrows():
            row = table.row()
            for v in r.tolist():
                row.cell(_latin1(_fmt(v)))
    if len(df) > max_rows:
        pdf.cell(0, 5, f"... {len(df) - max_rows} more rows in the CSV export.", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)


def build_pdf_report(
    title: str,
    summary: dict,
    counts: dict[str, int],
    faces: pd.DataFrame | None = None,
    tracks: pd.DataFrame | None = None,
    image_bgr: np.ndarray | None = None,
    changes: list[dict] | None = None,
) -> bytes:
    pdf = _Report(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _latin1(title), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 4, _latin1(config.DISCLAIMER + " Detection confidence = likelihood the box contains "
                                 "a face; expression confidence = classifier softmax probability."))
    pdf.set_text_color(0, 0, 0)

    _heading(pdf, "Session summary")
    summary_df = pd.DataFrame({"field": list(summary.keys()), "value": [_fmt(v) for v in summary.values()]})
    _table(pdf, summary_df)

    if image_bgr is not None:
        _heading(pdf, "Annotated image")
        ok, png = cv2.imencode(".png", image_bgr)
        if ok:
            h, w = image_bgr.shape[:2]
            width = min(180.0, 130.0 * w / h)  # cap height at ~130 mm for portrait images
            pdf.image(io.BytesIO(png.tobytes()), w=width)

    _heading(pdf, "Expression prediction counts")
    if sum(counts.values()) > 0:
        pdf.image(io.BytesIO(_counts_chart_png(counts)), w=170)
    _table(pdf, pd.DataFrame({"expression": list(counts.keys()), "predictions": list(counts.values())}))

    if faces is not None and not faces.empty:
        _heading(pdf, "Detected faces")
        _table(pdf, faces)
    if tracks is not None and not tracks.empty:
        _heading(pdf, "Tracked faces")
        cols = [c for c in ("track_id", "first_seen_s", "last_seen_s", "predictions",
                            "dominant_expression", "stability", "changes") if c in tracks.columns]
        _table(pdf, tracks[cols])
    if changes:
        _heading(pdf, "Detected expression changes (smoothed)")
        _table(pdf, pd.DataFrame(changes))

    return bytes(pdf.output())


def summary_json(summary: dict, counts: dict[str, int], tracks: list[dict] | None = None) -> bytes:
    payload = {"disclaimer": config.DISCLAIMER, "summary": summary, "expression_counts": counts,
               "tracks": tracks or []}
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")
