"""Plotly figures for the dashboard. Every function takes plain dicts/DataFrames and
returns a go.Figure, so charts are reusable for live sessions, stored sessions and reports."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config

EMOTION_COLORS = {
    "Angry": "#EF4444",
    "Disgust": "#22C55E",
    "Fear": "#A855F7",
    "Happy": "#FACC15",
    "Sad": "#3B82F6",
    "Surprise": "#F97316",
    "Neutral": "#94A3B8",
    config.UNCERTAIN_LABEL: "#6B7280",
}
ORDER = [*config.EMOTION_LABELS, config.UNCERTAIN_LABEL]
ACCENT = "#7C5CFF"


def _style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        title=dict(text=title or "", x=0.01, font=dict(size=15)),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, x=0),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)")
    return fig


def empty_figure(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#94A3B8"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, height)


def distribution_donut(counts: dict[str, int], title: str = "Expression distribution") -> go.Figure:
    items = [(k, v) for k, v in counts.items() if v > 0]
    if not items:
        return empty_figure("No predictions yet")
    labels, values = zip(*items)
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.6, sort=False,
        marker=dict(colors=[EMOTION_COLORS.get(l, ACCENT) for l in labels]),
        textinfo="percent", hovertemplate="%{label}: %{value} predictions (%{percent})<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{sum(values)}</b><br>predictions", showarrow=False, font=dict(size=14))
    return _style(fig, 360, title)


def distribution_bar(counts: dict[str, int], title: str = "Prediction counts") -> go.Figure:
    df = pd.DataFrame({"expression": list(counts.keys()), "count": list(counts.values())})
    if df["count"].sum() == 0:
        return empty_figure("No predictions yet")
    fig = px.bar(df, x="expression", y="count", color="expression", color_discrete_map=EMOTION_COLORS,
                 category_orders={"expression": ORDER}, text="count")
    fig.update_layout(showlegend=False)
    fig.update_traces(textposition="outside", cliponaxis=False)
    return _style(fig, 340, title)


def timeline_area(df: pd.DataFrame, time_col: str = "t", bin_seconds: float = 2.0,
                  title: str = "Expression frequency over time") -> go.Figure:
    """Stacked area of prediction counts per expression per time bin."""
    if df.empty:
        return empty_figure("No timeline data")
    data = df.copy()
    data["bin"] = (np.floor(data[time_col] / bin_seconds) * bin_seconds).round(2)
    pivot = data.groupby(["bin", "expression"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for label in ORDER:
        if label in pivot.columns:
            fig.add_trace(go.Scatter(
                x=pivot.index, y=pivot[label], name=label, mode="lines", stackgroup="one",
                line=dict(width=0.5, color=EMOTION_COLORS[label]),
                hovertemplate=f"{label}: %{{y}}<extra>t=%{{x}}s</extra>",
            ))
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title="predictions")
    return _style(fig, 360, title)


def track_history(df: pd.DataFrame, time_col: str = "t",
                  title: str = "Per-track expression history") -> go.Figure:
    """One row per track; each point is a prediction coloured by its smoothed label."""
    if df.empty:
        return empty_figure("No tracked faces")
    data = df.copy()
    data["track"] = "Track " + data["track_id"].astype(str)
    fig = px.scatter(
        data, x=time_col, y="track", color="expression", color_discrete_map=EMOTION_COLORS,
        category_orders={"expression": ORDER}, hover_data={"confidence": ":.2f"},
    )
    fig.update_traces(marker=dict(size=9, symbol="square", line=dict(width=0)))
    fig.update_xaxes(title="time (s)")
    fig.update_yaxes(title="", categoryorder="category descending")
    return _style(fig, max(260, 90 + 40 * data["track"].nunique()), title)


def confidence_histogram(df: pd.DataFrame, column: str = "confidence",
                         title: str = "Expression confidence distribution") -> go.Figure:
    if df.empty:
        return empty_figure("No predictions")
    fig = px.histogram(df, x=column, color="expression", nbins=20, color_discrete_map=EMOTION_COLORS,
                       category_orders={"expression": ORDER}, barmode="stack")
    fig.update_xaxes(title="softmax confidence", range=[0, 1])
    return _style(fig, 320, title)


def stability_bar(tracks: pd.DataFrame, title: str = "Prediction stability per track") -> go.Figure:
    if tracks.empty:
        return empty_figure("No tracks")
    data = tracks.copy()
    data["track"] = "Track " + data["track_id"].astype(str)
    fig = px.bar(data, x="track", y="stability", color="dominant_expression",
                 color_discrete_map=EMOTION_COLORS, hover_data=["predictions", "changes"])
    fig.update_yaxes(range=[0, 1], title="agreement with stable label")
    return _style(fig, 320, title)


def probability_bar(probabilities: np.ndarray, title: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(
        x=list(probabilities), y=list(config.EMOTION_LABELS), orientation="h",
        marker_color=[EMOTION_COLORS[l] for l in config.EMOTION_LABELS],
        text=[f"{p * 100:.1f}%" for p in probabilities], textposition="outside", cliponaxis=False,
    ))
    fig.update_xaxes(range=[0, 1.1], visible=False)
    fig.update_yaxes(autorange="reversed")
    return _style(fig, 260, title or None)


def sessions_bar(sessions: pd.DataFrame, title: str = "Faces per session") -> go.Figure:
    if sessions.empty:
        return empty_figure("No saved sessions")
    data = sessions.sort_values("started_at")
    fig = px.bar(data, x="started_at", y="total_faces", color="source",
                 color_discrete_sequence=[ACCENT, "#22D3EE", "#F472B6"],
                 hover_data=["session_id", "dominant_expression", "duration_s"])
    fig.update_xaxes(title="")
    return _style(fig, 320, title)


def model_comparison_bar(df: pd.DataFrame, metrics=("accuracy", "precision_macro", "recall_macro", "f1_macro"),
                         title: str = "Model comparison (test set)") -> go.Figure:
    if df.empty:
        return empty_figure("No evaluated models")
    long = df.melt(id_vars="model", value_vars=[m for m in metrics if m in df.columns],
                   var_name="metric", value_name="score")
    fig = px.bar(long, x="metric", y="score", color="model", barmode="group", text_auto=".3f",
                 color_discrete_sequence=[ACCENT, "#22D3EE", "#F472B6", "#FACC15", "#34D399"])
    fig.update_yaxes(range=[0, 1])
    return _style(fig, 360, title)


def latency_bar(df: pd.DataFrame, title: str = "Median inference latency (ms)") -> go.Figure:
    cols = [c for c in ("latency_1face_ms", "latency_8faces_ms") if c in df.columns]
    if df.empty or not cols:
        return empty_figure("No latency data")
    long = df.melt(id_vars="model", value_vars=cols, var_name="batch", value_name="ms")
    long["batch"] = long["batch"].map({"latency_1face_ms": "1 face", "latency_8faces_ms": "8 faces"})
    fig = px.bar(long, x="model", y="ms", color="batch", barmode="group", text_auto=".1f",
                 color_discrete_sequence=[ACCENT, "#22D3EE"])
    return _style(fig, 360, title)


def confusion_heatmap(cm: list[list[int]], labels: list[str], normalize: bool = True,
                      title: str = "Confusion matrix") -> go.Figure:
    m = np.array(cm, dtype=float)
    z = m / np.maximum(m.sum(axis=1, keepdims=True), 1) if normalize else m
    text = [[f"{z[i, j]:.2f}<br>({int(m[i, j])})" for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, text=text, texttemplate="%{text}", colorscale="Purples",
        zmin=0, zmax=1 if normalize else None, hovertemplate="true %{y} / pred %{x}: %{z:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="True", autorange="reversed")
    return _style(fig, 460, title)


def training_curves(history: pd.DataFrame, title: str = "Training curves") -> go.Figure:
    fig = go.Figure()
    for col, dash in (("accuracy", "solid"), ("val_accuracy", "dash"), ("loss", "solid"), ("val_loss", "dash")):
        if col in history.columns:
            fig.add_trace(go.Scatter(
                x=history["epoch"], y=history[col], name=col, line=dict(dash=dash),
                yaxis="y2" if "loss" in col else "y",
            ))
    fig.update_layout(
        yaxis=dict(title="accuracy", range=[0, 1]),
        yaxis2=dict(title="loss", overlaying="y", side="right", showgrid=False),
    )
    fig.update_xaxes(title="epoch")
    return _style(fig, 360, title)


def line_chart(df: pd.DataFrame, x: str, y: str, color: str, title: str, y_title: str = "") -> go.Figure:
    if df.empty:
        return empty_figure("No data")
    fig = px.line(df, x=x, y=y, color=color, markers=True,
                  color_discrete_sequence=[ACCENT, "#22D3EE", "#F472B6", "#FACC15", "#34D399"])
    fig.update_yaxes(title=y_title or y)
    return _style(fig, 340, title)


def per_class_f1(results: list[dict], title: str = "Per-class F1 vs. training support") -> go.Figure:
    rows = []
    for r in results:
        for label in config.EMOTION_LABELS:
            pc = r["per_class"][label]
            rows.append({"model": r["name"], "expression": label, "f1": pc["f1-score"], "support": pc["support"]})
    if not rows:
        return empty_figure("No evaluated models")
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="expression", y="f1", color="model", barmode="group",
                 color_discrete_sequence=[ACCENT, "#22D3EE", "#F472B6", "#FACC15", "#34D399"],
                 category_orders={"expression": list(config.EMOTION_LABELS)}, hover_data=["support"])
    fig.update_yaxes(range=[0, 1])
    return _style(fig, 360, title)
