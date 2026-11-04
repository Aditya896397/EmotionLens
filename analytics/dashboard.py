"""Reusable Streamlit UI components: theme CSS, KPI cards, banners."""
from __future__ import annotations

import html

import streamlit as st

import config

CSS = """
<style>
:root {
  --el-accent: #7C5CFF;
  --el-accent-2: #22D3EE;
  --el-card: #161B26;
  --el-border: rgba(255,255,255,0.08);
  --el-muted: #94A3B8;
}
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #121624 0%, #0E1117 100%); }
section[data-testid="stSidebar"] .stRadio label p { font-size: 0.95rem; }
h1, h2, h3 { letter-spacing: -0.01em; }

.el-hero { padding: 1.2rem 1.4rem; border-radius: 16px; margin-bottom: 1rem;
  background: radial-gradient(1200px 200px at 0% 0%, rgba(124,92,255,0.25), transparent),
              radial-gradient(800px 200px at 100% 0%, rgba(34,211,238,0.15), transparent), var(--el-card);
  border: 1px solid var(--el-border); }
.el-hero h1 { margin: 0; font-size: 1.9rem; }
.el-hero p { margin: .3rem 0 0 0; color: var(--el-muted); }

.el-kpi { background: var(--el-card); border: 1px solid var(--el-border); border-radius: 14px;
  padding: 0.9rem 1rem; height: 100%; position: relative; overflow: hidden; }
.el-kpi::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px;
  background: linear-gradient(180deg, var(--el-accent), var(--el-accent-2)); }
.el-kpi .label { color: var(--el-muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em; }
.el-kpi .value { font-size: 1.6rem; font-weight: 700; margin-top: .15rem; line-height: 1.2; }
.el-kpi .sub { color: var(--el-muted); font-size: 0.78rem; margin-top: .15rem; }

.el-note { border-left: 3px solid var(--el-accent); background: rgba(124,92,255,0.08);
  padding: .6rem .9rem; border-radius: 8px; color: #CBD5E1; font-size: .85rem; margin: .4rem 0 1rem 0; }
.el-pill { display: inline-block; padding: .1rem .55rem; border-radius: 999px; font-size: .75rem;
  border: 1px solid var(--el-border); background: rgba(255,255,255,0.04); margin-right: .3rem; }
.el-ok { color: #34D399; } .el-warn { color: #FBBF24; } .el-bad { color: #F87171; }
@media (max-width: 768px) { .el-kpi .value { font-size: 1.25rem; } .el-hero h1 { font-size: 1.4rem; } }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="el-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value, sub: str = "") -> str:
    return (
        f'<div class="el-kpi"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(str(value))}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div>'
    )


def kpi_row(cards: list[tuple]) -> None:
    """cards: list of (label, value) or (label, value, sub)."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(kpi_card(*card), unsafe_allow_html=True)


def kpi_row_into(placeholder, cards: list[tuple]) -> None:
    """Render a KPI row as one HTML block into an st.empty() placeholder (for live updates)."""
    inner = "".join(f'<div style="flex:1;min-width:140px">{kpi_card(*c)}</div>' for c in cards)
    placeholder.markdown(f'<div style="display:flex;gap:12px;flex-wrap:wrap">{inner}</div>',
                         unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f'<div class="el-note">{html.escape(text)}</div>', unsafe_allow_html=True)


def disclaimer() -> None:
    note(config.DISCLAIMER)


def fmt_pct(value) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def fmt_duration(seconds) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(float(seconds)))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"
