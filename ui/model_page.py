"""Model Performance page: measured evaluation results and research experiments."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

import config
from analytics import charts
from analytics.dashboard import hero, note
from ui.state import load_eval_results

EXPERIMENT_FILES = {
    "lighting": ("Robustness to lighting", "lighting.csv"),
    "multiface": ("Multiple faces: detection and latency", "multiface.csv"),
    "smoothing": ("Effect of temporal smoothing", "smoothing.csv"),
    "preprocessing": ("Preprocessing ablation", "preprocessing.csv"),
    "ablation": ("Training ablation", "ablation.csv"),
}


def _dataset_section() -> None:
    if not config.DATASET_REPORT.exists():
        return
    report = json.loads(config.DATASET_REPORT.read_text(encoding="utf-8"))
    st.subheader("Dataset and class imbalance")
    rows = [{"split": s, "expression": k, "images": v}
            for s, d in report["splits"].items() for k, v in d["per_class"].items()]
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="expression", columns="split", values="images").reindex(config.EMOTION_LABELS)
    pivot["class_weight"] = [report["class_weights_balanced"][l] for l in pivot.index]
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(charts.distribution_bar(report["splits"]["train"]["per_class"],
                                            title="Training images per class"), use_container_width=True)
    c2.dataframe(pivot, use_container_width=True)
    c2.caption(f"Imbalance ratio (max/min): {report['train_imbalance_ratio_max_over_min']} | "
               f"cleaning: {report['cleaning_train_only']}")


def _experiments_section() -> None:
    found = [(k, t, config.EXPERIMENTS_DIR / f) for k, (t, f) in EXPERIMENT_FILES.items()
             if (config.EXPERIMENTS_DIR / f).exists()]
    st.subheader("Research experiments")
    if not found:
        st.info("No experiment results yet. Run: `python -m scripts.run_experiments all`")
        return
    for key, title, path in found:
        with st.expander(title, expanded=key == "lighting"):
            df = pd.read_csv(path)
            st.dataframe(df, hide_index=True, use_container_width=True)
            fig_path = config.FIGURES_DIR / f"experiment_{key}.png"
            if fig_path.exists():
                st.image(str(fig_path), use_container_width=True)
            md = config.EXPERIMENTS_DIR / f"{key}_notes.md"
            if md.exists():
                st.markdown(md.read_text(encoding="utf-8"))


def render() -> None:
    hero("Model Performance", "Measured test-set metrics, confusion matrices, latency and experiments.")
    note("All numbers on this page are read from files produced by models/evaluate.py and "
         "scripts/run_experiments.py on this machine. Nothing is hard-coded.")

    results = load_eval_results()
    if not results:
        st.warning("No evaluation results found.")
        st.markdown("Train the models with **notebooks/train_colab.ipynb** on Google Colab, unzip the "
                    "downloaded results into the project folder, then measure them on this machine:")
        st.code("python -m models.evaluate
python -m scripts.run_experiments all", language="bash")
        _dataset_section()
        return

    comparison_path = config.EXPERIMENTS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comp = pd.read_csv(comparison_path)
        st.subheader("Model comparison")
        st.dataframe(comp, hide_index=True, use_container_width=True)
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(charts.model_comparison_bar(comp), use_container_width=True)
        c2.plotly_chart(charts.latency_bar(comp), use_container_width=True)

    st.plotly_chart(charts.per_class_f1(results), use_container_width=True)

    st.subheader("Per-model details")
    tabs = st.tabs([r["name"] for r in results])
    for tab, r in zip(tabs, results):
        with tab:
            normalize = st.toggle("Row-normalise", value=True, key=f"norm_{r['name']}")
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(charts.confusion_heatmap(r["confusion_matrix"], r["labels"], normalize,
                                                     title=f"Confusion matrix - {r['name']}"),
                            use_container_width=True)
            per_class = pd.DataFrame(r["per_class"]).T.round(3)
            c2.dataframe(per_class, use_container_width=True)
            c2.json({k: r.get(k) for k in ("accuracy", "f1_macro", "parameters", "model_file_mb",
                                           "training_seconds", "epochs_run", "augmentation",
                                           "class_weights", "test_set_throughput_img_per_s", "device")})
            hist = config.CHECKPOINT_DIR / f"{r['name']}_history.csv"
            if hist.exists():
                st.plotly_chart(charts.training_curves(pd.read_csv(hist), f"Training curves - {r['name']}"),
                                use_container_width=True)

    _dataset_section()
    _experiments_section()
