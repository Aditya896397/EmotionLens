"""Generate notebooks/train_colab.ipynb (kept as code so the notebook is reviewable in diffs).

    python -m scripts.make_notebook
"""
from __future__ import annotations

import json

import config

CELLS: list[tuple[str, str]] = [
    ("markdown", """# EmotionLens AI: training on Google Colab

This notebook trains the expression models on a free Colab GPU, evaluates them, and exports them
to **ONNX** so the laptop app can run them with ONNX Runtime (no TensorFlow needed on the laptop).

**Before you start:** *Runtime → Change runtime type → T4 GPU*, then run the cells top to bottom.

| Step | Time on a T4 (approx.) |
|---|---|
| Install + dataset | ~5 min |
| Custom CNN | ~10–15 min |
| MobileNetV2 | ~20–30 min |
| Ablations (optional) | ~25 min |

At the end, a zip with the trained models and measured results downloads. Unzip it into the
project folder on your laptop."""),
    ("code", """# 1. Check the GPU
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "No GPU: Runtime > Change runtime type > T4 GPU\""""),
    ("code", """# 2. Get the project code: either clone from GitHub or upload the project zip
REPO_URL = "https://github.com/Aditya896397/EmotionLens.git"   # set to "" to upload a project zip instead

import os, zipfile, shutil
os.chdir("/content")
shutil.rmtree("/content/EmotionLens-AI", ignore_errors=True)
if REPO_URL:
    !git clone -q {REPO_URL} /content/EmotionLens-AI
else:
    from google.colab import files
    uploaded = files.upload()          # choose EmotionLens-AI-colab.zip (git archive output)
    zipfile.ZipFile(next(iter(uploaded))).extractall("/content")
%cd /content/EmotionLens-AI
!ls"""),
    ("code", """# 3. Install the training stack, then restart the session automatically (numpy is downgraded).
#    Colab shows "Your session crashed": this is expected. Continue with the next cell.
!pip install -q -r requirements-train.txt
import os; os.kill(os.getpid(), 9)"""),
    ("code", """# 4. After the restart: verify versions and the GPU
%cd /content/EmotionLens-AI
import tensorflow as tf, numpy as np, keras, tf2onnx, onnxruntime
print("TF", tf.__version__, "| Keras", keras.__version__, "| numpy", np.__version__,
      "| tf2onnx", tf2onnx.__version__, "| onnxruntime", onnxruntime.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))"""),
    ("code", """# 5. Download FER-2013 from Kaggle and prepare it (cleaning, splits, class weights)
!mkdir -p data/raw/fer2013
!curl -sSL -o /content/fer2013.zip https://www.kaggle.com/api/v1/datasets/download/msambare/fer2013
!unzip -q -o /content/fer2013.zip -d data/raw/fer2013
!python -m scripts.prepare_dataset"""),
    ("code", """# 6. Train the custom CNN
!python -m models.train --model cnn"""),
    ("code", """# 7. Train MobileNetV2 (transfer learning, two stages)
!python -m models.train --model mobilenet"""),
    ("code", """# 8. (Optional) Ablations: effect of augmentation and class weights
RUN_ABLATIONS = True
if RUN_ABLATIONS:
    !python -m models.train --model cnn --no-augment --name cnn_noaug
    !python -m models.train --model cnn --no-class-weights --name cnn_noweights"""),
    ("code", """# 9. Evaluate with Keras on the GPU (accuracy, P/R/F1, confusion matrix)
#    Latency is re-measured on the laptop CPU later, with ONNX Runtime.
!python -m models.evaluate --backend keras"""),
    ("code", """# 10. Export every model to ONNX and check that it matches Keras
!python -m models.export_onnx
!ls -la models/checkpoints
import json, glob
for f in sorted(glob.glob("models/checkpoints/*_meta.json")):
    m = json.load(open(f))
    print(f"{m['name']:15s} val_acc={m.get('best_val_accuracy')}  onnx={m.get('onnx_file')}  "
          f"max_diff={m.get('max_abs_prob_diff')}  agreement={m.get('label_agreement')}")"""),
    ("code", """# 11. Download the results (models + measured reports) as one zip
import shutil, os
os.makedirs("/content/results/models", exist_ok=True)
shutil.copytree("models/checkpoints", "/content/results/models/checkpoints", dirs_exist_ok=True)
shutil.copytree("reports", "/content/results/reports", dirs_exist_ok=True)
shutil.make_archive("/content/emotionlens_results", "zip", "/content/results")
from google.colab import files
files.download("/content/emotionlens_results.zip")"""),
    ("markdown", """## On the laptop

Unzip `emotionlens_results.zip` **into the project folder** (it contains `models/` and `reports/`).
Then, in the project's virtual environment:

```
python -m models.evaluate              # same accuracy, latency measured on your CPU with ONNX Runtime
python -m scripts.run_experiments all  # lighting, preprocessing, smoothing, multi-face, ablation
streamlit run app.py
```"""),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def main() -> None:
    out = config.ROOT_DIR / "notebooks" / "train_colab.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
