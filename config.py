"""Central configuration for EmotionLens AI.

Every module imports paths, constants and default settings from here, so there is
exactly one place to change a directory, a label order or a default threshold.
Runtime settings edited in the dashboard are persisted to user_settings.json and
loaded on top of the defaults defined in AppSettings.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent

# Streamlit Community Cloud checks repositories out under /mount/src. There the server has no
# camera, so the Live page defaults to capturing from the visitor's browser instead.
IS_CLOUD = str(ROOT_DIR).replace("\\", "/").startswith("/mount/src") or os.environ.get("EMOTIONLENS_CLOUD") == "1"

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_NPZ = PROCESSED_DATA_DIR / "fer2013.npz"
DATASET_REPORT = PROCESSED_DATA_DIR / "dataset_report.json"
# Copy committed to the repository so deployed apps can show dataset statistics without the data.
DATASET_REPORT_PUBLIC = ROOT_DIR / "reports" / "dataset_report.json"

CHECKPOINT_DIR = ROOT_DIR / "models" / "checkpoints"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = REPORTS_DIR / "experiments"
EXPORTS_DIR = REPORTS_DIR / "exports"
LOG_DIR = ROOT_DIR / "logs"
DB_PATH = ROOT_DIR / "database" / "emotionlens.db"
SCHEMA_PATH = ROOT_DIR / "database" / "schema.sql"
USER_SETTINGS_PATH = ROOT_DIR / "user_settings.json"

# ---------------------------------------------------------------------------
# Dataset / labels
# ---------------------------------------------------------------------------
# Index order matches the original FER-2013 `emotion` column (0..6).
EMOTION_LABELS: tuple[str, ...] = (
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
)
UNCERTAIN_LABEL = "Uncertain"
NUM_CLASSES = len(EMOTION_LABELS)
IMG_SIZE = 48

RANDOM_SEED = 42
# Fraction of the training split held out for validation when the dataset
# source has no validation split of its own (the Kaggle image-folder version).
VAL_FRACTION = 0.10
# Images whose pixel standard deviation is below this are treated as blank/corrupt.
LOW_VARIANCE_STD_THRESHOLD = 2.0

DISCLAIMER = (
    "EmotionLens AI classifies visible facial expressions only. A predicted label "
    "is not a measurement of a person's actual emotional or psychological state."
)

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE = 64
CNN_EPOCHS = 60
CNN_LEARNING_RATE = 1e-3
MOBILENET_IMG_SIZE = 96          # FER faces are upsampled to this inside the model
MOBILENET_HEAD_EPOCHS = 10       # stage 1: frozen backbone
MOBILENET_FINETUNE_EPOCHS = 25   # stage 2: top layers unfrozen
MOBILENET_HEAD_LR = 1e-3
MOBILENET_FINETUNE_LR = 1e-4
MOBILENET_TRAINABLE_LAYERS = 40
EARLY_STOPPING_PATIENCE = 8
LR_PLATEAU_PATIENCE = 3


# ---------------------------------------------------------------------------
# Runtime settings (editable on the Settings page)
# ---------------------------------------------------------------------------
@dataclass
class AppSettings:
    # Model
    model_name: str = ""                       # "" = best evaluated model, else checkpoint stem

    # Face detection
    detector_backend: str = "auto"             # auto | mediapipe | haar
    detection_confidence: float = 0.5          # minimum face *detection* confidence
    mediapipe_model_selection: int = 0         # 0 = short range (<2 m), 1 = full range (<5 m)
    detection_input_width: int = 640           # frames wider than this are downscaled for detection
    min_face_size: int = 24                    # pixels; smaller boxes are ignored

    # Preprocessing
    align_faces: bool = True
    face_margin: float = 0.15                  # box expansion before cropping
    low_light_enhance: bool = True
    low_light_threshold: float = 70.0          # mean gray level below which enhancement runs

    # Expression classification
    expression_confidence_threshold: float = 0.40   # below this the label shows as "Uncertain"
    frame_skip: int = 1                        # run inference every (frame_skip + 1) frames

    # Temporal analysis
    smoothing_method: str = "ema"              # ema | window | none
    ema_alpha: float = 0.35
    smoothing_window: int = 8
    change_min_persist: int = 3                # updates a new label must persist to count as a change
    stability_window: int = 15

    # Tracking (ByteTrack)
    track_high_thresh: float = 0.6
    track_low_thresh: float = 0.1
    track_new_thresh: float = 0.6
    track_iou_thresh: float = 0.3
    track_max_lost: int = 30                   # processed frames a lost track is kept for re-identification
    track_min_hits: int = 2

    # Camera / live session
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    max_live_seconds: int = 600

    # Video
    video_max_seconds: int = 300               # only the first N seconds of an upload are processed

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def load_settings(path: Path = USER_SETTINGS_PATH) -> AppSettings:
    """Defaults overridden by user_settings.json (unknown keys are ignored)."""
    settings = AppSettings()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return settings
        valid = {f.name: f.type for f in fields(AppSettings)}
        for key, value in data.items():
            if key in valid:
                default = getattr(settings, key)
                try:
                    setattr(settings, key, type(default)(value))
                except (TypeError, ValueError):
                    continue
    return settings


def save_settings(settings: AppSettings, path: Path = USER_SETTINGS_PATH) -> None:
    path.write_text(settings.to_json(), encoding="utf-8")


def ensure_dirs() -> None:
    """Create every runtime directory the project writes to."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CHECKPOINT_DIR,
        FIGURES_DIR,
        EXPERIMENTS_DIR,
        EXPORTS_DIR,
        LOG_DIR,
        DB_PATH.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
