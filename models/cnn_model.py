"""Model definitions: a custom CNN and a MobileNetV2 transfer-learning model.

Both models share the same contract so the rest of the system can swap them freely:
    input : float32 tensor (batch, 48, 48, 1) with raw pixel values in [0, 255]
    output: softmax probabilities (batch, 7) in config.EMOTION_LABELS order

Normalisation, data augmentation and (for MobileNetV2) resizing + gray->RGB happen
*inside* the model. Augmentation layers are only active when training=True, so the
saved model can be used for inference without any extra preprocessing code.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from tensorflow import keras
from tensorflow.keras import layers, regularizers

import config

INPUT_SHAPE = (config.IMG_SIZE, config.IMG_SIZE, 1)


def augmentation_block(seed: int = config.RANDOM_SEED) -> keras.Sequential:
    """Geometric + photometric augmentation that keeps expressions recognisable.

    Horizontal flips are safe (expressions are roughly symmetric); vertical flips are not.
    Rotation/zoom/shift ranges mimic imperfect face crops from a detector.
    """
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.06, fill_mode="nearest", seed=seed),   # about +-22 degrees
            layers.RandomZoom(0.10, fill_mode="nearest", seed=seed),
            layers.RandomTranslation(0.08, 0.08, fill_mode="nearest", seed=seed),
            layers.RandomContrast(0.25, seed=seed),
            layers.RandomBrightness(0.2, value_range=(0.0, 255.0), seed=seed),
        ],
        name="augmentation",
    )


def _conv_block(x, filters: int, dropout: float, weight_decay: float, name: str):
    for i in (1, 2):
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False,
            kernel_regularizer=regularizers.l2(weight_decay), name=f"{name}_conv{i}",
        )(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i}")(x)
        x = layers.Activation("relu", name=f"{name}_relu{i}")(x)
    x = layers.MaxPooling2D(2, name=f"{name}_pool")(x)
    return layers.Dropout(dropout, name=f"{name}_drop")(x)


def build_custom_cnn(augment: bool = True, weight_decay: float = 1e-4) -> keras.Model:
    """VGG-style CNN: 4 blocks of (Conv-BN-ReLU) x2 + MaxPool + Dropout.

    Spatial size 48 -> 24 -> 12 -> 6 -> 3, channels 64 -> 128 -> 256 -> 256.
    BatchNorm stabilises training; Dropout + L2 fight over-fitting on a small,
    noisy dataset. About 2.9M parameters, fast enough for real-time CPU inference.
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="face_48x48_gray")
    x = augmentation_block()(inputs) if augment else inputs
    x = layers.Rescaling(1.0 / 255.0, name="normalize")(x)

    x = _conv_block(x, 64, 0.25, weight_decay, "block1")
    x = _conv_block(x, 128, 0.25, weight_decay, "block2")
    x = _conv_block(x, 256, 0.30, weight_decay, "block3")
    x = _conv_block(x, 256, 0.35, weight_decay, "block4")

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(weight_decay), name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_relu")(x)
    x = layers.Dropout(0.5, name="fc1_drop")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="expression")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def build_mobilenetv2(
    img_size: int = config.MOBILENET_IMG_SIZE, augment: bool = True, weights: str | None = "imagenet"
) -> tuple[keras.Model, keras.Model]:
    """MobileNetV2 (ImageNet weights) with a new classification head.

    Returns (full_model, backbone) so the trainer can freeze/unfreeze the backbone.
    The 48x48 grayscale face is upsampled to img_size and replicated to 3 channels
    because the ImageNet backbone expects RGB input at >= 32 px (96 is a standard size).
    """
    inputs = keras.Input(shape=INPUT_SHAPE, name="face_48x48_gray")
    x = augmentation_block()(inputs) if augment else inputs
    x = layers.Resizing(img_size, img_size, interpolation="bilinear", name="upsample")(x)
    x = layers.Concatenate(axis=-1, name="gray_to_rgb")([x, x, x])
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)  # [-1, 1]

    backbone = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3), include_top=False, weights=weights, alpha=1.0
    )
    backbone.trainable = False
    # training=False keeps BatchNorm statistics frozen even when layers are unfrozen later.
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="head_drop1")(x)
    x = layers.Dense(256, activation="relu", name="head_fc")(x)
    x = layers.Dropout(0.4, name="head_drop2")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="expression")(x)
    return keras.Model(inputs, outputs, name="mobilenetv2_transfer"), backbone


def unfreeze_top_layers(backbone: keras.Model, n_layers: int) -> int:
    """Unfreeze the last n layers of the backbone except BatchNorm. Returns count unfrozen."""
    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - n_layers)
    unfrozen = 0
    for i, layer in enumerate(backbone.layers):
        trainable = i >= cutoff and not isinstance(layer, layers.BatchNormalization)
        layer.trainable = trainable
        unfrozen += int(trainable)
    return unfrozen
