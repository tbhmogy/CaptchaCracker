"""모델 정의 및 관련 헬퍼."""
from __future__ import annotations

from typing import List

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

from .config import Settings


def _build_augment_layer(cfg: Settings) -> tf.keras.Sequential | None:
    rotation = abs(cfg.augment_rotation)
    if rotation <= 0:
        return None
    factor = min(rotation / 90.0, 0.5)
    augmentation_layers = [
        layers.RandomRotation(factor=(-factor, factor), fill_mode="constant", fill_value=0.0),
        layers.RandomZoom((-0.05, 0.05)),
        layers.RandomTranslation(0.05, 0.02),
    ]
    return tf.keras.Sequential(augmentation_layers, name="data_augmentation")


def build_model(cfg: Settings) -> tf.keras.Model:
    """다중 헤드 Softmax 분류 모델을 생성한다."""

    inputs = layers.Input(shape=(cfg.image_height, cfg.image_width, cfg.image_channels), name="image")
    x = layers.Rescaling(1.0, name="identity")(inputs)

    aug_layer = _build_augment_layer(cfg)
    if aug_layer is not None:
        x = aug_layer(x)

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu")(x)

    outputs: List[tf.Tensor] = []
    for idx in range(cfg.max_label_length):
        outputs.append(
            layers.Dense(cfg.num_classes, activation="softmax", name=f"char_{idx}")(x)
        )

    return models.Model(inputs=inputs, outputs=outputs, name="captcha_classifier")


def build_optimizer(cfg: Settings) -> optimizers.Optimizer:
    return optimizers.Adam(learning_rate=cfg.learning_rate)


def build_loss_dict(cfg: Settings) -> dict[str, tf.keras.losses.Loss]:
    return {
        f"char_{idx}": tf.keras.losses.SparseCategoricalCrossentropy()
        for idx in range(cfg.max_label_length)
    }


def build_metric_dict(cfg: Settings) -> dict[str, list[tf.keras.metrics.Metric]]:
    return {
        f"char_{idx}": [tf.keras.metrics.SparseCategoricalAccuracy(name=f"acc_char_{idx}")]
        for idx in range(cfg.max_label_length)
    }
