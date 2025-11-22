"""tf.data 기반 데이터 파이프라인."""
from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf

from .config import ConfigError, Settings

AUTOTUNE = tf.data.AUTOTUNE


@dataclass
class DatasetBundle:
    """학습/검증 데이터셋과 메타 정보를 묶어서 반환."""

    train: tf.data.Dataset
    validation: tf.data.Dataset
    train_count: int
    validation_count: int


def _find_image_paths(data_dir: Path) -> List[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg")
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(sorted(data_dir.glob(pattern)))
    return paths


def _load_image(path: tf.Tensor, cfg: Settings) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=cfg.image_channels, expand_animations=False)
    image = tf.image.resize(image, [cfg.image_height, cfg.image_width])
    image = tf.cast(image, tf.float32) / 255.0
    return image


def _format_labels(label_vec: tf.Tensor, cfg: Settings) -> dict:
    label_vec = tf.ensure_shape(label_vec, (cfg.max_label_length,))
    return {f"char_{idx}": tf.cast(label_vec[idx], tf.int32) for idx in range(cfg.max_label_length)}


def _dataset_from_arrays(paths: List[Path], labels: np.ndarray, cfg: Settings, training: bool) -> tf.data.Dataset:
    if not paths:
        raise ConfigError("데이터가 비어 있어 학습을 시작할 수 없습니다.")
    path_ds = tf.data.Dataset.from_tensor_slices([str(p) for p in paths])
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=cfg.random_seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, l: (_load_image(p, cfg), _format_labels(l, cfg)), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.batch_size).prefetch(AUTOTUNE)
    return ds


def build_datasets(cfg: Settings) -> DatasetBundle:
    data_dir = cfg.data_dir
    if not data_dir.exists():
        raise ConfigError(f"데이터 디렉터리가 존재하지 않습니다: {data_dir}")
    image_paths = _find_image_paths(data_dir)
    if len(image_paths) < 2:
        raise ConfigError("최소 2장 이상의 이미지가 필요합니다.")
    rng = random.Random(cfg.random_seed)
    rng.shuffle(image_paths)
    label_matrix = np.array([cfg.vocab.encode(path.stem, cfg.max_label_length) for path in image_paths], dtype=np.int32)

    val_size = max(1, int(len(image_paths) * cfg.validation_split))
    train_size = max(1, len(image_paths) - val_size)
    if train_size + val_size > len(image_paths):
        val_size = len(image_paths) - train_size
    if val_size <= 0:
        val_size = 1
        train_size = len(image_paths) - 1
    train_paths = image_paths[:train_size]
    val_paths = image_paths[train_size: train_size + val_size]
    train_labels = label_matrix[:train_size]
    val_labels = label_matrix[train_size: train_size + val_size]

    train_ds = _dataset_from_arrays(train_paths, train_labels, cfg, training=True)
    val_ds = _dataset_from_arrays(val_paths, val_labels, cfg, training=False)

    return DatasetBundle(
        train=train_ds,
        validation=val_ds,
        train_count=train_size,
        validation_count=val_size,
    )
