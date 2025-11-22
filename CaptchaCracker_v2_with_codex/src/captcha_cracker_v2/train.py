"""학습 스크립트."""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
from typing import Dict, Optional

import tensorflow as tf
from rich.console import Console
from rich.table import Table

from .config import Settings, load_settings
from .data_pipeline import DatasetBundle, build_datasets
from .model import build_loss_dict, build_metric_dict, build_model, build_optimizer
from .predictions import decode_output

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = PROJECT_ROOT / "documents" / "training_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

console = Console()


def _env_overrides(args: argparse.Namespace) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    mapping = {
        "data_dir": "CC_DATA_DIR",
        "epochs": "CC_EPOCHS",
        "batch_size": "CC_BATCH_SIZE",
        "learning_rate": "CC_LEARNING_RATE",
        "model_name": "CC_MODEL_NAME",
        "model_dir": "CC_MODEL_DIR",
    }
    for attr, key in mapping.items():
        value = getattr(args, attr)
        if value is not None:
            overrides[key] = str(value)
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CaptchaCracker v2 학습 진입점")
    parser.add_argument("--data-dir", help="학습 이미지가 들어있는 디렉터리", default=None)
    parser.add_argument("--epochs", type=int, default=None, help="훈련 epoch 수 (기본값은 .env 기반)")
    parser.add_argument("--batch-size", type=int, default=None, help="배치 크기")
    parser.add_argument("--learning-rate", type=float, default=None, help="학습률")
    parser.add_argument("--model-name", type=str, default=None, help="저장할 가중치 이름")
    parser.add_argument("--model-dir", type=str, default=None, help="모델 디렉터리")
    parser.add_argument("--log-extra", action="store_true", help="학습 후 샘플 추론을 로그로 남김")
    return parser.parse_args()


def _print_settings(cfg: Settings) -> None:
    table = Table(title="현재 학습 설정")
    table.add_column("항목", style="cyan", no_wrap=True)
    table.add_column("값", style="magenta")
    entries = {
        "데이터 경로": str(cfg.data_dir),
        "이미지 크기": f"{cfg.image_width}x{cfg.image_height}x{cfg.image_channels}",
        "문자 집합": cfg.charset,
        "최대 길이": str(cfg.max_label_length),
        "배치 크기": str(cfg.batch_size),
        "Epoch": str(cfg.epochs),
        "학습률": str(cfg.learning_rate),
        "검증 비율": str(cfg.validation_split),
        "증강 회전": f"{cfg.augment_rotation} 도",
        "모델 출력": str(cfg.model_dir / (cfg.model_name + ".keras")),
    }
    for key, value in entries.items():
        table.add_row(key, value)
    console.print(table)


def _prepare_callbacks(cfg: Settings) -> list[tf.keras.callbacks.Callback]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    weights_path = cfg.model_dir / f"{cfg.model_name}.keras"
    history_path = LOG_DIR / f"history_{timestamp}.csv"
    tensorboard_dir = LOG_DIR / "tensorboard" / timestamp
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(history_path)),
        tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_dir), write_graph=False),
    ]
    return callbacks


def _log_sample_predictions(model: tf.keras.Model, bundle: DatasetBundle, cfg: Settings) -> None:
    batch = next(iter(bundle.validation.take(1)))
    images, labels = batch
    preds = model.predict(images, verbose=0)
    decoded_preds = decode_output(preds, cfg)
    gt_matrix = tf.stack([labels[f"char_{idx}"] for idx in range(cfg.max_label_length)], axis=1)
    decoded_gt = [cfg.vocab.decode(row.numpy().tolist()) for row in gt_matrix]
    preview = Table(title="검증 샘플 예측 (상위 5개)")
    preview.add_column("Index", style="cyan")
    preview.add_column("정답", style="green")
    preview.add_column("모델 예측", style="magenta")
    for idx, (truth, pred) in enumerate(zip(decoded_gt, decoded_preds)):
        preview.add_row(str(idx), truth, pred)
        if idx >= 4:
            break
    console.print(preview)


def main() -> None:
    args = parse_args()
    overrides = _env_overrides(args)
    cfg = load_settings(overrides)
    _print_settings(cfg)

    tf.keras.utils.set_random_seed(cfg.random_seed)
    bundle = build_datasets(cfg)
    console.print(f"학습 샘플 수: {bundle.train_count}, 검증 샘플 수: {bundle.validation_count}")

    model = build_model(cfg)
    optimizer = build_optimizer(cfg)
    losses = build_loss_dict(cfg)
    metrics = build_metric_dict(cfg)
    model.compile(optimizer=optimizer, loss=losses, metrics=metrics)

    callbacks = _prepare_callbacks(cfg)

    history = model.fit(
        bundle.train,
        epochs=cfg.epochs,
        validation_data=bundle.validation,
        callbacks=callbacks,
        verbose=1,
    )
    console.print(f"최종 val_loss: {history.history['val_loss'][-1]:.4f}")

    if args.log_extra:
        _log_sample_predictions(model, bundle, cfg)


if __name__ == "__main__":
    main()
