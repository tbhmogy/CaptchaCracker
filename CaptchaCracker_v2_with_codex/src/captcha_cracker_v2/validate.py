"""검증 스크립트."""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from rich.console import Console
from rich.table import Table

from .config import load_settings
from .data_pipeline import build_datasets
from .model import build_loss_dict, build_metric_dict, build_model, build_optimizer
from .predictions import decode_output

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CaptchaCracker v2 검증")
    parser.add_argument("--weights", type=str, default=None, help="불러올 가중치 경로 (.keras)")
    parser.add_argument("--sample-count", type=int, default=5, help="로그에 뿌릴 예측 샘플 수")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_settings()
    weights_path = Path(args.weights) if args.weights else cfg.model_dir / f"{cfg.model_name}.keras"
    if not weights_path.exists():
        raise FileNotFoundError(f"가중치를 찾을 수 없습니다: {weights_path}")

    bundle = build_datasets(cfg)
    model = build_model(cfg)
    model.compile(optimizer=build_optimizer(cfg), loss=build_loss_dict(cfg), metrics=build_metric_dict(cfg))
    model.load_weights(str(weights_path))
    results = model.evaluate(bundle.validation, verbose=1)

    console.print("검증 결과")
    metrics_table = Table()
    metrics_table.add_column("지표", style="cyan")
    metrics_table.add_column("값", style="magenta")
    for name, value in zip(model.metrics_names, results):
        metrics_table.add_row(name, f"{value:.4f}")
    console.print(metrics_table)

    batch = next(iter(bundle.validation.take(1)))
    images, labels = batch
    preds = model.predict(images, verbose=0)
    decoded_preds = decode_output(preds, cfg)
    gt_matrix = tf.stack([labels[f"char_{idx}"] for idx in range(cfg.max_label_length)], axis=1)
    decoded_gt = [cfg.vocab.decode(row.numpy().tolist()) for row in gt_matrix]

    preview = Table(title="예측 샘플")
    preview.add_column("Index", style="cyan")
    preview.add_column("정답", style="green")
    preview.add_column("예측", style="magenta")
    for idx, (truth, pred) in enumerate(zip(decoded_gt, decoded_preds)):
        preview.add_row(str(idx), truth, pred)
        if idx + 1 >= args.sample_count:
            break
    console.print(preview)


if __name__ == "__main__":
    main()
