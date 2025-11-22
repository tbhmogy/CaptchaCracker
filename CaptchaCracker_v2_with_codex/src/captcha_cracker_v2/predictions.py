"""추론 및 디코딩 유틸리티."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import numpy as np

from .config import Settings


def _to_numpy(batch: Sequence) -> np.ndarray:
    if hasattr(batch, "numpy"):
        return batch.numpy()
    return np.asarray(batch)


def decode_output(predictions: Iterable, cfg: Settings) -> List[str]:
    """다중 헤드 Softmax 출력값을 문자열로 변환한다."""

    logits = [_to_numpy(pred) for pred in predictions]
    stacked = np.stack(logits, axis=1)
    indices = np.argmax(stacked, axis=-1)
    texts = [cfg.vocab.decode(sequence.tolist()) for sequence in indices]
    return texts
