"""환경 변수 기반 설정 로딩 유틸리티."""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

load_dotenv()


class ConfigError(ValueError):
    """설정 값이 유효하지 않을 때 발생하는 예외."""


@dataclass
class Vocabulary:
    """문자 집합을 관리하고 인덱스를 매핑하는 헬퍼."""

    charset: str
    blank_char: str = "_"
    tokens: str = field(init=False)
    char_to_index: Dict[str, int] = field(init=False, repr=False)
    index_to_char: Dict[int, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        charset = self.charset or "0123456789"
        charset = "".join(dict.fromkeys(charset))  # 중복 제거
        if not charset:
            raise ConfigError("문자 집합이 비어 있습니다.")
        if len(self.blank_char) != 1:
            raise ConfigError("blank_char 는 한 글자여야 합니다.")
        if self.blank_char not in charset:
            charset += self.blank_char
        self.tokens = charset
        self.char_to_index = {ch: idx for idx, ch in enumerate(self.tokens)}
        self.index_to_char = {idx: ch for ch, idx in self.char_to_index.items()}

    @property
    def size(self) -> int:
        return len(self.tokens)

    @property
    def padding_index(self) -> int:
        return self.char_to_index[self.blank_char]

    def encode(self, text: str, max_length: int) -> list[int]:
        cleaned = (text or "").strip()
        if any(ch not in self.char_to_index for ch in cleaned):
            raise ConfigError(f"허용되지 않은 문자가 포함되어 있습니다: {cleaned}")
        clipped = cleaned[:max_length]
        padded = clipped.ljust(max_length, self.blank_char)
        return [self.char_to_index[ch] for ch in padded]

    def decode(self, indices: list[int]) -> str:
        chars = [self.index_to_char.get(int(idx), self.blank_char) for idx in indices]
        return "".join(ch for ch in chars if ch != self.blank_char)


@dataclass
class Settings:
    """프로젝트 전역 설정."""

    data_dir: Path
    image_width: int
    image_height: int
    image_channels: int
    max_label_length: int
    charset: str
    blank_char: str
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    augment_rotation: float
    random_seed: int
    model_dir: Path
    model_name: str
    log_level: str
    num_workers: int
    tf_device: Optional[str]
    vocab: Vocabulary = field(init=False)

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ConfigError("이미지 크기는 0보다 커야 합니다.")
        if self.batch_size <= 0:
            raise ConfigError("batch_size 는 0보다 커야 합니다.")
        if self.max_label_length <= 0:
            raise ConfigError("max_label_length 는 0보다 커야 합니다.")
        if not 0.0 < self.validation_split < 1.0:
            raise ConfigError("validation_split 은 0과 1 사이여야 합니다.")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate 는 0보다 커야 합니다.")
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.model_dir = Path(self.model_dir).expanduser().resolve()
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.vocab = Vocabulary(self.charset, self.blank_char)

    @property
    def num_classes(self) -> int:
        return self.vocab.size

    @property
    def padding_index(self) -> int:
        return self.vocab.padding_index


def _get_env(key: str, default: str, env: Optional[dict[str, str]] = None) -> str:
    current_env = env or os.environ
    return current_env.get(key, default)


def load_settings(extra_env: Optional[dict[str, str]] = None) -> Settings:
    """환경 변수를 읽어 Settings 인스턴스를 생성한다."""

    env = os.environ.copy()
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    settings = Settings(
        data_dir=_get_env("CC_DATA_DIR", "./data", env),
        image_width=int(_get_env("CC_IMAGE_WIDTH", "200", env)),
        image_height=int(_get_env("CC_IMAGE_HEIGHT", "50", env)),
        image_channels=int(_get_env("CC_IMAGE_CHANNELS", "1", env)),
        max_label_length=int(_get_env("CC_MAX_LABEL_LENGTH", "4", env)),
        charset=_get_env("CC_CHARSET", "0123456789", env),
        blank_char=_get_env("CC_BLANK_CHAR", "_", env),
        batch_size=int(_get_env("CC_BATCH_SIZE", "64", env)),
        epochs=int(_get_env("CC_EPOCHS", "30", env)),
        learning_rate=float(_get_env("CC_LEARNING_RATE", "0.001", env)),
        validation_split=float(_get_env("CC_VALIDATION_SPLIT", "0.2", env)),
        augment_rotation=float(_get_env("CC_AUGMENT_ROTATION", "3.0", env)),
        random_seed=int(_get_env("CC_RANDOM_SEED", "42", env)),
        model_dir=_get_env("CC_MODEL_DIR", "./model_checkpoints", env),
        model_name=_get_env("CC_MODEL_NAME", "weights_v2", env),
        log_level=_get_env("CC_LOG_LEVEL", "INFO", env),
        num_workers=int(_get_env("CC_NUM_WORKERS", "4", env)),
        tf_device=_get_env("CC_TF_DEVICE", "", env) or None,
    )
    return settings
